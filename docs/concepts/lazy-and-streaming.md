# Lazy Evaluation and the Streaming Engine

Nexpresso is designed to be used with `LazyFrame`s: pass a `LazyFrame` in, get a
`LazyFrame` back, and nothing executes until you `collect()` or `sink_*()`. This
page documents which operations honour that contract and — just as importantly —
which parts of a nexpresso pipeline the Polars **streaming engine** can actually
run with bounded memory.

## The lazy contract

Every public operation preserves the input frame type (`FrameT`), and the lazy
path never calls `collect()` behind your back. Schema resolution
(`collect_schema()`) is used freely — it inspects the query plan's metadata and
moves no data.

| Operation | Lazy in → lazy out | Executes anything? |
|---|---|---|
| `pack` | ✅ | No |
| `unpack` | ✅ | No |
| `split_levels` / `normalize` | ✅ (dict of `LazyFrame`) | No |
| `denormalize` | ✅ | No |
| `build_from_tables` | ✅ | No |
| `promote_attribute`, `enrich`, `attribute_expr` | ✅ | No |
| `any_child_satisfies`, `all_children_satisfy` | ✅ | No |
| `validate` | — | **Yes** — one `collect()`, by definition |
| `pack_streaming` | ✅ (returns `LazyFrame`) | Sinks to Parquet when `defer=False` |

!!! tip "Collect a dict of frames in one call"
    `split_levels` / `normalize` return one `LazyFrame` per level, and every one
    of them branches off the same upstream pipeline. Collecting them one at a
    time re-executes that shared work per level; `pl.collect_all` runs them as a
    single graph:

    ```python
    tables = packer.normalize(lazy_df)
    frames = dict(zip(tables, pl.collect_all(list(tables.values()))))
    ```

    Measured 1.3–1.7× faster on a 300k-row three-level frame. Eager input
    already does this internally.

!!! warning "`validate_on_pack` is eager-only"
    The uniformity check inside `pack` has to execute the query to see the data,
    so running it on a `LazyFrame` would break the lazy contract. It is therefore
    **skipped for lazy input** — `pack(lazy_frame, ...)` returns an unexecuted
    plan regardless of `validate_on_pack`. Call `validate()` explicitly if you
    need the check on a lazy pipeline.

## What the streaming engine can and cannot run

Polars' streaming engine processes data in morsels with bounded memory, but it
does not implement every node. When it meets one it cannot run, it inserts an
**`in-memory-map`** node: that subtree is materialized in full and executed by
the in-memory engine. The query still succeeds — it just stops being
memory-bounded.

You can see this for yourself:

```bash
POLARS_VERBOSE=1 python your_script.py 2>&1 | grep "running .* in subgraph"
```

`running in-memory-map in subgraph` means a fallback happened.

### `unpack` streams natively ✅

`unpack` is `explode` + `unnest`, both of which the streaming engine runs
natively. Peak memory is bounded by the morsel size times the child-list fan-out,
not by the dataset. Disk-to-disk unpacking of an arbitrarily large file works:

```python
packer.unpack_streaming("packed/*.parquet", "street", sink_path="flat.parquet")
```

### `pack` does **not** stream ❌

Packing is a `group_by` that collects children into list columns. The streaming
group-by supports only *reducing* aggregations (`sum`, `min`, `max`,
`first`/`first_non_null`, `n_unique`, …); it has **no native list-collecting
aggregation**. The whole `group_by` therefore falls back to the in-memory engine,
and peak memory scales with the entire dataset.

This is a Polars engine limitation, not something the library can express its way
around: the output of a pack *is* one row per parent holding every child, so the
group state is inherently unbounded.

Everything else in the pack pipeline is streamable — the row index, the struct
construction, and the `first_non_null` aggregations used to collapse parent
attributes all run natively. It is specifically the list-collecting aggregation
that forces the fallback.

### Working around it: `pack_streaming`

`pack_streaming` bounds memory by *partitioning* rather than by streaming the
aggregation. It buckets the **root-level key** — so all rows of an entity always
land together — packs each bucket independently, and sinks each packed bucket to
Parquet:

```python
packed = packer.pack_streaming(
    "flat/*.parquet",   # DataFrame, LazyFrame, or Parquet path/glob
    "region",
    partitions=32,      # peak memory ≈ dataset / 32
    defer=False,        # sink eagerly, return a scan_parquet handle
)
```

Peak memory is bounded by one bucket. Bucketing itself is a single streaming
pass: the input is written once to a partitioned Parquet staging area and each
bucket is then read back exactly once. (On Polars versions without partitioned
sinks this degrades to one filtered pass per bucket — still correct, but it
re-reads the source `partitions` times.)

#### Why not just sort by the key first?

Because **`sort` is itself an in-memory fallback**, on every version tested:

```
POLARS_VERBOSE=1 … .sort("country.code").sink_parquet(…)
  polars 1.41.2 -> running in-memory-map in subgraph
  polars 1.43.2 -> running in-memory-map in subgraph
```

Sorting the data to make entities contiguous would cost exactly the memory this
method exists to bound. What *is* cheap is sorting the **per-entity row counts** —
`group_by(root_key).agg(pl.len())` is a reducing aggregation, which the streaming
engine runs natively with state proportional to the number of entities rather
than the number of rows. That is what `partition_strategy="balanced"` does.

#### `partition_strategy`

| | `"balanced"` (default) | `"hash"` |
|---|---|---|
| Assignment | contiguous key ranges of ≈ equal **rows** | `hash(key) % partitions` |
| Extra pass | one (per-entity counts) | none |
| Balances | rows — what actually bounds memory | entities |
| Bucket count | floats around `partitions` | exactly `partitions` |
| Output order | sorted by root key | not guaranteed |

Balancing rows matters when entity sizes are uneven. On a set with a few large
entities among many small ones (30 230 rows, 300 entities, largest entity
3 000 rows — the floor no scheme can beat, since an entity cannot be split):

| `partitions` | `"hash"` max bucket | `"balanced"` max bucket | peak reduction |
|---|---|---|---|
| 8 | 6 834 (2.28× floor) | **3 775 (1.26×)** | 45% |
| 16 | 6 472 (2.16× floor) | **3 000 (1.00×)** | 54% |
| 64 | 3 240 (1.08× floor) | **3 000 (1.00×)** | 7%, using 22 buckets not 64 |

`"balanced"` reaches the floor at `partitions=16` with 12 buckets, where `"hash"`
still needs 64 buckets to get close. Note the bucket count is a *target*: an
entity is never split, so a bucket closes early rather than overflow and more
buckets than requested may be produced. Reach for `"hash"` when you know entity
sizes are uniform and want to skip the counting pass.

Choose `defer` deliberately:

- `defer=True` (default) keeps the call chain lazy, but the packed result is
  materialized in memory at the defer boundary.
- `defer=False` sinks the buckets eagerly and hands back a `scan_parquet`
  handle, so downstream work streams straight from disk. Use this when the
  *packed* result also does not fit in memory.

### Cost summary

| Stage | Streaming? | Peak memory |
|---|---|---|
| Row index, struct build, `first_non_null` aggregations | ✅ native | bounded |
| `group_by` + list collection (the pack itself) | ❌ in-memory fallback | O(dataset) |
| `explode` + `unnest` (the unpack) | ✅ native | bounded |
| `join` (`denormalize`, `build_from_tables`, `parent_strategy="split_join"`) | ✅ native | O(build side) |
| `unique` (`split_levels` on flat levels, `split_join` dimension table) | ✅ native | O(distinct rows) |
| `pack_streaming` | partitioned | O(dataset / partitions) |

## Ordering and determinism

`pack` deliberately avoids a global `sort` (a pipeline breaker) and does not use
`maintain_order` on the `group_by`. Two consequences:

- **Top-level row order after packing is not guaranteed.** Compare packed frames
  order-independently. The exception is
  `pack_streaming(..., partition_strategy="balanced")`, whose buckets are
  contiguous ascending key ranges sorted within each bucket, so the concatenated
  result *is* ordered by root key.
- **Child-list order *is* preserved.** A row index is carried through the
  aggregation and the child list is sorted by it inside the `agg`, so
  `preserve_child_order=True` (the default) and any `LevelSpec.order_by`
  expressions both hold without a global sort.

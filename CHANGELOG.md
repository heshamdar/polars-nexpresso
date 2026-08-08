# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`HierarchyView` — query normalized storage through a nested interface.**
  `List[Struct]` is a good in-memory shape and a poor storage shape: Parquet
  shreds nested columns into per-leaf column chunks, but a row group holds N
  *top-level* rows, so packing collapses row-group skipping and predicate
  pushdown. On a 2M-row three-level hierarchy, querying the packed file is
  **30–196x slower** than the same data stored flat, while the packed file is
  only ~15% smaller.

  `HierarchyView` stores one flat table per level and presents them as if they
  were nested. Columns are addressed by dotted hierarchy path; the view routes
  each operation to the table that owns it and joins across levels only when an
  operation genuinely spans them — callers never write a join.

  ```python
  from nexpresso import HierarchyView

  HierarchyView.from_frame(flat_df, packer).sink_parquet("warehouse/")
  view = HierarchyView.scan_parquet("warehouse/", packer)

  hot = view.filter(pl.col("region.store.sale.amount") > 990)
  hot.tables()["sale"]     # cheapest: no join, no nesting
  hot.collect("sale")      # flat, joined to leaf granularity
  hot.collect_nested()     # the packed List[Struct] shape
  ```

  Operations: `filter`, `with_columns`, `drop`, `promote`,
  `any_child_satisfies`. Terminals: `tables`, `to_flat` / `collect`,
  `to_nested` / `collect_nested`, `sink_parquet`.

  **Cross-level references.** Polars rejects named columns inside `list.eval`
  ("named columns are not allowed in `eval` functions"), so in a packed frame a
  leaf value can never be combined with a parent attribute — there is no outer
  scope to reach into. On a view it is an ordinary expression, because
  underneath it is a join:

  ```python
  view.with_columns(
      (
          pl.col("region.store.sale.amount")
          * (1 - pl.col("region.store.discount"))   # parent
          * (1 + pl.col("region.tax_rate"))         # grandparent
      ).alias("region.store.sale.final")
  )
  ```

  `filter` and `any_child_satisfies` take cross-level predicates on the same
  terms. Referencing a parent *aggregate* from the child is `promote` followed
  by an ordinary expression, or a window over the parent key when no join is
  wanted.

  A predicate on an ancestor **key** is applied to every table carrying it —
  sound transitive pushdown, since `normalize()` replicates ancestor keys as
  foreign keys — so the deepest scan skips row groups with no join at all.

  `empty_parents` controls what happens to parents left with no surviving
  children: `"prune"` (default) matches `pack()`, `"keep"` retains them with
  empty child lists and skips the upward semi-join cascade.

- **`benchmarks/bench_storage.py`** — compares nested / flat / normalized
  layouts across nine representative queries, cross-checking every result
  across layouts before timing so a divergence fails the run.

- **`examples_hierarchy_view.py`** — a runnable tour: the `list.eval`
  limitation demonstrated directly, then cross-level expressions, rollups and
  shares, cross-level filtering, conditional aggregation, and a full pipeline
  ending in the packed shape.

- **`docs/concepts/storage-layouts.md`** and **`docs/api/view.md`** — the
  measurements behind the above, and the API reference.

## [0.5.0] - 2026-08-07

### Changed (breaking)

- **`split_levels` / `normalize` emit level-local tables.** Each table now holds
  only that level's own columns plus its ancestors' **key** columns as foreign
  keys. Previously every table duplicated all coarser columns, so the leaf table
  carried the whole hierarchy.

  ```text
  before  street: country.code, country.name, country.city.id,
                  country.city.population, country.city.street.name, ...
  after   street: country.code, country.city.id,
                  country.city.street.name, country.city.street.length
  ```

  A level that is still flat in the input (e.g. `country` in a frame packed only
  to `city`) now gets its own deduplicated table rather than riding along inside
  the finer ones, so no attribute is silently dropped. To read a coarser
  attribute alongside finer rows, join it back from that level's table.

- **`denormalize` is now a true inverse of `normalize`.** At the root it returns
  a single root struct column, matching `pack(df, root)`; previously it left the
  root's own columns flat alongside the nested child list. For every level `L`,
  `denormalize(normalize(df, root_level=L), target_level=L)` now equals
  `pack(df, L)` — schema order, dtypes and rows.

- **`convert_polars_schema` distinguishes `Array` from `List`.** `Array(inner, n)`
  now converts to the tuple `(inner, n)` instead of `[inner]`, which was
  indistinguishable from `List(inner)` and silently dropped the size. `List`,
  `Struct` and scalar handling are unchanged.

### Added

- **`pack_streaming(..., partition_strategy=...)`** with a new default,
  `"balanced"`. It counts rows per entity in one extra streaming pass, then cuts
  the key-ordered entities into contiguous buckets of roughly equal **row**
  count. Peak memory is bounded by the largest bucket, so balancing rows beats
  the previous `hash(key) % partitions`, which balances entities. On a skewed set
  (30 230 rows, 300 entities, largest entity 3 000 rows):

  | `partitions` | `"hash"` max bucket | `"balanced"` max bucket |
  |---|---|---|
  | 8 | 6 834 | **3 775** (45% lower) |
  | 16 | 6 472 | **3 000** (54% lower, at the floor) |
  | 64 | 3 240 | **3 000**, using 22 buckets not 64 |

  Buckets are contiguous ascending key ranges, so `"balanced"` output is sorted
  by root key. `partitions` becomes a *target*: an entity is never split, so a
  bucket closes early rather than overflow. Pass `partition_strategy="hash"` for
  the previous behaviour, which skips the counting pass.

- `docs/concepts/lazy-and-streaming.md` — which operations the streaming engine
  runs natively (`unpack`, joins, `unique`) and which fall back to the in-memory
  engine (`pack`'s list-collecting `group_by`, and `sort`), with the reasoning
  and the commands to verify it.
- A CI `lint` job running `black --check`, `ruff` and `mypy`.

### Deprecated

- **`HierarchySpec.key_aliases`.** Rename the column on the frame instead:
  `df.with_columns(pl.col("country.city.id").alias("country.code"))`. Synthesized
  keys are stripped from the tables `split_levels` emits, so a hierarchy relying
  on them cannot round-trip through `denormalize`. Passing a non-empty mapping
  emits a `DeprecationWarning`; behaviour is otherwise unchanged.

### Fixed

- `all_children_satisfy` dropped entities whose child list was **null**,
  contradicting its documented "entities with no children pass (vacuous truth)".
  An empty list passed but a null list — what a left join produces for a
  childless parent — did not. A child whose condition evaluates to null still
  counts as unsatisfied; that rule is now documented and tested.
- `pack(frame, root_level)` raised `ColumnNotFoundError` whenever a key alias had
  to be synthesized, which also broke `normalize()`.
- `unpack` skipped the explode for a level column typed as a fixed-size `Array`
  and then failed in `prefix_fields`.
- `denormalize` ran an eager `collect()` per level on `LazyFrame` input via the
  pack uniformity check, silently executing the query.
- `validate` collected once per key column, re-running the whole upstream plan
  each time; it is now a single pass.
- `denormalize` now reports a missing key column by level and name instead of
  surfacing an opaque Polars `ColumnNotFoundError` from inside a join plan.
- `nexpresso/nexpresso.py` was a drifted copy of `expressions.py` — it never
  picked up the `arr.eval()` version gate — rather than the alias it was
  documented to be. It is now a re-export shim.
- `__version__` reported `0.3.1` while the package was `0.4.0`.
- `pack_streaming` crashed on an **empty input** with `partitions > 1`:
  `PartitionBy` writes no partitions at all, so the staging directory never
  appeared (`FileNotFoundError`), and on Polars 1.30's fallback path a zero
  bucket count wrote no parts at all (`ComputeError: expected at least 1
  source`). It now returns an empty result with the correct schema, as eager
  `pack` already did.
- `pack_streaming(partition_strategy="balanced")` crashed on **null root keys**.
  `group_by` treats null as its own group, but a plain join does not match null
  to null, so those rows came out with a null bucket that `PartitionBy` wrote as
  `__HIVE_DEFAULT_PARTITION__` — not an integer bucket id. The join is now
  null-matching.

### Performance

- `split_levels` collects the per-level tables with `pl.collect_all` for eager
  input. The plans share the progressive-unpack chain, so collecting them one at
  a time re-ran that work per level (~1.3–1.7× faster on a 300k-row 3-level
  frame). For lazy input, collect the returned dict with `pl.collect_all`.
- `pack_streaming` bucketing is a single streaming pass via a partitioned Parquet
  sink instead of one filtered pass per bucket (2.3× faster at
  `partitions=64`, and the gap widens with input size). Polars versions without
  partitioned sinks keep the old behaviour.
- The pack uniformity check reduces violation counts inside the engine rather
  than pulling one row per group into Python.

[0.5.0]: https://github.com/heshamdar/polars-nexpresso/releases/tag/v0.5.0

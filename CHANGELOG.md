# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

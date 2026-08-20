# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`with_level` transforms can reference any ancestor attribute.** The
  transform used to receive the level's table exactly as stored — its own
  columns plus the ancestor *keys* `normalize` replicates — so the most common
  thing you would want to do, a cross-level derivation, was the one thing it
  could not express. Reaching an ancestor *attribute* meant dropping to
  `tables()`, writing the joins by hand, and remembering to drop the borrowed
  columns again, which is what `level()` exists to spare you:

  ```python
  view.with_level("sale", lambda lf: lf.with_columns(
      (pl.col(AMOUNT) * (1 - pl.col(DISCOUNT)) * (1 + pl.col(TAX)))
      .alias("region.store.sale.net")
  ))
  ```

  Ancestor attributes are joined in for the computation and dropped again, so
  the level keeps its own schema and `nested()` still places everything by path.
  Aliasing one to a path this level owns keeps that copy. Borrowing goes up the
  hierarchy only — a descendant column would fan the level out to child
  granularity, and now says so instead of surfacing a raw Polars error with the
  query plan attached.

  The widening costs one join per ancestor **level**, not per column, and is
  paid only when the transform actually names an ancestor column; a transform
  confined to its own level runs against the bare table and joins nothing.

### Changed

- `_augmented` groups its borrowed columns by owner, so pulling several
  attributes from one ancestor is a single join rather than one per column.
  `filter` gets the same reduction on a multi-column cross-level predicate.

### Fixed

- **`filter` could silently drop rows for a window-shaped predicate.** Routing
  broadcasts a predicate over a replicated ancestor key to every table carrying
  it, which is sound only when the predicate reads each row on its own. The
  check for that asked whether the expression preserved the row count, and a
  window-shaped predicate does: `col > col.mean()` and `col.rank() <= 2` map N
  rows to N while still depending on the whole column. Such a predicate was
  broadcast and its aggregate recomputed at each level's own granularity, so
  `view.filter(pl.col("region.id") > pl.col("region.id").mean())` used the mean
  over *sales* on the sale table instead of the mean over regions, dropping
  regions the documented semantics keep. The probe now compares evaluating the
  predicate over the whole frame against evaluating it a row at a time, over
  distinct non-null values, which only an elementwise predicate survives. Plain
  elementwise predicates still broadcast, so the pushdown the shortcut exists
  for is unaffected.

### Documentation

- `filter` spells out that an aggregate carries no implicit `over` — a predicate
  such as `pl.col(AMOUNT).sum() > 100` is one scalar over the whole level, so
  every row survives or none does — and shows the roll-up plus semi-join to use
  for the per-parent question instead.
- `nested()` notes that its root default is the opposite end from `level()`'s
  finest-level default, and why.

## [0.9.0] - 2026-08-14

### Changed (breaking)

- **`HierarchyView` is a grain accessor, not a mirror of the Polars API.** It had
  grown six routed operations — `filter`, `with_columns`, `select`, `drop`,
  `promote`, `any_child_satisfies` — each resolving expressions to whichever
  level could evaluate them, and each with its own rules. That is where the
  rough edges were: `select` took strings but not expressions, `with_columns`
  refused any descendant input, `filter` refused an aggregating ancestor-key
  predicate, and `promote` refused a non-immediate child while
  `any_child_satisfies` skipped levels happily.

  Polars already has an expression API. The view now hands you a frame and gets
  out of the way:

  ```python
  view.level("sale").group_by("region.id").agg(pl.col(AMOUNT).sum())
  ```

  `level(g)` joins the root → `g` axis and returns a `pl.LazyFrame`, which has
  exactly one granularity — so the ambiguity the routing existed to resolve
  never arises. Joining the whole axis costs almost nothing you do not read:
  projection pushdown reduces an unused ancestor level to its key columns, and a
  predicate on an ancestor attribute is evaluated inside that level's own scan,
  before the join. `tests/test_view_level_access.py` asserts both against real
  Parquet plans.

  | Removed | Replacement |
  |---|---|
  | `view.select(...)` / `.with_columns(...)` / `.drop(...)` | `view.level(g).select(...)` / `.with_columns(...)` / `.drop(...)` |
  | `view.promote("amt", from_level="sale", to_level="region", agg="sum")` | `view.level("sale").group_by(view.key_columns("region")).agg(pl.col(AMOUNT).sum())` |
  | `view.any_child_satisfies(p, at_level="region", child_level="sale")` | `view.level("region").join(view.level("sale").filter(p).select(keys).unique(), on=keys, how="semi")` |
  | `view.to_flat(g)` / `view.collect(g)` | `view.level(g)` / `view.level(g).collect()` |
  | `view.to_nested(l)` / `view.collect_nested(l)` | `view.nested(l)` / `view.nested(l).collect()` |
  | `view.schema` | `view.level(g).collect_schema()` or `view.nested().collect_schema()` |
  | `view.explain(g)` | `view.level(g).explain()` |

  The replacements are not merely equivalent — the roll-up and the semi-join
  both work across skipped levels, which `promote` refused and
  `packer.any_child_satisfies` still does, because `normalize` replicates *every*
  ancestor id into a level's table rather than only the immediate parent's.

  `view.any_child_satisfies(p, ...)` is **not** `view.filter(p)`: the semi-join
  restricts only the parent, while `filter` also restricts the children and
  prunes parents left childless.

- **`filter` stays, and is the only operation that does.** Restricting a
  normalized hierarchy implies restrictions on the other levels — orphaned
  children removed, childless parents pruned under `empty_parents="prune"` —
  which is what keeps `filter → nested` and `filter → sink_parquet` correct
  without ever materializing parent columns per child row. No flat frame can
  reproduce that.

- **`view.schema` and `view.explain` are gone** rather than renamed. Every
  granularity has a different schema and a different plan, so neither is well
  posed for "the view"; ask the frame you mean.

- **`level()` raises where `to_flat` fell back to a cross join.** A level sharing
  no key columns with its parent silently multiplying rows is never the intent.

### Added

- **`HierarchyView.level(at_level=None)`** — a flat `LazyFrame` with one row per
  `at_level` entity and every ancestor column in scope. Only the target's axis is
  joined; sibling branches are left out rather than crossed in.
- **`HierarchyView.nested(at_level=None)`** — the packed `List[Struct]` shape,
  lazily. Replaces `to_nested`, and `collect_nested` is now `.nested().collect()`.
- **`HierarchyView.with_level(level, transform)`** — the counterpart to `level()`:
  it applies a transform to one level's table and returns a **view**, so the
  result can still be filtered, nested or sunk. Doing this through `tables()` +
  `from_tables` works too, but silently resets `empty_parents` and skips two
  checks `with_level` performs — that the level's key columns survive, and that
  every column is named with the level's full dotted path. The latter matters
  because `nested()` places columns by path, so an unqualified name survives
  `level()` and is silently *dropped* by `nested()`.
- **`HierarchyView.key_columns(level)`** — ancestor foreign keys then own ids.
  Public because user code needs it now: a roll-up is
  `level(child).group_by(view.key_columns(parent)).agg(...)`.

### Performance

- **Existence queries are ~7.6× faster** (71.0 ms → 9.3 ms at 2M leaf rows). The
  removed `any_child_satisfies` filtered a scratch copy of the child table and
  discarded it; the semi-join it is replaced by does not. `cross_level_predicate`
  also improves (82.4 → 57.4 ms). No query regressed — measured base-vs-head on
  the same machine with `benchmarks/bench_storage.py --scale large`.
- Documented when to prefer `tables()[g]` over `level(g)`: a roll-up keyed only
  on ancestor **keys** needs no join, because `normalize` already put them on the
  child table. Grouping `level(g)` instead joins the axis for nothing and costs
  ~2.3× (31.0 ms → 76.0 ms).

### Fixed

- **A multi-character `granularity_separator` never worked.** Both path splitters
  compared one **character** at a time (`path[i] == separator`), so a separator
  such as `"__"` or `"::"` never matched and a path came back as a single
  unsplit component. `join_path` and `escape_field` were correct, so paths were
  *written* correctly and could not be read back: `split_path` was not the
  inverse of `join_path`, and since column ownership is resolved by splitting,
  `HierarchyView` could not route anything at all — `level_of` raised, and
  `with_level` rejected every column. Only a single-character separator (the
  `"."` default) worked.

  Both splitters now match the separator as a substring, and
  `HierarchicalPacker._split_path` delegates to the module-level
  `_split_path_static` so the two cannot drift again. The escape convention
  extends to it: `escape_char` before a *whole* separator occurrence is a
  literal separator in a field name.

  `tests/test_custom_separator.py` runs the packer and view suites across
  `".", "__", "::", "/", "|", " -> "` — 15 of its cases fail without this fix.

- **An aggregating predicate over an ancestor key is no longer refused.**
  `view.filter(pl.col("region.id").count() > 10)` raised `ValueError`; it now
  evaluates at the level that *owns* the column, so the count is the number of
  regions rather than the number of sales they flatten to. Broadcasting a
  predicate to every carrier is demoted to what it always was — a pushdown
  shortcut for row-wise predicates — and the `_is_row_wise` probe now fails
  *closed*, since correctness comes from routing and the downward cascade rather
  than from the probe's verdict.

## [0.8.0] - 2026-08-14

### Changed (breaking)

- **A level argument now always names the granularity of the resulting rows.**
  `unpack(df, "city")` returned city rows, but `pack(df, "city")` returned
  *country* rows — it named the level that got **nested**, landing one level
  coarser. The two disagreed about the same word, `unpack(pack(df, L), L)` was
  not a round trip, and `infer_current_level` had to document that it "reports
  row granularity, which is one level coarser than the argument to `pack`".

  `pack` now folds only the levels *strictly below* its target:

  ```python
  packer.pack(flat, "building")   # leaf rows    — nothing nested
  packer.pack(flat, "street")     # street rows  — …street.building nested
  packer.pack(flat, "city")       # city rows    — the same frame unpack("city") gives
  packer.pack(flat, "country")    # country rows — country.code flat, country.city nested
  ```

  which makes this hold for every level, including the leaf:

  ```python
  packer.infer_current_level(packer.pack(df, L)) == L
  packer.pack(df, L).equals(packer.unpack(nested, L))
  ```

  `pack` was the only outlier. `promote_attribute`, `any_child_satisfies`,
  `all_children_satisfy`, `attribute_expr`, `enrich`, `validate_schema`,
  `HierarchyView.promote` / `any_child_satisfies` / `to_flat`, and
  `infer_current_level`'s return value already meant row granularity and are
  unchanged.

  **Migration.** Shift the level by one, towards the root:

  | Before | After |
  |---|---|
  | `pack(df, X)` for non-root `X` | `pack(df, parent_of(X))` |
  | `pack(df, root)` | `pack(df, root)` — same rows, new shape (see below) |
  | `pack_streaming(src, X)` | `pack_streaming(src, parent_of(X))` |
  | `normalize(df, root_level=X)` | `normalize(df, at_level=parent_of(X))` |
  | `denormalize(t, target_level=X)` | `denormalize(t, at_level=parent_of(X))` |
  | `build_from_tables(t, target_level=X)` | `build_from_tables(t, at_level=parent_of(X))` |

- **The root is no longer folded into a single struct column.** Previously
  `pack(df, root)` wrapped the whole hierarchy in one `country` `Struct`; now
  the root's own fields are ordinary columns beside the nested child column.
  This is a change of *shape*, not of granularity — both give one row per root
  entity with the same information:

  ```text
  before:  country          Struct{code, name, city: List[…]}
  after:   country.code     String
           country.name     String
           country.city     List[Struct{…}]
  ```

  Everything that landed at root granularity is affected, including
  `denormalize(tables)`, `HierarchyView.to_nested()` and `collect_nested()`.
  Field access moves from `df["country"][0]["city"]` to `df["country.city"][0]`,
  and a nested-expression spec keyed on `{"country": {"city": …}}` becomes
  `{"country.city": …}` (with `use_with_columns=True`, since a bare `select`
  would now drop the root's sibling columns). If you need the old wrapper:

  ```python
  prefix = "country."
  df.select(
      pl.struct([pl.col(c).alias(c[len(prefix):]) for c in df.columns]).alias("country")
  )
  ```

- **Level parameters renamed to `at_level`**, on `pack`, `unpack`,
  `pack_streaming`, `unpack_streaming`, `normalize` (was `root_level`),
  `denormalize` and `build_from_tables` (was `target_level`), and
  `HierarchyView.from_frame` (was `root_level`). Passing an old keyword raises a
  `TypeError` naming the replacement *and* the shifted meaning, so a keyword
  caller cannot silently get a frame at the wrong granularity. Positional
  callers shift silently — there is no way to detect that, hence the table
  above.

- **`pack` now emits a canonical column order**: a depth-first walk of the
  hierarchy, each level's own columns followed by its children in declaration
  order. Previously the order depended on the input's layout (a branch already
  packed in the input kept its position, a branch folded during the call was
  appended) and on `group_by` hoisting its keys to the front. That made
  `pack(df, L)` and `unpack(nested, L)` — two routes to one granularity —
  disagree on column order, and left `denormalize`, which assembles from
  per-level tables and has no input order to inherit, unable to match either.
  Non-hierarchy columns keep their relative order at the end.

### Added

- `HierarchyView.to_nested()` and `collect_nested()` take an optional
  `at_level`, so a view can be materialized at any granularity rather than only
  the root.

### Fixed

- `split_levels` deduplicated coarser levels with `unique(keep="any")` and no
  `maintain_order`. The surviving row order becomes that level's order inside
  its parent's child list when the tables are denormalized again, and Polars is
  free to reorder — 1.41 and 1.43 disagreed. Under `preserve_child_order` (the
  default) the dedup is now stable.

### Removed

- Three root-only special cases, unreachable once the root is never folded:
  `pack`'s alias-scaffolding guard, `_pack_split_join`'s struct-field
  reattachment branch, and `denormalize`'s final root fold. The packer is
  smaller than it was before this change.

## [0.7.0] - 2026-08-14

### Added

- **Branching hierarchies — a level can carry several independent child
  branches.** `HierarchySpec` modelled a single chain: `levels[i + 1]` was "the
  child" of `levels[i]`, and the whole packer plus `HierarchyView` were built on
  that index arithmetic. It could not express a level with two children, so a
  hierarchy like

  ```
  country > city > street > building
  country > city > service          (police, fire, water, medical)
  ```

  needed two packers over two copies of the data, or `service` forced into the
  street chain where it does not belong.

  `LevelSpec` gains a `parent` field. Naming it makes the spec a tree:

  ```python
  spec = HierarchySpec.from_levels(
      LevelSpec(name="country",  id_fields=["code"]),
      LevelSpec(name="city",     id_fields=["id"],   parent="country", parent_keys=["code"]),
      LevelSpec(name="street",   id_fields=["id"],   parent="city",    parent_keys=["city_id"]),
      LevelSpec(name="building", id_fields=["id"],   parent="street",  parent_keys=["street_id"]),
      LevelSpec(name="service",  id_fields=["kind"], parent="city",    parent_keys=["city_id"]),
  )
  ```

  A level's ancestors remain a unique chain even in a tree, so column paths
  (`country.city.service.kind`), key propagation and cross-level attribute
  traversal are unchanged. What changes is that "the child" becomes "the
  children" and iteration becomes topological.

  Each root → level chain is an **axis**. A flat frame holds one granularity, so
  `pack` and `unpack` traverse the axis their target level names and leave
  sibling branches packed as `List[Struct]` columns, replicated onto each row —
  exploding both branches would cross every street with every service. Nothing
  is dropped, so re-packing either frame reproduces the original:

  ```python
  packer.unpack(nested, "building")   # street axis; country.city.service stays nested
  packer.unpack(nested, "service")    # service axis; country.city.street stays nested
  ```

  Normalized storage needs no special case: every level is its own table, and a
  parent simply receives one `List[Struct]` column per branch when
  `denormalize` reassembles it.

  `HierarchyView` routes across branches too. `to_flat(level)` joins only that
  level's axis; `promote` and `any_child_satisfies` work on any branch; and the
  consistency cascades now alternate to a fixpoint, so filtering `service`
  prunes cities *and* the streets under them. An expression spanning two
  branches is rejected with a message naming both, rather than silently answered
  with a cross join.

  **Existing specs are unaffected.** When no level declares `parent`, the spec is
  read as a linear chain in declaration order exactly as before. `parent` is
  all-or-nothing: if any non-root level declares it, every non-root level must —
  inferring the rest from declaration order is precisely how `service` would get
  silently attached to `building`.

- New hierarchy navigation, on `HierarchySpec` — `parent_of`, `children_of`,
  `ancestors_of`, `descendants_of`, `axis_of`, `is_ancestor_of`, `root`,
  `leaves`, `topological_levels`, `reverse_topological_levels` — and on
  `HierarchicalPacker` — `leaf_levels`, `axes`, `get_axis`, `get_child_levels`.
- `infer_current_level` takes an optional `axis=` to measure row granularity
  along one branch, and now reports it as the deepest level whose own columns
  are still flat.

### Fixed

- `HierarchyView` tested "is an ancestor" by comparing positions in the level
  list, and `attribute_expr` compared level indices. Both accepted a level on a
  *sibling* branch, which would have fanned a frame out to an unrelated
  granularity or emitted an expression referencing a struct field that does not
  exist. Both now test the actual ancestor relation.
- `split_levels` deduplicated coarser levels with `unique(keep="any")` and no
  `maintain_order`. The surviving row order becomes that level's order inside
  its parent's child list when the tables are denormalized again, and Polars is
  free to reorder — 1.41 and 1.43 disagreed. Under `preserve_child_order` (the
  default) the dedup is now stable.
- `_pack_column_order` grouped carried columns by level; it now walks the tree
  depth-first, which is the order a flat frame actually holds them in, so
  `denormalize` reproduces `pack`'s column order for targets below a branch
  point as well.
- Packing now emits a level's child branches last, in declaration order, so a
  struct's field order does not depend on which axis the caller unpacked.

### Changed

- `HierarchicalPacker.leaf_level` and `HierarchySpec.next_level` raise on a
  branching hierarchy, where they have no single answer, and point at
  `leaf_levels` / `children_of`. Unbranched specs are unaffected.
- `get_descendant_levels` returns the whole subtree (every branch), in
  topological order. Same answer as before for a chain.
- `describe()` marks every childless level as a leaf, not just the last one, and
  lists a level's branches when it has more than one.

## [0.6.0] - 2026-08-11

### Changed (breaking)

- **Minimum Polars raised to 1.41.1.** Newer Polars releases carry substantial
  changes, and supporting versions back to 1.20 meant carrying fallbacks for
  APIs that have long since shipped. The floor is 1.41.**1** rather than 1.41.0
  because 1.41.0 was yanked from PyPI and cannot be installed.

  The CI matrix is now `["1.41.1", "latest"]`, and these compatibility branches
  are removed as unreachable:

  - `_supports_arr_eval()` / `ARR_EVAL_MIN_VERSION` and the `ValueError` raised
    for `Array` types on older Polars — `arr.eval()` is now used directly.
  - `_supports_partitioned_sink()` and `pack_streaming`'s per-bucket filter
    fallback — `pl.PartitionBy` is always available, so the single-pass
    partitioned sink is the only path.
  - The `pl.defer` availability check in `pack_streaming(defer=True)` and its
    `RuntimeError`.
  - `_supports_explode_empty_as_null()` — `explode(empty_as_null=True)` is now
    passed unconditionally (still pinned explicitly, since Polars 2.0 flips the
    default).
  - `HierarchyView.sink_parquet`'s `collect().write_parquet()` fallback.

  The version-pinned skip markers in `tests/conftest.py` (`requires_arr_eval`,
  `requires_struct_with_fields`, `requires_list_eval`, `requires_collect_schema`,
  `requires_group_by_maintain_order`, `requires_streaming_pack`) are gone —
  every one was vacuous at 1.41. The generic helpers (`get_polars_version`,
  `polars_version_at_least`, `polars_version_below`, `skip_if_polars_below`)
  remain for gating features newer than the floor.

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

  Operations: `filter`, `with_columns`, `select`, `drop`, `promote`,
  `any_child_satisfies`. Terminals: `tables`, `to_flat` / `collect`,
  `to_nested` / `collect_nested`, `sink_parquet`.

  `with_columns` routes by the **output** column's path, so a column named
  `"region.store.sale.net"` lands on the `sale` table whatever levels its inputs
  came from, and accepts `**named_exprs` to spell that destination explicitly.
  Computing an ancestor-level column from descendant input is refused with a
  pointer to `promote()`.

  Broadcasting an ancestor-key predicate applies only to **row-wise**
  predicates; an aggregating one (`count`, `sum`, `quantile`, a window) is
  rejected, because each level holds the key at a different granularity and
  intersecting per-level aggregates is meaningless.

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

- **`HierarchicalPacker.split_path` / `join_path` / `escape_field`** — public
  counterparts of the private path helpers, so collaborators can build column
  paths that agree with the ones the packer emits.

- **`HierarchicalPacker.GROUP_AGGREGATIONS`** — group-by counterparts of
  `_LIST_AGGREGATIONS`, kept adjacent to them so the packed and normalized
  aggregation paths cannot drift on null handling.

- **`tests/test_view_packed_equivalence.py`** — 259 cases asserting the packed
  and view paths agree. A shared case table expresses each operation twice —
  once against the flat frame, once against the view — and four laws are checked
  per case: `collect_nested()` matches `pack()`, denormalizing the view's own
  tables matches `pack()`, the flat form matches `unpack()`, and every level's
  table matches `split_levels()`. Comparisons are strict: same values, dtypes,
  struct field order and child ordering, normalized only by a root-row sort.
  Operations the nested expression builder can express are additionally checked
  head-to-head against `apply_nested_operations`.

  Covers uneven fan-out, single-child parents and nulls; 2-, 3- and 4-level
  hierarchies; and both `empty_parents` modes. The suite was mutation-tested —
  removing either consistency cascade, weakening the ancestor join to an inner
  join, or evaluating a cross-level predicate at the wrong level each make it
  fail.

- **`examples_hierarchy_view.py`** — a runnable tour: the `list.eval`
  limitation demonstrated directly, then cross-level expressions, rollups and
  shares, cross-level filtering, conditional aggregation, and a full pipeline
  ending in the packed shape.

- **`docs/concepts/storage-layouts.md`** and **`docs/api/view.md`** — the
  measurements behind the above, and the API reference.

### Fixed

- **`HierarchyView` routing bugs** — multi-argument `filter`/`with_columns`
  no longer discard earlier arguments when a cross-level path is taken;
  `with_columns` routes by output path (not inputs); escaped separator
  handling in `_owner_of`/`promote` uses the public path helpers;
  aggregating ancestor-key predicates are rejected instead of silently
  returning wrong row counts.

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

[0.9.0]: https://github.com/heshamdar/polars-nexpresso/releases/tag/v0.9.0
[0.8.0]: https://github.com/heshamdar/polars-nexpresso/releases/tag/v0.8.0
[0.7.0]: https://github.com/heshamdar/polars-nexpresso/releases/tag/v0.7.0
[0.6.0]: https://github.com/heshamdar/polars-nexpresso/releases/tag/v0.6.0
[0.5.0]: https://github.com/heshamdar/polars-nexpresso/releases/tag/v0.5.0

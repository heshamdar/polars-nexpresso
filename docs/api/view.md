# HierarchyView API Reference

`HierarchyView` stores a hierarchy as one flat table per level and hands you the
right frame for whatever granularity you are working at. It exists because
`List[Struct]` is a good in-memory shape and a poor storage shape — see
[Storage Layouts](../concepts/storage-layouts.md) for the measurements.

```python
view = HierarchyView.scan_parquet("warehouse/", packer)
view.level("sale").filter(pl.col("region.store.sale.amount") > 990)
```

[`level()`](#level) joins the root → `sale` axis and returns an ordinary
`pl.LazyFrame`, so everything after it is plain Polars. There is no expression
API to learn and no question of which granularity an expression means, because
a frame has exactly one.

## The surface

The view is deliberately small, and the return type tells you which context you
are in. Three methods get data **out** as frames; two return a **view**, so the
hierarchy survives and can still be filtered, nested or sunk.

| Method | Returns | What it is for |
|---|---|---|
| [`.level(at_level=None)`](#level) | `pl.LazyFrame` | Work at one granularity — the main entry point |
| [`.nested(at_level=None)`](#nested) | `pl.LazyFrame` | The packed `List[Struct]` shape, at the boundary |
| [`.tables()`](#tables) | `dict[str, pl.LazyFrame]` | The per-level plans, no join at all |
| [`.with_level(level, transform)`](#with_level) | `HierarchyView` | Modify one level and keep a view |
| [`.filter(*predicates)`](#filter) | `HierarchyView` | Restrict the hierarchy consistently |
| `.sink_parquet(dest, *, pattern="{level}", **kwargs)` | `None` | Stream one file per level |

Columns are addressed by their full path from the root, exactly as in a
flat/unpacked frame, regardless of which physical table holds them.

The separator is the packer's, not a hard-coded `.`:

```python
packer = HierarchicalPacker(spec, granularity_separator="__")
view.level_of("region__store__sale__amount")   # -> "sale"
```

Any string works, including multi-character ones. Build paths with
`packer.join_path([...])` rather than an f-string, and let
`packer.escape_field(...)` handle a field name that itself contains the
separator (stored as `region.store.net\.sales` with the default). Everything the
view does — ownership, `key_columns`, routing, the `with_level` naming check —
goes through those helpers.

## Constructors

| Method | Purpose |
|---|---|
| `HierarchyView.from_tables(tables, packer, *, empty_parents="prune")` | Wrap tables already in `normalize()` shape |
| `HierarchyView.from_frame(frame, packer, *, at_level=None, empty_parents="prune")` | Normalize an existing flat or packed frame |
| `HierarchyView.scan_parquet(source, packer, *, pattern="{level}", empty_parents="prune", **scan_kwargs)` | Scan one Parquet dataset per level from a directory |

```python
# One-time conversion, then scan on every subsequent query.
HierarchyView.from_frame(flat_df, packer).sink_parquet("warehouse/")
view = HierarchyView.scan_parquet("warehouse/", packer)
```

`scan_parquet` accepts either `region.parquet` or a `region/` directory of
parts for each level, and forwards `**scan_kwargs` to `pl.scan_parquet`.

## level

```python
def level(self, at_level: str | None = None) -> pl.LazyFrame: ...
```

A flat frame with one row per `at_level` entity and every ancestor column in
scope. `at_level` defaults to the finest level in the view.

```python
view.level("sale").group_by("region.id").agg(pl.col(AMOUNT).sum())
view.level("store").select("region.name", "region.store.discount")
```

### Asking for the whole axis is close to free

`level()` joins every ancestor unconditionally and does not need to be told
which columns you intend to read, because the query planner prunes what you
ignore. Selecting only sale columns from `level("sale")` plans as:

```text
Parquet SCAN [region.parquet]   PROJECT 1/2 COLUMNS      <- key only
Parquet SCAN [store.parquet]    PROJECT 2/4 COLUMNS      <- keys only
Parquet SCAN [sale.parquet]     PROJECT 4/5 COLUMNS
```

An unused ancestor level contributes its key columns and nothing else, and a
predicate on an ancestor *attribute* is evaluated inside that level's own scan,
before the join:

```text
Parquet SCAN [region.parquet]
PROJECT */2 COLUMNS
SELECTION: [(col("region.name")) == ("east")]
```

This is the property the design rests on, so it is asserted in
`tests/test_view_level_access.py` rather than assumed.

### Branches

Only the levels on `at_level`'s **axis** are joined. Sibling branches are left
out rather than crossed in — a flat frame has one granularity, and joining two
branches would pair every street with every service. In a
[branching hierarchy](../concepts/hierarchical-data.md#multiple-branches-per-level)
each branch has its own finest level, so `at_level` must be named explicitly.

## nested

```python
def nested(self, at_level: str | None = None) -> pl.LazyFrame: ...
```

Reconstructs the packed `List[Struct]` shape. `at_level` names the granularity
of the rows — it defaults to the root, giving one row per root entity with its
descendants nested. Note that this is the opposite end from `level()`, which
defaults to the finest granularity: each default is the one that leaves nothing
out, and for a nested frame that is the root.

```python
view.nested().collect()          # one row per region
view.nested("store").collect()   # one row per store
```

Worth calling only at the boundary where something actually consumes nesting;
every query above it is cheaper on the flat frames `level()` returns.

## tables

```python
def tables(self) -> dict[str, pl.LazyFrame]: ...
```

The per-level plans with cross-level consistency applied, root → leaf. The
cheapest entry point: no join and no nesting. Use it to edit one level in place
and rebuild:

```python
tables = dict(view.tables())
tables["sale"] = tables["sale"].with_columns(expr)
view = HierarchyView.from_tables(tables, packer)
```

## with_level

```python
def with_level(
    self,
    level: str,
    transform: Callable[[pl.LazyFrame], pl.LazyFrame],
) -> HierarchyView: ...
```

`level()` hands you a frame and lets go, which is what you want for a query.
`with_level` keeps the hierarchy, so the result can still be filtered, nested or
sunk. `transform` receives that level's own `LazyFrame` — ancestor **keys** are
on it, ancestor *attributes* are not:

```python
view.with_level("sale", lambda lf: lf.with_columns(
    (pl.col("region.store.sale.amount") * 2).alias("region.store.sale.doubled")
))
```

Two things are checked, because both fail quietly otherwise:

- the level's **key columns** must survive, or it can no longer be related to the
  rest of the hierarchy;
- every column must be named with the level's **full path**. `nested()`
  places columns by path, so an unqualified name survives `level()` and is
  silently *dropped* by `nested()`. `with_level` raises instead.

Doing this by hand through `tables()` and `from_tables` also works, and is what
you want when several levels change at once — but it skips the checks above and
silently resets `empty_parents` to `"prune"`.

## filter

```python
def filter(self, *predicates: pl.Expr) -> HierarchyView: ...
```

The one operation that is *not* available on the frame `level()` returns,
because restricting a normalized hierarchy implies restrictions on the other
levels: a child whose parent was filtered away is orphaned, and under
`empty_parents="prune"` a parent left with no children disappears. That is what
keeps `filter → nested` and `filter → sink_parquet` correct without ever
materializing parent columns per child row.

| Predicate references | Routing |
|---|---|
| One level's columns | Applied to that level's table |
| An ancestor **key**, elementwise | Applied to *every* table carrying it (sound transitive pushdown; the deepest scan skips row groups with no join) |
| An ancestor **key**, reading the whole column | Applied once, at the level that owns the column |
| An ancestor **attribute** | Applied to the ancestor, propagated by semi-join |
| Columns across levels | Evaluated at the deepest level, ancestor columns joined in and dropped again |

An aggregate is evaluated at the granularity of the level that owns its columns,
so `pl.col("region.id").count()` is the number of regions — not the number of
sales they flatten to. Broadcasting to every carrier is a pushdown shortcut and
is skipped for such a predicate, since each level holds a replicated key at a
different granularity.

The line the shortcut is drawn on is **elementwise**, not "returns as many rows
as it was given". A window-shaped predicate such as `col > col.mean()` or
`col.rank() <= 2` preserves the row count while still reading the whole column,
so broadcasting it would recompute the aggregate once per level and quietly
return a different row set than the owning level would. Those take the
owning-level path too.

```python
view.filter(pl.col("region.store.sale.amount") > 990)
view.filter(pl.col("region.id") == 3)              # pushed to every level
view.filter(pl.col("region.id").count() > 10)      # at the region level
view.filter(pl.col("region.id") > pl.col("region.id").mean())  # at the region level
```

!!! warning "An aggregate has no implicit `over`"

    `view.filter(pl.col("region.store.sale.amount").sum() > 100)` is one scalar
    over the **entire** sale table, so every row survives or none does — it is
    not "regions whose sales sum to more than 100". For a per-parent question,
    roll up and semi-join back:

    ```python
    keys = view.key_columns("region")
    big = (
        view.level("sale")
        .group_by(keys)
        .agg(pl.col(AMOUNT).sum().alias("revenue"))
        .filter(pl.col("revenue") > 100)
    )
    view.level("region").join(big.select(keys), on=keys, how="semi")
    ```

Use `level(g).filter(...)` instead when you are asking a question *about* `g`
rows and want a frame back; use `view.filter(...)` when you want a smaller
hierarchy back.

## Introspection

| Member | Returns |
|---|---|
| `.levels` | Level names present, root → leaf |
| `.columns` | Every addressable column, as dotted paths |
| `.level_of(column)` | The level that owns `column` |
| `.key_columns(level)` | Ancestor foreign keys then own ids — what to join and group on |

There is no `.schema`: every granularity has a different one. Ask the frame you
mean, `view.level(g).collect_schema()` or `view.nested().collect_schema()`; both
read plan metadata only and move no data. Likewise `view.level(g).explain()`
rather than a view-level `explain`.

## Recipes

Everything below is plain Polars on `level()`. The old view methods that did
these are gone; these are the replacements, and each is covered by a test in
`tests/test_view_packed_equivalence.py`.

### Cross-level expressions

Inside `list.eval` Polars forbids named columns outright:

```text
ComputeError: named columns are not allowed in `eval` functions; consider using `element`
```

So a leaf value can never be combined with a parent attribute in a packed frame
— there is no outer scope to reach into. On `level()` it is one expression,
because underneath it is a join:

```python
view.level("sale").with_columns(
    (
        pl.col("region.store.sale.amount")
        * (1 - pl.col("region.store.discount"))     # parent
        * (1 + pl.col("region.tax_rate"))           # grandparent
    ).alias("final")
)
```

### Rolling up to an ancestor

A `group_by` on the ancestor's key columns. `normalize()` replicates *every*
ancestor id into a level's table, not just the immediate parent's, so this works
at any depth with no intermediate hop:

```python
view.tables()["sale"].group_by(view.key_columns("region")).agg(
    pl.col("region.store.sale.amount").sum().alias("revenue")
)
```

Note `tables()["sale"]`, not `level("sale")`. The keys are already on the child
table, so this roll-up needs no join at all — going through `level()` here
measures ~2.3× slower for the same answer. Use `level(g)` when the expression
mentions an ancestor *attribute*, `tables()[g]` when it only mentions keys.

### Referencing a parent aggregate from the child

Join the roll-up back, or — when the aggregate is over the immediate parent —
use a window over the parent key, which needs no join at all:

```python
view.level("sale").with_columns(
    (
        pl.col("region.store.sale.amount")
        / pl.col("region.store.sale.amount").sum().over("region.store.id")
    ).alias("share")
)
```

### Conditional aggregation

An ordinary masked aggregate, so non-matching children contribute zero rather
than disappearing from the parent:

```python
view.level("sale").group_by(view.key_columns("store")).agg(
    pl.col("region.store.sale.amount").sum().alias("revenue"),
    pl.when(pl.col("region.store.sale.qty") >= 3)
    .then(pl.col("region.store.sale.amount"))
    .otherwise(0.0)
    .sum()
    .alias("bulk_revenue"),
)
```

### Existence — "parents with at least one matching child"

A semi-join: no explode, no list construction, and the child scan still gets its
own predicate pushdown. Unlike `packer.any_child_satisfies` it may skip levels.

```python
keys = view.key_columns("region")
matching = view.level("sale").filter(pl.col(AMOUNT) > 990).select(keys).unique()
view.level("region").join(matching, on=keys, how="semi")
```

This is **not** the same as `view.filter(pl.col(AMOUNT) > 990)`, which also
restricts the sales themselves and prunes regions left with none.

### Getting a derived column into the nested shape

Two routes. If the column is local to one level, use
[`with_level`](#with_level) — nothing is ever widened to another granularity:

```python
view.with_level("sale", lambda lf: lf.with_columns(expr)).nested().collect()
```

If it is cross-level, compute it on `level()` and pack the result. Name derived
columns with their full dotted path, since `pack()` places columns by path — a
column called `net` is taken to belong above the leaf:

```python
packer.pack(
    view.level("sale").with_columns(expr.alias("region.store.sale.net")).collect(),
    "region",
)
```

## Types

### EmptyParentMode

```python
EmptyParentMode = Literal["prune", "keep"]
```

How to treat parents left with no surviving children after a filter.

| Value | Behavior |
|---|---|
| `"prune"` (default) | Childless parents disappear — matches `packer.pack()` |
| `"keep"` | Childless parents retained with empty child lists; skips the upward semi-join cascade |

## Consistency model

Filtering one level implies restrictions on the others. The view defers both
cascades until a terminal method is called, and applies them only for levels an
operation actually restricted:

- **Downward** (always): children are semi-joined to their surviving parents,
  so filtering a parent attribute never leaves orphans visible in `tables()`.
- **Upward** (`empty_parents="prune"`): parents that lost every child disappear.

An unfiltered view resolves to its scans with no joins added.

Where the hierarchy branches the two cascades feed each other, so they alternate
until nothing new is restricted: filtering `service` prunes cities upward, and
those pruned cities must then prune `street` downward — a branch the first
downward pass never touched. Each edge is semi-joined at most once per direction,
so this costs nothing extra; a chain settles after the first round. Pruning is
per branch: a city with no surviving streets disappears even if its services
survived, matching `pack` along the street axis.

Note that `tables()` reflects storage as it is, while `level(g)` is a frame at
`g` granularity *with ancestor columns on it*. Where referential integrity is
broken — a sale pointing at a store that is not there — the orphan is a row in
`tables()["sale"]` and absent from `level("sale")`, because the axis join is an
inner one.

## Example

```python
import polars as pl
from nexpresso import HierarchicalPacker, HierarchySpec, HierarchyView, LevelSpec

spec = HierarchySpec.from_levels(
    LevelSpec(name="region", id_fields=["id"]),
    LevelSpec(name="store", id_fields=["id"], parent_keys=["region_id"]),
    LevelSpec(name="sale", id_fields=["id"], parent_keys=["store_id"]),
)
packer = HierarchicalPacker(spec)
view = HierarchyView.scan_parquet("warehouse/", packer)

# Restrict the hierarchy once; the levels stay normalized.
hot = view.filter(pl.col("region.store.sale.amount") > 990)

# Then it is polars, at whatever granularity the question is about.
hot.level("sale").group_by(view.key_columns("store")).agg(
    pl.col("region.store.sale.amount").sum().alias("total")
).collect()

hot.tables()["store"].collect()   # no join, no nesting
hot.nested().collect()            # packed List[Struct], only if needed
```

## See also

- [Storage Layouts](../concepts/storage-layouts.md) — why this exists, with benchmarks
- [HierarchicalPacker API](packer.md) — the packing operations this complements

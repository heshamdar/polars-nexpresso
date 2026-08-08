# HierarchyView API Reference

`HierarchyView` presents a collection of normalized per-level tables *as if*
they were a single nested frame. It exists because `List[Struct]` is a good
in-memory shape and a poor storage shape — see
[Storage Layouts](../concepts/storage-layouts.md) for the measurements.

Every operation returns a **new** view; the underlying frames are never
mutated. Nothing executes until a terminal method is called.

## Classes

### HierarchyView

```python
class HierarchyView:
    def __init__(
        self,
        tables: Mapping[str, pl.LazyFrame | pl.DataFrame],
        packer: HierarchicalPacker,
        *,
        empty_parents: EmptyParentMode = "prune",
    ) -> None: ...
```

Columns are addressed by their full dotted path (`"region.store.sale.amount"`),
exactly as in a flat/unpacked frame, regardless of which physical table holds
them.

## Constructors

| Method | Purpose |
|---|---|
| `HierarchyView.from_tables(tables, packer, *, empty_parents="prune")` | Wrap tables already in `normalize()` shape |
| `HierarchyView.from_frame(frame, packer, *, root_level=None, empty_parents="prune")` | Normalize an existing flat or packed frame |
| `HierarchyView.scan_parquet(source, packer, *, pattern="{level}", empty_parents="prune", **scan_kwargs)` | Scan one Parquet dataset per level from a directory |

```python
# One-time conversion, then scan on every subsequent query.
HierarchyView.from_frame(flat_df, packer).sink_parquet("warehouse/")
view = HierarchyView.scan_parquet("warehouse/", packer)
```

`scan_parquet` accepts either `region.parquet` or a `region/` directory of
parts for each level, and forwards `**scan_kwargs` to `pl.scan_parquet`.

## Introspection

| Member | Returns |
|---|---|
| `.levels` | Level names present, root → leaf |
| `.columns` | Every addressable column, as dotted paths |
| `.schema` | The **nested** `pl.Schema` the view presents (moves no data) |
| `.level_of(column)` | The level that owns `column` |
| `.explain(level=None)` | Query plan for the flat join to `level`, or for the nested reconstruction |

## Operations

All return a new `HierarchyView`.

### filter

```python
def filter(self, *predicates: pl.Expr) -> HierarchyView: ...
```

Routes each predicate to the level(s) that can evaluate it — you never write a
join.

| Predicate references | Routing |
|---|---|
| One level's columns | Applied to that level's table |
| An ancestor **key** | Applied to *every* table carrying it (sound transitive pushdown; the deepest scan skips row groups with no join) |
| An ancestor **attribute** | Applied to the ancestor, propagated by semi-join |
| Columns across levels | Evaluated at the deepest level, ancestor columns joined in and dropped again |

### with_columns

```python
def with_columns(self, *exprs: pl.Expr) -> HierarchyView: ...
```

Adds or replaces columns. Each lands on the level of the deepest input it
references. Expressions must be aliased.

### drop

```python
def drop(self, *columns: str) -> HierarchyView: ...
```

Drops columns from whichever level carries them. Refuses key columns — they are
the join structure of the view.

### promote

```python
def promote(
    self,
    attribute: str,
    *,
    from_level: str,
    to_level: str,
    agg: PromoteAggregation = "list",
    alias: str | None = None,
) -> HierarchyView: ...
```

The relational counterpart of
[`promote_attribute`](packer.md): a `group_by` on the child table joined onto
the parent. Never builds an intermediate `List[Struct]`. `from_level` must be
the immediate child of `to_level`.

### any_child_satisfies

```python
def any_child_satisfies(
    self,
    predicate: pl.Expr,
    *,
    at_level: str,
    child_level: str,
) -> HierarchyView: ...
```

Keeps only `at_level` rows having at least one matching descendant — a
semi-join. `child_level` may skip levels.

## Terminal methods

| Method | Returns | Cost |
|---|---|---|
| `.tables()` | `dict[str, pl.LazyFrame]` | Cheapest — no join, no nesting |
| `.to_flat(level=None)` | `pl.LazyFrame` | The join you would otherwise write |
| `.collect(level=None)` | `pl.DataFrame` | `to_flat` executed |
| `.to_nested()` | `pl.LazyFrame` | Nested reconstruction, lazy |
| `.collect_nested()` | `pl.DataFrame` | Nested reconstruction, executed |
| `.sink_parquet(dest, *, pattern="{level}", **kwargs)` | `None` | Streams one file per level |

`level` defaults to the finest level in the view.

!!! tip "Prefer `.tables()` where it suffices"
    Most questions are answered entirely from one level's table. `.to_flat()`
    and `.collect_nested()` exist for the boundary where something genuinely
    needs the joined or nested shape.

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

report = (
    view.filter(pl.col("region.store.sale.amount") > 990)
    .promote("amount", from_level="sale", to_level="store", agg="sum", alias="hot_total")
    .any_child_satisfies(
        pl.col("region.store.sale.qty") > 10, at_level="region", child_level="sale"
    )
)

report.tables()["store"].collect()   # no join, no nesting
report.collect_nested()              # packed List[Struct], only if needed
```

## See also

- [Storage Layouts](../concepts/storage-layouts.md) — why this exists, with benchmarks
- [HierarchicalPacker API](packer.md) — the packing operations this complements

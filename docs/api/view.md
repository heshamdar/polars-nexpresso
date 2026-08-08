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
them. Paths are split with the packer's own escaping rules, so a field whose
name contains the separator (stored as `region.store.net\.sales`) resolves
correctly.

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
| An ancestor **key**, row-wise | Applied to *every* table carrying it (sound transitive pushdown; the deepest scan skips row groups with no join) |
| An ancestor **key**, aggregating | **Rejected** — see the note below |
| An ancestor **attribute** | Applied to the ancestor, propagated by semi-join |
| Columns across levels | Evaluated at the deepest level, ancestor columns joined in and dropped again |

!!! warning "Aggregating predicates over an ancestor key are refused"
    Broadcasting is valid only for **row-wise** predicates. Each level holds an
    ancestor key at a different granularity, so an aggregate over one —
    `count`, `sum`, `mean`, `quantile`, a window — means something different per
    level, and intersecting those results is meaningless.

    ```python
    view.filter(pl.col("region.id").count() > 10)   # ValueError
    ```

    Apply such a predicate to one level's table via `tables()` instead.

### with_columns

```python
def with_columns(self, *exprs: pl.Expr, **named_exprs: pl.Expr) -> HierarchyView: ...
```

Adds or replaces columns, routed by the **output** column's path — a column
named `"region.store.sale.net"` lands on the `sale` table regardless of which
levels its inputs came from. Ancestor inputs are joined in and dropped again.

The keyword form spells the destination explicitly and is usually clearer than
an `.alias()` chain, since the path is the important part:

```python
view.with_columns(**{
    "region.store.sale.net": pl.col("region.store.sale.amount") * (1 - pl.col("region.store.discount")),
})
```

Computing an *ancestor*-level column from descendant input is refused — that is
an aggregation, so use [`promote`](#promote):

```python
view.with_columns(pl.col("region.store.sale.amount").sum().alias("region.total"))
# ValueError: ... Use promote() to aggregate a child attribute upward.
```

### select

```python
def select(self, *columns: str) -> HierarchyView: ...
```

Keeps only the named columns. Key columns are always retained regardless of
whether they are listed — without them the levels cannot be joined or nested.
Projection is what makes per-level Parquet scans cheap, so this is usually the
better way to express intent than `drop`.

```python
view.select("region.name", "region.store.sale.amount")
```

### drop

```python
def drop(self, *columns: str, strict: bool = True) -> HierarchyView: ...
```

Drops columns from whichever level carries them. Refuses key columns — they are
the join structure of the view. A column absent from every level raises;
pass `strict=False` for best-effort dropping.

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

Aggregations come from `HierarchicalPacker.GROUP_AGGREGATIONS`, the group-by
counterpart of the list aggregations `promote_attribute` uses, so the two agree
on null handling (`set` and `single` drop nulls; `count` is 0 — not null — for a
parent with no surviving children).

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

## Cross-level references

Inside `list.eval` Polars forbids named columns outright:

```text
ComputeError: named columns are not allowed in `eval` functions; consider using `element`
```

So a leaf value can never be combined with a parent attribute in a packed
frame — there is no outer scope to reach into. On a view this is an ordinary
expression, because underneath it is a join:

```python
# leaf x parent
view.with_columns(
    (pl.col("region.store.sale.amount") * (1 - pl.col("region.store.discount")))
    .alias("region.store.sale.net")
)

# leaf x grandparent, and all three levels at once
view.with_columns(
    (
        pl.col("region.store.sale.amount")
        * (1 - pl.col("region.store.discount"))
        * (1 + pl.col("region.tax_rate"))
    ).alias("region.store.sale.final")
)
```

`filter` and `any_child_satisfies` accept cross-level predicates on the same
terms — the ancestor columns are joined in for the evaluation and dropped
again, so the target level keeps its own schema.

```python
# 'regions containing a sale that alone owes more than 15 in tax'
view.any_child_satisfies(
    pl.col("region.store.sale.amount") * pl.col("region.tax_rate") > 15.0,
    at_level="region",
    child_level="sale",
)
```

### Referencing a parent aggregate from the child

Two steps, both cheap: roll up with `promote`, then read the result back down.

```python
view.promote("amount", from_level="sale", to_level="store", agg="sum", alias="revenue")
    .with_columns(
        (pl.col("region.store.sale.amount") / pl.col("region.store.revenue"))
        .alias("region.store.sale.share")
    )
```

When the aggregate is over the immediate parent, a window over the parent key
is cheaper still and needs no join at all — `normalize()` puts the parent key
on the child table:

```python
view.with_columns(
    (pl.col("region.store.sale.amount") / pl.col("region.store.sale.amount").sum().over("region.store.id"))
    .alias("region.store.sale.share")
)
```

### Conditional aggregation

Mask at the leaf, then promote, so non-matching children contribute zero
instead of disappearing from the hierarchy:

```python
view.with_columns(
    pl.when(pl.col("region.store.sale.qty") >= 3)
    .then(pl.col("region.store.sale.amount"))
    .otherwise(0.0)
    .alias("region.store.sale.bulk")
).promote("bulk", from_level="sale", to_level="store", agg="sum", alias="bulk_revenue")
```

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

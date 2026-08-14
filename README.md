# Polars Nexpresso ☕

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Polars](https://img.shields.io/badge/polars-%3E%3D1.41.1-blue)](https://www.pola.rs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Polars Nexpresso** is a utility library for working with nested and hierarchical data in Polars. It provides two main capabilities:

1. **Nested Expression Builder** - Clean, intuitive syntax for transforming deeply nested structs and lists
2. **Hierarchical Packer** - Pack/unpack operations for hierarchical data, similar to pandas MultiIndex but using Polars' native nested types

*Nexpresso* = **N**ested **Express**ion + ☕ (espresso)

## Installation

```bash
pip install polars-nexpresso
```

Or using `uv`:

```bash
uv add polars-nexpresso
```

## Quick Start

### Nested Expression Builder

Transform deeply nested data with intuitive dictionary syntax:

```python
import polars as pl
from nexpresso import generate_nested_exprs

df = pl.DataFrame({
    "order": [
        {"customer": "Alice", "items": [{"name": "Laptop", "price": 999}, {"name": "Mouse", "price": 25}]},
        {"customer": "Bob", "items": [{"name": "Keyboard", "price": 75}]},
    ]
})

# Define transformations declaratively
fields = {
    "order": {
        "items": {
            "price": lambda x: x * 1.1,  # 10% price increase
            "discounted": pl.field("price") * 0.9,  # New field
        }
    }
}

exprs = generate_nested_exprs(fields, df.schema, struct_mode="with_fields")
result = df.select(exprs)
```

### Hierarchical Packer

Build and navigate hierarchical data from normalized tables:

```python
from nexpresso import HierarchicalPacker, HierarchySpec, LevelSpec

# Define hierarchy
spec = HierarchySpec.from_levels(
    LevelSpec(name="region", id_fields=["id"]),
    LevelSpec(name="store", id_fields=["id"], parent_keys=["region_id"]),
)

packer = HierarchicalPacker(spec)

# Build from separate tables (like database tables)
nested = packer.build_from_tables({
    "region": regions_df,
    "store": stores_df,
})

# Navigate between granularities. The level names the granularity you want,
# for both directions: one row per store, then one row per region.
flat = packer.unpack(nested, "store")       # Explode to store level
packed = packer.pack(flat, "region")        # Aggregate back to region level
```

## Features

### Nested Expression Builder

| Feature | Description |
|---------|-------------|
| **Field Selection** | Keep fields as-is with `None` |
| **Transformations** | Apply lambdas: `lambda x: x * 2` |
| **New Fields** | Create with `pl.Expr`: `pl.field("a") + pl.field("b")` |
| **Deep Nesting** | Works with any depth of structs/lists |
| **Two Modes** | `select` (keep specified) or `with_fields` (keep all) |

### Hierarchical Packer

| Feature | Description |
|---------|-------------|
| **Build from Tables** | Join normalized tables into nested hierarchy |
| **Pack/Unpack** | Navigate between granularity levels |
| **Multiple Branches** | A level can carry several independent child branches; pack/unpack along either axis |
| **Streaming Pack/Unpack** | Memory-bounded `pack_streaming` / `unpack_streaming` for large data |
| **Normalize/Denormalize** | Split into per-level tables and reconstruct |
| **Validation** | Check for null keys and data integrity |
| **Custom Separators** | Use any separator (default: `.`) |
| **Type Preservation** | DataFrame in = DataFrame out |

### Hierarchy View

Query normalized per-level storage through a nested interface.

| Feature | Description |
|---------|-------------|
| **Nested interface, flat storage** | Address columns by dotted path; the view routes to the owning table |
| **Cross-level expressions** | Combine leaf, parent and grandparent columns in one expression — impossible inside `list.eval` |
| **No hand-written joins** | Cross-level operations join automatically, only when needed |
| **Transitive key pushdown** | Ancestor-key predicates reach the deepest scan with no join |
| **Deferred consistency** | Filtering one level restricts the others, applied once at materialization |
| **Nest only at the boundary** | `collect_nested()` when a consumer genuinely needs `List[Struct]` |

## Core Concepts

### Field Value Types

When defining transformations:

- **`None`**: Keep the field unchanged
- **`dict`**: Recursively process nested structures
- **`Callable`**: Apply function to field (e.g., `lambda x: x * 2`)
- **`pl.Expr`**: Create/modify field with full expression

### Struct Modes

- **`"select"`**: Only keep fields specified in the dictionary
- **`"with_fields"`**: Keep all fields, add/modify specified ones

### Hierarchy Levels

Define your data hierarchy with `LevelSpec`:

```python
LevelSpec(
    name="store",           # Level identifier
    id_fields=["id"],       # Unique key columns
    parent_keys=["region_id"],  # Foreign key to parent (for build_from_tables)
    parent=None,            # Parent level name; None = the level declared before
)
```

### Multiple Branches

Levels form a chain by default. Naming a `parent` explicitly makes the hierarchy
a tree, so one level can carry several *independent* child branches — a city has
streets (which have buildings) and, orthogonally, services:

```python
spec = HierarchySpec.from_levels(
    LevelSpec(name="country",  id_fields=["code"]),
    LevelSpec(name="city",     id_fields=["id"],   parent="country", parent_keys=["code"]),
    LevelSpec(name="street",   id_fields=["id"],   parent="city",    parent_keys=["city_id"]),
    LevelSpec(name="building", id_fields=["id"],   parent="street",  parent_keys=["street_id"]),
    LevelSpec(name="service",  id_fields=["kind"], parent="city",    parent_keys=["city_id"]),
)
```

Each root → level chain is an **axis**. A flat frame holds one granularity, so
`pack` and `unpack` work along the axis their target names and leave sibling
branches packed:

```python
packer.unpack(nested, "building")   # street axis; `country.city.service` stays nested
packer.unpack(nested, "service")    # service axis; `country.city.street` stays nested
```

Nothing is dropped and nothing is cross-joined, so re-packing either frame
reproduces the original. `HierarchyView` follows the same rule: `to_flat(level)`
joins one axis, and a filter on one branch cascades to the other through their
shared ancestor. See
[Hierarchical Data](https://heshamdar.github.io/polars-nexpresso/concepts/hierarchical-data/#multiple-branches-per-level).

## Examples

### Lists of Structs

```python
df = pl.DataFrame({
    "items": [[{"name": "A", "qty": 5}, {"name": "B", "qty": 3}]]
})

fields = {
    "items": {
        "qty": lambda x: x * 2,
        "total": pl.field("qty") * 10,
    }
}

result = apply_nested_operations(df, fields, struct_mode="with_fields")
```

### Conditional Logic

```python
fields = {
    "customer": {
        "discount": pl.when(pl.field("tier") == "Gold")
            .then(0.15)
            .when(pl.field("tier") == "Silver")
            .then(0.10)
            .otherwise(0.05),
    }
}
```

### Building Hierarchies from Database Tables

```python
# Tables with foreign key relationships
regions = pl.DataFrame({"id": ["west", "east"], "name": ["West", "East"]})
stores = pl.DataFrame({
    "id": ["s1", "s2"], 
    "name": ["Store 1", "Store 2"],
    "region_id": ["west", "east"]
})

spec = HierarchySpec.from_levels(
    LevelSpec(name="region", id_fields=["id"]),
    LevelSpec(name="store", id_fields=["id"], parent_keys=["region_id"]),
)

packer = HierarchicalPacker(spec)
nested = packer.build_from_tables({"region": regions, "store": stores})
# Result: Stores nested within their regions
```

### Normalize and Denormalize

`normalize` / `split_levels` emit one **level-local** table per level: the
level's own id fields and attributes, plus the *key* columns of its ancestors as
foreign keys. Coarser attributes are not duplicated into the finer tables, and
descendant columns are never included.

```python
# Split nested data into separate tables
tables = packer.normalize(nested_df)
# region: region.id, region.name
# store : region.id, region.store.id, region.store.revenue
#         ^ foreign key   ^ own columns only

# Reconstruct from separate tables (round-trips back to the nested frame)
rebuilt = packer.denormalize(tables)
```

### Memory-bounded packing for large data

`pack` builds the nested result with a `group_by` that collects children into
list columns. Polars' streaming engine has no native list-collecting
aggregation, so that node falls back to the in-memory engine and peak memory
scales with the whole dataset. (`unpack` — `explode` + `unnest` — *does* stream
natively; see [Lazy Evaluation & Streaming](docs/concepts/lazy-and-streaming.md)
for the full breakdown.)

For datasets that don't fit comfortably in RAM, `pack_streaming` buckets
the input by the **root-level key** (keeping each entity's rows together), packs
each bucket independently while sinking to Parquet, and returns a chainable
`LazyFrame` — bounding peak memory to a single bucket.

```python
# Bound peak memory by processing the data in root-key buckets.
# Accepts a DataFrame, LazyFrame, or a Parquet path/glob (scanned lazily).
packed = packer.pack_streaming(flat_df, "region", partitions=32)

# It returns a LazyFrame, so you can keep composing lazily:
top = (
    packer.pack_streaming("s3_dump/*.parquet", "region", partitions=64)
    .filter(pl.col("region.id").is_in(active_regions))
    .collect(engine="streaming")
)

# defer=False sinks eagerly and returns a scan handle so downstream work streams
# straight from disk — safest when even the packed result is too big for RAM.
handle = packer.pack_streaming(flat_df, "region", partitions=64, defer=False)

# unpack already streams; unpack_streaming keeps it lazy / disk-to-disk.
leaves = packer.unpack_streaming("packed.parquet", "store", sink_path="leaves.parquet")
```

More buckets means lower peak memory (and more temporary files). See
[`benchmarks/`](benchmarks/) for a peak-RSS comparison of `pack` vs
`pack_streaming`.

### Heavy root attributes: `parent_strategy="split_join"`

When the **root** level carries heavy attributes that repeat across every leaf row
(e.g. a per-entity blob, thumbnail, or embedding), carrying them through the pack
aggregation is wasteful. `pack(..., parent_strategy="split_join")` instead pulls
those attributes into a small dimension table (unique per root key) and reattaches
them after packing — identical results, but the heavy column is touched once per
entity instead of once per leaf row.

```python
# Equivalent to the default pack, but far cheaper when root attributes dominate.
packed = packer.pack(flat_df, "region", parent_strategy="split_join")
```

This wins big when root attributes are heavy relative to the child data (measured
up to ~9x faster, half the memory); for child-dominated data it adds join overhead
for no gain, so it is opt-in. See [`benchmarks/`](benchmarks/) for the trade-offs.

> **Note on ordering:** packing no longer performs a global sort, so **top-level
> row order is not guaranteed**. Child-list order is still preserved when
> `preserve_child_order=True` (the default) or via a level's `order_by`.
> De-duplication and null handling of parent attributes are unaffected.

## API Reference

### Nested Expressions

#### `generate_nested_exprs(fields, schema, struct_mode="select")`

Generate Polars expressions for nested data operations.

**Parameters:**
- `fields`: Dictionary defining operations on columns/fields
- `schema`: DataFrame schema (or DataFrame/LazyFrame to extract schema)
- `struct_mode`: `"select"` or `"with_fields"`

**Returns:** `list[pl.Expr]`

#### `apply_nested_operations(df, fields, struct_mode="select", use_with_columns=False)`

Apply nested operations directly to a DataFrame.

### Hierarchical Packer

#### `HierarchicalPacker(spec, *, granularity_separator=".", escape_char="\\", preserve_child_order=True, validate_on_pack=True)`

Main class for hierarchical operations.

**Key Methods:**
- `pack(frame, at_level, *, extra_columns="preserve", parent_strategy="aggregate")` - Pack so
  each row is one `at_level` entity, nesting everything below it (`parent_strategy="split_join"`
  reattaches heavy root attributes via a join)
- `pack_streaming(source, at_level, *, partitions=16, partition_strategy="balanced", ...)` -
  Memory-bounded pack for large data (`"balanced"` buckets by row count and returns
  root-key-sorted output; `"hash"` is one pass cheaper but balances entities, not rows)
- `unpack(frame, at_level)` - Unpack so each row is one `at_level` entity — the exact
  inverse of `pack` at the same level
- `normalize(frame)` - Split into per-level tables
- `denormalize(tables)` - Reconstruct from per-level tables; a true inverse of
  `normalize`, so `denormalize(normalize(df, at_level=L), at_level=L) == pack(df, L)`
- `build_from_tables(tables)` - Build hierarchy from normalized tables
- `validate(frame)` - Check data integrity

#### `HierarchySpec.from_levels(*levels, key_aliases=None)`

Create a hierarchy specification from level definitions.

`key_aliases` is **deprecated** — rename the column on the frame instead
(`df.with_columns(pl.col("country.city.id").alias("country.code"))`). Synthesised
keys are stripped from the per-level tables, so a hierarchy relying on them cannot
round-trip through `normalize` / `denormalize`.

#### `LevelSpec(name, id_fields, required_fields=None, order_by=None, parent_keys=None, parent=None)`

Define a single level in the hierarchy. `parent` names this level's parent; leave
it `None` for a linear chain, or set it on every non-root level to branch.

## Running Examples

```bash
# Run comprehensive examples
python examples.py

# Or run specific module examples
python -m nexpresso.hierarchical_packer
```

## Storage Layouts

`List[Struct]` is a good in-memory shape and a poor storage shape. Parquet
shreds nested columns into one column chunk per leaf, but a row group holds N
**top-level** rows — so packing collapses row-group skipping, the main reason
Parquet is fast. On a 2M-row three-level hierarchy, querying the packed file is
**30-196x slower** than the same data stored flat, and the packed file is only
~15% smaller.

Store flat or normalized; pack at the boundary where something consumes nesting.
`HierarchyView` makes the normalized layout ergonomic:

```python
from nexpresso import HierarchyView

# One-time conversion.
HierarchyView.from_frame(flat_df, packer).sink_parquet("warehouse/")

# Every subsequent query scans one Parquet dataset per level.
view = HierarchyView.scan_parquet("warehouse/", packer)

hot = view.filter(pl.col("region.store.sale.amount") > 990)
hot.tables()["sale"]     # cheapest: no join, no nesting
hot.collect("sale")      # flat, joined to leaf granularity
hot.collect_nested()     # the packed List[Struct] shape
```

It also unlocks something `list.eval` cannot express at all — Polars rejects
named columns inside an eval context, so a leaf value can never be combined with
a parent attribute in a packed frame. On a view it is just an expression:

```python
view.with_columns(
    (
        pl.col("region.store.sale.amount")
        * (1 - pl.col("region.store.discount"))   # parent
        * (1 + pl.col("region.tax_rate"))         # grandparent
    ).alias("region.store.sale.final")
)
```

See [Storage Layouts](docs/concepts/storage-layouts.md) for the measurements,
`python examples_hierarchy_view.py` for a runnable tour, and
`benchmarks/bench_storage.py` to reproduce the numbers.

## Performance

Both components generate native Polars expressions, so performance is equivalent to hand-written code. All operations are lazy-compatible and benefit from Polars' query optimization.

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please feel free to submit issues and pull requests.

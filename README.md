# Polars Nexpresso ☕

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Polars](https://img.shields.io/badge/polars-%3E%3D1.20.0-blue)](https://www.pola.rs/)
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

# Navigate between granularities
flat = packer.unpack(nested, "store")      # Explode to store level
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
| **Normalize/Denormalize** | Split into per-level tables and reconstruct |
| **Validation** | Check for null keys and data integrity |
| **Custom Separators** | Use any separator (default: `.`) |
| **Type Preservation** | DataFrame in = DataFrame out |

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
)
```

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

```python
# Split nested data into separate tables
tables = packer.normalize(nested_df)
# {"region": region_df, "store": store_df, ...}

# Reconstruct from separate tables
rebuilt = packer.denormalize(tables)
```

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
- `pack(frame, to_level)` - Pack to coarser granularity
- `unpack(frame, to_level)` - Unpack to finer granularity
- `normalize(frame)` - Split into per-level tables
- `denormalize(tables)` - Reconstruct from per-level tables
- `build_from_tables(tables)` - Build hierarchy from normalized tables
- `validate(frame)` - Check data integrity

#### `HierarchySpec.from_levels(*levels, key_aliases=None)`

Create a hierarchy specification from level definitions.

#### `LevelSpec(name, id_fields, required_fields=None, order_by=None, parent_keys=None)`

Define a single level in the hierarchy.

## Running Examples

```bash
# Run comprehensive examples
python examples.py

# Or run specific module examples
python -m nexpresso.hierarchical_packer
```

## Performance

Both components generate native Polars expressions, so performance is equivalent to hand-written code. All operations are lazy-compatible and benefit from Polars' query optimization.

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please feel free to submit issues and pull requests.

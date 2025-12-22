# Polars Nexpresso

**Clean nested data transformations and hierarchical operations for Polars.**

Nexpresso provides two main capabilities for working with complex data structures in Polars:

1. **Nested Expression Builder** - Transform deeply nested structs and lists with intuitive dictionary syntax
2. **Hierarchical Packer** - Build, navigate, and manipulate hierarchical data (like pandas MultiIndex, but using Polars' native nested types)

## Why Nexpresso?

Working with nested data in Polars is powerful but can be verbose. Nexpresso reduces boilerplate and cognitive load:

=== "Without Nexpresso"

    ```python
    # Updating a field inside a list of structs
    df.with_columns(
        pl.col("orders").list.eval(
            pl.struct(
                pl.element().struct.field("item"),
                pl.element().struct.field("price"),
                (pl.element().struct.field("price") * 
                 pl.element().struct.field("qty")).alias("total")
            )
        )
    )
    ```

=== "With Nexpresso"

    ```python
    apply_nested_operations(df, {
        "orders": {
            "item": None,
            "price": None,
            "total": pl.field("price") * pl.field("qty")
        }
    }, struct_mode="with_fields")
    ```

## Quick Start

### Installation

```bash
pip install polars-nexpresso
```

Or with uv:

```bash
uv add polars-nexpresso
```

### Basic Example

```python
import polars as pl
from nexpresso import apply_nested_operations

# Create nested data
df = pl.DataFrame({
    "order": [
        {"customer": "Alice", "items": [{"name": "Laptop", "price": 999}]},
        {"customer": "Bob", "items": [{"name": "Mouse", "price": 25}]},
    ]
})

# Transform with nested expressions
result = apply_nested_operations(df, {
    "order": {
        "customer": None,  # Keep as-is
        "items": {
            "name": None,
            "price": None,
            "discounted": pl.field("price") * 0.9,  # Add new field
        }
    }
}, struct_mode="with_fields")
```

## Features

| Feature | Description |
|---------|-------------|
| **Intuitive Syntax** | Dictionary-based field specifications |
| **Deep Nesting** | Works with any depth of structs and lists |
| **Two Modes** | `select` (filter fields) or `with_fields` (preserve all) |
| **Array Support** | Full support for Polars Array types via `arr.eval()` |
| **Hierarchical Data** | Pack/unpack operations for multi-level data |
| **Database Integration** | Build hierarchies from normalized tables |
| **Type Preservation** | DataFrame in = DataFrame out, LazyFrame in = LazyFrame out |
| **Validation** | Check data integrity in hierarchies |

## Next Steps

- [Getting Started](getting-started.md) - Detailed installation and first steps
- [Nested Expressions](concepts/nested-expressions.md) - Understanding the expression builder
- [Hierarchical Data](concepts/hierarchical-data.md) - Working with pack/unpack operations
- [API Reference](api/expressions.md) - Complete API documentation

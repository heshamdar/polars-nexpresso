# Polars Nexpresso ☕

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Polars](https://img.shields.io/badge/polars-%3E%3D1.20.0-blue)](https://www.pola.rs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Polars Nexpresso** - A utility library for generating Polars expressions to work with nested data structures. Easily select, modify, and create columns and nested fields in Polars DataFrames, particularly for complex nested structures like lists of structs and deeply nested hierarchies.

*Nexpresso* = **N**ested **Express**ion + ☕ (espresso) - because why not?

## Motivation

Working with deeply nested columns in Polars can quickly become verbose and hard to read. Consider modifying a field nested within a list of structs that contains another list of structs:

**Without Nexpresso:**
```python
import polars as pl

df = pl.DataFrame(
    {
        "orders": [
            [
                {"items": [{"quantity": 1, "price": 10}, {"quantity": 2, "price": 20}]},
                {"items": [{"quantity": 3, "price": 30}, {"quantity": 4, "price": 40}]},
            ],
            [
                {"items": [{"quantity": 5, "price": 50}, {"quantity": 6, "price": 60}]},
                {"items": [{"quantity": 7, "price": 70}, {"quantity": 8, "price": 80}]},
            ],
        ]
    }
)


verbose_expr = (
    pl.col("orders")
    .list.eval(
        pl.element().struct.with_fields(
            pl.element()
            .struct.field("items")
            .list.eval(
                pl.element().struct.with_fields(
                    (pl.element().struct.field("quantity") * 2)
                )
            )
        )
    )
)

print(df.select(verbose_expr))
```

**With Nexpresso:**
```python
from nexpresso import generate_nested_exprs

nexpresso_expr = {
    'orders': {
        'items': {
            'quantity': lambda x: x * 2
        }
    }
}
exprs = generate_nested_exprs(nexpresso_expr, df.schema, struct_mode='with_fields')
print(df.select(exprs))
```

Much cleaner and easier to read! 🎉

## Installation

```bash
pip install polars-nexpresso
```

Or using `uv`:

```bash
uv add polars-nexpresso
```

## Quick Start

```python
import polars as pl
from nexpresso import generate_nested_exprs

# Create a DataFrame with nested structures
df = pl.DataFrame({
    "customer": [
        {"name": "Alice", "address": {"city": "NYC", "zip": "10001"}},
        {"name": "Bob", "address": {"city": "LA", "zip": "90001"}},
    ]
})

# Define operations on nested fields
fields = {
    "customer": {
        "name": None,  # Keep as-is 
        "address": {
            "city": None,  # Keep city
            "zip": lambda x: x.cast(pl.Int64),  # Transform zip to integer
            "state": pl.lit("Unknown"),  # Add new field
        },
    }
}

# Generate expressions and apply them
exprs = generate_nested_exprs(fields, df.schema, struct_mode="with_fields")
result = df.select(exprs)
```

## Core Concepts

### Field Value Types

When defining operations, you can use several types of values:

- **`None`**: Keep the field as-is (select it without modification)
- **`dict`**: Recursively process nested structures
- **`Callable`**: Apply a function to the field (e.g., `lambda x: x * 2`)
- **`pl.Expr`**: Use a full Polars expression to create/modify the field

### Struct Modes

- **`"select"`** (default): Only keep the fields specified in the dictionary
- **`"with_fields"`**: Keep all existing fields and add/modify only the specified ones. If this is used, the fields that are not specified will be kept as-is.

## Examples

### Lists of Structs

```python
df = pl.DataFrame({
    "order_id": [1, 2],
    "items": [
        [{"product": "Apple", "quantity": 5, "price": 1.0}],
        [{"product": "Banana", "quantity": 10, "price": 0.5}],
    ],
})

fields = {
    "order_id": None,
    "items": {
        "product": None,
        "quantity": lambda x: x * 2,  # Double quantity
        "price": None,
        "subtotal": pl.field("quantity") * pl.field("price"),  # Original qty * price
    },
}

exprs = generate_nested_exprs(fields, df.schema, struct_mode="with_fields")
result = df.select(exprs)
```

### Select Mode vs With Fields Mode

```python
df = pl.DataFrame({
    "product": [
        {"name": "Widget", "price": 10.0, "cost": 5.0, "stock": 100},
    ]
})

# Select mode: only keep specified fields
fields_select = {
    "product": {
        "name": None,
        "price": lambda x: x * 1.2,  # cost and stock are excluded
    }
}

# With fields mode: keep all fields, add/modify some
fields_with = {
    "product": {
        "price": lambda x: x * 1.2,  # Modify price
        "profit": pl.field("price") * 1.2 - pl.field("cost"),  # New field
        # name, cost, stock are kept as-is
    }
}

exprs_select = generate_nested_exprs(fields_select, df.schema, struct_mode="select")
exprs_with = generate_nested_exprs(fields_with, df.schema, struct_mode="with_fields")
```

### Convenience Function

```python
from nexpresso import apply_nested_operations

result = apply_nested_operations(
    df,
    {"data": {"value": lambda x: x * 2, "result": pl.field("value") * pl.field("multiplier")}},
    struct_mode="with_fields",
    use_with_columns=True,  # Use with_columns instead of select
)
```

## API Reference

### `generate_nested_exprs(fields, schema, struct_mode="select")`

Generate Polars expressions for nested data operations.

**Parameters:**

- **`fields`** (`dict[str, FieldValue]`): Dictionary defining operations on columns/fields
  - Keys are column/field names
  - Values specify the operation:
    - `None`: Select field as-is
    - `dict`: Recursively process nested structure
    - `Callable`: Apply function to field (e.g., `lambda x: x + 1`)
    - `pl.Expr`: Full expression to create/modify field

- **`schema`** (`pl.Schema`): The schema of the DataFrame to work with

- **`struct_mode`** (`Literal["select", "with_fields"]`): How to handle struct fields
  - `"select"`: Only keep specified fields (default)
  - `"with_fields"`: Keep all existing fields and add/modify specified ones

**Returns:**

- `list[pl.Expr]`: List of Polars expressions ready for use in `.select()` or `.with_columns()`

### `apply_nested_operations(df, fields, struct_mode="select", use_with_columns=False)`

Apply nested operations directly to a DataFrame or LazyFrame.

**Parameters:**

- **`df`** (`pl.DataFrame | pl.LazyFrame`): The DataFrame or LazyFrame to operate on
- **`fields`** (`dict[str, FieldValue]`): Dictionary defining operations (same as `generate_nested_exprs`)
- **`struct_mode`** (`Literal["select", "with_fields"]`): How to handle struct fields
- **`use_with_columns`** (`bool`): If `True`, use `.with_columns()` instead of `.select()`

**Returns:**

- `pl.DataFrame | pl.LazyFrame`: DataFrame or LazyFrame with operations applied

## Performance

The library generates native Polars expressions, so performance is equivalent to writing expressions manually. All operations are lazy and benefit from Polars' query optimization.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Built for the [Polars](https://www.pola.rs/) data processing library. Special thanks to the Polars team for creating such an excellent tool.

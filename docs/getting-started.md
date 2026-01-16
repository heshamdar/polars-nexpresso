# Getting Started

This guide will help you get up and running with Polars Nexpresso.

## Installation

### Using pip

```bash
pip install polars-nexpresso
```

### Using uv (recommended)

```bash
uv add polars-nexpresso
```

### From source

```bash
git clone https://github.com/heshamdar/polars-nexpresso.git
cd polars-nexpresso
uv sync
```

## Requirements

- Python 3.10+
- Polars 1.20.0+

## Your First Transformation

Let's start with a simple example of transforming nested data.

### 1. Create Some Nested Data

```python
import polars as pl
from nexpresso import apply_nested_operations

# Sample e-commerce order data
df = pl.DataFrame({
    "order_id": [1001, 1002],
    "customer": [
        {"name": "Alice", "tier": "Gold"},
        {"name": "Bob", "tier": "Silver"},
    ],
    "items": [
        [
            {"product": "Laptop", "price": 999.99, "qty": 1},
            {"product": "Mouse", "price": 29.99, "qty": 2},
        ],
        [
            {"product": "Keyboard", "price": 79.99, "qty": 1},
        ],
    ],
})

print("Original data:")
print(df)
```

### 2. Define Transformations

Use a dictionary to specify what to do with each field:

```python
fields = {
    "customer": {
        # Add new field based on tier
        "discount": pl.when(pl.field("tier") == "Gold")
            .then(0.15)
            .when(pl.field("tier") == "Silver")
            .then(0.10)
            .otherwise(0.05),
    },
    "items": {
        # Calculate line total
        "total": pl.field("price") * pl.field("qty"),
    },
}
```

### 3. Apply Transformations

```python
result = apply_nested_operations(df, fields, struct_mode="with_fields")

print("Transformed data:")
print(result)
```

### 4. Examine the Results

```python
# Access nested fields
print(f"Alice's discount: {result['customer'][0]['discount']}")  # 0.15
print(f"Laptop total: {result['items'][0][0]['total']}")  # 999.99
```

## Your First Hierarchy

Now let's build a hierarchy from separate tables.

### 1. Create Normalized Tables

```python
from nexpresso import HierarchicalPacker, HierarchySpec, LevelSpec

# Like database tables
regions = pl.DataFrame({
    "id": ["west", "east"],
    "name": ["West Coast", "East Coast"],
})

stores = pl.DataFrame({
    "id": ["s1", "s2", "s3"],
    "name": ["SF Store", "LA Store", "NYC Store"],
    "region_id": ["west", "west", "east"],  # Foreign key
})
```

### 2. Define the Hierarchy

```python
spec = HierarchySpec.from_levels(
    LevelSpec(name="region", id_fields=["id"]),
    LevelSpec(name="store", id_fields=["id"], parent_keys=["region_id"]),
)

packer = HierarchicalPacker(spec)
```

### 3. Build the Hierarchy

```python
nested = packer.build_from_tables({
    "region": regions,
    "store": stores,
})

print("Nested hierarchy:")
print(nested)
# Each region now contains its stores as a nested list
```

### 4. Navigate the Hierarchy

```python
# Unpack to store level (flatten)
flat = packer.unpack(nested, "store")
print("Flattened to store level:")
print(flat)

# Pack back up
repacked = packer.pack(flat, "region")
```

## Next Steps

Now that you've seen the basics, explore:

- [Nested Expressions](concepts/nested-expressions.md) - Deep dive into the expression builder
- [Hierarchical Data](concepts/hierarchical-data.md) - Understanding pack/unpack
- [Common Patterns](guides/common-patterns.md) - Real-world recipes

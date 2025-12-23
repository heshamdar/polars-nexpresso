# Examples

This page contains runnable examples demonstrating Nexpresso's capabilities.

## Running the Examples

The repository includes a comprehensive examples file:

```bash
python examples.py
```

Or run specific module examples:

```bash
python -m nexpresso.hierarchical_packer
python -m nexpresso.expressions
```

---

## Nested Expression Examples

### Basic Transformation

```python
import polars as pl
from nexpresso import apply_nested_operations

df = pl.DataFrame({
    "product": [
        {"name": "Laptop", "price": 999.99, "qty": 2},
        {"name": "Mouse", "price": 29.99, "qty": 5},
    ]
})

# Add calculated field
result = apply_nested_operations(df, {
    "product": {
        "total": pl.field("price") * pl.field("qty"),
    }
}, struct_mode="with_fields")

print(result)
```

### List of Structs

```python
df = pl.DataFrame({
    "orders": [
        [
            {"item": "A", "price": 10.0, "qty": 2},
            {"item": "B", "price": 20.0, "qty": 1},
        ],
        [
            {"item": "C", "price": 15.0, "qty": 3},
        ],
    ]
})

# Transform each item in the list
result = apply_nested_operations(df, {
    "orders": {
        "subtotal": pl.field("price") * pl.field("qty"),
    }
}, struct_mode="with_fields")

# First order totals: [20.0, 20.0]
# Second order totals: [45.0]
```

### Conditional Transformations

```python
df = pl.DataFrame({
    "customer": [
        {"name": "Alice", "tier": "Gold", "years": 5},
        {"name": "Bob", "tier": "Silver", "years": 2},
        {"name": "Carol", "tier": "Bronze", "years": 1},
    ]
})

result = apply_nested_operations(df, {
    "customer": {
        "discount": pl.when(pl.field("tier") == "Gold")
            .then(0.15)
            .when(pl.field("tier") == "Silver")
            .then(0.10)
            .otherwise(0.05),
        "status": pl.when(pl.field("years") >= 5)
            .then(pl.lit("VIP"))
            .otherwise(pl.lit("Regular")),
    }
}, struct_mode="with_fields")
```

### Deeply Nested Structures

```python
df = pl.DataFrame({
    "company": [
        {
            "name": "Acme",
            "departments": [
                {
                    "name": "Engineering",
                    "employees": [
                        {"name": "Alice", "salary": 100000},
                        {"name": "Bob", "salary": 90000},
                    ]
                },
            ]
        },
    ]
})

result = apply_nested_operations(df, {
    "company": {
        "departments": {
            "employees": {
                "annual_bonus": pl.field("salary") * 0.1,
            }
        }
    }
}, struct_mode="with_fields")
```

---

## Hierarchical Packer Examples

### Basic Pack/Unpack

```python
from nexpresso import HierarchicalPacker, HierarchySpec, LevelSpec

spec = HierarchySpec(
    levels=[
        LevelSpec(name="country", id_fields=["code"]),
        LevelSpec(name="city", id_fields=["id"]),
    ]
)
packer = HierarchicalPacker(spec)

# Flat data
flat = pl.DataFrame({
    "country.code": ["US", "US", "CA"],
    "country.name": ["United States", "United States", "Canada"],
    "country.city.id": ["NYC", "LA", "TOR"],
    "country.city.name": ["New York", "Los Angeles", "Toronto"],
})

# Pack to country level
packed = packer.pack(flat, "country")
print("Packed:")
print(packed)

# Unpack back
unpacked = packer.unpack(packed, "city")
print("Unpacked:")
print(unpacked)
```

### Building from Database Tables

```python
spec = HierarchySpec.from_levels(
    LevelSpec(name="region", id_fields=["id"]),
    LevelSpec(name="store", id_fields=["id"], parent_keys=["region_id"]),
    LevelSpec(name="product", id_fields=["id"], parent_keys=["store_id"]),
)
packer = HierarchicalPacker(spec)

regions = pl.DataFrame({
    "id": ["west", "east"],
    "name": ["West Coast", "East Coast"],
})

stores = pl.DataFrame({
    "id": ["s1", "s2", "s3"],
    "name": ["SF Store", "LA Store", "NYC Store"],
    "region_id": ["west", "west", "east"],
})

products = pl.DataFrame({
    "id": ["p1", "p2", "p3"],
    "name": ["Laptop", "Phone", "Tablet"],
    "price": [999.0, 699.0, 499.0],
    "store_id": ["s1", "s1", "s3"],
})

nested = packer.build_from_tables({
    "region": regions,
    "store": stores,
    "product": products,
})

print("Nested hierarchy:")
print(nested)
```

### Normalize and Denormalize

```python
# Split into separate tables
tables = packer.normalize(nested)

print("Regions table:")
print(tables["region"])

print("Stores table:")
print(tables["store"])

print("Products table:")
print(tables["product"])

# Reconstruct
rebuilt = packer.denormalize(tables)
```

### Custom Separator

```python
spec = HierarchySpec(
    levels=[
        LevelSpec(name="folder", id_fields=["name"]),
        LevelSpec(name="file", id_fields=["name"]),
    ]
)

# Use "/" as separator (like file paths)
packer = HierarchicalPacker(spec, granularity_separator="/")

df = pl.DataFrame({
    "folder/name": ["docs", "docs", "images"],
    "folder/file/name": ["readme.txt", "notes.txt", "photo.jpg"],
})

packed = packer.pack(df, "folder")
```

### Validation

```python
# Enable validation
packer = HierarchicalPacker(spec, validate_on_pack=True)

# Check for null keys
bad_data = pl.DataFrame({
    "parent.id": ["p1", None, "p3"],  # Null key!
    "parent.child.id": ["c1", "c2", "c3"],
})

errors = packer.validate(bad_data, raise_on_error=False)
for error in errors:
    print(f"Error: {error}")
```

---

## Combined Examples

### E-Commerce Analytics Pipeline

```python
import polars as pl
from nexpresso import (
    HierarchicalPacker,
    HierarchySpec,
    LevelSpec,
    apply_nested_operations,
)

# 1. Load data
regions = pl.DataFrame({
    "id": ["west", "east"],
    "name": ["West Coast", "East Coast"],
})

stores = pl.DataFrame({
    "id": ["s1", "s2"],
    "name": ["SF Store", "NYC Store"],
    "region_id": ["west", "east"],
})

products = pl.DataFrame({
    "id": ["p1", "p2", "p3"],
    "name": ["Laptop", "Phone", "Tablet"],
    "price": [999.0, 699.0, 499.0],
    "cost": [600.0, 400.0, 300.0],
    "units": [100, 200, 150],
    "store_id": ["s1", "s1", "s2"],
})

# 2. Build hierarchy
spec = HierarchySpec.from_levels(
    LevelSpec(name="region", id_fields=["id"]),
    LevelSpec(name="store", id_fields=["id"], parent_keys=["region_id"]),
    LevelSpec(name="product", id_fields=["id"], parent_keys=["store_id"]),
)
packer = HierarchicalPacker(spec)

nested = packer.build_from_tables({
    "region": regions,
    "store": stores,
    "product": products,
})

# 3. Add calculations
enriched = apply_nested_operations(nested, {
    "region": {
        "store": {
            "product": {
                "revenue": pl.field("price") * pl.field("units"),
                "profit": (pl.field("price") - pl.field("cost")) * pl.field("units"),
            }
        }
    }
}, struct_mode="with_fields")

# 4. Analyze
flat = packer.unpack(enriched, "product")

by_region = (
    flat
    .group_by("region.name")
    .agg([
        pl.col("region.store.product.revenue").sum().alias("total_revenue"),
        pl.col("region.store.product.profit").sum().alias("total_profit"),
    ])
    .with_columns([
        (pl.col("total_profit") / pl.col("total_revenue") * 100)
            .round(1)
            .alias("margin_pct")
    ])
)

print("Revenue by Region:")
print(by_region)
```

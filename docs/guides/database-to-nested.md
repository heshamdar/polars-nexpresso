# Building from Database Tables

This guide shows how to build nested hierarchical structures from normalized database tables.

## The Scenario

You have data in separate tables with foreign key relationships:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    regions      │     │     stores      │     │    products     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ id      (PK)    │◄────│ region_id (FK)  │     │ store_id  (FK)  │────►│
│ name            │     │ id       (PK)   │◄────│ id        (PK)  │     │
│ manager         │     │ name            │     │ name            │     │
└─────────────────┘     │ square_feet     │     │ price           │     │
                        └─────────────────┘     │ cost            │     │
                                                └─────────────────┘
```

You want to transform this into a nested structure:

```
Region
└── Store
    └── Product
```

## Step 1: Create Your Tables

```python
import polars as pl
from nexpresso import HierarchicalPacker, HierarchySpec, LevelSpec

# Regions table
regions = pl.DataFrame({
    "id": ["west", "east", "central"],
    "name": ["West Coast", "East Coast", "Central"],
    "manager": ["Alice", "Bob", "Charlie"],
})

# Stores table (with foreign key to regions)
stores = pl.DataFrame({
    "id": ["s1", "s2", "s3", "s4"],
    "name": ["SF Downtown", "LA Beach", "NYC Times Sq", "Chicago Loop"],
    "square_feet": [5000, 3500, 8000, 4500],
    "region_id": ["west", "west", "east", "central"],  # FK to regions.id
})

# Products table (with foreign key to stores)
products = pl.DataFrame({
    "id": ["p1", "p2", "p3", "p4", "p5"],
    "name": ["Laptop", "Phone", "Tablet", "Monitor", "Keyboard"],
    "price": [999.0, 699.0, 499.0, 399.0, 99.0],
    "cost": [600.0, 400.0, 300.0, 250.0, 50.0],
    "store_id": ["s1", "s1", "s2", "s3", "s4"],  # FK to stores.id
})
```

## Step 2: Define the Hierarchy

Use `HierarchySpec.from_levels()` with `parent_keys` to specify foreign key relationships:

```python
spec = HierarchySpec.from_levels(
    # Root level - no parent_keys
    LevelSpec(name="region", id_fields=["id"]),
    
    # Child level - parent_keys link to parent's id_fields
    LevelSpec(
        name="store", 
        id_fields=["id"], 
        parent_keys=["region_id"]  # Links to region.id
    ),
    
    # Grandchild level
    LevelSpec(
        name="product", 
        id_fields=["id"], 
        parent_keys=["store_id"]  # Links to store.id
    ),
)

packer = HierarchicalPacker(spec)
```

!!! note "parent_keys order matters"
    The order of `parent_keys` must match the order of the parent's `id_fields`.
    If the parent has `id_fields=["id1", "id2"]`, the child needs 
    `parent_keys=["fk1", "fk2"]` where `fk1` links to `id1` and `fk2` links to `id2`.

## Step 3: Build the Hierarchy

```python
nested = packer.build_from_tables({
    "region": regions,
    "store": stores,
    "product": products,
})

print(nested)
```

Output:
```
shape: (3, 1)
┌──────────────────────────────────────────────────────────────────────┐
│ region                                                               │
│ ---                                                                  │
│ struct[4]                                                            │
╞══════════════════════════════════════════════════════════════════════╡
│ {"west","West Coast","Alice",[{"s1","SF Downtown",5000,[{"p1","La... │
│ {"east","East Coast","Bob",[{"s3","NYC Times Sq",8000,[{"p4","Mon... │
│ {"central","Central","Charlie",[{"s4","Chicago Loop",4500,[{"p5",... │
└──────────────────────────────────────────────────────────────────────┘
```

## Step 4: Work with the Nested Data

### Unpack to See All Data

```python
flat = packer.unpack(nested, "product")
print(flat)
```

### Transform Nested Data

```python
from nexpresso import apply_nested_operations

# Add profit calculation to each product
result = apply_nested_operations(nested, {
    "region": {
        "id": None,
        "name": None,
        "manager": None,
        "store": {
            "id": None,
            "name": None,
            "square_feet": None,
            "product": {
                "id": None,
                "name": None,
                "price": None,
                "cost": None,
                "profit": pl.field("price") - pl.field("cost"),
                "margin_pct": (
                    (pl.field("price") - pl.field("cost")) / 
                    pl.field("price") * 100
                ).round(1),
            }
        }
    }
}, struct_mode="with_fields")
```

## Handling Join Types

By default, `build_from_tables` uses a left join:

```python
# Left join (default): Parents without children get null/empty lists
nested = packer.build_from_tables(tables, join_type="left")

# Inner join: Only parents with children are included
nested = packer.build_from_tables(tables, join_type="inner")
```

## Working with Existing Column Names

If your tables have different column naming conventions, use `prepare_level_table`:

```python
# Raw table with different column names
raw_products = pl.DataFrame({
    "product_sku": ["SKU001", "SKU002"],
    "product_name": ["Widget", "Gadget"],
    "unit_price": [9.99, 19.99],
    "store_id": ["s1", "s1"],
})

# Map to expected names
prepared = packer.prepare_level_table(
    "product",
    raw_products,
    column_mapping={
        "product_sku": "id",
        "product_name": "name",
        "unit_price": "price",
    },
)
```

## Complete Example

```python
import polars as pl
from nexpresso import (
    HierarchicalPacker, 
    HierarchySpec, 
    LevelSpec,
    apply_nested_operations,
)

# 1. Create tables
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
    "units_sold": [10, 20, 15],
    "store_id": ["s1", "s1", "s3"],
})

# 2. Define hierarchy
spec = HierarchySpec.from_levels(
    LevelSpec(name="region", id_fields=["id"]),
    LevelSpec(name="store", id_fields=["id"], parent_keys=["region_id"]),
    LevelSpec(name="product", id_fields=["id"], parent_keys=["store_id"]),
)

packer = HierarchicalPacker(spec)

# 3. Build hierarchy
nested = packer.build_from_tables({
    "region": regions,
    "store": stores,
    "product": products,
})

# 4. Add calculations
result = apply_nested_operations(nested, {
    "region": {
        "id": None,
        "name": None,
        "store": {
            "id": None,
            "name": None,
            "product": {
                "id": None,
                "name": None,
                "price": None,
                "units_sold": None,
                "revenue": pl.field("price") * pl.field("units_sold"),
            }
        }
    }
}, struct_mode="with_fields")

# 5. Analyze
flat = packer.unpack(result, "product")
total_revenue = flat.select(
    pl.col("region.store.product.revenue").sum()
).item()

print(f"Total Revenue: ${total_revenue:,.2f}")
```

# Analytics Workflow

This guide demonstrates a complete analytics pipeline using Nexpresso, from raw data to insights.

## The Scenario

You're an analyst at an e-commerce company with data across multiple systems. You need to:

1. Combine data from different sources
2. Calculate metrics at the product level
3. Aggregate to store and region levels
4. Generate a performance report

## Step 1: Load the Data

```python
import polars as pl
from nexpresso import (
    HierarchicalPacker,
    HierarchySpec,
    LevelSpec,
    apply_nested_operations,
)

# Region data (from CRM)
regions = pl.DataFrame({
    "id": ["west", "east", "central"],
    "name": ["West Coast", "East Coast", "Central"],
    "manager": ["Alice", "Bob", "Charlie"],
})

# Store data (from inventory system)
stores = pl.DataFrame({
    "id": ["s1", "s2", "s3", "s4", "s5"],
    "name": ["SF Downtown", "LA Beach", "NYC Times Sq", "Chicago Loop", "Denver Mall"],
    "square_feet": [5000, 3500, 8000, 4500, 3000],
    "opened_year": [2015, 2018, 2010, 2019, 2020],
    "region_id": ["west", "west", "east", "central", "central"],
})

# Product sales data (from POS system)
products = pl.DataFrame({
    "id": ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"],
    "name": ["Laptop", "Phone", "Tablet", "Monitor", "Keyboard", "Mouse", "Headphones", "Cable"],
    "category": ["Electronics", "Electronics", "Electronics", "Electronics", "Accessories", "Accessories", "Accessories", "Accessories"],
    "price": [999.0, 699.0, 499.0, 399.0, 99.0, 49.0, 149.0, 19.0],
    "cost": [600.0, 400.0, 300.0, 250.0, 50.0, 25.0, 75.0, 8.0],
    "units_sold": [150, 300, 200, 120, 400, 600, 250, 1000],
    "store_id": ["s1", "s1", "s2", "s3", "s3", "s4", "s5", "s5"],
})
```

## Step 2: Build the Hierarchy

```python
spec = HierarchySpec.from_levels(
    LevelSpec(name="region", id_fields=["id"]),
    LevelSpec(name="store", id_fields=["id"], parent_keys=["region_id"]),
    LevelSpec(name="product", id_fields=["id"], parent_keys=["store_id"]),
)

packer = HierarchicalPacker(spec)

# Build nested structure
nested = packer.build_from_tables({
    "region": regions,
    "store": stores,
    "product": products,
})

print(f"Built hierarchy with {nested.height} regions")
```

## Step 3: Calculate Product-Level Metrics

Use nested expressions to add calculations without unpacking:

```python
# Define product-level calculations
# With struct_mode="with_fields", we only need to specify the fields we're adding/modifying
# All existing fields are automatically preserved
fields = {
    "region": {
        "store": {
            "product": {
                # Add calculated fields - existing fields (id, name, price, etc.) are kept automatically
                "revenue": pl.field("price") * pl.field("units_sold"),
                "total_cost": pl.field("cost") * pl.field("units_sold"),
                "profit": (pl.field("price") - pl.field("cost")) * pl.field("units_sold"),
                "margin_pct": (
                    (pl.field("price") - pl.field("cost")) / pl.field("price") * 100
                ).round(1),
            }
        }
    }
}

enriched = apply_nested_operations(nested, fields, struct_mode="with_fields")
```

## Step 4: Aggregate to Store Level

Unpack and aggregate:

```python
# Unpack to product level for aggregation
flat = packer.unpack(enriched, "product")

# Aggregate to store level
store_metrics = (
    flat
    .group_by([
        "region.id",
        "region.name",
        "region.manager",
        "region.store.id",
        "region.store.name",
        "region.store.square_feet",
    ])
    .agg([
        pl.col("region.store.product.revenue").sum().alias("total_revenue"),
        pl.col("region.store.product.profit").sum().alias("total_profit"),
        pl.col("region.store.product.units_sold").sum().alias("total_units"),
        pl.len().alias("product_count"),
    ])
)

# Add store-level metrics
store_metrics = store_metrics.with_columns([
    (pl.col("total_revenue") / pl.col("region.store.square_feet"))
        .round(2)
        .alias("revenue_per_sqft"),
    (pl.col("total_profit") / pl.col("total_revenue") * 100)
        .round(1)
        .alias("profit_margin_pct"),
])

print("\n📊 Store Performance:")
print(store_metrics.sort("total_revenue", descending=True))
```

## Step 5: Aggregate to Region Level

```python
region_metrics = (
    store_metrics
    .group_by(["region.id", "region.name", "region.manager"])
    .agg([
        pl.col("total_revenue").sum().alias("region_revenue"),
        pl.col("total_profit").sum().alias("region_profit"),
        pl.col("total_units").sum().alias("region_units"),
        pl.col("region.store.id").count().alias("store_count"),
        pl.col("revenue_per_sqft").mean().round(2).alias("avg_revenue_per_sqft"),
    ])
    .with_columns([
        (pl.col("region_profit") / pl.col("region_revenue") * 100)
            .round(1)
            .alias("region_margin_pct"),
    ])
    .sort("region_revenue", descending=True)
)

print("\n🌍 Region Performance:")
print(region_metrics)
```

## Step 6: Generate Insights

```python
# Top performing region
top_region = region_metrics.row(0, named=True)
print(f"\n🏆 Top Region: {top_region['region.name']}")
print(f"   Revenue: ${top_region['region_revenue']:,.2f}")
print(f"   Profit Margin: {top_region['region_margin_pct']}%")

# Top performing store
top_store = store_metrics.sort("total_revenue", descending=True).row(0, named=True)
print(f"\n🏪 Top Store: {top_store['region.store.name']}")
print(f"   Revenue: ${top_store['total_revenue']:,.2f}")
print(f"   Revenue/sqft: ${top_store['revenue_per_sqft']}")

# Category breakdown
category_metrics = (
    flat
    .group_by("region.store.product.category")
    .agg([
        pl.col("region.store.product.revenue").sum().alias("category_revenue"),
        pl.col("region.store.product.profit").sum().alias("category_profit"),
    ])
    .sort("category_revenue", descending=True)
)

print("\n📦 Category Performance:")
print(category_metrics)
```

## Complete Script

Here's the entire workflow as a single script:

```python
#!/usr/bin/env python3
"""E-commerce analytics pipeline using Nexpresso."""

import polars as pl
from nexpresso import (
    HierarchicalPacker,
    HierarchySpec,
    LevelSpec,
    apply_nested_operations,
)


def main():
    # Load data
    regions = pl.DataFrame({
        "id": ["west", "east", "central"],
        "name": ["West Coast", "East Coast", "Central"],
        "manager": ["Alice", "Bob", "Charlie"],
    })
    
    stores = pl.DataFrame({
        "id": ["s1", "s2", "s3", "s4", "s5"],
        "name": ["SF Downtown", "LA Beach", "NYC Times Sq", "Chicago Loop", "Denver Mall"],
        "square_feet": [5000, 3500, 8000, 4500, 3000],
        "region_id": ["west", "west", "east", "central", "central"],
    })
    
    products = pl.DataFrame({
        "id": ["p1", "p2", "p3", "p4", "p5"],
        "name": ["Laptop", "Phone", "Tablet", "Monitor", "Keyboard"],
        "price": [999.0, 699.0, 499.0, 399.0, 99.0],
        "cost": [600.0, 400.0, 300.0, 250.0, 50.0],
        "units_sold": [150, 300, 200, 120, 400],
        "store_id": ["s1", "s1", "s2", "s3", "s4"],
    })
    
    # Build hierarchy
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
    
    # Add metrics - with_fields mode preserves all existing fields automatically
    enriched = apply_nested_operations(nested, {
        "region": {
            "store": {
                "product": {
                    "revenue": pl.field("price") * pl.field("units_sold"),
                    "profit": (pl.field("price") - pl.field("cost")) * pl.field("units_sold"),
                }
            }
        }
    }, struct_mode="with_fields")
    
    # Analyze
    flat = packer.unpack(enriched, "product")
    
    total_revenue = flat["region.store.product.revenue"].sum()
    total_profit = flat["region.store.product.profit"].sum()
    
    print(f"📊 Summary")
    print(f"   Total Revenue: ${total_revenue:,.2f}")
    print(f"   Total Profit: ${total_profit:,.2f}")
    print(f"   Overall Margin: {total_profit/total_revenue*100:.1f}%")


if __name__ == "__main__":
    main()
```

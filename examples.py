#!/usr/bin/env python3
"""
Polars Nexpresso - Comprehensive Examples
==========================================

This file demonstrates the complete workflow of polars-nexpresso, showing how to:

1. Build hierarchical data from normalized database tables
2. Pack/unpack data at different granularities
3. Transform nested fields using intuitive expressions
4. Split and normalize hierarchical data

The Scenario: E-Commerce Analytics
----------------------------------
We'll model an e-commerce system with the following hierarchy:

    Region → Store → Product (with sales data)

Starting from separate "database tables", we'll:
- Join them into a nested hierarchy
- Analyze and transform the data at various levels
- Use nested expressions for complex calculations

To run: python examples.py
"""

import polars as pl

from nexpresso import (
    HierarchicalPacker,
    HierarchySpec,
    LevelSpec,
    apply_nested_operations,
    generate_nested_exprs,
)
from nexpresso.hierarchical_packer import FrameT

# Configure Polars display for better output
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(200)
pl.Config.set_fmt_str_lengths(100)


def print_section(title: str, description: str = "") -> None:
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)
    if description:
        print(f"\n{description}\n")


def print_subsection(title: str) -> None:
    """Print a subsection header."""
    print(f"\n--- {title} ---\n")


# =============================================================================
# PART 1: THE DATA - Simulating Database Tables
# =============================================================================


def create_database_tables() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Create sample data representing normalized database tables.

    In a real scenario, these might come from:
    - PostgreSQL/MySQL queries
    - CSV files
    - API responses
    - Data warehouse exports
    """
    # Regions table (parent level)
    regions = pl.DataFrame(
        {
            "id": ["west", "east", "central"],
            "name": ["West Coast", "East Coast", "Central"],
            "timezone": ["PST", "EST", "CST"],
            "manager": ["Alice", "Bob", "Charlie"],
        }
    )

    # Stores table (child of region)
    stores = pl.DataFrame(
        {
            "id": ["s1", "s2", "s3", "s4", "s5"],
            "name": ["SF Downtown", "LA Beach", "NYC Times Sq", "Chicago Loop", "Denver Mall"],
            "square_feet": [5000, 3500, 8000, 4500, 3000],
            "opened_year": [2015, 2018, 2010, 2019, 2020],
            "region_id": ["west", "west", "east", "central", "central"],  # Foreign key
        }
    )

    # Products table (child of store) - with sales data
    products = pl.DataFrame(
        {
            "id": ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"],
            "name": [
                "Laptop",
                "Phone",
                "Tablet",
                "Laptop",
                "Phone",
                "Laptop",
                "Monitor",
                "Keyboard",
                "Mouse",
                "Headphones",
            ],
            "category": [
                "Electronics",
                "Electronics",
                "Electronics",
                "Electronics",
                "Electronics",
                "Electronics",
                "Electronics",
                "Accessories",
                "Accessories",
                "Accessories",
            ],
            "price": [
                999.99,
                699.99,
                449.99,
                1099.99,
                799.99,
                899.99,
                349.99,
                129.99,
                49.99,
                199.99,
            ],
            "cost": [700.00, 450.00, 300.00, 750.00, 500.00, 600.00, 200.00, 70.00, 25.00, 100.00],
            "units_sold": [150, 300, 200, 120, 250, 180, 90, 400, 600, 150],
            "store_id": ["s1", "s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4", "s5"],  # Foreign key
        }
    )

    return regions, stores, products


# =============================================================================
# PART 2: BUILDING THE HIERARCHY
# =============================================================================


def demonstrate_hierarchy_building():
    """Demonstrate building a nested hierarchy from flat tables."""
    print_section(
        "PART 1: Building a Hierarchy from Database Tables",
        "We start with three separate tables (like in a relational database)\n"
        "and combine them into a nested hierarchical structure.",
    )

    regions, stores, products = create_database_tables()

    print_subsection("Input Tables (Normalized Data)")
    print("REGIONS table:")
    print(regions)
    print("\nSTORES table:")
    print(stores)
    print("\nPRODUCTS table:")
    print(products)

    # Define the hierarchy with explicit parent-child relationships
    spec = HierarchySpec.from_levels(
        LevelSpec(name="region", id_fields=["id"]),
        LevelSpec(name="store", id_fields=["id"], parent_keys=["region_id"]),
        LevelSpec(name="product", id_fields=["id"], parent_keys=["store_id"]),
    )

    packer = HierarchicalPacker(spec)

    # Build the nested structure
    nested = packer.build_from_tables(
        {
            "region": regions,
            "store": stores,
            "product": products,
        }
    )

    print_subsection("Result: Nested Hierarchy")
    print("Data is now nested: Region → Store → Product")
    print(nested)

    return nested, packer


# =============================================================================
# PART 3: WORKING WITH HIERARCHICAL DATA
# =============================================================================


def demonstrate_pack_unpack(nested: pl.DataFrame, packer: HierarchicalPacker):
    """Demonstrate packing and unpacking operations."""
    print_section(
        "PART 2: Packing and Unpacking",
        "Navigate between different levels of granularity.\n"
        "Packing aggregates child rows into nested structures.\n"
        "Unpacking explodes nested structures into flat rows.",
    )

    print_subsection("Unpack to Product Level (Finest Granularity)")
    print("Each row represents one product with all parent data included:")
    flat = packer.unpack(nested, "product")
    print(flat)

    print_subsection("Pack to Store Level")
    print("Products are nested within each store:")
    store_level = packer.pack(flat, "store")
    print(store_level)

    print_subsection("Pack to Region Level (Coarsest Granularity)")
    print("Stores (with their products) are nested within each region:")
    region_level = packer.pack(flat, "region")
    print(region_level)

    return flat


# =============================================================================
# PART 4: NESTED EXPRESSIONS - THE POWER OF NEXPRESSO
# =============================================================================


def demonstrate_nested_expressions(flat: pl.DataFrame):
    """Demonstrate nested expression capabilities."""
    print_section(
        "PART 3: Transforming Nested Data with Nexpresso",
        "Use intuitive dictionary syntax to define transformations\n"
        "on deeply nested fields. Much cleaner than raw Polars expressions!",
    )

    # First, let's create a nicely nested structure to work with
    spec = HierarchySpec(
        levels=[
            LevelSpec(name="region", id_fields=["id"]),
            LevelSpec(name="store", id_fields=["id"]),
            LevelSpec(name="product", id_fields=["id"]),
        ]
    )
    packer = HierarchicalPacker(spec)
    nested = packer.pack(flat, "region")

    print_subsection("Starting Data (Region Level)")
    print(nested)

    # Example 1: Calculate metrics at the product level
    print_subsection("Example 1: Calculate Product Metrics")
    print("Adding profit margin and revenue calculations to each product:\n")

    fields = {
        "region": {
            "store": {
                "product": {
                    # Keep existing fields
                    "name": None,
                    "category": None,
                    "price": None,
                    "cost": None,
                    "units_sold": None,
                    # Calculate new metrics
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

    result = apply_nested_operations(nested, fields, struct_mode="with_fields")
    print("Result with calculated fields:")
    print(result)

    # Unpack to see the product details
    product_metrics = packer.unpack(result, "product")
    print("\nUnpacked to see all product metrics:")
    print(
        product_metrics.select(
            [
                "region.id",
                "region.store.id",
                "region.store.product.name",
                "region.store.product.revenue",
                "region.store.product.profit",
                "region.store.product.margin_pct",
            ]
        )
    )

    return result


def demonstrate_conditional_transformations():
    """Demonstrate conditional logic in nested expressions."""
    print_section(
        "PART 4: Conditional Transformations",
        "Apply business logic using when/then/otherwise expressions\n" "within nested structures.",
    )

    # Create order data with customer tiers
    orders = pl.DataFrame(
        {
            "order_id": [1001, 1002, 1003],
            "customer": [
                {"name": "Alice", "tier": "Gold", "years_member": 5},
                {"name": "Bob", "tier": "Silver", "years_member": 2},
                {"name": "Charlie", "tier": "Bronze", "years_member": 1},
            ],
            "items": [
                [
                    {"product": "Laptop", "price": 999.99, "qty": 1},
                    {"product": "Mouse", "price": 49.99, "qty": 2},
                ],
                [
                    {"product": "Keyboard", "price": 129.99, "qty": 1},
                ],
                [
                    {"product": "Monitor", "price": 349.99, "qty": 2},
                    {"product": "Cable", "price": 19.99, "qty": 3},
                ],
            ],
        }
    )

    print_subsection("Input: Order Data")
    print(orders)

    # Apply tiered discounts
    fields = {
        "order_id": None,
        "customer": {
            "name": None,
            "tier": None,
            "years_member": None,
            # Calculate discount based on tier
            "discount_pct": pl.when(pl.field("tier") == "Gold")
            .then(15)
            .when(pl.field("tier") == "Silver")
            .then(10)
            .otherwise(5),
            # Loyalty bonus for long-term members
            "loyalty_bonus": pl.when(pl.field("years_member") >= 5)
            .then(pl.lit("VIP"))
            .when(pl.field("years_member") >= 3)
            .then(pl.lit("Preferred"))
            .otherwise(pl.lit("Standard")),
        },
        "items": {
            "product": None,
            "price": None,
            "qty": None,
            "line_total": pl.field("price") * pl.field("qty"),
            # Flag high-value items
            "is_high_value": pl.field("price") > 100,
        },
    }

    result = apply_nested_operations(orders, fields, struct_mode="with_fields")

    print_subsection("Result: Orders with Business Logic Applied")
    print(result)


def demonstrate_select_vs_with_fields():
    """Demonstrate the difference between select and with_fields modes."""
    print_section(
        "PART 5: Select Mode vs With Fields Mode",
        "- select mode: Only keep explicitly specified fields\n"
        "- with_fields mode: Keep all fields, add/modify specified ones",
    )

    data = pl.DataFrame(
        {
            "product": [
                {"name": "Widget", "price": 10.0, "cost": 5.0, "stock": 100, "sku": "W001"},
                {"name": "Gadget", "price": 20.0, "cost": 8.0, "stock": 50, "sku": "G001"},
            ]
        }
    )

    print_subsection("Input Data")
    print(data)

    # Select mode - only keep specified fields
    select_fields = {
        "product": {
            "name": None,
            "price": lambda x: x * 1.1,  # 10% price increase
            "profit": pl.field("price") - pl.field("cost"),
        }
    }

    result_select = apply_nested_operations(data, select_fields, struct_mode="select")
    print_subsection("Select Mode Result")
    print("Only name, price, and profit are kept (cost, stock, sku dropped):")
    print(result_select)

    # With fields mode - keep all, modify some
    with_fields = {
        "product": {
            "price": lambda x: x * 1.1,  # 10% price increase
            "profit": pl.field("price") - pl.field("cost"),
        }
    }

    result_with = apply_nested_operations(data, with_fields, struct_mode="with_fields")
    print_subsection("With Fields Mode Result")
    print("All original fields kept, price modified, profit added:")
    print(result_with)


# =============================================================================
# PART 5: NORMALIZE AND DENORMALIZE
# =============================================================================


def demonstrate_normalize_denormalize():
    """Demonstrate splitting and reconstructing hierarchical data."""
    print_section(
        "PART 6: Normalize and Denormalize",
        "Split hierarchical data into separate tables per level,\n"
        "then reconstruct the nested structure.",
    )

    # Create sample data
    regions, stores, products = create_database_tables()

    spec = HierarchySpec.from_levels(
        LevelSpec(name="region", id_fields=["id"]),
        LevelSpec(name="store", id_fields=["id"], parent_keys=["region_id"]),
        LevelSpec(name="product", id_fields=["id"], parent_keys=["store_id"]),
    )
    packer = HierarchicalPacker(spec)

    # Build nested structure
    nested = packer.build_from_tables(
        {
            "region": regions,
            "store": stores,
            "product": products,
        }
    )

    print_subsection("Input: Nested Hierarchy")
    print(nested)

    # Normalize - split into separate tables
    normalized = packer.normalize(nested)

    print_subsection("Normalized: Separate Tables Per Level")
    for level_name, table in normalized.items():
        print(f"\n{level_name.upper()} table:")
        print(table)

    # Denormalize - reconstruct
    reconstructed = packer.denormalize(normalized)

    print_subsection("Denormalized: Reconstructed Hierarchy")
    print(reconstructed)


# =============================================================================
# PART 6: VALIDATION
# =============================================================================


def demonstrate_validation():
    """Demonstrate validation features."""
    print_section(
        "PART 7: Data Validation",
        "HierarchicalPacker can validate data integrity:\n"
        "- Check for null values in key columns\n"
        "- Ensure uniform values when aggregating",
    )

    spec = HierarchySpec(
        levels=[
            LevelSpec(name="parent", id_fields=["id"]),
            LevelSpec(name="child", id_fields=["id"]),
        ]
    )
    packer = HierarchicalPacker(spec)

    # Data with null key
    bad_data = pl.DataFrame(
        {
            "parent.id": ["p1", None, "p3"],  # Null in key!
            "parent.child.id": ["c1", "c2", "c3"],
            "parent.child.value": [10, 20, 30],
        }
    )

    print_subsection("Data with Null Key")
    print(bad_data)

    # Validate without raising
    errors = packer.validate(bad_data, raise_on_error=False)
    print("\nValidation errors found:")
    for error in errors:
        print(f"  - {error}")


# =============================================================================
# PART 7: REAL-WORLD WORKFLOW
# =============================================================================


def demonstrate_complete_workflow():
    """Demonstrate a complete real-world analytics workflow."""
    print_section(
        "PART 8: Complete Analytics Workflow",
        "A realistic end-to-end example combining all features:\n"
        "1. Load data from 'database tables'\n"
        "2. Build nested hierarchy\n"
        "3. Calculate metrics at multiple levels\n"
        "4. Apply business rules\n"
        "5. Generate reports at different granularities",
    )

    # Step 1: Load data
    print_subsection("Step 1: Load Data")
    regions, stores, products = create_database_tables()
    print(f"Loaded {len(regions)} regions, {len(stores)} stores, {len(products)} products")

    # Step 2: Build hierarchy
    print_subsection("Step 2: Build Hierarchy")
    spec = HierarchySpec.from_levels(
        LevelSpec(name="region", id_fields=["id"]),
        LevelSpec(name="store", id_fields=["id"], parent_keys=["region_id"]),
        LevelSpec(name="product", id_fields=["id"], parent_keys=["store_id"]),
    )
    packer = HierarchicalPacker(spec)

    nested = packer.build_from_tables(
        {
            "region": regions,
            "store": stores,
            "product": products,
        }
    )
    print("Hierarchy built successfully")

    # Step 3: Unpack and calculate product-level metrics
    print_subsection("Step 3: Calculate Product Metrics")
    flat = packer.unpack(nested, "product")

    # Use nexpresso for cleaner transformations
    product_metrics = flat.with_columns(
        [
            (
                pl.col("region.store.product.price") * pl.col("region.store.product.units_sold")
            ).alias("revenue"),
            (
                (pl.col("region.store.product.price") - pl.col("region.store.product.cost"))
                * pl.col("region.store.product.units_sold")
            ).alias("profit"),
        ]
    )

    print("Product-level metrics calculated:")
    print(
        product_metrics.select(
            ["region.name", "region.store.name", "region.store.product.name", "revenue", "profit"]
        )
    )

    # Step 4: Aggregate to store level
    print_subsection("Step 4: Store-Level Summary")
    store_summary = (
        product_metrics.group_by(
            ["region.id", "region.name", "region.store.id", "region.store.name"]
        )
        .agg(
            [
                pl.col("revenue").sum().alias("total_revenue"),
                pl.col("profit").sum().alias("total_profit"),
                pl.col("region.store.product.units_sold").sum().alias("total_units"),
                pl.col("region.store.product.name").count().alias("product_count"),
            ]
        )
        .sort("total_revenue", descending=True)
    )

    print("Store performance ranking:")
    print(store_summary)

    # Step 5: Region-level summary
    print_subsection("Step 5: Region-Level Summary")
    region_summary = (
        store_summary.group_by(["region.id", "region.name"])
        .agg(
            [
                pl.col("total_revenue").sum().alias("region_revenue"),
                pl.col("total_profit").sum().alias("region_profit"),
                pl.col("total_units").sum().alias("region_units"),
                pl.col("region.store.id").count().alias("store_count"),
            ]
        )
        .sort("region_revenue", descending=True)
    )

    print("Region performance:")
    print(region_summary)

    print_subsection("Workflow Complete!")
    print(
        "This demonstrates how nexpresso enables clean, maintainable\n"
        "data pipelines for hierarchical data analysis."
    )


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("  POLARS NEXPRESSO - COMPREHENSIVE EXAMPLES")
    print("  Working with Hierarchical and Nested Data")
    print("=" * 80)

    # Part 1 & 2: Building hierarchy and pack/unpack
    nested, packer = demonstrate_hierarchy_building()
    flat = demonstrate_pack_unpack(nested, packer)

    # Part 3 & 4: Nested expressions
    demonstrate_nested_expressions(flat)
    demonstrate_conditional_transformations()

    # Part 5: Select vs with_fields
    demonstrate_select_vs_with_fields()

    # Part 6: Normalize/Denormalize
    demonstrate_normalize_denormalize()

    # Part 7: Validation
    demonstrate_validation()

    # Part 8: Complete workflow
    demonstrate_complete_workflow()

    print("\n" + "=" * 80)
    print("  ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
    print("  ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

"""
Integration tests for the polars-nexpresso library.

These tests verify end-to-end workflows that combine:
- HierarchicalPacker for building and navigating hierarchies
- NestedExpressionBuilder for transforming nested data
- Full analytics pipelines from raw data to insights
"""

from __future__ import annotations

import polars as pl
import pytest

from nexpresso import (
    HierarchicalPacker,
    HierarchySpec,
    LevelSpec,
    apply_nested_operations,
    generate_nested_exprs,
)

# =============================================================================
# End-to-End Workflow Tests
# =============================================================================


class TestEndToEndWorkflows:
    """Tests for complete data processing workflows."""

    def test_database_to_analytics_workflow(self) -> None:
        """
        Complete workflow: database tables → hierarchy → transform → aggregate.

        Simulates a real analytics scenario where:
        1. Data comes from normalized database tables
        2. Tables are joined into a hierarchy
        3. Calculations are performed on nested data
        4. Results are aggregated at different levels
        """
        # Step 1: Create normalized "database" tables
        regions = pl.DataFrame(
            {
                "id": ["west", "east"],
                "name": ["West Coast", "East Coast"],
                "manager": ["Alice", "Bob"],
            }
        )

        stores = pl.DataFrame(
            {
                "id": ["s1", "s2", "s3"],
                "name": ["SF Store", "LA Store", "NYC Store"],
                "region_id": ["west", "west", "east"],
                "square_feet": [5000, 3500, 8000],
            }
        )

        products = pl.DataFrame(
            {
                "id": ["p1", "p2", "p3", "p4", "p5"],
                "name": ["Laptop", "Phone", "Tablet", "Monitor", "Keyboard"],
                "price": [999.0, 699.0, 499.0, 399.0, 99.0],
                "cost": [600.0, 400.0, 300.0, 250.0, 50.0],
                "units_sold": [10, 20, 15, 8, 50],
                "store_id": ["s1", "s1", "s2", "s3", "s3"],
            }
        )

        # Step 2: Build hierarchy
        spec = HierarchySpec.from_levels(
            LevelSpec(name="region", id_fields=["id"]),
            LevelSpec(name="store", id_fields=["id"], parent_keys=["region_id"]),
            LevelSpec(name="product", id_fields=["id"], parent_keys=["store_id"]),
        )
        packer = HierarchicalPacker(spec)

        nested = packer.build_from_tables({"region": regions, "store": stores, "product": products})

        assert nested.height == 2  # 2 regions
        assert "region" in nested.columns

        # Step 3: Transform nested data with calculations
        fields = {
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
                        "units_sold": None,
                        "revenue": pl.field("price") * pl.field("units_sold"),
                        "profit": (pl.field("price") - pl.field("cost")) * pl.field("units_sold"),
                        "margin_pct": (
                            (pl.field("price") - pl.field("cost")) / pl.field("price") * 100
                        ).round(1),
                    },
                },
            }
        }

        result = apply_nested_operations(nested, fields, struct_mode="with_fields")

        # Step 4: Verify calculations by unpacking
        flat = packer.unpack(result, "product")

        # Verify revenue calculation
        laptop = flat.filter(pl.col("region.store.product.name") == "Laptop")
        assert laptop["region.store.product.revenue"][0] == 9990.0  # 999 * 10

        # Verify profit calculation
        assert laptop["region.store.product.profit"][0] == 3990.0  # (999 - 600) * 10

        # Verify margin calculation
        assert laptop["region.store.product.margin_pct"][0] == 39.9  # (399/999) * 100

        # Step 5: Aggregate to store level
        store_summary = (
            flat.group_by(["region.id", "region.store.id"])
            .agg(
                [
                    pl.col("region.store.product.revenue").sum().alias("total_revenue"),
                    pl.col("region.store.product.profit").sum().alias("total_profit"),
                    pl.len().alias("product_count"),
                ]
            )
            .sort("total_revenue", descending=True)
        )

        assert store_summary.height == 3  # 3 stores
        assert store_summary["product_count"].sum() == 5  # 5 products total

    def test_conditional_transformation_workflow(self) -> None:
        """
        Workflow with conditional business logic applied to nested data.
        """
        # Create order data
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
                        {"product": "Laptop", "price": 1000.0, "qty": 1},
                        {"product": "Mouse", "price": 50.0, "qty": 2},
                    ],
                    [
                        {"product": "Monitor", "price": 400.0, "qty": 1},
                    ],
                    [
                        {"product": "Keyboard", "price": 100.0, "qty": 2},
                        {"product": "Headphones", "price": 150.0, "qty": 1},
                    ],
                ],
            }
        )

        # Apply tiered discounts and loyalty bonuses
        fields = {
            "order_id": None,
            "customer": {
                "name": None,
                "tier": None,
                "years_member": None,
                "discount_pct": pl.when(pl.field("tier") == "Gold")
                .then(15)
                .when(pl.field("tier") == "Silver")
                .then(10)
                .otherwise(5),
                "loyalty_status": pl.when(pl.field("years_member") >= 5)
                .then(pl.lit("VIP"))
                .when(pl.field("years_member") >= 3)
                .then(pl.lit("Preferred"))
                .otherwise(pl.lit("Standard")),
            },
            "items": {
                "product": None,
                "price": None,
                "qty": None,
                "subtotal": pl.field("price") * pl.field("qty"),
            },
        }

        result = apply_nested_operations(orders, fields, struct_mode="with_fields")

        # Verify conditional logic
        alice_order = result.filter(pl.col("order_id") == 1001)
        assert alice_order["customer"][0]["discount_pct"] == 15  # Gold tier
        assert alice_order["customer"][0]["loyalty_status"] == "VIP"  # 5 years

        bob_order = result.filter(pl.col("order_id") == 1002)
        assert bob_order["customer"][0]["discount_pct"] == 10  # Silver tier
        assert bob_order["customer"][0]["loyalty_status"] == "Standard"  # 2 years

        charlie_order = result.filter(pl.col("order_id") == 1003)
        assert charlie_order["customer"][0]["discount_pct"] == 5  # Bronze tier

        # Verify item subtotals
        alice_items = alice_order["items"][0]
        assert alice_items[0]["subtotal"] == 1000.0  # 1000 * 1
        assert alice_items[1]["subtotal"] == 100.0  # 50 * 2

    def test_normalize_transform_denormalize_workflow(self) -> None:
        """
        Workflow: pack → normalize → transform individual tables → denormalize.
        """
        # Create flat data
        df = pl.DataFrame(
            {
                "company.id": ["c1", "c1", "c1", "c2"],
                "company.name": ["Acme", "Acme", "Acme", "Globex"],
                "company.department.id": ["d1", "d1", "d2", "d3"],
                "company.department.name": ["Eng", "Eng", "Sales", "Ops"],
                "company.department.employee.id": ["e1", "e2", "e3", "e4"],
                "company.department.employee.name": ["Alice", "Bob", "Carol", "Dave"],
                "company.department.employee.salary": [100000, 90000, 80000, 95000],
            }
        )

        spec = HierarchySpec(
            levels=[
                LevelSpec(name="company", id_fields=["id"]),
                LevelSpec(name="department", id_fields=["id"]),
                LevelSpec(name="employee", id_fields=["id"]),
            ]
        )
        packer = HierarchicalPacker(spec)

        # Step 1: Normalize (split into per-level tables)
        normalized = packer.normalize(df)

        assert "company" in normalized
        assert "department" in normalized
        assert "employee" in normalized
        assert normalized["company"].height == 2  # 2 companies
        assert normalized["department"].height == 3  # 3 departments
        assert normalized["employee"].height == 4  # 4 employees

        # Step 2: Transform employee table independently
        emp_table = normalized["employee"]
        emp_table = emp_table.with_columns(
            [
                (pl.col("company.department.employee.salary") * 1.1)
                .round(0)
                .alias("company.department.employee.salary"),  # 10% raise
                pl.lit("Active").alias("company.department.employee.status"),  # Add status
            ]
        )
        normalized["employee"] = emp_table

        # Step 3: Denormalize back to nested structure
        rebuilt = packer.denormalize(normalized)

        # Verify transformations persisted
        flat = packer.unpack(rebuilt, "employee")
        alice = flat.filter(pl.col("company.department.employee.name") == "Alice")
        assert alice["company.department.employee.salary"][0] == 110000  # 10% raise

    def test_lazyframe_throughout_workflow(self) -> None:
        """
        Verify that LazyFrame type is preserved throughout the entire workflow.
        """
        # Create LazyFrame
        lf = pl.DataFrame(
            {
                "parent.id": ["p1", "p1", "p2"],
                "parent.name": ["Parent 1", "Parent 1", "Parent 2"],
                "parent.child.id": ["c1", "c2", "c3"],
                "parent.child.value": [10, 20, 30],
            }
        ).lazy()

        spec = HierarchySpec(
            levels=[
                LevelSpec(name="parent", id_fields=["id"]),
                LevelSpec(name="child", id_fields=["id"]),
            ]
        )
        packer = HierarchicalPacker(spec)

        # Pack - should return LazyFrame
        packed = packer.pack(lf, "parent")
        assert isinstance(packed, pl.LazyFrame)

        # Unpack - should return LazyFrame
        unpacked = packer.unpack(packed, "child")
        assert isinstance(unpacked, pl.LazyFrame)

        # Apply nested operations - should return LazyFrame
        fields = {"parent.child.value": lambda x: x * 2}
        result = apply_nested_operations(unpacked, fields, struct_mode="with_fields")
        assert isinstance(result, pl.LazyFrame)

        # Only collect at the end
        final = result.collect()
        assert isinstance(final, pl.DataFrame)
        assert final["parent.child.value"].sum() == 120  # (10+20+30) * 2


# =============================================================================
# Expression Builder with Hierarchy Integration
# =============================================================================


class TestExpressionBuilderIntegration:
    """Tests for using expression builder on hierarchical data."""

    def test_generate_exprs_on_packed_data(self) -> None:
        """Test using generate_nested_exprs directly on packed hierarchical data."""
        spec = HierarchySpec(
            levels=[
                LevelSpec(name="category", id_fields=["id"]),
                LevelSpec(name="product", id_fields=["sku"]),
            ]
        )
        packer = HierarchicalPacker(spec)

        # Create and pack data
        df = pl.DataFrame(
            {
                "category.id": ["electronics", "electronics"],
                "category.name": ["Electronics", "Electronics"],
                "category.product.sku": ["E001", "E002"],
                "category.product.name": ["Laptop", "Phone"],
                "category.product.price": [1000.0, 500.0],
            }
        )

        packed = packer.pack(df, "category")

        # Generate expressions for transformation
        exprs = generate_nested_exprs(
            {
                "category": {
                    "id": None,
                    "name": lambda x: x.str.to_uppercase(),
                    "product": {
                        "sku": None,
                        "name": None,
                        "price": None,
                        "price_with_tax": pl.field("price") * 1.08,
                    },
                }
            },
            packed.schema,
            struct_mode="with_fields",
        )

        result = packed.select(exprs)

        # Verify
        category = result["category"][0]
        assert category["name"] == "ELECTRONICS"
        products = category["product"]
        assert products[0]["price_with_tax"] == 1080.0

    def test_multiple_transformations_same_data(self) -> None:
        """Test applying multiple different transformations to same nested data."""
        df = pl.DataFrame(
            {
                "metrics": [
                    {"views": 1000, "clicks": 50, "conversions": 10},
                    {"views": 2000, "clicks": 100, "conversions": 25},
                ]
            }
        )

        # First transformation: calculate rates
        rates = apply_nested_operations(
            df,
            {
                "metrics": {
                    "views": None,
                    "clicks": None,
                    "conversions": None,
                    "click_rate": pl.field("clicks") / pl.field("views") * 100,
                    "conv_rate": pl.field("conversions") / pl.field("clicks") * 100,
                }
            },
            struct_mode="with_fields",
        )

        # Verify rates
        assert rates["metrics"][0]["click_rate"] == 5.0  # 50/1000 * 100
        assert rates["metrics"][0]["conv_rate"] == 20.0  # 10/50 * 100

        # Second transformation: normalize values
        normalized = apply_nested_operations(
            rates,
            {
                "metrics": {
                    "views": None,
                    "clicks": None,
                    "conversions": None,
                    "click_rate": None,
                    "conv_rate": None,
                    "views_normalized": pl.field("views") / 1000,
                    "performance_score": (pl.field("click_rate") + pl.field("conv_rate")) / 2,
                }
            },
            struct_mode="with_fields",
        )

        assert normalized["metrics"][0]["views_normalized"] == 1.0
        assert normalized["metrics"][0]["performance_score"] == 12.5  # (5 + 20) / 2


# =============================================================================
# Real-World Scenario Tests
# =============================================================================


class TestRealWorldScenarios:
    """Tests simulating real-world use cases."""

    def test_ecommerce_order_processing(self) -> None:
        """Simulate e-commerce order processing pipeline."""
        # Raw order data (like from an API)
        orders = pl.DataFrame(
            {
                "order": [
                    {
                        "id": "ORD001",
                        "customer_id": "C100",
                        "status": "completed",
                        "items": [
                            {"sku": "SKU001", "name": "Widget", "qty": 2, "unit_price": 29.99},
                            {"sku": "SKU002", "name": "Gadget", "qty": 1, "unit_price": 49.99},
                        ],
                        "shipping": {"method": "express", "cost": 9.99},
                    },
                    {
                        "id": "ORD002",
                        "customer_id": "C101",
                        "status": "pending",
                        "items": [
                            {"sku": "SKU001", "name": "Widget", "qty": 5, "unit_price": 29.99},
                        ],
                        "shipping": {"method": "standard", "cost": 4.99},
                    },
                ]
            }
        )

        # Calculate order totals
        result = apply_nested_operations(
            orders,
            {
                "order": {
                    "id": None,
                    "customer_id": None,
                    "status": None,
                    "items": {
                        "sku": None,
                        "name": None,
                        "qty": None,
                        "unit_price": None,
                        "line_total": pl.field("qty") * pl.field("unit_price"),
                    },
                    "shipping": None,
                }
            },
            struct_mode="with_fields",
        )

        # Verify line totals
        order1 = result["order"][0]
        assert order1["items"][0]["line_total"] == 59.98  # 2 * 29.99
        assert order1["items"][1]["line_total"] == 49.99  # 1 * 49.99

        order2 = result["order"][1]
        assert order2["items"][0]["line_total"] == 149.95  # 5 * 29.99

    def test_geographic_data_analysis(self) -> None:
        """Test geographic hierarchy analysis (continent → country → city)."""
        spec = HierarchySpec.from_levels(
            LevelSpec(name="continent", id_fields=["code"]),
            LevelSpec(name="country", id_fields=["code"], parent_keys=["continent_code"]),
            LevelSpec(name="city", id_fields=["id"], parent_keys=["country_code"]),
        )
        packer = HierarchicalPacker(spec)

        continents = pl.DataFrame({"code": ["NA", "EU"], "name": ["North America", "Europe"]})

        countries = pl.DataFrame(
            {
                "code": ["US", "CA", "UK", "DE"],
                "name": ["United States", "Canada", "United Kingdom", "Germany"],
                "continent_code": ["NA", "NA", "EU", "EU"],
            }
        )

        cities = pl.DataFrame(
            {
                "id": ["NYC", "LA", "TOR", "LON", "BER"],
                "name": ["New York", "Los Angeles", "Toronto", "London", "Berlin"],
                "population": [8_000_000, 4_000_000, 3_000_000, 9_000_000, 3_500_000],
                "country_code": ["US", "US", "CA", "UK", "DE"],
            }
        )

        # Build hierarchy
        geo = packer.build_from_tables(
            {"continent": continents, "country": countries, "city": cities}
        )

        assert geo.height == 2  # 2 continents

        # Unpack to city level for analysis
        flat = packer.unpack(geo, "city")
        assert flat.height == 5  # 5 cities

        # Aggregate by continent
        continent_pop = (
            flat.group_by("continent.code")
            .agg(pl.col("continent.country.city.population").sum().alias("total_population"))
            .sort("continent.code")
        )

        # EU: London (9M) + Berlin (3.5M) = 12.5M
        eu_pop = continent_pop.filter(pl.col("continent.code") == "EU")
        assert eu_pop["total_population"][0] == 12_500_000

        # NA: NYC (8M) + LA (4M) + TOR (3M) = 15M
        na_pop = continent_pop.filter(pl.col("continent.code") == "NA")
        assert na_pop["total_population"][0] == 15_000_000

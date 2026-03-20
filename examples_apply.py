#!/usr/bin/env python3
"""
Polars Nexpresso — apply() Examples
====================================

This script demonstrates the unified ``apply()`` method, which lets you
transform hierarchical data using the same dict-based field spec on both
the **nested** (HierarchicalPacker) and **normalized** (NormalizedPacker)
backends.

The Scenario: E-Commerce Analytics
-----------------------------------
We model a simple e-commerce hierarchy:

    Region → Store → Product

Each store has revenue and cost figures; each product has a price.
We use ``apply()`` to compute new columns, transform existing ones,
and navigate across hierarchy levels — all with the same syntax
regardless of how the data is physically stored.

To run::

    uv run python examples_apply.py
"""

import polars as pl

from nexpresso import (
    HierarchicalPacker,
    HierarchySpec,
    LevelSpec,
    NormalizedPacker,
)
from nexpresso.hierarchy_protocol import NestedBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def section(title: str, description: str = "") -> None:
    """Print a section header."""
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)
    if description:
        print(f"\n{description}\n")


def subsection(title: str) -> None:
    """Print a subsection header."""
    print(f"\n--- {title} ---\n")


# ---------------------------------------------------------------------------
# 1. SETUP — hierarchy spec & sample data
# ---------------------------------------------------------------------------


def create_data() -> (
    tuple[HierarchySpec, pl.DataFrame, pl.DataFrame, pl.DataFrame]
):
    """Create the hierarchy spec and sample data tables."""
    spec = HierarchySpec.from_levels(
        LevelSpec(name="region", id_fields=["id"]),
        LevelSpec(name="store", id_fields=["id"], parent_keys=["region_id"]),
        LevelSpec(name="product", id_fields=["id"], parent_keys=["store_id"]),
    )

    regions = pl.DataFrame({
        "id": ["r1", "r2"],
        "name": ["North", "South"],
    })

    stores = pl.DataFrame({
        "id": ["s1", "s2", "s3"],
        "name": ["Downtown", "Mall", "Airport"],
        "revenue": [500_000, 300_000, 400_000],
        "cost": [350_000, 200_000, 310_000],
        "region_id": ["r1", "r1", "r2"],
    })

    products = pl.DataFrame({
        "id": ["p1", "p2", "p3", "p4", "p5"],
        "name": ["Laptop", "Phone", "Tablet", "Monitor", "Keyboard"],
        "price": [999.0, 699.0, 499.0, 349.0, 79.0],
        "store_id": ["s1", "s1", "s2", "s3", "s3"],
    })

    return spec, regions, stores, products


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    spec, regions, stores, products = create_data()

    # Build both backends from the same data
    npacker = NormalizedPacker(
        spec, tables={"region": regions, "store": stores, "product": products}
    )

    hpacker = HierarchicalPacker(spec)
    packed = hpacker.build_from_tables(
        {"region": regions, "store": stores, "product": products}
    )
    nested_backend = NestedBackend(hpacker, packed)

    # ------------------------------------------------------------------
    # 2. BASIC apply() — flat field specs at a single level
    # ------------------------------------------------------------------
    section(
        "2. Basic apply() — Flat Field Specs",
        "Transform columns at a single hierarchy level using lambdas,\n"
        "pl.field() expressions, and None (keep-as-is).",
    )

    subsection("2a. Lambda — 10% revenue increase")
    result = (
        npacker.apply({"revenue": lambda x: x * 1.1}, at_level="store")
        .collect()
        .sort("region.store.id")
    )
    print("Original revenues:", stores["revenue"].to_list())
    print("After 10% bump:  ", result["region.store.revenue"].to_list())

    subsection("2b. pl.field() — compute a new 'margin' column")
    result = (
        npacker.apply(
            {"margin": pl.field("revenue") - pl.field("cost")},
            at_level="store",
        )
        .collect()
        .sort("region.store.id")
    )
    print("Revenues:", stores["revenue"].to_list())
    print("Costs:   ", stores["cost"].to_list())
    print("Margins: ", result["region.store.margin"].to_list())

    subsection("2c. Multiple operations in one call")
    result = (
        npacker.apply(
            {
                "revenue": lambda x: x * 1.1,       # bump revenue
                "margin": pl.field("revenue") - pl.field("cost"),  # add margin
            },
            at_level="store",
        )
        .collect()
        .sort("region.store.id")
    )
    print("Bumped revenues:", result["region.store.revenue"].to_list())
    print("Margins:        ", result["region.store.margin"].to_list())

    # ------------------------------------------------------------------
    # 3. NESTED DICT apply() — cross-level navigation
    # ------------------------------------------------------------------
    section(
        "3. Nested Dict apply() — Cross-Level Navigation",
        "Navigate into child levels using nested dicts.\n"
        "On the nested backend this traverses List[Struct] columns;\n"
        "on the normalized backend it routes to child tables and joins.",
    )

    subsection("3a. Reach into store level from region")
    print(
        'Spec: {"store": {"revenue": lambda x: x * 2}}  at_level="region"\n'
    )
    result = (
        npacker.apply(
            {"store": {"revenue": lambda x: x * 2}},
            at_level="region",
        )
        .collect()
        .sort("region.store.id")
    )
    print("Original store revenues:", stores["revenue"].to_list())
    print("Doubled via nested dict:", result["region.store.revenue"].to_list())

    subsection("3b. Mixed levels — parent AND child in one spec")
    print(
        "Spec: region name → uppercase, store revenue → tripled\n"
    )
    result = (
        npacker.apply(
            {
                "name": lambda x: x.str.to_uppercase(),
                "store": {"revenue": lambda x: x * 3},
            },
            at_level="region",
        )
        .collect()
        .sort("region.store.id")
    )
    print("Region names: ", sorted(result["region.name"].unique().to_list()))
    print("Store revenues:", result["region.store.revenue"].to_list())

    subsection("3c. Deep nesting — region → store → product")
    print(
        'Spec: {"store": {"product": {"price": lambda x: x * 1.05}}}\n'
        "       (5% price increase on all products)\n"
    )
    result = (
        npacker.apply(
            {"store": {"product": {"price": lambda x: x * 1.05}}},
            at_level="region",
        )
        .collect()
        .sort("region.store.product.id")
    )
    print("Original prices:", products.sort("id")["price"].to_list())
    print(
        "After 5% hike:  ",
        [round(p, 2) for p in result["region.store.product.price"].to_list()],
    )

    # ------------------------------------------------------------------
    # 4. CROSS-BACKEND EQUIVALENCE
    # ------------------------------------------------------------------
    section(
        "4. Cross-Backend Equivalence",
        "The same field spec produces identical results whether the data\n"
        "is stored as nested structs or as normalized tables.",
    )

    fields = {
        "name": lambda x: x.str.to_uppercase(),
        "store": {"revenue": lambda x: x * 2},
    }
    print(
        "Field spec:\n"
        '  {"name": upper, "store": {"revenue": double}}\n'
        '  at_level="region"\n'
    )

    norm_result = (
        npacker.apply(fields, at_level="region")
        .collect()
        .sort("region.store.id")
    )
    nested_result = (
        nested_backend.apply(fields, at_level="region")
        .collect()
        .sort("region.store.id")
    )

    norm_rev = norm_result["region.store.revenue"].to_list()
    nested_rev = nested_result["region.store.revenue"].to_list()
    print("Normalized backend revenues:", norm_rev)
    print("Nested backend revenues:    ", nested_rev)
    print("Match:", norm_rev == nested_rev)

    norm_names = sorted(norm_result["region.name"].unique().to_list())
    nested_names = sorted(nested_result["region.name"].unique().to_list())
    print("Normalized backend names:   ", norm_names)
    print("Nested backend names:       ", nested_names)
    print("Match:", norm_names == nested_names)

    # ------------------------------------------------------------------
    # 5. HIERARCHY PRESERVATION (HierarchicalPacker.apply)
    # ------------------------------------------------------------------
    section(
        "5. Hierarchy Preservation with HierarchicalPacker",
        "HierarchicalPacker.apply() modifies data IN the packed structure.\n"
        "The result is still a valid packed frame that you can unpack,\n"
        "re-pack, or feed into promote_attribute / enrich.",
    )

    subsection("5a. Packed frame before apply()")
    print(packed)
    print(f"\nColumns: {packed.columns}")

    subsection("5b. Apply revenue doubling at store level")
    modified = hpacker.apply(
        packed,
        {"revenue": lambda x: x * 2},
        at_level="store",
    )
    print("Modified packed frame (same structure):")
    print(modified)
    print(f"\nStill has the same columns: {modified.columns}")

    subsection("5c. Unpack to verify the changes")
    unpacked = hpacker.unpack(modified, "store")
    if isinstance(unpacked, pl.LazyFrame):
        unpacked = unpacked.collect()
    unpacked = unpacked.sort("region.store.id")
    print("Original revenues:", stores["revenue"].to_list())
    print("After apply:      ", unpacked["region.store.revenue"].to_list())

    # ------------------------------------------------------------------
    # 6. PRACTICAL WORKFLOW — margins + promotion
    # ------------------------------------------------------------------
    section(
        "6. Practical Workflow — Margins & Promotion",
        "A realistic workflow:\n"
        "  1. Compute store margins via apply()\n"
        "  2. Promote total margin to the region level\n"
        "  3. Compare regions by profitability",
    )

    subsection("Step 1: Compute store margins")
    with_margins = (
        npacker.apply(
            {"margin": pl.field("revenue") - pl.field("cost")},
            at_level="store",
        )
        .collect()
        .sort("region.store.id")
    )
    print(with_margins.select(
        "region.name", "region.store.name",
        "region.store.revenue", "region.store.cost", "region.store.margin",
    ))

    subsection("Step 2: Promote total margin to region level")
    # Update the store table to include the margin column, then promote
    stores_with_margin = stores.with_columns(
        (pl.col("revenue") - pl.col("cost")).alias("margin")
    )
    packer_for_promo = NormalizedPacker(
        HierarchySpec.from_levels(
            LevelSpec(name="region", id_fields=["id"]),
            LevelSpec(name="store", id_fields=["id"], parent_keys=["region_id"]),
        ),
        tables={"region": regions, "store": stores_with_margin},
    )
    region_margins = (
        packer_for_promo.promote_attribute(
            "margin",
            from_level="store",
            to_level="region",
            agg="sum",
            alias="total_margin",
        )
        .collect()
        .sort("region.id")
    )
    print(region_margins.select("region.id", "region.name", "region.total_margin"))

    subsection("Step 3: Compare regions")
    for row in region_margins.iter_rows(named=True):
        print(f"  {row['region.name']:>6}: total margin = {row['region.total_margin']:>10,}")

    print("\nDone!")


if __name__ == "__main__":
    main()

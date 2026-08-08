#!/usr/bin/env python3
"""
Polars Nexpresso - HierarchyView Examples
=========================================

`HierarchyView` stores hierarchical data as one flat table per level and
presents it as if it were a single nested frame. This file demonstrates the
operations that are awkward or impossible with `List[Struct]` and `list.eval`,
and shows how they become ordinary expressions against a view.

The headline capability: **cross-level references**. Inside `list.eval` Polars
forbids named columns outright ("named columns are not allowed in `eval`
functions"), so a child value can never be combined with a parent attribute.
Part 1 demonstrates that failure directly. Every part after it does exactly
that, in one expression, because underneath it is a join.

The Scenario: Retail Sales
--------------------------

    Region → Store → Sale

    region.tax_rate       a rate that applies to every sale beneath it
    store.discount        a rate that applies to every sale in that store
    sale.amount, qty      the leaf facts

Computing a sale's final price needs all three levels at once.

To run: python examples_hierarchy_view.py
"""

import shutil
import tempfile
from pathlib import Path

import polars as pl

from nexpresso import HierarchicalPacker, HierarchySpec, HierarchyView, LevelSpec

pl.Config.set_tbl_rows(12)
pl.Config.set_tbl_cols(12)
pl.Config.set_tbl_width_chars(200)

# Column paths, spelled once.
REGION_ID = "region.id"
TAX = "region.tax_rate"
STORE_ID = "region.store.id"
DISCOUNT = "region.store.discount"
SALE_ID = "region.store.sale.id"
AMOUNT = "region.store.sale.amount"
QTY = "region.store.sale.qty"

SPEC = HierarchySpec.from_levels(
    LevelSpec(name="region", id_fields=["id"]),
    LevelSpec(name="store", id_fields=["id"], parent_keys=["region_id"]),
    LevelSpec(name="sale", id_fields=["id"], parent_keys=["store_id"]),
)


def header(title: str) -> None:
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def span(frame: pl.DataFrame, n: int = 6) -> pl.DataFrame:
    """
    Evenly spaced rows in sale order.

    Sampling with ``head`` would show only the first store, whose discount is
    zero — making every cross-level calculation look like a no-op. Spanning the
    frame shows rows from different parents.
    """
    ordered = frame.sort(SALE_ID)
    if ordered.height <= n:
        return ordered
    return ordered.gather_every(max(1, ordered.height // n)).head(n)


def build_source_data() -> pl.DataFrame:
    """A flat retail frame: 3 regions x 3 stores x 4 sales."""
    n_region, n_store, n_sale = 3, 3, 4
    rows = n_region * n_store * n_sale
    per_region = n_store * n_sale
    tax_by_region = [0.10, 0.20, 0.05]
    discount_by_store = [0.00, 0.15, 0.25]
    # 13 values against 12 rows per region, so each region gets a different mix
    # rather than an identical copy (which would make every rollup look equal).
    amounts = [12.0, 45.0, 8.0, 31.0, 67.0, 23.0, 90.0, 15.0, 54.0, 38.0, 71.0, 26.0, 83.0]

    return pl.DataFrame(
        {
            REGION_ID: [i // per_region for i in range(rows)],
            "region.name": [f"region-{i // per_region}" for i in range(rows)],
            TAX: [tax_by_region[i // per_region] for i in range(rows)],
            STORE_ID: [i // n_sale for i in range(rows)],
            "region.store.name": [f"store-{i // n_sale}" for i in range(rows)],
            DISCOUNT: [discount_by_store[(i // n_sale) % n_store] for i in range(rows)],
            SALE_ID: list(range(rows)),
            AMOUNT: [amounts[i % len(amounts)] for i in range(rows)],
            QTY: [1 + (i * 3) % 5 for i in range(rows)],
        }
    )


# =============================================================================
# Part 1: The limitation this exists to solve
# =============================================================================


def demonstrate_the_problem(flat: pl.DataFrame, packer: HierarchicalPacker) -> None:
    header("PART 1: Why cross-level references are hard with list.eval")

    nested = packer.pack(flat, "region")
    print("\nPacked shape — sale.amount is buried two lists deep:")
    print(f"  {nested.schema['region']}")

    print("\nGoal: net = sale.amount * (1 - store.discount)")
    print("      i.e. a LEAF value combined with its PARENT's attribute.\n")

    print("Attempt: reference the parent field from inside list.eval")
    try:
        nested.select(
            pl.col("region")
            .struct.field("store")
            .list.eval(
                pl.element()
                .struct.field("sale")
                .list.eval(
                    pl.element().struct.field("amount")
                    * pl.col("region").struct.field("store").struct.field("discount")
                )
            )
        )
        print("  ... unexpectedly succeeded")
    except Exception as exc:
        print(f"  {type(exc).__name__}: {str(exc).splitlines()[0]}")

    print("\nPolars is explicit: an eval context sees only `element()`. There is no")
    print("outer scope, so the parent's discount is simply unreachable. Working")
    print("around it means unpacking, computing, and repacking by hand.")


# =============================================================================
# Part 2: Setting up a view
# =============================================================================


def build_view(flat: pl.DataFrame, packer: HierarchicalPacker, warehouse: Path) -> HierarchyView:
    header("PART 2: Storing the hierarchy as one table per level")

    HierarchyView.from_frame(flat, packer).sink_parquet(warehouse)
    print(f"\nWrote {warehouse}:")
    for path in sorted(warehouse.glob("*.parquet")):
        table = pl.read_parquet(path)
        print(f"  {path.name:<16} {table.height:>4} rows x {table.width} cols   {table.columns}")

    view = HierarchyView.scan_parquet(warehouse, packer)
    print(f"\n{view!r}")
    print("\nEach level is a real top-level Parquet table — its own row groups,")
    print("statistics and sort order — but the view still presents a nested schema:")
    print(f"  {view.schema['region']}")
    return view


# =============================================================================
# Part 3: Cross-level expressions
# =============================================================================


def demonstrate_cross_level(view: HierarchyView) -> None:
    header("PART 3: Cross-level expressions — parent, grandparent, all at once")

    print("\n(a) Leaf x PARENT attribute — the expression Part 1 could not write:")
    net = view.with_columns(
        (pl.col(AMOUNT) * (1 - pl.col(DISCOUNT))).alias("region.store.sale.net")
    )
    print(span(net.collect("sale")).select(STORE_ID, DISCOUNT, AMOUNT, "region.store.sale.net"))

    print("\n(b) Leaf x GRANDPARENT attribute — two levels up, same syntax:")
    taxed = view.with_columns((pl.col(AMOUNT) * pl.col(TAX)).alias("region.store.sale.tax"))
    print(span(taxed.collect("sale")).select(REGION_ID, TAX, AMOUNT, "region.store.sale.tax"))

    print("\n(c) All three levels in ONE expression:")
    final = view.with_columns(
        (pl.col(AMOUNT) * (1 - pl.col(DISCOUNT)) * (1 + pl.col(TAX))).alias(
            "region.store.sale.final"
        )
    )
    print(span(final.collect("sale")).select(TAX, DISCOUNT, AMOUNT, "region.store.sale.final"))
    print("\nThe view resolves each column to its owning level, joins in whatever")
    print("the deepest level is missing, evaluates, and drops the borrowed columns.")


# =============================================================================
# Part 4: Aggregating down and pushing back up
# =============================================================================


def demonstrate_rollup_and_share(view: HierarchyView) -> None:
    header("PART 4: Roll up to a parent, then reference it from the child")

    print("\n'What share of its store's revenue is each sale?' — a child->parent")
    print("aggregate, then read back down at child granularity:\n")
    share = view.promote(
        "amount", from_level="sale", to_level="store", agg="sum", alias="revenue"
    ).with_columns(
        (pl.col(AMOUNT) / pl.col("region.store.revenue")).alias("region.store.sale.share")
    )
    print(
        span(share.collect("sale")).select(
            STORE_ID, AMOUNT, "region.store.revenue", "region.store.sale.share"
        )
    )

    print("\nThe same question via a window over the parent key — no join at all,")
    print("because normalize() puts the parent key on the child table:\n")
    windowed = view.with_columns(
        (pl.col(AMOUNT) / pl.col(AMOUNT).sum().over(STORE_ID)).alias("region.store.sale.share2")
    )
    print(span(windowed.collect("sale")).select(STORE_ID, AMOUNT, "region.store.sale.share2"))

    print("\nMulti-level rollup chain — sale -> store -> region:")
    chained = view.promote(
        "amount", from_level="sale", to_level="store", agg="sum", alias="revenue"
    ).promote("revenue", from_level="store", to_level="region", agg="sum", alias="revenue")
    print(
        chained.collect("region").select(REGION_ID, "region.name", "region.revenue").sort(REGION_ID)
    )


# =============================================================================
# Part 5: Filtering against other levels
# =============================================================================


def demonstrate_filtering(view: HierarchyView) -> None:
    header("PART 5: Filtering across levels")

    total = view.collect("sale").height

    print("\n(a) Leaf rows compared to a PARENT-level aggregate")
    print("    'sales above their own store's average':")
    above = view.promote(
        "amount", from_level="sale", to_level="store", agg="mean", alias="avg"
    ).filter(pl.col(AMOUNT) > pl.col("region.store.avg"))
    print(f"    {above.collect('sale').height} of {total} sales")

    print("\n(b) Leaf rows compared to a GRANDPARENT attribute:")
    lopsided = view.filter(pl.col(AMOUNT) * pl.col(TAX) > 5.0)
    print(f"    {lopsided.collect('sale').height} of {total} sales carry over 5.0 of tax")

    print("\n(c) Existence, skipping a level, with a CROSS-LEVEL predicate —")
    print("    'regions containing a sale that alone owes more than 15 in tax'.")
    print("    The predicate is evaluated at sale level but reads region.tax_rate:")
    hot = view.any_child_satisfies(
        pl.col(AMOUNT) * pl.col(TAX) > 15.0, at_level="region", child_level="sale"
    )
    print(f"    regions: {sorted(hot.tables()['region'].collect()[REGION_ID].to_list())}")
    print("    (a semi-join — no explode, no list construction)")

    print("\n(d) Filtering a PARENT attribute restricts the children automatically:")
    one_region = view.filter(pl.col("region.name") == "region-1")
    print(f"    sale rows visible: {one_region.tables()['sale'].collect().height} of {total}")
    print("    — tables() performs no join, yet returns no orphans.")


# =============================================================================
# Part 6: Conditional aggregation
# =============================================================================


def demonstrate_conditional_rollup(view: HierarchyView) -> None:
    header("PART 6: Conditional aggregation")

    print("\n'Revenue from bulk sales only (qty >= 3), per store' — mask at the leaf,")
    print("then promote, so non-matching children contribute zero rather than")
    print("disappearing from the hierarchy:\n")
    bulk = (
        view.with_columns(
            pl.when(pl.col(QTY) >= 3)
            .then(pl.col(AMOUNT))
            .otherwise(0.0)
            .alias("region.store.sale.bulk")
        )
        .promote("bulk", from_level="sale", to_level="store", agg="sum", alias="bulk_revenue")
        .promote("amount", from_level="sale", to_level="store", agg="sum", alias="revenue")
    )
    stores = bulk.collect("store").with_columns(
        (pl.col("region.store.bulk_revenue") / pl.col("region.store.revenue"))
        .round(3)
        .alias("bulk_pct")
    )
    print(
        stores.select(
            STORE_ID, "region.store.revenue", "region.store.bulk_revenue", "bulk_pct"
        ).sort(STORE_ID)
    )


# =============================================================================
# Part 7: A full pipeline, ending in the nested shape
# =============================================================================


def demonstrate_full_pipeline(view: HierarchyView) -> None:
    header("PART 7: A full pipeline — and back to List[Struct] at the boundary")

    pipeline = (
        # 1. cross-level price calculation
        view.with_columns(
            (pl.col(AMOUNT) * (1 - pl.col(DISCOUNT)) * (1 + pl.col(TAX))).alias(
                "region.store.sale.final"
            )
        )
        # 2. roll the result up to the store
        .promote("final", from_level="sale", to_level="store", agg="sum", alias="net_revenue")
        # 3. rank each sale within its store
        .with_columns(
            pl.col("region.store.sale.final")
            .rank(descending=True)
            .over(STORE_ID)
            .alias("region.store.sale.rank")
        )
        # 4. keep only each store's top two sales
        .filter(pl.col("region.store.sale.rank") <= 2)
        # 5. and only regions that still have a big one
        .any_child_satisfies(
            pl.col("region.store.sale.final") > 50, at_level="region", child_level="sale"
        )
    )

    print("\nNothing has executed yet — the whole thing is one deferred plan.")
    print("\nCheapest terminal (one level, no join, no nesting):")
    print(
        pipeline.tables()["sale"]
        .collect()
        .select(STORE_ID, AMOUNT, "region.store.sale.final", "region.store.sale.rank")
        .sort(STORE_ID, "region.store.sale.rank")
    )

    print("\nFlat, joined to leaf granularity:")
    flat_result = pipeline.collect("sale")
    print(f"  {flat_result.height} rows x {flat_result.width} cols")

    print("\nAnd the packed List[Struct] shape, built only because we asked:")
    nested = pipeline.collect_nested()
    print(f"  {nested.height} region row(s)")
    print(f"  {nested.schema['region']}")
    print("\nThat last step is the only place nesting is materialized. Everything")
    print("above it ran against flat Parquet tables with full predicate pushdown.")


# =============================================================================


def main() -> None:
    print("\n" + "=" * 80)
    print("  POLARS NEXPRESSO - HierarchyView Examples")
    print("=" * 80)

    packer = HierarchicalPacker(SPEC)
    flat = build_source_data()
    warehouse = Path(tempfile.mkdtemp(prefix="nexpresso-demo-"))

    try:
        demonstrate_the_problem(flat, packer)
        view = build_view(flat, packer, warehouse)
        demonstrate_cross_level(view)
        demonstrate_rollup_and_share(view)
        demonstrate_filtering(view)
        demonstrate_conditional_rollup(view)
        demonstrate_full_pipeline(view)

        print("\n" + "=" * 80)
        print("  ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 80 + "\n")
    finally:
        shutil.rmtree(warehouse, ignore_errors=True)


if __name__ == "__main__":
    main()

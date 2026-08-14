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
    print(f"  region.store: {nested.schema['region.store']}")

    print("\nGoal: net = sale.amount * (1 - store.discount)")
    print("      i.e. a LEAF value combined with its PARENT's attribute.\n")

    print("Attempt: reference the parent field from inside list.eval")
    try:
        nested.select(
            pl.col("region.store").list.eval(
                pl.element()
                .struct.field("sale")
                .list.eval(
                    pl.element().struct.field("amount")
                    * pl.col("region.store").struct.field("discount")
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
    print("statistics and sort order. Ask the view for a granularity and it hands")
    print("back an ordinary LazyFrame joined down to it:\n")
    for level in view.levels:
        frame = view.level(level)
        print(f"  view.level({level!r}):{'':<4} {len(frame.collect_schema().names())} columns")
    return view


# =============================================================================
# Part 3: Cross-level expressions
# =============================================================================


def demonstrate_cross_level(view: HierarchyView) -> None:
    header("PART 3: Cross-level expressions — parent, grandparent, all at once")

    sales = view.level("sale")
    print("\nview.level('sale') is a LazyFrame with every ancestor column in scope,")
    print("so the expression Part 1 could not write is just an expression.\n")

    print("(a) Leaf x PARENT attribute:")
    net = sales.with_columns((pl.col(AMOUNT) * (1 - pl.col(DISCOUNT))).alias("net"))
    print(span(net.collect()).select(STORE_ID, DISCOUNT, AMOUNT, "net"))

    print("\n(b) Leaf x GRANDPARENT attribute — two levels up, same syntax:")
    taxed = sales.with_columns((pl.col(AMOUNT) * pl.col(TAX)).alias("tax"))
    print(span(taxed.collect()).select(REGION_ID, TAX, AMOUNT, "tax"))

    print("\n(c) All three levels in ONE expression:")
    final = sales.with_columns(
        (pl.col(AMOUNT) * (1 - pl.col(DISCOUNT)) * (1 + pl.col(TAX))).alias("final")
    )
    print(span(final.collect()).select(TAX, DISCOUNT, AMOUNT, "final"))

    print("\nNo routing rules to learn: it is polars, on a frame that has one")
    print("granularity. The join to get there is pruned to the columns actually")
    print("used — asking for the whole axis costs nothing you do not read.")


# =============================================================================
# Part 4: Rolling up to a parent
# =============================================================================


def demonstrate_rollup_and_share(view: HierarchyView) -> None:
    header("PART 4: Rolling up to a parent, and reading the result back down")

    print("\nA roll-up is a group_by on the parent's key columns. view.key_columns()")
    print(f"spells them for you — for 'store' that is {view.key_columns('store')}:\n")
    revenue = (
        view.level("sale")
        .group_by(view.key_columns("store"))
        .agg(pl.col(AMOUNT).sum().alias("revenue"))
    )
    print(revenue.collect().sort(STORE_ID).head(4))

    print("\n'What share of its store's revenue is each sale?' — join it back:\n")
    share = (
        view.level("sale")
        .join(revenue, on=view.key_columns("store"), how="left")
        .with_columns((pl.col(AMOUNT) / pl.col("revenue")).alias("share"))
    )
    print(span(share.collect()).select(STORE_ID, AMOUNT, "revenue", "share"))

    print("\nOr skip the join entirely with a window over the parent key, which")
    print("normalize() already put on the child table:\n")
    windowed = view.level("sale").with_columns(
        (pl.col(AMOUNT) / pl.col(AMOUNT).sum().over(STORE_ID)).alias("share")
    )
    print(span(windowed.collect()).select(STORE_ID, AMOUNT, "share"))

    print("\nRolling straight to the region needs no intermediate hop — every")
    print("level's table carries all of its ancestor keys, not just its parent's:\n")
    print(
        view.level("sale")
        .group_by(view.key_columns("region"))
        .agg(pl.col(AMOUNT).sum().alias("revenue"), pl.len().alias("sales"))
        .collect()
        .sort(REGION_ID)
    )


# =============================================================================
# Part 5: Filtering
# =============================================================================


def demonstrate_filtering(view: HierarchyView) -> None:
    header("PART 5: Filtering — two different questions")

    total = view.level("sale").collect().height

    print("\n(a) Asking a question about sales: filter the frame.")
    print("    'sales above their own store average':")
    above = (
        view.level("sale")
        .filter(pl.col(AMOUNT) > pl.col(AMOUNT).mean().over(STORE_ID))
        .collect()
    )
    print(f"    {above.height} of {total} sales")

    print("\n    'sales carrying over 5.0 of tax' — leaf against grandparent:")
    lopsided = view.level("sale").filter(pl.col(AMOUNT) * pl.col(TAX) > 5.0).collect()
    print(f"    {lopsided.height} of {total} sales")

    print("\n(b) Existence — 'regions containing a sale that alone owes over 15")
    print("    in tax'. A semi-join: no explode, no list construction.")
    keys = view.key_columns("region")
    hot = view.level("region").join(
        view.level("sale").filter(pl.col(AMOUNT) * pl.col(TAX) > 15.0).select(keys).unique(),
        on=keys,
        how="semi",
    )
    print(f"    regions: {sorted(hot.collect()[REGION_ID].to_list())}")

    print("\n(c) Restricting the HIERARCHY, not a frame: view.filter().")
    print("    This is the one operation a flat frame cannot express, because")
    print("    dropping a region has to drop its stores and sales too:\n")
    one_region = view.filter(pl.col("region.name") == "region-1")
    for level in one_region.levels:
        kept = one_region.tables()[level].collect().height
        whole = view.tables()[level].collect().height
        print(f"      {level:<8} {kept:>4} of {whole:>4} rows")
    print("\n    tables() performs no join, yet returns no orphans.")


# =============================================================================
# Part 6: Conditional aggregation
# =============================================================================


def demonstrate_conditional_rollup(view: HierarchyView) -> None:
    header("PART 6: Conditional aggregation")

    print("\n'Revenue from bulk sales only (qty >= 3), per store' — an ordinary")
    print("masked aggregate, so non-matching sales contribute zero rather than")
    print("disappearing from the store:\n")
    stores = (
        view.level("sale")
        .group_by(view.key_columns("store"))
        .agg(
            pl.col(AMOUNT).sum().alias("revenue"),
            pl.when(pl.col(QTY) >= 3).then(pl.col(AMOUNT)).otherwise(0.0).sum().alias("bulk"),
        )
        .with_columns((pl.col("bulk") / pl.col("revenue")).round(3).alias("bulk_pct"))
        .collect()
        .sort(STORE_ID)
    )
    print(stores.select(STORE_ID, "revenue", "bulk", "bulk_pct"))


# =============================================================================
# Part 7: Ending in the nested shape
# =============================================================================


def demonstrate_full_pipeline(view: HierarchyView, packer: HierarchicalPacker) -> None:
    header("PART 7: A full pipeline — and back to List[Struct] at the boundary")

    print("\nRestrict the hierarchy first — that part stays on the view, so the")
    print("levels stay normalized and no parent column is ever repeated per sale:")
    restricted = view.filter(pl.col(AMOUNT) * pl.col(TAX) > 5.0)
    print(f"  {restricted!r}")

    print("\nNothing has executed yet. Now do the analysis on the flat frame.")
    print("Note the dotted aliases: a derived column keeps its level's path if")
    print("you intend to pack the result back, since pack() places columns by")
    print("path. Name it 'final' and pack() will look for it above the leaf.")
    final, rank = "region.store.sale.final", "region.store.sale.rank"
    ranked = (
        restricted.level("sale")
        .with_columns((pl.col(AMOUNT) * (1 - pl.col(DISCOUNT)) * (1 + pl.col(TAX))).alias(final))
        .with_columns(pl.col(final).rank(descending=True).over(STORE_ID).alias(rank))
        .filter(pl.col(rank) <= 2)
    )
    print(ranked.collect().select(STORE_ID, AMOUNT, final, rank).sort(STORE_ID, rank).head(6))

    print("\nThe packed List[Struct] shape, built only where something needs it.")
    print("Straight from the view when the derived columns are not wanted:")
    nested = restricted.nested().collect()
    print(f"  {nested.height} region row(s); region.store: {nested.schema['region.store']}")

    print("\n...or by packing the analysed frame when they are:")
    packed = packer.pack(ranked.collect(), "region")
    print(f"  {packed.height} region row(s); region.store: {packed.schema['region.store']}")

    print("\nEverything above that ran against flat Parquet tables with full")
    print("projection and predicate pushdown. Nesting is a boundary format.")


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
        demonstrate_full_pipeline(view, packer)

        print("\n" + "=" * 80)
        print("  ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 80 + "\n")
    finally:
        shutil.rmtree(warehouse, ignore_errors=True)


if __name__ == "__main__":
    main()

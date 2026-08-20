#!/usr/bin/env python3
"""
Polars Nexpresso - HierarchyView recipes
========================================

A cookbook for `HierarchyView`, organised around the three things you need to
know to use it.

    1. THE THREE CONTEXTS   which entry point returns what, and when to use it
    2. EXPRESSIONS          what you may write (answer: anything Polars accepts)
    3. NAMING               how columns are addressed, and the one rule that bites

`examples_hierarchy_view.py` is the narrative tour of *why* the view exists.
This file is the reference you come back to.

To run: python examples_hierarchy_view_recipes.py
"""

import shutil
import tempfile
from pathlib import Path

import polars as pl

from nexpresso import HierarchicalPacker, HierarchySpec, HierarchyView, LevelSpec

pl.Config.set_tbl_rows(8)
pl.Config.set_tbl_width_chars(200)

# Column paths, spelled once.
REGION_ID = "region.id"
REGION_NAME = "region.name"
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
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def sub(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 72 - len(title)))


def build_flat() -> pl.DataFrame:
    """3 regions x 2 stores x 3 sales, with one null amount."""
    rows = []
    for r in range(3):
        for s in range(2):
            for k in range(3):
                store = r * 2 + s
                sale = store * 3 + k
                rows.append(
                    (
                        r,
                        f"region-{r}",
                        0.05 * (r + 1),
                        store,
                        0.1 * s,
                        sale,
                        None if sale == 4 else float(10 * (sale + 1)),
                        1 + sale % 5,
                    )
                )
    return pl.DataFrame(
        {
            REGION_ID: [r[0] for r in rows],
            REGION_NAME: [r[1] for r in rows],
            TAX: [r[2] for r in rows],
            STORE_ID: [r[3] for r in rows],
            DISCOUNT: [r[4] for r in rows],
            SALE_ID: [r[5] for r in rows],
            AMOUNT: [r[6] for r in rows],
            QTY: [r[7] for r in rows],
        }
    )


# =============================================================================
# 1. The three contexts
# =============================================================================


def recipe_contexts(view: HierarchyView, packer: HierarchicalPacker) -> None:
    header("1. THE THREE CONTEXTS")

    print(
        """
Every entry point answers a different question, and the return type tells you
which one you are in:

  view.tables()[g]  -> LazyFrame   one level's own table, exactly as stored.
                                   Ancestor KEYS are on it; attributes are
                                   not. No join.
  view.level(g)     -> LazyFrame   the root->g axis joined. Every ancestor
                                   column in scope. This is the query context.
  view.with_level() -> View        a modification that lands back on a level,
  view.filter()     -> View        so the hierarchy survives and can still be
                                   filtered, nested or sunk.

Ancestor attributes are in scope in BOTH level() and with_level(); only the
raw tables()[g] frame lacks them. So the question is not what you can write,
it is what you want back:

  a frame to query          -> level(g)
  a hierarchy to keep going -> with_level(g, ...) / filter(...)
  the stored table itself   -> tables()[g]
"""
    )

    sub("tables()[g] — no join at all")
    print(f"  columns: {view.tables()['sale'].collect_schema().names()}")
    print("  note region.id is present (a key) but region.tax_rate is not (an attribute)")

    sub("level(g) — the axis joined, one row per g entity")
    print(f"  columns: {view.level('sale').collect_schema().names()}")
    print(f"  rows: {view.level('sale').collect().height} sales, ", end="")
    print(f"{view.level('store').collect().height} stores, ", end="")
    print(f"{view.level('region').collect().height} regions")

    sub("with_level() — modify a level, keep a view")
    widened = view.with_level(
        "sale",
        lambda lf: lf.with_columns((pl.col(AMOUNT) * 2).alias("region.store.sale.doubled")),
    )
    print(f"  {widened!r}")
    print("  still a view, so everything else still applies:")
    kept = widened.filter(pl.col("region.store.sale.doubled") > 200)
    print(f"    .filter(...)  -> {kept.level('sale').collect().height} sales")
    inner = kept.nested().collect().schema["region.store"].inner.to_schema()
    print(f"    .nested()     -> sale fields {list(inner['sale'].inner.to_schema())}")

    sub("Why not just use level() and keep going?")
    print(
        """
  level() hands you a frame and lets go — there is no way back to a hierarchy
  from it, because a flat frame has lost which columns belonged to which level.
  Use it for questions. Use with_level when the answer has to stay a hierarchy.
"""
    )

    sub("The same thing the long way, for when you need several levels at once")
    tables = dict(view.tables())
    tables["sale"] = tables["sale"].with_columns(
        (pl.col(AMOUNT) * 2).alias("region.store.sale.doubled")
    )
    tables["store"] = tables["store"].with_columns(
        (pl.col(DISCOUNT) * 100).alias("region.store.discount_pct")
    )
    rebuilt = HierarchyView.from_tables(tables, packer)
    print(f"  {rebuilt!r}")
    print("  (from_tables resets empty_parents to 'prune' — with_level preserves it)")


# =============================================================================
# 2. Expressions
# =============================================================================


def recipe_expressions(view: HierarchyView) -> None:
    header("2. EXPRESSIONS — anything Polars accepts")

    print(
        """
There is no expression sublanguage to learn. level(g) returns a LazyFrame, so
every Polars expression works, including the ones the old view had no route
for: windows, when/then, struct and list operations, joins, group_by.
"""
    )

    sub("(a) Cross-level arithmetic — leaf x parent x grandparent")
    print(
        view.level("sale")
        .with_columns((pl.col(AMOUNT) * (1 - pl.col(DISCOUNT)) * (1 + pl.col(TAX))).alias("final"))
        .select(REGION_ID, TAX, DISCOUNT, AMOUNT, "final")
        .head(4)
        .collect()
    )

    sub("(b) Window functions over an ancestor key — no join needed")
    print(
        view.level("sale")
        .with_columns(
            (pl.col(AMOUNT) / pl.col(AMOUNT).sum().over(STORE_ID)).round(3).alias("store_share"),
            pl.col(AMOUNT).rank(descending=True).over(REGION_ID).alias("rank_in_region"),
        )
        .select(STORE_ID, AMOUNT, "store_share", "rank_in_region")
        .head(4)
        .collect()
    )

    sub("(c) Conditional aggregation — when/then inside agg")
    print(
        view.level("sale")
        .group_by(view.key_columns("store"))
        .agg(
            pl.col(AMOUNT).sum().alias("revenue"),
            pl.when(pl.col(QTY) >= 3).then(pl.col(AMOUNT)).otherwise(0.0).sum().alias("bulk"),
            pl.col(AMOUNT).null_count().alias("missing"),
        )
        .sort(STORE_ID)
        .head(4)
        .collect()
    )

    sub("(d) Aggregating to any ancestor, skipping levels")
    print("  sale -> region directly; every level carries ALL its ancestor keys:")
    print(
        view.tables()["sale"]
        .group_by(view.key_columns("region"))
        .agg(pl.col(AMOUNT).sum().alias("revenue"), pl.len().alias("sales"))
        .sort(REGION_ID)
        .collect()
    )

    sub("(e) Joining a roll-up back down")
    keys = view.key_columns("store")
    revenue = view.tables()["sale"].group_by(keys).agg(pl.col(AMOUNT).sum().alias("revenue"))
    print(
        view.level("sale")
        .join(revenue, on=keys, how="left")
        .with_columns((pl.col(AMOUNT) / pl.col("revenue")).round(3).alias("share"))
        .select(STORE_ID, AMOUNT, "revenue", "share")
        .head(4)
        .collect()
    )

    sub("(f) Existence — a semi-join, no explode, no list building")
    rkeys = view.key_columns("region")
    matching = view.tables()["sale"].filter(pl.col(AMOUNT) > 150).select(rkeys).unique()
    print("  regions containing a sale over 150:")
    print(
        view.level("region")
        .join(matching, on=rkeys, how="semi")
        .select(REGION_ID, REGION_NAME)
        .collect()
    )
    print("  NOTE this is not view.filter(amount > 150): that would also drop the")
    print("       non-matching sales, and any region left with none.")

    sub("(g) filter() on the view — restricting the whole hierarchy")
    restricted = view.filter(pl.col(AMOUNT) > 150)
    for level in restricted.levels:
        kept = restricted.tables()[level].collect().height
        whole = view.tables()[level].collect().height
        print(f"    {level:<8} {kept:>3} of {whole:>3} rows")

    sub("(h) An aggregate over an ancestor key counts THAT level's entities")
    n = view.tables()["region"].collect().height
    print(f"  there are {n} regions; a predicate over region.id is evaluated there:")
    print(f"    filter(region.id.count() == {n})  -> ", end="")
    print(f"{view.filter(pl.col(REGION_ID).count() == n).level('region').collect().height} regions")
    print(f"    filter(region.id.count() >  {n})  -> ", end="")
    print(f"{view.filter(pl.col(REGION_ID).count() > n).level('region').collect().height} regions")


# =============================================================================
# 3. Naming
# =============================================================================


def recipe_naming(view: HierarchyView, packer: HierarchicalPacker) -> None:
    header("3. NAMING — explicit dotted paths")

    print(
        """
Every column is addressed by its full path from the root, joined by the
packer's separator. The separator is configurable (`granularity_separator`,
default "."), and a field name containing it is escaped with `escape_char`
(default "\\"). The view never guesses: a name either resolves to a level or
it is an error.
"""
    )

    sub("Where a column lives is read off its path")
    for column in (REGION_NAME, DISCOUNT, AMOUNT):
        print(f"  {column:<28} owned by {view.level_of(column)!r}")

    sub("key_columns() spells the identifying columns for a level")
    for level in view.levels:
        print(f"  {level:<8} {view.key_columns(level)}")

    sub("Build paths with the packer rather than f-strings")
    print("  packer.join_path(['region', 'store', 'sale', 'amount'])")
    print(f"    -> {packer.join_path(['region', 'store', 'sale', 'amount'])!r}")
    print("  a field whose NAME contains the separator must be escaped:")
    print(f"    packer.escape_field('net.sales') -> {packer.escape_field('net.sales')!r}")
    print("    packer.split_path('region.store.net\\\\.sales')")
    print(f"      -> {packer.split_path(chr(92).join(['region.store.net', '.sales']))}")

    sub("The separator is configurable — nothing assumes '.'")
    for sep in (".", "__", " -> "):
        other = HierarchicalPacker(SPEC, granularity_separator=sep)
        amount = other.join_path(["region", "store", "sale", "amount"])
        # A full round trip on that separator: build a frame, view it, resolve
        # ownership, roll up. Ownership is resolved by SPLITTING the path, so it
        # is the strictest check that the separator is honoured throughout.
        frame = pl.DataFrame(
            {
                other.join_path(["region", "id"]): [1, 1],
                other.join_path(["region", "store", "region_id"]): [1, 1],
                other.join_path(["region", "store", "id"]): [10, 11],
                other.join_path(["region", "store", "sale", "store_id"]): [10, 11],
                other.join_path(["region", "store", "sale", "id"]): [100, 110],
                amount: [5.0, 6.0],
            }
        )
        v = HierarchyView.from_frame(frame, other)
        total = (
            v.tables()["sale"]
            .group_by(v.key_columns("region"))
            .agg(pl.col(amount).sum().alias("t"))
            .collect()["t"]
            .item()
        )
        print(f"  sep={sep!r:8} level_of({amount!r}) = {v.level_of(amount)!r}, rollup = {total}")

    sub("THE RULE THAT BITES: naming decides whether a column survives nesting")
    print(
        """
  On level(g) you may alias anything — it is just a LazyFrame, and "net" is a
  perfectly good column name. But nested() and pack() place columns by PATH.
  A column not named for its level has nowhere to go and is dropped.

  with_level() checks this for you, so the mistake is an error, not silence:
"""
    )
    try:
        view.with_level("sale", lambda lf: lf.with_columns(pl.col(AMOUNT).alias("net")))
    except ValueError as exc:
        print(f"    ValueError: {str(exc)[:150]}...")

    print("\n  Named for its level, it survives all the way into the nested shape:")
    ok = view.with_level(
        "sale", lambda lf: lf.with_columns(pl.col(AMOUNT).alias("region.store.sale.net"))
    )
    store = ok.nested().collect().schema["region.store"].inner.to_schema()
    print(f"    nested sale fields: {list(store['sale'].inner.to_schema())}")

    sub("Same rule when packing a frame you built with level()")
    print("  pack() has no with_level() to warn you, so compare the two directly.\n")
    derive = pl.col(AMOUNT) * (1 - pl.col(DISCOUNT))
    for alias in ("net", "region.store.sale.net"):
        frame = view.level("sale").with_columns(derive.alias(alias)).collect()
        try:
            inner = packer.pack(frame, "region").schema["region.store"].inner.to_schema()
            fields = list(inner["sale"].inner.to_schema())
            verdict = "kept" if "net" in fields else "LOST"
            print(f"    alias={alias!r:26} -> sale fields {fields}  [{verdict}]")
        except Exception as exc:
            print(
                f"    alias={alias!r:26} -> {type(exc).__name__}: {str(exc).splitlines()[0][:70]}"
            )
    print("\n  An unqualified name is read as belonging to a coarser level, so pack()")
    print("  either drops it or refuses the frame. Qualified, it lands on the leaf.")


# =============================================================================
# Quick reference
# =============================================================================


def recipe_cheatsheet() -> None:
    header("QUICK REFERENCE")
    print(
        """
  WANT                                  WRITE
  ------------------------------------  --------------------------------------
  a frame at some granularity           view.level("sale")
  one level's table, no join            view.tables()["sale"]
  the packed List[Struct] shape         view.nested().collect()
  the identifying columns of a level    view.key_columns("store")
  which level owns a column             view.level_of("region.store.discount")

  a derived column, keep the view       view.with_level("sale", lambda lf: ...)
  ...using an ancestor attribute        same — they are in scope, and are
                                        joined in only if you name one
  a derived column, just querying       view.level("sale").with_columns(...)
  restrict the whole hierarchy          view.filter(pl.col(AMOUNT) > 990)
  restrict just this query              view.level("sale").filter(...)

  roll a child up to an ancestor        view.tables()["sale"]
                                            .group_by(view.key_columns("region"))
                                            .agg(...)
  ...using an ancestor attribute too    view.level("sale").group_by(...).agg(...)
  parents with a matching child         view.level("region").join(
                                            matching_keys, on=keys, how="semi")

  NAMING
  ------------------------------------  --------------------------------------
  querying only                         any alias you like
  result must nest or pack              full dotted path, e.g.
                                        "region.store.sale.net"
  field name contains the separator     packer.escape_field("net.sales")
  building a path                       packer.join_path([...])
"""
    )


def main() -> None:
    print("\n" + "=" * 78)
    print("  POLARS NEXPRESSO - HierarchyView recipes")
    print("=" * 78)

    packer = HierarchicalPacker(SPEC)
    warehouse = Path(tempfile.mkdtemp(prefix="nexpresso-recipes-"))
    try:
        HierarchyView.from_frame(build_flat(), packer).sink_parquet(warehouse)
        view = HierarchyView.scan_parquet(warehouse, packer)

        recipe_contexts(view, packer)
        recipe_expressions(view)
        recipe_naming(view, packer)
        recipe_cheatsheet()

        print("\n" + "=" * 78)
        print("  ALL RECIPES COMPLETED SUCCESSFULLY!")
        print("=" * 78 + "\n")
    finally:
        shutil.rmtree(warehouse, ignore_errors=True)


if __name__ == "__main__":
    main()

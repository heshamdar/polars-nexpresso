#!/usr/bin/env python3
"""
Storage-layout benchmark: nested vs. flat vs. normalized ``HierarchyView``.

Nesting is a good in-memory shape and a poor storage shape. This benchmark
quantifies that by running the same question against the same data in three
layouts:

``nested``
    One Parquet file holding the packed ``List[Struct]`` frame. Parquet shreds
    it into one leaf column chunk per field, but a row group holds N *top-level*
    rows — so packing collapses the number of row groups and takes row-group
    skipping off the table.

``flat``
    One denormalized Parquet file, sorted by hierarchy key. Parent columns
    repeat per leaf row, but RLE + dictionary encoding makes that nearly free,
    and every predicate gets full pushdown.

``view``
    One Parquet file per level plus :class:`~nexpresso.HierarchyView`. Each
    level is a real top-level table with its own row groups and statistics; the
    view presents them as if they were nested.

Every query is checked across layouts before it is timed, so a divergence shows
up as a failure rather than a suspiciously fast number.

Usage::

    python -m benchmarks.bench_storage
    python -m benchmarks.bench_storage --scale large --repeat 7
    python -m benchmarks.bench_storage --json results.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl

from nexpresso import HierarchicalPacker, HierarchySpec, HierarchyView, LevelSpec

# =============================================================================
# Fixture
# =============================================================================

SCALES: dict[str, tuple[int, int, int]] = {
    # name: (regions, stores per region, sales per store)
    "small": (10, 20, 100),  # 20k leaf rows
    "medium": (20, 50, 300),  # 300k leaf rows
    "large": (50, 100, 400),  # 2M leaf rows
}

REGION_ID = "region.id"
STORE_ID = "region.store.id"
SALE_ID = "region.store.sale.id"
AMOUNT = "region.store.sale.amount"
REGION_NAME = "region.name"

SPEC = HierarchySpec.from_levels(
    LevelSpec(name="region", id_fields=["id"]),
    LevelSpec(name="store", id_fields=["id"], parent_keys=["region_id"]),
    LevelSpec(name="sale", id_fields=["id"], parent_keys=["store_id"]),
)


def build_flat(n_region: int, n_store: int, n_sale: int) -> pl.DataFrame:
    """
    A three-level sales hierarchy, sorted by hierarchy key.

    ``note`` and ``sku`` exist to make the leaf level genuinely wide — without a
    heavy column that queries do *not* touch, projection pruning has nothing to
    prune and the comparison flatters nesting.
    """
    rows = n_region * n_store * n_sale
    per_region = n_store * n_sale
    return pl.DataFrame(
        {
            REGION_ID: [i // per_region for i in range(rows)],
            REGION_NAME: [f"region-{i // per_region}" for i in range(rows)],
            STORE_ID: [i // n_sale for i in range(rows)],
            "region.store.name": [f"store-{i // n_sale}" for i in range(rows)],
            SALE_ID: list(range(rows)),
            AMOUNT: [float(i % 997) for i in range(rows)],
            "region.store.sale.qty": [i % 13 for i in range(rows)],
            "region.store.sale.sku": [f"sku-{i % 5000}" for i in range(rows)],
            "region.store.sale.note": [
                f"a fairly chunky descriptive note for sale {i}" for i in range(rows)
            ],
        }
    ).sort(REGION_ID, STORE_ID, SALE_ID)


@dataclass
class Layouts:
    """On-disk paths and sizes for the three layouts."""

    root: Path
    nested: Path
    flat: Path
    view_dir: Path
    sizes: dict[str, int]


def write_layouts(
    flat: pl.DataFrame, packer: HierarchicalPacker, root: Path, row_group_size: int
) -> Layouts:
    """Materialize all three layouts from one flat frame."""
    root.mkdir(parents=True, exist_ok=True)
    nested_path, flat_path = root / "nested.parquet", root / "flat.parquet"
    view_dir = root / "levels"

    packer.pack(flat, "region").write_parquet(nested_path, statistics=True)
    flat.write_parquet(flat_path, row_group_size=row_group_size, statistics=True)
    HierarchyView.from_frame(flat, packer).sink_parquet(
        view_dir, row_group_size=row_group_size, statistics=True
    )

    def total(path: Path) -> int:
        if path.is_dir():
            return sum(p.stat().st_size for p in path.rglob("*.parquet"))
        return path.stat().st_size

    return Layouts(
        root=root,
        nested=nested_path,
        flat=flat_path,
        view_dir=view_dir,
        sizes={
            "nested": total(nested_path),
            "flat": total(flat_path),
            "view": total(view_dir),
        },
    )


# =============================================================================
# Query suite
# =============================================================================
# Each query is (name, description, {layout: callable}). Callables return a
# comparable value so layouts can be cross-checked before timing.

QueryFn = Callable[[Layouts, HierarchicalPacker], Any]

_AMOUNT_IN_LISTS = pl.col("region.store").list.eval(
    pl.element().struct.field("sale").list.eval(pl.element().struct.field("amount"))
)


def _nested_leaf(layouts: Layouts, packer: HierarchicalPacker) -> pl.LazyFrame:
    """Nested file unpacked back to leaf granularity."""
    return packer.unpack(pl.scan_parquet(layouts.nested), "sale")


def _view(layouts: Layouts, packer: HierarchicalPacker) -> HierarchyView:
    return HierarchyView.scan_parquet(layouts.view_dir, packer)


def _rollup(view: HierarchyView, child: str, parent: str) -> pl.LazyFrame:
    """Total ``AMOUNT`` per ``parent``, grouped on the child's own table."""
    return (
        view.tables()[child]
        .group_by(view.key_columns(parent))
        .agg(pl.col(AMOUNT).sum().alias("total"))
    )


QUERIES: dict[str, dict[str, Any]] = {
    "root_key_filter": {
        "description": "Sum of sale amounts for a single region (selective root-key predicate)",
        "nested": lambda lay, p: _nested_leaf(lay, p)
        .filter(pl.col(REGION_ID) == 3)
        .select(pl.col(AMOUNT).sum())
        .collect()
        .item(),
        "flat": lambda lay, p: pl.scan_parquet(lay.flat)
        .filter(pl.col(REGION_ID) == 3)
        .select(pl.col(AMOUNT).sum())
        .collect()
        .item(),
        "view": lambda lay, p: _view(lay, p)
        .filter(pl.col(REGION_ID) == 3)
        .tables()["sale"]
        .select(pl.col(AMOUNT).sum())
        .collect()
        .item(),
    },
    "leaf_projection": {
        "description": "Read + total one leaf column out of a wide leaf level",
        # The nested path cannot ask for one leaf: Polars projects whole
        # top-level columns, so all nine leaves are read to reach `amount`.
        "nested": lambda lay, p: pl.scan_parquet(lay.nested)
        .select(_AMOUNT_IN_LISTS.alias("a"))
        .explode("a")
        .explode("a")
        .select(pl.col("a").sum())
        .collect()
        .item(),
        "flat": lambda lay, p: pl.scan_parquet(lay.flat)
        .select(pl.col(AMOUNT).sum())
        .collect()
        .item(),
        "view": lambda lay, p: _view(lay, p)
        .tables()["sale"]
        .select(pl.col(AMOUNT).sum())
        .collect()
        .item(),
    },
    "leaf_filter": {
        "description": "Selective filter on a leaf attribute, counted",
        "nested": lambda lay, p: _nested_leaf(lay, p)
        .filter(pl.col(AMOUNT) > 990)
        .select(pl.len())
        .collect()
        .item(),
        "flat": lambda lay, p: pl.scan_parquet(lay.flat)
        .filter(pl.col(AMOUNT) > 990)
        .select(pl.len())
        .collect()
        .item(),
        "view": lambda lay, p: _view(lay, p)
        .filter(pl.col(AMOUNT) > 990)
        .tables()["sale"]
        .select(pl.len())
        .collect()
        .item(),
    },
    "ancestor_attribute_filter": {
        "description": "Filter on a parent *attribute* — the case that needs a join",
        "nested": lambda lay, p: _nested_leaf(lay, p)
        .filter(pl.col(REGION_NAME) == "region-3")
        .select(pl.col(AMOUNT).sum())
        .collect()
        .item(),
        "flat": lambda lay, p: pl.scan_parquet(lay.flat)
        .filter(pl.col(REGION_NAME) == "region-3")
        .select(pl.col(AMOUNT).sum())
        .collect()
        .item(),
        # The parent attribute lives only on the region table, so this is the
        # case normalization actually charges for. The view answers it with a
        # key semi-join rather than a full three-way join, because the
        # downward cascade has already restricted the sale table.
        "view": lambda lay, p: _view(lay, p)
        .filter(pl.col(REGION_NAME) == "region-3")
        .tables()["sale"]
        .select(pl.col(AMOUNT).sum())
        .collect()
        .item(),
    },
    "rollup_to_parent": {
        "description": "Total sale amount per store (child -> parent aggregation)",
        "nested": lambda lay, p: p.promote_attribute(
            _nested_leaf(lay, p),
            "amount",
            from_level="sale",
            to_level="store",
            agg="sum",
            alias="total",
        )
        .select(pl.col("region.store.total").sum())
        .collect()
        .item(),
        "flat": lambda lay, p: pl.scan_parquet(lay.flat)
        .group_by(STORE_ID)
        .agg(pl.col(AMOUNT).sum().alias("total"))
        .select(pl.col("total").sum())
        .collect()
        .item(),
        # The roll-up needs no ancestor *attribute*, only the parent key -- which
        # normalize() already put on the child table. Grouping tables()["sale"]
        # therefore skips the axis join that level("sale") would perform; going
        # through level() here costs 2.3x for nothing.
        "view": lambda lay, p: _rollup(_view(lay, p), "sale", "store")
        .select(pl.col("total").sum())
        .collect()
        .item(),
    },
    "existence": {
        "description": "How many stores have at least one sale over a threshold",
        "nested": lambda lay, p: _nested_leaf(lay, p)
        .filter(pl.col(AMOUNT) > 990)
        .select(pl.col(STORE_ID).n_unique())
        .collect()
        .item(),
        "flat": lambda lay, p: pl.scan_parquet(lay.flat)
        .filter(pl.col(AMOUNT) > 990)
        .select(pl.col(STORE_ID).n_unique())
        .collect()
        .item(),
        "view": lambda lay, p: _view(lay, p)
        .level("sale")
        .filter(pl.col(AMOUNT) > 990)
        .select(pl.col(STORE_ID).n_unique())
        .collect()
        .item(),
    },
    "cross_level_predicate": {
        "description": "Predicate comparing a leaf column against an ancestor column",
        "nested": lambda lay, p: _nested_leaf(lay, p)
        .filter(pl.col(AMOUNT) > pl.col(REGION_ID) * 20)
        .select(pl.len())
        .collect()
        .item(),
        "flat": lambda lay, p: pl.scan_parquet(lay.flat)
        .filter(pl.col(AMOUNT) > pl.col(REGION_ID) * 20)
        .select(pl.len())
        .collect()
        .item(),
        "view": lambda lay, p: _view(lay, p)
        .filter(pl.col(AMOUNT) > pl.col(REGION_ID) * 20)
        .tables()["sale"]
        .select(pl.len())
        .collect()
        .item(),
    },
    "materialize_nested": {
        "description": "Produce the packed List[Struct] shape (the boundary case)",
        "nested": lambda lay, p: pl.scan_parquet(lay.nested).collect().height,
        "flat": lambda lay, p: p.pack(pl.scan_parquet(lay.flat), "region").collect().height,
        "view": lambda lay, p: _view(lay, p).nested().collect().height,
    },
    "filtered_nested": {
        "description": "Filter on a leaf, then hand back the packed shape",
        "nested": lambda lay, p: p.pack(_nested_leaf(lay, p).filter(pl.col(AMOUNT) > 990), "region")
        .collect()
        .height,
        "flat": lambda lay, p: p.pack(
            pl.scan_parquet(lay.flat).filter(pl.col(AMOUNT) > 990), "region"
        )
        .collect()
        .height,
        "view": lambda lay, p: _view(lay, p).filter(pl.col(AMOUNT) > 990).nested().collect().height,
    },
}

LAYOUT_ORDER = ("nested", "flat", "view")


# =============================================================================
# Harness
# =============================================================================


@dataclass
class QueryResult:
    query: str
    description: str
    timings_ms: dict[str, float]
    value: Any
    agreed: bool
    disagreements: dict[str, Any]


def time_call(fn: Callable[[], Any], repeat: int) -> tuple[float, Any]:
    """Best-of-``repeat`` wall time in ms, plus the returned value."""
    best, value = float("inf"), None
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        if elapsed < best:
            best, value = elapsed, result
    return best * 1000, value


def run_query(
    name: str, spec: dict[str, Any], layouts: Layouts, packer: HierarchicalPacker, repeat: int
) -> QueryResult:
    """Run one query across all layouts, cross-checking results."""
    timings: dict[str, float] = {}
    values: dict[str, Any] = {}
    for layout in LAYOUT_ORDER:
        fn = spec[layout]
        timings[layout], values[layout] = time_call(lambda: fn(layouts, packer), repeat)

    reference = values["flat"]
    disagreements = {
        layout: value for layout, value in values.items() if not _equivalent(value, reference)
    }
    return QueryResult(
        query=name,
        description=spec["description"],
        timings_ms=timings,
        value=reference,
        agreed=not disagreements,
        disagreements=disagreements,
    )


def _equivalent(a: Any, b: Any) -> bool:
    """Numeric-tolerant equality for cross-layout checking."""
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
        except (TypeError, ValueError):
            return False
    return bool(a == b)


def format_report(results: list[QueryResult], layouts: Layouts, rows: int) -> str:
    """Human-readable comparison table."""
    lines: list[str] = []
    lines.append(f"\nleaf rows: {rows:,}")
    lines.append("\non-disk size")
    baseline = layouts.sizes["nested"]
    for name in LAYOUT_ORDER:
        size = layouts.sizes[name]
        delta = f"{(size / baseline - 1) * 100:+6.1f}% vs nested" if baseline else ""
        lines.append(f"  {name:<8} {size / 1e6:8.2f} MB   {delta}")

    width = max(len(r.query) for r in results) + 2
    lines.append("\nquery timings (best of N, ms)")
    header = f"  {'query':<{width}}" + "".join(f"{n:>12}" for n in LAYOUT_ORDER)
    lines.append(header + f"{'speedup':>12}")
    lines.append("  " + "-" * (width + 12 * 4))
    for result in results:
        row = f"  {result.query:<{width}}"
        for layout in LAYOUT_ORDER:
            row += f"{result.timings_ms[layout]:>12.1f}"
        best_alt = min(result.timings_ms["flat"], result.timings_ms["view"])
        speedup = result.timings_ms["nested"] / best_alt if best_alt else float("nan")
        row += f"{speedup:>11.1f}x"
        if not result.agreed:
            row += "  <-- MISMATCH"
        lines.append(row)

    mismatched = [r for r in results if not r.agreed]
    if mismatched:
        lines.append("\nMISMATCHES (layouts disagreed — treat timings as invalid):")
        for result in mismatched:
            lines.append(
                f"  {result.query}: expected {result.value!r}, got {result.disagreements!r}"
            )
    else:
        lines.append("\nall layouts agreed on every query.")

    lines.append("\ndescriptions")
    for result in results:
        lines.append(f"  {result.query:<{width}} {result.description}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--scale", choices=sorted(SCALES), default="medium", help="Fixture size preset."
    )
    parser.add_argument("--repeat", type=int, default=5, help="Timing repeats per query.")
    parser.add_argument(
        "--row-group-size", type=int, default=50_000, help="Parquet row group size."
    )
    parser.add_argument(
        "--queries", help="Comma-separated subset of queries to run (default: all)."
    )
    parser.add_argument("--json", type=Path, help="Write results as JSON to this path.")
    parser.add_argument(
        "--keep", type=Path, help="Write the Parquet fixtures here instead of a temp dir."
    )
    args = parser.parse_args(argv)

    selected = list(QUERIES)
    if args.queries:
        selected = [q.strip() for q in args.queries.split(",") if q.strip()]
        unknown = [q for q in selected if q not in QUERIES]
        if unknown:
            parser.error(f"Unknown queries: {unknown}. Available: {sorted(QUERIES)}")

    n_region, n_store, n_sale = SCALES[args.scale]
    packer = HierarchicalPacker(SPEC)

    root = args.keep if args.keep else Path(tempfile.mkdtemp(prefix="nexpresso-storage-"))
    try:
        print(f"building fixtures ({args.scale}) ...", file=sys.stderr)
        flat = build_flat(n_region, n_store, n_sale)
        layouts = write_layouts(flat, packer, root, args.row_group_size)

        results = [
            run_query(name, QUERIES[name], layouts, packer, args.repeat) for name in selected
        ]
        print(format_report(results, layouts, flat.height))

        if args.json:
            args.json.write_text(
                json.dumps(
                    {
                        "scale": args.scale,
                        "rows": flat.height,
                        "repeat": args.repeat,
                        "polars_version": pl.__version__,
                        "sizes_bytes": layouts.sizes,
                        "results": [asdict(r) for r in results],
                    },
                    indent=2,
                    default=str,
                )
            )
            print(f"\nwrote {args.json}", file=sys.stderr)

        return 0 if all(r.agreed for r in results) else 1
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

"""Experimental pack strategies for benchmarking.

These are **not** part of the public ``nexpresso`` API. Each function reproduces
the *contents* of :meth:`HierarchicalPacker.pack` (compared order-independently)
while exploring a different execution strategy, so the harness can measure their
time/memory trade-offs. Strategies that win clearly can later be promoted into the
library.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import polars as pl
from polars.io.partition import PartitionBy

from nexpresso import HierarchicalPacker

FrameOrLazy = pl.DataFrame | pl.LazyFrame


def _to_lazy(source: FrameOrLazy) -> pl.LazyFrame:
    """Return a LazyFrame view of ``source``."""
    return source.lazy() if isinstance(source, pl.DataFrame) else source


def _root_keys(packer: HierarchicalPacker) -> list[str]:
    """Qualified id columns of the root (coarsest) level."""
    return list(packer._levels_meta[0].id_columns)


def _bucket_expr(root_keys: list[str], partitions: int) -> pl.Expr:
    """Hash root keys into ``partitions`` buckets, keeping each entity whole."""
    return pl.struct(root_keys).hash() % partitions


def pack_filter_rescan(
    packer: HierarchicalPacker,
    source: FrameOrLazy,
    to_level: str,
    *,
    partitions: int,
    tmp_dir: str | Path,
) -> pl.LazyFrame:
    """
    Baseline partitioned pack: filter the source to each bucket and pack it.

    This mirrors the current ``pack_streaming`` inner loop. It rescans the source
    ``partitions`` times (one filter per bucket).
    """
    out = Path(tmp_dir)
    out.mkdir(parents=True, exist_ok=True)
    lazy = _to_lazy(source)
    bucket = _bucket_expr(_root_keys(packer), partitions)

    for i in range(partitions):
        part = packer.pack(lazy.filter(bucket == i), to_level)
        part.sink_parquet(out / f"part_{i:05d}.parquet")

    return pl.scan_parquet(str(out / "part_*.parquet"))


def pack_single_pass(
    packer: HierarchicalPacker,
    source: FrameOrLazy,
    to_level: str,
    *,
    partitions: int,
    tmp_dir: str | Path,
) -> pl.LazyFrame:
    """
    Single-pass partitioned pack: split the source into bucket files in one
    streaming pass via ``sink_parquet(PartitionBy(...))``, then pack each bucket.

    Trades the K source rescans of :func:`pack_filter_rescan` for one source scan
    plus an extra disk round-trip (write + read of the bucketed source).
    """
    out = Path(tmp_dir)
    buckets_dir = out / "buckets"
    parts_dir = out / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    lazy = _to_lazy(source)
    bucket = _bucket_expr(_root_keys(packer), partitions).alias("__bucket")
    lazy.with_columns(bucket).sink_parquet(
        PartitionBy(buckets_dir, key="__bucket", include_key=False), mkdir=True
    )

    for bucket_path in sorted(buckets_dir.glob("__bucket=*")):
        suffix = bucket_path.name.split("=", 1)[1]
        bucket_lf = pl.scan_parquet(str(bucket_path / "*.parquet"))
        packer.pack(bucket_lf, to_level).sink_parquet(parts_dir / f"part_{suffix}.parquet")

    return pl.scan_parquet(str(parts_dir / "part_*.parquet"))


def pack_split_join(
    packer: HierarchicalPacker,
    source: FrameOrLazy,
    to_level: str,
) -> pl.LazyFrame:
    """
    Split-and-join pack: reattach the root level's heavy attributes via a join
    instead of carrying them through the aggregation.

    This now delegates to the shipped library implementation
    (``pack(..., parent_strategy="split_join")``) so the benchmark measures the
    real code path. See :meth:`HierarchicalPacker.pack`.
    """
    return _to_lazy(packer.pack(_to_lazy(source), to_level, parent_strategy="split_join"))


def _canonical_rows(frame: FrameOrLazy) -> list[str]:
    """Order- and field-order-independent canonical form of a frame's rows."""
    df = frame.collect() if isinstance(frame, pl.LazyFrame) else frame
    return sorted(json.dumps(row, sort_keys=True, default=str) for row in df.to_dicts())


def assert_strategies_match_pack(
    packer: HierarchicalPacker, source: FrameOrLazy, to_level: str
) -> None:
    """
    Assert every experimental strategy reproduces ``pack``'s contents.

    Compared order-independently (top-level row order and struct field order are
    not significant). Raises ``AssertionError`` on the first mismatch.
    """
    lazy = _to_lazy(source)
    expected = _canonical_rows(packer.pack(lazy, to_level))

    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        variants = {
            "pack_filter_rescan": pack_filter_rescan(
                packer, lazy, to_level, partitions=4, tmp_dir=d1
            ),
            "pack_single_pass": pack_single_pass(packer, lazy, to_level, partitions=4, tmp_dir=d2),
            "pack_split_join": pack_split_join(packer, lazy, to_level),
        }
        for name, variant in variants.items():
            if _canonical_rows(variant) != expected:
                raise AssertionError(f"Strategy {name!r} does not match pack() contents.")

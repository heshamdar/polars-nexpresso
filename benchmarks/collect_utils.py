"""Streaming collect helper for benchmarks (targets a modern Polars)."""

from __future__ import annotations

import polars as pl


def streaming_collect(lazy_frame: pl.LazyFrame) -> pl.DataFrame:
    """Collect a LazyFrame using the streaming engine."""
    return lazy_frame.collect(engine="streaming")

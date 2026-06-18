"""Polars version-aware streaming collect helpers for benchmarks."""

from __future__ import annotations

from functools import lru_cache

import polars as pl
from packaging import version


@lru_cache(maxsize=1)
def _polars_version() -> version.Version:
    """Return the installed Polars version."""
    return version.parse(pl.__version__)


@lru_cache(maxsize=1)
def streaming_collect_mode() -> str:
    """
    Return how this Polars version requests streaming execution.

    Returns:
        ``"engine"`` when ``collect(engine="streaming")`` is supported, else
        ``"streaming_flag"`` for ``collect(streaming=True)``.
    """
    if _polars_version() >= version.parse("1.30.0"):
        return "engine"
    return "streaming_flag"


def streaming_collect(lazy_frame: pl.LazyFrame) -> pl.DataFrame:
    """
    Collect a LazyFrame using the streaming engine for the installed Polars version.

    Polars < 1.30 uses ``streaming=True``; newer versions use ``engine="streaming"``.
    """
    if streaming_collect_mode() == "engine":
        return lazy_frame.collect(engine="streaming")
    return lazy_frame.collect(streaming=True)  # type: ignore[call-overload, no-any-return]

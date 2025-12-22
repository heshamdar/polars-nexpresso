"""
Pytest configuration and fixtures for polars-nexpresso tests.

This module provides:
- Version checking utilities for Polars feature compatibility
- Shared fixtures for tests
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import polars as pl
import pytest
from packaging import version

if TYPE_CHECKING:
    pass


@lru_cache(maxsize=1)
def get_polars_version() -> version.Version:
    """Get the current Polars version as a parsed Version object."""
    return version.parse(pl.__version__)


def polars_version_at_least(min_version: str) -> bool:
    """
    Check if the installed Polars version is at least the specified version.

    Args:
        min_version: Minimum required version (e.g., "1.0.0")

    Returns:
        True if current version >= min_version
    """
    return get_polars_version() >= version.parse(min_version)


def polars_version_below(max_version: str) -> bool:
    """
    Check if the installed Polars version is below the specified version.

    Args:
        max_version: Maximum version (exclusive) (e.g., "1.0.0")

    Returns:
        True if current version < max_version
    """
    return get_polars_version() < version.parse(max_version)


# =============================================================================
# Skip Markers for Version-Specific Features
# =============================================================================

# arr.eval() was added in Polars 1.35.1
requires_arr_eval = pytest.mark.skipif(
    polars_version_below("1.35.1"),
    reason="arr.eval() requires Polars >= 1.35.1",
)

# struct.with_fields() behavior changed in 0.19.0
requires_struct_with_fields = pytest.mark.skipif(
    polars_version_below("0.19.0"),
    reason="struct.with_fields() requires Polars >= 0.19.0",
)

# list.eval() with pl.element() became stable in 0.18.0
requires_list_eval = pytest.mark.skipif(
    polars_version_below("0.18.0"),
    reason="list.eval() with pl.element() requires Polars >= 0.18.0",
)

# LazyFrame.collect_schema() was added in 0.20.0
requires_collect_schema = pytest.mark.skipif(
    polars_version_below("0.20.0"),
    reason="LazyFrame.collect_schema() requires Polars >= 0.20.0",
)

# group_by with maintain_order parameter
requires_group_by_maintain_order = pytest.mark.skipif(
    polars_version_below("0.19.0"),
    reason="group_by(maintain_order=True) requires Polars >= 0.19.0",
)


# =============================================================================
# Helper function for custom version requirements
# =============================================================================


def skip_if_polars_below(min_version: str, reason: str | None = None):
    """
    Create a pytest skip marker for tests requiring a minimum Polars version.

    Args:
        min_version: Minimum required Polars version (e.g., "1.5.0")
        reason: Custom reason message (optional)

    Returns:
        pytest.mark.skipif marker

    Example:
        @skip_if_polars_below("1.5.0")
        def test_new_feature():
            ...
    """
    if reason is None:
        reason = f"Test requires Polars >= {min_version}"

    return pytest.mark.skipif(
        polars_version_below(min_version),
        reason=reason,
    )


# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
def simple_nested_df() -> pl.DataFrame:
    """Create a simple DataFrame with nested structure for testing."""
    return pl.DataFrame(
        {
            "id": [1, 2],
            "data": [
                {"name": "Alice", "value": 100},
                {"name": "Bob", "value": 200},
            ],
        }
    )


@pytest.fixture
def list_of_structs_df() -> pl.DataFrame:
    """Create a DataFrame with list of structs for testing."""
    return pl.DataFrame(
        {
            "id": [1, 2],
            "items": [
                [{"name": "A", "qty": 2}, {"name": "B", "qty": 3}],
                [{"name": "C", "qty": 1}],
            ],
        }
    )

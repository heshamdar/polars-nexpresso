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

from nexpresso import HierarchicalPacker, HierarchySpec, LevelSpec

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


# =============================================================================
# Branching (multi-axis) hierarchy fixtures
# =============================================================================
#
#   country
#     └── city
#           ├── street ── building     (the "building" axis)
#           └── service                (the "service" axis)
#
# ``street`` and ``service`` are siblings: both are properties of a city, and
# neither is a stage of the other's chain.


@pytest.fixture
def branching_spec() -> HierarchySpec:
    """A hierarchy where ``city`` carries two independent child branches."""
    return HierarchySpec.from_levels(
        LevelSpec(name="country", id_fields=["code"]),
        LevelSpec(name="city", id_fields=["id"], parent="country", parent_keys=["code"]),
        LevelSpec(name="street", id_fields=["id"], parent="city", parent_keys=["id"]),
        LevelSpec(name="building", id_fields=["id"], parent="street", parent_keys=["id"]),
        LevelSpec(name="service", id_fields=["kind"], parent="city", parent_keys=["id"]),
    )


@pytest.fixture
def branching_packer(branching_spec: HierarchySpec) -> HierarchicalPacker:
    """A packer over :func:`branching_spec`."""
    return HierarchicalPacker(branching_spec)


@pytest.fixture
def branching_tables() -> dict[str, pl.DataFrame]:
    """
    Normalized per-level tables for the branching hierarchy.

    NYC has two streets and two services; LA and PAR have one of each. This is
    the shape :meth:`HierarchicalPacker.normalize` emits — level-local columns
    plus ancestor keys as foreign keys.
    """
    return {
        "country": pl.DataFrame({"country.code": ["US", "FR"], "country.name": ["USA", "France"]}),
        "city": pl.DataFrame(
            {
                "country.code": ["US", "US", "FR"],
                "country.city.id": ["NYC", "LA", "PAR"],
                "country.city.population": [8, 4, 2],
            }
        ),
        "street": pl.DataFrame(
            {
                "country.code": ["US", "US", "US", "FR"],
                "country.city.id": ["NYC", "NYC", "LA", "PAR"],
                "country.city.street.id": ["s1", "s2", "s3", "s4"],
                "country.city.street.length": [100, 200, 300, 400],
            }
        ),
        "building": pl.DataFrame(
            {
                "country.code": ["US", "US", "US", "FR"],
                "country.city.id": ["NYC", "NYC", "LA", "PAR"],
                "country.city.street.id": ["s1", "s2", "s3", "s4"],
                "country.city.street.building.id": ["b1", "b2", "b3", "b4"],
                "country.city.street.building.floors": [10, 20, 30, 40],
            }
        ),
        "service": pl.DataFrame(
            {
                "country.code": ["US", "US", "US", "FR"],
                "country.city.id": ["NYC", "NYC", "LA", "PAR"],
                "country.city.service.kind": ["police", "fire", "water", "medical"],
                "country.city.service.budget": [100, 200, 300, 400],
            }
        ),
    }


@pytest.fixture
def branching_nested(
    branching_packer: HierarchicalPacker, branching_tables: dict[str, pl.DataFrame]
) -> pl.DataFrame:
    """The branching hierarchy fully packed into a single nested column."""
    return branching_packer.denormalize(  # type: ignore[return-value]
        branching_tables, at_level="country"
    )

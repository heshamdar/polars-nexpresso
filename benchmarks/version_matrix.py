"""Shared Polars version lists for matrix runners."""

from __future__ import annotations

# Keep aligned with tests/test_matrix.py DEFAULT_VERSIONS.
DEFAULT_POLARS_VERSIONS: tuple[str, ...] = (
    "1.20.0",  # Minimum supported version
    "1.30.0",  # Intermediate version
    "1.35.1",  # arr.eval() baseline
    "latest",  # Latest available version
)


def resolve_versions(
    *,
    versions: list[str] | None,
    min_version: str | None,
    skip_versions: list[str],
) -> list[str]:
    """Resolve which Polars versions to run from CLI options."""
    if versions:
        selected = versions
    elif min_version:
        selected = [v for v in DEFAULT_POLARS_VERSIONS if v == "latest" or v >= min_version]
    else:
        selected = list(DEFAULT_POLARS_VERSIONS)

    return [version for version in selected if version not in skip_versions]

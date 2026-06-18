"""Synthetic leaf-level DataFrame generation for packer benchmarks."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

from benchmarks.config import BenchmarkConfig
from nexpresso import HierarchySpec, LevelSpec

PIXELS_COLUMN = "image.tile.patch.pixels"

IMAGE_SPEC = HierarchySpec.from_levels(
    LevelSpec(name="image", id_fields=["id"]),
    LevelSpec(name="tile", id_fields=["id"]),
    LevelSpec(name="patch", id_fields=["id"]),
)


def _polars_pixel_dtype(config: BenchmarkConfig) -> pl.Array:
    """Return the Polars Array dtype for the pixel payload column."""
    if config.pixel_dtype == "f32":
        inner_dtype: pl.DataType = pl.Float32()
    else:
        inner_dtype = pl.UInt8()
    return pl.Array(inner_dtype, config.pixels_per_patch)


def _grid_dims(count: int) -> tuple[int, int]:
    """Return (rows, cols) for a roughly square grid holding ``count`` cells."""
    side = max(1, int(math.sqrt(count)))
    cols = max(1, math.ceil(count / side))
    return side, cols


def _make_pixel_series(
    config: BenchmarkConfig,
    rng: np.random.Generator,
    n_leaf_rows: int,
) -> pl.Series:
    """Build the heavy pixel payload column."""
    shape = (n_leaf_rows, config.patch_height, config.patch_width)

    if config.pixel_dtype == "f32":
        float_data = rng.random(shape, dtype=np.float32)
        data: np.ndarray[Any, Any] = float_data
    else:
        data = rng.integers(0, 256, size=shape, dtype=np.uint8)

    if config.payload_type == "array":
        flat = data.reshape(n_leaf_rows, config.pixels_per_patch)
        return pl.Series(PIXELS_COLUMN, flat, dtype=_polars_pixel_dtype(config))

    inner = pl.Float32 if config.pixel_dtype == "f32" else pl.UInt8
    return pl.Series(
        PIXELS_COLUMN,
        data.tolist(),
        dtype=pl.List(pl.List(inner)),
    )


def generate_leaf_dataframe(config: BenchmarkConfig) -> pl.DataFrame:
    """
    Generate a flat leaf-level DataFrame for the image → tile → patch hierarchy.

    Args:
        config: Benchmark scale and payload configuration.

    Returns:
        DataFrame with one row per patch and dotted hierarchy column names.
    """
    n_leaf_rows = config.n_leaf_rows
    rng = np.random.default_rng(config.seed)

    image_idx = np.repeat(
        np.arange(config.n_images), config.tiles_per_image * config.patches_per_tile
    )
    tile_idx = np.tile(
        np.repeat(np.arange(config.tiles_per_image), config.patches_per_tile),
        config.n_images,
    )
    patch_idx = np.tile(
        np.arange(config.patches_per_tile), config.n_images * config.tiles_per_image
    )

    tile_rows, tile_cols = _grid_dims(config.tiles_per_image)
    patch_rows, patch_cols = _grid_dims(config.patches_per_tile)

    image_width = config.patch_width * tile_cols * patch_cols
    image_height = config.patch_height * tile_rows * patch_rows

    pixel_series = _make_pixel_series(config, rng, n_leaf_rows)

    return pl.DataFrame(
        {
            "image.id": [f"img-{i}" for i in image_idx],
            "image.width": pl.Series(
                "image.width",
                np.full(n_leaf_rows, image_width, dtype=np.int32),
            ),
            "image.height": pl.Series(
                "image.height",
                np.full(n_leaf_rows, image_height, dtype=np.int32),
            ),
            "image.tile.id": [f"tile-{i}-{t}" for i, t in zip(image_idx, tile_idx, strict=True)],
            "image.tile.row": (tile_idx // tile_cols).astype(np.int32),
            "image.tile.col": (tile_idx % tile_cols).astype(np.int32),
            "image.tile.patch.id": [
                f"patch-{i}-{t}-{p}" for i, t, p in zip(image_idx, tile_idx, patch_idx, strict=True)
            ],
            "image.tile.patch.row": (patch_idx // patch_cols).astype(np.int32),
            "image.tile.patch.col": (patch_idx % patch_cols).astype(np.int32),
            PIXELS_COLUMN: pixel_series,
        }
    )


def check_invariants(
    leaf_df: pl.DataFrame,
    packed_df: pl.DataFrame,
    unpacked_df: pl.DataFrame,
    config: BenchmarkConfig,
) -> None:
    """
    Verify cheap row-count invariants after pack/unpack.

    Raises:
        AssertionError: If any invariant fails.
    """
    if leaf_df.height != config.n_leaf_rows:
        raise AssertionError(f"Expected {config.n_leaf_rows} leaf rows, got {leaf_df.height}")
    if packed_df.height != config.n_images:
        raise AssertionError(f"Expected {config.n_images} packed rows, got {packed_df.height}")
    if unpacked_df.height != config.n_leaf_rows:
        raise AssertionError(
            f"Expected {config.n_leaf_rows} unpacked rows, got {unpacked_df.height}"
        )
    if PIXELS_COLUMN not in unpacked_df.columns:
        raise AssertionError(f"Missing payload column {PIXELS_COLUMN!r} after unpack")

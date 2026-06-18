"""Configuration for hierarchical packer streaming benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PayloadType = Literal["array", "list_of_lists"]
PixelDtype = Literal["f32", "u8"]
PresetName = Literal["smoke", "medium", "large", "very_large"]


@dataclass(frozen=True)
class BenchmarkConfig:
    """Parameters controlling synthetic data scale and payload shape."""

    n_images: int = 50
    tiles_per_image: int = 16
    patches_per_tile: int = 16
    patch_height: int = 64
    patch_width: int = 64
    payload_type: PayloadType = "array"
    pixel_dtype: PixelDtype = "f32"
    seed: int = 0

    @property
    def n_leaf_rows(self) -> int:
        """Total patch-level rows in the flat leaf DataFrame."""
        return self.n_images * self.tiles_per_image * self.patches_per_tile

    @property
    def pixels_per_patch(self) -> int:
        """Number of scalar pixel values per patch."""
        return self.patch_height * self.patch_width

    @property
    def payload_bytes_per_row(self) -> int:
        """Approximate payload bytes per leaf row (pixels only)."""
        bytes_per_pixel = 4 if self.pixel_dtype == "f32" else 1
        return self.pixels_per_patch * bytes_per_pixel

    @property
    def estimated_payload_mb(self) -> float:
        """Rough total payload size in megabytes."""
        return (self.n_leaf_rows * self.payload_bytes_per_row) / (1024 * 1024)

    def label(self) -> str:
        """Short human-readable label for result tables."""
        return (
            f"images={self.n_images},tiles={self.tiles_per_image},"
            f"patches={self.patches_per_tile},{self.patch_height}x{self.patch_width},"
            f"{self.payload_type},{self.pixel_dtype}"
        )


PRESETS: dict[PresetName, BenchmarkConfig] = {
    "smoke": BenchmarkConfig(
        n_images=5,
        tiles_per_image=4,
        patches_per_tile=4,
        patch_height=16,
        patch_width=16,
    ),
    "medium": BenchmarkConfig(),
    "large": BenchmarkConfig(
        n_images=200,
        tiles_per_image=16,
        patches_per_tile=16,
        patch_height=64,
        patch_width=64,
    ),
    "very_large": BenchmarkConfig(
        n_images=400,
        tiles_per_image=16,
        patches_per_tile=16,
        patch_height=64,
        patch_width=64,
    ),
}

#!/usr/bin/env python3
"""CLI for streaming hierarchical pack/unpack benchmarks."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import polars as pl

from benchmarks import strategies
from benchmarks.collect_utils import streaming_collect
from benchmarks.config import PRESETS, BenchmarkConfig
from benchmarks.data_generator import IMAGE_SPEC, check_invariants, generate_leaf_dataframe
from benchmarks.harness import (
    BenchmarkResult,
    OperationName,
    run_benchmark_repeats,
    summarize_results,
)
from nexpresso import HierarchicalPacker

DEFAULT_OPERATIONS: tuple[OperationName, ...] = (
    "pack_to_image",
    "unpack_to_patch",
    "roundtrip",
)


def _parse_operations(raw: str | None) -> list[OperationName]:
    """Parse a comma-separated operations string."""
    if raw is None:
        return list(DEFAULT_OPERATIONS)

    aliases = {
        "pack": "pack_to_image",
        "pack_streaming": "pack_streaming_to_image",
        "pack_streaming_singlepass": "pack_streaming_singlepass",
        "pack_singlepass": "pack_streaming_singlepass",
        "pack_no_order": "pack_no_child_order",
        "pack_no_child_order": "pack_no_child_order",
        "pack_split_join": "pack_split_join",
        "pack_splitjoin": "pack_split_join",
        "unpack": "unpack_to_patch",
        "roundtrip": "roundtrip",
        "pack_to_image": "pack_to_image",
        "pack_streaming_to_image": "pack_streaming_to_image",
        "unpack_to_patch": "unpack_to_patch",
    }
    operations: list[OperationName] = []
    for part in raw.split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name not in aliases:
            raise argparse.ArgumentTypeError(
                f"Unknown operation {name!r}. Choose from: pack, pack_streaming, "
                "pack_singlepass, pack_no_order, pack_split_join, unpack, roundtrip"
            )
        operations.append(aliases[name])  # type: ignore[arg-type]
    if not operations:
        raise argparse.ArgumentTypeError("At least one operation is required")
    return operations


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Benchmark HierarchicalPacker pack/unpack with streaming collect.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        help="Use a built-in scale preset (smoke, medium, large).",
    )
    parser.add_argument("--n-images", type=int, help="Number of root image entities.")
    parser.add_argument("--tiles-per-image", type=int, help="Tile children per image.")
    parser.add_argument("--patches-per-tile", type=int, help="Patch children per tile.")
    parser.add_argument("--patch-height", type=int, help="Pixel rows per patch.")
    parser.add_argument("--patch-width", type=int, help="Pixel columns per patch.")
    parser.add_argument(
        "--payload-type",
        choices=["array", "list_of_lists"],
        help="Pixel payload storage type.",
    )
    parser.add_argument(
        "--pixel-dtype",
        choices=["f32", "u8"],
        help="Scalar dtype for pixel values.",
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducible payloads.")
    parser.add_argument(
        "--stream-partitions",
        type=int,
        help="Root-key buckets for pack_streaming (more = lower peak memory).",
    )
    parser.add_argument(
        "--parent-payload-pixels",
        type=int,
        help="Heavy per-image thumbnail payload (Float32 pixels) carried on every "
        "leaf row. Makes split-and-join vs dedup meaningful.",
    )
    parser.add_argument(
        "--parent-attr-count",
        type=int,
        help="Number of extra redundant scalar attributes at the image (root) level.",
    )
    parser.add_argument(
        "--operations",
        type=_parse_operations,
        help="Comma-separated operations: pack, unpack, roundtrip.",
    )
    parser.add_argument("--warmup", type=int, default=0, help="Untimed warmup runs per operation.")
    parser.add_argument("--repeat", type=int, default=1, help="Timed repeats per operation.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path for results.")
    parser.add_argument("--quiet", action="store_true", help="Suppress non-result stdout.")
    parser.add_argument(
        "--check-invariants",
        action="store_true",
        help=(
            "Run pack/unpack row-count invariant checks before benchmarking. "
            "This materializes the generated dataset in the parent process."
        ),
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    """Resolve final benchmark config from CLI args and optional preset."""
    has_overrides = any(
        value is not None
        for value in (
            args.n_images,
            args.tiles_per_image,
            args.patches_per_tile,
            args.patch_height,
            args.patch_width,
            args.payload_type,
            args.pixel_dtype,
            args.seed,
            args.parent_payload_pixels,
            args.parent_attr_count,
        )
    )
    if args.preset is not None:
        base = PRESETS[args.preset]
    elif has_overrides:
        base = BenchmarkConfig()
    else:
        base = PRESETS["medium"]

    return BenchmarkConfig(
        n_images=args.n_images if args.n_images is not None else base.n_images,
        tiles_per_image=(
            args.tiles_per_image if args.tiles_per_image is not None else base.tiles_per_image
        ),
        patches_per_tile=(
            args.patches_per_tile if args.patches_per_tile is not None else base.patches_per_tile
        ),
        patch_height=args.patch_height if args.patch_height is not None else base.patch_height,
        patch_width=args.patch_width if args.patch_width is not None else base.patch_width,
        payload_type=args.payload_type or base.payload_type,
        pixel_dtype=args.pixel_dtype or base.pixel_dtype,
        seed=args.seed if args.seed is not None else base.seed,
        stream_partitions=(
            args.stream_partitions if args.stream_partitions is not None else base.stream_partitions
        ),
        parent_payload_pixels=(
            args.parent_payload_pixels
            if args.parent_payload_pixels is not None
            else base.parent_payload_pixels
        ),
        parent_attr_count=(
            args.parent_attr_count if args.parent_attr_count is not None else base.parent_attr_count
        ),
    )


def _log(message: str, *, quiet: bool) -> None:
    if not quiet:
        print(message)


def _run_invariant_checks(config: BenchmarkConfig, quiet: bool) -> float:
    """Generate data and verify pack/unpack row-count invariants."""
    _log("Running invariant checks...", quiet=quiet)
    start = time.perf_counter()
    leaf_df = generate_leaf_dataframe(config)
    gen_elapsed = time.perf_counter() - start

    packer = HierarchicalPacker(IMAGE_SPEC, validate_on_pack=False)
    packed_df = streaming_collect(packer.pack(leaf_df.lazy(), "image"))
    unpacked_df = streaming_collect(packer.unpack(packed_df.lazy(), "patch"))
    check_invariants(leaf_df, packed_df, unpacked_df, config)

    # The experimental strategies must reproduce pack()'s contents exactly.
    strategies.assert_strategies_match_pack(packer, leaf_df.lazy(), "image")

    _log(f"Invariant checks passed (data generation: {gen_elapsed:.3f}s)", quiet=quiet)
    return gen_elapsed


def _log_generation_memory_note(config: BenchmarkConfig, quiet: bool) -> None:
    """Print guidance for large synthetic payloads."""
    if config.estimated_payload_mb < 1024 and config.payload_type == "array":
        return

    _log(
        "Data generation materializes the full pixel payload before benchmarking; "
        f"raw pixels alone are about {config.estimated_payload_mb:.1f} MiB.",
        quiet=quiet,
    )
    if config.payload_type == "list_of_lists":
        _log(
            "`list_of_lists` converts the NumPy buffer through Python lists, so peak memory "
            "can be several times the raw payload size.",
            quiet=quiet,
        )


def _add_baseline_ratios(rows: list[dict[str, Any]], baseline: str = "pack_to_image") -> bool:
    """
    Annotate rows with time/memory ratios relative to the baseline operation.

    Returns:
        True if the baseline was present and ratio columns were added.
    """
    base = next((r for r in rows if r["operation"] == baseline), None)
    if base is None or len(rows) < 2:
        return False

    base_t = base["_median_s"]
    base_m = base["_peak_rss_mb"]
    for row in rows:
        row["t_vs_pack"] = f"{row['_median_s'] / base_t:.2f}x" if base_t else "-"
        row["mem_vs_pack"] = f"{row['_peak_rss_mb'] / base_m:.2f}x" if base_m else "-"
    return True


def _format_table(rows: list[dict[str, Any]], *, with_ratios: bool = False) -> str:
    """Format benchmark summary rows as a fixed-width table."""
    headers = ["operation", "rows", "median_s", "min_s", "peak_rss_mb"]
    if with_ratios:
        headers += ["t_vs_pack", "mem_vs_pack"]
    headers += ["polars"]

    col_widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            col_widths[header] = max(col_widths[header], len(str(row.get(header, ""))))

    def fmt_row(values: dict[str, Any]) -> str:
        return "  ".join(str(values.get(h, "")).ljust(col_widths[h]) for h in headers)

    lines = [fmt_row({h: h for h in headers}), fmt_row({h: "-" * col_widths[h] for h in headers})]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run pack/unpack streaming benchmarks."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    operations = args.operations or list(DEFAULT_OPERATIONS)

    _log(f"Polars {pl.__version__} (streaming engine)", quiet=args.quiet)
    _log(f"Config: {config.label()}", quiet=args.quiet)
    _log(
        f"Leaf rows: {config.n_leaf_rows:,} | est. payload: {config.estimated_payload_mb:.1f} MB",
        quiet=args.quiet,
    )
    _log_generation_memory_note(config, quiet=args.quiet)

    gen_elapsed = 0.0
    if args.check_invariants:
        gen_elapsed = _run_invariant_checks(config, quiet=args.quiet)

    all_results: list[BenchmarkResult] = []
    summary_rows: list[dict[str, Any]] = []

    for operation in operations:
        _log(
            f"\nBenchmarking {operation} (warmup={args.warmup}, repeat={args.repeat})...",
            quiet=args.quiet,
        )
        repeats = run_benchmark_repeats(
            config,
            operation,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        all_results.extend(repeats)
        stats = summarize_results(repeats)
        summary_rows.append(
            {
                "operation": operation,
                "rows": config.n_leaf_rows,
                "median_s": f"{stats['median_elapsed_s']:.3f}",
                "min_s": f"{stats['min_elapsed_s']:.3f}",
                "peak_rss_mb": f"{stats['max_peak_rss_mb']:.1f}",
                "polars": pl.__version__,
                "_median_s": stats["median_elapsed_s"],
                "_peak_rss_mb": stats["max_peak_rss_mb"],
            }
        )
        if operation == "roundtrip" and repeats:
            pack_times = [r.pack_elapsed_s for r in repeats if r.pack_elapsed_s is not None]
            unpack_times = [r.unpack_elapsed_s for r in repeats if r.unpack_elapsed_s is not None]
            if pack_times and unpack_times:
                _log(
                    f"  roundtrip sub-timings: pack={statistics.median(pack_times):.3f}s, "
                    f"unpack={statistics.median(unpack_times):.3f}s",
                    quiet=args.quiet,
                )

    with_ratios = _add_baseline_ratios(summary_rows)
    _log("\n" + _format_table(summary_rows, with_ratios=with_ratios), quiet=args.quiet)
    if with_ratios:
        _log("\n(t_vs_pack / mem_vs_pack are relative to pack_to_image)", quiet=args.quiet)
    if gen_elapsed:
        _log(f"\nParent invariant data generation: {gen_elapsed:.3f}s", quiet=args.quiet)

    if args.output is not None:
        public_summary = [
            {k: v for k, v in row.items() if not k.startswith("_")} for row in summary_rows
        ]
        payload = {
            "config": asdict(config),
            "data_generation_s": gen_elapsed,
            "results": [result.to_dict() for result in all_results],
            "summary": public_summary,
        }
        args.output.write_text(json.dumps(payload, indent=2))
        _log(f"\nWrote results to {args.output}", quiet=args.quiet)

    return 0


if __name__ == "__main__":
    sys.exit(main())

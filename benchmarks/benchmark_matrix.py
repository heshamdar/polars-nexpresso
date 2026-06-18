#!/usr/bin/env python3
"""
Benchmark matrix runner for polars-nexpresso.

Runs streaming pack/unpack benchmarks against multiple Polars versions using
isolated uv environments, similar to tests/test_matrix.py.

Usage:
    uv run python -m benchmarks.benchmark_matrix --preset smoke
    uv run python -m benchmarks.benchmark_matrix --versions 1.20.0 1.35.1 latest --preset medium
    uv run python -m benchmarks.benchmark_matrix --min-version 1.30.0 --operations pack
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from benchmarks.version_matrix import DEFAULT_POLARS_VERSIONS, resolve_versions

MATRIX_OUTPUT_DEFAULT = "benchmark_matrix_results.json"


def run_command(cmd: list[str], *, cwd: Path | None = None) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, and stderr."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as exc:  # noqa: BLE001 - surface subprocess setup failures
        return 1, "", str(exc)


def _polars_dependency_spec(version: str) -> str:
    """Return a uv/pip dependency string for a Polars version target."""
    return "polars" if version == "latest" else f"polars=={version}"


def _write_matrix_pyproject(env_dir: Path, polars_version: str) -> None:
    """Create an isolated project definition for one Polars version."""
    polars_spec = _polars_dependency_spec(polars_version)
    pyproject_content = f"""[project]
name = "polars-nexpresso-benchmark"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "{polars_spec}",
    "packaging>=21.0",
    "psutil>=6.0",
    "numpy>=1.26",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["nexpresso"]
"""
    (env_dir / "pyproject.toml").write_text(pyproject_content)


def _copy_project_sources(project_root: Path, env_dir: Path) -> None:
    """Copy library and benchmark sources into an isolated environment."""
    shutil.copytree(
        project_root / "nexpresso",
        env_dir / "nexpresso",
        dirs_exist_ok=True,
    )
    shutil.copytree(
        project_root / "benchmarks",
        env_dir / "benchmarks",
        dirs_exist_ok=True,
    )


def _get_installed_polars_version(env_dir: Path) -> str:
    """Return the Polars version installed in an isolated environment."""
    cmd = [
        "uv",
        "run",
        "--project",
        str(env_dir),
        "python",
        "-c",
        "import polars; print(polars.__version__)",
    ]
    exit_code, stdout, stderr = run_command(cmd)
    if exit_code != 0:
        raise RuntimeError(f"Failed to determine Polars version:\n{stderr}")
    return stdout.strip()


def _setup_version_environment(project_root: Path, polars_version: str) -> tuple[Path, Any]:
    """Create a temp directory with sources and sync dependencies for one version."""
    tmpdir = tempfile.TemporaryDirectory(prefix=f"polars-bench-{polars_version}-")
    env_dir = Path(tmpdir.name)
    _write_matrix_pyproject(env_dir, polars_version)
    _copy_project_sources(project_root, env_dir)

    sync_cmd = ["uv", "sync", "--project", str(env_dir)]
    exit_code, stdout, stderr = run_command(sync_cmd)
    if exit_code != 0:
        tmpdir.cleanup()
        raise RuntimeError(
            f"Failed to setup environment for Polars {polars_version}:\n{stderr}\n{stdout}"
        )

    return env_dir, tmpdir


def run_benchmark_for_version(
    polars_version: str,
    project_root: Path,
    bench_argv: list[str],
    *,
    output_path: Path,
) -> tuple[bool, dict[str, Any], str]:
    """
    Run benchmarks in an isolated environment for one Polars version.

    Returns:
        Tuple of (success, result_payload, actual_polars_version).
    """
    print(f"\n{'=' * 80}")
    print(f"Benchmarking with Polars {polars_version}")
    print(f"{'=' * 80}")

    env_dir, tmpdir = _setup_version_environment(project_root, polars_version)
    try:
        actual_version = _get_installed_polars_version(env_dir)
        if polars_version == "latest":
            print(f"Installed Polars version: {actual_version}")

        bench_cmd = [
            "uv",
            "run",
            "--project",
            str(env_dir),
            "python",
            "-m",
            "benchmarks.bench_packer",
            *bench_argv,
            "--output",
            str(output_path),
            "--quiet",
        ]
        print(f"Running: {' '.join(bench_cmd)}")
        exit_code, stdout, stderr = run_command(bench_cmd)

        if exit_code != 0:
            error_msg = f"Benchmark failed for Polars {actual_version}:\n{stderr}\n{stdout}"
            print(f"FAILED {error_msg}")
            return False, {"error": error_msg}, actual_version

        payload = json.loads(output_path.read_text())
        payload["requested_polars_version"] = polars_version
        payload["actual_polars_version"] = actual_version
        print(f"Completed Polars {actual_version}")
        if stdout.strip():
            print(stdout.strip())
        return True, payload, actual_version
    finally:
        tmpdir.cleanup()


def _format_comparison_table(version_payloads: dict[str, dict[str, Any]]) -> str:
    """Build a cross-version comparison table from per-version benchmark JSON."""
    versions = list(version_payloads.keys())
    operations: list[str] = []
    for payload in version_payloads.values():
        for row in payload.get("summary", []):
            operation = row["operation"]
            if operation not in operations:
                operations.append(operation)

    if not versions or not operations:
        return "No benchmark results to compare."

    headers = ["operation", *versions]
    table_rows: list[dict[str, str]] = []
    for operation in operations:
        table_row: dict[str, str] = {"operation": operation}
        for version in versions:
            summary = next(
                (
                    item
                    for item in version_payloads[version].get("summary", [])
                    if item["operation"] == operation
                ),
                None,
            )
            if summary is None:
                table_row[version] = "n/a"
            else:
                table_row[version] = f"{summary['median_s']}s / {summary['peak_rss_mb']}MB"
        table_rows.append(table_row)

    col_widths = {header: len(header) for header in headers}
    for table_row in table_rows:
        for header in headers:
            col_widths[header] = max(col_widths[header], len(table_row.get(header, "")))

    def fmt_row(values: dict[str, str]) -> str:
        return "  ".join(values[header].ljust(col_widths[header]) for header in headers)

    lines = [
        fmt_row({header: header for header in headers}),
        fmt_row({header: "-" * col_widths[header] for header in headers}),
    ]
    lines.extend(fmt_row(table_row) for table_row in table_rows)
    return "\n".join(lines)


def _build_matrix_parser() -> argparse.ArgumentParser:
    """Build the matrix CLI parser; unknown args are forwarded to bench_packer."""
    parser = argparse.ArgumentParser(
        description="Run pack/unpack benchmarks across multiple Polars versions.",
        epilog=(
            "All unrecognized arguments are forwarded to benchmarks.bench_packer "
            "(e.g. --preset smoke --operations pack,unpack --repeat 3)."
        ),
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        help=f"Polars versions to benchmark (default: {' '.join(DEFAULT_POLARS_VERSIONS)})",
    )
    parser.add_argument(
        "--min-version",
        help="Benchmark from this Polars version onward.",
    )
    parser.add_argument(
        "--skip-versions",
        nargs="+",
        default=[],
        help="Polars versions to skip.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop on the first failed version.",
    )
    parser.add_argument(
        "--matrix-output",
        type=Path,
        default=Path(MATRIX_OUTPUT_DEFAULT),
        help=f"Aggregate JSON output path (default: {MATRIX_OUTPUT_DEFAULT}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Polars version benchmark matrix."""
    parser = _build_matrix_parser()
    args, bench_argv = parser.parse_known_args(argv)

    versions_to_run = resolve_versions(
        versions=args.versions,
        min_version=args.min_version,
        skip_versions=args.skip_versions,
    )
    if not versions_to_run:
        print("No Polars versions selected.")
        return 1

    project_root = Path(__file__).resolve().parent.parent
    print(f"Benchmarking against {len(versions_to_run)} Polars version(s)")
    print(f"Versions: {', '.join(versions_to_run)}")
    if bench_argv:
        print(f"Forwarded bench args: {' '.join(bench_argv)}")

    version_results: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="polars-bench-results-") as results_dir:
        results_path = Path(results_dir)
        for version in versions_to_run:
            output_path = results_path / f"bench_{version.replace('.', '_')}.json"
            success, payload, actual_version = run_benchmark_for_version(
                version,
                project_root,
                bench_argv,
                output_path=output_path,
            )
            version_results[actual_version] = payload
            if not success:
                failures.append(actual_version)
                if args.stop_on_failure:
                    print(f"\nStopping on first failure (Polars {actual_version})")
                    break

    aggregate = {
        "versions_requested": versions_to_run,
        "version_results": version_results,
        "failures": failures,
    }
    args.matrix_output.write_text(json.dumps(aggregate, indent=2))

    print(f"\n{'=' * 80}")
    print("BENCHMARK MATRIX SUMMARY (median_s / peak_rss_mb)")
    print(f"{'=' * 80}")
    print(_format_comparison_table(version_results))
    print(f"\nWrote aggregate results to {args.matrix_output}")

    if failures:
        print(f"\nFailed versions: {', '.join(failures)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

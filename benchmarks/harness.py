"""Subprocess benchmark harness with timing and peak RSS measurement."""

from __future__ import annotations

import gc
import multiprocessing as mp
import statistics
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

import polars as pl
import psutil  # type: ignore[import-untyped]

from benchmarks.collect_utils import streaming_collect
from benchmarks.config import BenchmarkConfig
from benchmarks.data_generator import IMAGE_SPEC, generate_leaf_dataframe
from nexpresso import HierarchicalPacker

OperationName = Literal["pack_to_image", "pack_streaming_to_image", "unpack_to_patch", "roundtrip"]


@dataclass(frozen=True)
class BenchmarkResult:
    """Single benchmark measurement."""

    operation: str
    config_label: str
    n_leaf_rows: int
    elapsed_s: float
    peak_rss_mb: float
    polars_version: str
    repeat_index: int = 0
    pack_elapsed_s: float | None = None
    unpack_elapsed_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class _WorkerResult:
    """Internal result payload from a benchmark worker process."""

    elapsed_s: float
    peak_rss_mb: float
    pack_elapsed_s: float | None = None
    unpack_elapsed_s: float | None = None
    error: str | None = None


def _bytes_to_mb(value: int) -> float:
    return value / (1024 * 1024)


def _monitor_peak_rss(
    peak_holder: list[int], stop_event: threading.Event, interval_s: float = 0.05
) -> None:
    """Poll process RSS until ``stop_event`` is set."""
    process = psutil.Process()
    while not stop_event.is_set():
        rss = process.memory_info().rss
        if rss > peak_holder[0]:
            peak_holder[0] = rss
        stop_event.wait(interval_s)


def _run_timed_collect(
    lazy_frame: pl.LazyFrame,
) -> tuple[pl.DataFrame, float, float]:
    """
    Collect a LazyFrame with the streaming engine while tracking peak RSS.

    Returns:
        Tuple of (result DataFrame, elapsed seconds, peak RSS in MB).
    """
    gc.collect()
    baseline_rss = psutil.Process().memory_info().rss
    peak_holder = [baseline_rss]
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=_monitor_peak_rss,
        args=(peak_holder, stop_event),
        daemon=True,
    )
    monitor.start()

    start = time.perf_counter()
    try:
        result = streaming_collect(lazy_frame)
    finally:
        stop_event.set()
        monitor.join(timeout=1.0)

    elapsed = time.perf_counter() - start
    peak_mb = _bytes_to_mb(peak_holder[0])
    return result, elapsed, peak_mb


def _worker_entry(
    config_dict: dict[str, Any],
    operation: OperationName,
    result_queue: mp.Queue[dict[str, Any]],
) -> None:
    """Benchmark worker executed in an isolated child process."""
    try:
        config = BenchmarkConfig(**config_dict)
        leaf_df = generate_leaf_dataframe(config)
        packer = HierarchicalPacker(IMAGE_SPEC, validate_on_pack=False)
        leaf_lf = leaf_df.lazy()

        if operation == "pack_to_image":
            _, elapsed, peak_mb = _run_timed_collect(packer.pack(leaf_lf, "image"))
            result_queue.put(
                asdict(
                    _WorkerResult(
                        elapsed_s=elapsed,
                        peak_rss_mb=peak_mb,
                    )
                )
            )
            return

        if operation == "pack_streaming_to_image":
            # Memory-bounded pack: partition by root key, sink each bucket, and
            # consume the result via a streaming count so the full nested result
            # is never materialized. Peak RSS reflects one bucket, not the dataset.
            gc.collect()
            baseline_rss = psutil.Process().memory_info().rss
            peak_holder = [baseline_rss]
            stop_event = threading.Event()
            monitor = threading.Thread(
                target=_monitor_peak_rss,
                args=(peak_holder, stop_event),
                daemon=True,
            )
            monitor.start()
            start = time.perf_counter()
            try:
                tmp_dir = tempfile.mkdtemp(prefix="nexpresso_bench_")
                packed_lf = packer.pack_streaming(
                    leaf_lf,
                    "image",
                    partitions=config.stream_partitions,
                    tmp_dir=tmp_dir,
                    defer=False,
                )
                packed_lf.select(pl.len()).collect(engine="streaming")
            finally:
                stop_event.set()
                monitor.join(timeout=1.0)
            elapsed = time.perf_counter() - start
            result_queue.put(
                asdict(
                    _WorkerResult(
                        elapsed_s=elapsed,
                        peak_rss_mb=_bytes_to_mb(peak_holder[0]),
                    )
                )
            )
            return

        if operation == "unpack_to_patch":
            packed_df, _, _ = _run_timed_collect(packer.pack(leaf_lf, "image"))
            gc.collect()
            _, elapsed, peak_mb = _run_timed_collect(
                packer.unpack(packed_df.lazy(), "patch"),
            )
            result_queue.put(
                asdict(
                    _WorkerResult(
                        elapsed_s=elapsed,
                        peak_rss_mb=peak_mb,
                    )
                )
            )
            return

        if operation == "roundtrip":
            gc.collect()
            baseline_rss = psutil.Process().memory_info().rss
            peak_holder = [baseline_rss]
            stop_event = threading.Event()
            monitor = threading.Thread(
                target=_monitor_peak_rss,
                args=(peak_holder, stop_event),
                daemon=True,
            )
            monitor.start()

            total_start = time.perf_counter()
            try:
                packed_df, pack_elapsed, _ = _run_timed_collect(packer.pack(leaf_lf, "image"))
                unpacked_df, unpack_elapsed, _ = _run_timed_collect(
                    packer.unpack(packed_df.lazy(), "patch"),
                )
            finally:
                stop_event.set()
                monitor.join(timeout=1.0)

            total_elapsed = time.perf_counter() - total_start
            _ = unpacked_df
            result_queue.put(
                asdict(
                    _WorkerResult(
                        elapsed_s=total_elapsed,
                        peak_rss_mb=_bytes_to_mb(peak_holder[0]),
                        pack_elapsed_s=pack_elapsed,
                        unpack_elapsed_s=unpack_elapsed,
                    )
                )
            )
            return

        result_queue.put(
            asdict(_WorkerResult(elapsed_s=0.0, peak_rss_mb=0.0, error=f"Unknown op: {operation}"))
        )
    except Exception as exc:  # noqa: BLE001 - propagate worker failures to parent
        result_queue.put(asdict(_WorkerResult(elapsed_s=0.0, peak_rss_mb=0.0, error=str(exc))))


def run_benchmark(
    config: BenchmarkConfig,
    operation: OperationName,
    *,
    timeout_s: float = 3600.0,
) -> _WorkerResult:
    """
    Run a single benchmark scenario in a fresh child process.

    Args:
        config: Benchmark configuration.
        operation: Which pack/unpack scenario to measure.
        timeout_s: Maximum seconds to wait for the child process.

    Returns:
        Worker result with timing and peak RSS.

    Raises:
        RuntimeError: If the worker fails or times out.
    """
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue[dict[str, Any]] = ctx.Queue()
    process = ctx.Process(
        target=_worker_entry,
        args=(asdict(config), operation, result_queue),
    )
    process.start()
    process.join(timeout=timeout_s)

    if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)
        raise RuntimeError(f"Benchmark worker timed out after {timeout_s}s for {operation}")

    if process.exitcode != 0:
        raise RuntimeError(f"Benchmark worker exited with code {process.exitcode} for {operation}")

    if result_queue.empty():
        raise RuntimeError(f"Benchmark worker produced no result for {operation}")

    payload = _WorkerResult(**result_queue.get())
    if payload.error is not None:
        raise RuntimeError(f"Benchmark worker failed for {operation}: {payload.error}")

    return payload


def run_benchmark_repeats(
    config: BenchmarkConfig,
    operation: OperationName,
    *,
    warmup: int,
    repeat: int,
) -> list[BenchmarkResult]:
    """
    Run warmup iterations and repeated benchmark measurements.

    Args:
        config: Benchmark configuration.
        operation: Scenario to benchmark.
        warmup: Number of untimed warmup runs.
        repeat: Number of timed runs to record.

    Returns:
        One ``BenchmarkResult`` per timed repeat (median should be computed by caller).
    """
    for _ in range(warmup):
        run_benchmark(config, operation)

    results: list[BenchmarkResult] = []
    for index in range(repeat):
        payload = run_benchmark(config, operation)
        results.append(
            BenchmarkResult(
                operation=operation,
                config_label=config.label(),
                n_leaf_rows=config.n_leaf_rows,
                elapsed_s=payload.elapsed_s,
                peak_rss_mb=payload.peak_rss_mb,
                polars_version=pl.__version__,
                repeat_index=index,
                pack_elapsed_s=payload.pack_elapsed_s,
                unpack_elapsed_s=payload.unpack_elapsed_s,
            )
        )
    return results


def summarize_results(results: list[BenchmarkResult]) -> dict[str, float]:
    """Compute median elapsed time and max peak RSS across repeats."""
    elapsed_values = [result.elapsed_s for result in results]
    return {
        "median_elapsed_s": statistics.median(elapsed_values),
        "min_elapsed_s": min(elapsed_values),
        "max_peak_rss_mb": max(result.peak_rss_mb for result in results),
    }

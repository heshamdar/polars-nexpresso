# Hierarchical packer benchmarks

Standalone performance harness for `HierarchicalPacker` pack/unpack operations. This is **not** part of the pytest correctness suite and is **not** run in CI.

## What it measures

- **Time:** wall-clock seconds for `collect(engine="streaming")` on LazyFrame pack/unpack pipelines
- **Memory:** peak process RSS (MB) sampled via `psutil` in an isolated child process per scenario

Default hierarchy: **image → tile → patch** (3 levels), with image-like pixel payloads at the patch level (`pl.Array` or `pl.List(pl.List(...))`).

Operations: `pack` (whole-dataset `group_by`), `pack_streaming` (root-key
partitioned, memory-bounded), `unpack`, and `roundtrip`. Compare `pack` vs
`pack_streaming` to see the peak-RSS reduction; tune the bucket count with
`--stream-partitions` (more buckets = lower peak memory).

```bash
# Peak-RSS comparison of pack vs the memory-bounded pack_streaming
uv run python -m benchmarks.bench_packer --preset large \
  --operations pack,pack_streaming --stream-partitions 64
```

## Setup

```bash
uv sync --group benchmark
```

## Usage

```bash
# Quick smoke run (80 leaf rows)
uv run python -m benchmarks.bench_packer --preset smoke

# Default medium preset (12.8k leaf rows)
uv run python -m benchmarks.bench_packer

# Custom scale
uv run python -m benchmarks.bench_packer \
  --n-images 100 --tiles-per-image 16 --patches-per-tile 16 \
  --patch-height 64 --patch-width 64 \
  --payload-type array --repeat 3

# JSON output for tracking over time
uv run python -m benchmarks.bench_packer --preset medium --output /tmp/bench.json

# Specific operations only
uv run python -m benchmarks.bench_packer --preset smoke --operations pack,unpack

# Optional correctness invariant check before benchmarking
uv run python -m benchmarks.bench_packer --preset smoke --check-invariants
```

### Multi-version matrix

Compare performance across Polars versions using isolated uv environments (same approach as `tests/test_matrix.py`):

```bash
# Default versions: 1.20.0, 1.30.0, 1.35.1, latest
uv run python -m benchmarks.benchmark_matrix --preset smoke

# Specific versions with forwarded bench_packer flags
uv run python -m benchmarks.benchmark_matrix \
  --versions 1.20.0 1.35.1 latest \
  --preset medium --repeat 3 --operations pack,unpack

# From a minimum version onward
uv run python -m benchmarks.benchmark_matrix --min-version 1.30.0 --preset smoke

# Aggregate JSON for tracking regressions across releases
uv run python -m benchmarks.benchmark_matrix --preset smoke --matrix-output /tmp/matrix.json
```

The matrix runner prints a comparison table (`median_s / peak_rss_mb` per operation per version) and writes per-version details plus an aggregate JSON file.

### Presets

| Preset | Images | Tiles/image | Patches/tile | Patch size | ~Leaf rows |
|--------|--------|-------------|--------------|------------|------------|
| `smoke` | 5 | 4 | 4 | 16×16 | 80 |
| `medium` | 50 | 16 | 16 | 64×64 | 12,800 |
| `large` | 200 | 16 | 16 | 64×64 | 51,200 |

## Interpreting results

- **median_s / min_s:** lower is faster; use `--warmup` and `--repeat` for stable medians on larger presets
- **peak_rss_mb:** peak resident set size during streaming collect in a fresh process; useful for comparing payload types and scale
- Data generation happens inside each benchmark worker and is **not** included in benchmark timings

## Notes

- All benchmarks use streaming collect. Polars >= 1.30 uses `collect(engine="streaming")`; older supported versions use `collect(streaming=True)`. Whether Polars fully streams a given query plan depends on the Polars version and operations involved.
- Invariant checks (row counts after pack/unpack) are opt-in via `--check-invariants`. They materialize a full generated dataset in the parent process, which is useful for smoke checks but expensive for large configurations.
- Large image payloads scale quickly. For example, `200 images * 16 tiles/image * 16 patches/tile * 128 * 128 * 4 bytes` is about 3.2 GiB of raw pixel data before Polars, strings, nested list conversion, or allocator overhead.
- Prefer `--payload-type array` for large runs. `list_of_lists` converts the NumPy buffer through Python lists and can require several times the raw payload memory during construction.

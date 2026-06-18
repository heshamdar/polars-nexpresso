# Hierarchical packer benchmarks

Standalone performance harness for `HierarchicalPacker` pack/unpack operations. This is **not** part of the pytest correctness suite and is **not** run in CI.

## What it measures

- **Time:** wall-clock seconds for `collect(engine="streaming")` on LazyFrame pack/unpack pipelines
- **Memory:** peak process RSS (MB) sampled via `psutil` in an isolated child process per scenario

Default hierarchy: **image → tile → patch** (3 levels), with image-like pixel payloads at the patch level (`pl.Array` or `pl.List(pl.List(...))`).

### Operations

| Operation (alias) | What it measures |
|-------------------|------------------|
| `pack` | Whole-dataset `group_by` pack (baseline). |
| `pack_streaming` | Root-key partitioned, memory-bounded pack (`pack_streaming`, filter-per-bucket). |
| `pack_singlepass` | Experimental partitioned pack that splits the source in one `sink_parquet(PartitionBy)` pass instead of re-filtering per bucket. |
| `pack_no_order` | `pack` with `preserve_child_order=False` — measures the cost of the child-list sort. |
| `pack_split_join` | `pack(..., parent_strategy="split_join")` — reattaches heavy root attributes via a join instead of carrying them through the aggregation. |
| `unpack` | `explode` + `unnest` back to the leaf level. |
| `roundtrip` | `pack` then `unpack`, with per-phase sub-timings. |

When `pack` is included, the summary table adds `t_vs_pack` / `mem_vs_pack`
ratio columns so each variant is read relative to the baseline.

```bash
# Time + peak-RSS comparison of pack vs the memory-bounded variants
uv run python -m benchmarks.bench_packer --preset large \
  --operations pack,pack_streaming,pack_singlepass,pack_no_order --stream-partitions 64

# Split-and-join vs aggregation, with heavy redundant root attributes
uv run python -m benchmarks.bench_packer --preset parent_heavy \
  --operations pack,pack_split_join
```

### Heavy parent attributes

`--parent-payload-pixels N` adds an `image.thumbnail` `Array(Float32, N)` column
and `--parent-attr-count M` adds `M` scalar `image.attr_*` columns. Both are
generated once per image and **broadcast across every leaf row**, so they model
heavy, redundant root-level attributes. This is the regime where
`pack_split_join` can win (see Results). The `parent_heavy` preset enables them.

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

| Preset | Images | Tiles/image | Patches/tile | Patch size | Parent payload | ~Leaf rows |
|--------|--------|-------------|--------------|------------|----------------|------------|
| `smoke` | 5 | 4 | 4 | 16×16 | — | 80 |
| `medium` | 50 | 16 | 16 | 64×64 | — | 12,800 |
| `large` | 200 | 16 | 16 | 64×64 | — | 51,200 |
| `very_large` | 400 | 16 | 16 | 64×64 | — | 102,400 |
| `parent_heavy` | 200 | 16 | 16 | 64×64 | 4,096 px + 8 attrs | 51,200 |

## Interpreting results

- **median_s / min_s:** lower is faster; use `--warmup` and `--repeat` for stable medians on larger presets
- **peak_rss_mb:** peak resident set size during streaming collect in a fresh process; useful for comparing payload types and scale
- **t_vs_pack / mem_vs_pack:** time and peak-memory relative to the `pack` baseline (shown when `pack` is in the run)
- Data generation happens inside each benchmark worker and is **not** included in benchmark timings

## Results

Measured on Polars 1.41.2 (16-core Linux container), `--warmup 1 --repeat 3`.
**Absolute numbers are hardware-specific; read the ratios.** Reproduce with the
commands above.

### Partitioned (`pack_streaming`) vs non-partitioned (`pack`) — `large`, no heavy parent

| operation | median_s | peak_rss_mb | t_vs_pack | mem_vs_pack |
|-----------|---------:|------------:|----------:|------------:|
| `pack` | 1.51 | 2497 | 1.00x | 1.00x |
| `pack_streaming` (64 parts) | 8.76 | 1053 | 5.8x | **0.42x** |
| `pack_singlepass` (64 parts) | 21.3 | 1873 | 14x | 0.75x |
| `pack_no_child_order` | 1.34 | 2495 | **0.89x** | 1.00x |

- **Partitioning is a memory-for-time trade.** `pack_streaming` bounds peak RSS to
  ~0.42x (one bucket + the final streaming scan) but costs ~5.8x time — almost all
  of it the disk round-trip of the heavy payload, not the partitioning itself.
- **More partitions → lower peak memory at ~flat time:**

  | partitions | 8 | 16 | 32 | 64 |
  |-----------|---:|---:|---:|---:|
  | peak_rss_mb | 1533 | 1246 | 1162 | 1053 |
  | median_s | 8.5 | 8.7 | 8.5 | 8.2 |

- **Single-pass `PartitionBy` lost on both axes** (slower *and* more memory): it
  sinks the full heavy payload to disk before re-reading, so it is **not** promoted
  to the library and stays a benchmark-only experiment.
- **Child-list sorting (`preserve_child_order=True`, the default) costs ~11% time**
  and no extra memory here. It is purely cosmetic (deterministic child order); set
  `preserve_child_order=False` if you don't need it.

### Split-and-join vs aggregation for heavy root attributes

| scenario | operation | median_s | peak_rss_mb | t_vs_pack | mem_vs_pack |
|----------|-----------|---------:|------------:|----------:|------------:|
| `parent_heavy` (child payload also heavy) | `pack` | 1.34 | 4156 | 1.00x | 1.00x |
| | `pack_split_join` | 1.57 | 3908 | 1.17x | 0.94x |
| parent-dominant¹ | `pack` | 0.27 | 936 | 1.00x | 1.00x |
| | `pack_split_join` | 0.03 | 470 | **0.11x** | **0.50x** |

¹ 100 images, 8×8 patches, `--parent-payload-pixels 16384 --parent-attr-count 16` (6,400 leaf rows): tiny children, dominant root payload.

- Polars' columnar `group_by(...).first()` already collapses redundant parent
  columns cheaply, so when the **child** payload dominates, `split_join` only adds
  join overhead (~17% slower) for a marginal memory gain.
- When the **root** payload dominates (heavy per-entity blobs/embeddings replicated
  across every leaf row), `split_join` is dramatically better — here **9x faster
  and half the memory** — because the heavy column is touched once per entity
  instead of once per leaf row. It is available as
  `pack(..., parent_strategy="split_join")`.

## Notes

- All benchmarks use streaming collect (`collect(engine="streaming")`). Whether Polars fully streams a given query plan depends on the operations involved.
- Invariant checks (row counts after pack/unpack) are opt-in via `--check-invariants`. They materialize a full generated dataset in the parent process, which is useful for smoke checks but expensive for large configurations.
- Large image payloads scale quickly. For example, `200 images * 16 tiles/image * 16 patches/tile * 128 * 128 * 4 bytes` is about 3.2 GiB of raw pixel data before Polars, strings, nested list conversion, or allocator overhead.
- Prefer `--payload-type array` for large runs. `list_of_lists` converts the NumPy buffer through Python lists and can require several times the raw payload memory during construction.

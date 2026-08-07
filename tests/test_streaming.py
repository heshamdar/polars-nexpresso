"""Tests for streaming / memory-bounded pack & unpack and the order-independence
of packing introduced by the streaming-friendly aggregation."""

from __future__ import annotations

import json

import polars as pl
import pytest

from nexpresso import HierarchicalPacker, HierarchySpec, LevelSpec, hierarchical_packer
from tests.conftest import requires_streaming_pack

SPEC = HierarchySpec.from_levels(
    LevelSpec(name="country", id_fields=["id"]),
    LevelSpec(name="city", id_fields=["id"], parent_keys=["country_id"]),
    LevelSpec(name="street", id_fields=["id"], parent_keys=["city_id"]),
)


def _canonical_rows(df: pl.DataFrame | pl.LazyFrame) -> list[str]:
    """Order-independent row comparison that preserves within-list order."""
    frame = df.collect() if isinstance(df, pl.LazyFrame) else df
    cols = sorted(frame.columns)
    return sorted(json.dumps(row, sort_keys=True) for row in frame.select(cols).to_dicts())


def _same(left, right) -> bool:
    return _canonical_rows(left) == _canonical_rows(right)


@pytest.fixture()
def packer() -> HierarchicalPacker:
    return HierarchicalPacker(SPEC, validate_on_pack=False)


@pytest.fixture()
def flat_df() -> pl.DataFrame:
    rows = []
    counts = [1, 3, 2, 4, 1, 2]
    for ci in range(6):
        for si in range(counts[ci]):
            rows.append(
                {
                    "country.id": f"C{ci % 3}",
                    "country.city.id": f"city{ci}",
                    "country.city.country_id": f"C{ci % 3}",
                    "country.city.street.id": f"s{ci}_{si}",
                    "country.city.street.city_id": f"city{ci}",
                }
            )
    return pl.DataFrame(rows)


# =============================================================================
# Order-independence / null & dedup regression (Change 1)
# =============================================================================


def test_pack_is_order_independent(packer, flat_df):
    """Packing the data and a row-shuffled copy yields identical contents.

    Compared at leaf granularity (via unpack) so the check is independent of both
    top-level row order and child-list order, isolating *contents* equality.
    """
    ref = packer.unpack(packer.pack(flat_df, "country"), "street")
    shuffled_packed = packer.pack(flat_df.sample(fraction=1.0, shuffle=True, seed=11), "country")
    shuffled = packer.unpack(shuffled_packed, "street")
    assert _same(ref, shuffled)


def test_pack_recovers_null_parent_attribute_regardless_of_order():
    """drop_nulls().first() collapses parent attrs and recovers non-null values,
    independent of row order (the global sort never did this)."""
    spec = HierarchySpec.from_levels(
        LevelSpec(name="country", id_fields=["id"]),
        LevelSpec(name="city", id_fields=["id"], parent_keys=["country_id"]),
    )
    packer = HierarchicalPacker(spec, validate_on_pack=False)
    flat = pl.DataFrame(
        {
            "country.id": ["US", "US", "US", "CA"],
            "country.name": ["USA", None, "USA", "Canada"],  # null gap on a US row
            "country.city.id": ["NYC", "LA", "SF", "TOR"],
            "country.city.country_id": ["US", "US", "US", "CA"],
        }
    )

    def names(df):
        inner = df.unnest("country").sort("id")
        return dict(zip(inner["id"], inner["name"]))

    packed = packer.pack(flat, "country")
    shuffled = packer.pack(flat.sample(fraction=1.0, shuffle=True, seed=3), "country")
    assert names(packed) == {"US": "USA", "CA": "Canada"}
    assert names(packed) == names(shuffled)


def test_preserve_child_order_keeps_child_list_order(packer, flat_df):
    """With preserve_child_order=True (default), child lists follow original order."""
    packed = packer.pack(flat_df, "country")
    by_city = {}
    for country in packed["country"].to_list():
        for city in country["city"]:
            by_city[city["id"]] = [s["id"] for s in city["street"]]
    # city1 has 3 streets inserted as s1_0, s1_1, s1_2 in original order
    assert by_city["city1"] == ["s1_0", "s1_1", "s1_2"]
    assert by_city["city3"] == ["s3_0", "s3_1", "s3_2", "s3_3"]


def test_order_by_sorts_child_list_inside_agg():
    """order_by expressions still control child-list order after the rewrite."""
    spec = HierarchySpec.from_levels(
        LevelSpec(name="country", id_fields=["id"]),
        LevelSpec(
            name="city",
            id_fields=["id"],
            parent_keys=["country_id"],
            order_by=[pl.col("country.city.id")],  # ascending by city id
        ),
    )
    packer = HierarchicalPacker(spec, validate_on_pack=False)
    flat = pl.DataFrame(
        {
            "country.id": ["US", "US", "US"],
            "country.city.id": ["c", "a", "b"],
            "country.city.country_id": ["US", "US", "US"],
        }
    )
    packed = packer.pack(flat, "country")
    cities = [c["id"] for c in packed["country"][0]["city"]]
    assert cities == ["a", "b", "c"]


# =============================================================================
# pack_streaming (Change 2)
# =============================================================================


@requires_streaming_pack
@pytest.mark.parametrize("strategy", ["balanced", "hash"])
@pytest.mark.parametrize("partitions", [1, 4, 64])
def test_pack_streaming_matches_pack(packer, flat_df, partitions, strategy):
    ref = packer.pack(flat_df, "country")
    out = packer.pack_streaming(
        flat_df, "country", partitions=partitions, partition_strategy=strategy
    )
    assert isinstance(out, pl.LazyFrame)
    assert _same(out, ref)


@requires_streaming_pack
def test_pack_streaming_eager_sink_returns_scan(packer, flat_df):
    ref = packer.pack(flat_df, "country")
    out = packer.pack_streaming(flat_df, "country", partitions=4, defer=False)
    assert isinstance(out, pl.LazyFrame)
    assert _same(out, ref)


@requires_streaming_pack
def test_pack_streaming_accepts_lazyframe(packer, flat_df):
    ref = packer.pack(flat_df, "country")
    out = packer.pack_streaming(flat_df.lazy(), "country", partitions=4)
    assert _same(out, ref)


@requires_streaming_pack
def test_pack_streaming_accepts_parquet_path(packer, flat_df, tmp_path):
    src = tmp_path / "src.parquet"
    flat_df.write_parquet(src)
    ref = packer.pack(flat_df, "country")
    out = packer.pack_streaming(str(src), "country", partitions=4, tmp_dir=tmp_path / "parts")
    assert _same(out, ref)


@requires_streaming_pack
def test_pack_streaming_chains_lazily(packer, flat_df):
    ref = packer.pack(flat_df, "country")
    n = (
        packer.pack_streaming(flat_df, "country", partitions=4)
        .select(pl.len().alias("n"))
        .collect()["n"][0]
    )
    assert n == ref.height


@requires_streaming_pack
def test_pack_streaming_to_intermediate_level(packer, flat_df):
    ref = packer.pack(flat_df, "city")
    out = packer.pack_streaming(flat_df, "city", partitions=4)
    assert _same(out, ref)


def test_pack_streaming_rejects_bad_partitions(packer, flat_df):
    with pytest.raises(ValueError, match="partitions must be >= 1"):
        packer.pack_streaming(flat_df, "country", partitions=0)


def test_pack_streaming_rejects_unknown_strategy(packer, flat_df):
    with pytest.raises(ValueError, match="Invalid partition_strategy"):
        packer.pack_streaming(flat_df, "country", partition_strategy="round-robin")


# =============================================================================
# Balanced vs hash partitioning
# =============================================================================


@pytest.fixture()
def skewed_df() -> pl.DataFrame:
    """A few large entities among many small ones — where hashing balances badly."""
    rows = []
    for entity in range(120):
        n_children = 400 if entity < 4 else (entity % 7) + 1
        rows.extend(
            {
                "country.id": f"C{entity:04d}",
                "country.city.id": f"c{entity}_{i}",
                "country.city.country_id": f"C{entity:04d}",
            }
            for i in range(n_children)
        )
    return pl.DataFrame(rows)


@requires_streaming_pack
def test_balanced_output_is_sorted_by_root_key(packer, skewed_df):
    """Contiguous ascending key ranges make the concatenated result sorted."""
    out = packer.pack_streaming(
        skewed_df, "country", partitions=8, partition_strategy="balanced"
    ).collect()

    ids = out["country"].struct.field("id").to_list()
    assert ids == sorted(ids)
    assert _same(out, packer.pack(skewed_df, "country"))


@requires_streaming_pack
def test_balanced_lowers_peak_bucket_size(packer, skewed_df):
    """Balancing rows beats hashing entities when entity sizes are uneven.

    Peak memory is bounded by the largest bucket, and no scheme can go below the
    largest single entity, which cannot be split.
    """
    lf = skewed_df.lazy()
    keys = ["country.id"]
    floor = lf.group_by(keys).agg(pl.len().alias("n")).collect()["n"].max()

    bucket_map = packer._balanced_bucket_map(lf, keys, 8)
    balanced = (
        lf.join(bucket_map.lazy(), on=keys, how="left")
        .group_by(hierarchical_packer.BUCKET_COLUMN)
        .agg(pl.len())
        .collect()["len"]
        .to_list()
    )
    hashed = (
        lf.with_columns((pl.struct(keys).hash() % 8).alias("b"))
        .group_by("b")
        .agg(pl.len())
        .collect()["len"]
        .to_list()
    )

    assert max(balanced) < max(hashed)
    assert max(balanced) >= floor  # cannot beat the floor
    # Every entity lands in exactly one bucket.
    assert bucket_map.height == skewed_df["country.id"].n_unique()


@requires_streaming_pack
def test_balanced_bucket_count_floats_above_target(packer, skewed_df):
    """An entity is never split, so a bucket closes early rather than overflow."""
    lf = skewed_df.lazy()
    bucket_map = packer._balanced_bucket_map(lf, ["country.id"], 4)
    n_buckets = int(bucket_map[hierarchical_packer.BUCKET_COLUMN].max()) + 1

    # Four 400-child entities alone exceed a quarter of the rows each, so the
    # greedy pass must open more buckets than requested.
    assert n_buckets > 4
    # ...and it must not dump the remainder into one giant final bucket.
    sizes = bucket_map.group_by(hierarchical_packer.BUCKET_COLUMN).agg(pl.len())["len"].to_list()
    assert max(sizes) < bucket_map.height


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda p, lf: p.pack(lf, "country"), id="pack"),
        pytest.param(lambda p, lf: p.unpack(p.pack(lf, "country"), "street"), id="unpack"),
        pytest.param(lambda p, lf: p.normalize(lf), id="normalize"),
        pytest.param(lambda p, lf: p.split_levels(p.pack(lf, "country")), id="split_levels"),
        pytest.param(lambda p, lf: p.denormalize(p.normalize(lf)), id="denormalize"),
        pytest.param(
            lambda p, lf: p.promote_attribute(
                lf, "id", from_level="street", to_level="city", agg="count"
            ),
            id="promote_attribute",
        ),
    ],
)
def test_lazy_operations_do_not_execute(packer, flat_df, monkeypatch, operation):
    """Lazy input must produce a lazy plan without any hidden execution.

    ``collect_schema()`` is fine (metadata only); actually executing the plan
    would materialize data behind the caller's back and break streaming
    pipelines. Both entry points are watched: ``LazyFrame.collect`` and the
    module-level ``pl.collect_all`` (which ``split_levels`` uses for eager input
    and which a method-level spy would not catch).
    """
    executed: list[str] = []

    original_collect = pl.LazyFrame.collect
    original_collect_all = pl.collect_all

    def spy_collect(self, *args, **kwargs):
        executed.append("LazyFrame.collect")
        return original_collect(self, *args, **kwargs)

    def spy_collect_all(*args, **kwargs):
        executed.append("collect_all")
        return original_collect_all(*args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "collect", spy_collect)
    monkeypatch.setattr(pl, "collect_all", spy_collect_all)
    operation(packer, flat_df.lazy())

    assert not executed, f"operation executed the query instead of staying lazy: {executed}"


@requires_streaming_pack
@pytest.mark.parametrize("strategy", ["balanced", "hash"])
def test_pack_streaming_bucketing_paths_agree(packer, flat_df, tmp_path, monkeypatch, strategy):
    """The single-pass partitioned sink and the per-bucket filter fallback agree.

    Polars versions before partitioned sinks (1.30) take the fallback, so both
    paths have to carry every strategy.
    """
    ref = packer.pack(flat_df, "country")

    fast_dir = tmp_path / "fast"
    fast = packer.pack_streaming(
        flat_df,
        "country",
        partitions=4,
        tmp_dir=fast_dir,
        defer=False,
        partition_strategy=strategy,
    )

    monkeypatch.setattr(hierarchical_packer, "_supports_partitioned_sink", lambda: False)
    slow_dir = tmp_path / "slow"
    slow = packer.pack_streaming(
        flat_df,
        "country",
        partitions=4,
        tmp_dir=slow_dir,
        defer=False,
        partition_strategy=strategy,
    )

    assert _same(fast, ref)
    assert _same(slow, ref)
    # The staging area used by the partitioned sink is cleaned up.
    assert not (fast_dir / "_stage").exists()


@requires_streaming_pack
@pytest.mark.parametrize("strategy", ["balanced", "hash"])
def test_pack_streaming_keeps_entities_whole(packer, flat_df, strategy):
    """Every root entity ends up in exactly one output row, never split across buckets."""
    out = packer.pack_streaming(
        flat_df, "country", partitions=32, partition_strategy=strategy
    ).collect()
    codes = out["country"].struct.field("id").to_list()
    assert sorted(codes) == sorted(set(codes))
    assert len(codes) == packer.pack(flat_df, "country").height


# =============================================================================
# unpack_streaming (Change 2)
# =============================================================================


def test_unpack_streaming_matches_unpack(packer, flat_df):
    packed = packer.pack(flat_df, "country")
    ref = packer.unpack(packed, "street")
    out = packer.unpack_streaming(packed, "street")
    assert isinstance(out, pl.LazyFrame)
    assert _same(out, ref)


def test_unpack_streaming_parquet_source_and_sink(packer, flat_df, tmp_path):
    packed_path = tmp_path / "packed.parquet"
    packer.pack(flat_df, "country").write_parquet(packed_path)
    ref = packer.unpack(packer.pack(flat_df, "country"), "street")

    sink = tmp_path / "unpacked.parquet"
    out = packer.unpack_streaming(str(packed_path), "street", sink_path=sink)
    assert sink.exists()
    assert _same(out, ref)

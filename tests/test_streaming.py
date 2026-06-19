"""Tests for streaming / memory-bounded pack & unpack and the order-independence
of packing introduced by the streaming-friendly aggregation."""

from __future__ import annotations

import json

import polars as pl
import pytest

from nexpresso import HierarchicalPacker, HierarchySpec, LevelSpec
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
@pytest.mark.parametrize("partitions", [1, 4, 64])
def test_pack_streaming_matches_pack(packer, flat_df, partitions):
    ref = packer.pack(flat_df, "country")
    out = packer.pack_streaming(flat_df, "country", partitions=partitions)
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

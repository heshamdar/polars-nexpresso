"""Tests for streaming / memory-bounded pack & unpack and the order-independence
of packing introduced by the streaming-friendly aggregation."""

from __future__ import annotations

import json
import pathlib

import polars as pl
import pytest

from nexpresso import HierarchicalPacker, HierarchySpec, LevelSpec, hierarchical_packer

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


def _schema_of(frame: pl.DataFrame | pl.LazyFrame) -> list[tuple[str, pl.DataType]]:
    """Column names, order and dtypes — what ``_same`` cannot see."""
    schema = frame.collect_schema() if isinstance(frame, pl.LazyFrame) else frame.schema
    return list(schema.items())


def _assert_matches_pack(actual, expected) -> None:
    """Strict equivalence: same rows *and* same schema (names, order, dtypes).

    ``_same`` JSON-compares values, so on its own it would pass a result whose
    dtypes had silently drifted.
    """
    assert _schema_of(actual) == _schema_of(
        expected
    ), f"schema mismatch\n  actual   {_schema_of(actual)}\n  expected {_schema_of(expected)}"
    assert _same(actual, expected)


def _children_ignoring_order(frame, root: str = "country", child: str = "city") -> list:
    """Root keys with their child ids as sets.

    For ``preserve_child_order=False`` the child-list order is legitimately
    nondeterministic, so ``_same`` (which preserves list order) is too strict.
    """
    df = frame.collect() if isinstance(frame, pl.LazyFrame) else frame
    return sorted(
        (row[root]["id"], sorted(c["id"] for c in (row[root][child] or [])))
        for row in df.to_dicts()
    )


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


@pytest.mark.parametrize("strategy", ["balanced", "hash"])
@pytest.mark.parametrize("partitions", [1, 4, 64])
def test_pack_streaming_matches_pack(packer, flat_df, partitions, strategy):
    ref = packer.pack(flat_df, "country")
    out = packer.pack_streaming(
        flat_df, "country", partitions=partitions, partition_strategy=strategy
    )
    assert isinstance(out, pl.LazyFrame)
    assert _same(out, ref)


def test_pack_streaming_eager_sink_returns_scan(packer, flat_df):
    ref = packer.pack(flat_df, "country")
    out = packer.pack_streaming(flat_df, "country", partitions=4, defer=False)
    assert isinstance(out, pl.LazyFrame)
    assert _same(out, ref)


def test_pack_streaming_accepts_lazyframe(packer, flat_df):
    ref = packer.pack(flat_df, "country")
    out = packer.pack_streaming(flat_df.lazy(), "country", partitions=4)
    assert _same(out, ref)


def test_pack_streaming_accepts_parquet_path(packer, flat_df, tmp_path):
    src = tmp_path / "src.parquet"
    flat_df.write_parquet(src)
    ref = packer.pack(flat_df, "country")
    out = packer.pack_streaming(str(src), "country", partitions=4, tmp_dir=tmp_path / "parts")
    assert _same(out, ref)


def test_pack_streaming_chains_lazily(packer, flat_df):
    ref = packer.pack(flat_df, "country")
    n = (
        packer.pack_streaming(flat_df, "country", partitions=4)
        .select(pl.len().alias("n"))
        .collect()["n"][0]
    )
    assert n == ref.height


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


def test_balanced_output_is_sorted_by_root_key(packer, skewed_df):
    """Contiguous ascending key ranges make the concatenated result sorted."""
    out = packer.pack_streaming(
        skewed_df, "country", partitions=8, partition_strategy="balanced"
    ).collect()

    ids = out["country"].struct.field("id").to_list()
    assert ids == sorted(ids)
    assert _same(out, packer.pack(skewed_df, "country"))


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


def test_balanced_sorted_with_many_buckets(packer, skewed_df, tmp_path):
    """Sortedness must survive >= 10 buckets.

    ``PartitionBy`` names directories ``<key>=<value>``; sorting those as strings
    puts ``=10`` before ``=2``. With fewer than ten buckets that bug is invisible.
    """
    out = packer.pack_streaming(
        skewed_df,
        "country",
        partitions=64,
        tmp_dir=tmp_path,
        defer=False,
        partition_strategy="balanced",
    ).collect()

    n_parts = len(list(tmp_path.glob("part_*.parquet")))
    assert n_parts >= 10, f"need >= 10 buckets to exercise the ordering, got {n_parts}"

    ids = out["country"].struct.field("id").to_list()
    assert ids == sorted(ids)
    assert _same(out, packer.pack(skewed_df, "country"))


def test_balanced_degenerates_to_one_bucket_per_entity(packer, skewed_df):
    """Raising partitions past the entity count is the partition-by-key limit."""
    n_entities = skewed_df["country.id"].n_unique()
    bucket_map = packer._balanced_bucket_map(skewed_df.lazy(), ["country.id"], 1_000_000)

    n_buckets = bucket_map.select(pl.col(hierarchical_packer.BUCKET_COLUMN).n_unique()).item()
    assert n_buckets == n_entities


def test_balanced_and_hash_agree_on_skewed_data(packer, skewed_df, tmp_path):
    """The two strategies differ in bucketing, never in contents."""
    results = {
        strategy: packer.pack_streaming(
            skewed_df,
            "country",
            partitions=8,
            tmp_dir=tmp_path / strategy,
            defer=False,
            partition_strategy=strategy,
        ).collect()
        for strategy in ("balanced", "hash")
    }

    _assert_matches_pack(results["balanced"], packer.pack(skewed_df, "country"))
    _assert_matches_pack(results["hash"], packer.pack(skewed_df, "country"))


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


@pytest.mark.parametrize("strategy", ["balanced", "hash"])
def test_pack_streaming_bucketing_matches_pack(packer, flat_df, tmp_path, strategy):
    """Every bucketing strategy reproduces a plain pack()."""
    ref = packer.pack(flat_df, "country")

    out_dir = tmp_path / "out"
    result = packer.pack_streaming(
        flat_df,
        "country",
        partitions=4,
        tmp_dir=out_dir,
        defer=False,
        partition_strategy=strategy,
    )

    assert _same(result, ref)
    # The staging area used by the partitioned sink is cleaned up.
    assert not (out_dir / "_stage").exists()


def test_pack_streaming_cleans_staging_on_failure(packer, flat_df, tmp_path, monkeypatch):
    """The staging area is removed even when a bucket pack blows up."""
    boom = RuntimeError("bucket pack failed")

    original_pack = HierarchicalPacker.pack
    calls = {"n": 0}

    def exploding_pack(self, frame, to_level, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:  # let the up-front schema probe through
            raise boom
        return original_pack(self, frame, to_level, **kwargs)

    monkeypatch.setattr(HierarchicalPacker, "pack", exploding_pack)

    with pytest.raises(RuntimeError, match="bucket pack failed"):
        packer.pack_streaming(flat_df, "country", partitions=4, tmp_dir=tmp_path, defer=False)

    assert not (tmp_path / "_stage").exists()


@pytest.mark.parametrize("strategy", ["balanced", "hash"])
def test_pack_streaming_defer_does_not_execute_at_call_time(packer, flat_df, tmp_path, strategy):
    """``defer=True`` must not touch the data until the caller collects.

    Schema resolution (``collect_schema``) legitimately runs at call time; what
    must not happen is any partitioning, packing or file writing.
    """
    out = packer.pack_streaming(
        flat_df,
        "country",
        partitions=4,
        tmp_dir=tmp_path,
        defer=True,
        partition_strategy=strategy,
    )

    assert isinstance(out, pl.LazyFrame)
    assert not list(tmp_path.glob("part_*.parquet")), "work ran before collect()"
    assert not (tmp_path / "_stage").exists()

    collected = out.collect()

    assert list(tmp_path.glob("part_*.parquet")), "collect() did not trigger the work"
    _assert_matches_pack(collected, packer.pack(flat_df, "country"))


def test_pack_streaming_eager_returns_a_real_scan(packer, flat_df, tmp_path):
    """``defer=False`` hands back a Parquet scan, so downstream work streams.

    This is the difference that matters when the packed result also does not fit
    in memory: ``defer=True`` collapses to an opaque PYTHON SCAN that
    materializes, while ``defer=False`` supports predicate pushdown.
    """
    out = packer.pack_streaming(flat_df, "country", partitions=4, tmp_dir=tmp_path, defer=False)

    plan = out.explain(optimized=False)
    assert "PYTHON SCAN" not in plan
    assert "SCAN" in plan

    pushed = out.filter(pl.col("country").struct.field("id") == "C0").explain(optimized=True)
    assert "SELECTION" in pushed or "FILTER" in pushed


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
# Edge cases — regression guards for crashes the original suite missed
# =============================================================================

TWO_LEVEL = HierarchySpec(
    levels=[LevelSpec(name="country", id_fields=["id"]), LevelSpec(name="city", id_fields=["id"])]
)


@pytest.fixture()
def two_level_packer() -> HierarchicalPacker:
    return HierarchicalPacker(TWO_LEVEL, validate_on_pack=False)


def _two_level_df(pairs) -> pl.DataFrame:
    return pl.DataFrame(
        {"country.id": [k for k, _ in pairs], "country.city.id": [v for _, v in pairs]},
        schema={"country.id": pl.String, "country.city.id": pl.String},
    )


@pytest.mark.parametrize("strategy", ["balanced", "hash"])
@pytest.mark.parametrize("partitions", [1, 4])
def test_pack_streaming_empty_input(two_level_packer, tmp_path, strategy, partitions):
    """An empty frame packs to an empty result, not a crash.

    ``PartitionBy`` writes no partitions at all for an empty frame, so the
    staging directory never appears. This used to raise FileNotFoundError for
    partitions > 1 while eager ``pack`` handled it fine.
    """
    empty = _two_level_df([])

    out = two_level_packer.pack_streaming(
        empty,
        "country",
        partitions=partitions,
        tmp_dir=tmp_path / f"{strategy}{partitions}",
        defer=False,
        partition_strategy=strategy,
    ).collect()

    assert out.height == 0
    _assert_matches_pack(out, two_level_packer.pack(empty, "country"))


@pytest.mark.parametrize("strategy", ["balanced", "hash"])
@pytest.mark.parametrize(
    ("pairs", "label"),
    [
        ([("A", "a"), (None, "b"), ("B", "c")], "some-null"),
        ([(None, "a"), (None, "b")], "all-null"),
    ],
)
def test_pack_streaming_null_root_keys(two_level_packer, tmp_path, strategy, pairs, label):
    """Null root keys form their own group, as they do in eager ``pack``.

    ``group_by`` treats null as a group, but a plain join does not match null to
    null — those rows used to come out with a null bucket, which PartitionBy
    writes as ``__HIVE_DEFAULT_PARTITION__`` and the bucket-ordering helper then
    failed to parse as an integer.
    """
    df = _two_level_df(pairs)

    out = two_level_packer.pack_streaming(
        df,
        "country",
        partitions=4,
        tmp_dir=tmp_path / f"{strategy}-{label}",
        defer=False,
        partition_strategy=strategy,
    ).collect()

    _assert_matches_pack(out, two_level_packer.pack(df, "country"))


@pytest.mark.parametrize("strategy", ["balanced", "hash"])
@pytest.mark.parametrize(
    ("pairs", "label"),
    [
        ([("A", "a")], "single-row"),
        ([("A", "a"), ("A", "b"), ("A", "c")], "single-entity"),
    ],
)
def test_pack_streaming_degenerate_inputs(two_level_packer, tmp_path, strategy, pairs, label):
    df = _two_level_df(pairs)

    out = two_level_packer.pack_streaming(
        df,
        "country",
        partitions=8,  # far more partitions than entities
        tmp_dir=tmp_path / f"{strategy}-{label}",
        defer=False,
        partition_strategy=strategy,
    ).collect()

    _assert_matches_pack(out, two_level_packer.pack(df, "country"))


# =============================================================================
# Feature interactions
# =============================================================================


@pytest.mark.parametrize("strategy", ["balanced", "hash"])
def test_pack_streaming_multi_column_root_key(tmp_path, strategy):
    spec = HierarchySpec(
        levels=[
            LevelSpec(name="country", id_fields=["code", "region"]),
            LevelSpec(name="city", id_fields=["id"]),
        ]
    )
    p = HierarchicalPacker(spec, validate_on_pack=False)
    df = pl.DataFrame(
        {
            "country.code": ["US", "US", "CA"],
            "country.region": ["N", "N", "S"],
            "country.city.id": ["a", "b", "c"],
        }
    )

    out = p.pack_streaming(
        df,
        "country",
        partitions=4,
        tmp_dir=tmp_path / strategy,
        defer=False,
        partition_strategy=strategy,
    ).collect()

    _assert_matches_pack(out, p.pack(df, "country"))


@pytest.mark.parametrize("strategy", ["balanced", "hash"])
def test_pack_streaming_non_string_root_key(two_level_packer, tmp_path, strategy):
    """Bucketing must not assume the key is a string."""
    df = pl.DataFrame({"country.id": [1, 1, 2], "country.city.id": ["a", "b", "c"]})

    out = two_level_packer.pack_streaming(
        df,
        "country",
        partitions=4,
        tmp_dir=tmp_path / strategy,
        defer=False,
        partition_strategy=strategy,
    ).collect()

    _assert_matches_pack(out, two_level_packer.pack(df, "country"))


def test_pack_streaming_honours_order_by(tmp_path):
    """A level's order_by still controls child-list order through the streaming path."""
    spec = HierarchySpec(
        levels=[
            LevelSpec(name="country", id_fields=["id"]),
            LevelSpec(name="city", id_fields=["id"], order_by=[pl.col("country.city.id")]),
        ]
    )
    p = HierarchicalPacker(spec, validate_on_pack=False)
    df = pl.DataFrame({"country.id": ["A"] * 3, "country.city.id": ["c", "a", "b"]})

    out = p.pack_streaming(df, "country", partitions=2, tmp_dir=tmp_path, defer=False).collect()

    assert [c["id"] for c in out["country"][0]["city"]] == ["a", "b", "c"]


@pytest.mark.parametrize("strategy", ["balanced", "hash"])
def test_pack_streaming_without_preserve_child_order(tmp_path, strategy):
    """Contents still match; only child-list order is unspecified."""
    p = HierarchicalPacker(TWO_LEVEL, validate_on_pack=False, preserve_child_order=False)
    df = _two_level_df([("A", "a"), ("A", "b"), ("A", "c"), ("B", "d")])

    out = p.pack_streaming(
        df,
        "country",
        partitions=4,
        tmp_dir=tmp_path / strategy,
        defer=False,
        partition_strategy=strategy,
    ).collect()

    assert _schema_of(out) == _schema_of(p.pack(df, "country"))
    assert _children_ignoring_order(out) == _children_ignoring_order(p.pack(df, "country"))


@pytest.mark.parametrize("extra_columns", ["preserve", "drop"])
def test_pack_streaming_extra_columns(two_level_packer, tmp_path, extra_columns):
    """Non-hierarchy columns follow the same rules as eager ``pack``."""
    df = _two_level_df([("A", "a"), ("A", "b"), ("B", "c")]).with_columns(
        pl.lit("x").alias("unrelated")
    )

    out = two_level_packer.pack_streaming(
        df,
        "country",
        partitions=4,
        tmp_dir=tmp_path / extra_columns,
        defer=False,
        extra_columns=extra_columns,
    ).collect()

    _assert_matches_pack(out, two_level_packer.pack(df, "country", extra_columns=extra_columns))


def test_pack_streaming_with_validation_enabled(tmp_path):
    """validate_on_pack=True must not break the per-bucket packs."""
    p = HierarchicalPacker(TWO_LEVEL, validate_on_pack=True)
    df = _two_level_df([("A", "a"), ("A", "b"), ("B", "c")])

    out = p.pack_streaming(df, "country", partitions=2, tmp_dir=tmp_path, defer=False).collect()

    _assert_matches_pack(out, p.pack(df, "country"))


@pytest.mark.parametrize("source_kind", ["dataframe", "lazyframe", "str_path", "path", "glob"])
@pytest.mark.parametrize("strategy", ["balanced", "hash"])
def test_pack_streaming_source_forms(packer, flat_df, tmp_path, source_kind, strategy):
    if source_kind == "dataframe":
        source = flat_df
    elif source_kind == "lazyframe":
        source = flat_df.lazy()
    else:
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        flat_df.write_parquet(src_dir / "data.parquet")
        source = {
            "str_path": str(src_dir / "data.parquet"),
            "path": src_dir / "data.parquet",
            "glob": str(src_dir / "*.parquet"),
        }[source_kind]

    out = packer.pack_streaming(
        source,
        "country",
        partitions=4,
        tmp_dir=tmp_path / f"{source_kind}-{strategy}",
        defer=False,
        partition_strategy=strategy,
    )

    _assert_matches_pack(out.collect(), packer.pack(flat_df, "country"))


def test_pack_streaming_tmp_dir_accepts_str(packer, flat_df, tmp_path):
    out = packer.pack_streaming(
        flat_df, "country", partitions=4, tmp_dir=str(tmp_path / "as_str"), defer=False
    )
    assert _same(out, packer.pack(flat_df, "country"))
    assert list((tmp_path / "as_str").glob("part_*.parquet"))


# =============================================================================
# Error paths
# =============================================================================


def test_pack_streaming_requires_root_id_fields(flat_df):
    spec = HierarchySpec(
        levels=[LevelSpec(name="country"), LevelSpec(name="city", id_fields=["id"])]
    )
    p = HierarchicalPacker(spec, validate_on_pack=False)

    with pytest.raises(hierarchical_packer.HierarchyValidationError, match="id_fields"):
        p.pack_streaming(flat_df, "country", partitions=2)


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


@pytest.mark.parametrize("source_kind", ["dataframe", "lazyframe", "str_path", "path", "glob"])
def test_unpack_streaming_source_forms(packer, flat_df, tmp_path, source_kind):
    packed = packer.pack(flat_df, "country")
    ref = packer.unpack(packed, "street")

    if source_kind == "dataframe":
        source = packed
    elif source_kind == "lazyframe":
        source = packed.lazy()
    else:
        packed.write_parquet(tmp_path / "packed.parquet")
        source = {
            "str_path": str(tmp_path / "packed.parquet"),
            "path": tmp_path / "packed.parquet",
            "glob": str(tmp_path / "*.parquet"),
        }[source_kind]

    _assert_matches_pack(packer.unpack_streaming(source, "street"), ref)


@pytest.mark.parametrize("to_level", ["city", "street"])
def test_unpack_streaming_to_each_level(packer, flat_df, to_level):
    packed = packer.pack(flat_df, "country")
    _assert_matches_pack(packer.unpack_streaming(packed, to_level), packer.unpack(packed, to_level))


def test_unpack_streaming_sink_path_as_str(packer, flat_df, tmp_path):
    packed = packer.pack(flat_df, "country")
    sink = str(tmp_path / "out.parquet")

    out = packer.unpack_streaming(packed, "street", sink_path=sink)

    assert pathlib.Path(sink).exists()
    _assert_matches_pack(out, packer.unpack(packed, "street"))


def test_unpack_streaming_stays_lazy_without_sink(packer, flat_df, monkeypatch):
    """Without sink_path it must be a plan, not an execution."""
    packed = packer.pack(flat_df, "country").lazy()
    executed: list[str] = []
    original = pl.LazyFrame.collect
    monkeypatch.setattr(
        pl.LazyFrame,
        "collect",
        lambda self, *a, **k: (executed.append("collect"), original(self, *a, **k))[1],
    )

    packer.unpack_streaming(packed, "street")

    assert not executed


@pytest.mark.parametrize("strategy", ["balanced", "hash"])
def test_pack_streaming_unpack_streaming_round_trip(packer, flat_df, tmp_path, strategy):
    """pack_streaming -> unpack_streaming returns the original rows."""
    packed = packer.pack_streaming(
        flat_df,
        "country",
        partitions=4,
        tmp_dir=tmp_path / "parts",
        defer=False,
        partition_strategy=strategy,
    )

    out = packer.unpack_streaming(packed, "street")

    assert _same(out, flat_df)

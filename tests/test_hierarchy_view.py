"""Tests for HierarchyView — the deferred nested view over normalized tables."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from nexpresso import HierarchicalPacker, HierarchySpec, HierarchyView, LevelSpec

N_REGION, N_STORE, N_SALE = 4, 5, 6
ROWS = N_REGION * N_STORE * N_SALE
SALE_ID = "region.store.sale.id"
AMOUNT = "region.store.sale.amount"


@pytest.fixture
def spec() -> HierarchySpec:
    return HierarchySpec.from_levels(
        LevelSpec(name="region", id_fields=["id"]),
        LevelSpec(name="store", id_fields=["id"], parent_keys=["region_id"]),
        LevelSpec(name="sale", id_fields=["id"], parent_keys=["store_id"]),
    )


@pytest.fixture
def packer(spec: HierarchySpec) -> HierarchicalPacker:
    return HierarchicalPacker(spec)


@pytest.fixture
def flat() -> pl.DataFrame:
    """Three-level flat frame: 4 regions x 5 stores x 6 sales."""
    return pl.DataFrame(
        {
            "region.id": [i // (N_STORE * N_SALE) for i in range(ROWS)],
            "region.name": [f"r{i // (N_STORE * N_SALE)}" for i in range(ROWS)],
            "region.store.id": [i // N_SALE for i in range(ROWS)],
            "region.store.name": [f"s{i // N_SALE}" for i in range(ROWS)],
            SALE_ID: list(range(ROWS)),
            AMOUNT: [float(i % 17) for i in range(ROWS)],
        }
    )


@pytest.fixture
def view(flat: pl.DataFrame, packer: HierarchicalPacker) -> HierarchyView:
    return HierarchyView.from_frame(flat, packer)


def assert_same_rows(got: pl.DataFrame, want: pl.DataFrame, sort_by: str) -> None:
    """Compare frames ignoring column order and row order."""
    assert_frame_equal(
        got.sort(sort_by).select(sorted(got.columns)),
        want.sort(sort_by).select(sorted(want.columns)),
        check_dtypes=False,
    )


class TestConstruction:
    """Building views from frames, tables, and Parquet."""

    def test_from_frame_exposes_all_levels(self, view: HierarchyView):
        assert view.levels == ["region", "store", "sale"]

    def test_from_tables_matches_from_frame(
        self, flat: pl.DataFrame, packer: HierarchicalPacker, view: HierarchyView
    ):
        direct = HierarchyView.from_tables(packer.normalize(flat), packer)
        assert_same_rows(direct.level("sale").collect(), view.level("sale").collect(), SALE_ID)

    def test_rejects_unknown_level(self, flat: pl.DataFrame, packer: HierarchicalPacker):
        tables = packer.normalize(flat)
        tables["galaxy"] = tables["region"]
        with pytest.raises(ValueError, match="Unknown level"):
            HierarchyView.from_tables(tables, packer)

    def test_rejects_empty_tables(self, packer: HierarchicalPacker):
        with pytest.raises(ValueError, match="must not be empty"):
            HierarchyView.from_tables({}, packer)

    def test_rejects_bad_empty_parents_mode(self, flat: pl.DataFrame, packer: HierarchicalPacker):
        with pytest.raises(ValueError, match="Invalid empty_parents"):
            HierarchyView.from_frame(flat, packer, empty_parents="sometimes")  # type: ignore[arg-type]

    def test_parquet_round_trip(self, view: HierarchyView, packer: HierarchicalPacker, tmp_path):
        view.sink_parquet(tmp_path)
        assert {p.name for p in tmp_path.glob("*.parquet")} == {
            "region.parquet",
            "store.parquet",
            "sale.parquet",
        }
        rescanned = HierarchyView.scan_parquet(tmp_path, packer)
        assert_same_rows(rescanned.level("sale").collect(), view.level("sale").collect(), SALE_ID)

    def test_scan_parquet_reports_missing_levels(
        self, view: HierarchyView, packer: HierarchicalPacker, tmp_path
    ):
        view.sink_parquet(tmp_path)
        (tmp_path / "sale.parquet").unlink()
        with pytest.raises(FileNotFoundError, match="sale"):
            HierarchyView.scan_parquet(tmp_path, packer)

    def test_scan_parquet_reports_empty_directory(self, packer: HierarchicalPacker, tmp_path):
        with pytest.raises(FileNotFoundError, match="No per-level Parquet datasets"):
            HierarchyView.scan_parquet(tmp_path, packer)


class TestIntrospection:
    """What the view can tell you about itself without moving data."""

    def test_schema_comes_from_the_frame_you_ask_for(
        self, view: HierarchyView, packer: HierarchicalPacker, flat: pl.DataFrame
    ):
        """Each granularity has its own schema, so each is asked for by name."""
        assert view.nested().collect_schema() == packer.pack(flat, "region").schema
        assert (
            view.level("sale").collect_schema()
            == packer.unpack(packer.pack(flat, "region"), "sale").schema
        )

    def test_key_columns_are_ancestor_keys_then_own_ids(self, view: HierarchyView):
        assert view.key_columns("region") == ["region.id"]
        assert view.key_columns("sale") == ["region.id", "region.store.id", SALE_ID]

    def test_columns_are_dotted_paths(self, view: HierarchyView):
        assert AMOUNT in view.columns
        assert "region.name" in view.columns

    def test_level_of_resolves_by_path(self, view: HierarchyView):
        assert view.level_of(AMOUNT) == "sale"
        assert view.level_of("region.store.name") == "store"
        assert view.level_of("region.name") == "region"

    def test_level_of_rejects_unknown(self, view: HierarchyView):
        with pytest.raises(KeyError, match="not owned by any level"):
            view.level_of("region.store.sale.nonexistent.deep")

    def test_plans_are_available_per_granularity(self, view: HierarchyView):
        assert "JOIN" in view.level("sale").explain().upper()
        assert view.nested().explain() != ""

    def test_explain_scan_shows_pushdown(
        self, view: HierarchyView, packer: HierarchicalPacker, tmp_path
    ):
        """Against Parquet, the per-level scans carry their own predicates."""
        view.sink_parquet(tmp_path)
        plan = (
            HierarchyView.scan_parquet(tmp_path, packer)
            .filter(pl.col("region.id") == 2)
            .level("sale")
            .explain()
        )
        assert "SCAN" in plan.upper()
        assert "SELECTION" in plan.upper()

    def test_repr_lists_levels(self, view: HierarchyView):
        assert "region" in repr(view) and "prune" in repr(view)

    def test_nothing_executes_until_terminal(self, view: HierarchyView):
        assert isinstance(view.level("sale"), pl.LazyFrame)
        assert isinstance(view.nested(), pl.LazyFrame)
        assert all(isinstance(t, pl.LazyFrame) for t in view.tables().values())


class TestFilterRouting:
    """Filters land on the level that can evaluate them — without user joins."""

    def test_leaf_attribute(self, view: HierarchyView, flat: pl.DataFrame):
        assert_same_rows(
            view.filter(pl.col(AMOUNT) > 12).level("sale").collect(),
            flat.filter(pl.col(AMOUNT) > 12),
            SALE_ID,
        )

    def test_ancestor_key_is_pushed_to_every_carrier(self, view: HierarchyView):
        """region.id is a foreign key on all three tables, so all three filter."""
        filtered = view.filter(pl.col("region.id") == 2)
        for level in ("region", "store", "sale"):
            table = filtered.tables()[level].collect()
            assert table["region.id"].unique().to_list() == [2]

    def test_ancestor_key_predicate_reaches_every_scan(
        self, view: HierarchyView, packer: HierarchicalPacker, tmp_path
    ):
        """
        The predicate must sit ON each level's scan, not merely be implied.

        Applying it to only one level would still be *correct* — the
        consistency cascade restores the same rows — so no data assertion can
        catch the difference. But the deepest scan would then read every row
        group and filter afterwards, losing exactly the row-group skipping this
        routing exists to buy. Hence a plan assertion.
        """
        view.sink_parquet(tmp_path)
        filtered = HierarchyView.scan_parquet(tmp_path, packer).filter(pl.col("region.id") == 2)
        plan = filtered.tables()["sale"].explain().upper()
        assert plan.count("SELECTION") >= len(filtered.levels), (
            "the ancestor-key predicate did not reach every level's scan:\n" + plan
        )

    def test_ancestor_key_matches_flat(self, view: HierarchyView, flat: pl.DataFrame):
        assert_same_rows(
            view.filter(pl.col("region.id") == 2).level("sale").collect(),
            flat.filter(pl.col("region.id") == 2),
            SALE_ID,
        )

    def test_ancestor_attribute_requires_join(self, view: HierarchyView, flat: pl.DataFrame):
        assert_same_rows(
            view.filter(pl.col("region.name") == "r1").level("sale").collect(),
            flat.filter(pl.col("region.name") == "r1"),
            SALE_ID,
        )

    def test_cross_level_predicate(self, view: HierarchyView, flat: pl.DataFrame):
        predicate = pl.col(AMOUNT) > pl.col("region.id") * 4
        assert_same_rows(
            view.filter(predicate).level("sale").collect(), flat.filter(predicate), SALE_ID
        )

    def test_cross_level_predicate_drops_borrowed_columns(self, view: HierarchyView):
        """Columns joined in to evaluate a predicate must not leak into the table."""
        filtered = view.filter(pl.col(AMOUNT) > pl.col("region.name").str.len_chars())
        assert "region.name" not in filtered.tables()["sale"].collect_schema().names()

    def test_multiple_predicates_compose(self, view: HierarchyView, flat: pl.DataFrame):
        assert_same_rows(
            view.filter(pl.col(AMOUNT) > 5, pl.col("region.id") == 1).level("sale").collect(),
            flat.filter(pl.col(AMOUNT) > 5, pl.col("region.id") == 1),
            SALE_ID,
        )

    def test_chained_filters_compose(self, view: HierarchyView, flat: pl.DataFrame):
        assert_same_rows(
            view.filter(pl.col(AMOUNT) > 5)
            .filter(pl.col("region.id") == 1)
            .level("sale")
            .collect(),
            flat.filter(pl.col(AMOUNT) > 5).filter(pl.col("region.id") == 1),
            SALE_ID,
        )

    def test_unknown_column_raises(self, view: HierarchyView):
        with pytest.raises(KeyError, match="[Uu]nknown column"):
            view.filter(pl.col("region.store.sale.nope.deep") > 1)

    def test_source_view_is_unchanged(self, view: HierarchyView):
        before = view.level("sale").collect().height
        view.filter(pl.col(AMOUNT) > 12)
        assert view.level("sale").collect().height == before


class TestCrossLevelConsistency:
    """Filtering one level must restrict the others — from every entry point.

    ``tables()`` performs no join of its own, so if the cascades were missing it
    would happily hand back orphaned child rows whose parent was filtered away.
    """

    def test_ancestor_filter_restricts_descendant_tables(
        self, view: HierarchyView, flat: pl.DataFrame
    ):
        filtered = view.filter(pl.col("region.name") == "r1")
        want = flat.filter(pl.col("region.name") == "r1")
        assert filtered.tables()["sale"].collect().height == want.height
        assert filtered.tables()["store"].collect().height == want["region.store.id"].n_unique()

    def test_descendant_filter_restricts_ancestor_tables(
        self, view: HierarchyView, flat: pl.DataFrame
    ):
        predicate = pl.col(AMOUNT) > 15
        filtered = view.filter(predicate)
        want = flat.filter(predicate)
        assert filtered.tables()["store"].collect().height == want["region.store.id"].n_unique()
        assert filtered.tables()["region"].collect().height == want["region.id"].n_unique()

    def test_tables_and_collect_agree(self, view: HierarchyView):
        """The cheap terminal and the joined terminal must tell the same story."""
        filtered = view.filter(pl.col("region.name") == "r1")
        assert filtered.tables()["sale"].collect().height == filtered.level("sale").collect().height

    def test_unfiltered_view_adds_no_joins(self, view: HierarchyView):
        """An untouched view must resolve to its scans, not to a join cascade."""
        assert "JOIN" not in view.tables()["sale"].explain().upper()

    def test_filtered_view_adds_cascade(self, view: HierarchyView):
        filtered = view.filter(pl.col("region.name") == "r1")
        assert "JOIN" in filtered.tables()["sale"].explain().upper()

    def test_keep_mode_still_restricts_downward(
        self, flat: pl.DataFrame, packer: HierarchicalPacker
    ):
        """'keep' relaxes upward pruning only; orphans are always wrong."""
        filtered = HierarchyView.from_frame(flat, packer, empty_parents="keep").filter(
            pl.col("region.name") == "r1"
        )
        want = flat.filter(pl.col("region.name") == "r1")
        assert filtered.tables()["sale"].collect().height == want.height


class TestEmptyParentSemantics:
    """Filtering children changes which parents survive — a real decision."""

    RARE = pl.col(AMOUNT) > 15  # matches in only some stores

    def test_prune_matches_pack(
        self, view: HierarchyView, flat: pl.DataFrame, packer: HierarchicalPacker
    ):
        got = view.filter(self.RARE).nested().collect()
        want = packer.pack(flat.filter(self.RARE), "region")
        assert_frame_equal(got.sort("region.id"), want.sort("region.id"), check_dtypes=False)

    def test_prune_is_the_default(self, view: HierarchyView):
        assert "prune" in repr(view)

    def test_keep_retains_childless_parents(self, flat: pl.DataFrame, packer: HierarchicalPacker):
        kept = (
            HierarchyView.from_frame(flat, packer, empty_parents="keep")
            .filter(self.RARE)
            .nested()
            .collect()
        )
        pruned = HierarchyView.from_frame(flat, packer).filter(self.RARE).nested().collect()

        def total_stores(frame: pl.DataFrame) -> int:
            return frame.select(pl.col("region.store").list.len().sum()).item()

        assert total_stores(kept) == N_REGION * N_STORE
        assert total_stores(pruned) < total_stores(kept)

    def test_mode_survives_operations(self, flat: pl.DataFrame, packer: HierarchicalPacker):
        chained = (
            HierarchyView.from_frame(flat, packer, empty_parents="keep")
            .filter(pl.col(AMOUNT) > 1)
            .filter(pl.col("region.id") >= 0)
        )
        assert "keep" in repr(chained)


class TestNestedRoundTrip:
    """The view can always hand back the packed shape."""

    def test_unfiltered(self, view: HierarchyView, flat: pl.DataFrame, packer: HierarchicalPacker):
        assert_frame_equal(
            view.nested().collect().sort("region.id"),
            packer.pack(flat, "region").sort("region.id"),
            check_dtypes=False,
        )

    def test_filtered(self, view: HierarchyView, flat: pl.DataFrame, packer: HierarchicalPacker):
        predicate = pl.col(AMOUNT) > 12
        assert_frame_equal(
            view.filter(predicate).nested().collect().sort("region.id"),
            packer.pack(flat.filter(predicate), "region").sort("region.id"),
            check_dtypes=False,
        )

    def test_collect_at_intermediate_level(self, view: HierarchyView):
        stores = view.level("store").collect()
        assert stores.height == N_REGION * N_STORE
        assert AMOUNT not in stores.columns

    def test_collect_defaults_to_finest_level(self, view: HierarchyView):
        assert view.level().collect().height == ROWS

    def test_collect_unknown_level(self, view: HierarchyView):
        with pytest.raises(KeyError, match="not present in this view"):
            view.level("planet").collect()


class TestLazyContract:
    """Lazy in, lazy out — nothing executes without a terminal call."""

    def test_filter_returns_a_new_view(self, view: HierarchyView):
        filtered = view.filter(pl.col(AMOUNT) > 1)
        assert isinstance(filtered, HierarchyView)
        assert filtered is not view

    def test_lazy_input_stays_lazy(self, flat: pl.DataFrame, packer: HierarchicalPacker):
        view = HierarchyView.from_frame(flat.lazy(), packer)
        assert isinstance(view.level("sale"), pl.LazyFrame)

    def test_tables_are_unexecuted(self, view: HierarchyView):
        tables = view.filter(pl.col(AMOUNT) > 12).tables()
        assert all(isinstance(t, pl.LazyFrame) for t in tables.values())


# =============================================================================
# Regressions found by design review
# =============================================================================


class TestMultiArgumentRouting:
    """
    Every argument in one call must survive.

    Routing a cross-level argument used to rebuild that level's frame from the
    *original* tables, discarding whatever earlier arguments in the same call
    had already written there. Chaining hid it, and so did predicates where one
    implies the other — these use orthogonal ones on purpose.
    """

    LEAF_ONLY = pl.col(SALE_ID) % 2 == 0
    CROSS_LEVEL = pl.col(AMOUNT) > pl.col("region.name").str.len_chars()

    def test_filter_keeps_every_predicate(self, view: HierarchyView, flat: pl.DataFrame):
        assert (
            view.filter(self.LEAF_ONLY, self.CROSS_LEVEL).level("sale").collect().height
            == flat.filter(self.LEAF_ONLY, self.CROSS_LEVEL).height
        )

    def test_filter_multi_arg_matches_chained(self, view: HierarchyView):
        assert (
            view.filter(self.LEAF_ONLY, self.CROSS_LEVEL).level("sale").collect().height
            == view.filter(self.LEAF_ONLY).filter(self.CROSS_LEVEL).level("sale").collect().height
        )

    def test_two_cross_level_predicates_both_apply(self, view: HierarchyView, flat: pl.DataFrame):
        a = pl.col(AMOUNT) > pl.col("region.name").str.len_chars()
        b = pl.col(AMOUNT) < pl.col("region.name").str.len_chars() * 3
        assert view.filter(a, b).level("sale").collect().height == flat.filter(a, b).height


class TestEscapedSeparators:
    """Field names containing the separator are escaped by the packer."""

    @pytest.fixture
    def escaped(self, packer: HierarchicalPacker) -> tuple[HierarchyView, str]:
        frame = pl.DataFrame(
            {
                "region.id": [0, 0, 1],
                "region.name": ["a", "a", "b"],
                "region.store.id": [0, 1, 2],
                "region.store.net\\.sales": [1.0, 2.0, 3.0],
                "region.store.sale.id": [0, 1, 2],
                AMOUNT: [10.0, 20.0, 30.0],
            }
        )
        return HierarchyView.from_frame(frame, packer), "region.store.net\\.sales"

    def test_level_of_resolves_escaped_column(self, escaped):
        view, column = escaped
        assert column in view.columns
        assert view.level_of(column) == "store"

    def test_cross_level_filter_on_escaped_column(self, escaped):
        view, column = escaped
        result = view.filter(pl.col(column) > pl.col("region.id")).level("store").collect()
        assert result.height > 0

    def test_escaped_column_survives_the_axis_join(self, escaped):
        view, column = escaped
        assert column in view.level("store").collect_schema().names()

    def test_rollup_of_an_escaped_attribute(self, escaped):
        """A group_by on key_columns does not care that the name contains a dot."""
        view, column = escaped
        rolled = (
            view.level("store")
            .group_by(view.key_columns("region"))
            .agg(pl.col(column).sum().alias("region.net\\.sales"))
            .collect()
        )
        assert "region.net\\.sales" in rolled.columns


class TestAggregatingPredicateRouting:
    """
    Broadcasting an ancestor-key predicate to every carrier is only sound for
    row-wise predicates. Each table holds the key at a different granularity, so
    an aggregate over it means something different per level. Such a predicate
    is therefore evaluated once, at the level that owns the column.
    """

    def test_aggregating_predicate_evaluates_at_the_owning_level(self, view: HierarchyView):
        """count() over an ancestor key counts entities of that level."""
        n_regions = view.tables()["region"].collect().height

        kept = view.filter(pl.col("region.id").count() == n_regions)
        assert kept.tables()["region"].collect().height == n_regions

        dropped = view.filter(pl.col("region.id").count() > n_regions)
        assert dropped.tables()["region"].collect().height == 0
        assert dropped.tables()["sale"].collect().height == 0, "cascade must follow"

    def test_row_wise_key_predicate_still_broadcasts(
        self, view: HierarchyView, packer: HierarchicalPacker, tmp_path
    ):
        view.sink_parquet(tmp_path)
        filtered = HierarchyView.scan_parquet(tmp_path, packer).filter(pl.col("region.id") == 2)
        plan = filtered.tables()["sale"].explain().upper()
        assert plan.count("SELECTION") >= len(filtered.levels)

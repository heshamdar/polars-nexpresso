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
        assert_same_rows(direct.collect("sale"), view.collect("sale"), SALE_ID)

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
        assert_same_rows(rescanned.collect("sale"), view.collect("sale"), SALE_ID)

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
    """The view presents a nested face over flat tables."""

    def test_schema_is_nested(
        self, view: HierarchyView, packer: HierarchicalPacker, flat: pl.DataFrame
    ):
        assert view.schema == packer.pack(flat, "region").schema

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

    def test_explain_returns_plans(self, view: HierarchyView):
        assert "JOIN" in view.explain("sale").upper()
        assert view.explain() != ""

    def test_explain_scan_shows_pushdown(
        self, view: HierarchyView, packer: HierarchicalPacker, tmp_path
    ):
        """Against Parquet, the per-level scans carry their own predicates."""
        view.sink_parquet(tmp_path)
        plan = (
            HierarchyView.scan_parquet(tmp_path, packer)
            .filter(pl.col("region.id") == 2)
            .explain("sale")
        )
        assert "SCAN" in plan.upper()
        assert "SELECTION" in plan.upper()

    def test_repr_lists_levels(self, view: HierarchyView):
        assert "region" in repr(view) and "prune" in repr(view)

    def test_nothing_executes_until_terminal(self, view: HierarchyView):
        assert isinstance(view.to_flat("sale"), pl.LazyFrame)
        assert isinstance(view.to_nested(), pl.LazyFrame)
        assert all(isinstance(t, pl.LazyFrame) for t in view.tables().values())


class TestFilterRouting:
    """Filters land on the level that can evaluate them — without user joins."""

    def test_leaf_attribute(self, view: HierarchyView, flat: pl.DataFrame):
        assert_same_rows(
            view.filter(pl.col(AMOUNT) > 12).collect("sale"),
            flat.filter(pl.col(AMOUNT) > 12),
            SALE_ID,
        )

    def test_ancestor_key_is_pushed_to_every_carrier(self, view: HierarchyView):
        """region.id is a foreign key on all three tables, so all three filter."""
        filtered = view.filter(pl.col("region.id") == 2)
        for level in ("region", "store", "sale"):
            table = filtered.tables()[level].collect()
            assert table["region.id"].unique().to_list() == [2]

    def test_ancestor_key_matches_flat(self, view: HierarchyView, flat: pl.DataFrame):
        assert_same_rows(
            view.filter(pl.col("region.id") == 2).collect("sale"),
            flat.filter(pl.col("region.id") == 2),
            SALE_ID,
        )

    def test_ancestor_attribute_requires_join(self, view: HierarchyView, flat: pl.DataFrame):
        assert_same_rows(
            view.filter(pl.col("region.name") == "r1").collect("sale"),
            flat.filter(pl.col("region.name") == "r1"),
            SALE_ID,
        )

    def test_cross_level_predicate(self, view: HierarchyView, flat: pl.DataFrame):
        predicate = pl.col(AMOUNT) > pl.col("region.id") * 4
        assert_same_rows(view.filter(predicate).collect("sale"), flat.filter(predicate), SALE_ID)

    def test_cross_level_predicate_drops_borrowed_columns(self, view: HierarchyView):
        """Columns joined in to evaluate a predicate must not leak into the table."""
        filtered = view.filter(pl.col(AMOUNT) > pl.col("region.name").str.len_chars())
        assert "region.name" not in filtered.tables()["sale"].collect_schema().names()

    def test_multiple_predicates_compose(self, view: HierarchyView, flat: pl.DataFrame):
        assert_same_rows(
            view.filter(pl.col(AMOUNT) > 5, pl.col("region.id") == 1).collect("sale"),
            flat.filter(pl.col(AMOUNT) > 5, pl.col("region.id") == 1),
            SALE_ID,
        )

    def test_chained_filters_compose(self, view: HierarchyView, flat: pl.DataFrame):
        assert_same_rows(
            view.filter(pl.col(AMOUNT) > 5).filter(pl.col("region.id") == 1).collect("sale"),
            flat.filter(pl.col(AMOUNT) > 5).filter(pl.col("region.id") == 1),
            SALE_ID,
        )

    def test_unknown_column_raises(self, view: HierarchyView):
        with pytest.raises(KeyError, match="unknown column"):
            view.filter(pl.col("region.store.sale.nope.deep") > 1)

    def test_source_view_is_unchanged(self, view: HierarchyView):
        before = view.collect("sale").height
        view.filter(pl.col(AMOUNT) > 12)
        assert view.collect("sale").height == before


class TestWithColumns:
    """Derived columns land on the level of their deepest input."""

    def test_leaf_expression(self, view: HierarchyView, flat: pl.DataFrame):
        expr = (pl.col(AMOUNT) * 2).alias("region.store.sale.doubled")
        assert_same_rows(view.with_columns(expr).collect("sale"), flat.with_columns(expr), SALE_ID)

    def test_lands_on_owning_level(self, view: HierarchyView):
        widened = view.with_columns((pl.col(AMOUNT) * 2).alias("region.store.sale.doubled"))
        assert "region.store.sale.doubled" in widened.tables()["sale"].collect_schema().names()
        assert (
            "region.store.sale.doubled" not in widened.tables()["region"].collect_schema().names()
        )

    def test_cross_level_expression(self, view: HierarchyView, flat: pl.DataFrame):
        expr = (pl.col(AMOUNT) + pl.col("region.id")).alias("region.store.sale.adj")
        assert_same_rows(view.with_columns(expr).collect("sale"), flat.with_columns(expr), SALE_ID)

    def test_unknown_column_raises(self, view: HierarchyView):
        with pytest.raises(KeyError, match="unknown column"):
            view.with_columns((pl.col("region.store.sale.nope.deep") * 2).alias("x"))


class TestDrop:
    def test_drops_attribute(self, view: HierarchyView):
        dropped = view.drop("region.store.name")
        assert "region.store.name" not in dropped.collect("sale").columns

    def test_refuses_key_columns(self, view: HierarchyView):
        with pytest.raises(ValueError, match="key column"):
            view.drop("region.id")


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
        assert filtered.tables()["sale"].collect().height == filtered.collect("sale").height

    def test_any_child_satisfies_restricts_descendants(self, view: HierarchyView):
        restricted = view.any_child_satisfies(
            pl.col(AMOUNT) > 15, at_level="store", child_level="sale"
        )
        surviving = set(restricted.tables()["store"].collect()["region.store.id"].to_list())
        sale_stores = set(restricted.tables()["sale"].collect()["region.store.id"].to_list())
        assert sale_stores <= surviving

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
        got = view.filter(self.RARE).collect_nested()
        want = packer.pack(flat.filter(self.RARE), "region")
        assert_frame_equal(got.sort("region"), want.sort("region"), check_dtypes=False)

    def test_prune_is_the_default(self, view: HierarchyView):
        assert "prune" in repr(view)

    def test_keep_retains_childless_parents(self, flat: pl.DataFrame, packer: HierarchicalPacker):
        kept = (
            HierarchyView.from_frame(flat, packer, empty_parents="keep")
            .filter(self.RARE)
            .collect_nested()
        )
        pruned = HierarchyView.from_frame(flat, packer).filter(self.RARE).collect_nested()

        def total_stores(frame: pl.DataFrame) -> int:
            return frame.select(pl.col("region").struct.field("store").list.len().sum()).item()

        assert total_stores(kept) == N_REGION * N_STORE
        assert total_stores(pruned) < total_stores(kept)

    def test_mode_survives_operations(self, flat: pl.DataFrame, packer: HierarchicalPacker):
        chained = (
            HierarchyView.from_frame(flat, packer, empty_parents="keep")
            .filter(pl.col(AMOUNT) > 1)
            .drop("region.store.name")
        )
        assert "keep" in repr(chained)


class TestNestedRoundTrip:
    """The view can always hand back the packed shape."""

    def test_unfiltered(self, view: HierarchyView, flat: pl.DataFrame, packer: HierarchicalPacker):
        assert_frame_equal(
            view.collect_nested().sort("region"),
            packer.pack(flat, "region").sort("region"),
            check_dtypes=False,
        )

    def test_filtered(self, view: HierarchyView, flat: pl.DataFrame, packer: HierarchicalPacker):
        predicate = pl.col(AMOUNT) > 12
        assert_frame_equal(
            view.filter(predicate).collect_nested().sort("region"),
            packer.pack(flat.filter(predicate), "region").sort("region"),
            check_dtypes=False,
        )

    def test_collect_at_intermediate_level(self, view: HierarchyView):
        stores = view.collect("store")
        assert stores.height == N_REGION * N_STORE
        assert AMOUNT not in stores.columns

    def test_collect_defaults_to_finest_level(self, view: HierarchyView):
        assert view.collect().height == ROWS

    def test_collect_unknown_level(self, view: HierarchyView):
        with pytest.raises(KeyError, match="not present in this view"):
            view.collect("planet")


class TestPromote:
    """Relational attribute promotion, without building List[Struct]."""

    def test_matches_promote_attribute(
        self, view: HierarchyView, flat: pl.DataFrame, packer: HierarchicalPacker
    ):
        got = view.promote(
            "amount", from_level="sale", to_level="store", agg="sum", alias="total"
        ).collect("store")
        want = packer.promote_attribute(
            flat, "amount", from_level="sale", to_level="store", agg="sum", alias="total"
        )
        cols = ["region.store.id", "region.store.total"]
        assert_same_rows(got.select(cols), want.select(cols), "region.store.id")

    def test_lands_on_parent_level(self, view: HierarchyView):
        promoted = view.promote("amount", from_level="sale", to_level="store", agg="sum")
        assert "region.store.amount" in promoted.tables()["store"].collect_schema().names()

    @pytest.mark.parametrize("agg", ["sum", "mean", "min", "max", "count", "first", "last"])
    def test_aggregations(self, view: HierarchyView, agg):
        promoted = view.promote("amount", from_level="sale", to_level="store", agg=agg, alias="agg")
        assert promoted.collect("store").height == N_REGION * N_STORE

    def test_list_aggregation_yields_list(self, view: HierarchyView):
        promoted = view.promote(
            "amount", from_level="sale", to_level="store", agg="list", alias="all"
        )
        dtype = promoted.tables()["store"].collect_schema()["region.store.all"]
        assert isinstance(dtype, pl.List)

    def test_requires_immediate_child(self, view: HierarchyView):
        with pytest.raises(ValueError, match="immediate child"):
            view.promote("amount", from_level="sale", to_level="region", agg="sum")

    def test_rejects_missing_attribute(self, view: HierarchyView):
        with pytest.raises(ValueError, match="not found at level"):
            view.promote("nope", from_level="sale", to_level="store", agg="sum")

    def test_rejects_unknown_level(self, view: HierarchyView):
        with pytest.raises(KeyError, match="not present in this view"):
            view.promote("amount", from_level="sale", to_level="planet", agg="sum")


class TestAnyChildSatisfies:
    """Existence questions become semi-joins."""

    def test_matches_flat_ground_truth(self, view: HierarchyView, flat: pl.DataFrame):
        predicate = pl.col(AMOUNT) > 15
        kept = (
            view.any_child_satisfies(predicate, at_level="store", child_level="sale")
            .tables()["store"]
            .collect()
        )
        want = flat.filter(predicate)["region.store.id"].n_unique()
        assert kept.height == want

    def test_skips_levels(self, view: HierarchyView, flat: pl.DataFrame):
        predicate = pl.col(AMOUNT) > 15
        kept = (
            view.any_child_satisfies(predicate, at_level="region", child_level="sale")
            .tables()["region"]
            .collect()
        )
        assert kept.height == flat.filter(predicate)["region.id"].n_unique()

    def test_requires_finer_child(self, view: HierarchyView):
        with pytest.raises(ValueError, match="must be finer"):
            view.any_child_satisfies(
                pl.col("region.name") == "r0", at_level="sale", child_level="region"
            )

    def test_rejects_unknown_level(self, view: HierarchyView):
        with pytest.raises(KeyError, match="not present in this view"):
            view.any_child_satisfies(pl.col(AMOUNT) > 1, at_level="planet", child_level="sale")


class TestLazyContract:
    """Lazy in, lazy out — nothing executes without a terminal call."""

    def test_operations_return_new_views(self, view: HierarchyView):
        assert isinstance(view.filter(pl.col(AMOUNT) > 1), HierarchyView)
        assert isinstance(view.drop("region.store.name"), HierarchyView)
        assert isinstance(
            view.promote("amount", from_level="sale", to_level="store", agg="sum"),
            HierarchyView,
        )

    def test_lazy_input_stays_lazy(self, flat: pl.DataFrame, packer: HierarchicalPacker):
        view = HierarchyView.from_frame(flat.lazy(), packer)
        assert isinstance(view.to_flat("sale"), pl.LazyFrame)

    def test_tables_are_unexecuted(self, view: HierarchyView):
        tables = view.filter(pl.col(AMOUNT) > 12).tables()
        assert all(isinstance(t, pl.LazyFrame) for t in tables.values())

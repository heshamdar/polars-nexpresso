"""Comprehensive tests for the NormalizedPacker."""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from nexpresso import (
    F,
    HierarchySpec,
    HierarchyValidationError,
    LevelAttribute,
    LevelSpec,
    NormalizedPacker,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def two_level_spec() -> HierarchySpec:
    """Region → Store hierarchy."""
    return HierarchySpec.from_levels(
        LevelSpec(name="region", id_fields=["id"]),
        LevelSpec(name="store", id_fields=["id"], parent_keys=["region_id"]),
    )


@pytest.fixture
def three_level_spec() -> HierarchySpec:
    """Region → Store → Product hierarchy."""
    return HierarchySpec.from_levels(
        LevelSpec(name="region", id_fields=["id"]),
        LevelSpec(name="store", id_fields=["id"], parent_keys=["region_id"]),
        LevelSpec(name="product", id_fields=["id"], parent_keys=["store_id"]),
    )


@pytest.fixture
def regions_df() -> pl.DataFrame:
    return pl.DataFrame({
        "id": ["r1", "r2"],
        "name": ["North", "South"],
    })


@pytest.fixture
def stores_df() -> pl.DataFrame:
    return pl.DataFrame({
        "id": ["s1", "s2", "s3"],
        "name": ["Store A", "Store B", "Store C"],
        "revenue": [100_000, 200_000, 150_000],
        "region_id": ["r1", "r1", "r2"],
    })


@pytest.fixture
def products_df() -> pl.DataFrame:
    return pl.DataFrame({
        "id": ["p1", "p2", "p3", "p4"],
        "name": ["Widget", "Gadget", "Doohickey", "Thingamajig"],
        "price": [9.99, 19.99, 4.99, 29.99],
        "store_id": ["s1", "s1", "s2", "s3"],
    })


@pytest.fixture
def two_level_packer(
    two_level_spec: HierarchySpec,
    regions_df: pl.DataFrame,
    stores_df: pl.DataFrame,
) -> NormalizedPacker:
    return NormalizedPacker(
        two_level_spec,
        tables={"region": regions_df, "store": stores_df},
    )


@pytest.fixture
def three_level_packer(
    three_level_spec: HierarchySpec,
    regions_df: pl.DataFrame,
    stores_df: pl.DataFrame,
    products_df: pl.DataFrame,
) -> NormalizedPacker:
    return NormalizedPacker(
        three_level_spec,
        tables={"region": regions_df, "store": stores_df, "product": products_df},
    )


# ============================================================================
# Construction & validation
# ============================================================================


class TestConstruction:
    """Test NormalizedPacker creation and input validation."""

    def test_basic_construction(self, two_level_packer: NormalizedPacker) -> None:
        assert two_level_packer.level_names == ["region", "store"]

    def test_construction_from_lazyframes(
        self, two_level_spec: HierarchySpec, regions_df: pl.DataFrame, stores_df: pl.DataFrame
    ) -> None:
        packer = NormalizedPacker(
            two_level_spec,
            tables={"region": regions_df.lazy(), "store": stores_df.lazy()},
        )
        assert packer.level_names == ["region", "store"]

    def test_missing_table_raises(self, two_level_spec: HierarchySpec, regions_df: pl.DataFrame) -> None:
        with pytest.raises(HierarchyValidationError, match="Missing table"):
            NormalizedPacker(two_level_spec, tables={"region": regions_df})

    def test_missing_id_field_raises(self, two_level_spec: HierarchySpec) -> None:
        bad_regions = pl.DataFrame({"wrong_col": ["r1"]})
        stores = pl.DataFrame({"id": ["s1"], "region_id": ["r1"]})
        with pytest.raises(HierarchyValidationError, match="id_field 'id' not found"):
            NormalizedPacker(two_level_spec, tables={"region": bad_regions, "store": stores})

    def test_missing_parent_key_raises(self, two_level_spec: HierarchySpec) -> None:
        regions = pl.DataFrame({"id": ["r1"], "name": ["North"]})
        bad_stores = pl.DataFrame({"id": ["s1"], "name": ["Store A"]})
        with pytest.raises(HierarchyValidationError, match="parent_key 'region_id' not found"):
            NormalizedPacker(two_level_spec, tables={"region": regions, "store": bad_stores})


# ============================================================================
# Introspection
# ============================================================================


class TestIntrospection:
    """Test introspection properties and methods."""

    def test_level_names(self, three_level_packer: NormalizedPacker) -> None:
        assert three_level_packer.level_names == ["region", "store", "product"]

    def test_root_level(self, three_level_packer: NormalizedPacker) -> None:
        assert three_level_packer.root_level == "region"

    def test_leaf_level(self, three_level_packer: NormalizedPacker) -> None:
        assert three_level_packer.leaf_level == "product"

    def test_get_ancestor_levels(self, three_level_packer: NormalizedPacker) -> None:
        assert three_level_packer.get_ancestor_levels("product") == ["region", "store"]
        assert three_level_packer.get_ancestor_levels("store") == ["region"]
        assert three_level_packer.get_ancestor_levels("region") == []

    def test_get_descendant_levels(self, three_level_packer: NormalizedPacker) -> None:
        assert three_level_packer.get_descendant_levels("region") == ["store", "product"]
        assert three_level_packer.get_descendant_levels("store") == ["product"]
        assert three_level_packer.get_descendant_levels("product") == []

    def test_get_level_keys_short(self, two_level_packer: NormalizedPacker) -> None:
        assert two_level_packer.get_level_keys("region") == ["id"]

    def test_get_level_keys_long(self, two_level_packer: NormalizedPacker) -> None:
        assert two_level_packer.get_level_keys("store", form="long") == ["region.store.id"]

    def test_get_level_fields(self, two_level_packer: NormalizedPacker) -> None:
        fields = two_level_packer.get_level_fields("store")
        assert "id" in fields
        assert "name" in fields
        assert "revenue" in fields

    def test_describe(self, two_level_packer: NormalizedPacker) -> None:
        desc = two_level_packer.describe()
        assert "region" in desc
        assert "store" in desc

    def test_tables_property(self, two_level_packer: NormalizedPacker) -> None:
        tables = two_level_packer.tables
        assert "region" in tables
        assert "store" in tables
        assert isinstance(tables["region"], pl.LazyFrame)


# ============================================================================
# collect()
# ============================================================================


class TestCollect:
    """Test joining tables into flat frames at various levels."""

    def test_collect_root_level(self, two_level_packer: NormalizedPacker) -> None:
        """Collecting at root should return the root table with qualified names."""
        result = two_level_packer.collect("region").collect()
        assert "region.id" in result.columns
        assert "region.name" in result.columns
        assert result.height == 2

    def test_collect_child_level(self, two_level_packer: NormalizedPacker) -> None:
        """Collecting at child level joins parent and child."""
        result = two_level_packer.collect("store").collect()
        assert "region.id" in result.columns
        assert "region.name" in result.columns
        assert "region.store.id" in result.columns
        assert "region.store.name" in result.columns
        assert "region.store.revenue" in result.columns
        # 3 stores, each joined with their region
        assert result.height == 3

    def test_collect_preserves_data(
        self, two_level_packer: NormalizedPacker
    ) -> None:
        """Check that join produces correct data."""
        result = two_level_packer.collect("store").collect().sort("region.store.id")
        # s1 is in region r1 (North)
        row_s1 = result.filter(pl.col("region.store.id") == "s1")
        assert row_s1["region.name"].to_list() == ["North"]
        # s3 is in region r2 (South)
        row_s3 = result.filter(pl.col("region.store.id") == "s3")
        assert row_s3["region.name"].to_list() == ["South"]

    def test_collect_three_levels(self, three_level_packer: NormalizedPacker) -> None:
        """Collecting at leaf level joins all three tables."""
        result = three_level_packer.collect("product").collect()
        assert "region.id" in result.columns
        assert "region.store.id" in result.columns
        assert "region.store.product.id" in result.columns
        assert "region.store.product.price" in result.columns
        assert result.height == 4  # 4 products

    def test_collect_returns_lazyframe(self, two_level_packer: NormalizedPacker) -> None:
        """collect() always returns a LazyFrame."""
        result = two_level_packer.collect("store")
        assert isinstance(result, pl.LazyFrame)

    def test_collect_left_join_preserves_parents(
        self, two_level_spec: HierarchySpec
    ) -> None:
        """Left join preserves parents with no children."""
        regions = pl.DataFrame({"id": ["r1", "r2"], "name": ["A", "B"]})
        stores = pl.DataFrame({
            "id": ["s1"],
            "name": ["Store"],
            "revenue": [100],
            "region_id": ["r1"],
        })
        packer = NormalizedPacker(
            two_level_spec,
            tables={"region": regions, "store": stores},
            join_type="left",
        )
        result = packer.collect("store").collect()
        # r2 has no stores, but should still appear with null store columns
        assert result.height == 2

    def test_collect_inner_join_drops_childless_parents(
        self, two_level_spec: HierarchySpec
    ) -> None:
        """Inner join drops parents with no children."""
        regions = pl.DataFrame({"id": ["r1", "r2"], "name": ["A", "B"]})
        stores = pl.DataFrame({
            "id": ["s1"],
            "name": ["Store"],
            "revenue": [100],
            "region_id": ["r1"],
        })
        packer = NormalizedPacker(
            two_level_spec,
            tables={"region": regions, "store": stores},
            join_type="inner",
        )
        result = packer.collect("store").collect()
        assert result.height == 1


# ============================================================================
# promote_attribute()
# ============================================================================


class TestPromoteAttribute:
    """Test aggregating child attributes to parent level."""

    def test_promote_sum(self, two_level_packer: NormalizedPacker) -> None:
        result = two_level_packer.promote_attribute(
            "revenue", from_level="store", to_level="region", agg="sum"
        ).collect().sort("region.id")
        # r1 has stores s1 (100k) + s2 (200k) = 300k
        # r2 has store s3 (150k)
        r1 = result.filter(pl.col("region.id") == "r1")
        assert r1["region.revenue"].to_list() == [300_000]
        r2 = result.filter(pl.col("region.id") == "r2")
        assert r2["region.revenue"].to_list() == [150_000]

    def test_promote_mean(self, two_level_packer: NormalizedPacker) -> None:
        result = two_level_packer.promote_attribute(
            "revenue", from_level="store", to_level="region", agg="mean"
        ).collect().sort("region.id")
        r1 = result.filter(pl.col("region.id") == "r1")
        assert r1["region.revenue"].to_list() == [150_000.0]

    def test_promote_count(self, two_level_packer: NormalizedPacker) -> None:
        result = two_level_packer.promote_attribute(
            "revenue", from_level="store", to_level="region", agg="count"
        ).collect().sort("region.id")
        r1 = result.filter(pl.col("region.id") == "r1")
        assert r1["region.revenue"].to_list() == [2]

    def test_promote_min(self, two_level_packer: NormalizedPacker) -> None:
        result = two_level_packer.promote_attribute(
            "revenue", from_level="store", to_level="region", agg="min"
        ).collect().sort("region.id")
        r1 = result.filter(pl.col("region.id") == "r1")
        assert r1["region.revenue"].to_list() == [100_000]

    def test_promote_max(self, two_level_packer: NormalizedPacker) -> None:
        result = two_level_packer.promote_attribute(
            "revenue", from_level="store", to_level="region", agg="max"
        ).collect().sort("region.id")
        r1 = result.filter(pl.col("region.id") == "r1")
        assert r1["region.revenue"].to_list() == [200_000]

    def test_promote_first(self, two_level_packer: NormalizedPacker) -> None:
        result = two_level_packer.promote_attribute(
            "name", from_level="store", to_level="region", agg="first",
            alias="first_store_name",
        ).collect().sort("region.id")
        # First store name for r1
        r1 = result.filter(pl.col("region.id") == "r1")
        assert r1["region.first_store_name"].to_list()[0] in ["Store A", "Store B"]

    def test_promote_list(self, two_level_packer: NormalizedPacker) -> None:
        result = two_level_packer.promote_attribute(
            "revenue", from_level="store", to_level="region", agg="list"
        ).collect().sort("region.id")
        r1 = result.filter(pl.col("region.id") == "r1")
        revenues = sorted(r1["region.revenue"].to_list()[0])
        assert revenues == [100_000, 200_000]

    def test_promote_with_alias(self, two_level_packer: NormalizedPacker) -> None:
        result = two_level_packer.promote_attribute(
            "revenue",
            from_level="store",
            to_level="region",
            agg="sum",
            alias="total_revenue",
        ).collect()
        assert "region.total_revenue" in result.columns

    def test_promote_non_adjacent_raises(self, three_level_packer: NormalizedPacker) -> None:
        with pytest.raises(ValueError, match="immediate child"):
            three_level_packer.promote_attribute(
                "price", from_level="product", to_level="region", agg="sum"
            )

    def test_promote_missing_attribute_raises(self, two_level_packer: NormalizedPacker) -> None:
        with pytest.raises(ValueError, match="not found"):
            two_level_packer.promote_attribute(
                "nonexistent", from_level="store", to_level="region", agg="sum"
            )


# ============================================================================
# any_child_satisfies()
# ============================================================================


class TestAnyChildSatisfies:
    """Test existential child filtering."""

    def test_basic_filter(self, two_level_packer: NormalizedPacker) -> None:
        """Filter to regions where any store has revenue > 180k."""
        result = two_level_packer.any_child_satisfies(
            from_level="store",
            to_level="region",
            condition=F("revenue") > 180_000,
        ).collect()
        # Only r1 has store s2 with 200k
        assert result.height == 1
        assert result["region.id"].to_list() == ["r1"]

    def test_all_match(self, two_level_packer: NormalizedPacker) -> None:
        """When all parents have matching children, all are returned."""
        result = two_level_packer.any_child_satisfies(
            from_level="store",
            to_level="region",
            condition=F("revenue") > 50_000,
        ).collect()
        assert result.height == 2

    def test_none_match(self, two_level_packer: NormalizedPacker) -> None:
        """When no children match, empty result."""
        result = two_level_packer.any_child_satisfies(
            from_level="store",
            to_level="region",
            condition=F("revenue") > 1_000_000,
        ).collect()
        assert result.height == 0

    def test_compound_condition(self, two_level_packer: NormalizedPacker) -> None:
        """Test with compound LevelExpr condition."""
        result = two_level_packer.any_child_satisfies(
            from_level="store",
            to_level="region",
            condition=(F("revenue") > 100_000) & (F("name") == "Store B"),
        ).collect()
        assert result.height == 1
        assert result["region.id"].to_list() == ["r1"]

    def test_non_adjacent_raises(self, three_level_packer: NormalizedPacker) -> None:
        with pytest.raises(ValueError, match="immediate child"):
            three_level_packer.any_child_satisfies(
                from_level="product",
                to_level="region",
                condition=F("price") > 10,
            )


# ============================================================================
# all_children_satisfy()
# ============================================================================


class TestAllChildrenSatisfy:
    """Test universal child filtering."""

    def test_basic_filter(self, two_level_packer: NormalizedPacker) -> None:
        """Filter to regions where ALL stores have revenue > 120k."""
        result = two_level_packer.all_children_satisfy(
            from_level="store",
            to_level="region",
            condition=F("revenue") > 120_000,
        ).collect()
        # r1 has s1 (100k < 120k), so r1 fails
        # r2 has s3 (150k > 120k), so r2 passes
        assert result.height == 1
        assert result["region.id"].to_list() == ["r2"]

    def test_vacuous_truth(self, two_level_spec: HierarchySpec) -> None:
        """Parents with no children satisfy vacuously."""
        regions = pl.DataFrame({"id": ["r1", "r2"], "name": ["A", "B"]})
        stores = pl.DataFrame({
            "id": ["s1"],
            "name": ["Store"],
            "revenue": [100],
            "region_id": ["r1"],
        })
        packer = NormalizedPacker(
            two_level_spec,
            tables={"region": regions, "store": stores},
        )
        # revenue > 200 — s1 fails, so r1 fails; r2 has no children → passes
        result = packer.all_children_satisfy(
            from_level="store",
            to_level="region",
            condition=F("revenue") > 200,
        ).collect()
        assert result.height == 1
        assert result["region.id"].to_list() == ["r2"]

    def test_all_pass(self, two_level_packer: NormalizedPacker) -> None:
        """When all children satisfy, all parents returned."""
        result = two_level_packer.all_children_satisfy(
            from_level="store",
            to_level="region",
            condition=F("revenue") > 0,
        ).collect()
        assert result.height == 2


# ============================================================================
# enrich()
# ============================================================================


class TestEnrich:
    """Test batch attribute promotion."""

    def test_single_enrichment(self, two_level_packer: NormalizedPacker) -> None:
        result = two_level_packer.enrich(
            LevelAttribute("revenue", "store", "sum", alias="total_rev"),
            at_level="region",
        ).collect().sort("region.id")
        assert "region.total_rev" in result.columns
        r1 = result.filter(pl.col("region.id") == "r1")
        assert r1["region.total_rev"].to_list() == [300_000]

    def test_multiple_enrichments(self, two_level_packer: NormalizedPacker) -> None:
        result = two_level_packer.enrich(
            LevelAttribute("revenue", "store", "sum", alias="total_rev"),
            LevelAttribute("revenue", "store", "count", alias="store_count"),
            LevelAttribute("revenue", "store", "mean", alias="avg_rev"),
            at_level="region",
        ).collect().sort("region.id")
        assert "region.total_rev" in result.columns
        assert "region.store_count" in result.columns
        assert "region.avg_rev" in result.columns


# ============================================================================
# pack()
# ============================================================================


class TestPack:
    """Test escape hatch to nested form."""

    def test_pack_to_root(self, two_level_packer: NormalizedPacker) -> None:
        result = two_level_packer.pack("region").collect()
        assert "store" in result.columns or "region" in result.columns
        # The result should be packed — store should be a List[Struct]
        # After pack, we should have region-level data with nested stores

    def test_pack_returns_lazyframe(self, two_level_packer: NormalizedPacker) -> None:
        result = two_level_packer.pack("region")
        assert isinstance(result, pl.LazyFrame)


# ============================================================================
# validate()
# ============================================================================


class TestValidate:
    """Test referential integrity validation."""

    def test_valid_data(self, two_level_packer: NormalizedPacker) -> None:
        errors = two_level_packer.validate(raise_on_error=False)
        assert errors == []

    def test_orphan_children(self, two_level_spec: HierarchySpec) -> None:
        regions = pl.DataFrame({"id": ["r1"], "name": ["North"]})
        stores = pl.DataFrame({
            "id": ["s1", "s2"],
            "name": ["A", "B"],
            "revenue": [100, 200],
            "region_id": ["r1", "r_missing"],
        })
        packer = NormalizedPacker(
            two_level_spec,
            tables={"region": regions, "store": stores},
        )
        errors = packer.validate(raise_on_error=False)
        assert len(errors) == 1
        assert "orphan" in errors[0].lower()

    def test_null_keys(self, two_level_spec: HierarchySpec) -> None:
        regions = pl.DataFrame({"id": ["r1", None], "name": ["North", "South"]})
        stores = pl.DataFrame({
            "id": ["s1"],
            "name": ["A"],
            "revenue": [100],
            "region_id": ["r1"],
        })
        packer = NormalizedPacker(
            two_level_spec,
            tables={"region": regions, "store": stores},
        )
        with pytest.raises(HierarchyValidationError, match="null"):
            packer.validate()


# ============================================================================
# update_table()
# ============================================================================


class TestUpdateTable:
    """Test table replacement."""

    def test_update_table(self, two_level_packer: NormalizedPacker) -> None:
        new_stores = pl.DataFrame({
            "id": ["s10"],
            "name": ["New Store"],
            "revenue": [500_000],
            "region_id": ["r1"],
        })
        two_level_packer.update_table("store", new_stores)
        result = two_level_packer.collect("store").collect()
        # Left join: r1 has 1 store, r2 has 0 stores (null row)
        assert result.height == 2
        non_null = result.filter(pl.col("region.store.id").is_not_null())
        assert non_null.height == 1
        assert non_null["region.store.name"].to_list() == ["New Store"]

    def test_update_invalid_level_raises(self, two_level_packer: NormalizedPacker) -> None:
        with pytest.raises(KeyError):
            two_level_packer.update_table("nonexistent", pl.DataFrame())


# ============================================================================
# Equivalence with HierarchicalPacker
# ============================================================================


class TestEquivalenceWithHierarchicalPacker:
    """Verify that NormalizedPacker produces results equivalent to
    HierarchicalPacker.build_from_tables() for the same data."""

    def test_collect_matches_unpack(
        self,
        two_level_spec: HierarchySpec,
        regions_df: pl.DataFrame,
        stores_df: pl.DataFrame,
    ) -> None:
        """collect() should produce the same data as build_from_tables + unpack."""
        from nexpresso import HierarchicalPacker

        # Normalized approach
        npacker = NormalizedPacker(
            two_level_spec,
            tables={"region": regions_df, "store": stores_df},
        )
        norm_result = npacker.collect("store").collect().sort("region.store.id")

        # Nested approach
        hpacker = HierarchicalPacker(two_level_spec)
        nested = hpacker.build_from_tables(
            {"region": regions_df, "store": stores_df},
            target_level="region",
        )
        hier_result = hpacker.unpack(nested, "store").sort("region.store.id")
        if isinstance(hier_result, pl.LazyFrame):
            hier_result = hier_result.collect()

        # Compare — both should have the same columns and data
        # Normalized may have extra join columns; compare shared columns
        shared_cols = sorted(
            set(norm_result.columns) & set(hier_result.columns)
        )
        assert len(shared_cols) > 0
        assert_frame_equal(
            norm_result.select(shared_cols),
            hier_result.select(shared_cols),
        )

    def test_promote_attribute_matches(
        self,
        two_level_spec: HierarchySpec,
        regions_df: pl.DataFrame,
        stores_df: pl.DataFrame,
    ) -> None:
        """promote_attribute sum should produce same values as HierarchicalPacker."""
        from nexpresso import HierarchicalPacker

        # Normalized
        npacker = NormalizedPacker(
            two_level_spec,
            tables={"region": regions_df, "store": stores_df},
        )
        norm_result = (
            npacker.promote_attribute(
                "revenue",
                from_level="store",
                to_level="region",
                agg="sum",
                alias="total_rev",
            )
            .collect()
            .sort("region.id")
        )

        # Nested: build_from_tables packs to root, so we need to first unpack
        # to store level, then use promote_attribute to go from store→region.
        # But promote_attribute needs from_level packed. Let's pack to store
        # level (which keeps stores as rows with region keys), then promote.
        hpacker = HierarchicalPacker(two_level_spec)
        nested = hpacker.build_from_tables(
            {"region": regions_df, "store": stores_df},
            target_level="store",
        )
        hier_result = hpacker.promote_attribute(
            nested,
            "revenue",
            from_level="store",
            to_level="region",
            agg="sum",
            alias="total_rev",
        )
        if isinstance(hier_result, pl.LazyFrame):
            hier_result = hier_result.collect()
        hier_result = hier_result.sort("region.id")

        # Compare the promoted column
        assert (
            norm_result["region.total_rev"].to_list()
            == hier_result["region.total_rev"].to_list()
        )

    def test_three_level_collect_matches(
        self,
        three_level_spec: HierarchySpec,
        regions_df: pl.DataFrame,
        stores_df: pl.DataFrame,
        products_df: pl.DataFrame,
    ) -> None:
        """Three-level collect matches build_from_tables + unpack."""
        from nexpresso import HierarchicalPacker

        tables = {"region": regions_df, "store": stores_df, "product": products_df}

        npacker = NormalizedPacker(three_level_spec, tables=tables)
        norm_result = (
            npacker.collect("product")
            .collect()
            .sort("region.store.product.id")
        )

        hpacker = HierarchicalPacker(three_level_spec)
        nested = hpacker.build_from_tables(tables)
        hier_result = hpacker.unpack(nested, "product")
        if isinstance(hier_result, pl.LazyFrame):
            hier_result = hier_result.collect()
        hier_result = hier_result.sort("region.store.product.id")

        shared_cols = sorted(
            set(norm_result.columns) & set(hier_result.columns)
        )
        assert len(shared_cols) > 0
        assert_frame_equal(
            norm_result.select(shared_cols),
            hier_result.select(shared_cols),
        )

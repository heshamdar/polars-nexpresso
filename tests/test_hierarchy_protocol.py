"""Cross-backend equivalence tests.

These tests verify that both :class:`NestedBackend` (wrapping
:class:`HierarchicalPacker`) and :class:`NormalizedPacker` produce
identical results for the same operations on the same data, validating
that the normalized backend is a correct alternative to physical nesting.

Both backends accept the SAME Polars expression syntax — conditions use
``pl.element().struct.field()`` — no separate DSL is needed.
"""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from nexpresso import (
    HierarchicalPacker,
    HierarchySpec,
    LevelAttribute,
    LevelSpec,
    NormalizedPacker,
)
from nexpresso.hierarchy_protocol import NestedBackend

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def spec() -> HierarchySpec:
    return HierarchySpec.from_levels(
        LevelSpec(name="region", id_fields=["id"]),
        LevelSpec(name="store", id_fields=["id"], parent_keys=["region_id"]),
    )


@pytest.fixture
def regions_df() -> pl.DataFrame:
    return pl.DataFrame({
        "id": ["r1", "r2", "r3"],
        "name": ["North", "South", "West"],
    })


@pytest.fixture
def stores_df() -> pl.DataFrame:
    return pl.DataFrame({
        "id": ["s1", "s2", "s3", "s4"],
        "name": ["Alpha", "Beta", "Gamma", "Delta"],
        "revenue": [100_000, 200_000, 150_000, 300_000],
        "region_id": ["r1", "r1", "r2", "r3"],
    })


@pytest.fixture
def nested_backend(
    spec: HierarchySpec,
    regions_df: pl.DataFrame,
    stores_df: pl.DataFrame,
) -> NestedBackend:
    packer = HierarchicalPacker(spec)
    packed = packer.build_from_tables(
        {"region": regions_df, "store": stores_df}
    )
    return NestedBackend(packer, packed)


@pytest.fixture
def normalized_backend(
    spec: HierarchySpec,
    regions_df: pl.DataFrame,
    stores_df: pl.DataFrame,
) -> NormalizedPacker:
    return NormalizedPacker(
        spec,
        tables={"region": regions_df, "store": stores_df},
    )


# ============================================================================
# Introspection equivalence
# ============================================================================


class TestIntrospectionEquivalence:
    """Verify introspection methods return identical results."""

    def test_level_names(
        self, nested_backend: NestedBackend, normalized_backend: NormalizedPacker
    ) -> None:
        assert nested_backend.level_names == normalized_backend.level_names

    def test_root_level(
        self, nested_backend: NestedBackend, normalized_backend: NormalizedPacker
    ) -> None:
        assert nested_backend.root_level == normalized_backend.root_level

    def test_leaf_level(
        self, nested_backend: NestedBackend, normalized_backend: NormalizedPacker
    ) -> None:
        assert nested_backend.leaf_level == normalized_backend.leaf_level

    def test_ancestor_levels(
        self, nested_backend: NestedBackend, normalized_backend: NormalizedPacker
    ) -> None:
        assert (
            nested_backend.get_ancestor_levels("store")
            == normalized_backend.get_ancestor_levels("store")
        )

    def test_descendant_levels(
        self, nested_backend: NestedBackend, normalized_backend: NormalizedPacker
    ) -> None:
        assert (
            nested_backend.get_descendant_levels("region")
            == normalized_backend.get_descendant_levels("region")
        )

    def test_level_keys(
        self, nested_backend: NestedBackend, normalized_backend: NormalizedPacker
    ) -> None:
        assert (
            nested_backend.get_level_keys("store")
            == normalized_backend.get_level_keys("store")
        )
        assert (
            nested_backend.get_level_keys("region", form="long")
            == normalized_backend.get_level_keys("region", form="long")
        )


# ============================================================================
# collect() equivalence
# ============================================================================


class TestCollectEquivalence:
    """Verify that collect() produces same data from both backends."""

    def test_collect_at_root(
        self, nested_backend: NestedBackend, normalized_backend: NormalizedPacker
    ) -> None:
        nested_result = nested_backend.collect("region").collect().sort("region.id")
        norm_result = normalized_backend.collect("region").collect().sort("region.id")

        shared = sorted(set(nested_result.columns) & set(norm_result.columns))
        assert len(shared) > 0
        assert_frame_equal(
            nested_result.select(shared), norm_result.select(shared)
        )

    def test_collect_at_leaf(
        self, nested_backend: NestedBackend, normalized_backend: NormalizedPacker
    ) -> None:
        nested_result = (
            nested_backend.collect("store").collect().sort("region.store.id")
        )
        norm_result = (
            normalized_backend.collect("store").collect().sort("region.store.id")
        )

        shared = sorted(set(nested_result.columns) & set(norm_result.columns))
        assert len(shared) > 0
        assert_frame_equal(
            nested_result.select(shared), norm_result.select(shared)
        )


# ============================================================================
# promote_attribute() equivalence
# ============================================================================


class TestPromoteAttributeEquivalence:
    """Verify promote_attribute produces same aggregations from both backends."""

    @pytest.mark.parametrize(
        "agg", ["sum", "mean", "min", "max", "count", "first", "last"]
    )
    def test_promote_revenue_aggs(
        self,
        nested_backend: NestedBackend,
        normalized_backend: NormalizedPacker,
        agg: str,
    ) -> None:
        alias = f"agg_{agg}"

        nested_result = (
            nested_backend.promote_attribute(
                "revenue",
                from_level="store",
                to_level="region",
                agg=agg,
                alias=alias,
            )
            .collect()
            .sort("region.id")
        )

        norm_result = (
            normalized_backend.promote_attribute(
                "revenue",
                from_level="store",
                to_level="region",
                agg=agg,
                alias=alias,
            )
            .collect()
            .sort("region.id")
        )

        col = f"region.{alias}"
        assert col in nested_result.columns, f"{col} not in nested: {nested_result.columns}"
        assert col in norm_result.columns, f"{col} not in norm: {norm_result.columns}"
        assert nested_result[col].to_list() == norm_result[col].to_list()


# ============================================================================
# any_child_satisfies() equivalence
# ============================================================================


class TestAnyChildSatisfiesEquivalence:
    """Verify existential filters produce same results from both backends.

    Both backends receive the EXACT SAME pl.Expr — no DSL translation needed.
    """

    def test_revenue_filter(
        self, nested_backend: NestedBackend, normalized_backend: NormalizedPacker
    ) -> None:
        condition = pl.element().struct.field("revenue") > 180_000

        nested_result = (
            nested_backend.any_child_satisfies(
                from_level="store", to_level="region", condition=condition
            )
            .collect()
            .sort("region.id")
        )

        norm_result = (
            normalized_backend.any_child_satisfies(
                from_level="store", to_level="region", condition=condition
            )
            .collect()
            .sort("region.id")
        )

        assert nested_result["region.id"].to_list() == norm_result["region.id"].to_list()

    def test_compound_condition(
        self, nested_backend: NestedBackend, normalized_backend: NormalizedPacker
    ) -> None:
        condition = (
            (pl.element().struct.field("revenue") >= 100_000)
            & (pl.element().struct.field("name") == "Alpha")
        )

        nested_ids = (
            nested_backend.any_child_satisfies(
                from_level="store", to_level="region", condition=condition
            )
            .collect()
            .sort("region.id")["region.id"]
            .to_list()
        )

        norm_ids = (
            normalized_backend.any_child_satisfies(
                from_level="store", to_level="region", condition=condition
            )
            .collect()
            .sort("region.id")["region.id"]
            .to_list()
        )

        assert nested_ids == norm_ids


# ============================================================================
# all_children_satisfy() equivalence
# ============================================================================


class TestAllChildrenSatisfyEquivalence:
    """Verify universal filters produce same results from both backends."""

    def test_all_revenue_above_threshold(
        self, nested_backend: NestedBackend, normalized_backend: NormalizedPacker
    ) -> None:
        condition = pl.element().struct.field("revenue") > 120_000

        nested_ids = sorted(
            nested_backend.all_children_satisfy(
                from_level="store", to_level="region", condition=condition
            )
            .collect()["region.id"]
            .to_list()
        )

        norm_ids = sorted(
            normalized_backend.all_children_satisfy(
                from_level="store", to_level="region", condition=condition
            )
            .collect()["region.id"]
            .to_list()
        )

        assert nested_ids == norm_ids


# ============================================================================
# enrich() equivalence
# ============================================================================


class TestEnrichEquivalence:
    """Verify batch enrichment produces same results from both backends."""

    def test_multiple_enrichments(
        self, nested_backend: NestedBackend, normalized_backend: NormalizedPacker
    ) -> None:
        specs = (
            LevelAttribute("revenue", "store", "sum", alias="total_rev"),
            LevelAttribute("revenue", "store", "count", alias="store_count"),
        )

        nested_result = (
            nested_backend.enrich(*specs, at_level="region")
            .collect()
            .sort("region.id")
        )

        norm_result = (
            normalized_backend.enrich(*specs, at_level="region")
            .collect()
            .sort("region.id")
        )

        for col in ["region.total_rev", "region.store_count"]:
            assert col in nested_result.columns
            assert col in norm_result.columns
            assert nested_result[col].to_list() == norm_result[col].to_list()


# ============================================================================
# Expression equivalence (nested and flat produce same filter results)
# ============================================================================


class TestExpressionEquivalence:
    """Verify that the same pl.Expr produces equivalent results on nested
    and flat data — proving the translation is correct."""

    def test_filter_same_result(self) -> None:
        """pl.element().struct.field("x") > 3 filters identically."""
        data = [{"x": 1, "y": "a"}, {"x": 5, "y": "b"}, {"x": 3, "y": "c"}]

        # Flat (via translation)
        from nexpresso.normalized_packer import _translate_nested_to_flat

        flat_df = pl.DataFrame(data)
        cond = pl.element().struct.field("x") > 3
        flat_cond = _translate_nested_to_flat(cond)
        flat_result = flat_df.filter(flat_cond).sort("x")

        # Nested — apply the original expression inside list.eval
        nested_df = pl.DataFrame({"items": [data]})
        nested_result = (
            nested_df.select(
                pl.col("items").list.eval(
                    pl.when(cond).then(pl.element()).otherwise(None)
                )
            )
            .explode("items")
            .drop_nulls()
            .unnest("items")
            .sort("x")
        )

        assert_frame_equal(flat_result, nested_result)

    def test_arithmetic_same_result(self) -> None:
        """Arithmetic expressions translate correctly."""
        from nexpresso.normalized_packer import _translate_nested_to_flat

        data = [{"a": 2, "b": 3}, {"a": 4, "b": 5}]

        flat_df = pl.DataFrame(data)
        expr = pl.element().struct.field("a") * pl.element().struct.field("b")
        flat_expr = _translate_nested_to_flat(expr)
        flat_vals = flat_df.select(flat_expr.alias("product"))["product"].to_list()

        nested_df = pl.DataFrame({"items": [data]})
        nested_vals = nested_df.select(
            pl.col("items").list.eval(expr)
        )["items"].to_list()[0]

        assert flat_vals == nested_vals


# ============================================================================
# apply() equivalence
# ============================================================================


class TestApplyEquivalence:
    """Verify that apply() produces equivalent results from both backends."""

    @pytest.fixture
    def spec_with_cost(self) -> HierarchySpec:
        return HierarchySpec.from_levels(
            LevelSpec(name="region", id_fields=["id"]),
            LevelSpec(name="store", id_fields=["id"], parent_keys=["region_id"]),
        )

    @pytest.fixture
    def nested_with_cost(self, spec_with_cost: HierarchySpec) -> NestedBackend:
        packer = HierarchicalPacker(spec_with_cost)
        regions = pl.DataFrame({"id": ["r1", "r2"], "name": ["North", "South"]})
        stores = pl.DataFrame({
            "id": ["s1", "s2", "s3"],
            "name": ["A", "B", "C"],
            "revenue": [100, 200, 150],
            "cost": [30, 80, 60],
            "region_id": ["r1", "r1", "r2"],
        })
        packed = packer.build_from_tables(
            {"region": regions, "store": stores}
        )
        return NestedBackend(packer, packed)

    @pytest.fixture
    def normalized_with_cost(self, spec_with_cost: HierarchySpec) -> NormalizedPacker:
        regions = pl.DataFrame({"id": ["r1", "r2"], "name": ["North", "South"]})
        stores = pl.DataFrame({
            "id": ["s1", "s2", "s3"],
            "name": ["A", "B", "C"],
            "revenue": [100, 200, 150],
            "cost": [30, 80, 60],
            "region_id": ["r1", "r1", "r2"],
        })
        return NormalizedPacker(
            spec_with_cost, tables={"region": regions, "store": stores}
        )

    def test_lambda_equivalence(
        self,
        nested_with_cost: NestedBackend,
        normalized_with_cost: NormalizedPacker,
    ) -> None:
        """Lambda transformation produces same results on both backends."""
        fields = {"revenue": lambda x: x * 2}

        nested_result = (
            nested_with_cost.apply(fields, at_level="store")
            .collect()
            .sort("region.store.id")
        )
        norm_result = (
            normalized_with_cost.apply(fields, at_level="store")
            .collect()
            .sort("region.store.id")
        )

        assert (
            nested_result["region.store.revenue"].to_list()
            == norm_result["region.store.revenue"].to_list()
        )

    def test_pl_field_expr_equivalence(
        self,
        nested_with_cost: NestedBackend,
        normalized_with_cost: NormalizedPacker,
    ) -> None:
        """pl.field() expression produces same results on both backends."""
        fields = {"profit": pl.field("revenue") - pl.field("cost")}

        nested_result = (
            nested_with_cost.apply(fields, at_level="store")
            .collect()
            .sort("region.store.id")
        )
        norm_result = (
            normalized_with_cost.apply(fields, at_level="store")
            .collect()
            .sort("region.store.id")
        )

        assert (
            nested_result["region.store.profit"].to_list()
            == norm_result["region.store.profit"].to_list()
        )

    def test_root_level_equivalence(
        self,
        nested_with_cost: NestedBackend,
        normalized_with_cost: NormalizedPacker,
    ) -> None:
        """apply() at root level produces same results on both backends."""
        fields = {"name": lambda x: x.str.to_uppercase()}

        nested_result = (
            nested_with_cost.apply(fields, at_level="region")
            .collect()
            .sort("region.id")
        )
        norm_result = (
            normalized_with_cost.apply(fields, at_level="region")
            .collect()
            .sort("region.id")
        )

        assert (
            nested_result["region.name"].to_list()
            == norm_result["region.name"].to_list()
        )

    def test_nested_dict_equivalence(
        self,
        nested_with_cost: NestedBackend,
        normalized_with_cost: NormalizedPacker,
    ) -> None:
        """Nested dict spec produces same results on both backends."""
        fields: dict[str, object] = {
            "store": {"revenue": lambda x: x * 2},
        }

        nested_result = (
            nested_with_cost.apply(fields, at_level="region")
            .collect()
            .sort("region.store.id")
        )
        norm_result = (
            normalized_with_cost.apply(fields, at_level="region")
            .collect()
            .sort("region.store.id")
        )

        assert (
            nested_result["region.store.revenue"].to_list()
            == norm_result["region.store.revenue"].to_list()
        )

    def test_mixed_levels_equivalence(
        self,
        nested_with_cost: NestedBackend,
        normalized_with_cost: NormalizedPacker,
    ) -> None:
        """Mixed parent + child field spec matches across backends."""
        fields: dict[str, object] = {
            "name": lambda x: x.str.to_uppercase(),
            "store": {"revenue": lambda x: x * 3},
        }

        nested_result = (
            nested_with_cost.apply(fields, at_level="region")
            .collect()
            .sort("region.store.id")
        )
        norm_result = (
            normalized_with_cost.apply(fields, at_level="region")
            .collect()
            .sort("region.store.id")
        )

        assert sorted(nested_result["region.name"].unique().to_list()) == sorted(
            norm_result["region.name"].unique().to_list()
        )
        assert (
            nested_result["region.store.revenue"].to_list()
            == norm_result["region.store.revenue"].to_list()
        )

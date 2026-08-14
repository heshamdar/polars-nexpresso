"""
Tests for branching (multi-axis) hierarchies.

The fixtures in ``conftest.py`` describe a ``city`` with two independent child
branches::

    country > city > street > building
    country > city > service

``street`` and ``service`` are orthogonal properties of a city, not stages of
one chain. These tests cover how the spec is declared and validated, how
``pack`` / ``unpack`` traverse one axis while leaving the other packed, the
normalized round-trip, and how :class:`HierarchyView` routes across branches.
"""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from nexpresso import (
    HierarchicalPacker,
    HierarchySpec,
    HierarchyView,
    LevelSpec,
)

# ``pack`` groups without maintain_order, so top-level row order is not part of
# its contract; every comparison below is order-insensitive.
UNORDERED = {"check_row_order": False}


class TestSpecTree:
    """Declaring and validating a branching HierarchySpec."""

    def test_chain_parent_inference_unchanged(self):
        """A spec with no parent= is still read as a linear chain."""
        spec = HierarchySpec.from_levels(
            LevelSpec(name="country", id_fields=["code"]),
            LevelSpec(name="city", id_fields=["id"]),
            LevelSpec(name="street", id_fields=["id"]),
        )
        assert spec.parent_of("country") is None
        assert spec.parent_of("city") == "country"
        assert spec.parent_of("street") == "city"
        assert spec.children_of("city") == ["street"]
        assert spec.leaves == ["street"]

    def test_tree_navigation(self, branching_spec: HierarchySpec):
        assert branching_spec.root == "country"
        assert branching_spec.leaves == ["building", "service"]
        assert branching_spec.children_of("city") == ["street", "service"]
        assert branching_spec.children_of("building") == []
        assert branching_spec.parent_of("service") == "city"
        assert branching_spec.ancestors_of("building") == ["country", "city", "street"]
        assert branching_spec.ancestors_of("service") == ["country", "city"]
        assert branching_spec.axis_of("service") == ["country", "city", "service"]
        assert branching_spec.descendants_of("city") == ["street", "building", "service"]
        assert branching_spec.descendants_of("street") == ["building"]

    def test_siblings_are_not_ancestors(self, branching_spec: HierarchySpec):
        """The relation that every routing decision now turns on."""
        assert branching_spec.is_ancestor_of("city", "service")
        assert branching_spec.is_ancestor_of("city", "building")
        assert not branching_spec.is_ancestor_of("street", "service")
        assert not branching_spec.is_ancestor_of("service", "street")
        assert not branching_spec.is_ancestor_of("building", "service")

    def test_partial_parent_declaration_rejected(self):
        """Inferring the rest from declaration order is exactly the bug to avoid."""
        with pytest.raises(ValueError, match="all-or-nothing"):
            HierarchySpec.from_levels(
                LevelSpec(name="country", id_fields=["code"]),
                LevelSpec(name="city", id_fields=["id"], parent="country"),
                LevelSpec(name="street", id_fields=["id"]),
            )

    def test_unknown_parent_rejected(self):
        with pytest.raises(ValueError, match="unknown parent 'region'"):
            HierarchySpec.from_levels(
                LevelSpec(name="country", id_fields=["code"]),
                LevelSpec(name="city", id_fields=["id"], parent="region"),
            )

    def test_self_parent_rejected(self):
        with pytest.raises(ValueError, match="cannot be its own parent"):
            HierarchySpec.from_levels(
                LevelSpec(name="country", id_fields=["code"]),
                LevelSpec(name="city", id_fields=["id"], parent="city"),
            )

    def test_child_declared_before_parent_rejected(self):
        with pytest.raises(ValueError, match="declared before its parent"):
            HierarchySpec.from_levels(
                LevelSpec(name="country", id_fields=["code"]),
                LevelSpec(name="building", id_fields=["id"], parent="street"),
                LevelSpec(name="street", id_fields=["id"], parent="country"),
            )

    def test_root_with_parent_rejected(self):
        with pytest.raises(ValueError, match="must not have one"):
            HierarchySpec.from_levels(
                LevelSpec(name="country", id_fields=["code"], parent="world"),
                LevelSpec(name="city", id_fields=["id"], parent="country"),
            )

    def test_parent_keys_validated_against_declared_parent(self):
        """The arity check follows parent=, not the level declared before."""
        with pytest.raises(ValueError, match="parent_keys"):
            HierarchySpec.from_levels(
                LevelSpec(name="country", id_fields=["code"]),
                LevelSpec(name="city", id_fields=["id", "region"], parent="country"),
                LevelSpec(name="street", id_fields=["id"], parent="city", parent_keys=["id"]),
                # Declared after `street`, but its parent is `city` (2 id_fields).
                LevelSpec(name="service", id_fields=["kind"], parent="city", parent_keys=["id"]),
            )

    def test_next_level_ambiguous_on_branch(self, branching_spec: HierarchySpec):
        assert branching_spec.next_level("street").name == "building"
        assert branching_spec.next_level("building") is None
        with pytest.raises(ValueError, match="ambiguous"):
            branching_spec.next_level("city")


class TestPackerIntrospection:
    """Packer-level views of the tree."""

    def test_leaf_level_ambiguous(self, branching_packer: HierarchicalPacker):
        assert branching_packer.leaf_levels == ["building", "service"]
        with pytest.raises(ValueError, match="leaf levels"):
            _ = branching_packer.leaf_level

    def test_axes(self, branching_packer: HierarchicalPacker):
        assert branching_packer.axes == [
            ["country", "city", "street", "building"],
            ["country", "city", "service"],
        ]
        assert branching_packer.get_axis("service") == ["country", "city", "service"]

    def test_paths_extend_the_parent_not_the_previous_level(
        self, branching_packer: HierarchicalPacker
    ):
        """``service`` is declared last but hangs off ``city``, not ``building``."""
        meta = {m.name: m for m in branching_packer._levels_meta}  # noqa: SLF001
        assert meta["service"].path == "country.city.service"
        assert meta["building"].path == "country.city.street.building"
        assert meta["service"].ancestor_keys == ("country.code", "country.city.id")

    def test_level_keys_follow_the_axis(self, branching_packer: HierarchicalPacker):
        assert branching_packer.get_level_keys("service", form="long") == [
            "country.city.service.kind"
        ]
        assert branching_packer.get_level_keys("service", include_ancestors=True) == [
            "country.code",
            "country.city.id",
            "country.city.service.kind",
        ]

    def test_describe_marks_every_leaf(self, branching_packer: HierarchicalPacker):
        description = branching_packer.describe()
        assert "building  (leaf)" in description
        assert "service  (leaf)" in description
        assert "country  (root)" in description
        assert "Branches: street, service" in description


class TestUnpackLeavesSiblingsPacked:
    """unpack() walks one axis; the other branch rides along nested."""

    def test_building_axis(
        self, branching_packer: HierarchicalPacker, branching_nested: pl.DataFrame
    ):
        flat = branching_packer.unpack(branching_nested, "building")

        assert "country.city.street.building.id" in flat.columns
        assert "country.city.service" in flat.columns
        assert isinstance(flat.schema["country.city.service"], pl.List)
        # Not exploded: no flat service columns at all.
        assert not [c for c in flat.columns if c.startswith("country.city.service.")]
        assert flat.height == 4

        nyc = flat.filter(pl.col("country.city.id") == "NYC")
        assert nyc.height == 2
        # The city's services are replicated onto every building row, intact.
        for services in nyc["country.city.service"].to_list():
            assert [s["kind"] for s in services] == ["police", "fire"]

    def test_service_axis(
        self, branching_packer: HierarchicalPacker, branching_nested: pl.DataFrame
    ):
        flat = branching_packer.unpack(branching_nested, "service")

        assert "country.city.service.kind" in flat.columns
        assert "country.city.street" in flat.columns
        assert isinstance(flat.schema["country.city.street"], pl.List)
        assert not [c for c in flat.columns if c.startswith("country.city.street.")]
        assert flat.height == 4
        assert sorted(flat["country.city.service.kind"].to_list()) == [
            "fire",
            "medical",
            "police",
            "water",
        ]

    def test_unpack_to_the_branch_point_leaves_both_packed(
        self, branching_packer: HierarchicalPacker, branching_nested: pl.DataFrame
    ):
        flat = branching_packer.unpack(branching_nested, "city")
        assert flat.height == 3
        assert isinstance(flat.schema["country.city.street"], pl.List)
        assert isinstance(flat.schema["country.city.service"], pl.List)


class TestPack:
    """pack() folds every branch of the subtree it targets."""

    def test_city_struct_carries_both_branches(
        self, branching_packer: HierarchicalPacker, branching_nested: pl.DataFrame
    ):
        city_fields = branching_nested.schema["country.city"].inner.to_schema()
        assert "street" in city_fields
        assert "service" in city_fields
        assert isinstance(city_fields["street"], pl.List)
        assert isinstance(city_fields["service"], pl.List)

    def test_repack_round_trips_each_axis(
        self, branching_packer: HierarchicalPacker, branching_nested: pl.DataFrame
    ):
        for axis_leaf in ("building", "service"):
            flat = branching_packer.unpack(branching_nested, axis_leaf)
            repacked = branching_packer.pack(flat, "country")
            assert_frame_equal(repacked, branching_nested, **UNORDERED)

    def test_pack_to_a_level_below_the_branch_point_keeps_the_sibling(
        self, branching_packer: HierarchicalPacker, branching_nested: pl.DataFrame
    ):
        flat = branching_packer.unpack(branching_nested, "building")
        packed = branching_packer.pack(flat, "street")
        # Rows are at street granularity; the service branch is still there.
        assert "country.city.service" in packed.columns
        assert isinstance(packed.schema["country.city.street.building"], pl.List)

    def test_infer_current_level_per_axis(
        self, branching_packer: HierarchicalPacker, branching_nested: pl.DataFrame
    ):
        assert branching_packer.infer_current_level(branching_nested) == "country"
        building_flat = branching_packer.unpack(branching_nested, "building")
        service_flat = branching_packer.unpack(branching_nested, "service")
        assert branching_packer.infer_current_level(building_flat) == "building"
        assert branching_packer.infer_current_level(service_flat) == "service"
        # Restricting to an axis measures only that branch.
        assert branching_packer.infer_current_level(building_flat, axis="service") == "city"


class TestNormalizedRoundTrip:
    """split_levels / normalize / denormalize on a tree."""

    def test_split_levels_emits_one_table_per_level(
        self, branching_packer: HierarchicalPacker, branching_nested: pl.DataFrame
    ):
        tables = branching_packer.split_levels(branching_nested)
        assert set(tables) == {"country", "city", "street", "building", "service"}
        assert tables["service"].columns == [
            "country.code",
            "country.city.id",
            "country.city.service.kind",
            "country.city.service.budget",
        ]
        # A branch table carries its ancestors' keys and nothing from its sibling.
        assert not [c for c in tables["service"].columns if "street" in c]
        assert tables["service"].height == 4
        assert tables["building"].height == 4

    def test_split_levels_deduplicates_a_branch_packed_inside_a_finer_frame(
        self, branching_packer: HierarchicalPacker, branching_nested: pl.DataFrame
    ):
        """
        In a building-granularity frame the service list is replicated per row.
        Exploding it must not multiply NYC's two services by its two buildings.
        """
        flat = branching_packer.unpack(branching_nested, "building")
        tables = branching_packer.split_levels(flat)
        assert tables["service"].height == 4

    @pytest.mark.parametrize(
        ("level", "axis"),
        [
            ("country", "building"),
            ("city", "building"),
            ("street", "building"),
            ("building", "building"),
            ("city", "service"),
            ("service", "service"),
        ],
    )
    def test_denormalize_inverts_normalize(
        self,
        branching_packer: HierarchicalPacker,
        branching_nested: pl.DataFrame,
        level: str,
        axis: str,
    ):
        """
        The round-trip identity holds per axis: the source frame must be flat
        along the axis containing ``level``, since a flat frame carries only one.
        """
        flat = branching_packer.unpack(branching_nested, axis)
        got = branching_packer.denormalize(
            branching_packer.normalize(flat, at_level=level), at_level=level
        )
        want = branching_packer.pack(flat, level)
        assert got.columns == want.columns
        assert_frame_equal(got, want, **UNORDERED)  # type: ignore[arg-type]

    def test_denormalize_below_the_branch_point_keeps_the_sibling(
        self, branching_packer: HierarchicalPacker, branching_tables: dict[str, pl.DataFrame]
    ):
        """
        A target under ``street`` must still carry ``city``'s service branch —
        ``pack`` does, so the inverse has to as well.
        """
        result = branching_packer.denormalize(branching_tables, at_level="street")
        assert "country.city.service" in result.columns
        assert isinstance(result.schema["country.city.service"], pl.List)

    def test_denormalize_tolerates_a_missing_branch(
        self, branching_packer: HierarchicalPacker, branching_tables: dict[str, pl.DataFrame]
    ):
        """A branch off the target's axis is optional."""
        without_service = {k: v for k, v in branching_tables.items() if k != "service"}
        result = branching_packer.denormalize(without_service, at_level="building")
        assert "country.city.service" not in result.columns

    def test_denormalize_requires_the_target_axis(
        self, branching_packer: HierarchicalPacker, branching_tables: dict[str, pl.DataFrame]
    ):
        without_street = {k: v for k, v in branching_tables.items() if k != "street"}
        with pytest.raises(Exception, match="Missing table for .*'street'"):
            branching_packer.denormalize(without_street, at_level="building")


class TestBuildFromTables:
    """Independent source tables assembled into a branching hierarchy."""

    def test_both_branches_appear(self, branching_packer: HierarchicalPacker):
        tables = {
            "country": pl.DataFrame({"code": ["US"], "name": ["USA"]}),
            "city": pl.DataFrame({"id": ["NYC"], "population": [8], "code": ["US"]}),
            "street": pl.DataFrame({"id": ["s1"], "length": [100], "city_id": ["NYC"]}),
            "building": pl.DataFrame({"id": ["b1"], "floors": [10], "street_id": ["s1"]}),
            "service": pl.DataFrame({"kind": ["police"], "budget": [100], "city_id": ["NYC"]}),
        }
        spec = HierarchySpec.from_levels(
            LevelSpec(name="country", id_fields=["code"]),
            LevelSpec(name="city", id_fields=["id"], parent="country", parent_keys=["code"]),
            LevelSpec(name="street", id_fields=["id"], parent="city", parent_keys=["city_id"]),
            LevelSpec(
                name="building", id_fields=["id"], parent="street", parent_keys=["street_id"]
            ),
            LevelSpec(name="service", id_fields=["kind"], parent="city", parent_keys=["city_id"]),
        )
        result = HierarchicalPacker(spec).build_from_tables(tables, at_level="country")
        city = result.schema["country.city"].inner.to_schema()  # type: ignore[union-attr]
        assert "street" in city
        assert "service" in city

    def test_missing_branch_table_is_allowed(self, branching_packer: HierarchicalPacker):
        """Only the target's own axis is required."""
        tables = {
            "country": pl.DataFrame({"code": ["US"], "name": ["USA"]}),
            "city": pl.DataFrame({"id": ["NYC"], "population": [8], "code": ["US"]}),
        }
        spec = HierarchySpec.from_levels(
            LevelSpec(name="country", id_fields=["code"]),
            LevelSpec(name="city", id_fields=["id"], parent="country", parent_keys=["code"]),
            LevelSpec(name="street", id_fields=["id"], parent="city", parent_keys=["city_id"]),
            LevelSpec(name="service", id_fields=["kind"], parent="city", parent_keys=["city_id"]),
        )
        result = HierarchicalPacker(spec).build_from_tables(tables, at_level="city")
        assert result.height == 1  # type: ignore[union-attr]


class TestCrossLevelAttributes:
    """promote_attribute / attribute_expr across a branch."""

    def test_promote_from_the_second_branch(
        self, branching_packer: HierarchicalPacker, branching_nested: pl.DataFrame
    ):
        packed = branching_packer.pack(branching_packer.unpack(branching_nested, "service"), "city")
        result = branching_packer.promote_attribute(
            packed,
            "budget",
            from_level="service",
            to_level="city",
            agg="sum",
            alias="total_budget",
        )
        totals = dict(
            zip(
                result["country.city.id"].to_list(),
                result["country.city.total_budget"].to_list(),
            )
        )
        assert totals == {"NYC": 300, "LA": 300, "PAR": 400}

    def test_attribute_expr_descends_one_branch(
        self, branching_packer: HierarchicalPacker, branching_nested: pl.DataFrame
    ):
        packed = branching_packer.pack(
            branching_packer.unpack(branching_nested, "building"), "country"
        )
        expr = branching_packer.attribute_expr("budget", "service", "country", "sum")
        # Rows are at country granularity: US = 100+200+300, FR = 400.
        assert sorted(packed.select(expr.alias("total"))["total"].to_list()) == [400, 600]

    def test_attribute_expr_across_branches_rejected(self, branching_packer: HierarchicalPacker):
        with pytest.raises(ValueError, match="different branch"):
            branching_packer.attribute_expr("budget", "service", "street", "sum")

    def test_promote_across_branches_rejected(self, branching_packer: HierarchicalPacker):
        with pytest.raises(ValueError, match="immediate child"):
            branching_packer.promote_attribute(
                pl.DataFrame(), "budget", from_level="service", to_level="street"
            )


class TestBranchingView:
    """HierarchyView over a branching hierarchy."""

    @pytest.fixture
    def view(
        self, branching_packer: HierarchicalPacker, branching_tables: dict[str, pl.DataFrame]
    ) -> HierarchyView:
        return HierarchyView.from_tables(branching_tables, branching_packer)

    def test_to_flat_joins_only_one_axis(self, view: HierarchyView):
        building = view.to_flat("building").collect_schema().names()
        assert "country.city.street.building.id" in building
        assert not [c for c in building if "service" in c]

        service = view.to_flat("service").collect_schema().names()
        assert "country.city.service.kind" in service
        assert not [c for c in service if "street" in c]

    def test_to_flat_without_a_level_is_ambiguous(self, view: HierarchyView):
        with pytest.raises(ValueError, match="leaf levels"):
            view.to_flat()

    def test_filter_on_one_branch_cascades_to_the_other(self, view: HierarchyView):
        """
        Filtering ``service`` prunes cities, and those pruned cities must in turn
        prune ``street`` and ``building`` — a branch the first downward pass
        never touched.
        """
        hot = view.filter(pl.col("country.city.service.budget") >= 300)
        counts = {name: lf.collect().height for name, lf in hot.tables().items()}
        assert counts == {"country": 2, "city": 2, "street": 2, "building": 2, "service": 2}
        assert sorted(hot.tables()["city"].collect()["country.city.id"].to_list()) == [
            "LA",
            "PAR",
        ]

    def test_filter_keeps_empty_parents_when_asked(
        self, branching_packer: HierarchicalPacker, branching_tables: dict[str, pl.DataFrame]
    ):
        keep = HierarchyView.from_tables(
            branching_tables, branching_packer, empty_parents="keep"
        ).filter(pl.col("country.city.service.budget") >= 300)
        assert keep.tables()["city"].collect().height == 3

    def test_cross_branch_predicate_rejected(self, view: HierarchyView):
        with pytest.raises(ValueError, match="different branches"):
            view.filter(
                pl.col("country.city.street.length") > pl.col("country.city.service.budget")
            )

    def test_cross_branch_borrow_rejected(self, view: HierarchyView):
        with pytest.raises(ValueError, match="not an ancestor"):
            view.with_columns(
                (pl.col("country.city.service.budget") * 2).alias("country.city.street.doubled")
            )

    def test_ancestor_borrow_still_works(self, view: HierarchyView):
        widened = view.with_columns(
            (pl.col("country.city.population") * 2).alias("country.city.service.scaled")
        )
        rows = widened.tables()["service"].collect()
        assert rows["country.city.service.scaled"].to_list() == [16, 16, 8, 4]

    def test_promote_from_the_second_branch(self, view: HierarchyView):
        promoted = view.promote(
            "budget", from_level="service", to_level="city", agg="sum", alias="total_budget"
        )
        rows = promoted.tables()["city"].collect().sort("country.city.id")
        assert rows["country.city.total_budget"].to_list() == [300, 300, 400]

    def test_any_child_satisfies_on_the_second_branch(self, view: HierarchyView):
        matched = view.any_child_satisfies(
            pl.col("country.city.service.kind") == "water",
            at_level="city",
            child_level="service",
        )
        assert matched.tables()["city"].collect()["country.city.id"].to_list() == ["LA"]

    def test_any_child_satisfies_across_branches_rejected(self, view: HierarchyView):
        with pytest.raises(ValueError, match="must be finer"):
            view.any_child_satisfies(
                pl.col("country.city.service.kind") == "water",
                at_level="street",
                child_level="service",
            )

    def test_nested_matches_the_packer(
        self,
        view: HierarchyView,
        branching_packer: HierarchicalPacker,
        branching_tables: dict[str, pl.DataFrame],
    ):
        direct = branching_packer.denormalize(branching_tables, at_level="country")
        assert_frame_equal(
            view.collect_nested(),
            direct,  # type: ignore[arg-type]
            **UNORDERED,
        )

    def test_roundtrip_through_parquet(self, view: HierarchyView, tmp_path):
        view.sink_parquet(tmp_path)
        restored = HierarchyView.scan_parquet(tmp_path, view._packer)  # noqa: SLF001
        assert_frame_equal(
            restored.to_flat("service").collect().sort("country.city.service.kind"),
            view.to_flat("service").collect().sort("country.city.service.kind"),
        )
        assert_frame_equal(
            restored.to_flat("building").collect().sort("country.city.street.building.id"),
            view.to_flat("building").collect().sort("country.city.street.building.id"),
        )

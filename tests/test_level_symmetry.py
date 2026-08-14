"""
Tests that a level argument means the same thing everywhere.

``pack(df, L)`` and ``unpack(df, L)`` both produce a frame whose rows are ``L``
entities: everything below ``L`` is nested, everything at ``L`` and above stays
flat. Before 0.8.0 ``pack`` named the level that got *nested* instead, so it
landed one level coarser than ``unpack`` and ``infer_current_level(pack(df, L))``
was never ``L``.
"""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from nexpresso import HierarchicalPacker, HierarchySpec, HierarchyView, LevelSpec

LINEAR_LEVELS = ["country", "city", "street"]


@pytest.fixture
def linear_packer() -> HierarchicalPacker:
    return HierarchicalPacker(
        HierarchySpec.from_levels(
            LevelSpec(name="country", id_fields=["code"]),
            LevelSpec(name="city", id_fields=["id"], parent_keys=["code"]),
            LevelSpec(name="street", id_fields=["id"], parent_keys=["id"]),
        )
    )


@pytest.fixture
def linear_flat() -> pl.DataFrame:
    """2 countries, 3 cities, 4 streets — every level has a distinct row count."""
    return pl.DataFrame(
        {
            "country.code": ["US", "US", "US", "FR"],
            "country.name": ["USA", "USA", "USA", "France"],
            "country.city.id": ["NYC", "NYC", "LA", "PAR"],
            "country.city.population": [8, 8, 4, 2],
            "country.city.street.id": ["s1", "s2", "s3", "s4"],
            "country.city.street.length": [100, 200, 300, 400],
        }
    )


HEIGHTS = {"country": 2, "city": 3, "street": 4}


class TestRowGranularity:
    """The level argument names the granularity of the resulting rows."""

    @pytest.mark.parametrize("level", LINEAR_LEVELS)
    def test_pack_gives_one_row_per_entity(
        self, linear_packer: HierarchicalPacker, linear_flat: pl.DataFrame, level: str
    ):
        assert linear_packer.pack(linear_flat, level).height == HEIGHTS[level]

    @pytest.mark.parametrize("level", LINEAR_LEVELS)
    def test_infer_current_level_inverts_pack(
        self, linear_packer: HierarchicalPacker, linear_flat: pl.DataFrame, level: str
    ):
        """
        The identity the old convention could not satisfy: packing to a level and
        asking what granularity the result is at must give back that same level.
        """
        packed = linear_packer.pack(linear_flat, level)
        assert linear_packer.infer_current_level(packed) == level

    @pytest.mark.parametrize("level", LINEAR_LEVELS)
    def test_pack_and_unpack_agree_at_the_same_level(
        self, linear_packer: HierarchicalPacker, linear_flat: pl.DataFrame, level: str
    ):
        """``pack`` and ``unpack`` are two routes to one granularity."""
        nested = linear_packer.pack(linear_flat, "country")
        from_pack = linear_packer.pack(linear_flat, level)
        from_unpack = linear_packer.unpack(nested, level)

        assert from_pack.columns == from_unpack.columns
        assert_frame_equal(
            from_pack.sort(from_pack.columns[0]),
            from_unpack.sort(from_unpack.columns[0]),
            check_row_order=False,
        )

    @pytest.mark.parametrize("level", LINEAR_LEVELS)
    def test_own_columns_stay_flat_and_descendants_nest(
        self, linear_packer: HierarchicalPacker, linear_flat: pl.DataFrame, level: str
    ):
        packed = linear_packer.pack(linear_flat, level)
        meta = {m.name: m for m in linear_packer._levels_meta}  # noqa: SLF001

        # The level's own id column is a flat column, not buried in a struct.
        for column in meta[level].id_columns:
            assert column in packed.columns
            assert not isinstance(packed.schema[column], (pl.List, pl.Struct))

        # Each child is a single nested column.
        for child in linear_packer.get_child_levels(level):
            assert isinstance(packed.schema[meta[child].path], pl.List)


class TestTheRootStaysFlat:
    """Packing to the root no longer wraps the hierarchy in one struct column."""

    def test_root_columns_are_top_level(
        self, linear_packer: HierarchicalPacker, linear_flat: pl.DataFrame
    ):
        packed = linear_packer.pack(linear_flat, "country")
        assert packed.columns == ["country.code", "country.name", "country.city"]
        assert packed.schema["country.code"] == pl.String
        assert isinstance(packed.schema["country.city"], pl.List)

    def test_no_column_is_named_for_the_root_alone(
        self, linear_packer: HierarchicalPacker, linear_flat: pl.DataFrame
    ):
        assert "country" not in linear_packer.pack(linear_flat, "country").columns


class TestRoundTrips:
    """pack and unpack undo each other."""

    @pytest.mark.parametrize("level", LINEAR_LEVELS)
    def test_unpack_after_pack_recovers_the_flat_frame(
        self, linear_packer: HierarchicalPacker, linear_flat: pl.DataFrame, level: str
    ):
        recovered = linear_packer.unpack(linear_packer.pack(linear_flat, level), "street")
        assert_frame_equal(
            recovered.select(sorted(recovered.columns)),
            linear_flat.select(sorted(linear_flat.columns)),
            check_row_order=False,
        )

    def test_pack_to_the_leaf_is_a_no_op(
        self, linear_packer: HierarchicalPacker, linear_flat: pl.DataFrame
    ):
        """A leaf-granularity frame is already flat, so there is nothing to fold."""
        assert_frame_equal(linear_packer.pack(linear_flat, "street"), linear_flat)

    @pytest.mark.parametrize("level", LINEAR_LEVELS)
    def test_packing_to_the_granularity_you_have_is_a_no_op(
        self, linear_packer: HierarchicalPacker, linear_flat: pl.DataFrame, level: str
    ):
        packed = linear_packer.pack(linear_flat, level)
        assert_frame_equal(linear_packer.pack(packed, level), packed)

    @pytest.mark.parametrize("level", LINEAR_LEVELS)
    def test_denormalize_inverts_normalize(
        self, linear_packer: HierarchicalPacker, linear_flat: pl.DataFrame, level: str
    ):
        rebuilt = linear_packer.denormalize(
            linear_packer.normalize(linear_flat, at_level=level), at_level=level
        )
        expected = linear_packer.pack(linear_flat, level)
        assert list(rebuilt.schema.items()) == list(expected.schema.items())  # type: ignore[union-attr]
        assert_frame_equal(rebuilt, expected, check_row_order=False)  # type: ignore[arg-type]


class TestColumnOrderIsCanonical:
    """Packing the same data by different routes gives the same column order."""

    def test_pack_order_does_not_depend_on_input_order(
        self, linear_packer: HierarchicalPacker, linear_flat: pl.DataFrame
    ):
        shuffled = linear_flat.select(sorted(linear_flat.columns))
        assert (
            linear_packer.pack(shuffled, "country").columns
            == linear_packer.pack(linear_flat, "country").columns
        )

    def test_split_join_matches_the_aggregate_strategy(
        self, linear_packer: HierarchicalPacker, linear_flat: pl.DataFrame
    ):
        assert_frame_equal(
            linear_packer.pack(linear_flat, "country", parent_strategy="split_join"),
            linear_packer.pack(linear_flat, "country"),
            check_row_order=False,
        )


class TestBranchingHierarchy:
    """The same rule holds per axis when a level carries several branches."""

    @pytest.mark.parametrize(
        ("level", "height"),
        [("country", 2), ("city", 3), ("street", 4), ("building", 4), ("service", 4)],
    )
    def test_pack_row_granularity(
        self,
        branching_packer: HierarchicalPacker,
        branching_nested: pl.DataFrame,
        level: str,
        height: int,
    ):
        axis_leaf = "service" if level == "service" else "building"
        flat = branching_packer.unpack(branching_nested, axis_leaf)
        assert branching_packer.pack(flat, level).height == height

    @pytest.mark.parametrize("level", ["country", "city", "street", "building"])
    def test_infer_current_level_inverts_pack(
        self, branching_packer: HierarchicalPacker, branching_nested: pl.DataFrame, level: str
    ):
        flat = branching_packer.unpack(branching_nested, "building")
        assert branching_packer.infer_current_level(branching_packer.pack(flat, level)) == level

    def test_pack_agrees_across_axes(
        self, branching_packer: HierarchicalPacker, branching_nested: pl.DataFrame
    ):
        """
        Both axes reach ``city`` granularity, and the shared levels must come out
        identical whichever branch the caller flattened first.
        """
        via_building = branching_packer.pack(
            branching_packer.unpack(branching_nested, "building"), "city"
        )
        via_service = branching_packer.pack(
            branching_packer.unpack(branching_nested, "service"), "city"
        )
        assert via_building.columns == via_service.columns
        assert_frame_equal(via_building, via_service, check_row_order=False)


class TestMigrationGuard:
    """The renamed level keywords raise rather than silently changing meaning."""

    @pytest.mark.parametrize(
        ("call", "old_kwarg"),
        [
            ("pack", "to_level"),
            ("unpack", "to_level"),
            ("pack_streaming", "to_level"),
            ("unpack_streaming", "to_level"),
            ("normalize", "root_level"),
            ("denormalize", "target_level"),
            ("build_from_tables", "target_level"),
        ],
    )
    def test_old_keyword_is_rejected(
        self,
        linear_packer: HierarchicalPacker,
        linear_flat: pl.DataFrame,
        call: str,
        old_kwarg: str,
    ):
        subject: object = linear_flat if call not in ("denormalize", "build_from_tables") else {}
        with pytest.raises(TypeError, match=f"'{old_kwarg}' was renamed to 'at_level'"):
            getattr(linear_packer, call)(subject, **{old_kwarg: "city"})

    def test_view_from_frame_rejects_root_level(
        self, linear_packer: HierarchicalPacker, linear_flat: pl.DataFrame
    ):
        with pytest.raises(TypeError, match="'root_level' was renamed to 'at_level'"):
            HierarchyView.from_frame(linear_flat, linear_packer, root_level="city")

    def test_unknown_keyword_still_reports_itself(
        self, linear_packer: HierarchicalPacker, linear_flat: pl.DataFrame
    ):
        """The guard must not swallow ordinary typos."""
        with pytest.raises(TypeError, match="unexpected keyword argument 'levl'"):
            linear_packer.pack(linear_flat, "city", levl="city")

    def test_missing_level_is_still_a_missing_argument(
        self, linear_packer: HierarchicalPacker, linear_flat: pl.DataFrame
    ):
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            linear_packer.pack(linear_flat)

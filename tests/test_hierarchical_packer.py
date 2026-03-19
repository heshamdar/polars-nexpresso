"""Tests for the hierarchical_packer module."""

from __future__ import annotations

import json

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from nexpresso.hierarchical_packer import (
    DiscoveredLevel,
    HierarchicalPacker,
    HierarchySpec,
    HierarchyValidationError,
    LevelAttribute,
    LevelSpec,
    SchemaValidationResult,
)

TEST_HIERARCHY = HierarchySpec(
    levels=[
        LevelSpec(name="country", id_fields=["code"]),
        LevelSpec(name="city", id_fields=["id", "name"]),
        LevelSpec(name="street", id_fields=["name"]),
        LevelSpec(name="building", id_fields=["number"]),
        LevelSpec(name="apartment", id_fields=["id"], required_fields=["id"]),
    ],
    key_aliases={"country.code": "country.city.id"},
)


@pytest.fixture()
def packer():
    return HierarchicalPacker(TEST_HIERARCHY)


@pytest.fixture()
def apartment_level_df():
    return pl.DataFrame(
        {
            "country.code": ["US", "US", "US", "CA"],
            "country.city.id": ["NYC", "NYC", "NYC", "TOR"],
            "country.city.name": ["New York", "New York", "New York", "Toronto"],
            "country.city.street.name": ["Main St", "Main St", "Main St", "Queen St"],
            "country.city.street.building.number": [100, 100, 101, 200],
            "country.city.street.building.id": [
                "bldg-100",
                "bldg-100",
                "bldg-101",
                "bldg-200",
            ],
            "country.city.street.building.apartment.id": [
                "apt-1",
                "apt-2",
                "apt-3",
                "apt-4",
            ],
            "country.city.street.building.apartment.area": [50.5, 75.0, 90.2, 60.8],
        }
    )


FrameLike = pl.DataFrame | pl.LazyFrame


def _materialize(df: FrameLike) -> pl.DataFrame:
    return df if isinstance(df, pl.DataFrame) else df.collect()


def _canonical_rows(df: FrameLike) -> list[str]:
    frame = _materialize(df)
    cols = sorted(frame.columns)
    ordered = frame.select(cols).sort(cols)
    return sorted(json.dumps(row, sort_keys=True) for row in ordered.to_dicts())


def _assert_same_rows(left: FrameLike, right: FrameLike):
    assert_frame_equal(left, right)


def test_pack_unpack_roundtrip(packer, apartment_level_df):
    street_level_df = packer.pack(apartment_level_df, "street")
    assert "country.city.street" in street_level_df.columns

    unpacked_df = packer.unpack(street_level_df, "apartment")

    _assert_same_rows(unpacked_df, apartment_level_df)


def test_pack_handles_missing_country_code_alias(packer, apartment_level_df):
    df_no_country_code = apartment_level_df.drop("country.code")

    packed_df = packer.pack(df_no_country_code, "street")
    assert "country.code" not in packed_df.columns

    roundtrip_df = packer.unpack(packed_df, "apartment")

    _assert_same_rows(roundtrip_df, df_no_country_code)


def test_split_levels_outputs_expected_tables(packer, apartment_level_df):
    city_level_df = packer.pack(apartment_level_df, "city")

    split_tables = packer.split_levels(city_level_df)

    assert set(split_tables.keys()) == {"city", "street", "building", "apartment"}

    apartment_table = split_tables["apartment"]
    _assert_same_rows(apartment_table, apartment_level_df)

    street_table = split_tables["street"]
    assert all(not col.startswith("country.city.street.building") for col in street_table.columns)
    expected_street_rows = apartment_level_df.select(
        ["country.city.id", "country.city.street.name"]
    ).unique()
    assert street_table.height == expected_street_rows.height

    city_table = split_tables["city"]
    assert all(
        col.startswith("country.") and not col.startswith("country.city.street")
        for col in city_table.columns
    )


def test_normalize_matches_manual_split(packer, apartment_level_df):
    normalized = packer.normalize(apartment_level_df)
    manual = packer.split_levels(packer.pack(apartment_level_df, "country"))

    assert normalized.keys() == manual.keys()
    for level_name, manual_table in manual.items():
        _assert_same_rows(normalized[level_name], manual_table)


def test_denormalize_reconstructs_nested(packer, apartment_level_df):
    normalized = packer.normalize(apartment_level_df)
    rebuilt = packer.denormalize(normalized, target_level="apartment")
    expected = packer.pack(apartment_level_df, "apartment")

    _assert_same_rows(rebuilt, expected)


def test_pack_without_preserve_order(apartment_level_df: pl.DataFrame) -> None:
    """Test that packing without order preservation works correctly."""
    relaxed_packer = HierarchicalPacker(TEST_HIERARCHY, preserve_child_order=False)

    street_level = relaxed_packer.pack(apartment_level_df, "street")
    assert "__hier_row_id" not in street_level.columns

    unpacked = relaxed_packer.unpack(street_level, "apartment")
    _assert_same_rows(unpacked, apartment_level_df)


# =============================================================================
# New Tests: Separator Escaping
# =============================================================================


class TestSeparatorEscaping:
    """Tests for separator escaping functionality."""

    def test_escape_unescape_roundtrip(self) -> None:
        """Test that escaping and unescaping a field name is reversible."""
        packer = HierarchicalPacker(
            HierarchySpec(levels=[LevelSpec(name="level", id_fields=["id"])]),
            granularity_separator=".",
            escape_char="\\",
        )

        # Field with separator
        original = "field.with.dots"
        escaped = packer._escape_field(original)
        unescaped = packer._unescape_field(escaped)

        assert escaped == "field\\.with\\.dots"
        assert unescaped == original

    def test_escape_char_in_field_name(self) -> None:
        """Test escaping when field name contains escape char."""
        packer = HierarchicalPacker(
            HierarchySpec(levels=[LevelSpec(name="level", id_fields=["id"])]),
            granularity_separator=".",
            escape_char="\\",
        )

        original = "field\\name"
        escaped = packer._escape_field(original)
        unescaped = packer._unescape_field(escaped)

        assert escaped == "field\\\\name"
        assert unescaped == original

    def test_split_path_with_escapes(self) -> None:
        """Test splitting a path that contains escaped separators."""
        packer = HierarchicalPacker(
            HierarchySpec(levels=[LevelSpec(name="level", id_fields=["id"])]),
            granularity_separator=".",
            escape_char="\\",
        )

        # Path with escaped dot
        path = "level\\.one.level\\.two"
        parts = packer._split_path(path)

        assert parts == ["level.one", "level.two"]

    def test_join_path_escapes_components(self) -> None:
        """Test that join_path properly escapes components."""
        packer = HierarchicalPacker(
            HierarchySpec(levels=[LevelSpec(name="level", id_fields=["id"])]),
            granularity_separator=".",
            escape_char="\\",
        )

        components = ["level.one", "level.two"]
        joined = packer._join_path(components)

        assert joined == "level\\.one.level\\.two"

    def test_custom_separator(self) -> None:
        """Test using a custom separator."""
        spec = HierarchySpec(
            levels=[
                LevelSpec(name="parent", id_fields=["id"]),
                LevelSpec(name="child", id_fields=["id"]),
            ]
        )
        packer = HierarchicalPacker(spec, granularity_separator="/")

        df = pl.DataFrame(
            {
                "parent/id": ["p1", "p1"],
                "parent/child/id": ["c1", "c2"],
                "parent/child/value": [10, 20],
            }
        )

        packed = packer.pack(df, "parent")
        assert "parent" in packed.columns

        unpacked = packer.unpack(packed, "child")
        _assert_same_rows(unpacked, df)

    def test_escape_char_same_as_separator_raises(self) -> None:
        """Test that using same char for escape and separator raises error."""
        with pytest.raises(ValueError, match="cannot be the same"):
            HierarchicalPacker(
                HierarchySpec(levels=[LevelSpec(name="level", id_fields=["id"])]),
                granularity_separator=".",
                escape_char=".",
            )


# =============================================================================
# New Tests: Validation
# =============================================================================


class TestValidation:
    """Tests for validation functionality."""

    def test_validate_detects_null_keys(self) -> None:
        """Test that validate() detects null values in key columns."""
        spec = HierarchySpec(
            levels=[
                LevelSpec(name="parent", id_fields=["id"]),
                LevelSpec(name="child", id_fields=["id"]),
            ]
        )
        packer = HierarchicalPacker(spec)

        df = pl.DataFrame(
            {
                "parent.id": ["p1", None, "p3"],  # Null in key column
                "parent.child.id": ["c1", "c2", "c3"],
                "parent.child.value": [10, 20, 30],
            }
        )

        with pytest.raises(HierarchyValidationError, match="null values"):
            packer.validate(df)

    def test_validate_returns_errors_when_not_raising(self) -> None:
        """Test that validate() returns errors without raising when configured."""
        spec = HierarchySpec(
            levels=[
                LevelSpec(name="parent", id_fields=["id"]),
            ]
        )
        packer = HierarchicalPacker(spec)

        df = pl.DataFrame({"parent.id": ["p1", None, "p3"]})

        errors = packer.validate(df, raise_on_error=False)

        assert len(errors) == 1
        assert errors[0].level == "parent"
        assert "null values" in str(errors[0])

    def test_validate_passes_for_valid_data(self) -> None:
        """Test that validate() passes for valid data."""
        spec = HierarchySpec(
            levels=[
                LevelSpec(name="parent", id_fields=["id"]),
            ]
        )
        packer = HierarchicalPacker(spec)

        df = pl.DataFrame({"parent.id": ["p1", "p2", "p3"]})

        errors = packer.validate(df, raise_on_error=False)
        assert len(errors) == 0

    def test_aggregation_validation_detects_non_uniform_values(self) -> None:
        """Test that aggregation validation detects non-uniform values."""
        spec = HierarchySpec(
            levels=[
                LevelSpec(name="parent", id_fields=["id"]),
                LevelSpec(name="child", id_fields=["id"]),
            ]
        )
        packer = HierarchicalPacker(spec, validate_on_pack=True)

        # Same parent.id but different parent.name values
        df = pl.DataFrame(
            {
                "parent.id": ["p1", "p1"],
                "parent.name": ["Name1", "Name2"],  # Non-uniform!
                "parent.child.id": ["c1", "c2"],
                "parent.child.value": [10, 20],
            }
        )

        with pytest.raises(HierarchyValidationError, match="non-uniform values"):
            packer.pack(df, "parent")

    def test_aggregation_validation_can_be_disabled(self) -> None:
        """Test that aggregation validation can be disabled."""
        spec = HierarchySpec(
            levels=[
                LevelSpec(name="parent", id_fields=["id"]),
                LevelSpec(name="child", id_fields=["id"]),
            ]
        )
        packer = HierarchicalPacker(spec, validate_on_pack=False)

        # Same parent.id but different parent.name values
        df = pl.DataFrame(
            {
                "parent.id": ["p1", "p1"],
                "parent.name": ["Name1", "Name2"],  # Non-uniform!
                "parent.child.id": ["c1", "c2"],
                "parent.child.value": [10, 20],
            }
        )

        # Should not raise - just picks first value
        packed = packer.pack(df, "parent")
        assert packed.height == 1


# =============================================================================
# New Tests: Frame Type Preservation
# =============================================================================


class TestFrameTypePreservation:
    """Tests for preserving DataFrame/LazyFrame types."""

    def test_pack_preserves_dataframe_type(
        self, packer: HierarchicalPacker, apartment_level_df: pl.DataFrame
    ) -> None:
        """Test that pack() returns DataFrame when given DataFrame."""
        result = packer.pack(apartment_level_df, "street")
        assert isinstance(result, pl.DataFrame)

    def test_pack_preserves_lazyframe_type(
        self, packer: HierarchicalPacker, apartment_level_df: pl.DataFrame
    ) -> None:
        """Test that pack() returns LazyFrame when given LazyFrame."""
        lf = apartment_level_df.lazy()
        result = packer.pack(lf, "street")
        assert isinstance(result, pl.LazyFrame)

    def test_unpack_preserves_dataframe_type(
        self, packer: HierarchicalPacker, apartment_level_df: pl.DataFrame
    ) -> None:
        """Test that unpack() returns DataFrame when given DataFrame."""
        packed = packer.pack(apartment_level_df, "street")
        result = packer.unpack(packed, "apartment")
        assert isinstance(result, pl.DataFrame)

    def test_unpack_preserves_lazyframe_type(
        self, packer: HierarchicalPacker, apartment_level_df: pl.DataFrame
    ) -> None:
        """Test that unpack() returns LazyFrame when given LazyFrame."""
        packed = packer.pack(apartment_level_df.lazy(), "street")
        result = packer.unpack(packed, "apartment")
        assert isinstance(result, pl.LazyFrame)


# =============================================================================
# New Tests: Empty DataFrames
# =============================================================================


class TestEmptyDataFrames:
    """Tests for handling empty DataFrames."""

    def test_pack_empty_dataframe(self) -> None:
        """Test that packing an empty DataFrame works correctly."""
        spec = HierarchySpec(
            levels=[
                LevelSpec(name="parent", id_fields=["id"]),
                LevelSpec(name="child", id_fields=["id"]),
            ]
        )
        packer = HierarchicalPacker(spec)

        df = pl.DataFrame(
            {
                "parent.id": pl.Series([], dtype=pl.Utf8),
                "parent.child.id": pl.Series([], dtype=pl.Utf8),
                "parent.child.value": pl.Series([], dtype=pl.Int64),
            }
        )

        packed = packer.pack(df, "parent")
        assert packed.height == 0
        assert "parent" in packed.columns

    def test_unpack_empty_dataframe(self) -> None:
        """Test that unpacking an empty packed DataFrame works correctly."""
        spec = HierarchySpec(
            levels=[
                LevelSpec(name="parent", id_fields=["id"]),
                LevelSpec(name="child", id_fields=["id"]),
            ]
        )
        packer = HierarchicalPacker(spec)

        df = pl.DataFrame(
            {
                "parent.id": pl.Series([], dtype=pl.Utf8),
                "parent.child.id": pl.Series([], dtype=pl.Utf8),
                "parent.child.value": pl.Series([], dtype=pl.Int64),
            }
        )

        packed = packer.pack(df, "parent")
        unpacked = packer.unpack(packed, "child")

        assert unpacked.height == 0


# =============================================================================
# New Tests: Build from Tables
# =============================================================================


class TestBuildFromTables:
    """Tests for building hierarchies from normalized tables."""

    def test_build_from_tables_simple(self) -> None:
        """Test building a simple hierarchy from normalized tables."""
        spec = HierarchySpec.from_levels(
            LevelSpec(name="city", id_fields=["id"]),
            LevelSpec(name="street", id_fields=["id"], parent_keys=["city_id"]),
        )
        packer = HierarchicalPacker(spec)

        city_df = pl.DataFrame({"id": ["NYC", "LA"], "name": ["New York", "Los Angeles"]})

        street_df = pl.DataFrame(
            {
                "id": ["st1", "st2", "st3"],
                "name": ["Broadway", "Main St", "Sunset Blvd"],
                "city_id": ["NYC", "NYC", "LA"],
            }
        )

        result = packer.build_from_tables({"city": city_df, "street": street_df})

        assert isinstance(result, pl.DataFrame)
        assert "city" in result.columns
        assert result.height == 2  # Two cities

        # Unpack and verify
        unpacked = packer.unpack(result, "street")
        assert unpacked.height == 3  # Three streets

    def test_build_from_tables_with_lazyframes(self) -> None:
        """Test that build_from_tables preserves LazyFrame type."""
        spec = HierarchySpec.from_levels(
            LevelSpec(name="parent", id_fields=["id"]),
            LevelSpec(name="child", id_fields=["id"], parent_keys=["parent_id"]),
        )
        packer = HierarchicalPacker(spec)

        parent_lf = pl.DataFrame({"id": ["p1"], "name": ["Parent 1"]}).lazy()
        child_lf = pl.DataFrame({"id": ["c1"], "name": ["Child 1"], "parent_id": ["p1"]}).lazy()

        result = packer.build_from_tables({"parent": parent_lf, "child": child_lf})

        assert isinstance(result, pl.LazyFrame)

    def test_build_from_tables_missing_table_raises(self) -> None:
        """Test that missing tables raise an appropriate error."""
        spec = HierarchySpec.from_levels(
            LevelSpec(name="parent", id_fields=["id"]),
            LevelSpec(name="child", id_fields=["id"], parent_keys=["parent_id"]),
        )
        packer = HierarchicalPacker(spec)

        parent_df = pl.DataFrame({"id": ["p1"], "name": ["Parent 1"]})

        # When target_level is child, we need the child table
        with pytest.raises(HierarchyValidationError, match="Missing table"):
            packer.build_from_tables({"parent": parent_df}, target_level="child")

    def test_build_from_tables_missing_parent_keys_raises(self) -> None:
        """Test that missing parent_keys on child level raises an error."""
        spec = HierarchySpec(
            levels=[
                LevelSpec(name="parent", id_fields=["id"]),
                LevelSpec(name="child", id_fields=["id"]),  # No parent_keys!
            ]
        )
        packer = HierarchicalPacker(spec)

        parent_df = pl.DataFrame({"id": ["p1"], "name": ["Parent 1"]})
        child_df = pl.DataFrame({"id": ["c1"], "name": ["Child 1"], "parent_id": ["p1"]})

        with pytest.raises(HierarchyValidationError, match="parent_keys"):
            packer.build_from_tables({"parent": parent_df, "child": child_df})


# =============================================================================
# New Tests: Composable Levels
# =============================================================================


class TestComposableLevels:
    """Tests for composable level definitions."""

    def test_from_levels_creates_hierarchy(self) -> None:
        """Test that from_levels creates a valid HierarchySpec."""
        spec = HierarchySpec.from_levels(
            LevelSpec(name="country", id_fields=["code"]),
            LevelSpec(name="city", id_fields=["id"], parent_keys=["country_code"]),
            LevelSpec(name="street", id_fields=["name"], parent_keys=["city_id"]),
        )

        assert len(spec.levels) == 3
        assert spec.levels[0].name == "country"
        assert spec.levels[1].name == "city"
        assert spec.levels[2].name == "street"

    def test_from_levels_validates_parent_keys_count(self) -> None:
        """Test that from_levels validates parent_keys count matches parent id_fields."""
        with pytest.raises(ValueError, match="parent_keys"):
            HierarchySpec.from_levels(
                LevelSpec(name="parent", id_fields=["id1", "id2"]),  # Two id fields
                LevelSpec(
                    name="child", id_fields=["id"], parent_keys=["parent_id"]
                ),  # Only one parent_key!
            )

    def test_from_levels_rejects_parent_keys_on_root(self) -> None:
        """Test that from_levels rejects parent_keys on root level."""
        with pytest.raises(ValueError, match="Root level"):
            HierarchySpec.from_levels(
                LevelSpec(name="root", id_fields=["id"], parent_keys=["invalid"]),
            )

    def test_from_levels_with_key_aliases(self) -> None:
        """Test that from_levels accepts key_aliases."""
        spec = HierarchySpec.from_levels(
            LevelSpec(name="parent", id_fields=["id"]),
            key_aliases={"parent.id": "parent.child.parent_id"},
        )

        assert spec.key_aliases == {"parent.id": "parent.child.parent_id"}


# =============================================================================
# New Tests: Error Messages
# =============================================================================


class TestErrorMessages:
    """Tests for error message quality."""

    def test_validation_error_includes_level(self) -> None:
        """Test that HierarchyValidationError includes level context."""
        error = HierarchyValidationError(
            "Test error message",
            level="test_level",
            details={"key": "value"},
        )

        assert "[Level: test_level]" in str(error)
        assert error.level == "test_level"
        assert error.details == {"key": "value"}

    def test_missing_level_error_is_descriptive(self) -> None:
        """Test that missing level errors are descriptive."""
        spec = HierarchySpec(levels=[LevelSpec(name="known", id_fields=["id"])])

        with pytest.raises(KeyError, match="not found"):
            spec.index_of("unknown")


# =============================================================================
# New Tests: Prepare Level Table
# =============================================================================


class TestPrepareLevelTable:
    """Tests for prepare_level_table functionality."""

    def test_prepare_level_table_adds_prefix(self) -> None:
        """Test that prepare_level_table adds the correct prefix to columns."""
        spec = HierarchySpec.from_levels(
            LevelSpec(name="parent", id_fields=["id"]),
            LevelSpec(name="child", id_fields=["id"], parent_keys=["parent_id"]),
        )
        packer = HierarchicalPacker(spec)

        raw_df = pl.DataFrame(
            {"id": ["c1", "c2"], "name": ["Child 1", "Child 2"], "parent_id": ["p1", "p1"]}
        )

        prepared = packer.prepare_level_table("child", raw_df)

        assert "parent.child.id" in prepared.columns
        assert "parent.child.name" in prepared.columns
        assert "parent.child.parent_id" in prepared.columns

    def test_prepare_level_table_with_column_mapping(self) -> None:
        """Test that prepare_level_table respects column mapping."""
        spec = HierarchySpec.from_levels(
            LevelSpec(name="item", id_fields=["id"]),
        )
        packer = HierarchicalPacker(spec)

        raw_df = pl.DataFrame({"item_id": [1, 2], "item_name": ["A", "B"]})

        prepared = packer.prepare_level_table(
            "item",
            raw_df,
            column_mapping={"item_id": "id", "item_name": "name"},
        )

        assert "item.id" in prepared.columns
        assert "item.name" in prepared.columns

    def test_prepare_level_table_preserves_dataframe_type(self) -> None:
        """Test that prepare_level_table preserves DataFrame type."""
        spec = HierarchySpec(levels=[LevelSpec(name="level", id_fields=["id"])])
        packer = HierarchicalPacker(spec)

        raw_df = pl.DataFrame({"id": [1, 2]})
        result = packer.prepare_level_table("level", raw_df)

        assert isinstance(result, pl.DataFrame)

    def test_prepare_level_table_preserves_lazyframe_type(self) -> None:
        """Test that prepare_level_table preserves LazyFrame type."""
        spec = HierarchySpec(levels=[LevelSpec(name="level", id_fields=["id"])])
        packer = HierarchicalPacker(spec)

        raw_lf = pl.DataFrame({"id": [1, 2]}).lazy()
        result = packer.prepare_level_table("level", raw_lf)

        assert isinstance(result, pl.LazyFrame)


# =============================================================================
# New Tests: Get Level Columns
# =============================================================================


class TestGetLevelColumns:
    """Tests for get_level_columns functionality."""

    def test_get_level_columns_returns_expected(self) -> None:
        """Test that get_level_columns returns the expected columns."""
        spec = HierarchySpec(
            levels=[
                LevelSpec(name="parent", id_fields=["id"], required_fields=["name"]),
                LevelSpec(name="child", id_fields=["id", "code"]),
            ]
        )
        packer = HierarchicalPacker(spec)

        parent_cols = packer.get_level_columns("parent")
        assert "parent.id" in parent_cols
        assert "parent.name" in parent_cols

        child_cols = packer.get_level_columns("child")
        assert "parent.child.id" in child_cols
        assert "parent.child.code" in child_cols


# =============================================================================
# Promote Attribute Tests
# =============================================================================


class TestPromoteAttribute:
    """Tests for the promote_attribute method."""

    @pytest.fixture()
    def promote_spec(self):
        return HierarchySpec(
            levels=[
                LevelSpec(name="country", id_fields=["code"]),
                LevelSpec(name="city", id_fields=["id"]),
                LevelSpec(name="street", id_fields=["name"]),
            ]
        )

    @pytest.fixture()
    def promote_packer(self, promote_spec):
        return HierarchicalPacker(promote_spec)

    @pytest.fixture()
    def promote_df(self):
        return pl.DataFrame(
            {
                "country.code": ["US", "US", "US", "CA", "CA"],
                "country.name": ["United States", "United States", "United States", "Canada", "Canada"],
                "country.city.id": ["NYC", "NYC", "LA", "TOR", "TOR"],
                "country.city.population": [8_000_000, 8_000_000, 4_000_000, 3_000_000, 3_000_000],
                "country.city.street.name": ["Broadway", "5th Ave", "Sunset Blvd", "Queen St", "King St"],
                "country.city.street.length_km": [21.0, 10.0, 35.0, 5.0, 3.0],
            }
        )

    def test_sum_city_to_country(self, promote_packer, promote_df):
        """Sum city populations to country level."""
        result = promote_packer.promote_attribute(
            promote_df, "population", from_level="city", to_level="country", agg="sum"
        )
        # US: NYC(8M) + LA(4M) = 12M, CA: TOR(3M) = 3M
        vals = result.sort("country.code").select("country.population").to_series().to_list()
        assert vals == [3_000_000, 12_000_000]

    def test_sum_street_to_city(self, promote_packer, promote_df):
        """Sum street lengths to city level."""
        result = promote_packer.promote_attribute(
            promote_df, "length_km", from_level="street", to_level="city", agg="sum"
        )
        vals = dict(
            zip(
                result.select("country.city.id").to_series().to_list(),
                result.select("country.city.length_km").to_series().to_list(),
            )
        )
        assert vals["NYC"] == 31.0
        assert vals["LA"] == 35.0
        assert vals["TOR"] == 8.0

    def test_list_aggregation(self, promote_packer, promote_df):
        """Collect street lengths as list at city level."""
        result = promote_packer.promote_attribute(
            promote_df, "length_km", from_level="street", to_level="city", agg="list"
        )
        nyc_row = result.filter(pl.col("country.city.id") == "NYC")
        lengths = nyc_row.select("country.city.length_km").to_series().to_list()[0]
        assert sorted(lengths) == [10.0, 21.0]

    def test_set_aggregation(self, promote_packer, promote_df):
        """Collect unique values."""
        result = promote_packer.promote_attribute(
            promote_df, "id", from_level="city", to_level="country", agg="set",
            alias="city_ids",
        )
        assert "country.city_ids" in result.columns
        us_row = result.filter(pl.col("country.code") == "US")
        ids = us_row.select("country.city_ids").to_series().to_list()[0]
        assert sorted(ids) == ["LA", "NYC"]

    def test_mean_aggregation(self, promote_packer, promote_df):
        result = promote_packer.promote_attribute(
            promote_df, "length_km", from_level="street", to_level="city", agg="mean"
        )
        nyc_row = result.filter(pl.col("country.city.id") == "NYC")
        assert nyc_row.select("country.city.length_km").to_series().to_list()[0] == 15.5

    def test_min_max(self, promote_packer, promote_df):
        r_min = promote_packer.promote_attribute(
            promote_df, "length_km", from_level="street", to_level="city", agg="min"
        )
        r_max = promote_packer.promote_attribute(
            promote_df, "length_km", from_level="street", to_level="city", agg="max"
        )
        nyc_min = r_min.filter(pl.col("country.city.id") == "NYC").select("country.city.length_km").item()
        nyc_max = r_max.filter(pl.col("country.city.id") == "NYC").select("country.city.length_km").item()
        assert nyc_min == 10.0
        assert nyc_max == 21.0

    def test_count_aggregation(self, promote_packer, promote_df):
        result = promote_packer.promote_attribute(
            promote_df, "length_km", from_level="street", to_level="city", agg="count"
        )
        nyc_count = result.filter(pl.col("country.city.id") == "NYC").select("country.city.length_km").item()
        assert nyc_count == 2

    def test_first_last(self, promote_packer, promote_df):
        r_first = promote_packer.promote_attribute(
            promote_df, "name", from_level="street", to_level="city", agg="first",
            alias="first_street",
        )
        r_last = promote_packer.promote_attribute(
            promote_df, "name", from_level="street", to_level="city", agg="last",
            alias="last_street",
        )
        nyc_first = r_first.filter(pl.col("country.city.id") == "NYC").select("country.city.first_street").item()
        nyc_last = r_last.filter(pl.col("country.city.id") == "NYC").select("country.city.last_street").item()
        assert nyc_first == "Broadway"
        assert nyc_last == "5th Ave"

    def test_single_uniform(self, promote_packer):
        """Single agg returns the unique value when all values are identical."""
        df = pl.DataFrame(
            {
                "country.code": ["US", "US"],
                "country.city.id": ["NYC", "LA"],
                "country.city.currency": ["USD", "USD"],
                "country.city.street.name": ["Broadway", "Sunset"],
                "country.city.street.length_km": [21.0, 35.0],
            }
        )
        result = promote_packer.promote_attribute(
            df, "currency", from_level="city", to_level="country", agg="single"
        )
        assert result.select("country.currency").to_series().to_list()[0] == "USD"

    def test_single_non_uniform_returns_first_unique(self, promote_packer, promote_df):
        """Single agg with non-uniform values returns the first unique value."""
        result = promote_packer.promote_attribute(
            promote_df, "population", from_level="city", to_level="country", agg="single"
        )
        # US has NYC(8M) and LA(4M) — non-uniform, returns first unique
        us_val = result.filter(pl.col("country.code") == "US").select("country.population").item()
        assert us_val in (8_000_000, 4_000_000)

    def test_alias_parameter(self, promote_packer, promote_df):
        """Custom alias for the output column."""
        result = promote_packer.promote_attribute(
            promote_df, "length_km", from_level="street", to_level="city",
            agg="sum", alias="total_street_length",
        )
        assert "country.city.total_street_length" in result.columns

    def test_invalid_direction_raises(self, promote_packer, promote_df):
        """Promoting from coarser to finer level raises ValueError."""
        with pytest.raises(ValueError, match="must be the immediate child"):
            promote_packer.promote_attribute(
                promote_df, "code", from_level="country", to_level="city", agg="list"
            )

    def test_non_adjacent_levels_raises(self, promote_packer, promote_df):
        """Skipping levels raises ValueError."""
        with pytest.raises(ValueError, match="must be the immediate child"):
            promote_packer.promote_attribute(
                promote_df, "length_km", from_level="street", to_level="country", agg="sum"
            )

    def test_missing_attribute_raises(self, promote_packer, promote_df):
        """Missing attribute raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            promote_packer.promote_attribute(
                promote_df, "nonexistent", from_level="street", to_level="city", agg="sum"
            )

    def test_from_packed_frame(self, promote_packer, promote_df):
        """Works correctly when input frame is already packed."""
        packed = promote_packer.pack(promote_df, "city")
        result = promote_packer.promote_attribute(
            packed, "population", from_level="city", to_level="country", agg="sum"
        )
        assert "country.population" in result.columns
        vals = result.sort("country.code").select("country.population").to_series().to_list()
        assert vals == [3_000_000, 12_000_000]

    def test_preserves_frame_type_lazy(self, promote_packer, promote_df):
        """Returns LazyFrame when input is LazyFrame."""
        result = promote_packer.promote_attribute(
            promote_df.lazy(), "length_km", from_level="street", to_level="city", agg="sum"
        )
        assert isinstance(result, pl.LazyFrame)


# ---------------------------------------------------------------------------
# Shared fixtures for attribute_expr / enrich / existential tests
# ---------------------------------------------------------------------------

CROSS_LEVEL_SPEC = HierarchySpec(
    levels=[
        LevelSpec(name="country", id_fields=["code"]),
        LevelSpec(name="city", id_fields=["id"]),
        LevelSpec(name="street", id_fields=["name"]),
    ]
)

CROSS_LEVEL_DF = pl.DataFrame(
    {
        "country.code": ["US", "US", "US", "CA", "CA"],
        "country.name": ["United States"] * 3 + ["Canada"] * 2,
        "country.city.id": ["NYC", "NYC", "LA", "TOR", "TOR"],
        "country.city.population": [8_000_000, 8_000_000, 4_000_000, 3_000_000, 3_000_000],
        "country.city.street.name": ["Broadway", "5th Ave", "Sunset Blvd", "Queen St", "King St"],
        "country.city.street.length_km": [21.0, 10.0, 35.0, 5.0, 3.0],
    }
)


@pytest.fixture()
def cl_packer():
    return HierarchicalPacker(CROSS_LEVEL_SPEC)


@pytest.fixture()
def cl_df():
    return CROSS_LEVEL_DF.clone()


# ---------------------------------------------------------------------------
# TestAttributeExpr
# ---------------------------------------------------------------------------


class TestAttributeExpr:
    """Tests for HierarchicalPacker.attribute_expr."""

    def test_same_level_returns_column(self, cl_packer, cl_df):
        """Same-level access returns the attribute as a direct column expr."""
        packed = cl_packer.pack(cl_df, "city")  # country-level frame
        expr = cl_packer.attribute_expr("name", "country", "country")
        result = sorted(packed.select(expr).to_series().to_list())
        assert result == ["Canada", "United States"]

    def test_immediate_child_sum(self, cl_packer, cl_df):
        """Cross-level sum aggregation over immediate child level."""
        packed = cl_packer.pack(cl_df, "city")
        expr = cl_packer.attribute_expr("population", "city", "country", "sum")
        vals = dict(
            zip(
                packed.select("country.code").to_series().to_list(),
                packed.select(expr).to_series().to_list(),
            )
        )
        assert vals["US"] == 12_000_000
        assert vals["CA"] == 3_000_000

    def test_immediate_child_count(self, cl_packer, cl_df):
        """Cross-level count gives number of child entities."""
        packed = cl_packer.pack(cl_df, "city")
        expr = cl_packer.attribute_expr("id", "city", "country", "count")
        vals = dict(
            zip(
                packed.select("country.code").to_series().to_list(),
                packed.select(expr).to_series().to_list(),
            )
        )
        assert vals["US"] == 2  # NYC and LA
        assert vals["CA"] == 1  # TOR

    def test_two_hop_sum(self, cl_packer, cl_df):
        """Sum across two hops (street → country) cascades correctly."""
        packed = cl_packer.pack(cl_df, "city")
        expr = cl_packer.attribute_expr("length_km", "street", "country", "sum")
        vals = dict(
            zip(
                packed.select("country.code").to_series().to_list(),
                packed.select(expr).to_series().to_list(),
            )
        )
        # US: Broadway(21) + 5th Ave(10) + Sunset Blvd(35) = 66
        assert vals["US"] == pytest.approx(66.0)
        # CA: Queen St(5) + King St(3) = 8
        assert vals["CA"] == pytest.approx(8.0)

    def test_two_hop_count(self, cl_packer, cl_df):
        """Count across two hops gives total number of from_level entities."""
        packed = cl_packer.pack(cl_df, "city")
        expr = cl_packer.attribute_expr("name", "street", "country", "count")
        vals = dict(
            zip(
                packed.select("country.code").to_series().to_list(),
                packed.select(expr).to_series().to_list(),
            )
        )
        assert vals["US"] == 3  # Broadway, 5th Ave, Sunset Blvd
        assert vals["CA"] == 2  # Queen St, King St

    def test_used_as_filter(self, cl_packer, cl_df):
        """attribute_expr result can be used directly in filter()."""
        packed = cl_packer.pack(cl_df, "city")
        expr = cl_packer.attribute_expr("id", "city", "country", "count")
        result = packed.filter(expr > 1)
        assert result.select("country.code").to_series().to_list() == ["US"]

    def test_used_as_sort_key(self, cl_packer, cl_df):
        """attribute_expr result can be used as a sort key."""
        packed = cl_packer.pack(cl_df, "city")
        expr = cl_packer.attribute_expr("population", "city", "country", "sum")
        result = packed.sort(expr, descending=True).select("country.code").to_series().to_list()
        assert result[0] == "US"

    def test_expression_arithmetic(self, cl_packer, cl_df):
        """Two attribute_expr results compose naturally with Polars arithmetic."""
        packed = cl_packer.pack(cl_df, "city")
        city_count = cl_packer.attribute_expr("id", "city", "country", "count")
        total_pop = cl_packer.attribute_expr("population", "city", "country", "sum")
        result = packed.with_columns((total_pop / city_count).alias("avg_pop"))
        us_row = result.filter(pl.col("country.code") == "US")
        assert us_row.select("avg_pop").item() == pytest.approx(6_000_000.0)

    def test_coarser_from_level_raises(self, cl_packer, cl_df):
        """Requesting attribute from a coarser level raises ValueError."""
        with pytest.raises(ValueError, match="coarser"):
            cl_packer.attribute_expr("code", "country", "city")

    def test_preserves_lazyframe(self, cl_packer, cl_df):
        """Works on LazyFrame; result is a plain pl.Expr regardless."""
        packed = cl_packer.pack(cl_df.lazy(), "city")
        expr = cl_packer.attribute_expr("id", "city", "country", "count")
        result = packed.filter(expr >= 1).collect()
        assert len(result) == 2


# ---------------------------------------------------------------------------
# TestEnrich
# ---------------------------------------------------------------------------


class TestEnrich:
    """Tests for HierarchicalPacker.enrich."""

    def test_single_spec(self, cl_packer, cl_df):
        """enrich with a single LevelAttribute adds the column."""
        packed = cl_packer.pack(cl_df, "city")
        result = cl_packer.enrich(
            packed,
            LevelAttribute("id", "city", "count", alias="city_count"),
            at_level="country",
        )
        assert "country.city_count" in result.columns
        vals = dict(
            zip(
                result.select("country.code").to_series().to_list(),
                result.select("country.city_count").to_series().to_list(),
            )
        )
        assert vals["US"] == 2
        assert vals["CA"] == 1

    def test_multiple_specs(self, cl_packer, cl_df):
        """enrich adds multiple attribute columns at once."""
        packed = cl_packer.pack(cl_df, "city")
        result = cl_packer.enrich(
            packed,
            LevelAttribute("id", "city", "count", alias="city_count"),
            LevelAttribute("population", "city", "sum", alias="total_pop"),
            at_level="country",
        )
        assert "country.city_count" in result.columns
        assert "country.total_pop" in result.columns

    def test_same_level_spec(self, cl_packer, cl_df):
        """enrich works for same-level attribute access."""
        packed = cl_packer.pack(cl_df, "city")
        result = cl_packer.enrich(
            packed,
            LevelAttribute("name", "country", "single", alias="cname"),
            at_level="country",
        )
        assert "country.cname" in result.columns

    def test_default_alias(self, cl_packer, cl_df):
        """When alias is None, column name defaults to the attribute name."""
        packed = cl_packer.pack(cl_df, "city")
        result = cl_packer.enrich(
            packed,
            LevelAttribute("population", "city", "sum"),
            at_level="country",
        )
        assert "country.population" in result.columns

    def test_preserves_lazyframe(self, cl_packer, cl_df):
        """enrich preserves LazyFrame type."""
        packed = cl_packer.pack(cl_df.lazy(), "city")
        result = cl_packer.enrich(
            packed,
            LevelAttribute("id", "city", "count", alias="city_count"),
            at_level="country",
        )
        assert isinstance(result, pl.LazyFrame)


# ---------------------------------------------------------------------------
# TestAnyAllChildSatisfies
# ---------------------------------------------------------------------------


class TestAnyAllChildSatisfies:
    """Tests for any_child_satisfies and all_children_satisfy."""

    def test_any_child_satisfies_basic(self, cl_packer, cl_df):
        """Filter countries where any city has population > 5M."""
        packed = cl_packer.pack(cl_df, "city")
        result = cl_packer.any_child_satisfies(
            packed,
            from_level="city",
            to_level="country",
            condition=pl.element().struct.field("population") > 5_000_000,
        )
        codes = sorted(result.select("country.code").to_series().to_list())
        assert codes == ["US"]  # only US has NYC(8M) > 5M

    def test_any_child_satisfies_all_pass(self, cl_packer, cl_df):
        """When all entities have qualifying children, all rows are returned."""
        packed = cl_packer.pack(cl_df, "city")
        result = cl_packer.any_child_satisfies(
            packed,
            from_level="city",
            to_level="country",
            condition=pl.element().struct.field("population") > 0,
        )
        assert len(result) == len(packed)

    def test_any_child_satisfies_none_pass(self, cl_packer, cl_df):
        """When no entities have qualifying children, result is empty."""
        packed = cl_packer.pack(cl_df, "city")
        result = cl_packer.any_child_satisfies(
            packed,
            from_level="city",
            to_level="country",
            condition=pl.element().struct.field("population") > 1_000_000_000,
        )
        assert len(result) == 0

    def test_all_children_satisfy_basic(self, cl_packer, cl_df):
        """Filter countries where ALL cities have population > 2M."""
        packed = cl_packer.pack(cl_df, "city")
        result = cl_packer.all_children_satisfy(
            packed,
            from_level="city",
            to_level="country",
            condition=pl.element().struct.field("population") > 2_000_000,
        )
        # US: NYC(8M) > 2M ✓, LA(4M) > 2M ✓  → pass
        # CA: TOR(3M) > 2M ✓                   → pass
        codes = sorted(result.select("country.code").to_series().to_list())
        assert codes == ["CA", "US"]

    def test_all_children_satisfy_partial(self, cl_packer, cl_df):
        """Filter countries where ALL cities have population > 5M."""
        packed = cl_packer.pack(cl_df, "city")
        result = cl_packer.all_children_satisfy(
            packed,
            from_level="city",
            to_level="country",
            condition=pl.element().struct.field("population") > 5_000_000,
        )
        # US: NYC(8M) ✓ but LA(4M) ✗  → fail
        # CA: TOR(3M) ✗               → fail
        assert len(result) == 0

    def test_non_adjacent_levels_raises(self, cl_packer, cl_df):
        """Skipping a level raises ValueError."""
        packed = cl_packer.pack(cl_df, "city")
        with pytest.raises(ValueError, match="immediate child"):
            cl_packer.any_child_satisfies(
                packed,
                from_level="street",
                to_level="country",
                condition=pl.element().struct.field("length_km") > 10,
            )

    def test_preserves_lazyframe(self, cl_packer, cl_df):
        """any_child_satisfies preserves LazyFrame type."""
        packed = cl_packer.pack(cl_df.lazy(), "city")
        result = cl_packer.any_child_satisfies(
            packed,
            from_level="city",
            to_level="country",
            condition=pl.element().struct.field("population") > 5_000_000,
        )
        assert isinstance(result, pl.LazyFrame)

# =============================================================================
# Usability Helper Tests
# =============================================================================


class TestUsabilityHelpers:
    """Tests for introspection and navigation helper methods."""

    # ------------------------------------------------------------------
    # Properties: level_names, root_level, leaf_level
    # ------------------------------------------------------------------

    def test_level_names(self, packer):
        assert packer.level_names == ["country", "city", "street", "building", "apartment"]

    def test_root_level(self, packer):
        assert packer.root_level == "country"

    def test_leaf_level(self, packer):
        assert packer.leaf_level == "apartment"

    # ------------------------------------------------------------------
    # get_ancestor_levels / get_descendant_levels
    # ------------------------------------------------------------------

    def test_get_ancestor_levels_root_returns_empty(self, packer):
        assert packer.get_ancestor_levels("country") == []

    def test_get_ancestor_levels_middle(self, packer):
        assert packer.get_ancestor_levels("street") == ["country", "city"]

    def test_get_ancestor_levels_leaf(self, packer):
        assert packer.get_ancestor_levels("apartment") == [
            "country",
            "city",
            "street",
            "building",
        ]

    def test_get_descendant_levels_leaf_returns_empty(self, packer):
        assert packer.get_descendant_levels("apartment") == []

    def test_get_descendant_levels_middle(self, packer):
        assert packer.get_descendant_levels("city") == ["street", "building", "apartment"]

    def test_get_descendant_levels_root(self, packer):
        assert packer.get_descendant_levels("country") == [
            "city",
            "street",
            "building",
            "apartment",
        ]

    def test_get_ancestor_levels_unknown_raises(self, packer):
        with pytest.raises(KeyError, match="unknown"):
            packer.get_ancestor_levels("unknown")

    # ------------------------------------------------------------------
    # get_level_keys
    # ------------------------------------------------------------------

    def test_get_level_keys_short_root(self, packer):
        assert packer.get_level_keys("country") == ["code"]

    def test_get_level_keys_short_multi_key(self, packer):
        assert packer.get_level_keys("city") == ["id", "name"]

    def test_get_level_keys_long(self, packer):
        assert packer.get_level_keys("city", form="long") == [
            "country.city.id",
            "country.city.name",
        ]

    def test_get_level_keys_with_ancestors(self, packer):
        keys = packer.get_level_keys("city", include_ancestors=True)
        assert keys == ["country.code", "country.city.id", "country.city.name"]

    def test_get_level_keys_ancestors_always_long_form(self, packer):
        # include_ancestors=True always forces long form regardless of form argument
        keys_default = packer.get_level_keys("city", include_ancestors=True)
        keys_explicit_long = packer.get_level_keys("city", include_ancestors=True, form="long")
        assert keys_default == keys_explicit_long
        assert all("." in k for k in keys_default), "ancestor keys should be fully qualified"

    def test_get_level_keys_leaf_with_ancestors(self, packer):
        keys = packer.get_level_keys("apartment", include_ancestors=True)
        assert "country.code" in keys
        assert "country.city.street.building.apartment.id" in keys

    # ------------------------------------------------------------------
    # get_level_fields — flat schema
    # ------------------------------------------------------------------

    def test_get_level_fields_flat_short(self, packer, apartment_level_df):
        fields = packer.get_level_fields("building", apartment_level_df)
        assert set(fields) == {"number", "id"}

    def test_get_level_fields_flat_long(self, packer, apartment_level_df):
        fields = packer.get_level_fields("building", apartment_level_df, form="long")
        assert set(fields) == {
            "country.city.street.building.number",
            "country.city.street.building.id",
        }

    def test_get_level_fields_excludes_child_columns(self, packer, apartment_level_df):
        # city fields should NOT include street/building/apartment columns
        fields = packer.get_level_fields("city", apartment_level_df)
        assert "id" in fields
        assert "name" in fields
        assert "street" not in fields
        # should not include apartment or street sub-fields
        assert not any("street" in f for f in fields)

    def test_get_level_fields_accepts_schema(self, packer, apartment_level_df):
        fields_from_df = packer.get_level_fields("city", apartment_level_df)
        fields_from_schema = packer.get_level_fields("city", apartment_level_df.schema)
        assert fields_from_df == fields_from_schema

    def test_get_level_fields_accepts_lazyframe(self, packer, apartment_level_df):
        fields_from_df = packer.get_level_fields("city", apartment_level_df)
        fields_from_lazy = packer.get_level_fields("city", apartment_level_df.lazy())
        assert fields_from_df == fields_from_lazy

    # ------------------------------------------------------------------
    # get_level_fields — packed schema
    # ------------------------------------------------------------------

    def test_get_level_fields_packed_short(self, packer, apartment_level_df):
        packed = packer.pack(apartment_level_df, "city")
        fields = packer.get_level_fields("city", packed)
        # city struct fields should be id and name (not street sub-struct)
        assert "id" in fields
        assert "name" in fields
        assert "street" not in fields

    def test_get_level_fields_packed_long(self, packer, apartment_level_df):
        packed = packer.pack(apartment_level_df, "city")
        fields = packer.get_level_fields("city", packed, form="long")
        assert "country.city.id" in fields
        assert "country.city.name" in fields
        assert not any("street" in f for f in fields)

    # ------------------------------------------------------------------
    # infer_current_level
    # ------------------------------------------------------------------

    def test_infer_current_level_flat_is_leaf(self, packer, apartment_level_df):
        assert packer.infer_current_level(apartment_level_df) == "apartment"

    def test_infer_current_level_packed_to_street(self, packer, apartment_level_df):
        # pack(df, "street") packs street and below into a List[Struct] column,
        # so each row represents a city (the level above the first packed column).
        packed = packer.pack(apartment_level_df, "street")
        assert packer.infer_current_level(packed) == "city"

    def test_infer_current_level_packed_to_city(self, packer, apartment_level_df):
        # pack(df, "city") packs city and below, so each row represents a country.
        packed = packer.pack(apartment_level_df, "city")
        assert packer.infer_current_level(packed) == "country"

    def test_infer_current_level_packed_to_country(self, packer, apartment_level_df):
        packed = packer.pack(apartment_level_df, "country")
        assert packer.infer_current_level(packed) == "country"

    def test_infer_current_level_accepts_schema(self, packer, apartment_level_df):
        assert packer.infer_current_level(apartment_level_df.schema) == "apartment"

    def test_infer_current_level_accepts_lazyframe(self, packer, apartment_level_df):
        assert packer.infer_current_level(apartment_level_df.lazy()) == "apartment"

    # ------------------------------------------------------------------
    # get_level_schema
    # ------------------------------------------------------------------

    def test_get_level_schema_flat(self, packer, apartment_level_df):
        level_schema = packer.get_level_schema("building", apartment_level_df)
        assert "number" in level_schema
        assert "id" in level_schema
        assert "apartment" not in level_schema

    def test_get_level_schema_packed(self, packer, apartment_level_df):
        packed = packer.pack(apartment_level_df, "city")
        level_schema = packer.get_level_schema("city", packed)
        assert "id" in level_schema
        assert "name" in level_schema
        # child struct should be excluded
        assert "street" not in level_schema

    def test_get_level_schema_returns_correct_types(self, packer, apartment_level_df):
        level_schema = packer.get_level_schema("country", apartment_level_df)
        assert "code" in level_schema
        assert level_schema["code"] == pl.String

    # ------------------------------------------------------------------
    # describe
    # ------------------------------------------------------------------

    def test_describe_contains_level_names(self, packer):
        desc = packer.describe()
        for name in packer.level_names:
            assert name in desc

    def test_describe_contains_root_leaf_tags(self, packer):
        desc = packer.describe()
        assert "root" in desc
        assert "leaf" in desc

    def test_describe_contains_separator(self, packer):
        desc = packer.describe()
        assert 'separator="."' in desc

    def test_describe_contains_key_names(self, packer):
        desc = packer.describe()
        assert "code" in desc  # country key
        assert "number" in desc  # building key


# =============================================================================
# Hierarchy Discovery Tests
# =============================================================================


class TestDiscoverLevels:
    """Tests for hierarchy discovery from schema."""

    def test_discover_from_flat_schema(self, apartment_level_df):
        """Discover levels from a fully flat DataFrame."""
        levels = HierarchicalPacker.discover_levels(apartment_level_df)
        names = [lvl.name for lvl in levels]
        assert names == ["country", "city", "street", "building", "apartment"]

    def test_discover_depths(self, apartment_level_df):
        """Discovered levels have correct depths."""
        levels = HierarchicalPacker.discover_levels(apartment_level_df)
        depths = {lvl.name: lvl.depth for lvl in levels}
        assert depths == {
            "country": 0,
            "city": 1,
            "street": 2,
            "building": 3,
            "apartment": 4,
        }

    def test_discover_paths(self, apartment_level_df):
        """Discovered levels have correct full paths."""
        levels = HierarchicalPacker.discover_levels(apartment_level_df)
        paths = {lvl.name: lvl.path for lvl in levels}
        assert paths["country"] == "country"
        assert paths["city"] == "country.city"
        assert paths["apartment"] == "country.city.street.building.apartment"

    def test_discover_parents(self, apartment_level_df):
        """Discovered levels have correct parent references."""
        levels = HierarchicalPacker.discover_levels(apartment_level_df)
        parents = {lvl.name: lvl.parent for lvl in levels}
        assert parents["country"] is None
        assert parents["city"] == "country"
        assert parents["street"] == "city"
        assert parents["building"] == "street"
        assert parents["apartment"] == "building"

    def test_discover_fields_correct(self, apartment_level_df):
        """Discovered levels report correct non-level field names."""
        levels = HierarchicalPacker.discover_levels(apartment_level_df)
        fields_by_name = {lvl.name: lvl.fields for lvl in levels}
        assert "code" in fields_by_name["country"]
        assert "id" in fields_by_name["city"]
        assert "name" in fields_by_name["city"]
        assert "area" in fields_by_name["apartment"]

    def test_discover_flat_not_packed(self, apartment_level_df):
        """Flat schema levels should not be marked as packed."""
        levels = HierarchicalPacker.discover_levels(apartment_level_df)
        assert all(not lvl.is_packed for lvl in levels)

    def test_discover_from_packed_schema(self, packer, apartment_level_df):
        """Discover levels from a packed DataFrame."""
        packed = packer.pack(apartment_level_df, "city")
        levels = HierarchicalPacker.discover_levels(packed)
        names = [lvl.name for lvl in levels]
        assert "country" in names
        assert "city" in names
        assert "street" in names
        assert "building" in names
        assert "apartment" in names

    def test_discover_packed_levels_marked(self, packer, apartment_level_df):
        """Levels inside packed columns should be marked is_packed=True."""
        packed = packer.pack(apartment_level_df, "city")
        levels = HierarchicalPacker.discover_levels(packed)
        by_name = {lvl.name: lvl for lvl in levels}
        # country is flat, city is a packed List[Struct] column
        assert not by_name["country"].is_packed
        assert by_name["city"].is_packed
        # levels inside the struct are also packed
        assert by_name["street"].is_packed

    def test_discover_from_partially_packed(self, packer, apartment_level_df):
        """Discover levels from partially packed data."""
        packed = packer.pack(apartment_level_df, "street")
        levels = HierarchicalPacker.discover_levels(packed)
        names = [lvl.name for lvl in levels]
        # All 5 levels should still be discoverable
        assert len(names) == 5
        by_name = {lvl.name: lvl for lvl in levels}
        # country and city are flat
        assert not by_name["country"].is_packed
        assert not by_name["city"].is_packed
        # street is packed (it's a List[Struct] column)
        assert by_name["street"].is_packed

    def test_discover_single_level(self):
        """Schema with only one level."""
        schema = pl.Schema({"entity.id": pl.String, "entity.name": pl.String})
        levels = HierarchicalPacker.discover_levels(schema)
        assert len(levels) == 1
        assert levels[0].name == "entity"
        assert levels[0].depth == 0
        assert levels[0].parent is None
        assert set(levels[0].fields) == {"id", "name"}

    def test_discover_custom_separator(self):
        """Discovery with non-default separator."""
        schema = pl.Schema({
            "region/code": pl.String,
            "region/city/id": pl.String,
        })
        levels = HierarchicalPacker.discover_levels(schema, separator="/")
        names = [lvl.name for lvl in levels]
        assert names == ["region", "city"]

    def test_discover_sibling_branches(self):
        """Struct with multiple List[Struct] fields (sibling branches)."""
        # A city with both "street" and "park" as List[Struct] children
        schema = pl.Schema({
            "city.name": pl.String,
            "city": pl.List(
                pl.Struct({
                    "id": pl.String,
                    "street": pl.List(pl.Struct({"name": pl.String})),
                    "park": pl.List(pl.Struct({"name": pl.String, "area": pl.Float64})),
                })
            ),
        })
        levels = HierarchicalPacker.discover_levels(schema)
        names = [lvl.name for lvl in levels]
        assert "city" in names
        assert "street" in names
        assert "park" in names
        # Both siblings should have same parent
        by_name = {lvl.name: lvl for lvl in levels}
        assert by_name["street"].parent == "city"
        assert by_name["park"].parent == "city"
        assert by_name["street"].depth == by_name["park"].depth

    def test_discover_empty_schema(self):
        """Empty schema returns empty list."""
        levels = HierarchicalPacker.discover_levels(pl.Schema({}))
        assert levels == []

    def test_discover_no_hierarchy_columns(self):
        """Schema with only top-level columns (no separator) returns empty."""
        schema = pl.Schema({"foo": pl.String, "bar": pl.Int64})
        levels = HierarchicalPacker.discover_levels(schema)
        assert levels == []

    def test_discover_accepts_lazyframe(self, apartment_level_df):
        """Works with LazyFrame input."""
        levels_df = HierarchicalPacker.discover_levels(apartment_level_df)
        levels_lf = HierarchicalPacker.discover_levels(apartment_level_df.lazy())
        assert levels_df == levels_lf

    def test_discover_accepts_schema(self, apartment_level_df):
        """Works with pl.Schema input."""
        levels_df = HierarchicalPacker.discover_levels(apartment_level_df)
        levels_schema = HierarchicalPacker.discover_levels(apartment_level_df.schema)
        assert levels_df == levels_schema

    def test_discover_depth_ordering(self, apartment_level_df):
        """Results are ordered by depth then name."""
        levels = HierarchicalPacker.discover_levels(apartment_level_df)
        depths = [lvl.depth for lvl in levels]
        assert depths == sorted(depths)

    def test_discover_intermediate_levels_created(self):
        """Levels with no direct fields are still discovered as intermediates."""
        # Only leaf columns exist, but intermediate levels should be inferred
        schema = pl.Schema({
            "a.b.c.value": pl.Int64,
        })
        levels = HierarchicalPacker.discover_levels(schema)
        names = [lvl.name for lvl in levels]
        assert "a" in names
        assert "b" in names
        assert "c" in names


# =============================================================================
# Schema Validation Tests
# =============================================================================


class TestValidateSchema:
    """Tests for schema compatibility validation."""

    def test_compatible_flat_schema(self, packer, apartment_level_df):
        """Flat schema matching the hierarchy is compatible."""
        result = packer.validate_schema(apartment_level_df)
        assert result.is_compatible
        assert result.inferred_level == "apartment"
        assert len(result.errors) == 0
        assert len(result.present_levels) == 5

    def test_compatible_packed_schema(self, packer, apartment_level_df):
        """Packed schema is compatible."""
        packed = packer.pack(apartment_level_df, "city")
        result = packer.validate_schema(packed)
        assert result.is_compatible
        assert result.inferred_level == "country"

    def test_incompatible_no_hierarchy_columns(self, packer):
        """Schema with no hierarchy columns is incompatible."""
        schema = pl.Schema({"foo": pl.String, "bar": pl.Int64})
        result = packer.validate_schema(schema)
        assert not result.is_compatible
        assert len(result.errors) > 0
        assert len(result.missing_levels) > 0
        assert len(result.present_levels) == 0

    def test_partial_levels_missing(self, packer, apartment_level_df):
        """Schema with some levels missing reports them."""
        df = apartment_level_df.drop(
            "country.city.street.building.apartment.id",
            "country.city.street.building.apartment.area",
        )
        result = packer.validate_schema(df)
        assert "apartment" in result.missing_levels

    def test_expected_level_matches(self, packer, apartment_level_df):
        """Specifying expected_level that matches is fine."""
        result = packer.validate_schema(apartment_level_df, expected_level="apartment")
        assert result.is_compatible

    def test_expected_level_mismatch(self, packer, apartment_level_df):
        """Specifying wrong expected_level produces an error."""
        result = packer.validate_schema(apartment_level_df, expected_level="city")
        assert not result.is_compatible
        assert any("expected" in e.lower() for e in result.errors)

    def test_wrong_column_types(self, packer):
        """Columns with wrong types produce errors."""
        schema = pl.Schema({
            "country.code": pl.List(pl.String),  # Should be scalar
            "country.city.id": pl.String,
            "country.city.name": pl.String,
            "country.city.street.name": pl.String,
            "country.city.street.building.number": pl.Int64,
            "country.city.street.building.id": pl.String,
            "country.city.street.building.apartment.id": pl.String,
            "country.city.street.building.apartment.area": pl.Float64,
        })
        result = packer.validate_schema(schema)
        assert any("type" in e.lower() for e in result.errors)

    def test_accepts_lazyframe(self, packer, apartment_level_df):
        """Works with LazyFrame input."""
        result = packer.validate_schema(apartment_level_df.lazy())
        assert result.is_compatible

    def test_accepts_schema_object(self, packer, apartment_level_df):
        """Works with pl.Schema input."""
        result = packer.validate_schema(apartment_level_df.schema)
        assert result.is_compatible

    def test_result_fields_populated(self, packer, apartment_level_df):
        """All result fields are populated correctly."""
        result = packer.validate_schema(apartment_level_df)
        assert isinstance(result, SchemaValidationResult)
        assert isinstance(result.is_compatible, bool)
        assert isinstance(result.present_levels, list)
        assert isinstance(result.missing_levels, list)
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)

    def test_present_and_missing_levels_disjoint(self, packer, apartment_level_df):
        """present_levels and missing_levels should not overlap."""
        # Full schema
        result = packer.validate_schema(apartment_level_df)
        assert set(result.present_levels).isdisjoint(set(result.missing_levels))

        # Partial schema
        df = apartment_level_df.drop(
            "country.city.street.building.apartment.id",
            "country.city.street.building.apartment.area",
        )
        result = packer.validate_schema(df)
        assert set(result.present_levels).isdisjoint(set(result.missing_levels))

    def test_packed_struct_missing_id_field(self, packer, apartment_level_df):
        """Packed struct missing expected key field produces error."""
        # Create a packed schema where we manually drop a key field from the struct
        # by constructing a schema with a malformed packed column
        schema = pl.Schema({
            "country.code": pl.String,
            "country.city": pl.List(
                pl.Struct({
                    # "id" is missing — it's a key for city level
                    "name": pl.String,
                    "street": pl.List(pl.Struct({"name": pl.String})),
                })
            ),
        })
        result = packer.validate_schema(schema)
        assert any(
            "missing" in e.lower() and "key" in e.lower()
            for e in result.errors
        )

"""Tests for the hierarchical_packer module."""

from __future__ import annotations

import json

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from nexpresso.hierarchical_packer import (
    HierarchicalPacker,
    HierarchySpec,
    HierarchyValidationError,
    LevelSpec,
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

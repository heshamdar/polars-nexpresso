import json

import polars as pl
import pytest

from nexpresso.hierarchical_packer import HierarchicalPacker, HierarchySpec, LevelSpec

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
    assert _canonical_rows(left) == _canonical_rows(right)


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


def test_pack_without_preserve_order(apartment_level_df):
    relaxed_packer = HierarchicalPacker(TEST_HIERARCHY, preserve_child_order=False)

    street_level = relaxed_packer.pack(apartment_level_df, "street")
    assert "__hier_row_id" not in street_level.columns

    unpacked = relaxed_packer.unpack(street_level, "apartment")
    _assert_same_rows(unpacked, apartment_level_df)

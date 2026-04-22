"""Tests for NestedFilterBuilder, generate_nested_filter_expr, filter_nested_elements."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from nexpresso import (
    NestedFilterBuilder,
    apply_nested_filter,
    filter_nested_elements,
    generate_nested_filter_expr,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def city_df() -> pl.DataFrame:
    """DataFrame with scalar, struct, and List<Struct> columns."""
    return pl.DataFrame(
        {
            "city": ["Alpha", "Beta", "Gamma"],
            "population": [50_000, 200_000, 80_000],
            "info": [
                {"founded": 1800, "area_km2": 120.0},
                {"founded": 1950, "area_km2": 300.0},
                {"founded": 1700, "area_km2": 90.0},
            ],
            "streets": [
                [{"name": "Main", "length_km": 2.5}, {"name": "Oak", "length_km": 0.4}],
                [{"name": "Pine", "length_km": 0.3}, {"name": "Elm", "length_km": 0.2}],
                [{"name": "River", "length_km": 1.1}, {"name": "Hill", "length_km": 1.8}],
            ],
        }
    )


@pytest.fixture
def city_lf(city_df: pl.DataFrame) -> pl.LazyFrame:
    return city_df.lazy()


# =============================================================================
# Row-level filter — generate_nested_filter_expr
# =============================================================================


class TestGenerateNestedFilterExpr:
    def test_scalar_column_callable(self, city_df: pl.DataFrame) -> None:
        pred = generate_nested_filter_expr(
            {"population": lambda x: x > 100_000},
            city_df.schema,
        )
        result = city_df.filter(pred)
        assert result["city"].to_list() == ["Beta"]

    def test_struct_field_predicate(self, city_df: pl.DataFrame) -> None:
        pred = generate_nested_filter_expr(
            {"info": {"area_km2": lambda x: x > 200.0}},
            city_df.schema,
        )
        result = city_df.filter(pred)
        assert result["city"].to_list() == ["Beta"]

    def test_struct_multiple_fields_and(self, city_df: pl.DataFrame) -> None:
        # founded < 1900 AND area > 100 → only Alpha qualifies (area=120, founded=1800)
        pred = generate_nested_filter_expr(
            {
                "info": {
                    "founded": lambda x: x < 1900,
                    "area_km2": lambda x: x > 100.0,
                }
            },
            city_df.schema,
        )
        result = city_df.filter(pred)
        assert result["city"].to_list() == ["Alpha"]

    def test_list_struct_field_any_semantics(self, city_df: pl.DataFrame) -> None:
        # max length > 1 → any street > 1km
        # Alpha: max=2.5 ✓, Beta: max=0.3 ✗, Gamma: max=1.8 ✓
        pred = generate_nested_filter_expr(
            {"streets": {"length_km": lambda x: x.list.max() > 1}},
            city_df.schema,
        )
        result = city_df.filter(pred)
        assert result["city"].to_list() == ["Alpha", "Gamma"]

    def test_list_struct_field_all_semantics(self, city_df: pl.DataFrame) -> None:
        # min length > 1 → all streets > 1km
        # Alpha: min=0.4 ✗, Beta: min=0.2 ✗, Gamma: min=1.1 ✓
        pred = generate_nested_filter_expr(
            {"streets": {"length_km": lambda x: x.list.min() > 1}},
            city_df.schema,
        )
        result = city_df.filter(pred)
        assert result["city"].to_list() == ["Gamma"]

    def test_list_struct_field_explicit_list_any(self, city_df: pl.DataFrame) -> None:
        # User writes explicit list.eval(...).list.any() — same result as list.max() > 1
        pred = generate_nested_filter_expr(
            {
                "streets": {
                    "length_km": lambda x: x.list.eval(pl.element() > 1).list.any()
                }
            },
            city_df.schema,
        )
        result = city_df.filter(pred)
        assert result["city"].to_list() == ["Alpha", "Gamma"]

    def test_list_column_full_callable(self, city_df: pl.DataFrame) -> None:
        # User writes a full list-level expression directly
        # cities with more than 1 street
        pred = generate_nested_filter_expr(
            {"streets": lambda x: x.list.len() > 1},
            city_df.schema,
        )
        result = city_df.filter(pred)
        assert len(result) == 3  # all cities have 2 streets

    def test_multiple_columns_and(self, city_df: pl.DataFrame) -> None:
        # population > 70_000 AND any street > 1km
        # Alpha: pop=50k ✗; Beta: pop=200k ✓ but max_street=0.3 ✗; Gamma: pop=80k ✓ max=1.8 ✓
        pred = generate_nested_filter_expr(
            {
                "population": lambda x: x > 70_000,
                "streets": {"length_km": lambda x: x.list.max() > 1},
            },
            city_df.schema,
        )
        result = city_df.filter(pred)
        assert result["city"].to_list() == ["Gamma"]

    def test_accepts_dataframe_as_schema(self, city_df: pl.DataFrame) -> None:
        pred = generate_nested_filter_expr(
            {"population": lambda x: x > 100_000},
            city_df,
        )
        result = city_df.filter(pred)
        assert result["city"].to_list() == ["Beta"]

    def test_accepts_lazyframe_as_schema(self, city_lf: pl.LazyFrame) -> None:
        pred = generate_nested_filter_expr(
            {"population": lambda x: x > 100_000},
            city_lf,
        )
        result = city_lf.filter(pred).collect()
        assert result["city"].to_list() == ["Beta"]

    def test_lazyframe_type_preserved(self, city_lf: pl.LazyFrame) -> None:
        pred = generate_nested_filter_expr(
            {"population": lambda x: x > 100_000},
            city_lf,
        )
        result = city_lf.filter(pred)
        assert isinstance(result, pl.LazyFrame)

    def test_empty_spec_returns_all_rows(self, city_df: pl.DataFrame) -> None:
        pred = generate_nested_filter_expr({}, city_df.schema)
        result = city_df.filter(pred)
        assert_frame_equal(result, city_df)

    def test_plain_expr_value(self, city_df: pl.DataFrame) -> None:
        pred = generate_nested_filter_expr(
            {"population": pl.col("population") > 100_000},
            city_df.schema,
        )
        result = city_df.filter(pred)
        assert result["city"].to_list() == ["Beta"]

    def test_list_mean_aggregation(self, city_df: pl.DataFrame) -> None:
        # mean length > 1: Alpha mean=(2.5+0.4)/2=1.45 ✓, Beta mean=0.25 ✗, Gamma mean=1.45 ✓
        pred = generate_nested_filter_expr(
            {"streets": {"length_km": lambda x: x.list.mean() > 1}},
            city_df.schema,
        )
        result = city_df.filter(pred)
        assert result["city"].to_list() == ["Alpha", "Gamma"]


# =============================================================================
# apply_nested_filter convenience wrapper
# =============================================================================


class TestApplyNestedFilter:
    def test_basic(self, city_df: pl.DataFrame) -> None:
        result = apply_nested_filter(
            city_df, {"population": lambda x: x > 100_000}
        )
        assert result["city"].to_list() == ["Beta"]

    def test_preserves_dataframe_type(self, city_df: pl.DataFrame) -> None:
        result = apply_nested_filter(city_df, {"population": lambda x: x > 0})
        assert isinstance(result, pl.DataFrame)

    def test_preserves_lazyframe_type(self, city_lf: pl.LazyFrame) -> None:
        result = apply_nested_filter(city_lf, {"population": lambda x: x > 0})
        assert isinstance(result, pl.LazyFrame)


# =============================================================================
# Element-level filter — filter_nested_elements
# =============================================================================


class TestFilterNestedElements:
    def test_list_struct_single_field(self, city_df: pl.DataFrame) -> None:
        # Keep only streets longer than 1km
        result = filter_nested_elements(
            city_df, {"streets": {"length_km": lambda x: x > 1}}
        )
        streets = result["streets"].to_list()
        # Alpha: [Main(2.5)] — Oak(0.4) removed
        assert len(streets[0]) == 1
        assert streets[0][0]["name"] == "Main"
        # Beta: [] — both streets removed (0.3 and 0.2)
        assert streets[1] == []
        # Gamma: [River(1.1), Hill(1.8)] — both kept
        assert len(streets[2]) == 2

    def test_list_struct_multiple_fields_and(self, city_df: pl.DataFrame) -> None:
        # Keep streets where length > 0.3 AND name starts with a letter before 'P'
        result = filter_nested_elements(
            city_df,
            {
                "streets": {
                    "length_km": lambda x: x > 0.3,
                    "name": lambda x: x < pl.lit("P"),  # lexicographic: M, O, E, H
                }
            },
        )
        streets = result["streets"].to_list()
        # Alpha: Main(len=2.5, M<P ✓), Oak(len=0.4, O<P ✓) → both
        assert {s["name"] for s in streets[0]} == {"Main", "Oak"}
        # Beta: Pine(len=0.3 NOT >0.3 ✗), Elm(len=0.2 NOT >0.3 ✗) → empty
        assert streets[1] == []
        # Gamma: River(R>P ✗), Hill(len=1.8 ✓, H<P ✓) → Hill only
        assert len(streets[2]) == 1
        assert streets[2][0]["name"] == "Hill"

    def test_all_elements_filtered_gives_empty_list_not_null(
        self, city_df: pl.DataFrame
    ) -> None:
        result = filter_nested_elements(
            city_df, {"streets": {"length_km": lambda x: x > 100}}
        )
        streets = result["streets"].to_list()
        for row in streets:
            assert row == []

    def test_row_count_unchanged(self, city_df: pl.DataFrame) -> None:
        result = filter_nested_elements(
            city_df, {"streets": {"length_km": lambda x: x > 1}}
        )
        assert len(result) == len(city_df)

    def test_other_columns_unchanged(self, city_df: pl.DataFrame) -> None:
        result = filter_nested_elements(
            city_df, {"streets": {"length_km": lambda x: x > 1}}
        )
        assert result["city"].to_list() == city_df["city"].to_list()
        assert result["population"].to_list() == city_df["population"].to_list()

    def test_multiple_list_columns(self) -> None:
        df = pl.DataFrame(
            {
                "streets": [
                    [{"length_km": 2.0}, {"length_km": 0.4}],
                ],
                "parks": [
                    [{"area_ha": 5.0}, {"area_ha": 0.5}],
                ],
            }
        )
        result = filter_nested_elements(
            df,
            {
                "streets": {"length_km": lambda x: x > 1},
                "parks": {"area_ha": lambda x: x > 1},
            },
        )
        assert len(result["streets"][0]) == 1
        assert result["streets"][0][0]["length_km"] == 2.0
        assert len(result["parks"][0]) == 1
        assert result["parks"][0][0]["area_ha"] == 5.0

    def test_preserves_dataframe_type(self, city_df: pl.DataFrame) -> None:
        result = filter_nested_elements(city_df, {"streets": {"length_km": lambda x: x > 1}})
        assert isinstance(result, pl.DataFrame)

    def test_preserves_lazyframe_type(self, city_lf: pl.LazyFrame) -> None:
        result = filter_nested_elements(city_lf, {"streets": {"length_km": lambda x: x > 1}})
        assert isinstance(result, pl.LazyFrame)

    def test_lazyframe_correct_result(self, city_lf: pl.LazyFrame) -> None:
        result = filter_nested_elements(
            city_lf, {"streets": {"length_km": lambda x: x > 1}}
        ).collect()
        streets = result["streets"].to_list()
        assert len(streets[0]) == 1  # Alpha: only Main(2.5)
        assert streets[1] == []  # Beta: all removed
        assert len(streets[2]) == 2  # Gamma: both kept

    def test_plain_expr_element_predicate(self) -> None:
        df = pl.DataFrame({"nums": [[1, 3, 5, 2, 4]]})
        result = filter_nested_elements(df, {"nums": pl.element() > 3})
        assert result["nums"][0].to_list() == [5, 4]


# =============================================================================
# NestedFilterBuilder direct usage
# =============================================================================


class TestNestedFilterBuilderDirect:
    def test_build_row_filter_empty(self, city_df: pl.DataFrame) -> None:
        builder = NestedFilterBuilder(city_df.schema)
        pred = builder.build_row_filter({})
        # All rows pass
        assert len(city_df.filter(pred)) == 3

    def test_build_element_filters_returns_exprs(self, city_df: pl.DataFrame) -> None:
        builder = NestedFilterBuilder(city_df.schema)
        exprs = builder.build_element_filters({"streets": {"length_km": lambda x: x > 1}})
        assert len(exprs) == 1
        result = city_df.with_columns(exprs)
        assert len(result["streets"][0]) == 1


# =============================================================================
# Error cases
# =============================================================================


class TestFilterErrors:
    def test_row_filter_column_not_found(self, city_df: pl.DataFrame) -> None:
        with pytest.raises(ValueError, match="not found in schema"):
            generate_nested_filter_expr(
                {"missing": lambda x: x > 0}, city_df.schema
            )

    def test_row_filter_dict_on_scalar_column(self, city_df: pl.DataFrame) -> None:
        with pytest.raises(ValueError, match="dict filter spec"):
            generate_nested_filter_expr(
                {"population": {"sub": lambda x: x > 0}}, city_df.schema
            )

    def test_row_filter_invalid_spec_type(self, city_df: pl.DataFrame) -> None:
        with pytest.raises(TypeError, match="Invalid filter spec type"):
            generate_nested_filter_expr(
                {"population": 42},  # type: ignore[dict-item]
                city_df.schema,
            )

    def test_row_filter_struct_field_not_found(self, city_df: pl.DataFrame) -> None:
        with pytest.raises(ValueError, match="not found in struct"):
            generate_nested_filter_expr(
                {"info": {"missing_field": lambda x: x > 0}}, city_df.schema
            )

    def test_row_filter_list_non_struct_inner_with_dict(self) -> None:
        df = pl.DataFrame({"nums": [[1, 2, 3]]})
        with pytest.raises(ValueError, match="element type to be a Struct"):
            generate_nested_filter_expr(
                {"nums": {"value": lambda x: x > 0}}, df.schema
            )

    def test_element_filter_non_list_column(self, city_df: pl.DataFrame) -> None:
        with pytest.raises(ValueError, match="List columns"):
            filter_nested_elements(city_df, {"population": lambda x: x > 0})

    def test_element_filter_column_not_found(self, city_df: pl.DataFrame) -> None:
        with pytest.raises(ValueError, match="not found in schema"):
            filter_nested_elements(city_df, {"missing": lambda x: x > 0})

    def test_element_filter_invalid_spec_type(self, city_df: pl.DataFrame) -> None:
        with pytest.raises(TypeError, match="Invalid element filter spec type"):
            filter_nested_elements(
                city_df,
                {"streets": 99},  # type: ignore[dict-item]
            )

    def test_element_filter_struct_field_not_found(self, city_df: pl.DataFrame) -> None:
        with pytest.raises(ValueError, match="not found in struct element"):
            filter_nested_elements(
                city_df, {"streets": {"missing_field": lambda x: x > 0}}
            )

"""Tests for the nested-to-flat expression translation."""

from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from nexpresso.normalized_packer import _translate_nested_to_flat


class TestTranslateNestedToFlat:
    """Test _translate_nested_to_flat() roundtrip translation."""

    def test_simple_field_reference(self) -> None:
        """pl.element().struct.field("x") → pl.col("x")."""
        nested = pl.element().struct.field("x")
        flat = _translate_nested_to_flat(nested)

        df = pl.DataFrame({"x": [1, 2, 3]})
        result = df.select(flat)
        assert result["x"].to_list() == [1, 2, 3]

    def test_field_reference_with_prefix(self) -> None:
        """Translation with prefix prepends it to column names."""
        nested = pl.element().struct.field("revenue")
        flat = _translate_nested_to_flat(nested, prefix="store.")

        df = pl.DataFrame({"store.revenue": [100, 200]})
        result = df.select(flat)
        assert result["store.revenue"].to_list() == [100, 200]

    def test_comparison_gt(self) -> None:
        """pl.element().struct.field("x") > 5 translates correctly."""
        nested = pl.element().struct.field("x") > 5
        flat = _translate_nested_to_flat(nested)

        df = pl.DataFrame({"x": [3, 7, 5, 10]})
        result = df.filter(flat)
        assert result["x"].to_list() == [7, 10]

    def test_comparison_eq(self) -> None:
        """Equality comparison translates correctly."""
        nested = pl.element().struct.field("name") == "Alice"
        flat = _translate_nested_to_flat(nested)

        df = pl.DataFrame({"name": ["Alice", "Bob", "Alice"]})
        result = df.filter(flat)
        assert result.height == 2

    def test_compound_and(self) -> None:
        """Compound (a > 1) & (b < 10) translates correctly."""
        nested = (pl.element().struct.field("a") > 1) & (
            pl.element().struct.field("b") < 10
        )
        flat = _translate_nested_to_flat(nested)

        df = pl.DataFrame({"a": [0, 5, 3], "b": [5, 15, 3]})
        result = df.filter(flat)
        assert result["a"].to_list() == [3]

    def test_compound_or(self) -> None:
        """Compound (a > 10) | (b == 1) translates correctly."""
        nested = (pl.element().struct.field("a") > 10) | (
            pl.element().struct.field("b") == 1
        )
        flat = _translate_nested_to_flat(nested)

        df = pl.DataFrame({"a": [0, 15, 3], "b": [1, 5, 1]})
        result = df.filter(flat)
        # Row 0: a=0 not >10, b=1 ==1 → pass
        # Row 1: a=15 >10 → pass
        # Row 2: a=3 not >10, b=1 ==1 → pass
        assert result.height == 3

    def test_arithmetic_mul(self) -> None:
        """pl.element().struct.field("a") * pl.element().struct.field("b")."""
        nested = pl.element().struct.field("a") * pl.element().struct.field("b")
        flat = _translate_nested_to_flat(nested)

        df = pl.DataFrame({"a": [2, 3], "b": [5, 4]})
        result = df.select(flat.alias("product"))
        assert result["product"].to_list() == [10, 12]

    def test_arithmetic_in_condition(self) -> None:
        """(a * b) > 10 translates correctly."""
        nested = (
            pl.element().struct.field("price") * pl.element().struct.field("qty")
        ) > 10
        flat = _translate_nested_to_flat(nested)

        df = pl.DataFrame({"price": [2, 5, 3], "qty": [3, 4, 2]})
        result = df.filter(flat)
        # price*qty: 6, 20, 6 — only 20 > 10
        assert result.height == 1
        assert result["price"].to_list() == [5]

    def test_negation(self) -> None:
        """~(x > 5) translates correctly."""
        nested = ~(pl.element().struct.field("x") > 5)
        flat = _translate_nested_to_flat(nested)

        df = pl.DataFrame({"x": [3, 7, 5, 10]})
        result = df.filter(flat)
        assert result["x"].to_list() == [3, 5]

    def test_nested_and_flat_produce_same_filter_result(self) -> None:
        """The same condition filters identically on nested and flat data."""
        data = [{"x": 1, "y": "a"}, {"x": 5, "y": "b"}, {"x": 3, "y": "c"}]

        # Flat
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

    def test_literal_only_expression(self) -> None:
        """Expression with no element references passes through."""
        # Just a literal — should round-trip without error
        nested = pl.lit(True)
        flat = _translate_nested_to_flat(nested)

        df = pl.DataFrame({"x": [1, 2, 3]})
        result = df.filter(flat)
        assert result.height == 3

    def test_mixed_literal_and_field(self) -> None:
        """Mixing field refs and literals works correctly."""
        nested = pl.element().struct.field("x") + 10
        flat = _translate_nested_to_flat(nested)

        df = pl.DataFrame({"x": [1, 2, 3]})
        result = df.select(flat.alias("result"))
        assert result["result"].to_list() == [11, 12, 13]


class TestTranslationWithPrefix:
    """Test prefix handling in expression translation."""

    def test_prefix_applied_to_all_fields(self) -> None:
        """All field references get the prefix."""
        nested = (pl.element().struct.field("a") > pl.element().struct.field("b"))
        flat = _translate_nested_to_flat(nested, prefix="level.")

        df = pl.DataFrame({"level.a": [5, 1], "level.b": [3, 7]})
        result = df.filter(flat)
        assert result.height == 1
        assert result["level.a"].to_list() == [5]

    def test_empty_prefix(self) -> None:
        """Empty prefix produces unqualified column names."""
        nested = pl.element().struct.field("x") > 0
        flat = _translate_nested_to_flat(nested, prefix="")

        df = pl.DataFrame({"x": [1, -1, 0]})
        result = df.filter(flat)
        assert result["x"].to_list() == [1]

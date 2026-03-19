"""Tests for the LevelExpr backend-agnostic expression DSL."""

from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from nexpresso.level_expr import (
    ArithmeticExpr,
    ComparisonExpr,
    F,
    FieldRef,
    LiteralExpr,
    LogicalExpr,
    NotExpr,
)

# ============================================================================
# Construction
# ============================================================================


class TestConstruction:
    """Test building expression trees with the F() constructor."""

    def test_field_ref(self) -> None:
        ref = F("revenue")
        assert isinstance(ref, FieldRef)
        assert ref.name == "revenue"

    def test_comparison_gt(self) -> None:
        expr = F("x") > 5
        assert isinstance(expr, ComparisonExpr)
        assert expr.op == ">"

    def test_comparison_lt(self) -> None:
        expr = F("x") < 5
        assert isinstance(expr, ComparisonExpr)
        assert expr.op == "<"

    def test_comparison_ge(self) -> None:
        expr = F("x") >= 5
        assert isinstance(expr, ComparisonExpr)
        assert expr.op == ">="

    def test_comparison_le(self) -> None:
        expr = F("x") <= 5
        assert isinstance(expr, ComparisonExpr)
        assert expr.op == "<="

    def test_comparison_eq(self) -> None:
        expr = F("x") == 5
        assert isinstance(expr, ComparisonExpr)
        assert expr.op == "=="

    def test_comparison_ne(self) -> None:
        expr = F("x") != 5
        assert isinstance(expr, ComparisonExpr)
        assert expr.op == "!="

    def test_arithmetic_add(self) -> None:
        expr = F("x") + F("y")
        assert isinstance(expr, ArithmeticExpr)
        assert expr.op == "+"

    def test_arithmetic_sub(self) -> None:
        expr = F("x") - 1
        assert isinstance(expr, ArithmeticExpr)
        assert expr.op == "-"

    def test_arithmetic_mul(self) -> None:
        expr = F("price") * F("qty")
        assert isinstance(expr, ArithmeticExpr)
        assert expr.op == "*"

    def test_arithmetic_div(self) -> None:
        expr = F("total") / 2
        assert isinstance(expr, ArithmeticExpr)
        assert expr.op == "/"

    def test_radd(self) -> None:
        expr = 10 + F("x")
        assert isinstance(expr, ArithmeticExpr)
        assert isinstance(expr.left, LiteralExpr)
        assert expr.left.value == 10

    def test_rmul(self) -> None:
        expr = 2 * F("x")
        assert isinstance(expr, ArithmeticExpr)
        assert isinstance(expr.left, LiteralExpr)
        assert expr.left.value == 2

    def test_logical_and(self) -> None:
        expr = (F("x") > 1) & (F("y") < 10)
        assert isinstance(expr, LogicalExpr)
        assert expr.op == "and"

    def test_logical_or(self) -> None:
        expr = (F("x") > 1) | (F("y") < 10)
        assert isinstance(expr, LogicalExpr)
        assert expr.op == "or"

    def test_not(self) -> None:
        expr = ~(F("x") > 5)
        assert isinstance(expr, NotExpr)

    def test_complex_expression(self) -> None:
        """Test composing multiple operators into a complex tree."""
        expr = ((F("price") * F("qty")) > 500) & (F("active") == True)  # noqa: E712
        assert isinstance(expr, LogicalExpr)
        assert isinstance(expr.left, ComparisonExpr)
        assert isinstance(expr.left.left, ArithmeticExpr)


# ============================================================================
# Compilation to nested expressions
# ============================================================================


class TestNestedCompilation:
    """Test compiling to pl.element().struct.field(...) expressions."""

    def test_field_ref_nested(self) -> None:
        """FieldRef compiles to pl.element().struct.field(name)."""
        df = pl.DataFrame(
            {"items": [[{"x": 1, "y": 2}, {"x": 3, "y": 4}]]}
        )
        expr = F("x").to_nested_expr()
        result = df.select(pl.col("items").list.eval(expr))
        assert result["items"].to_list() == [[1, 3]]

    def test_comparison_nested(self) -> None:
        """Comparison compiles correctly in nested context."""
        df = pl.DataFrame(
            {"items": [[{"x": 1}, {"x": 5}, {"x": 10}]]}
        )
        cond = (F("x") > 3).to_nested_expr()
        result = df.select(
            pl.col("items").list.eval(cond.cast(pl.UInt8)).list.sum()
        )
        assert result["items"].to_list() == [2]

    def test_arithmetic_nested(self) -> None:
        """Arithmetic compiles correctly in nested context."""
        df = pl.DataFrame(
            {"items": [[{"price": 10.0, "qty": 3}, {"price": 5.0, "qty": 2}]]}
        )
        expr = (F("price") * F("qty")).to_nested_expr()
        result = df.select(pl.col("items").list.eval(expr))
        assert result["items"].to_list() == [[30.0, 10.0]]

    def test_logical_and_nested(self) -> None:
        """Logical AND compiles correctly in nested context."""
        df = pl.DataFrame(
            {"items": [[{"x": 5, "y": 2}, {"x": 1, "y": 8}, {"x": 5, "y": 8}]]}
        )
        cond = ((F("x") > 3) & (F("y") > 5)).to_nested_expr()
        result = df.select(
            pl.col("items").list.eval(cond.cast(pl.UInt8)).list.sum()
        )
        # Only {x: 5, y: 8} satisfies both conditions
        assert result["items"].to_list() == [1]

    def test_logical_or_nested(self) -> None:
        """Logical OR compiles correctly in nested context."""
        df = pl.DataFrame(
            {"items": [[{"x": 1, "y": 2}, {"x": 5, "y": 1}]]}
        )
        cond = ((F("x") > 3) | (F("y") > 1)).to_nested_expr()
        result = df.select(
            pl.col("items").list.eval(cond.cast(pl.UInt8)).list.sum()
        )
        # Both elements satisfy at least one condition
        assert result["items"].to_list() == [2]

    def test_not_nested(self) -> None:
        """Negation compiles correctly in nested context."""
        df = pl.DataFrame(
            {"items": [[{"x": 1}, {"x": 5}]]}
        )
        cond = (~(F("x") > 3)).to_nested_expr()
        result = df.select(
            pl.col("items").list.eval(cond.cast(pl.UInt8)).list.sum()
        )
        assert result["items"].to_list() == [1]

    def test_literal_nested(self) -> None:
        """Literal values compile to pl.lit()."""
        expr = LiteralExpr(42).to_nested_expr()
        df = pl.DataFrame({"x": [1]})
        result = df.select(expr)
        assert result["literal"].to_list() == [42]


# ============================================================================
# Compilation to flat expressions
# ============================================================================


class TestFlatCompilation:
    """Test compiling to pl.col(...) expressions."""

    def test_field_ref_flat_no_prefix(self) -> None:
        """FieldRef without prefix compiles to pl.col(name)."""
        df = pl.DataFrame({"revenue": [100, 200, 300]})
        expr = F("revenue").to_flat_expr()
        result = df.select(expr)
        assert result["revenue"].to_list() == [100, 200, 300]

    def test_field_ref_flat_with_prefix(self) -> None:
        """FieldRef with prefix compiles to pl.col(prefix + name)."""
        df = pl.DataFrame({"store.revenue": [100, 200]})
        expr = F("revenue").to_flat_expr(prefix="store.")
        result = df.select(expr)
        assert result["store.revenue"].to_list() == [100, 200]

    def test_comparison_flat(self) -> None:
        """Comparison compiles correctly on flat data."""
        df = pl.DataFrame({"revenue": [50, 150, 250]})
        cond = (F("revenue") > 100).to_flat_expr()
        result = df.filter(cond)
        assert result["revenue"].to_list() == [150, 250]

    def test_arithmetic_flat(self) -> None:
        """Arithmetic compiles correctly on flat data."""
        df = pl.DataFrame({"price": [10.0, 5.0], "qty": [3, 2]})
        expr = (F("price") * F("qty")).to_flat_expr()
        result = df.select(expr)
        assert result["price"].to_list() == [30.0, 10.0]

    def test_comparison_flat_with_prefix(self) -> None:
        """Comparison with prefix compiles correctly."""
        df = pl.DataFrame({"store.revenue": [50, 150, 250]})
        cond = (F("revenue") > 100).to_flat_expr(prefix="store.")
        result = df.filter(cond)
        assert result["store.revenue"].to_list() == [150, 250]

    def test_logical_flat(self) -> None:
        """Logical ops compile correctly on flat data."""
        df = pl.DataFrame({"x": [1, 5, 3], "y": [10, 2, 8]})
        cond = ((F("x") > 2) & (F("y") > 5)).to_flat_expr()
        result = df.filter(cond)
        assert result["x"].to_list() == [3]

    def test_not_flat(self) -> None:
        """Negation compiles correctly on flat data."""
        df = pl.DataFrame({"x": [1, 5, 3]})
        cond = (~(F("x") > 3)).to_flat_expr()
        result = df.filter(cond)
        assert result["x"].to_list() == [1, 3]

    def test_scalar_on_left(self) -> None:
        """Scalar on left side of arithmetic (radd/rmul)."""
        df = pl.DataFrame({"x": [2, 4]})
        expr = (10 + F("x")).to_flat_expr()
        result = df.select(expr)
        assert result["literal"].to_list() == [12, 14]


# ============================================================================
# Repr
# ============================================================================


class TestRepr:
    """Test string representations."""

    def test_field_ref_repr(self) -> None:
        assert repr(F("x")) == "F('x')"

    def test_comparison_repr(self) -> None:
        assert repr(F("x") > 5) == "(F('x') > Lit(5))"

    def test_logical_repr(self) -> None:
        expr = (F("x") > 1) & (F("y") < 10)
        assert repr(expr) == "((F('x') > Lit(1)) & (F('y') < Lit(10)))"

    def test_not_repr(self) -> None:
        expr = ~(F("x") > 5)
        assert repr(expr) == "~(F('x') > Lit(5))"


# ============================================================================
# Cross-compilation equivalence
# ============================================================================


class TestCrossCompilationEquivalence:
    """Verify that nested and flat compilation produce equivalent results on
    the same underlying data."""

    def test_filter_equivalence(self) -> None:
        """Same condition applied nested vs flat produces same matches."""
        data = [
            {"id": 1, "value": 10},
            {"id": 2, "value": 50},
            {"id": 3, "value": 90},
        ]

        # Flat DataFrame
        flat_df = pl.DataFrame(data)
        flat_cond = (F("value") > 30).to_flat_expr()
        flat_result = flat_df.filter(flat_cond).sort("id")

        # Nested: wrap as single-row list-of-struct, filter, extract
        nested_df = pl.DataFrame({"items": [data]})
        nested_cond = (F("value") > 30).to_nested_expr()
        # Use list.eval to get boolean mask, then filter items
        filtered = nested_df.select(
            pl.col("items").list.eval(
                pl.when(nested_cond).then(pl.element()).otherwise(None)
            )
        ).explode("items").drop_nulls().unnest("items").sort("id")

        assert_frame_equal(flat_result, filtered)

    def test_arithmetic_equivalence(self) -> None:
        """Same arithmetic applied nested vs flat produces same values."""
        data = [{"price": 10.0, "qty": 3}, {"price": 5.0, "qty": 2}]

        flat_df = pl.DataFrame(data)
        flat_expr = (F("price") * F("qty")).to_flat_expr()
        flat_result = flat_df.select(flat_expr)["price"].to_list()

        nested_df = pl.DataFrame({"items": [data]})
        nested_expr = (F("price") * F("qty")).to_nested_expr()
        nested_result = nested_df.select(
            pl.col("items").list.eval(nested_expr)
        )["items"].to_list()[0]

        assert flat_result == nested_result

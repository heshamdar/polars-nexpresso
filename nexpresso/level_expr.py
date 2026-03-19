"""
Lightweight expression DSL for backend-agnostic hierarchy conditions.

This module provides a small expression tree that can be compiled to either
nested Polars expressions (``pl.element().struct.field(...)``) or flat
column-based expressions (``pl.col(...)``), depending on whether the
underlying backend stores data as nested structs or normalized tables.

The primary user-facing entry point is the :func:`F` constructor::

    from nexpresso import F

    condition = F("revenue") > 100_000
    combined  = (F("price") * F("qty")) > 500

These expressions are accepted by any :class:`~nexpresso.hierarchy_protocol.HierarchyOperator`
implementation and compiled to the appropriate Polars expression at execution time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

# ---------------------------------------------------------------------------
# Public type alias – anything that can appear as a LevelExpr operand.
# ---------------------------------------------------------------------------
Scalar = int | float | str | bool | None


# ============================================================================
# Base class
# ============================================================================


class LevelExpr:
    """Base class for backend-agnostic hierarchy expressions.

    Subclasses represent different expression nodes (field references,
    comparisons, arithmetic, logical combinators).  Each node knows how to
    compile itself into two flavours of ``pl.Expr``:

    * **nested** – for use inside ``list.eval()`` on packed ``List[Struct]``
      columns, where fields are accessed via ``pl.element().struct.field(name)``.
    * **flat** – for use on joined/flat tables, where fields are accessed via
      ``pl.col(prefix + name)``.
    """

    # -- comparison operators ------------------------------------------------

    def __gt__(self, other: LevelExpr | Scalar) -> ComparisonExpr:
        return ComparisonExpr(self, _wrap(other), ">")

    def __ge__(self, other: LevelExpr | Scalar) -> ComparisonExpr:
        return ComparisonExpr(self, _wrap(other), ">=")

    def __lt__(self, other: LevelExpr | Scalar) -> ComparisonExpr:
        return ComparisonExpr(self, _wrap(other), "<")

    def __le__(self, other: LevelExpr | Scalar) -> ComparisonExpr:
        return ComparisonExpr(self, _wrap(other), "<=")

    def __eq__(self, other: object) -> ComparisonExpr:  # type: ignore[override]
        return ComparisonExpr(self, _wrap(other), "==")

    def __ne__(self, other: object) -> ComparisonExpr:  # type: ignore[override]
        return ComparisonExpr(self, _wrap(other), "!=")

    # -- arithmetic operators ------------------------------------------------

    def __add__(self, other: LevelExpr | Scalar) -> ArithmeticExpr:
        return ArithmeticExpr(self, _wrap(other), "+")

    def __radd__(self, other: LevelExpr | Scalar) -> ArithmeticExpr:
        return ArithmeticExpr(_wrap(other), self, "+")

    def __sub__(self, other: LevelExpr | Scalar) -> ArithmeticExpr:
        return ArithmeticExpr(self, _wrap(other), "-")

    def __rsub__(self, other: LevelExpr | Scalar) -> ArithmeticExpr:
        return ArithmeticExpr(_wrap(other), self, "-")

    def __mul__(self, other: LevelExpr | Scalar) -> ArithmeticExpr:
        return ArithmeticExpr(self, _wrap(other), "*")

    def __rmul__(self, other: LevelExpr | Scalar) -> ArithmeticExpr:
        return ArithmeticExpr(_wrap(other), self, "*")

    def __truediv__(self, other: LevelExpr | Scalar) -> ArithmeticExpr:
        return ArithmeticExpr(self, _wrap(other), "/")

    def __rtruediv__(self, other: LevelExpr | Scalar) -> ArithmeticExpr:
        return ArithmeticExpr(_wrap(other), self, "/")

    # -- logical operators ---------------------------------------------------

    def __and__(self, other: LevelExpr) -> LogicalExpr:
        return LogicalExpr(self, other, "and")

    def __or__(self, other: LevelExpr) -> LogicalExpr:
        return LogicalExpr(self, other, "or")

    def __invert__(self) -> NotExpr:
        return NotExpr(self)

    # -- compilation ---------------------------------------------------------

    def to_nested_expr(self) -> pl.Expr:
        """Compile to a Polars expression for use inside ``list.eval()`` on
        packed ``List[Struct]`` columns.

        Field references become ``pl.element().struct.field(name)``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement to_nested_expr()"
        )

    def to_flat_expr(self, prefix: str = "") -> pl.Expr:
        """Compile to a Polars expression for use on flat/joined tables.

        Field references become ``pl.col(prefix + name)``.

        Args:
            prefix: Optional column name prefix (e.g. ``"store."``).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement to_flat_expr()"
        )


# ============================================================================
# Leaf nodes
# ============================================================================


@dataclass(frozen=True, eq=False)
class FieldRef(LevelExpr):
    """Reference to a field at a hierarchy level.

    Args:
        name: Unqualified field name (e.g. ``"revenue"``).

    Examples:
        >>> ref = FieldRef("revenue")
        >>> ref.to_nested_expr()   # pl.element().struct.field("revenue")
        >>> ref.to_flat_expr("store.")  # pl.col("store.revenue")
    """

    name: str

    def to_nested_expr(self) -> pl.Expr:
        return pl.element().struct.field(self.name)

    def to_flat_expr(self, prefix: str = "") -> pl.Expr:
        return pl.col(f"{prefix}{self.name}")

    def __repr__(self) -> str:
        return f"F({self.name!r})"


@dataclass(frozen=True, eq=False)
class LiteralExpr(LevelExpr):
    """A scalar literal value.

    Args:
        value: The literal Python value.
    """

    value: Any

    def to_nested_expr(self) -> pl.Expr:
        return pl.lit(self.value)

    def to_flat_expr(self, prefix: str = "") -> pl.Expr:
        return pl.lit(self.value)

    def __repr__(self) -> str:
        return f"Lit({self.value!r})"


# ============================================================================
# Composite nodes
# ============================================================================

# Mapping from operator token to the corresponding Polars Expr method name.
_CMP_OPS: dict[str, str] = {
    ">": "__gt__",
    ">=": "__ge__",
    "<": "__lt__",
    "<=": "__le__",
    "==": "__eq__",
    "!=": "__ne__",
}

_ARITH_OPS: dict[str, str] = {
    "+": "__add__",
    "-": "__sub__",
    "*": "__mul__",
    "/": "__truediv__",
}


@dataclass(frozen=True, eq=False)
class ComparisonExpr(LevelExpr):
    """Binary comparison (e.g. ``F("revenue") > 100_000``).

    Args:
        left: Left-hand operand.
        right: Right-hand operand.
        op: Comparison operator string (``">"`` etc.).
    """

    left: LevelExpr
    right: LevelExpr
    op: str

    def to_nested_expr(self) -> pl.Expr:
        method = _CMP_OPS[self.op]
        result: pl.Expr = getattr(self.left.to_nested_expr(), method)(
            self.right.to_nested_expr()
        )
        return result

    def to_flat_expr(self, prefix: str = "") -> pl.Expr:
        method = _CMP_OPS[self.op]
        result: pl.Expr = getattr(self.left.to_flat_expr(prefix), method)(
            self.right.to_flat_expr(prefix)
        )
        return result

    def __repr__(self) -> str:
        return f"({self.left!r} {self.op} {self.right!r})"


@dataclass(frozen=True, eq=False)
class ArithmeticExpr(LevelExpr):
    """Binary arithmetic (e.g. ``F("price") * F("qty")``).

    Args:
        left: Left-hand operand.
        right: Right-hand operand.
        op: Arithmetic operator string (``"+"`` etc.).
    """

    left: LevelExpr
    right: LevelExpr
    op: str

    def to_nested_expr(self) -> pl.Expr:
        method = _ARITH_OPS[self.op]
        result: pl.Expr = getattr(self.left.to_nested_expr(), method)(
            self.right.to_nested_expr()
        )
        return result

    def to_flat_expr(self, prefix: str = "") -> pl.Expr:
        method = _ARITH_OPS[self.op]
        result: pl.Expr = getattr(self.left.to_flat_expr(prefix), method)(
            self.right.to_flat_expr(prefix)
        )
        return result

    def __repr__(self) -> str:
        return f"({self.left!r} {self.op} {self.right!r})"


@dataclass(frozen=True, eq=False)
class LogicalExpr(LevelExpr):
    """Logical combination (e.g. ``(F("x") > 1) & (F("y") < 10)``).

    Args:
        left: Left-hand operand.
        right: Right-hand operand.
        op: ``"and"`` or ``"or"``.
    """

    left: LevelExpr
    right: LevelExpr
    op: str

    def to_nested_expr(self) -> pl.Expr:
        l_expr = self.left.to_nested_expr()
        r_expr = self.right.to_nested_expr()
        if self.op == "and":
            return l_expr & r_expr
        return l_expr | r_expr

    def to_flat_expr(self, prefix: str = "") -> pl.Expr:
        l_expr = self.left.to_flat_expr(prefix)
        r_expr = self.right.to_flat_expr(prefix)
        if self.op == "and":
            return l_expr & r_expr
        return l_expr | r_expr

    def __repr__(self) -> str:
        sym = "&" if self.op == "and" else "|"
        return f"({self.left!r} {sym} {self.right!r})"


@dataclass(frozen=True, eq=False)
class NotExpr(LevelExpr):
    """Logical negation (e.g. ``~(F("x") > 5)``).

    Args:
        inner: The expression to negate.
    """

    inner: LevelExpr

    def to_nested_expr(self) -> pl.Expr:
        return ~self.inner.to_nested_expr()

    def to_flat_expr(self, prefix: str = "") -> pl.Expr:
        return ~self.inner.to_flat_expr(prefix)

    def __repr__(self) -> str:
        return f"~{self.inner!r}"


# ============================================================================
# Convenience constructor
# ============================================================================


def F(name: str) -> FieldRef:  # noqa: N802 – intentionally uppercase for API clarity
    """Create a field reference for use in hierarchy conditions.

    This is the primary user-facing entry point for building backend-agnostic
    expressions.  The returned :class:`FieldRef` supports comparison,
    arithmetic, and logical operators, producing an expression tree that can
    be compiled to either nested or flat Polars expressions.

    Args:
        name: Unqualified field name (e.g. ``"revenue"``).

    Returns:
        A :class:`FieldRef` that can be combined with operators.

    Examples:
        >>> from nexpresso import F
        >>> cond = F("revenue") > 100_000
        >>> cond = (F("price") * F("qty")) > 500
        >>> cond = (F("x") > 1) & (F("y") < 10)
    """
    return FieldRef(name)


# ============================================================================
# Internal helpers
# ============================================================================


def _wrap(value: Any) -> LevelExpr:
    """Wrap a raw Python value into a :class:`LevelExpr` node.

    If *value* is already a :class:`LevelExpr`, return it unchanged.
    Otherwise wrap it as a :class:`LiteralExpr`.
    """
    if isinstance(value, LevelExpr):
        return value
    return LiteralExpr(value)


__all__ = [
    "LevelExpr",
    "FieldRef",
    "LiteralExpr",
    "ComparisonExpr",
    "ArithmeticExpr",
    "LogicalExpr",
    "NotExpr",
    "F",
]

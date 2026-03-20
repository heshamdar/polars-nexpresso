"""
Abstract interface for hierarchy backends (nested and normalized).

This module defines the :class:`HierarchyOperator` protocol — a common API
that both :class:`~nexpresso.hierarchical_packer.HierarchicalPacker` (nested
backend) and :class:`~nexpresso.normalized_packer.NormalizedPacker`
(normalized backend) can satisfy.

It also provides :class:`NestedBackend`, an adapter that wraps an existing
``HierarchicalPacker`` together with a packed frame to expose it through the
``HierarchyOperator`` interface.

Typical usage::

    import polars as pl
    from nexpresso import NestedBackend, NormalizedPacker

    # Use the SAME Polars expression with either backend
    cond = pl.element().struct.field("revenue") > 100_000

    # Normalized backend
    result_n = npacker.any_child_satisfies(
        from_level="store", to_level="region", condition=cond
    )

    # Nested backend (via adapter)
    backend = NestedBackend(packer, packed_frame)
    result_h = backend.any_child_satisfies(
        from_level="store", to_level="region", condition=cond
    )
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import polars as pl

from nexpresso.hierarchical_packer import (
    HierarchicalPacker,
    HierarchySpec,
    LevelAttribute,
    PromoteAggregation,
)

if TYPE_CHECKING:
    from nexpresso.expressions import FieldValue, StructMode

# ============================================================================
# Protocol
# ============================================================================


@runtime_checkable
class HierarchyOperator(Protocol):
    """Common interface for hierarchy backends.

    Both the nested (``HierarchicalPacker``) and normalized
    (``NormalizedPacker``) backends implement this protocol, enabling
    backend-agnostic code that works with either representation.

    All methods return ``pl.LazyFrame`` for consistency and to maximise
    Polars query optimisation.
    """

    @property
    def spec(self) -> HierarchySpec:
        """The underlying hierarchy specification."""
        ...

    @property
    def level_names(self) -> list[str]:
        """Level names from root (coarsest) to leaf (finest)."""
        ...

    @property
    def root_level(self) -> str:
        """Name of the coarsest (root) level."""
        ...

    @property
    def leaf_level(self) -> str:
        """Name of the finest (leaf) level."""
        ...

    def get_ancestor_levels(self, level: str) -> list[str]:
        """Ancestor level names above *level*, ordered root → parent."""
        ...

    def get_descendant_levels(self, level: str) -> list[str]:
        """Descendant level names below *level*, ordered child → leaf."""
        ...

    def get_level_keys(
        self,
        level: str,
        *,
        include_ancestors: bool = False,
        form: Literal["short", "long"] = "short",
    ) -> list[str]:
        """Identifying key column names for *level*."""
        ...

    def collect(self, level: str) -> pl.LazyFrame:
        """Return flat data at *level* granularity.

        For the nested backend this explodes/unnests; for the normalized
        backend this joins the relevant tables.
        """
        ...

    def promote_attribute(
        self,
        attribute: str,
        *,
        from_level: str,
        to_level: str,
        agg: PromoteAggregation = "list",
        alias: str | None = None,
    ) -> pl.LazyFrame:
        """Aggregate an attribute from a child level to its immediate parent."""
        ...

    def any_child_satisfies(
        self,
        *,
        from_level: str,
        to_level: str,
        condition: pl.Expr,
    ) -> pl.LazyFrame:
        """Filter to parent rows where at least one child matches *condition*.

        *condition* should use ``pl.element().struct.field()`` syntax.
        The nested backend passes it through directly; the normalized
        backend translates it to flat column references automatically.
        """
        ...

    def all_children_satisfy(
        self,
        *,
        from_level: str,
        to_level: str,
        condition: pl.Expr,
    ) -> pl.LazyFrame:
        """Filter to parent rows where every child matches *condition*.

        *condition* should use ``pl.element().struct.field()`` syntax.
        """
        ...

    def enrich(
        self,
        *specs: LevelAttribute,
        at_level: str,
    ) -> pl.LazyFrame:
        """Add cross-level attribute columns at *at_level* granularity."""
        ...

    def apply(
        self,
        fields: dict[str, FieldValue],
        *,
        at_level: str,
        struct_mode: StructMode = "with_fields",
    ) -> pl.LazyFrame:
        """Apply field transformations at a specific hierarchy level.

        Uses the same dict-based ``FieldValue`` syntax as
        ``generate_nested_exprs``.  Both backends accept the same spec
        and produce equivalent results.
        """
        ...

    def describe(self) -> str:
        """Human-readable summary of the hierarchy."""
        ...


# ============================================================================
# Nested backend adapter
# ============================================================================


class NestedBackend:
    """Adapter wrapping a :class:`HierarchicalPacker` and a packed frame.

    Exposes the :class:`HierarchyOperator` interface.  Conditions are
    standard Polars expressions using ``pl.element().struct.field()``
    syntax, which are passed through directly to the nested packer
    (no translation needed).

    The *frame* argument should be packed at **root level** (or at least at
    the coarsest level you intend to query).

    Args:
        packer: An existing :class:`HierarchicalPacker`.
        frame: A packed frame (``DataFrame`` or ``LazyFrame``).
    """

    def __init__(
        self,
        packer: HierarchicalPacker,
        frame: pl.DataFrame | pl.LazyFrame,
    ) -> None:
        self._packer = packer
        self._frame = frame

    # -- introspection -------------------------------------------------------

    @property
    def spec(self) -> HierarchySpec:
        return self._packer.spec

    @property
    def level_names(self) -> list[str]:
        return self._packer.level_names

    @property
    def root_level(self) -> str:
        return self._packer.root_level

    @property
    def leaf_level(self) -> str:
        return self._packer.leaf_level

    def get_ancestor_levels(self, level: str) -> list[str]:
        return self._packer.get_ancestor_levels(level)

    def get_descendant_levels(self, level: str) -> list[str]:
        return self._packer.get_descendant_levels(level)

    def get_level_keys(
        self,
        level: str,
        *,
        include_ancestors: bool = False,
        form: Literal["short", "long"] = "short",
    ) -> list[str]:
        return self._packer.get_level_keys(
            level, include_ancestors=include_ancestors, form=form
        )

    # -- data operations -----------------------------------------------------

    def collect(self, level: str) -> pl.LazyFrame:
        """Unpack the stored frame to *level* granularity."""
        lf = self._packer._to_lazy(self._frame)
        result = self._packer.unpack(lf, level)
        return self._packer._to_lazy(result)

    def _frame_at_level(self, level: str) -> pl.LazyFrame:
        """Get the stored data re-packed at *level* granularity.

        The stored frame may be packed at any level (typically root).
        This unpacks to the leaf and re-packs to the requested level so
        that operations like ``promote_attribute`` receive data in the
        expected shape.
        """
        lf = self._packer._to_lazy(self._frame)
        flat = self._packer.unpack(lf, self._packer.leaf_level)
        repacked = self._packer.pack(flat, level)
        return self._packer._to_lazy(repacked)

    def promote_attribute(
        self,
        attribute: str,
        *,
        from_level: str,
        to_level: str,
        agg: PromoteAggregation = "list",
        alias: str | None = None,
    ) -> pl.LazyFrame:
        # promote_attribute expects the frame at from_level granularity
        frame_at_from = self._frame_at_level(from_level)
        result = self._packer.promote_attribute(
            frame_at_from,
            attribute,
            from_level=from_level,
            to_level=to_level,
            agg=agg,
            alias=alias,
        )
        return self._packer._to_lazy(result)

    def any_child_satisfies(
        self,
        *,
        from_level: str,
        to_level: str,
        condition: pl.Expr,
    ) -> pl.LazyFrame:
        # Condition is already a pl.Expr in nested form — pass through
        # Pack at from_level so from_level appears as List[Struct] column
        frame_at_from = self._frame_at_level(from_level)
        result = self._packer.any_child_satisfies(
            frame_at_from,
            from_level=from_level,
            to_level=to_level,
            condition=condition,
        )
        return self._packer._to_lazy(result)

    def all_children_satisfy(
        self,
        *,
        from_level: str,
        to_level: str,
        condition: pl.Expr,
    ) -> pl.LazyFrame:
        # Condition is already a pl.Expr in nested form — pass through
        frame_at_from = self._frame_at_level(from_level)
        result = self._packer.all_children_satisfy(
            frame_at_from,
            from_level=from_level,
            to_level=to_level,
            condition=condition,
        )
        return self._packer._to_lazy(result)

    def enrich(
        self,
        *specs: LevelAttribute,
        at_level: str,
    ) -> pl.LazyFrame:
        # enrich expects frame packed at at_level, with immediate children as
        # nested List[Struct] columns.  Pack to the immediate child level of
        # at_level so the child data is visible.
        child_level = self._packer.spec.next_level(at_level)
        if child_level is None:
            raise ValueError(f"No child level below '{at_level}' to enrich from.")
        frame_at_child = self._frame_at_level(child_level.name)
        result = self._packer.enrich(frame_at_child, *specs, at_level=at_level)
        return self._packer._to_lazy(result)

    def apply(
        self,
        fields: dict[str, FieldValue],
        *,
        at_level: str,
        struct_mode: StructMode = "with_fields",
    ) -> pl.LazyFrame:
        # Convert to LazyFrame first to avoid FrameT variance issues
        lf = self._packer._to_lazy(self._frame)
        modified = self._packer.apply(
            lf, fields, at_level=at_level, struct_mode=struct_mode
        )
        # Unpack to the deepest level referenced in the field spec so
        # that child-level modifications are visible as flat columns,
        # matching the normalized backend's join behaviour.
        unpack_level = self._deepest_level(fields, at_level)
        result = self._packer.unpack(modified, unpack_level)
        return self._packer._to_lazy(result)

    def _deepest_level(
        self,
        fields: Mapping[str, object],
        at_level: str,
    ) -> str:
        """Return the deepest level name referenced in a nested field spec."""
        level_names = {lvl.name for lvl in self._packer.spec.levels}
        deepest = at_level

        for key, value in fields.items():
            if isinstance(value, dict) and key in level_names:
                child_deepest = self._deepest_level(value, key)
                if self._packer.spec.index_of(child_deepest) > self._packer.spec.index_of(
                    deepest
                ):
                    deepest = child_deepest

        return deepest

    def describe(self) -> str:
        return self._packer.describe()


__all__ = [
    "HierarchyOperator",
    "NestedBackend",
]

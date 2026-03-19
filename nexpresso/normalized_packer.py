"""
Normalized backend for hierarchical data operations.

This module provides :class:`NormalizedPacker`, which offers the same
declarative hierarchy API as :class:`~nexpresso.hierarchical_packer.HierarchicalPacker`
but operates on **separate normalized tables** rather than a single nested
frame.  Behind the scenes it performs lazy joins only when needed, leveraging
Polars predicate/projection pushdown for performance.

Typical usage::

    import polars as pl
    from nexpresso import NormalizedPacker, HierarchySpec, LevelSpec

    spec = HierarchySpec.from_levels(
        LevelSpec(name="region", id_fields=["id"]),
        LevelSpec(name="store", id_fields=["id"], parent_keys=["region_id"]),
    )

    npacker = NormalizedPacker(
        spec,
        tables={"region": regions_df, "store": stores_df},
    )

    # Flat data at store level (joins region + store lazily)
    store_data = npacker.collect("store")

    # Promote attribute without physical nesting
    result = npacker.promote_attribute(
        "revenue", from_level="store", to_level="region", agg="sum"
    )

    # Filter using standard Polars expressions (same syntax as HierarchicalPacker)
    filtered = npacker.any_child_satisfies(
        from_level="store", to_level="region",
        condition=pl.element().struct.field("revenue") > 100_000,
    )
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import polars as pl

from nexpresso.hierarchical_packer import (
    ColumnSelector,
    HierarchicalPacker,
    HierarchySpec,
    HierarchyValidationError,
    LevelAttribute,
    LevelMetadata,
    PromoteAggregation,
)

DEFAULT_SEPARATOR = "."
DEFAULT_ESCAPE_CHAR = "\\"


# ============================================================================
# Flat aggregation mapping (group_by equivalents of _LIST_AGGREGATIONS)
# ============================================================================

_FLAT_AGGREGATIONS: dict[PromoteAggregation, Callable[[pl.Expr], pl.Expr]] = {
    "list": lambda col: col,
    "set": lambda col: col.drop_nulls().unique(),
    "sum": lambda col: col.sum(),
    "mean": lambda col: col.mean(),
    "min": lambda col: col.min(),
    "max": lambda col: col.max(),
    "first": lambda col: col.first(),
    "last": lambda col: col.last(),
    "count": lambda col: col.count(),
    "single": lambda col: col.drop_nulls().unique().first(),
}


# ============================================================================
# Expression translation (nested → flat)
# ============================================================================


def _translate_nested_to_flat(expr: pl.Expr, prefix: str = "") -> pl.Expr:
    """Translate a nested Polars condition to a flat column-based condition.

    Converts ``pl.element().struct.field("x")`` patterns to ``pl.col("prefix.x")``
    by serializing the expression to JSON, walking the tree, and deserializing.
    This allows users to write the exact same Polars expressions for both nested
    and normalized backends.

    Args:
        expr: A Polars expression using ``pl.element().struct.field()`` syntax.
        prefix: Column name prefix to prepend to translated field names.

    Returns:
        A new ``pl.Expr`` with flat column references.
    """
    raw = expr.meta.serialize(format="json")
    tree = json.loads(raw)
    transformed = _walk_expr_tree(tree, prefix)
    return pl.Expr.deserialize(json.dumps(transformed).encode(), format="json")


def _walk_expr_tree(node: Any, prefix: str) -> Any:
    """Recursively walk a serialized expression tree.

    Replaces ``Element + StructExpr/FieldByName`` patterns with
    ``Column`` references using the given *prefix*.
    """
    if isinstance(node, dict):
        if "Function" in node:
            fn = node["Function"]
            if (
                fn.get("input") == ["Element"]
                and isinstance(fn.get("function"), dict)
                and "StructExpr" in fn["function"]
                and "FieldByName" in fn["function"]["StructExpr"]
            ):
                field_name: str = fn["function"]["StructExpr"]["FieldByName"]
                return {"Column": f"{prefix}{field_name}"}
        return {k: _walk_expr_tree(v, prefix) for k, v in node.items()}
    elif isinstance(node, list):
        return [_walk_expr_tree(item, prefix) for item in node]
    return node


# ============================================================================
# NormalizedPacker
# ============================================================================


class NormalizedPacker:
    """Hierarchical operations over normalized (separate) tables using lazy joins.

    This class provides the same high-level operations as
    :class:`~nexpresso.hierarchical_packer.HierarchicalPacker` — collecting
    data at a given level, promoting attributes, filtering by child
    conditions — but operates on separate per-level tables internally.  Joins
    are performed lazily via ``pl.LazyFrame``, allowing Polars to push down
    predicates and projections.

    Args:
        spec: The hierarchy specification defining levels and their relationships.
        tables: Mapping of ``{level_name: table}`` for each level in the
            hierarchy.  Tables should have unqualified column names (e.g.
            ``"id"`` not ``"region.id"``).  Child tables must include the
            ``parent_keys`` columns defined in their ``LevelSpec``.
        granularity_separator: Character(s) used to separate hierarchy levels
            in qualified column names.  Defaults to ``"."``.
        escape_char: Character used to escape the separator in field names
            that naturally contain it.  Defaults to ``"\\\\"``.
        join_type: Default join strategy when assembling tables.
            ``"left"`` preserves parents without children;
            ``"inner"`` drops them.  Defaults to ``"left"``.

    Raises:
        HierarchyValidationError: If required tables or columns are missing.
    """

    def __init__(
        self,
        spec: HierarchySpec,
        tables: Mapping[str, pl.LazyFrame | pl.DataFrame],
        *,
        granularity_separator: str = DEFAULT_SEPARATOR,
        escape_char: str = DEFAULT_ESCAPE_CHAR,
        join_type: Literal["left", "inner"] = "left",
    ) -> None:
        if escape_char == granularity_separator:
            raise ValueError(
                f"escape_char '{escape_char}' cannot be the same as "
                f"granularity_separator '{granularity_separator}'."
            )

        self.spec: HierarchySpec = spec
        self.separator: str = granularity_separator
        self.escape_char: str = escape_char
        self.join_type: Literal["left", "inner"] = join_type

        self._levels_meta: list[LevelMetadata] = self._build_metadata()

        # Store tables as lazy frames
        self._tables: dict[str, pl.LazyFrame] = {}
        for meta in self._levels_meta:
            if meta.name not in tables:
                raise HierarchyValidationError(
                    f"Missing table for level '{meta.name}'.",
                    level=meta.name,
                    details={"provided_levels": list(tables.keys())},
                )
            table = tables[meta.name]
            self._tables[meta.name] = (
                table if isinstance(table, pl.LazyFrame) else table.lazy()
            )

        self._validate_table_columns()

    # ------------------------------------------------------------------
    # Introspection Helpers
    # ------------------------------------------------------------------

    @property
    def level_names(self) -> list[str]:
        """Return all level names ordered from root (coarsest) to leaf (finest).

        Returns:
            List of level name strings.
        """
        return [m.name for m in self._levels_meta]

    @property
    def root_level(self) -> str:
        """Return the name of the coarsest (root) level."""
        return self._levels_meta[0].name

    @property
    def leaf_level(self) -> str:
        """Return the name of the finest (leaf) level."""
        return self._levels_meta[-1].name

    def get_ancestor_levels(self, level: str) -> list[str]:
        """Return all ancestor level names above *level*, ordered root → parent.

        Args:
            level: The level whose ancestors to retrieve.

        Returns:
            List of ancestor level names.  Empty list if *level* is the root.
        """
        idx = self.spec.index_of(level)
        return [m.name for m in self._levels_meta[:idx]]

    def get_descendant_levels(self, level: str) -> list[str]:
        """Return all descendant level names below *level*, ordered child → leaf.

        Args:
            level: The level whose descendants to retrieve.

        Returns:
            List of descendant level names.  Empty list if *level* is the leaf.
        """
        idx = self.spec.index_of(level)
        return [m.name for m in self._levels_meta[idx + 1 :]]

    def get_level_keys(
        self,
        level: str,
        *,
        include_ancestors: bool = False,
        form: Literal["short", "long"] = "short",
    ) -> list[str]:
        """Return the identifying key column names for *level*.

        Args:
            level: The level whose keys to retrieve.
            include_ancestors: If ``True``, also include all ancestor key
                columns.  Forces ``form="long"`` to avoid ambiguity.
            form: ``"short"`` for unqualified names, ``"long"`` for fully
                qualified paths.

        Returns:
            List of key column name strings.
        """
        meta = self._levels_meta[self.spec.index_of(level)]
        if include_ancestors:
            return list(meta.ancestor_keys) + list(meta.id_columns)
        if form == "long":
            return list(meta.id_columns)
        return [col[len(meta.prefix) :] for col in meta.id_columns]

    def get_level_fields(
        self,
        level: str,
        *,
        form: Literal["short", "long"] = "short",
    ) -> list[str]:
        """Return all column names for *level* from the underlying table.

        Args:
            level: The level whose fields to extract.
            form: ``"short"`` for unqualified names, ``"long"`` for
                qualified paths.

        Returns:
            List of field name strings.
        """
        meta = self._levels_meta[self.spec.index_of(level)]
        schema = self._tables[meta.name].collect_schema()
        if form == "long":
            return [f"{meta.prefix}{col}" for col in schema.keys()]
        return list(schema.keys())

    @property
    def tables(self) -> dict[str, pl.LazyFrame]:
        """Direct access to the underlying per-level tables (read-only copies).

        Returns:
            Dict mapping level names to their LazyFrames.
        """
        return dict(self._tables)

    def describe(self) -> str:
        """Return a human-readable summary of the hierarchy and tables.

        Returns:
            Multi-line string describing the hierarchy structure.
        """
        lines = ["NormalizedPacker Hierarchy:"]
        for meta in self._levels_meta:
            level_spec = self.spec.levels[meta.index]
            schema = self._tables[meta.name].collect_schema()
            indent = "  " * meta.index
            parent_info = ""
            if level_spec.parent_keys:
                parent_info = f" (parent_keys: {list(level_spec.parent_keys)})"
            lines.append(
                f"{indent}└─ {meta.name}: {list(schema.keys())}{parent_info}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Core Operations
    # ------------------------------------------------------------------

    def collect(self, level: str) -> pl.LazyFrame:
        """Return flat data at *level* granularity by joining tables lazily.

        Joins all tables from the root down to the requested level, producing
        a flat ``LazyFrame`` with fully qualified column names
        (e.g. ``"region.name"``, ``"region.store.revenue"``).

        Args:
            level: Target level.  All tables from root to *level* are joined.

        Returns:
            ``pl.LazyFrame`` at *level* granularity with qualified column names.

        Raises:
            KeyError: If *level* is not found in the hierarchy.
        """
        target_idx = self.spec.index_of(level)
        return self._join_to_level(target_idx)

    def promote_attribute(
        self,
        attribute: str,
        *,
        from_level: str,
        to_level: str,
        agg: PromoteAggregation = "list",
        alias: str | None = None,
    ) -> pl.LazyFrame:
        """Aggregate an attribute from a child level to its immediate parent.

        Performs a ``group_by`` on the child table's ``parent_keys`` and joins
        the aggregated result to the parent table.  No physical nesting is
        required.

        Args:
            attribute: Unqualified field name at *from_level*
                (e.g. ``"revenue"``).
            from_level: The level where the attribute lives.  Must be the
                immediate child of *to_level*.
            to_level: The coarser level to promote the attribute to.
            agg: Aggregation strategy.
            alias: Output column name (unqualified).  Defaults to *attribute*.

        Returns:
            ``pl.LazyFrame`` at *to_level* granularity with the promoted column.

        Raises:
            KeyError: If either level is not found.
            ValueError: If *from_level* is not the immediate child of *to_level*.
        """
        from_idx = self.spec.index_of(from_level)
        to_idx = self.spec.index_of(to_level)
        if from_idx != to_idx + 1:
            raise ValueError(
                f"from_level '{from_level}' must be the immediate child of "
                f"to_level '{to_level}'. Got indices {from_idx} and {to_idx}."
            )

        child_table = self._tables[from_level]
        child_schema = child_table.collect_schema()
        if attribute not in child_schema:
            raise ValueError(
                f"Attribute '{attribute}' not found in level '{from_level}' table. "
                f"Available columns: {list(child_schema.keys())}"
            )

        from_spec = self.spec.levels[from_idx]
        parent_keys = list(from_spec.parent_keys or [])
        if not parent_keys:
            raise HierarchyValidationError(
                f"Level '{from_level}' must have parent_keys defined.",
                level=from_level,
            )

        # Aggregate on child table
        # Use qualified name to avoid collisions with parent columns
        to_meta = self._levels_meta[to_idx]
        out_col = alias or attribute
        qualified_out = f"{to_meta.prefix}{self._escape_field(out_col)}"
        agg_fn = _FLAT_AGGREGATIONS[agg]
        agg_expr = agg_fn(pl.col(attribute)).alias(qualified_out)
        aggregated = child_table.group_by(parent_keys).agg(agg_expr)

        # Get parent table with qualified columns
        parent_qualified = self._prefix_table(to_meta)

        # Qualified parent id columns for join
        parent_id_qualified = list(to_meta.id_columns)

        # Join aggregated child to parent
        result = parent_qualified.join(
            aggregated,
            left_on=parent_id_qualified,
            right_on=parent_keys,
            how="left",
        )

        return result

    def any_child_satisfies(
        self,
        *,
        from_level: str,
        to_level: str,
        condition: pl.Expr,
    ) -> pl.LazyFrame:
        """Filter to parent rows where at least one child matches *condition*.

        *from_level* must be the immediate child of *to_level*.

        Args:
            from_level: Child level whose rows are tested.
            to_level: Parent level; one row per entity in the result.
            condition: A Polars expression using
                ``pl.element().struct.field()`` syntax (same as the nested
                backend).  Internally translated to flat column references.

        Returns:
            Filtered ``pl.LazyFrame`` at *to_level* granularity.
        """
        from_idx = self.spec.index_of(from_level)
        to_idx = self.spec.index_of(to_level)
        if from_idx != to_idx + 1:
            raise ValueError(
                f"from_level '{from_level}' must be the immediate child of "
                f"to_level '{to_level}'."
            )

        child_table = self._tables[from_level]
        from_spec = self.spec.levels[from_idx]
        parent_keys = list(from_spec.parent_keys or [])

        # Translate nested expression to flat column references (no prefix —
        # raw child columns)
        flat_cond = _translate_nested_to_flat(condition, prefix="")
        matching_children = child_table.filter(flat_cond)

        # Get distinct parent keys from matching children
        distinct_parent_keys = matching_children.select(parent_keys).unique()

        # Semi-join parent table
        to_meta = self._levels_meta[to_idx]
        parent_qualified = self._prefix_table(to_meta)
        parent_id_qualified = list(to_meta.id_columns)

        result = parent_qualified.join(
            distinct_parent_keys,
            left_on=parent_id_qualified,
            right_on=parent_keys,
            how="semi",
        )

        return result

    def all_children_satisfy(
        self,
        *,
        from_level: str,
        to_level: str,
        condition: pl.Expr,
    ) -> pl.LazyFrame:
        """Filter to parent rows where **every** child matches *condition*.

        Entities with no children pass the filter (vacuous truth).

        *from_level* must be the immediate child of *to_level*.

        Args:
            from_level: Child level whose rows are tested.
            to_level: Parent level; one row per entity in the result.
            condition: A Polars expression using
                ``pl.element().struct.field()`` syntax (same as the nested
                backend).  Internally translated to flat column references.

        Returns:
            Filtered ``pl.LazyFrame`` at *to_level* granularity.
        """
        from_idx = self.spec.index_of(from_level)
        to_idx = self.spec.index_of(to_level)
        if from_idx != to_idx + 1:
            raise ValueError(
                f"from_level '{from_level}' must be the immediate child of "
                f"to_level '{to_level}'."
            )

        child_table = self._tables[from_level]
        from_spec = self.spec.levels[from_idx]
        parent_keys = list(from_spec.parent_keys or [])

        flat_cond = _translate_nested_to_flat(condition, prefix="")

        # Count total children per parent
        total_counts = child_table.group_by(parent_keys).agg(
            pl.len().alias("__total")
        )

        # Count matching children per parent
        matching_counts = (
            child_table.filter(flat_cond)
            .group_by(parent_keys)
            .agg(pl.len().alias("__matching"))
        )

        # Join counts: parents where total == matching (or no children at all)
        counts = total_counts.join(matching_counts, on=parent_keys, how="left")
        counts = counts.with_columns(pl.col("__matching").fill_null(0))
        passing_parents = counts.filter(
            pl.col("__total") == pl.col("__matching")
        ).select(parent_keys)

        # Semi-join or anti-join parent table
        to_meta = self._levels_meta[to_idx]
        parent_qualified = self._prefix_table(to_meta)
        parent_id_qualified = list(to_meta.id_columns)

        # Parents that pass: those in passing_parents OR those with no children
        parents_with_children = total_counts.select(parent_keys)

        # Parents without children (vacuous truth)
        parents_no_children = parent_qualified.join(
            parents_with_children,
            left_on=parent_id_qualified,
            right_on=parent_keys,
            how="anti",
        )

        # Parents with all children satisfying
        parents_all_pass = parent_qualified.join(
            passing_parents,
            left_on=parent_id_qualified,
            right_on=parent_keys,
            how="semi",
        )

        result = pl.concat([parents_all_pass, parents_no_children])
        return result

    def enrich(
        self,
        *specs: LevelAttribute,
        at_level: str,
    ) -> pl.LazyFrame:
        """Add cross-level attribute columns at *at_level* granularity.

        Each :class:`~nexpresso.hierarchical_packer.LevelAttribute` is
        converted to a ``group_by`` + join operation on the appropriate
        child table.

        Args:
            *specs: One or more :class:`LevelAttribute` specs.
            at_level: The granularity level of the output.

        Returns:
            ``pl.LazyFrame`` at *at_level* with promoted columns appended.
        """
        at_idx = self.spec.index_of(at_level)
        at_meta = self._levels_meta[at_idx]

        result = self._prefix_table(at_meta)
        at_id_qualified = list(at_meta.id_columns)

        for spec in specs:
            from_idx = self.spec.index_of(spec.from_level)
            if from_idx != at_idx + 1:
                raise ValueError(
                    f"from_level '{spec.from_level}' must be the immediate "
                    f"child of at_level '{at_level}'."
                )

            child_table = self._tables[spec.from_level]
            from_spec = self.spec.levels[from_idx]
            parent_keys = list(from_spec.parent_keys or [])

            out_col = spec.alias or spec.attribute
            qualified_out = f"{at_meta.prefix}{self._escape_field(out_col)}"
            agg_fn = _FLAT_AGGREGATIONS[spec.agg]
            agg_expr = agg_fn(pl.col(spec.attribute)).alias(qualified_out)
            aggregated = child_table.group_by(parent_keys).agg(agg_expr)

            # Join to result
            result = result.join(
                aggregated,
                left_on=at_id_qualified,
                right_on=parent_keys,
                how="left",
            )

        return result

    def pack(self, to_level: str | None = None) -> pl.LazyFrame:
        """Build nested hierarchy by delegating to :class:`HierarchicalPacker`.

        This is the escape hatch for when you need physically nested data
        (e.g. for serialization or interop with nested-aware code).

        Args:
            to_level: Pack to this level (default: root level).

        Returns:
            ``pl.LazyFrame`` packed at *to_level* with nested ``List[Struct]``
            columns.
        """
        packer = HierarchicalPacker(
            self.spec,
            granularity_separator=self.separator,
            escape_char=self.escape_char,
        )
        # Convert internal lazy tables to dict for build_from_tables
        result = packer.build_from_tables(
            self._tables,
            target_level=to_level,
        )
        if isinstance(result, pl.DataFrame):
            return result.lazy()
        return result

    def validate(self, *, raise_on_error: bool = True) -> list[str]:
        """Validate referential integrity across tables.

        Checks:
        - Child ``parent_keys`` values have matching values in parent
          ``id_fields``.
        - Key columns are not null.

        Args:
            raise_on_error: If ``True``, raise on first error.

        Returns:
            List of error messages (empty if valid).

        Raises:
            HierarchyValidationError: If *raise_on_error* is ``True`` and
                validation fails.
        """
        errors: list[str] = []

        for idx, meta in enumerate(self._levels_meta):
            table = self._tables[meta.name]
            schema = table.collect_schema()
            level_spec = self.spec.levels[idx]

            # Check for null keys
            for id_field in level_spec.id_fields:
                if isinstance(id_field, str) and id_field in schema:
                    null_count = (
                        table.select(pl.col(id_field).is_null().sum())
                        .collect()
                        .item()
                    )
                    if null_count > 0:
                        msg = (
                            f"Level '{meta.name}': id_field '{id_field}' "
                            f"has {null_count} null values."
                        )
                        errors.append(msg)
                        if raise_on_error:
                            raise HierarchyValidationError(msg, level=meta.name)

            # Check referential integrity for child levels
            if idx > 0 and level_spec.parent_keys:
                parent_meta = self._levels_meta[idx - 1]
                parent_table = self._tables[parent_meta.name]
                parent_spec = self.spec.levels[idx - 1]
                parent_id_fields = [
                    f for f in parent_spec.id_fields if isinstance(f, str)
                ]

                for pk, pid in zip(
                    level_spec.parent_keys, parent_id_fields, strict=False
                ):
                    if pk not in schema:
                        continue
                    # Find orphan children
                    parent_ids = parent_table.select(pl.col(pid)).unique()
                    child_pks = table.select(pl.col(pk)).unique()
                    orphans = child_pks.join(
                        parent_ids,
                        left_on=pk,
                        right_on=pid,
                        how="anti",
                    )
                    orphan_count = orphans.collect().height
                    if orphan_count > 0:
                        msg = (
                            f"Level '{meta.name}': {orphan_count} orphan "
                            f"values in '{pk}' with no match in parent "
                            f"'{parent_meta.name}.{pid}'."
                        )
                        errors.append(msg)
                        if raise_on_error:
                            raise HierarchyValidationError(msg, level=meta.name)

        return errors

    def update_table(
        self,
        level: str,
        table: pl.LazyFrame | pl.DataFrame,
    ) -> None:
        """Replace the table for a specific level.

        Args:
            level: The level whose table to replace.
            table: The new table data.

        Raises:
            KeyError: If *level* is not found in the hierarchy.
        """
        self.spec.index_of(level)  # validate level exists
        self._tables[level] = (
            table if isinstance(table, pl.LazyFrame) else table.lazy()
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_metadata(self) -> list[LevelMetadata]:
        """Build per-level metadata (mirrors HierarchicalPacker._build_metadata)."""
        metas: list[LevelMetadata] = []
        path_components: list[str] = []
        ancestor_keys: list[str] = []

        for index, level in enumerate(self.spec.levels):
            path_components.append(level.name)
            path = self.separator.join(path_components)
            prefix = f"{path}{self.separator}" if path else ""

            id_columns, id_exprs = self._resolve_fields(index, level.id_fields, prefix)
            required_columns, required_exprs = self._resolve_fields(
                index, level.required_fields or (), prefix
            )

            metas.append(
                LevelMetadata(
                    index=index,
                    name=level.name,
                    path=path,
                    prefix=prefix,
                    ancestor_keys=tuple(ancestor_keys),
                    id_columns=tuple(id_columns),
                    id_exprs=tuple(id_exprs),
                    required_columns=tuple(required_columns),
                    required_exprs=tuple(required_exprs),
                    order_by=tuple(level.order_by or ()),
                )
            )

            ancestor_keys.extend(id_columns)

        return metas

    def _resolve_fields(
        self,
        level_idx: int,
        selectors: Sequence[ColumnSelector],
        prefix: str,
    ) -> tuple[list[str], list[pl.Expr]]:
        """Resolve field selectors into qualified column names and expressions."""
        columns: list[str] = []
        exprs: list[pl.Expr] = []

        for selector in selectors:
            if isinstance(selector, pl.Expr):
                alias = selector.meta.output_name()
                if alias is None:
                    raise ValueError(
                        f"Expression provided for level "
                        f"'{self.spec.levels[level_idx].name}' "
                        "must have an alias via .alias(...)."
                    )
                columns.append(alias)
                exprs.append(selector)
            else:
                escaped = self._escape_field(selector)
                columns.append(f"{prefix}{escaped}")

        return columns, exprs

    def _escape_field(self, name: str) -> str:
        """Escape separator characters in a field name."""
        escaped = name.replace(self.escape_char, self.escape_char + self.escape_char)
        return escaped.replace(self.separator, self.escape_char + self.separator)

    def _get_id_fields_short(self, level_idx: int) -> list[str]:
        """Get unqualified id field names for a level."""
        level_spec = self.spec.levels[level_idx]
        return [f for f in level_spec.id_fields if isinstance(f, str)]

    def _prefix_table(self, meta: LevelMetadata) -> pl.LazyFrame:
        """Return the table for a level with all columns qualified with the level prefix."""
        table = self._tables[meta.name]
        schema = table.collect_schema()
        rename_exprs = [
            pl.col(col).alias(f"{meta.prefix}{col}") for col in schema.keys()
        ]
        return table.select(rename_exprs)

    def _join_to_level(self, target_idx: int) -> pl.LazyFrame:
        """Join tables from root down to *target_idx*, returning qualified flat frame."""
        root_meta = self._levels_meta[0]
        result = self._prefix_table(root_meta)

        for level_idx in range(1, target_idx + 1):
            meta = self._levels_meta[level_idx]
            level_spec = self.spec.levels[level_idx]

            parent_meta = self._levels_meta[level_idx - 1]
            parent_id_qualified = list(parent_meta.id_columns)

            parent_keys = list(level_spec.parent_keys or [])
            if not parent_keys:
                raise HierarchyValidationError(
                    f"Level '{meta.name}' must have parent_keys defined "
                    "to join to parent.",
                    level=meta.name,
                )

            # Prefix child table columns
            child_qualified = self._prefix_table(meta)

            # Qualify parent keys in child table for join
            qualified_parent_keys = [f"{meta.prefix}{pk}" for pk in parent_keys]

            # Join child to accumulated result
            result = result.join(
                child_qualified,
                left_on=parent_id_qualified,
                right_on=qualified_parent_keys,
                how=self.join_type,
            )

        return result

    def _validate_table_columns(self) -> None:
        """Validate that required columns exist in each table."""
        for idx, meta in enumerate(self._levels_meta):
            table = self._tables[meta.name]
            schema = table.collect_schema()
            level_spec = self.spec.levels[idx]

            # Check id_fields exist
            for id_field in level_spec.id_fields:
                if isinstance(id_field, str) and id_field not in schema:
                    raise HierarchyValidationError(
                        f"Level '{meta.name}': id_field '{id_field}' "
                        f"not found in table. "
                        f"Available columns: {list(schema.keys())}",
                        level=meta.name,
                    )

            # Check parent_keys exist (for child levels)
            if level_spec.parent_keys:
                for pk in level_spec.parent_keys:
                    if pk not in schema:
                        raise HierarchyValidationError(
                            f"Level '{meta.name}': parent_key '{pk}' "
                            f"not found in table. "
                            f"Available columns: {list(schema.keys())}",
                            level=meta.name,
                        )


__all__ = [
    "NormalizedPacker",
    "_translate_nested_to_flat",
]

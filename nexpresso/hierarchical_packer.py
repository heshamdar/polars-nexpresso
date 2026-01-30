"""
Generic packing/unpacking helpers for hierarchically-structured Polars datasets.

Example
-------
>>> from nexpresso import HierarchicalPacker, HierarchySpec, LevelSpec
>>> spec = HierarchySpec(levels=[
...     LevelSpec(name="country", id_fields=["code"]),
...     LevelSpec(name="city", id_fields=["id"]),
... ])
>>> packer = HierarchicalPacker(spec)
>>> country_level = packer.pack(flat_df, "country")
>>> city_level = packer.unpack(country_level, "city")
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeVar

import polars as pl
from polars.expr.expr import Expr

FrameT = TypeVar("FrameT", pl.LazyFrame, pl.DataFrame)

ColumnSelector = str | pl.Expr
ExtraColumnsMode = Literal["preserve", "drop", "error"]
PromoteAggregation = Literal[
    "list", "set", "sum", "mean", "min", "max", "first", "last", "count", "single"
]

ROW_ID_COLUMN = "__hier_row_id"
DEFAULT_SEPARATOR = "."
DEFAULT_ESCAPE_CHAR = "\\"

__all__ = [
    "LevelSpec",
    "HierarchySpec",
    "HierarchicalPacker",
    "HierarchyValidationError",
    "PromoteAggregation",
]


class HierarchyValidationError(Exception):
    """
    Exception raised when hierarchy validation fails.

    Attributes:
        message: Human-readable error description.
        level: The hierarchy level where the error occurred.
        details: Additional context about the error.
    """

    def __init__(self, message: str, level: str | None = None, details: dict | None = None) -> None:
        """
        Initialize a HierarchyValidationError.

        Args:
            message: Human-readable error description.
            level: The hierarchy level where the error occurred.
            details: Additional context about the error.
        """
        self.level = level
        self.details = details or {}
        prefix = f"[Level: {level}] " if level else ""
        super().__init__(f"{prefix}{message}")


@dataclass(frozen=True)
class LevelSpec:
    """
    Declarative description of a hierarchy level.

    Args:
        name: Logical identifier for the level (e.g. ``"country"``). The final
            column path follows the convention ``parent.child`` determined by
            the ordering of levels in :class:`HierarchySpec`.
        id_fields: Columns or expressions that uniquely identify records at
            this level. Strings are treated as relative column names that will
            be qualified with the level path. Expressions must include an alias
            (via ``.alias(...)``) so that the derived column can be referenced.
        required_fields: Optional list of columns/expressions that must be
            non-null when emitting standalone tables via
            :meth:`HierarchicalPacker.split_levels`.
        order_by: Optional list of expressions that enforce deterministic
            ordering of children before grouping into list-of-struct columns.
        parent_keys: Column names in this level's table that link to the parent
            level's id_fields. Used when building hierarchies from normalized
            tables via :meth:`HierarchicalPacker.build_from_tables`. Order
            matters: ``parent_keys[i]`` joins to parent's ``id_fields[i]``.
    """

    name: str
    id_fields: Sequence[ColumnSelector] = ()
    required_fields: Sequence[ColumnSelector] | None = None
    order_by: Sequence[pl.Expr] | None = None
    parent_keys: Sequence[str] | None = None


@dataclass(frozen=True)
class LevelMetadata:
    index: int
    name: str
    path: str
    prefix: str
    ancestor_keys: tuple[str, ...]
    id_columns: tuple[str, ...]
    id_exprs: tuple[pl.Expr, ...]
    required_columns: tuple[str, ...]
    required_exprs: tuple[pl.Expr, ...]
    order_by: tuple[pl.Expr, ...]


@dataclass(frozen=True)
class HierarchySpec:
    """
    Collection of ``LevelSpec`` objects ordered from coarse → fine granularity.

    Args:
        levels: Sequence of LevelSpec objects from root to leaf.
        key_aliases: Mapping of {target_column: source_column} for aliasing keys.
    """

    levels: Sequence[LevelSpec]
    key_aliases: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that level names are unique."""
        level_names = [lvl.name for lvl in self.levels]
        if len(level_names) != len(set(level_names)):
            raise ValueError("Level names must be unique inside a HierarchySpec.")

    @classmethod
    def from_levels(
        cls, *levels: LevelSpec, key_aliases: Mapping[str, str] | None = None
    ) -> HierarchySpec:
        """
        Build a HierarchySpec from an ordered sequence of LevelSpec objects.

        This is a convenience constructor that validates compatibility between
        levels based on their parent_keys definitions.

        Args:
            *levels: LevelSpec objects ordered from root (coarsest) to leaf (finest).
            key_aliases: Optional mapping of {target_column: source_column}.

        Returns:
            A new HierarchySpec instance.

        Raises:
            ValueError: If parent_keys don't match parent's id_fields in count.
        """
        # Validate parent_keys compatibility
        for i, level in enumerate(levels):
            if i == 0:
                # Root level should not have parent_keys
                if level.parent_keys:
                    raise ValueError(
                        f"Root level '{level.name}' should not have parent_keys defined."
                    )
            else:
                parent = levels[i - 1]
                if level.parent_keys:
                    # Get parent's id_fields as strings (just count for validation)
                    parent_id_count = len(parent.id_fields)
                    if len(level.parent_keys) != parent_id_count:
                        raise ValueError(
                            f"Level '{level.name}' has {len(level.parent_keys)} parent_keys "
                            f"but parent '{parent.name}' has {parent_id_count} id_fields. "
                            "These must match."
                        )

        return cls(levels=list(levels), key_aliases=key_aliases or {})

    @property
    def levels_by_name(self) -> Mapping[str, LevelSpec]:
        """Get a mapping of level name to LevelSpec."""
        return {level.name: level for level in self.levels}

    def index_of(self, level_name: str) -> int:
        """
        Get the index of a level by name.

        Args:
            level_name: The name of the level to find.

        Returns:
            The zero-based index of the level.

        Raises:
            KeyError: If the level is not found.
        """
        for idx, level in enumerate(self.levels):
            if level.name == level_name:
                return idx
        raise KeyError(f"Level '{level_name}' not found in hierarchy.")

    def level(self, level_name: str) -> LevelSpec:
        """
        Get a LevelSpec by name.

        Args:
            level_name: The name of the level to get.

        Returns:
            The LevelSpec for the given name.
        """
        return self.levels[self.index_of(level_name)]

    def next_level(self, level_name: str) -> LevelSpec | None:
        """
        Get the next (child) level after the given level.

        Args:
            level_name: The name of the current level.

        Returns:
            The next LevelSpec, or None if this is the leaf level.
        """
        idx = self.index_of(level_name)
        if idx + 1 >= len(self.levels):
            return None
        return self.levels[idx + 1]


class HierarchicalPacker:
    """
    General-purpose helper for packing/unpacking nested hierarchies in Polars.

    The implementation assumes a configurable separator-based naming scheme and a
    strict tree (no cross-links). All behavior is driven by a ``HierarchySpec``
    instance.

    Args:
        spec: The hierarchy specification defining levels and their relationships.
        granularity_separator: Character(s) used to separate hierarchy levels in
            column names. Defaults to ".".
        escape_char: Character used to escape the separator in field names that
            naturally contain it. Defaults to "\\".
        preserve_child_order: Whether to maintain the original row order when
            packing children into list columns. Defaults to True.
        validate_on_pack: Whether to validate data integrity during pack operations.
            Defaults to True.
    """

    def __init__(
        self,
        spec: HierarchySpec,
        *,
        granularity_separator: str = DEFAULT_SEPARATOR,
        escape_char: str = DEFAULT_ESCAPE_CHAR,
        preserve_child_order: bool = True,
        validate_on_pack: bool = True,
    ) -> None:
        """
        Initialize the HierarchicalPacker.

        Args:
            spec: The hierarchy specification.
            granularity_separator: Separator for hierarchy levels in column names.
            escape_char: Character to escape separators in field names.
            preserve_child_order: Whether to maintain original row order.
            validate_on_pack: Whether to validate during pack operations.
        """
        if escape_char == granularity_separator:
            raise ValueError(
                f"escape_char '{escape_char}' cannot be the same as "
                f"granularity_separator '{granularity_separator}'."
            )

        self.spec: HierarchySpec = spec
        self.separator: str = granularity_separator
        self.escape_char: str = escape_char
        self.preserve_child_order: bool = preserve_child_order
        self.validate_on_pack: bool = validate_on_pack
        self._levels_meta: list[LevelMetadata] = self._build_metadata()
        self._computed_exprs: dict[str, Expr] = self._collect_computed_exprs()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def pack(
        self,
        frame: FrameT,
        to_level: str,
        *,
        extra_columns: ExtraColumnsMode = "preserve",
    ) -> FrameT:
        """
        Pack flattened columns down to ``to_level`` so that rows represent the
        requested granularity.

        Args:
            frame: The DataFrame or LazyFrame to pack.
            to_level: The target level name to pack down to.
            extra_columns: How to handle columns that don't belong to the hierarchy:
                - ``"preserve"``: Keep extra columns if they have uniform values
                  within each group (default). Raises error if values differ.
                - ``"drop"``: Silently drop extra columns.
                - ``"error"``: Raise an error if any extra columns are present.

        Returns:
            Packed frame with nested structures, same type as input.

        Raises:
            KeyError: If the level is not found in the hierarchy.
            HierarchyValidationError: If validation is enabled and data integrity
                issues are detected, or if extra_columns="error" and extra columns
                are present.
        """
        lf, added_cols, schema = self._prepare_frame(frame)

        # Identify and handle extra columns
        extra_cols = self._identify_extra_columns(schema)
        if extra_cols:
            if extra_columns == "error":
                raise HierarchyValidationError(
                    f"Found {len(extra_cols)} column(s) not part of the hierarchy: "
                    f"{extra_cols[:5]}{'...' if len(extra_cols) > 5 else ''}. "
                    "Use extra_columns='preserve' to keep them or 'drop' to remove them.",
                    details={"extra_columns": extra_cols},
                )
            elif extra_columns == "drop":
                lf = lf.drop(*extra_cols)
                schema = lf.collect_schema()

        target_idx = self.spec.index_of(to_level)
        for level_idx in reversed(range(target_idx, len(self._levels_meta))):
            lf, schema = self._pack_single_level(lf, level_idx, schema)

        if added_cols:
            lf = lf.drop(*added_cols)

        lf = self._drop_internal_columns(lf)
        return self._match_frame_type(lf, frame)

    def unpack(self, frame: FrameT, to_level: str) -> FrameT:
        """
        Unpack nested list-of-struct columns until ``to_level`` is reached,
        mirroring :func:`explode` + :func:`unnest` per level.

        Args:
            frame: The DataFrame or LazyFrame to unpack.
            to_level: The target level name to unpack to.

        Returns:
            Unpacked frame with flattened columns, same type as input.

        Raises:
            KeyError: If the level is not found in the hierarchy.
        """
        lf = self._to_lazy(frame)
        schema = lf.collect_schema()

        for level in self._levels_meta:
            if level.path not in schema:
                continue

            lf, schema = self._explode_and_unnest(lf, level, schema)
            if level.name == to_level:
                break

        lf = self._drop_internal_columns(lf)
        return self._match_frame_type(lf, frame)

    def split_levels(self, frame: FrameT) -> dict[str, FrameT]:
        """
        Split a packed frame into standalone tables—one per hierarchy level.

        Args:
            frame: The packed DataFrame or LazyFrame to split.

        Returns:
            Dictionary mapping level names to their respective tables.
        """
        lf, added_cols, schema = self._prepare_frame(frame)

        outputs: dict[str, pl.LazyFrame] = {}
        current = lf

        for level in self._levels_meta:
            if level.path not in schema:
                # Try to get updated schema
                schema = current.collect_schema()
                if level.path not in schema:
                    continue

            level_table = self.unpack(current, level.name)
            level_schema = level_table.collect_schema()
            output_table = level_table

            next_meta = (
                self._levels_meta[level.index + 1]
                if level.index + 1 < len(self._levels_meta)
                else None
            )
            if next_meta:
                drop_cols = [
                    col
                    for col in level_schema.keys()
                    if col.startswith(next_meta.prefix) or col == next_meta.path
                ]
                if drop_cols:
                    output_table = output_table.drop(*drop_cols)
                level_schema = output_table.collect_schema()
                next_drop_subset = [col for col in next_meta.ancestor_keys if col in level_schema]
                if next_drop_subset:
                    output_table = output_table.drop_nulls(subset=next_drop_subset)
            elif level.required_columns:
                level_schema = output_table.collect_schema()
                required_subset = [col for col in level.required_columns if col in level_schema]
                if required_subset:
                    output_table = output_table.drop_nulls(subset=required_subset)

            if added_cols:
                level_schema = output_table.collect_schema()
                drop_candidates = [col for col in added_cols if col in level_schema]
                if drop_candidates:
                    output_table = output_table.drop(*drop_candidates)

            outputs[level.name] = self._drop_internal_columns(output_table)
            current = level_table
            schema = level_schema

        # Match output types to input type
        if isinstance(frame, pl.DataFrame):
            return {name: tbl.collect() for name, tbl in outputs.items()}  # type: ignore[misc]
        return outputs  # type: ignore[return-value]

    def normalize(self, frame: FrameT, *, root_level: str | None = None) -> dict[str, FrameT]:
        """
        Convenience wrapper that packs to the root level and splits into
        normalized per-level tables.

        Args:
            frame: The DataFrame or LazyFrame to normalize.
            root_level: Optional root level to pack to (defaults to first level).

        Returns:
            Dictionary mapping level names to their respective normalized tables.
        """
        target = root_level or self._levels_meta[0].name
        packed = self.pack(frame, target)
        return self.split_levels(packed)

    def denormalize(
        self,
        tables: Mapping[str, pl.LazyFrame | pl.DataFrame],
        *,
        target_level: str | None = None,
    ) -> pl.LazyFrame | pl.DataFrame:
        """
        Reconstruct nested columns by progressively attaching child tables to
        their parents. The input should be a mapping produced by
        :meth:`normalize`.

        Args:
            tables: Mapping of level name to table.
            target_level: Optional target level (defaults to root).

        Returns:
            Denormalized frame with nested structures.

        Raises:
            ValueError: If tables is empty.
            KeyError: If required tables are missing.
        """
        if not tables:
            raise HierarchyValidationError(
                "Expected at least one table to denormalize.",
                details={"tables_provided": 0},
            )

        target_name = target_level or self._levels_meta[0].name
        target_idx = self.spec.index_of(target_name)

        root_name = self._levels_meta[0].name
        if root_name not in tables:
            raise HierarchyValidationError(
                f"Missing root level '{root_name}' in table mapping.",
                level=root_name,
                details={"provided_levels": list(tables.keys())},
            )

        prepared_tables: dict[str, pl.LazyFrame] = {}
        alias_map: dict[str, tuple[str, ...]] = {}

        for name, table in tables.items():
            lf = self._to_lazy(table)
            schema = lf.collect_schema()
            lf, added, schema = self._ensure_key_columns(lf, schema)
            if self.preserve_child_order:
                lf, schema = self._with_row_id(lf, schema)
            lf, schema = self._ensure_computed_fields(lf, schema)
            prepared_tables[name] = lf
            alias_map[name] = tuple(added)

        # Propagate child structures upward from deepest level.
        for level_idx in reversed(range(1, len(self._levels_meta))):
            level = self._levels_meta[level_idx]
            parent_meta = self._levels_meta[level_idx - 1]
            parent_name = parent_meta.name

            child_lf = prepared_tables.get(level.name)
            if child_lf is None:
                if level_idx <= target_idx:
                    raise HierarchyValidationError(
                        f"Missing table for level '{level.name}'.",
                        level=level.name,
                        details={"provided_levels": list(tables.keys())},
                    )
                continue

            parent_lf = prepared_tables.get(parent_name)
            if parent_lf is None:
                raise HierarchyValidationError(
                    f"Missing table for parent level '{parent_name}'.",
                    level=parent_name,
                    details={"provided_levels": list(tables.keys())},
                )

            child_schema = child_lf.collect_schema()
            child_packed, _ = self._pack_single_level(child_lf, level_idx, child_schema)
            child_struct = level.path
            join_keys = list(level.ancestor_keys)
            child_struct_frame = child_packed.select(
                [pl.col(key) for key in join_keys] + [pl.col(child_struct)]
            )
            child_added = alias_map.get(level.name, ())
            if child_added:
                child_packed = child_packed.drop(*child_added)
                child_struct_frame = child_struct_frame.drop(*child_added, strict=False)

            prepared_tables[level.name] = child_packed
            prepared_tables[parent_name] = parent_lf.join(
                child_struct_frame, on=join_keys, how="left"
            )

        target_name = self._levels_meta[target_idx].name
        result = prepared_tables.get(target_name)
        if result is None:
            raise HierarchyValidationError(
                f"Missing table for level '{target_name}'.",
                level=target_name,
            )

        added_aliases = alias_map.get(target_name, ())
        if added_aliases:
            result = result.drop(*added_aliases)

        result = self._drop_internal_columns(result)

        # Match output type to the target table's input type
        target_table = tables[target_name]
        if isinstance(target_table, pl.DataFrame):
            return result.collect()
        return result

    def build_from_tables(
        self,
        tables: Mapping[str, pl.LazyFrame | pl.DataFrame],
        *,
        target_level: str | None = None,
        join_type: Literal["left", "inner"] = "left",
    ) -> pl.LazyFrame | pl.DataFrame:
        """
        Build nested hierarchy from independent normalized tables.

        This method takes separate tables for each level (like database tables)
        where each table has its own column naming and joins them into a nested
        hierarchy structure.

        Args:
            tables: Mapping of level_name -> table. Each table should have:
                - Its own columns (no prefix required)
                - parent_keys columns for joining to parent level (if not root)
            target_level: Pack to this level (default: root level).
            join_type: How to join child tables to parents ("left" or "inner").

        Returns:
            Nested frame packed to the target level.

        Raises:
            HierarchyValidationError: If required tables or columns are missing.

        Example:
            >>> city_df = pl.DataFrame({"id": ["NYC"], "name": ["New York"]})
            >>> street_df = pl.DataFrame({
            ...     "id": ["st1"], "name": ["Broadway"], "city_id": ["NYC"]
            ... })
            >>> spec = HierarchySpec.from_levels(
            ...     LevelSpec(name="city", id_fields=["id"]),
            ...     LevelSpec(name="street", id_fields=["id"], parent_keys=["city_id"]),
            ... )
            >>> packer = HierarchicalPacker(spec)
            >>> result = packer.build_from_tables({"city": city_df, "street": street_df})
        """
        if not tables:
            raise HierarchyValidationError(
                "Expected at least one table to build from.",
                details={"tables_provided": 0},
            )

        target_name = target_level or self._levels_meta[0].name
        target_idx = self.spec.index_of(target_name)

        # Check that we have all required levels
        for i, meta in enumerate(self._levels_meta):
            if i > target_idx:
                break
            if meta.name not in tables:
                raise HierarchyValidationError(
                    f"Missing table for level '{meta.name}'.",
                    level=meta.name,
                    details={"provided_levels": list(tables.keys())},
                )

        # Determine output type based on first table
        first_table = next(iter(tables.values()))
        output_lazy = isinstance(first_table, pl.LazyFrame)

        # Prepare tables with proper prefixes
        prepared_tables: dict[str, pl.LazyFrame] = {}

        for level_idx, meta in enumerate(self._levels_meta):
            if meta.name not in tables:
                continue

            table = tables[meta.name]
            lf = self._to_lazy(table)

            # Rename columns with level prefix
            lf = self._prepare_level_table_internal(lf, meta.name, level_idx)
            prepared_tables[meta.name] = lf

        # Join tables from leaf to root
        # Start from deepest level and work up
        for level_idx in reversed(range(1, len(self._levels_meta))):
            level = self._levels_meta[level_idx]
            level_spec = self.spec.levels[level_idx]

            if level.name not in prepared_tables:
                continue

            parent_meta = self._levels_meta[level_idx - 1]
            parent_name = parent_meta.name

            if parent_name not in prepared_tables:
                continue

            child_lf = prepared_tables[level.name]
            parent_lf = prepared_tables[parent_name]

            # Get join keys from parent_keys
            parent_keys = level_spec.parent_keys
            if not parent_keys:
                raise HierarchyValidationError(
                    f"Level '{level.name}' must have parent_keys defined for build_from_tables.",
                    level=level.name,
                )

            # Map parent_keys to the qualified parent id columns
            parent_id_cols = list(parent_meta.id_columns)
            if len(parent_keys) != len(parent_id_cols):
                raise HierarchyValidationError(
                    f"Level '{level.name}' has {len(parent_keys)} parent_keys "
                    f"but parent '{parent_name}' has {len(parent_id_cols)} id_fields.",
                    level=level.name,
                    details={
                        "parent_keys": list(parent_keys),
                        "parent_id_columns": parent_id_cols,
                    },
                )

            # Create join: child's qualified parent_keys -> parent's id columns
            qualified_parent_keys = [f"{level.prefix}{pk}" for pk in parent_keys]

            # Join child to parent
            joined = parent_lf.join(
                child_lf,
                left_on=parent_id_cols,
                right_on=qualified_parent_keys,
                how=join_type,
            )

            # Drop the duplicate parent key columns from child
            joined = joined.drop(*qualified_parent_keys, strict=False)

            prepared_tables[parent_name] = joined

        # Get the result from root level and pack to target
        root_name = self._levels_meta[0].name
        result = prepared_tables[root_name]

        # Pack to target level
        result = self.pack(result, target_name)

        if output_lazy:
            return result
        return result.collect() if isinstance(result, pl.LazyFrame) else result

    def prepare_level_table(
        self,
        level_name: str,
        data: pl.DataFrame | pl.LazyFrame,
        column_mapping: dict[str, str] | None = None,
    ) -> pl.DataFrame | pl.LazyFrame:
        """
        Prepare a raw table for use in build_from_tables.

        Renames columns to match hierarchy naming convention.

        Args:
            level_name: Target level in hierarchy.
            data: Raw data table.
            column_mapping: Optional {raw_col: target_field} for non-obvious mappings.
                           If None, assumes column names match field names.

        Returns:
            Table with columns prefixed appropriately (e.g., "name" -> "city.street.name").
        """
        level_idx = self.spec.index_of(level_name)
        lf = self._to_lazy(data)

        if column_mapping:
            # Rename columns first according to mapping
            rename_exprs = [
                pl.col(raw_col).alias(target_col)
                for raw_col, target_col in column_mapping.items()
                if raw_col in lf.collect_schema()
            ]
            if rename_exprs:
                # Select all columns, applying renames
                all_cols = lf.collect_schema().keys()
                select_exprs: list[pl.Expr] = []
                for col in all_cols:
                    if col in column_mapping:
                        select_exprs.append(pl.col(col).alias(column_mapping[col]))
                    else:
                        select_exprs.append(pl.col(col))
                lf = lf.select(select_exprs)

        result = self._prepare_level_table_internal(lf, level_name, level_idx)

        if isinstance(data, pl.DataFrame):
            return result.collect()
        return result

    def _prepare_level_table_internal(
        self, lf: pl.LazyFrame, level_name: str, level_idx: int
    ) -> pl.LazyFrame:
        """
        Internal helper to add level prefixes to columns.

        Args:
            lf: The LazyFrame to process.
            level_name: The level name.
            level_idx: The level index.

        Returns:
            LazyFrame with prefixed columns.
        """
        meta = self._levels_meta[level_idx]
        schema = lf.collect_schema()

        # Add prefix to all columns except parent_keys (if at child level)
        level_spec = self.spec.levels[level_idx]
        parent_keys = set(level_spec.parent_keys or [])

        rename_exprs: list[pl.Expr] = []
        for col in schema.keys():
            if col in parent_keys:
                # Parent keys get the level prefix too
                rename_exprs.append(pl.col(col).alias(f"{meta.prefix}{col}"))
            else:
                # Regular columns get prefixed
                rename_exprs.append(pl.col(col).alias(f"{meta.prefix}{col}"))

        return lf.select(rename_exprs)

    def validate(
        self, frame: FrameT, *, level: str | None = None, raise_on_error: bool = True
    ) -> list[HierarchyValidationError]:
        """
        Validate hierarchy constraints on a frame.

        Checks:
        - Key columns are not null (unless entire entity is null)
        - Grouped values are identical for coarser-level attributes

        Args:
            frame: The DataFrame or LazyFrame to validate.
            level: Optional specific level to validate (validates all if None).
            raise_on_error: If True, raise on first error. If False, collect all errors.

        Returns:
            List of validation errors (empty if valid).

        Raises:
            HierarchyValidationError: If raise_on_error is True and validation fails.
        """
        errors: list[HierarchyValidationError] = []
        lf = self._to_lazy(frame)
        schema = lf.collect_schema()

        levels_to_check = self._levels_meta
        if level:
            level_idx = self.spec.index_of(level)
            levels_to_check = [self._levels_meta[level_idx]]

        for meta in levels_to_check:
            # Check key columns for nulls
            for key_col in meta.id_columns:
                if key_col not in schema:
                    continue

                # Count nulls in key column
                null_count = lf.select(pl.col(key_col).is_null().sum()).collect().item()

                if null_count > 0:
                    error = HierarchyValidationError(
                        f"Key column '{key_col}' contains {null_count} null values. "
                        "Key columns must not be null unless the entire entity is null.",
                        level=meta.name,
                        details={"column": key_col, "null_count": null_count},
                    )
                    if raise_on_error:
                        raise error
                    errors.append(error)

        return errors

    def get_level_columns(self, level: str) -> list[str]:
        """
        Return all columns belonging to a level.

        Args:
            level: The level name.

        Returns:
            List of column names for the level.
        """
        meta = self._levels_meta[self.spec.index_of(level)]
        # Return the prefix pattern that would match this level's columns
        return list(meta.id_columns) + list(meta.required_columns)

    def promote_attribute(
        self,
        frame: FrameT,
        attribute: str,
        *,
        from_level: str,
        to_level: str,
        agg: PromoteAggregation = "list",
        alias: str | None = None,
    ) -> FrameT:
        """
        Promote an attribute from a child level to its immediate parent level.

        Packs the frame so that ``from_level`` is a nested list-of-struct column,
        then uses ``list.eval`` / ``list.<agg>`` to extract and aggregate the
        attribute — no explode / group_by round-trips.

        Args:
            frame: The input frame (at any granularity).
            attribute: Unqualified field name at ``from_level`` (e.g. ``"population"``).
            from_level: The level where the attribute currently lives.  Must be an
                immediate child of ``to_level``.
            to_level: The coarser level to promote the attribute to.
            agg: Aggregation strategy:
                ``"list"`` | ``"set"`` | ``"sum"`` | ``"mean"`` | ``"min"`` |
                ``"max"`` | ``"first"`` | ``"last"`` | ``"count"`` | ``"single"``.
            alias: Optional output field name (unqualified).  Defaults to
                ``attribute``, qualified with the ``to_level`` prefix.

        Returns:
            Frame at ``to_level`` granularity with the promoted column added.
            Preserves input type (DataFrame / LazyFrame).

        Raises:
            KeyError: If either level is not found.
            ValueError: If ``from_level`` is not the immediate child of ``to_level``,
                or if the attribute does not exist at ``from_level``.
            HierarchyValidationError: If ``agg="single"`` and values are not
                uniform within groups.

        Examples:
            >>> result = packer.promote_attribute(
            ...     flat_df, "population",
            ...     from_level="city", to_level="country", agg="sum",
            ... )
        """
        from_idx = self.spec.index_of(from_level)
        to_idx = self.spec.index_of(to_level)

        if from_idx != to_idx + 1:
            raise ValueError(
                f"from_level '{from_level}' must be the immediate child of "
                f"to_level '{to_level}'. Got indices {from_idx} and {to_idx}."
            )

        from_meta = self._levels_meta[from_idx]
        to_meta = self._levels_meta[to_idx]

        # Pack so from_level becomes a list-of-struct column.
        packed = self.pack(frame, from_level)
        packed_lf = self._to_lazy(packed)
        packed_schema = packed_lf.collect_schema()

        # The packed column for from_level (e.g. "country.city")
        list_col = from_meta.path
        if list_col not in packed_schema:
            raise ValueError(
                f"Expected packed column '{list_col}' not found in schema. "
                f"Available columns: {list(packed_schema.keys())}"
            )

        # Validate the attribute exists inside the struct
        dtype = packed_schema[list_col]
        inner = dtype.inner if isinstance(dtype, pl.List) else dtype
        if not isinstance(inner, pl.Struct):
            raise ValueError(
                f"Expected struct inside list column '{list_col}', got {inner}."
            )
        field_names = [f.name for f in inner.fields]
        if attribute not in field_names:
            raise ValueError(
                f"Attribute '{attribute}' not found at level '{from_level}'. "
                f"Available fields: {field_names}"
            )

        # Build extraction expression:
        #   pl.col("<list_col>").list.eval(pl.element().struct.field("<attr>"))
        # This yields a List[T] column — one list per row.
        extract = pl.col(list_col).list.eval(
            pl.element().struct.field(attribute)
        )

        # Apply aggregation on the list
        out_field = alias or attribute
        out_col = f"{to_meta.prefix}{self._escape_field(out_field)}"
        agg_expr = self._apply_list_agg(extract, agg, from_level, packed_lf, out_col)

        result = packed_lf.with_columns(agg_expr.alias(out_col))
        return self._match_frame_type(result, frame)

    def _apply_list_agg(
        self,
        list_expr: pl.Expr,
        agg: PromoteAggregation,
        from_level: str,
        packed_lf: pl.LazyFrame,
        out_col: str,
    ) -> pl.Expr:
        """
        Apply an aggregation to a ``List[T]`` expression produced by ``list.eval``.

        Args:
            list_expr: Expression producing a ``List[T]`` column.
            agg: The aggregation name.
            from_level: Source level name (for error messages).
            packed_lf: The packed LazyFrame (used for eager validation in ``"single"``).
            out_col: Output column name (for error messages).

        Returns:
            Expression producing the aggregated result.
        """
        if agg == "list":
            return list_expr
        elif agg == "set":
            return list_expr.list.eval(pl.element().unique())
        elif agg == "sum":
            return list_expr.list.sum()
        elif agg == "mean":
            return list_expr.list.mean()
        elif agg == "min":
            return list_expr.list.min()
        elif agg == "max":
            return list_expr.list.max()
        elif agg == "first":
            return list_expr.list.first()
        elif agg == "last":
            return list_expr.list.last()
        elif agg == "count":
            return list_expr.list.len()
        elif agg == "single":
            # Validate that every list contains at most one unique non-null value.
            check = packed_lf.select(list_expr.alias("__check"))
            check_df = check.collect()
            for row_vals in check_df["__check"].to_list():
                if row_vals is not None:
                    unique_vals = set(v for v in row_vals if v is not None)
                    if len(unique_vals) > 1:
                        raise HierarchyValidationError(
                            f"Column '{out_col}' has non-uniform values within "
                            "groups. Values at coarser granularity should be "
                            "identical when using 'single' aggregation.",
                            level=from_level,
                            details={"column": out_col, "unique_values": unique_vals},
                        )
            return list_expr.list.first()
        else:
            raise ValueError(
                f"Unknown aggregation '{agg}'. Expected one of: "
                "list, set, sum, mean, min, max, first, last, count, single."
            )

    # ------------------------------------------------------------------
    # Separator Escaping
    # ------------------------------------------------------------------
    def _escape_field(self, name: str) -> str:
        """
        Escape separator characters in a field name.

        Args:
            name: The field name to escape.

        Returns:
            The escaped field name with separators escaped.
        """
        # First escape any existing escape characters, then escape separators
        escaped = name.replace(self.escape_char, self.escape_char + self.escape_char)
        return escaped.replace(self.separator, self.escape_char + self.separator)

    def _unescape_field(self, name: str) -> str:
        """
        Unescape separator characters in a field name.

        Args:
            name: The escaped field name.

        Returns:
            The original field name with escape sequences resolved.
        """
        # Unescape separators first, then unescape escape characters
        unescaped = name.replace(self.escape_char + self.separator, self.separator)
        return unescaped.replace(self.escape_char + self.escape_char, self.escape_char)

    def _split_path(self, path: str) -> list[str]:
        """
        Split a path by separator, respecting escaped separators.

        Args:
            path: The path to split.

        Returns:
            List of path components.
        """
        if not path:
            return []

        # Use a simple state machine to handle escapes
        components: list[str] = []
        current: list[str] = []
        i = 0
        while i < len(path):
            if path[i] == self.escape_char and i + 1 < len(path):
                # Escaped character - include the next character literally
                current.append(path[i + 1])
                i += 2
            elif path[i] == self.separator:
                # Unescaped separator - end current component
                components.append("".join(current))
                current = []
                i += 1
            else:
                current.append(path[i])
                i += 1

        # Add final component
        components.append("".join(current))
        return components

    def _join_path(self, components: Sequence[str]) -> str:
        """
        Join path components with separator, escaping as needed.

        Args:
            components: The path components to join.

        Returns:
            The joined path with escaped separators.
        """
        return self.separator.join(self._escape_field(c) for c in components)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_frame(
        self, frame: FrameT, schema: pl.Schema | None = None
    ) -> tuple[pl.LazyFrame, tuple[str, ...], pl.Schema]:
        """
        Prepare a frame for packing/unpacking.

        Args:
            frame: The frame to prepare.
            schema: Optional pre-collected schema to avoid re-collection.

        Returns:
            Tuple of (prepared LazyFrame, added column names, schema).
        """
        lf = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
        if schema is None:
            schema = lf.collect_schema()

        lf, added, schema = self._ensure_key_columns(lf, schema)

        if self.preserve_child_order:
            lf, schema = self._with_row_id(lf, schema)

        lf, schema = self._ensure_computed_fields(lf, schema)
        return lf, tuple(added), schema

    def _with_row_id(self, lf: pl.LazyFrame, schema: pl.Schema) -> tuple[pl.LazyFrame, pl.Schema]:
        """
        Add row ID column if needed for order preservation.

        Args:
            lf: The LazyFrame to modify.
            schema: Current schema.

        Returns:
            Tuple of (modified LazyFrame, updated schema).
        """
        if not self.preserve_child_order:
            return lf, schema
        if ROW_ID_COLUMN in schema:
            return lf, schema
        lf = lf.with_row_index(ROW_ID_COLUMN)
        # Update schema with the new column
        new_schema = lf.collect_schema()
        return lf, new_schema

    def _ensure_key_columns(
        self, lf: pl.LazyFrame, schema: pl.Schema
    ) -> tuple[pl.LazyFrame, list[str], pl.Schema]:
        """
        Ensure key alias columns exist.

        Args:
            lf: The LazyFrame to modify.
            schema: Current schema.

        Returns:
            Tuple of (modified LazyFrame, list of added columns, updated schema).
        """
        exprs: list[pl.Expr] = []
        added: list[str] = []

        for target, source in self.spec.key_aliases.items():
            if target in schema or source not in schema:
                continue
            exprs.append(pl.col(source).alias(target))
            added.append(target)

        if exprs:
            lf = lf.with_columns(*exprs)
            schema = lf.collect_schema()

        return lf, added, schema

    def _ensure_computed_fields(
        self, lf: pl.LazyFrame, schema: pl.Schema
    ) -> tuple[pl.LazyFrame, pl.Schema]:
        """
        Ensure computed field columns exist.

        Args:
            lf: The LazyFrame to modify.
            schema: Current schema.

        Returns:
            Tuple of (modified LazyFrame, updated schema).
        """
        if not self._computed_exprs:
            return lf, schema

        missing = [expr for alias, expr in self._computed_exprs.items() if alias not in schema]
        if missing:
            lf = lf.with_columns(*missing)
            schema = lf.collect_schema()

        return lf, schema

    def _to_lazy(self, frame: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
        """
        Convert frame to LazyFrame if needed.

        Args:
            frame: DataFrame or LazyFrame.

        Returns:
            LazyFrame.
        """
        return frame.lazy() if isinstance(frame, pl.DataFrame) else frame

    def _match_frame_type(self, result: pl.LazyFrame, original: FrameT) -> FrameT:
        """
        Match the result frame type to the original input type.

        Args:
            result: The LazyFrame result.
            original: The original input frame.

        Returns:
            Result as same type as original.
        """
        if isinstance(original, pl.DataFrame):
            return result.collect()  # type: ignore[return-value]
        return result  # type: ignore[return-value]

    def _drop_internal_columns(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Drop internal tracking columns.

        Args:
            lf: The LazyFrame to clean.

        Returns:
            LazyFrame with internal columns removed.
        """
        if self.preserve_child_order:
            lf = lf.drop(ROW_ID_COLUMN, strict=False)
        return lf

    def _identify_extra_columns(self, schema: pl.Schema) -> list[str]:
        """
        Identify columns that don't belong to any level in the hierarchy.

        A column belongs to the hierarchy if:
        - It starts with the root level name followed by the separator (e.g., "country.")
        - OR it's an internal column (like __hier_row_id)
        - OR it's a key alias column

        Args:
            schema: The current schema.

        Returns:
            List of column names that are not part of the hierarchy.
        """
        extra_cols: list[str] = []
        root_prefix = f"{self._levels_meta[0].name}{self.separator}"

        # Get all known hierarchy prefixes
        hierarchy_prefixes = [meta.prefix for meta in self._levels_meta if meta.prefix]

        # Also consider the root level path itself (for packed data)
        hierarchy_paths = {meta.path for meta in self._levels_meta}

        # Key alias targets are also valid
        key_alias_targets = set(self.spec.key_aliases.keys())

        for col in schema.keys():
            # Skip internal columns
            if col == ROW_ID_COLUMN:
                continue

            # Check if column is a known hierarchy path (for packed data)
            if col in hierarchy_paths:
                continue

            # Check if column is a key alias target
            if col in key_alias_targets:
                continue

            # Check if column starts with any hierarchy prefix
            is_hierarchy_col = any(col.startswith(prefix) for prefix in hierarchy_prefixes)
            if not is_hierarchy_col:
                # Also check if it's the root level itself (without children)
                if not col.startswith(root_prefix) and col != self._levels_meta[0].name:
                    extra_cols.append(col)

        return extra_cols

    def _qualify_field(self, level_idx: int, field: str) -> str:
        """
        Qualify a field name with the level path prefix.

        Args:
            level_idx: The level index.
            field: The field name.

        Returns:
            Fully qualified field name.
        """
        # Check if already contains an unescaped separator (already qualified)
        parts = self._split_path(field)
        if len(parts) > 1:
            return field

        level_names = [lvl.name for lvl in self.spec.levels[: level_idx + 1]]
        path = self._join_path(level_names)
        prefix = f"{path}{self.separator}" if path else ""
        escaped_field = self._escape_field(field)
        return f"{prefix}{escaped_field}" if prefix else escaped_field

    def _resolve_fields(
        self, level_idx: int, selectors: Sequence[ColumnSelector]
    ) -> tuple[list[str], list[pl.Expr]]:
        columns: list[str] = []
        exprs: list[pl.Expr] = []

        for selector in selectors:
            if isinstance(selector, pl.Expr):
                alias = selector.meta.output_name()
                if alias is None:
                    raise ValueError(
                        f"Expression provided for level '{self.spec.levels[level_idx].name}' "
                        "must have an alias via .alias(...)."
                    )
                columns.append(alias)
                exprs.append(selector)
            else:
                columns.append(self._qualify_field(level_idx, selector))

        return columns, exprs

    def _build_metadata(self) -> list[LevelMetadata]:
        metas: list[LevelMetadata] = []
        path_components: list[str] = []
        ancestor_keys: list[str] = []

        for index, level in enumerate(self.spec.levels):
            path_components.append(level.name)
            path = self.separator.join(path_components)
            prefix = f"{path}{self.separator}" if path else ""

            id_columns, id_exprs = self._resolve_fields(index, level.id_fields)
            required_columns, required_exprs = self._resolve_fields(
                index, level.required_fields or ()
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

    def _collect_computed_exprs(self) -> dict[str, pl.Expr]:
        exprs: dict[str, pl.Expr] = {}
        for meta in self._levels_meta:
            for expression in (*meta.id_exprs, *meta.required_exprs):
                alias = expression.meta.output_name()
                if alias:
                    exprs[alias] = expression
        return exprs

    def _pack_single_level(
        self, lf: pl.LazyFrame, level_idx: int, schema: pl.Schema
    ) -> tuple[pl.LazyFrame, pl.Schema]:
        """
        Pack a single level into a struct column.

        Args:
            lf: The LazyFrame to pack.
            level_idx: The level index to pack.
            schema: Current schema.

        Returns:
            Tuple of (packed LazyFrame, updated schema).
        """
        if self.preserve_child_order:
            lf, schema = self._with_row_id(lf, schema)

        meta = self._levels_meta[level_idx]
        level_cols = [
            name for name in schema.keys() if meta.prefix and name.startswith(meta.prefix)
        ]

        if not level_cols:
            return lf, schema

        group_keys = list(meta.ancestor_keys)

        sort_keys: list[pl.Expr | str] = []
        if meta.order_by:
            sort_keys.extend(meta.order_by)
        if self.preserve_child_order:
            sort_keys.append(pl.col(ROW_ID_COLUMN))
        if sort_keys:
            lf = lf.sort(sort_keys)

        struct_expr = pl.struct(
            [pl.col(col).alias(col[len(meta.prefix) :]) for col in level_cols]
        ).alias(meta.path)

        lf = lf.select(pl.all().exclude(level_cols), struct_expr)
        schema = lf.collect_schema()

        if not group_keys:
            return lf, schema

        excluded = set(group_keys) | {meta.path}
        if self.preserve_child_order:
            excluded.add(ROW_ID_COLUMN)
        remaining_cols = [col for col in schema.keys() if col not in excluded]

        # Validate that grouped values are identical if validation is enabled
        if self.validate_on_pack and remaining_cols:
            self._validate_aggregation_uniformity(lf, group_keys, remaining_cols, meta.name)

        agg_exprs = [pl.col(col).drop_nulls().first().alias(col) for col in remaining_cols]
        agg_exprs.append(pl.col(meta.path))

        lf = lf.group_by(group_keys, maintain_order=True).agg(agg_exprs)
        schema = lf.collect_schema()

        return lf, schema

    def _validate_aggregation_uniformity(
        self,
        lf: pl.LazyFrame,
        group_keys: list[str],
        value_cols: list[str],
        level_name: str,
    ) -> None:
        """
        Validate that values being aggregated are uniform within groups.

        Args:
            lf: The LazyFrame to validate.
            group_keys: Columns to group by.
            value_cols: Columns that will be aggregated with .first().
            level_name: The level name for error context.

        Raises:
            HierarchyValidationError: If values differ within a group.
        """
        # Check each value column for uniformity within groups
        # Use n_unique to detect non-uniform values
        for col in value_cols:
            check_expr = (
                lf.group_by(group_keys)
                .agg(pl.col(col).drop_nulls().n_unique().alias("n_unique"))
                .filter(pl.col("n_unique") > 1)
                .select(pl.len())
            )
            non_uniform_count = check_expr.collect().item()

            if non_uniform_count > 0:
                raise HierarchyValidationError(
                    f"Column '{col}' has non-uniform values within groups. "
                    f"Found {non_uniform_count} groups with differing values. "
                    "Values at coarser granularity should be identical within each group.",
                    level=level_name,
                    details={
                        "column": col,
                        "non_uniform_groups": non_uniform_count,
                        "group_keys": group_keys,
                    },
                )

    def _explode_and_unnest(
        self, lf: pl.LazyFrame, meta: LevelMetadata, schema: pl.Schema
    ) -> tuple[pl.LazyFrame, pl.Schema]:
        """
        Explode and unnest a level's nested column.

        Args:
            lf: The LazyFrame to process.
            meta: The level metadata.
            schema: Current schema.

        Returns:
            Tuple of (processed LazyFrame, updated schema).
        """
        dtype = schema[meta.path]
        if getattr(dtype, "base_type", lambda: None)() == pl.List:
            lf = lf.explode(meta.path)

        lf = lf.with_columns(
            pl.col(meta.path).name.prefix_fields(f"{meta.path}{self.separator}")
        ).unnest(meta.path)

        schema = lf.collect_schema()
        return lf, schema


if __name__ == "__main__":
    # ==========================================================================
    # Example Usage of HierarchicalPacker
    # ==========================================================================
    #
    # This module helps you work with hierarchical data in Polars, similar to
    # how pandas MultiIndex works but using nested struct/list columns.
    #
    # Run this file directly to see the examples:
    #   python -m nexpresso.hierarchical_packer
    # ==========================================================================

    print("=" * 80)
    print("HierarchicalPacker Examples")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # Example 1: Basic Pack/Unpack Operations
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Example 1: Basic Pack/Unpack Operations")
    print("=" * 80)

    # Define a simple hierarchy: Country -> City -> Street
    simple_spec = HierarchySpec(
        levels=[
            LevelSpec(name="country", id_fields=["code"]),
            LevelSpec(name="city", id_fields=["id"]),
            LevelSpec(name="street", id_fields=["name"]),
        ]
    )
    packer = HierarchicalPacker(simple_spec)

    # Create a flat DataFrame at the street level
    flat_df = pl.DataFrame(
        {
            "country.code": ["US", "US", "US", "CA", "CA"],
            "country.name": ["United States", "United States", "United States", "Canada", "Canada"],
            "country.city.id": ["NYC", "NYC", "LA", "TOR", "TOR"],
            "country.city.name": ["New York", "New York", "Los Angeles", "Toronto", "Toronto"],
            "country.city.population": [8_000_000, 8_000_000, 4_000_000, 3_000_000, 3_000_000],
            "country.city.street.name": [
                "Broadway",
                "5th Ave",
                "Sunset Blvd",
                "Queen St",
                "King St",
            ],
            "country.city.street.length_km": [21.0, 10.0, 35.0, 5.0, 3.0],
        }
    )

    print("\nOriginal flat DataFrame (5 rows at street level):")
    print(flat_df)

    # Pack to city level - streets become nested lists
    city_level = packer.pack(flat_df, "city")
    print("\nPacked to city level (3 rows, streets are nested):")
    print(city_level)

    # Pack further to country level
    country_level = packer.pack(flat_df, "country")
    print("\nPacked to country level (2 rows, cities and streets are nested):")
    print(country_level)

    # Unpack back to street level
    unpacked = packer.unpack(country_level, "street")
    print("\nUnpacked back to street level (5 rows):")
    print(unpacked)

    # --------------------------------------------------------------------------
    # Example 2: Normalize and Denormalize
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Example 2: Normalize and Denormalize")
    print("=" * 80)

    # Normalize splits the data into separate tables per level
    normalized = packer.normalize(flat_df)

    print("\nNormalized tables:")
    for level_name, table in normalized.items():
        print(f"\n{level_name.upper()} table:")
        print(table)

    # Denormalize reconstructs the nested structure
    denormalized = packer.denormalize(normalized)
    print("\nDenormalized (reconstructed nested structure):")
    print(denormalized)

    # --------------------------------------------------------------------------
    # Example 3: Building from Normalized Tables (Relational Data)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Example 3: Building from Normalized Tables (Like Database Tables)")
    print("=" * 80)

    # Define hierarchy with parent_keys for joining
    relational_spec = HierarchySpec.from_levels(
        LevelSpec(name="company", id_fields=["id"]),
        LevelSpec(name="department", id_fields=["id"], parent_keys=["company_id"]),
        LevelSpec(name="employee", id_fields=["id"], parent_keys=["dept_id"]),
    )
    relational_packer = HierarchicalPacker(relational_spec)

    # Create separate tables like you'd have in a database
    companies = pl.DataFrame(
        {
            "id": ["acme", "globex"],
            "name": ["Acme Corp", "Globex Inc"],
            "founded": [1990, 2005],
        }
    )

    departments = pl.DataFrame(
        {
            "id": ["eng", "sales", "hr"],
            "name": ["Engineering", "Sales", "Human Resources"],
            "company_id": ["acme", "acme", "globex"],
        }
    )

    employees = pl.DataFrame(
        {
            "id": ["e1", "e2", "e3", "e4"],
            "name": ["Alice", "Bob", "Charlie", "Diana"],
            "salary": [100000, 90000, 80000, 95000],
            "dept_id": ["eng", "eng", "sales", "hr"],
        }
    )

    print("\nInput tables (like database tables):")
    print("\nCOMPANIES:")
    print(companies)
    print("\nDEPARTMENTS:")
    print(departments)
    print("\nEMPLOYEES:")
    print(employees)

    # Build nested structure from these tables
    nested = relational_packer.build_from_tables(
        {
            "company": companies,
            "department": departments,
            "employee": employees,
        }
    )
    print("\nBuilt nested hierarchy:")
    print(nested)

    # Unpack to see the joined data
    all_employees = relational_packer.unpack(nested, "employee")
    print("\nUnpacked to employee level (all data joined):")
    print(all_employees)

    # --------------------------------------------------------------------------
    # Example 4: Validation
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Example 4: Validation")
    print("=" * 80)

    valid_spec = HierarchySpec(
        levels=[
            LevelSpec(name="parent", id_fields=["id"]),
            LevelSpec(name="child", id_fields=["id"]),
        ]
    )

    # Create data with validation issues
    data_with_nulls = pl.DataFrame(
        {
            "parent.id": ["p1", None, "p3"],  # Null in key column!
            "parent.child.id": ["c1", "c2", "c3"],
        }
    )

    validator = HierarchicalPacker(valid_spec)

    print("\nData with null key values:")
    print(data_with_nulls)

    # Validate without raising
    errors = validator.validate(data_with_nulls, raise_on_error=False)
    print(f"\nValidation errors found: {len(errors)}")
    for error in errors:
        print(f"  - {error}")

    # --------------------------------------------------------------------------
    # Example 5: Custom Separator
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Example 5: Custom Separator (using '/' instead of '.')")
    print("=" * 80)

    slash_spec = HierarchySpec(
        levels=[
            LevelSpec(name="folder", id_fields=["name"]),
            LevelSpec(name="file", id_fields=["name"]),
        ]
    )
    slash_packer = HierarchicalPacker(slash_spec, granularity_separator="/")

    files_df = pl.DataFrame(
        {
            "folder/name": ["docs", "docs", "images"],
            "folder/file/name": ["readme.txt", "notes.txt", "photo.jpg"],
            "folder/file/size_kb": [10, 25, 5000],
        }
    )

    print("\nFlat DataFrame with '/' separator:")
    print(files_df)

    packed_files = slash_packer.pack(files_df, "folder")
    print("\nPacked to folder level:")
    print(packed_files)

    # --------------------------------------------------------------------------
    # Example 6: Composable Level Definitions
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Example 6: Composable Level Definitions")
    print("=" * 80)

    # Define levels independently - they can be reused across hierarchies
    region_level = LevelSpec(name="region", id_fields=["code"])
    store_level = LevelSpec(name="store", id_fields=["id"], parent_keys=["region_code"])
    product_level = LevelSpec(name="product", id_fields=["sku"], parent_keys=["store_id"])

    # Compose into a hierarchy
    retail_spec = HierarchySpec.from_levels(
        region_level,
        store_level,
        product_level,
    )

    print("\nComposed hierarchy from independent level definitions:")
    for i, level in enumerate(retail_spec.levels):
        parent_info = f", parent_keys={list(level.parent_keys)}" if level.parent_keys else ""
        print(f"  {i}. {level.name} (id_fields={list(level.id_fields)}{parent_info})")

    # --------------------------------------------------------------------------
    # Example 7: Using prepare_level_table for Column Mapping
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Example 7: Preparing Tables with Column Mapping")
    print("=" * 80)

    # Raw data with different column names
    raw_products = pl.DataFrame(
        {
            "product_sku": ["SKU001", "SKU002"],
            "product_name": ["Widget", "Gadget"],
            "unit_price": [9.99, 19.99],
            "store_id": ["store1", "store1"],
        }
    )

    print("\nRaw product table (different column names):")
    print(raw_products)

    # Prepare with column mapping
    retail_packer = HierarchicalPacker(retail_spec)
    prepared = retail_packer.prepare_level_table(
        "product",
        raw_products,
        column_mapping={
            "product_sku": "sku",
            "product_name": "name",
            "unit_price": "price",
        },
    )

    print("\nPrepared table with hierarchy prefixes:")
    print(prepared)

    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)

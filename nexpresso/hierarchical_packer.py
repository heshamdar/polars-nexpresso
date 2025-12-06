"""
Generic packing/unpacking helpers for dot-notated Polars datasets.

Example
-------
>>> from dap_data_preprocessing.hierarchical_packer import (
...     HierarchicalPacker,
...     DEFAULT_HIERARCHY,
... )
>>> packer = HierarchicalPacker(DEFAULT_HIERARCHY)
>>> image_level = packer.pack(flat_df, "image")
>>> slice_level = packer.unpack(image_level, "slice")

Pipelines that previously relied on ``structuring_utils.pack_cols_to_level``
can switch to ``HierarchicalPacker(DEFAULT_HIERARCHY).pack(..., to_level)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TypeVar

import polars as pl
from polars.expr.expr import Expr

FrameT = TypeVar("FrameT", pl.LazyFrame, pl.DataFrame)

ColumnSelector = str | pl.Expr
ROW_ID_COLUMN = "__hier_row_id"

__all__ = [
    "LevelSpec",
    "HierarchySpec",
    "HierarchicalPacker",
]


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
    """

    name: str
    id_fields: Sequence[ColumnSelector] = ()
    required_fields: Sequence[ColumnSelector] | None = None
    order_by: Sequence[pl.Expr] | None = None


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
    """

    levels: Sequence[LevelSpec]
    key_aliases: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        level_names = [lvl.name for lvl in self.levels]
        if len(level_names) != len(set(level_names)):
            raise ValueError("Level names must be unique inside a HierarchySpec.")

    @property
    def levels_by_name(self) -> Mapping[str, LevelSpec]:
        return {level.name: level for level in self.levels}

    def index_of(self, level_name: str) -> int:
        for idx, level in enumerate(self.levels):
            if level.name == level_name:
                return idx
        raise KeyError(f"Level '{level_name}' not found in hierarchy.")

    def level(self, level_name: str) -> LevelSpec:
        return self.levels[self.index_of(level_name)]

    def next_level(self, level_name: str) -> LevelSpec | None:
        idx = self.index_of(level_name)
        if idx + 1 >= len(self.levels):
            return None
        return self.levels[idx + 1]


class HierarchicalPacker:
    """
    General-purpose helper for packing/unpacking nested hierarchies in Polars.

    The implementation assumes a dot-notation naming scheme and a strict tree
    (no cross-links). All behavior is driven by a ``HierarchySpec`` instance.
    """

    def __init__(
        self,
        spec: HierarchySpec,
        *,
        granularity_separator: str = ".",
        preserve_child_order: bool = True,
    ) -> None:
        self.spec: HierarchySpec = spec
        self.separator: str = granularity_separator
        self.preserve_child_order: bool = preserve_child_order
        self._levels_meta: list[LevelMetadata] = self._build_metadata()
        self._computed_exprs: dict[str, Expr] = self._collect_computed_exprs()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def pack(self, frame: FrameT, to_level: str) -> FrameT:
        """
        Pack flattened columns down to ``to_level`` so that rows represent the
        requested granularity.
        """

        lf, added_cols, was_lazy = self._prepare_frame(frame)

        target_idx = self.spec.index_of(to_level)
        for level_idx in reversed(range(target_idx, len(self._levels_meta))):
            lf = self._pack_single_level(lf, level_idx)

        if added_cols:
            lf = lf.drop(*added_cols)

        lf = self._drop_internal_columns(lf)
        return self._finalize(lf, was_lazy)

    def unpack(self, frame: FrameT, to_level: str) -> FrameT:
        """
        Unpack nested list-of-struct columns until ``to_level`` is reached,
        mirroring :func:`explode` + :func:`unnest` per level.
        """

        was_lazy = isinstance(frame, pl.LazyFrame)
        lf = frame.lazy()

        for level in self._levels_meta:
            schema = lf.collect_schema()
            if level.path not in schema:
                continue

            lf = self._explode_and_unnest(lf, level)
            if level.name == to_level:
                break

        lf = self._drop_internal_columns(lf)

        return self._finalize(lf, was_lazy)

    def split_levels(self, frame: FrameT) -> dict[str, FrameT]:
        """
        Split a packed frame into standalone tables—one per hierarchy level.
        """

        lf, added_cols, was_lazy = self._prepare_frame(frame)

        outputs: dict[str, pl.LazyFrame] = {}
        current = lf.lazy()

        for level in self._levels_meta:
            schema = current.collect_schema()
            if level.path not in schema:
                continue

            level_table = current
            level_table = self.unpack(level_table, level.name)
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
                drop_candidates = [col for col in added_cols if col in level_schema]
                if drop_candidates:
                    output_table = output_table.drop(*drop_candidates)

            outputs[level.name] = self._drop_internal_columns(output_table)
            current = level_table

        if added_cols:
            lf = lf.drop(*added_cols)

        if not was_lazy:
            return {name: tbl.lazy().collect() for name, tbl in outputs.items()}

        return outputs

    def normalize(self, frame: FrameT, *, root_level: str | None = None) -> dict[str, FrameT]:
        """
        Convenience wrapper that packs to the root level and splits into
        normalized per-level tables.
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
        """

        if not tables:
            raise ValueError("Expected at least one table to denormalize.")

        target_name = target_level or self._levels_meta[0].name
        target_idx = self.spec.index_of(target_name)

        if self._levels_meta[0].name not in tables:
            raise KeyError(f"Missing root level '{self._levels_meta[0].name}' in table mapping.")

        prepared_tables: dict[str, pl.LazyFrame] = {}
        alias_map: dict[str, tuple[str, ...]] = {}

        for name, table in tables.items():
            lf = table.lazy()
            lf, added = self._ensure_key_columns(lf)
            if self.preserve_child_order:
                lf = self._with_row_id(lf)
            lf = self._ensure_computed_fields(lf)
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
                    raise KeyError(f"Missing table for level '{level.name}'.")
                continue

            parent_lf = prepared_tables.get(parent_name)
            if parent_lf is None:
                raise KeyError(f"Missing table for parent level '{parent_name}'.")

            child_packed = self._pack_single_level(child_lf, level_idx)
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
            raise KeyError(f"Missing table for level '{target_name}'.")

        added_aliases = alias_map.get(target_name, ())
        if added_aliases:
            result = result.drop(*added_aliases)

        was_lazy = isinstance(tables[target_name], pl.LazyFrame)
        result = self._drop_internal_columns(result)
        return self._finalize(result, was_lazy)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_frame(self, frame: FrameT) -> tuple[pl.LazyFrame, tuple[str, ...], bool]:
        """
        Prepare a frame for packing/unpacking.
        """

        was_lazy = isinstance(frame, pl.LazyFrame)
        lf = frame.lazy()
        lf, added = self._ensure_key_columns(lf)

        if self.preserve_child_order:
            lf = self._with_row_id(lf)
        lf = self._ensure_computed_fields(lf)
        return lf, tuple(added), was_lazy

    def _with_row_id(self, lf: FrameT) -> FrameT:
        if not self.preserve_child_order:
            return lf
        schema = lf.collect_schema()
        if ROW_ID_COLUMN in schema:
            return lf
        return lf.with_row_index(ROW_ID_COLUMN)

    def _ensure_key_columns(self, lf: pl.LazyFrame) -> tuple[pl.LazyFrame, list[str]]:
        schema = lf.collect_schema()
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

        return lf, added

    def _ensure_computed_fields(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        if not self._computed_exprs:
            return lf

        schema = lf.collect_schema()
        missing = [expr for alias, expr in self._computed_exprs.items() if alias not in schema]
        if missing:
            lf = lf.with_columns(*missing)

        return lf

    def _finalize(self, lf: FrameT, was_lazy: bool) -> pl.LazyFrame | pl.DataFrame:
        if was_lazy:
            return lf.lazy()

        df = lf.lazy().collect()

        return df

    def _drop_internal_columns(self, lf: FrameT) -> FrameT:
        if self.preserve_child_order:
            lf = lf.drop(ROW_ID_COLUMN, strict=False)
        return lf

    def _qualify_field(self, level_idx: int, field: str) -> str:
        if self.separator in field:
            return field
        parts = [lvl.name for lvl in self.spec.levels[: level_idx + 1]]
        path = self.separator.join(parts)
        prefix = f"{path}{self.separator}" if path else ""
        return f"{prefix}{field}" if prefix else field

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

    def _pack_single_level(self, lf: FrameT, level_idx: int) -> FrameT:
        if self.preserve_child_order:
            lf = self._with_row_id(lf)
        meta = self._levels_meta[level_idx]
        schema = lf.collect_schema()
        level_cols = [
            name for name in schema.keys() if meta.prefix and name.startswith(meta.prefix)
        ]

        if not level_cols:
            return lf

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

        if not group_keys:
            return lf

        excluded = set(group_keys) | {meta.path}
        if self.preserve_child_order:
            excluded.add(ROW_ID_COLUMN)
        remaining_cols = [col for col in lf.collect_schema().keys() if col not in excluded]

        agg_exprs = [pl.col(col).drop_nulls().first().alias(col) for col in remaining_cols]
        agg_exprs.append(pl.col(meta.path))

        lf = lf.group_by(group_keys, maintain_order=True).agg(agg_exprs)

        return lf

    def _explode_and_unnest(self, lf: pl.LazyFrame, meta: LevelMetadata) -> pl.LazyFrame:
        dtype = lf.collect_schema()[meta.path]
        if getattr(dtype, "base_type", lambda: None)() == pl.List:
            lf = lf.explode(meta.path)
        return lf.with_columns(
            pl.col(meta.path).name.prefix_fields(f"{meta.path}{self.separator}")
        ).unnest(meta.path)

"""
Deferred nested *views* over normalized per-level tables.

Nesting is an excellent in-memory and interchange shape and a poor storage
shape. A ``List[Struct]`` column is shredded by Parquet into one leaf column
chunk per field, so the bytes stay columnar — but a Parquet row group holds N
*top-level* rows, and a list is never split across one. Packing 300k sale rows
into 20 region rows turns six skippable row groups into one, and no reader can
skip part of a list. Predicate pushdown and row-group pruning, the main reasons
Parquet is fast, are gone.

:class:`HierarchyView` resolves that tension. Data is stored **normalized** —
one flat table per level, as emitted by
:meth:`~nexpresso.HierarchicalPacker.normalize` — so every level is a real
top-level table with its own row groups, sort order, statistics and
partitioning. The view then presents that collection *as if* it were a single
nested frame: you address columns by their dotted hierarchy path, filter and
transform them, and the view routes each operation to the table that owns it,
joining across levels only when an operation genuinely spans them. Nothing
executes until you call a terminal method, and the nested shape is materialized
only if you actually ask for it.

    >>> view = HierarchyView.scan_parquet("warehouse/", packer)
    >>> view.filter(pl.col("region.store.sale.amount") > 990).collect("sale")

See ``docs/concepts/storage-layouts.md`` for the measurements behind this.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import polars as pl

from nexpresso.hierarchical_packer import (
    HierarchicalPacker,
    LevelMetadata,
    PromoteAggregation,
)

__all__ = ["HierarchyView", "EmptyParentMode"]

EmptyParentMode = Literal["prune", "keep"]
"""How to treat parents left with no surviving children.

``"prune"`` matches :meth:`~nexpresso.HierarchicalPacker.pack` semantics: a
parent whose children were all filtered away disappears. ``"keep"`` retains it
with an empty child list, which is cheaper (no semi-join cascade) and is what
you want when the parent row is meaningful on its own.
"""


class HierarchyView:
    """
    A lazy, nested-looking view over one flat table per hierarchy level.

    The view holds a ``LazyFrame`` per level plus the
    :class:`~nexpresso.HierarchicalPacker` that describes the hierarchy. Every
    operation returns a new view (the underlying frames are never mutated) and
    nothing executes until a terminal method is called.

    Columns are addressed by their full dotted path — ``"region.store.sale.amount"``
    — exactly as they appear in a flat/unpacked frame, regardless of which
    physical table they live in.

    Args:
        tables: Mapping of level name to that level's table, in the level-local
            shape produced by :meth:`~nexpresso.HierarchicalPacker.split_levels`
            (own columns plus ancestor key columns as foreign keys).
            ``DataFrame`` values are converted to ``LazyFrame``.
        packer: The packer describing the hierarchy.
        empty_parents: How to treat parents with no surviving children when
            materializing. Defaults to ``"prune"`` to match ``pack()``.

    Raises:
        ValueError: If ``tables`` is empty or names a level not in the hierarchy.

    Examples:
        >>> view = HierarchyView.from_tables(packer.normalize(df), packer)
        >>> hot = view.filter(pl.col("region.store.sale.amount") > 990)
        >>> hot.collect("sale")        # flat, joined to sale granularity
        >>> hot.collect_nested()       # the packed List[Struct] shape
    """

    def __init__(
        self,
        tables: Mapping[str, pl.LazyFrame | pl.DataFrame],
        packer: HierarchicalPacker,
        *,
        empty_parents: EmptyParentMode = "prune",
        _restricted: frozenset[str] = frozenset(),
    ) -> None:
        if empty_parents not in ("prune", "keep"):
            raise ValueError(
                f"Invalid empty_parents: {empty_parents!r}. Must be 'prune' or 'keep'."
            )
        if not tables:
            raise ValueError("tables must not be empty.")

        known = {lvl.name for lvl in packer.spec.levels}
        unknown = [name for name in tables if name not in known]
        if unknown:
            raise ValueError(
                f"Unknown level(s) in tables: {unknown}. " f"Hierarchy levels are: {sorted(known)}."
            )

        self._packer = packer
        self._empty_parents: EmptyParentMode = empty_parents
        # Levels whose row set may have shrunk. Only these need a consistency
        # cascade at materialization, so an unfiltered view costs no joins.
        self._restricted = frozenset(_restricted)
        # Keep hierarchy order (root -> leaf) regardless of mapping order.
        self._tables: dict[str, pl.LazyFrame] = {
            lvl.name: (
                tables[lvl.name].lazy()
                if isinstance(tables[lvl.name], pl.DataFrame)
                else tables[lvl.name]  # type: ignore[misc]
            )
            for lvl in packer.spec.levels
            if lvl.name in tables
        }
        self._meta: dict[str, LevelMetadata] = {
            meta.name: meta
            for meta in packer._levels_meta  # noqa: SLF001 - internal collaboration
            if meta.name in self._tables
        }

    # =========================================================================
    # Construction
    # =========================================================================

    @classmethod
    def from_tables(
        cls,
        tables: Mapping[str, pl.LazyFrame | pl.DataFrame],
        packer: HierarchicalPacker,
        *,
        empty_parents: EmptyParentMode = "prune",
    ) -> HierarchyView:
        """
        Build a view directly from normalized per-level tables.

        Args:
            tables: Level name -> level-local table, as produced by
                :meth:`~nexpresso.HierarchicalPacker.normalize`.
            packer: The packer describing the hierarchy.
            empty_parents: See :class:`HierarchyView`.

        Returns:
            A new view over ``tables``.
        """
        return cls(tables, packer, empty_parents=empty_parents)

    @classmethod
    def from_frame(
        cls,
        frame: pl.LazyFrame | pl.DataFrame,
        packer: HierarchicalPacker,
        *,
        root_level: str | None = None,
        empty_parents: EmptyParentMode = "prune",
    ) -> HierarchyView:
        """
        Normalize an existing flat or packed frame into a view.

        Convenience wrapper around
        :meth:`~nexpresso.HierarchicalPacker.normalize`. Useful for testing and
        for one-off conversions; for repeated queries, persist the normalized
        tables with :meth:`sink_parquet` and use :meth:`scan_parquet` instead.

        Args:
            frame: A flat or packed frame covering the hierarchy.
            packer: The packer describing the hierarchy.
            root_level: Optional root level to normalize to.
            empty_parents: See :class:`HierarchyView`.

        Returns:
            A new view over the normalized tables.
        """
        lazy = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
        return cls(
            packer.normalize(lazy, root_level=root_level), packer, empty_parents=empty_parents
        )

    @classmethod
    def scan_parquet(
        cls,
        source: str | Path,
        packer: HierarchicalPacker,
        *,
        pattern: str = "{level}",
        empty_parents: EmptyParentMode = "prune",
        **scan_kwargs: Any,
    ) -> HierarchyView:
        """
        Scan one Parquet dataset per level from a directory.

        Each level is scanned lazily and independently, so Polars applies
        projection and predicate pushdown — including row-group skipping — to
        each level's table on its own. That is the whole point of the layout:
        the child level is a first-class table, not a list buried inside a
        parent row.

        Args:
            source: Directory containing one Parquet file or directory per level.
            packer: The packer describing the hierarchy.
            pattern: Naming pattern for each level's dataset, with ``{level}``
                substituted. ``"{level}"`` matches both ``region.parquet`` and a
                ``region/`` directory of parts.
            empty_parents: See :class:`HierarchyView`.
            **scan_kwargs: Forwarded to :func:`polars.scan_parquet`.

        Returns:
            A new view over the scanned tables.

        Raises:
            FileNotFoundError: If no dataset is found for any level.
        """
        root = Path(source)
        tables: dict[str, pl.LazyFrame] = {}
        missing: list[str] = []
        for level in packer.spec.levels:
            stem = pattern.format(level=level.name)
            single, directory = root / f"{stem}.parquet", root / stem
            if single.exists():
                target: Path | str = single
            elif directory.is_dir():
                target = str(directory / "**/*.parquet")
            else:
                missing.append(level.name)
                continue
            tables[level.name] = pl.scan_parquet(target, **scan_kwargs)

        if not tables:
            raise FileNotFoundError(
                f"No per-level Parquet datasets found under {root} "
                f"using pattern {pattern!r}. Expected e.g. "
                f"{root / (pattern.format(level=packer.spec.levels[0].name) + '.parquet')}."
            )
        if missing:
            raise FileNotFoundError(
                f"No Parquet dataset found for level(s) {missing} under {root} "
                f"using pattern {pattern!r}."
            )
        return cls(tables, packer, empty_parents=empty_parents)

    # =========================================================================
    # Introspection
    # =========================================================================

    @property
    def levels(self) -> list[str]:
        """Level names present in this view, ordered root -> leaf."""
        return list(self._tables)

    @property
    def columns(self) -> list[str]:
        """Every column addressable through this view, as dotted paths."""
        seen: dict[str, None] = {}
        for lf in self._tables.values():
            for name in lf.collect_schema().names():
                seen.setdefault(name, None)
        return list(seen)

    @property
    def schema(self) -> pl.Schema:
        """
        The **nested** schema this view presents.

        Resolving it inspects query-plan metadata only and moves no data.
        """
        nested = self._packer.denormalize(self._resolved_tables())
        return nested.collect_schema()  # type: ignore[union-attr]

    def level_of(self, column: str) -> str:
        """
        The level that *owns* ``column``.

        Ownership follows the dotted path: ``"region.store.sale.amount"`` is
        owned by the level whose path is ``"region.store.sale"``. Ancestor key
        columns are replicated into descendant tables as foreign keys, but they
        are owned by the ancestor that declares them.

        Args:
            column: A dotted column path.

        Returns:
            The owning level's name.

        Raises:
            KeyError: If no level owns ``column``.
        """
        owner = self._owner_of(column)
        if owner is None:
            raise KeyError(
                f"Column {column!r} is not owned by any level in this view. "
                f"Known columns: {self.columns[:10]}..."
            )
        return owner

    def explain(self, level: str | None = None) -> str:
        """
        The query plan for materializing this view.

        Args:
            level: Plan for the flat join to this level's granularity. When
                ``None``, plans the nested reconstruction instead.

        Returns:
            The formatted query plan.
        """
        if level is None:
            nested = self._packer.denormalize(self._resolved_tables())
            return nested.explain()  # type: ignore[union-attr]
        return self.to_flat(level).explain()

    # =========================================================================
    # Internal resolution
    # =========================================================================

    def _owner_of(self, column: str) -> str | None:
        """Longest level path that prefixes ``column`` and owns it directly."""
        best: str | None = None
        best_len = -1
        for name, meta in self._meta.items():
            prefix = meta.path + self._packer.separator
            if column.startswith(prefix) and len(meta.path) > best_len:
                # Must be a direct field of this level, not of a descendant.
                remainder = column[len(prefix) :]
                if self._packer.separator not in remainder:
                    best, best_len = name, len(meta.path)
        return best

    def _key_columns(self, level: str) -> list[str]:
        """Ancestor foreign keys plus this level's own ids."""
        meta = self._meta[level]
        return [*meta.ancestor_keys, *meta.id_columns]

    def _columns_of(self, level: str) -> set[str]:
        return set(self._tables[level].collect_schema().names())

    def _carriers_of(self, columns: Iterable[str]) -> list[str]:
        """Levels whose table carries *every* one of ``columns``."""
        needed = set(columns)
        return [name for name in self._tables if needed <= self._columns_of(name)]

    def _rebuild(
        self,
        tables: Mapping[str, pl.LazyFrame],
        *,
        restricted: Iterable[str] = (),
    ) -> HierarchyView:
        return HierarchyView(
            tables,
            self._packer,
            empty_parents=self._empty_parents,
            _restricted=self._restricted | frozenset(restricted),
        )

    def _semi_join(
        self, target: pl.LazyFrame, source: pl.LazyFrame, keys: list[str]
    ) -> pl.LazyFrame:
        return target.join(source.select(keys).unique(), on=keys, how="semi")

    def _resolved_tables(self) -> dict[str, pl.LazyFrame]:
        """
        Apply deferred cross-level consistency before materializing.

        Filtering one level implies restrictions on the others, and the view
        owes callers a coherent hierarchy from any entry point — including
        :meth:`tables`, which performs no join of its own. Two cascades deliver
        that, and both run once here rather than after every operation:

        **Downward** (always): a child row whose parent was filtered away is no
        longer part of the hierarchy, so children are semi-joined to their
        surviving parents. Without this, filtering a parent *attribute* would
        leave orphaned rows visible in the child tables.

        **Upward** (``empty_parents="prune"``): a parent left with no surviving
        children disappears, matching :meth:`~nexpresso.HierarchicalPacker.pack`.

        Only levels recorded as restricted trigger a cascade, so an unfiltered
        view resolves to its scans untouched.
        """
        tables = dict(self._tables)
        if not self._restricted:
            return tables

        ordered = list(tables)

        # Downward: children follow their surviving parents.
        dirty = set(self._restricted)
        for parent, child in zip(ordered[:-1], ordered[1:]):
            if parent not in dirty:
                continue
            keys = [c for c in self._key_columns(parent) if c in self._columns_of(child)]
            if not keys:
                continue
            tables[child] = self._semi_join(tables[child], tables[parent], keys)
            dirty.add(child)

        if self._empty_parents == "keep":
            return tables

        # Upward: parents that lost every child disappear.
        for child, parent in zip(reversed(ordered[1:]), reversed(ordered[:-1])):
            if child not in dirty:
                continue
            keys = [c for c in self._key_columns(parent) if c in self._columns_of(child)]
            if not keys:
                continue
            tables[parent] = self._semi_join(tables[parent], tables[child], keys)
            dirty.add(parent)
        return tables

    def _evaluate_at(self, level: str, predicate: pl.Expr) -> pl.LazyFrame:
        """
        ``level``'s table filtered by ``predicate``, whatever levels it spans.

        Ancestor columns the predicate needs are joined in for the evaluation
        and dropped again, so the result keeps ``level``'s own schema.
        """
        roots = predicate.meta.root_names()
        present = self._columns_of(level)
        if not roots or set(roots) <= present:
            return self._tables[level].filter(predicate)
        widened, added = self._augmented(level, roots)
        return widened.filter(predicate).drop(added)

    def _augmented(self, level: str, columns: Iterable[str]) -> tuple[pl.LazyFrame, list[str]]:
        """
        ``level``'s table widened with ancestor-owned ``columns``.

        Ancestor keys are already present in every descendant table, so pulling
        an ancestor *attribute* down is a single join on those keys.

        Returns:
            The widened frame and the list of columns that were added (so the
            caller can drop them again).
        """
        lf = self._tables[level]
        present = self._columns_of(level)
        added: list[str] = []
        for name in columns:
            if name in present:
                continue
            owner = self._owner_of(name)
            if owner is None or owner not in self._tables:
                raise KeyError(
                    f"Column {name!r} is not available in this view "
                    f"(owning level missing or unknown)."
                )
            keys = [c for c in self._key_columns(owner) if c in present]
            if not keys:
                raise ValueError(
                    f"Cannot pull {name!r} from level {owner!r} into level {level!r}: "
                    f"no shared key columns. Expected {self._key_columns(owner)} "
                    f"to be present as foreign keys."
                )
            lf = lf.join(self._tables[owner].select([*keys, name]), on=keys, how="left")
            added.append(name)
        return lf, added

    # =========================================================================
    # Operations
    # =========================================================================

    def filter(self, *predicates: pl.Expr) -> HierarchyView:
        """
        Filter rows, routing each predicate to the level(s) that can evaluate it.

        A predicate over a single level's columns is applied to that level's
        table. Because :meth:`~nexpresso.HierarchicalPacker.normalize`
        replicates ancestor **keys** into descendant tables, a predicate on an
        ancestor key is applied to *every* table that carries it — sound
        transitive pushdown that lets the deepest scan skip row groups without
        any join. A predicate spanning several levels is evaluated at the
        deepest level involved, with the ancestor columns joined in and dropped
        again afterwards.

        Args:
            *predicates: Boolean expressions over dotted column paths.

        Returns:
            A new view with the predicates applied.

        Raises:
            KeyError: If a predicate references an unknown column.

        Examples:
            >>> view.filter(pl.col("region.store.sale.amount") > 990)
            >>> view.filter(pl.col("region.id") == 3)  # pushed to every level
        """
        tables = dict(self._tables)
        touched: set[str] = set()
        for predicate in predicates:
            roots = predicate.meta.root_names()
            if not roots:
                # Literal / no column reference: apply at the root level.
                first = next(iter(tables))
                tables[first] = tables[first].filter(predicate)
                touched.add(first)
                continue

            carriers = [name for name in self._carriers_of(roots) if name in tables]
            if carriers:
                for name in carriers:
                    tables[name] = tables[name].filter(predicate)
                touched.update(carriers)
                continue

            # Spans levels: evaluate at the deepest owner, joining the rest in.
            owners = [self._owner_of(c) for c in roots]
            unknown = [c for c, o in zip(roots, owners) if o is None]
            if unknown:
                raise KeyError(
                    f"Predicate references unknown column(s): {unknown}. "
                    f"Known columns: {self.columns[:10]}..."
                )
            ordered = list(self._tables)
            deepest = max((o for o in owners if o is not None), key=ordered.index)
            tables[deepest] = self._evaluate_at(deepest, predicate)
            touched.add(deepest)
        return self._rebuild(tables, restricted=touched)

    def with_columns(self, *exprs: pl.Expr) -> HierarchyView:
        """
        Add or replace columns, routed to the level that owns their inputs.

        The new column lands on the level of the deepest input it references,
        which is where it belongs in the hierarchy. Ancestor inputs are joined
        in for the computation and dropped again.

        Args:
            *exprs: Aliased expressions over dotted column paths.

        Returns:
            A new view with the columns added.

        Raises:
            KeyError: If an expression references an unknown column.
            ValueError: If an expression has no output name (missing ``.alias``).
        """
        tables = dict(self._tables)
        for expr in exprs:
            roots = expr.meta.root_names()
            try:
                output_name = expr.meta.output_name()
            except Exception as exc:  # pragma: no cover - polars raises variously
                raise ValueError(
                    f"Expression has no resolvable output name; add .alias(...): {expr}"
                ) from exc

            carriers = [name for name in self._carriers_of(roots) if name in tables]
            if carriers:
                ordered = list(self._tables)
                target = max(carriers, key=ordered.index) if roots else next(iter(tables))
                tables[target] = tables[target].with_columns(expr)
                continue

            owners = [self._owner_of(c) for c in roots]
            unknown = [c for c, o in zip(roots, owners) if o is None]
            if unknown:
                raise KeyError(
                    f"Expression {output_name!r} references unknown column(s): {unknown}."
                )
            ordered = list(self._tables)
            deepest = max((o for o in owners if o is not None), key=ordered.index)
            widened, added = self._augmented(deepest, roots)
            tables[deepest] = widened.with_columns(expr).drop(added)
        return self._rebuild(tables)

    def drop(self, *columns: str) -> HierarchyView:
        """
        Drop columns from whichever level carries them.

        Key columns are refused: they are the join structure of the view.

        Args:
            *columns: Dotted column paths to drop.

        Returns:
            A new view without those columns.

        Raises:
            ValueError: If a column is a key column of any level.
        """
        tables = dict(self._tables)
        for column in columns:
            for name in tables:
                if column in self._key_columns(name):
                    raise ValueError(
                        f"Cannot drop {column!r}: it is a key column of level {name!r} "
                        "and is required to relate the levels."
                    )
            for name in list(tables):
                if column in self._columns_of(name):
                    tables[name] = tables[name].drop(column)
        return self._rebuild(tables)

    def promote(
        self,
        attribute: str,
        *,
        from_level: str,
        to_level: str,
        agg: PromoteAggregation = "list",
        alias: str | None = None,
    ) -> HierarchyView:
        """
        Aggregate a child attribute up onto its parent level.

        The relational counterpart of
        :meth:`~nexpresso.HierarchicalPacker.promote_attribute`: a ``group_by``
        on the child table joined onto the parent table. Unlike the packed
        version it never builds an intermediate ``List[Struct]``, and unlike a
        flat ``group_by`` it never materializes the parent columns per child row.

        Args:
            attribute: Unqualified field name at ``from_level``.
            from_level: Level the attribute lives on. Must be the immediate
                child of ``to_level``.
            to_level: Level to promote onto.
            agg: Aggregation to apply. See
                :data:`~nexpresso.PromoteAggregation`.
            alias: Output field name (unqualified). Defaults to ``attribute``.

        Returns:
            A new view with the promoted column added to ``to_level``.

        Raises:
            KeyError: If either level is absent from the view.
            ValueError: If ``from_level`` is not the immediate child of
                ``to_level``, or the attribute is missing.
        """
        for name in (from_level, to_level):
            if name not in self._tables:
                raise KeyError(f"Level {name!r} is not present in this view: {self.levels}.")
        from_idx = self._packer.spec.index_of(from_level)
        to_idx = self._packer.spec.index_of(to_level)
        if from_idx != to_idx + 1:
            raise ValueError(
                f"from_level {from_level!r} must be the immediate child of "
                f"to_level {to_level!r}. Got indices {from_idx} and {to_idx}."
            )

        source = f"{self._meta[from_level].path}{self._packer.separator}{attribute}"
        if source not in self._columns_of(from_level):
            raise ValueError(
                f"Attribute {attribute!r} not found at level {from_level!r} "
                f"(looked for column {source!r})."
            )
        target = f"{self._meta[to_level].path}{self._packer.separator}{alias or attribute}"
        keys = [c for c in self._key_columns(to_level) if c in self._columns_of(from_level)]
        if not keys:
            raise ValueError(
                f"Cannot promote to {to_level!r}: level {from_level!r} carries none of "
                f"its key columns {self._key_columns(to_level)}."
            )

        rolled = (
            self._tables[from_level]
            .group_by(keys)
            .agg(_aggregate(pl.col(source), agg).alias(target))
        )
        tables = dict(self._tables)
        tables[to_level] = tables[to_level].join(rolled, on=keys, how="left")
        return self._rebuild(tables)

    def any_child_satisfies(
        self,
        predicate: pl.Expr,
        *,
        at_level: str,
        child_level: str,
    ) -> HierarchyView:
        """
        Keep only ``at_level`` rows having at least one descendant that matches.

        A semi-join, which is what this question is in relational form — no
        explode, no list construction, and the child scan still gets its own
        predicate pushdown.

        Args:
            predicate: Boolean expression evaluated at ``child_level``. It may
                reference ancestor columns too — those are joined in for the
                evaluation and dropped again.
            at_level: Level to filter.
            child_level: Descendant level the predicate applies to.

        Returns:
            A new view with ``at_level`` restricted.

        Raises:
            KeyError: If either level is absent from the view.
            ValueError: If ``child_level`` is not a descendant of ``at_level``.
        """
        for name in (at_level, child_level):
            if name not in self._tables:
                raise KeyError(f"Level {name!r} is not present in this view: {self.levels}.")
        if self._packer.spec.index_of(child_level) <= self._packer.spec.index_of(at_level):
            raise ValueError(
                f"child_level {child_level!r} must be finer than at_level {at_level!r}."
            )

        keys = [c for c in self._key_columns(at_level) if c in self._columns_of(child_level)]
        if not keys:
            raise ValueError(
                f"Level {child_level!r} carries none of {at_level!r}'s key columns "
                f"{self._key_columns(at_level)}."
            )
        matching = self._evaluate_at(child_level, predicate).select(keys).unique()
        tables = dict(self._tables)
        tables[at_level] = tables[at_level].join(matching, on=keys, how="semi")
        return self._rebuild(tables, restricted={at_level})

    # =========================================================================
    # Materialization
    # =========================================================================

    def tables(self) -> dict[str, pl.LazyFrame]:
        """
        The per-level plans, with cross-level consistency applied.

        This is the cheapest terminal: no join and no nesting. Most questions
        ("sum of sale amounts in region 3") are answered entirely from one
        level's table.

        Returns:
            Level name -> unexecuted ``LazyFrame``, ordered root -> leaf.
        """
        return self._resolved_tables()

    def to_flat(self, level: str | None = None) -> pl.LazyFrame:
        """
        Join the levels into a single flat frame at ``level`` granularity.

        This is the join the caller would otherwise have to write by hand. Each
        level is attached to its parent on the parent's key columns, which
        :meth:`~nexpresso.HierarchicalPacker.normalize` guarantees are present
        as foreign keys.

        Args:
            level: Target granularity. Defaults to the finest level in the view.

        Returns:
            An unexecuted ``LazyFrame`` at ``level`` granularity, with the same
            columns an unpacked frame would have.

        Raises:
            KeyError: If ``level`` is absent from the view.
        """
        ordered = list(self._tables)
        target = level if level is not None else ordered[-1]
        if target not in self._tables:
            raise KeyError(f"Level {target!r} is not present in this view: {ordered}.")

        tables = self._resolved_tables()
        lf = tables[ordered[0]]
        for name in ordered[1 : ordered.index(target) + 1]:
            parent = ordered[ordered.index(name) - 1]
            keys = [c for c in self._key_columns(parent) if c in self._columns_of(name)]
            lf = (
                lf.join(tables[name], on=keys, how="inner")
                if keys
                else lf.join(tables[name], how="cross")
            )
        return lf

    def collect(self, level: str | None = None) -> pl.DataFrame:
        """
        Execute and return a flat frame at ``level`` granularity.

        Args:
            level: Target granularity. Defaults to the finest level in the view.

        Returns:
            The materialized flat frame.
        """
        return self.to_flat(level).collect()

    def to_nested(self) -> pl.LazyFrame:
        """
        Reconstruct the packed ``List[Struct]`` shape, lazily.

        Returns:
            An unexecuted ``LazyFrame`` with the nested schema.
        """
        return self._packer.denormalize(self._resolved_tables())  # type: ignore[return-value]

    def collect_nested(self) -> pl.DataFrame:
        """
        Execute and return the packed ``List[Struct]`` frame.

        Only worth calling at the boundary where something actually consumes
        nesting; every query above is cheaper on the flat tables.

        Returns:
            The materialized nested frame.
        """
        return self.to_nested().collect()

    def sink_parquet(
        self,
        destination: str | Path,
        *,
        pattern: str = "{level}",
        **sink_kwargs: Any,
    ) -> None:
        """
        Write one Parquet file per level, in the layout :meth:`scan_parquet` reads.

        Prefers ``sink_parquet`` so the write is streamed rather than collected
        into memory first. Polars < 1.25 cannot sink Parquet from the standard
        engine, so those versions fall back to ``collect().write_parquet()`` —
        correct, but no longer memory-bounded.

        Args:
            destination: Directory to write into. Created if absent.
            pattern: Naming pattern with ``{level}`` substituted.
            **sink_kwargs: Forwarded to :meth:`polars.LazyFrame.sink_parquet`,
                or to :meth:`polars.DataFrame.write_parquet` on the fallback path.
        """
        root = Path(destination)
        root.mkdir(parents=True, exist_ok=True)
        for name, lf in self._resolved_tables().items():
            path = root / f"{pattern.format(level=name)}.parquet"
            try:
                lf.sink_parquet(path, **sink_kwargs)
            except pl.exceptions.InvalidOperationError:
                # Older Polars: no Parquet sink outside the streaming engine.
                lf.collect().write_parquet(path, **sink_kwargs)

    def __repr__(self) -> str:
        levels = ", ".join(f"{name}({len(self._columns_of(name))} cols)" for name in self._tables)
        return f"<HierarchyView empty_parents={self._empty_parents!r} levels=[{levels}]>"


def _aggregate(expr: pl.Expr, agg: PromoteAggregation) -> pl.Expr:
    """Apply a :data:`~nexpresso.PromoteAggregation` to a grouped column."""
    match agg:
        case "list":
            return expr
        case "set":
            return expr.unique()
        case "sum":
            return expr.sum()
        case "mean":
            return expr.mean()
        case "min":
            return expr.min()
        case "max":
            return expr.max()
        case "first":
            return expr.first()
        case "last":
            return expr.last()
        case "count":
            return expr.len()
        case "single":
            return expr.first()
        case _:
            raise ValueError(f"Unsupported aggregation: {agg!r}.")

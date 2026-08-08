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
        self._schema_cache: dict[str, set[str]] = {}
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
        """
        The level that owns ``column``, resolved by hierarchy path.

        Splitting is delegated to the packer so that escaped separators are
        honoured: a field literally named ``net.sales`` is stored as
        ``region.store.net\\.sales`` and must resolve to ``store``, not fail as
        though it named a deeper level.
        """
        parts = self._packer.split_path(column)
        if len(parts) < 2:
            return None
        # A column is a direct field of the level whose path is everything but
        # the last component.
        owner_path = self._packer.join_path(parts[:-1])
        for name, meta in self._meta.items():
            if meta.path == owner_path:
                return name
        return None

    def _qualified(self, level: str, field: str) -> str:
        """Full column path for an unqualified ``field`` at ``level``."""
        return self._packer.join_path([*self._packer.split_path(self._meta[level].path), field])

    def _key_columns(self, level: str) -> list[str]:
        """Ancestor foreign keys plus this level's own ids."""
        meta = self._meta[level]
        return [*meta.ancestor_keys, *meta.id_columns]

    def _columns_of(self, level: str, tables: Mapping[str, pl.LazyFrame] | None = None) -> set[str]:
        """
        Column names at ``level``, cached per view.

        ``self._tables`` never changes for the lifetime of a view, so resolving
        each level's schema once avoids re-planning on every routing decision.
        A ``tables`` override is used while an operation is still accumulating,
        where the working frames differ from the view's own.
        """
        if tables is not None:
            return set(tables[level].collect_schema().names())
        cached = self._schema_cache.get(level)
        if cached is None:
            cached = set(self._tables[level].collect_schema().names())
            self._schema_cache[level] = cached
        return cached

    def _carriers_of(
        self, columns: Iterable[str], tables: Mapping[str, pl.LazyFrame] | None = None
    ) -> list[str]:
        """Levels whose table carries *every* one of ``columns``."""
        needed = set(columns)
        source = tables if tables is not None else self._tables
        return [name for name in source if needed <= self._columns_of(name, tables)]

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

    def _augmented(
        self,
        tables: dict[str, pl.LazyFrame],
        level: str,
        columns: Iterable[str],
    ) -> tuple[pl.LazyFrame, list[str]]:
        """
        ``level``'s working frame widened with ancestor-owned ``columns``.

        Ancestor keys are already present in every descendant table, so pulling
        an ancestor *attribute* down is a single join on those keys. The join is
        a LEFT join on purpose: borrowing a column to evaluate an expression
        must never drop rows, even where referential integrity is broken.

        Reads from ``tables`` rather than ``self._tables`` so that earlier
        arguments in the same operation are not discarded.

        Args:
            tables: The working frames for this operation.
            level: Level to widen.
            columns: Columns the expression references.

        Returns:
            The widened frame and the columns that were added, for the caller to
            drop again.

        Raises:
            KeyError: If a column is not available in this view.
            ValueError: If the column is owned by a descendant rather than an
                ancestor, or no shared key columns exist.
        """
        lf = tables[level]
        present = self._columns_of(level, tables)
        order = list(tables)
        added: list[str] = []
        for name in columns:
            if name in present:
                continue
            owner = self._owner_of(name)
            if owner is None or owner not in tables:
                raise KeyError(
                    f"Unknown column {name!r}: not available in this view. "
                    f"Known columns: {sorted(self.columns)}."
                )
            # Borrowing only ever goes *up* the hierarchy. Pulling a descendant
            # column into an ancestor would fan the ancestor out to child
            # granularity — silently turning 4 region rows into 120.
            if order.index(owner) >= order.index(level):
                raise ValueError(
                    f"Cannot compute a {level!r}-level column from {owner!r}-level input "
                    f"{name!r}: {owner!r} is not an ancestor of {level!r}. "
                    "Use promote() to aggregate a child attribute upward."
                )
            keys = [c for c in self._key_columns(owner) if c in present]
            if not keys:
                raise ValueError(
                    f"Cannot pull {name!r} from level {owner!r} into level {level!r}: "
                    f"no shared key columns. Expected {self._key_columns(owner)} "
                    f"to be present as foreign keys."
                )
            lf = lf.join(tables[owner].select([*keys, name]), on=keys, how="left")
            added.append(name)
        return lf, added

    def _apply_at(
        self,
        tables: dict[str, pl.LazyFrame],
        level: str,
        expr: pl.Expr,
        method: str,
    ) -> None:
        """
        Run ``expr`` against ``level`` via ``method``, borrowing what it needs.

        Ancestor columns the expression references are joined in for the
        evaluation and dropped again, so the level keeps its own schema.
        """
        roots = expr.meta.root_names()
        if not roots or set(roots) <= self._columns_of(level, tables):
            tables[level] = getattr(tables[level], method)(expr)
            return
        widened, borrowed = self._augmented(tables, level, roots)
        tables[level] = getattr(widened, method)(expr).drop(borrowed)

    def _deepest_owner(self, tables: Mapping[str, pl.LazyFrame], roots: Iterable[str]) -> str:
        """The finest level owning any of ``roots``; that is where they can meet."""
        order = list(tables)
        owners = [(c, self._owner_of(c)) for c in roots]
        unknown = [c for c, owner in owners if owner is None or owner not in tables]
        if unknown:
            known = sorted(self.columns)
            hint = (
                " (present in the view but not resolvable from its path)"
                if any(c in known for c in unknown)
                else ""
            )
            raise KeyError(f"Unknown column(s): {unknown}{hint}. Known columns: {known}.")
        return max((owner for _, owner in owners if owner), key=order.index)

    # =========================================================================
    # Operations
    # =========================================================================

    def filter(self, *predicates: pl.Expr) -> HierarchyView:
        """
        Filter rows, routing each predicate to the level(s) that can evaluate it.

        A predicate over a single level's columns is applied to that level's
        table. Because :meth:`~nexpresso.HierarchicalPacker.normalize`
        replicates ancestor **keys** into descendant tables, a *row-wise*
        predicate on an ancestor key is applied to every table that carries it —
        sound transitive pushdown that lets the deepest scan skip row groups
        without any join. A predicate spanning several levels is evaluated at
        the deepest level involved, with the ancestor columns joined in and
        dropped again afterwards.

        Args:
            *predicates: Boolean expressions over dotted column paths.

        Returns:
            A new view with the predicates applied.

        Raises:
            KeyError: If a predicate references an unknown column.
            ValueError: If a predicate aggregates over a replicated ancestor
                key — see the note below.

        Note:
            Broadcasting is only valid for **row-wise** predicates. Each level
            holds an ancestor key at a different granularity, so an aggregate
            over one — ``count``, ``sum``, ``mean``, ``quantile``, a window —
            means something different per level, and intersecting those results
            is meaningless. Such a predicate is rejected rather than silently
            answered wrongly. Apply it to a single level's table via
            :meth:`tables` if that is what you want.

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

            carriers = self._carriers_of(roots, tables)
            if len(carriers) > 1 and not self._is_row_wise(
                predicate, roots, tables[carriers[0]].collect_schema()
            ):
                self._reject_aggregating_broadcast(predicate, roots, carriers)
            if carriers:
                for name in carriers:
                    self._apply_at(tables, name, predicate, "filter")
                touched.update(carriers)
                continue

            # Spans levels: evaluate at the deepest owner, joining the rest in.
            deepest = self._deepest_owner(tables, roots)
            self._apply_at(tables, deepest, predicate, "filter")
            touched.add(deepest)
        return self._rebuild(tables, restricted=touched)

    _PROBE_HEIGHT = 3

    def _is_row_wise(self, predicate: pl.Expr, roots: Iterable[str], schema: pl.Schema) -> bool:
        """
        Whether ``predicate`` yields one value per row rather than collapsing.

        Polars exposes no public "is this elementwise" API, so ask the
        expression: evaluate it against a small correctly-typed probe frame and
        see whether the output keeps the row count. A row-wise predicate maps
        N rows to N; any aggregate collapses to 1. Probing needs more than one
        row, since an aggregate over a single row also returns a single row.

        Detection failures are treated as row-wise — the guard exists to catch
        an obvious footgun, not to police every expression.
        """
        try:
            probe = pl.DataFrame(
                {c: pl.Series([None] * self._PROBE_HEIGHT, dtype=schema[c]) for c in roots}
            )
            return probe.lazy().select(predicate).collect().height == self._PROBE_HEIGHT
        except Exception:  # pragma: no cover - unusual expressions
            return True

    def _reject_aggregating_broadcast(
        self, predicate: pl.Expr, roots: Iterable[str], carriers: list[str]
    ) -> None:
        """Refuse to broadcast a predicate whose meaning depends on row multiplicity."""
        raise ValueError(
            f"Cannot broadcast an aggregating predicate over ancestor key(s) {sorted(roots)}: "
            f"they are replicated across levels {carriers} at different granularities, so the "
            "aggregate means something different on each and intersecting the results is "
            "meaningless. Apply it to one level's table via tables() instead."
        )

    def with_columns(self, *exprs: pl.Expr, **named_exprs: pl.Expr) -> HierarchyView:
        """
        Add or replace columns, routed by the **output** column's path.

        A derived column belongs where its name says it does:
        ``"region.store.sale.net"`` lands on the ``sale`` table regardless of
        which levels its inputs came from. Ancestor inputs are joined in for the
        computation and dropped again.

        Keyword form spells the destination explicitly and is usually clearer
        than an ``.alias()`` chain, since the path is the important part::

            view.with_columns(**{
                "region.store.sale.net": pl.col(amount) * (1 - pl.col(discount)),
            })

        Args:
            *exprs: Expressions whose output name is a full dotted column path
                (i.e. aliased).
            **named_exprs: Expressions keyed by their output path.

        Returns:
            A new view with the columns added.

        Raises:
            KeyError: If an expression references an unknown column, or its
                output name does not resolve to a level in this view.
            ValueError: If an expression has no output name, or computes an
                ancestor-level column from descendant-level input (use
                :meth:`promote` for that).
        """
        tables = dict(self._tables)
        for expr in (*exprs, *(e.alias(name) for name, e in named_exprs.items())):
            try:
                output_name = expr.meta.output_name()
            except Exception as exc:  # pragma: no cover - polars raises variously
                raise ValueError(
                    f"Expression has no resolvable output name; add .alias(...): {expr}"
                ) from exc

            target = self._owner_of(output_name)
            if target is None or target not in tables:
                raise KeyError(
                    f"Output column {output_name!r} does not name a level in this view. "
                    f"Use a full dotted path such as "
                    f"{self._qualified(list(tables)[-1], 'my_column')!r}."
                )
            self._apply_at(tables, target, expr, "with_columns")
        return self._rebuild(tables)

    def select(self, *columns: str) -> HierarchyView:
        """
        Keep only the named columns, plus the keys that relate the levels.

        Projection is what makes per-level Parquet scans cheap — each level's
        scan reads only the leaves you asked for — so this is the counterpart to
        :meth:`drop` and usually the better way to express intent.

        Key columns are always retained regardless of whether they are listed;
        without them the levels cannot be joined or nested. A level left with
        nothing but its keys is kept, so the hierarchy's shape is preserved.

        Args:
            *columns: Dotted column paths to keep.

        Returns:
            A new view restricted to those columns.

        Raises:
            ValueError: If a column is not present in any level.

        Examples:
            >>> view.select("region.name", "region.store.sale.amount")
        """
        wanted = set(columns)
        unknown = sorted(wanted - set(self.columns))
        if unknown:
            raise ValueError(
                f"Column(s) {unknown} not found in any level of this view. "
                f"Known columns: {sorted(self.columns)}."
            )

        tables = dict(self._tables)
        for name in tables:
            present = self._columns_of(name)
            keep = [
                column
                for column in self._tables[name].collect_schema().names()
                if column in wanted or column in self._key_columns(name)
            ]
            if set(keep) != present:
                tables[name] = tables[name].select(keep)
        return self._rebuild(tables)

    def drop(self, *columns: str, strict: bool = True) -> HierarchyView:
        """
        Drop columns from whichever level carries them.

        Key columns are refused: they are the join structure of the view.

        Args:
            *columns: Dotted column paths to drop.
            strict: Raise if a column is not present in any level. Dotted paths
                are long and hand-written, so a typo silently doing nothing is
                worse than an error; pass ``False`` for best-effort dropping.

        Returns:
            A new view without those columns.

        Raises:
            ValueError: If a column is a key column of any level, or is absent
                from every level and ``strict`` is True.
        """
        tables = dict(self._tables)
        for column in columns:
            for name in tables:
                if column in self._key_columns(name):
                    raise ValueError(
                        f"Cannot drop {column!r}: it is a key column of level {name!r} "
                        "and is required to relate the levels."
                    )
            holders = [name for name in tables if column in self._columns_of(name)]
            if not holders:
                if strict:
                    raise ValueError(
                        f"Column {column!r} not found in any level of this view. "
                        f"Known columns: {sorted(self.columns)}. "
                        "Pass strict=False to ignore missing columns."
                    )
                continue
            for name in holders:
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

        source = self._qualified(from_level, attribute)
        if source not in self._columns_of(from_level):
            raise ValueError(
                f"Attribute {attribute!r} not found at level {from_level!r} "
                f"(looked for column {source!r})."
            )
        target = self._qualified(to_level, alias or attribute)
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
        joined = tables[to_level].join(rolled, on=keys, how="left")
        if agg == "count":
            # A parent with no surviving children gets no group at all, so the
            # left join leaves null. list.len() on an empty list gives 0, and
            # promote_attribute agrees; match it.
            joined = joined.with_columns(pl.col(target).fill_null(0))
        tables[to_level] = joined
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
        tables = dict(self._tables)
        self._apply_at(tables, child_level, predicate, "filter")
        matching = tables[child_level].select(keys).unique()
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

        Uses ``sink_parquet``, so the write is streamed rather than collected
        into memory first.

        Args:
            destination: Directory to write into. Created if absent.
            pattern: Naming pattern with ``{level}`` substituted.
            **sink_kwargs: Forwarded to :meth:`polars.LazyFrame.sink_parquet`.
        """
        root = Path(destination)
        root.mkdir(parents=True, exist_ok=True)
        for name, lf in self._resolved_tables().items():
            lf.sink_parquet(root / f"{pattern.format(level=name)}.parquet", **sink_kwargs)

    def __repr__(self) -> str:
        levels = ", ".join(f"{name}({len(self._columns_of(name))} cols)" for name in self._tables)
        return f"<HierarchyView empty_parents={self._empty_parents!r} levels=[{levels}]>"


def _aggregate(expr: pl.Expr, agg: PromoteAggregation) -> pl.Expr:
    """
    Apply a :data:`~nexpresso.PromoteAggregation` to a grouped column.

    Delegates to :data:`HierarchicalPacker.GROUP_AGGREGATIONS` so that
    ``promote`` and ``promote_attribute`` cannot drift apart on null handling.
    """
    try:
        return HierarchicalPacker.GROUP_AGGREGATIONS[agg](expr)
    except KeyError:
        raise ValueError(
            f"Unsupported aggregation: {agg!r}. "
            f"Must be one of {sorted(HierarchicalPacker.GROUP_AGGREGATIONS)}."
        ) from None

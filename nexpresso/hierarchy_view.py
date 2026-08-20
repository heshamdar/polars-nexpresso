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
partitioning. The view's job is then to hand you the right frame for the
granularity you are working at:

    >>> view = HierarchyView.scan_parquet("warehouse/", packer)
    >>> view.level("sale").filter(pl.col("region.store.sale.amount") > 990)

:meth:`~HierarchyView.level` joins the root → ``sale`` axis and returns an
ordinary ``LazyFrame``, so everything after it is plain Polars. Joining the
whole axis is close to free when you do not use it: projection pushdown reduces
an unused ancestor level to its key columns, and a predicate on an ancestor
attribute runs inside that level's own scan, before the join.

Two things need the hierarchy itself and so stay on the view.
:meth:`~HierarchyView.nested` rebuilds the packed ``List[Struct]`` shape, and
:meth:`~HierarchyView.filter` restricts the hierarchy *as a whole* — dropping a
sale can orphan nothing, but dropping a region orphans its stores and sales, and
a region whose sales all vanish is itself gone. Nothing executes until you ask.

See ``docs/concepts/storage-layouts.md`` for the measurements behind this.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import polars as pl

from nexpresso.hierarchical_packer import (
    HierarchicalPacker,
    LevelMetadata,
    _reject_legacy_level_kwarg,
)

__all__ = ["HierarchyView", "EmptyParentMode", "PromoteMode"]

EmptyParentMode = Literal["prune", "keep"]
"""How to treat parents left with no surviving children.

``"prune"`` matches :meth:`~nexpresso.HierarchicalPacker.pack` semantics: a
parent whose children were all filtered away disappears. ``"keep"`` retains it
with an empty child list, which is cheaper (no semi-join cascade) and is what
you want when the parent row is meaningful on its own.

Where the hierarchy branches, ``"prune"`` applies per branch: a city left with
no streets is dropped even if its services survived, because packing along the
street axis has nothing to pack for it.
"""


PromoteMode = Literal["first", "list"]
"""How :meth:`HierarchyView.with_level` collapses a column named for an ancestor.

A transform runs at one level's granularity, so a column it names for an
*ancestor* has many values per ancestor row and something must reduce them.

``"first"`` takes one value per ancestor group and trusts you that they are all
the same — the mode for a window roll-up such as
``pl.col(amount).sum().over(region_keys)``, which is constant within the group by
construction. Nothing verifies that; a genuinely varying column silently keeps an
arbitrary value, so reach for ``"list"`` or an explicit aggregate when unsure.

``"list"`` gathers every value into a ``List`` column on the ancestor instead,
which is well defined whether or not they agree.
"""


class HierarchyView:
    """
    A lazy view over one flat table per hierarchy level.

    The view holds a ``LazyFrame`` per level plus the
    :class:`~nexpresso.HierarchicalPacker` that describes the hierarchy. It has
    a small surface on purpose: :meth:`level` gives you a frame at whatever
    granularity you name and Polars takes it from there, :meth:`nested` rebuilds
    the packed shape, :meth:`filter` restricts the hierarchy consistently, and
    :meth:`tables` hands back the per-level plans untouched. Every operation
    returns a new view — the underlying frames are never mutated — and nothing
    executes until you collect or sink.

    Columns are addressed by their full path from the root, joined by the
    packer's separator — ``"region.store.sale.amount"`` with the default ``"."``
    — exactly as they appear in a flat/unpacked frame, regardless of which
    physical table they live in. Build paths with
    :meth:`~nexpresso.HierarchicalPacker.join_path` rather than an f-string when
    the separator may not be the default.

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
        >>> view.level("sale")         # a LazyFrame at sale granularity
        >>> hot = view.filter(pl.col("region.store.sale.amount") > 990)
        >>> hot.nested().collect()     # the packed List[Struct] shape
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
        at_level: str | None = None,
        empty_parents: EmptyParentMode = "prune",
        **_legacy: Any,
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
            at_level: Optional level to pack to before splitting.
            empty_parents: See :class:`HierarchyView`.

        Returns:
            A new view over the normalized tables.

        Raises:
            TypeError: If the pre-0.8.0 ``root_level`` keyword is passed.
        """
        _reject_legacy_level_kwarg("from_frame", _legacy)

        lazy = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
        return cls(packer.normalize(lazy, at_level=at_level), packer, empty_parents=empty_parents)

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
        """Every column addressable through this view, as full separator-joined paths."""
        seen: dict[str, None] = {}
        for lf in self._tables.values():
            for name in lf.collect_schema().names():
                seen.setdefault(name, None)
        return list(seen)

    def key_columns(self, level: str) -> list[str]:
        """
        The columns that identify a ``level`` row: ancestor foreign keys then own ids.

        These are the columns :meth:`level` joins on, and the ones to group by
        when rolling a descendant up onto ``level``::

            view.level("sale").group_by(view.key_columns("region")).agg(
                pl.col("region.store.sale.amount").sum()
            )

        Args:
            level: A level present in this view.

        Returns:
            Ancestor key columns followed by this level's own id columns.
        """
        return self._key_columns(level)

    def level_of(self, column: str) -> str:
        """
        The level that *owns* ``column``.

        Ownership follows the column's path: ``"region.store.sale.amount"`` is
        owned by the level whose path is ``"region.store.sale"``. Ancestor key
        columns are replicated into descendant tables as foreign keys, but they
        are owned by the ancestor that declares them.

        Args:
            column: A full column path.

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

    def _parent_in_view(self, level: str) -> str | None:
        """
        The nearest ancestor of ``level`` present in this view.

        A view may hold a subset of the hierarchy's levels, so the structural
        parent is not always available; the nearest present ancestor is what the
        foreign keys still connect to.
        """
        for name in reversed(self._packer.spec.ancestors_of(level)):
            if name in self._tables:
                return name
        return None

    def _edges(self) -> list[tuple[str, str]]:
        """
        ``(parent, child)`` pairs over the levels present, parents first.

        In a chain this is each level paired with the one before it. Where the
        hierarchy branches, a parent appears once per branch — which is what
        makes the consistency cascades and the ancestor tests below fan out
        correctly instead of following declaration order.
        """
        edges: list[tuple[str, str]] = []
        for name in self._tables:
            parent = self._parent_in_view(name)
            if parent is not None:
                edges.append((parent, name))
        return edges

    def _ancestors_in_view(self, level: str) -> list[str]:
        """Ancestors of ``level`` present in this view, root first."""
        return [name for name in self._packer.spec.ancestors_of(level) if name in self._tables]

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

    def _ancestor_attributes(self, level: str) -> list[str]:
        """
        Ancestor-owned columns that ``level``'s own table does not already carry.

        ``normalize`` replicates ancestor *keys* into every descendant table, so
        those are already present; this is the rest — the attributes — which a
        cross-level expression has to borrow.
        """
        present = self._columns_of(level)
        return [
            name
            for ancestor in self._ancestors_in_view(level)
            for name in sorted(self._columns_of(ancestor))
            if name not in present and self._owner_of(name) == ancestor
        ]

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

        edges = self._edges()
        dirty = set(self._restricted)
        applied: set[tuple[str, str, str]] = set()

        def cascade(parent: str, child: str, direction: str) -> bool:
            """Semi-join one edge once; report whether it added a restriction."""
            if (parent, child, direction) in applied:
                return False
            keys = [c for c in self._key_columns(parent) if c in self._columns_of(child)]
            if not keys:
                return False
            applied.add((parent, child, direction))
            if direction == "down":
                tables[child] = self._semi_join(tables[child], tables[parent], keys)
                dirty.add(child)
            else:
                tables[parent] = self._semi_join(tables[parent], tables[child], keys)
                dirty.add(parent)
            return True

        # The two cascades feed each other once the hierarchy branches: filtering
        # ``service`` prunes cities upward, and those pruned cities must then
        # prune ``street`` downward — a branch the first downward pass never
        # touched. Alternate until nothing new is restricted. Each edge is
        # semi-joined at most once per direction, so this adds no repeated work;
        # an unbranched chain settles after the first round.
        changed = True
        while changed:
            changed = False
            # Downward: children follow their surviving parents. Parents-first
            # order carries a restriction the full depth of a branch in one pass.
            for parent, child in edges:
                if parent in dirty:
                    changed |= cascade(parent, child, "down")

            if self._empty_parents == "keep":
                continue

            # Upward: parents that lost every child disappear. Each branch prunes
            # independently, so a city with no surviving streets is dropped even
            # if its services survived — matching ``pack`` along the street axis,
            # where a childless parent has nothing to pack.
            for parent, child in reversed(edges):
                if child in dirty:
                    changed |= cascade(parent, child, "up")

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
        ancestors = set(self._packer.spec.ancestors_of(level))
        added: list[str] = []
        # Group by owner first: one join per ancestor level, not per column. A
        # borrowed column is projected away again afterwards, but Polars cannot
        # drop the join itself -- it has no way to know the ancestor's keys are
        # unique, so the join might change the row count.
        wanted: dict[str, tuple[list[str], list[str]]] = {}
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
            # granularity — silently turning 4 region rows into 120. A level on
            # a *sibling* branch is no better: it shares only an ancestor, so
            # joining it in would cross every street with every service.
            if owner not in ancestors:
                raise ValueError(
                    f"Cannot evaluate at level {level!r} using {owner!r}-level input "
                    f"{name!r}: {owner!r} is not an ancestor of {level!r}, so the two do "
                    "not line up row for row. Work on the frame level() returns for the "
                    "granularity you want."
                )
            keys = [c for c in self._key_columns(owner) if c in present]
            if not keys:
                raise ValueError(
                    f"Cannot pull {name!r} from level {owner!r} into level {level!r}: "
                    f"no shared key columns. Expected {self._key_columns(owner)} "
                    f"to be present as foreign keys."
                )
            wanted.setdefault(owner, ([], keys))[0].append(name)
            added.append(name)

        for owner, (names, join_keys) in wanted.items():
            lf = lf.join(tables[owner].select([*join_keys, *names]), on=join_keys, how="left")
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
        """
        The finest level owning any of ``roots``; that is where they can meet.

        Such a level exists only when every other owner is one of its ancestors.
        Owners on different branches share nothing below their common ancestor,
        so there is no granularity at which the columns line up row-for-row —
        joining them would pair every street with every service.
        """
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

        spec = self._packer.spec
        names = {owner for _, owner in owners if owner}
        for candidate in names:
            others = names - {candidate}
            if all(spec.is_ancestor_of(other, candidate) for other in others):
                return candidate

        branches = sorted(names)
        raise ValueError(
            f"Columns span levels {branches}, which lie on different branches of the "
            "hierarchy: no level is a descendant of all of them, so there is no "
            "granularity at which they align. Filter each branch separately, or roll one "
            "up onto the shared ancestor with a group_by on level()."
        )

    # =========================================================================
    # Operations
    # =========================================================================

    @staticmethod
    def _missing_column(exc: BaseException) -> str | None:
        """The column name a Polars "not found" error names, if it names one."""
        match = re.search(r'unable to find column "((?:[^"\\]|\\.)*)"', str(exc))
        return match.group(1) if match else None

    def _reject_unreachable_column(self, exc: BaseException, level: str) -> None:
        """
        Re-raise a missing-column failure as the reason that column is unreachable.

        A transform runs against ``level`` widened with every ancestor
        attribute, so a name that still does not resolve is either owned by a
        level that cannot be borrowed from -- a descendant or a sibling branch,
        which do not line up row for row -- or not in the view at all. Both
        deserve better than a raw error printing the query plan. Anything this
        cannot explain is left for the caller to re-raise unchanged.
        """
        name = self._missing_column(exc)
        if name is None:
            return

        owner = self._owner_of(name)
        # _owner_of resolves by path, so a name that merely *looks* like a
        # level's column ("region.nope") comes back owned. Check it exists.
        if owner is None or owner not in self._tables or name not in self._columns_of(owner):
            raise KeyError(
                f"Unknown column {name!r}: not available in this view. "
                f"Known columns: {sorted(self.columns)}."
            ) from exc
        if owner not in set(self._packer.spec.ancestors_of(level)):
            raise ValueError(
                f"Cannot evaluate at level {level!r} using {owner!r}-level input "
                f"{name!r}: {owner!r} is not an ancestor of {level!r}, so the two do "
                "not line up row for row. Work on the frame level() returns for the "
                "granularity you want."
            ) from exc

    def with_level(
        self,
        level: str,
        transform: Callable[[pl.LazyFrame], pl.LazyFrame],
        *,
        promote: PromoteMode | None = None,
    ) -> HierarchyView:
        """
        Replace one level's table, keeping a view.

        :meth:`level` gives you a frame and lets go; this keeps the hierarchy, so
        the result can still be filtered, nested or sunk. ``transform`` receives
        that level's ``LazyFrame`` and returns the new one:

            view.with_level("sale", lambda lf: lf.with_columns(
                (pl.col("region.store.sale.amount") * 2).alias("region.store.sale.dbl")
            ))

        Any **ancestor attribute** may be referenced, not just this level's own
        columns and the ancestor keys ``normalize`` replicates. They are joined
        in for the computation and dropped again, so the level keeps its own
        schema — a cross-level derivation needs no manual join:

            view.with_level("sale", lambda lf: lf.with_columns(
                (pl.col("region.store.sale.amount")
                 * (1 - pl.col("region.store.discount"))
                 * (1 + pl.col("region.tax_rate"))).alias("region.store.sale.net")
            ))

        To keep an ancestor value on this level rather than merely borrow it,
        alias it to a path this level owns; that copy is not one of the borrowed
        columns, so it survives.

        The widening costs one join per ancestor level, which Polars cannot
        optimize away — it has no way to know an ancestor's keys are unique — so
        it is only paid when the transform actually names an ancestor column.
        A transform confined to this level runs against its bare table.

        A column's **name decides where it lands**. Name one for an ancestor and,
        with ``promote`` set, that is the table it goes to — so a roll-up is an
        ordinary window expression rather than a separate operation:

            view.with_level("sale", lambda lf: lf.with_columns(
                pl.col(amount).sum().over("region.id").alias("region.revenue")
            ), promote="first")

        ``promote`` defaults to ``None``, which refuses such a column: a value
        arriving on a level you are not working at is worth being explicit
        about. See :data:`~nexpresso.PromoteMode` for what the modes mean.
        Nothing verifies that a ``"first"`` column is really constant within its
        group — that is the caller's bargain, and ``"list"`` is there for when it
        is not.

        Only *ancestors* can take a column this way. Their rows are coarser, so
        the values computed here reduce onto one; a descendant's are finer, and
        there is no answer to which of them each value belongs to. A sibling
        branch shares nothing below the common ancestor, so it is refused too.

        Doing this by hand through :meth:`tables` and
        :meth:`from_tables` works too, but silently drops ``empty_parents`` and
        skips the naming check below.

        Args:
            level: The level whose table to replace.
            transform: Called with the level's ``LazyFrame``, returns the new one.
            promote: How to reduce a column named for an ancestor onto that
                level — ``"first"`` or ``"list"``. ``None`` (the default)
                refuses such a column instead.

        Returns:
            A new view with that level rebuilt.

        Raises:
            KeyError: If ``level`` is absent from the view, or the transform
                references a column no level owns.
            ValueError: If the result drops a key column; carries a column that
                names no level at all — see the note; names one for another
                level without ``promote``; names one for a level that is not an
                ancestor, or that already has that column; if ``promote`` is not
                a recognised mode; or if the transform references a descendant
                or sibling-branch column.

        Note:
            Columns must be named with ``level``'s full path, because that
            is how :meth:`nested` knows where to put them. An unqualified name
            survives :meth:`level` and is silently **dropped** by :meth:`nested`,
            which is a quiet way to lose a column, so it is rejected here
            instead. Use :meth:`~nexpresso.HierarchicalPacker.join_path` or
            :meth:`~nexpresso.HierarchicalPacker.escape_field` when a field name
            itself contains the separator.
        """
        if level not in self._tables:
            raise KeyError(f"Level {level!r} is not present in this view: {self.levels}.")

        # Ancestor attributes are in scope, but joining them in costs a join per
        # ancestor that Polars cannot optimize away. So try the level's own
        # table first: a transform that never names an ancestor column pays
        # nothing, and one that does fails here on the missing name and is
        # retried against the widened frame.
        borrowed: list[str] = []
        try:
            result = transform(self._tables[level])
            produced = result.collect_schema().names()
        except Exception:
            working = dict(self._tables)
            widened, borrowed = self._augmented(working, level, self._ancestor_attributes(level))
            try:
                result = transform(widened)
                produced = result.collect_schema().names()
            except Exception as exc:
                # Every ancestor attribute was in scope and the transform still
                # could not resolve a name. Say which one and why, rather than
                # letting a raw Polars error print the whole plan.
                self._reject_unreachable_column(exc, level)
                raise

        # Ancestor attributes were lent for the computation, not adopted: drop
        # whatever survived so the level keeps its own schema. A transform that
        # wants to keep one aliases it to a path this level owns, and that copy
        # is not in ``borrowed`` so it stays.
        leftover = [name for name in borrowed if name in produced]
        if leftover:
            result = result.drop(leftover)
            produced = [name for name in produced if name not in leftover]

        keys = self.key_columns(level)
        missing = [key for key in keys if key not in produced]
        if missing:
            raise ValueError(
                f"Transform of level {level!r} dropped key column(s) {missing}, which "
                "relate it to the rest of the hierarchy. Keep "
                f"{keys} on the frame."
            )

        allowed = set(keys)
        foreign = [
            name for name in produced if name not in allowed and self._owner_of(name) != level
        ]

        tables = dict(self._tables)
        if foreign:
            promoted = self._promote_foreign(tables, level, result, foreign, promote)
            result = result.drop(promoted)
            produced = [name for name in produced if name not in promoted]

        tables[level] = result
        return self._rebuild(tables)

    def _promote_foreign(
        self,
        tables: dict[str, pl.LazyFrame],
        level: str,
        result: pl.LazyFrame,
        foreign: list[str],
        promote: PromoteMode | None,
    ) -> list[str]:
        """
        Move ``foreign`` columns onto the ancestor levels their paths name.

        ``result`` is at ``level`` granularity, so a column named for an ancestor
        has many values per ancestor row. ``promote`` says how to reduce them and,
        by being ``None`` by default, that reduction never happens unless it was
        asked for -- a column landing on another level is a big enough thing to
        be explicit about.

        Mutates ``tables`` and returns the names to drop from ``level``.

        Raises:
            ValueError: If ``promote`` is ``None``, if a name belongs to no level
                or to one that is not an ancestor, or if ``promote`` is not a
                recognised mode.
        """
        if promote is not None and promote not in ("first", "list"):
            raise ValueError(f"Invalid promote: {promote!r}. Must be 'first', 'list' or None.")

        unowned = [name for name in foreign if self._owner_of(name) not in self._tables]
        if unowned:
            example = self._qualified(level, "my_column")
            raise ValueError(
                f"Transform of level {level!r} produced column(s) {unowned} that name no "
                f"level in this view. nested() places columns by their path, so an "
                f"unqualified name would be dropped on the way out. Name them for a "
                f"level, e.g. {example!r}."
            )

        ancestors = set(self._packer.spec.ancestors_of(level))
        by_owner: dict[str, list[str]] = {}
        for name in foreign:
            owner = self._owner_of(name)
            assert owner is not None  # unowned names were rejected above
            if owner not in ancestors:
                relation = (
                    "a descendant of"
                    if self._packer.spec.is_ancestor_of(level, owner)
                    else "on a different branch from"
                )
                raise ValueError(
                    f"Transform of level {level!r} produced {name!r}, which belongs to "
                    f"level {owner!r} — {relation} {level!r}. A column can only be "
                    f"promoted to an *ancestor*, whose rows are coarser, so the values "
                    f"computed here reduce onto one. {owner!r} rows are not, so there is "
                    f"no answer to which of them each value belongs to. Compute it in "
                    f"with_level({owner!r}, ...) instead."
                )
            by_owner.setdefault(owner, []).append(name)

        if promote is None:
            raise ValueError(
                f"Transform of level {level!r} produced column(s) {sorted(foreign)} named "
                f"for another level. Pass promote='first' to reduce each to one value per "
                f"owning row -- correct when the column is constant within the group, as a "
                f"window aggregate is -- or promote='list' to gather the values into a "
                "List column. Nothing checks uniformity; 'first' takes you at your word."
            )

        promoted: list[str] = []
        for owner, names in by_owner.items():
            owner_keys = self.key_columns(owner)
            existing = [name for name in names if name in self._columns_of(owner)]
            if existing:
                raise ValueError(
                    f"Transform of level {level!r} produced column(s) {existing}, which "
                    f"level {owner!r} already has. Promoting would collide with the stored "
                    f"column, and a borrowed column is lent, not adopted -- to change "
                    f"{owner!r}'s own data, call with_level({owner!r}, ...)."
                )
            rolled = result.group_by(owner_keys).agg(
                [
                    (pl.col(name).first() if promote == "first" else pl.col(name)).alias(name)
                    for name in names
                ]
            )
            tables[owner] = tables[owner].join(rolled, on=owner_keys, how="left")
            promoted.extend(names)
        return promoted

    def filter(self, *predicates: pl.Expr) -> HierarchyView:
        """
        Filter rows, routing each predicate to the level(s) that can evaluate it.

        This is the one operation that cannot be done on the frame
        :meth:`level` returns, because restricting a normalized hierarchy
        implies restrictions on the *other* levels: a child whose parent was
        filtered away is orphaned, and under ``empty_parents="prune"`` a parent
        left with no children disappears. Those cascades run once, at
        materialization, so ``filter`` composes without paying a join per call.

        A predicate over a single level's columns is applied to that level's
        table. Because :meth:`~nexpresso.HierarchicalPacker.normalize`
        replicates ancestor **keys** into descendant tables, a *row-wise*
        predicate on an ancestor key is applied to every table that carries it —
        sound transitive pushdown that lets the deepest scan skip row groups
        without any join. A predicate spanning several levels is evaluated at
        the deepest level involved, with the ancestor columns joined in and
        dropped again afterwards.

        A predicate that reads the whole column rather than each row on its own
        is evaluated at the level that *owns* its columns, so
        ``pl.col("region.id").count()`` is the number of regions. The broadcast
        above is a pushdown shortcut and is skipped for such a predicate, since
        each level holds a replicated key at a different granularity. That
        covers window-shaped predicates too — ``col > col.mean()`` keeps the row
        count but still depends on every other row, so broadcasting it would
        recompute the mean once per level.

        Note:
            An aggregate carries no implicit ``over``: ``pl.col(AMOUNT).sum() >
            100`` is one scalar over the entire level, so every row survives or
            none does. For a per-parent question, roll up with
            ``level(child).group_by(key_columns(parent)).agg(...)`` and
            semi-join the result back.

        Args:
            *predicates: Boolean expressions over full column paths.

        Returns:
            A new view with the predicates applied.

        Raises:
            KeyError: If a predicate references an unknown column.
            ValueError: If a predicate spans two branches of the hierarchy,
                which share no granularity.

        Examples:
            >>> view.filter(pl.col("region.store.sale.amount") > 990)
            >>> view.filter(pl.col("region.id") == 3)  # pushed to every level
            >>> view.filter(pl.col("region.id").count() > 10)  # at the region level
            >>> view.filter(pl.col("region.id") > pl.col("region.id").mean())  # ditto
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
            # Broadcasting to every carrier is a pushdown shortcut, sound only
            # for a row-wise predicate: each table holds a replicated ancestor
            # key at a different granularity, so an aggregate over one means
            # something different on each. Where that cannot be shown, drop the
            # shortcut and evaluate at the owning level, where the predicate has
            # exactly one meaning; the downward cascade restricts the rest.
            if len(carriers) > 1 and not self._is_row_wise(
                predicate, roots, tables[carriers[0]].collect_schema()
            ):
                carriers = []
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
        Whether ``predicate`` evaluates each row independently of the others.

        Polars exposes no public "is this elementwise" API, so ask the
        expression: evaluate it against a small correctly-typed probe frame and
        compare that against evaluating it one row at a time. An elementwise
        predicate cannot tell the difference; anything that reads the rest of
        the column gives itself away.

        Preserving the row count is *not* enough on its own. It rules out a
        plain aggregate, which collapses N rows to 1, but a window-shaped
        predicate such as ``col > col.mean()`` or ``col.rank() <= 2`` maps N
        rows to N while still depending on the whole column. Broadcasting one of
        those to every carrier would recompute the aggregate at each level's own
        granularity — over regions on the region table, over sales on the sale
        table — and quietly return a different row set than the level that owns
        the columns would.

        The probe therefore needs distinct, non-null values: an all-null column
        makes ``col > col.mean()`` return nulls either way, and a constant one
        makes it agree by accident.

        This gates an optimization only. Correctness comes from routing the
        predicate to the level that owns its columns and cascading from there,
        so any doubt is answered ``False``: the broadcast is skipped and the
        predicate takes the slower but always-sound path.
        """
        try:
            probe = pl.DataFrame({c: self._probe_column(schema[c]) for c in roots})
            # A column the cast collapsed to a single value cannot tell the two
            # evaluations apart, so it cannot clear the predicate either.
            if any(probe[c].n_unique() < 2 for c in probe.columns):
                return False
            whole = probe.lazy().select(predicate).collect()
            if whole.height != self._PROBE_HEIGHT:
                return False
            per_row = pl.concat(
                [probe[i : i + 1].lazy().select(predicate).collect() for i in range(probe.height)]
            )
            return whole.equals(per_row)
        except Exception:  # pragma: no cover - unusual expressions
            return False

    def _probe_column(self, dtype: pl.DataType) -> pl.Series:
        """Distinct, non-null probe values of ``dtype``, for :meth:`_is_row_wise`."""
        counter = pl.int_range(self._PROBE_HEIGHT, eager=True)
        try:
            return counter.cast(dtype, strict=True)
        except Exception:
            # A few dtypes take no integer directly — Categorical among them —
            # but accept the same values spelled as strings.
            return counter.cast(pl.String).cast(dtype, strict=True)

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

    def level(self, at_level: str | None = None) -> pl.LazyFrame:
        """
        The frame for working at ``at_level`` granularity: flat, one row per entity.

        This is the view's main entry point. It hands back an ordinary
        ``LazyFrame``, so everything downstream is plain Polars — ``select``,
        ``with_columns``, ``group_by``, ``sort``, joins — with no ambiguity about
        which granularity an expression means, because a frame has exactly one.

        The whole root → ``at_level`` axis is joined, so every ancestor
        attribute is in scope. That costs close to nothing when you do not use
        them: projection pushdown reduces an unused ancestor level to its key
        columns, and a predicate on an ancestor attribute is evaluated inside
        that level's own scan, before the join.

        Only the levels on ``at_level``'s **axis** are joined. Sibling branches
        are left out rather than crossed in: a flat frame has one granularity,
        and joining two branches would pair every street with every service.
        Call ``level`` once per branch and combine the results yourself.

        Args:
            at_level: Target granularity. Defaults to the finest level in the
                view when the hierarchy has only one leaf.

        Returns:
            An unexecuted ``LazyFrame`` at ``at_level`` granularity, with the
            same columns an unpacked frame would have.

        Raises:
            KeyError: If ``at_level`` is absent from the view.
            ValueError: If ``at_level`` is omitted and the view has several leaf
                levels, where there is no single finest granularity; or if a
                level shares no key columns with its parent.

        Examples:
            >>> view.level("sale").filter(pl.col("region.store.sale.amount") > 990)
            >>> view.level("sale").group_by("region.id").agg(pl.col(AMOUNT).sum())
        """
        ordered = list(self._tables)
        target = at_level if at_level is not None else self._default_target(ordered)
        if target not in self._tables:
            raise KeyError(f"Level {target!r} is not present in this view: {ordered}.")

        tables = self._resolved_tables()
        axis = [*self._ancestors_in_view(target), target]
        lf = tables[axis[0]]
        for parent, name in zip(axis[:-1], axis[1:]):
            keys = [c for c in self.key_columns(parent) if c in self._columns_of(name)]
            if not keys:
                # Crossing instead would silently multiply the rows of a frame
                # that is supposed to have one row per entity.
                raise ValueError(
                    f"Cannot flatten to {target!r}: level {name!r} carries none of "
                    f"{parent!r}'s key columns {self.key_columns(parent)}, so the two "
                    "cannot be related."
                )
            lf = lf.join(tables[name], on=keys, how="inner")
        return lf

    def _default_target(self, ordered: list[str]) -> str:
        """
        The level :meth:`level` flattens to when the caller names none.

        Well defined only for a single-branch view: with several leaves the
        "finest level" is a different granularity per branch.
        """
        leaves = [name for name in ordered if not self._children_in_view(name)]
        if len(leaves) > 1:
            raise ValueError(
                f"This view has {len(leaves)} leaf levels {leaves}, one per branch, so "
                "there is no single finest granularity. Name the level to flatten to."
            )
        return leaves[0] if leaves else ordered[-1]

    def _children_in_view(self, level: str) -> list[str]:
        """Levels in this view whose nearest present ancestor is ``level``."""
        return [name for name in self._tables if self._parent_in_view(name) == level]

    def nested(self, at_level: str | None = None) -> pl.LazyFrame:
        """
        Reconstruct the packed ``List[Struct]`` shape, lazily.

        Only worth calling at the boundary where something actually consumes
        nesting; every query above it is cheaper on the flat frames
        :meth:`level` returns.

        Args:
            at_level: The level each row should represent. Defaults to the root,
                giving one row per root entity with its descendants nested.

        Returns:
            An unexecuted ``LazyFrame`` with the nested schema.

        Examples:
            >>> view.nested().collect()             # one row per root entity
            >>> view.nested("store").collect()      # one row per store
        """
        return self._packer.denormalize(  # type: ignore[return-value]
            self._resolved_tables(), at_level=at_level
        )

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

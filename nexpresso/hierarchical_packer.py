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

import math
import shutil
import tempfile
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import polars as pl
from polars.expr.expr import Expr

if TYPE_CHECKING:
    from polars._typing import PolarsDataType

FrameT = TypeVar("FrameT", pl.LazyFrame, pl.DataFrame)

ColumnSelector = str | pl.Expr
ExtraColumnsMode = Literal["preserve", "drop", "error"]
ParentStrategy = Literal["aggregate", "split_join"]
PartitionStrategy = Literal["balanced", "hash"]
PromoteAggregation = Literal[
    "list", "set", "sum", "mean", "min", "max", "first", "last", "count", "single"
]
SchemaInput = pl.Schema | pl.DataFrame | pl.LazyFrame

ROW_ID_COLUMN = "__hier_row_id"
ORDER_TEMP_COLUMN_PREFIX = "__hier_order_"
BUCKET_COLUMN = "__hier_bucket"
DEFAULT_SEPARATOR = "."
DEFAULT_ESCAPE_CHAR = "\\"


def _sorted_bucket_dirs(stage_dir: Path) -> list[Path]:
    """
    Bucket directories written by ``pl.PartitionBy``, in ascending bucket order.

    ``PartitionBy`` names them ``<key>=<value>``. Sorting the names as strings
    puts ``...=10`` before ``...=2``, which would scramble the contiguous key
    ranges that make the concatenated output sorted, so order by the parsed
    integer value instead.

    An empty input produces no partitions at all, so the directory may not even
    exist; that is reported as "no buckets" rather than an error.

    Args:
        stage_dir: Directory holding the partitioned output.

    Returns:
        Bucket directories ordered by their numeric bucket value, or an empty
        list if nothing was staged.
    """
    if not stage_dir.is_dir():
        return []

    def bucket_value(path: Path) -> int:
        _, _, raw = path.name.partition("=")
        return int(raw)

    return sorted((p for p in stage_dir.iterdir() if p.is_dir()), key=bucket_value)


def _split_path_static(
    path: str,
    separator: str = DEFAULT_SEPARATOR,
    escape_char: str = DEFAULT_ESCAPE_CHAR,
) -> list[str]:
    """
    Split a path by separator, respecting escaped separators.

    Standalone version of :meth:`HierarchicalPacker._split_path` for use
    without an instance (e.g. in :meth:`HierarchicalPacker.discover_levels`).

    The separator is matched as a **substring**, not a character, so a
    multi-character separator such as ``"__"`` splits correctly and an escaped
    one (``"a\\__b"``) is a literal part of the field name.

    Args:
        path: The path string to split.
        separator: Separator between path components; may be several characters.
        escape_char: Character used to escape literal separators in field names.

    Returns:
        List of path components.
    """
    if not path:
        return []

    escaped_separator = escape_char + separator
    components: list[str] = []
    current: list[str] = []
    i = 0
    while i < len(path):
        # Longest match first: an escaped separator is a literal separator,
        # however many characters the separator happens to be.
        if path.startswith(escaped_separator, i):
            current.append(separator)
            i += len(escaped_separator)
        elif path.startswith(escape_char, i) and i + len(escape_char) < len(path):
            # Any other escape keeps the following character as-is (``\\`` -> ``\``).
            current.append(path[i + len(escape_char)])
            i += len(escape_char) + 1
        elif path.startswith(separator, i):
            components.append("".join(current))
            current = []
            i += len(separator)
        else:
            current.append(path[i])
            i += 1

    components.append("".join(current))
    return components


# Level parameters renamed in 0.8.0, mapped to their replacement. The rename went
# with an inversion of meaning, so a silently-accepted old keyword would quietly
# return a frame at the wrong granularity — worth refusing loudly instead.
_RENAMED_LEVEL_KWARGS = {
    "to_level": "at_level",
    "root_level": "at_level",
    "target_level": "at_level",
}

# Stands in for a required argument so that a caller passing only the old keyword
# reaches the migration message below instead of Python's bare "missing 1
# required positional argument".
_MISSING_LEVEL: Any = object()


def _reject_legacy_level_kwarg(
    method: str, kwargs: Mapping[str, object], at_level: object = None
) -> None:
    """
    Raise a migration ``TypeError`` for a pre-0.8.0 level keyword.

    Args:
        method: Name of the method that was called, for the message.
        kwargs: The caught ``**_legacy`` mapping. Empty in the normal case.
        at_level: The value the caller passed for the new parameter, when it is
            required. Left as ``None`` by methods whose level is optional.

    Raises:
        TypeError: If ``kwargs`` is non-empty, or if a required ``at_level`` was
            omitted. Renamed level parameters get a migration message; any other
            keyword gets the usual unexpected-keyword error, since swallowing
            typos would be worse than the rename.
    """
    if kwargs:
        name = next(iter(kwargs))
        replacement = _RENAMED_LEVEL_KWARGS.get(name)
        if replacement is None:
            raise TypeError(f"{method}() got an unexpected keyword argument '{name}'")

        raise TypeError(
            f"{method}() parameter '{name}' was renamed to '{replacement}' in 0.8.0, "
            "and its meaning changed: it now names the level each ROW represents, "
            "not the level that gets nested. Rewrite "
            f'{method}(..., {name}="<level>") as '
            f'{method}(..., {replacement}="<parent of that level>").'
        )

    if at_level is _MISSING_LEVEL:
        raise TypeError(f"{method}() missing 1 required positional argument: 'at_level'")


__all__ = [
    "LevelSpec",
    "LevelAttribute",
    "DiscoveredLevel",
    "SchemaValidationResult",
    "HierarchySpec",
    "HierarchicalPacker",
    "HierarchyValidationError",
    "ParentStrategy",
    "PartitionStrategy",
    "PromoteAggregation",
    "SchemaInput",
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
        parent: Name of this level's parent level. Leave ``None`` for a simple
            linear chain — :class:`HierarchySpec` then infers each level's
            parent as the level declared before it. Set it to branch: two levels
            naming the same ``parent`` are siblings, each rooting its own axis::

                LevelSpec(name="street",  parent="city", ...)
                LevelSpec(name="service", parent="city", ...)

            It is all-or-nothing across a spec: if any non-root level declares
            ``parent``, every non-root level must.
    """

    name: str
    id_fields: Sequence[ColumnSelector] = ()
    required_fields: Sequence[ColumnSelector] | None = None
    order_by: Sequence[pl.Expr] | None = None
    parent_keys: Sequence[str] | None = None
    parent: str | None = None


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
    parent: str | None = None
    children: tuple[str, ...] = ()


@dataclass(frozen=True)
class LevelAttribute:
    """
    Declarative specification of an attribute derived from a particular level.

    Used with :meth:`HierarchicalPacker.enrich` to annotate a packed frame with
    multiple cross-level attributes in a single call.

    Args:
        attribute: Column / field name at ``from_level``.
        from_level: The level where the attribute lives.  May be the same as
            the target level (same-level access) or any descendant.
        agg: Aggregation applied when rolling up to the target level.
            Defaults to ``"list"``.
        alias: Output column name (unqualified).  Defaults to ``attribute``.
    """

    attribute: str
    from_level: str
    agg: PromoteAggregation = "list"
    alias: str | None = None


@dataclass(frozen=True)
class DiscoveredLevel:
    """
    A hierarchy level inferred from schema inspection.

    Produced by :meth:`HierarchicalPacker.discover_levels` when examining a
    schema without a pre-existing :class:`HierarchySpec`.

    Args:
        name: Inferred level name (the path component, e.g. ``"city"``).
        depth: Zero-based depth in the hierarchy tree (0 = root).
        path: Full separator-joined path from root to this level
            (e.g. ``"country.city"``).
        fields: Non-level scalar field names at this level.
        parent: Name of the parent level, or ``None`` for the root.
        is_packed: ``True`` if this level was discovered inside a
            ``List[Struct]`` or ``Struct`` type rather than from flat
            dotted column names.
    """

    name: str
    depth: int
    path: str
    fields: tuple[str, ...]
    parent: str | None
    is_packed: bool = False


@dataclass(frozen=True)
class SchemaValidationResult:
    """
    Result of schema compatibility validation.

    Produced by :meth:`HierarchicalPacker.validate_schema`.

    Args:
        is_compatible: ``True`` if the schema is usable with this packer.
        inferred_level: The current packing level inferred from the schema,
            or ``None`` if inference failed.
        present_levels: Level names whose columns/fields were found.
        missing_levels: Level names whose expected columns are absent.
        errors: Fatal incompatibilities (human-readable descriptions).
        warnings: Non-fatal issues (e.g. missing optional fields).
    """

    is_compatible: bool
    inferred_level: str | None
    present_levels: list[str]
    missing_levels: list[str]
    errors: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class HierarchySpec:
    """
    Collection of ``LevelSpec`` objects describing a hierarchy tree.

    Levels are declared coarse → fine, parents before children. In the common
    case the hierarchy is a single chain and each level's parent is simply the
    level declared before it::

        HierarchySpec.from_levels(
            LevelSpec(name="country", id_fields=["code"]),
            LevelSpec(name="city", id_fields=["id"]),
            LevelSpec(name="street", id_fields=["id"]),
        )

    A level may instead name its :attr:`LevelSpec.parent` explicitly, which lets
    one level carry several independent child branches. ``street`` and
    ``service`` below are siblings — orthogonal properties of a city, not stages
    of one chain::

        HierarchySpec.from_levels(
            LevelSpec(name="country", id_fields=["code"]),
            LevelSpec(name="city", id_fields=["id"], parent="country"),
            LevelSpec(name="street", id_fields=["id"], parent="city"),
            LevelSpec(name="building", id_fields=["id"], parent="street"),
            LevelSpec(name="service", id_fields=["kind"], parent="city"),
        )

    Each root → level chain is an **axis**. Because a level's ancestors are
    unique even in a branching tree, column paths
    (``country.city.service.kind``) and key propagation work identically in both
    shapes; only iteration over children becomes plural. ``pack`` / ``unpack``
    act along the axis implied by their target level, leaving sibling branches
    packed as ``List[Struct]`` columns.

    Args:
        levels: Sequence of LevelSpec objects, parents before children.
        key_aliases: **Deprecated.** Mapping of {target_column: source_column}
            used to synthesize a missing key column from another column. Rename
            the column on the frame instead — it is one expression and the
            renamed column then behaves like any other, including surviving
            :meth:`HierarchicalPacker.normalize` /
            :meth:`HierarchicalPacker.denormalize` round-trips, which synthesized
            keys do not::

                df = df.with_columns(pl.col("country.city.id").alias("country.code"))
    """

    levels: Sequence[LevelSpec]
    key_aliases: Mapping[str, str] = field(default_factory=dict)

    # Resolved tree, derived in __post_init__ (never passed by callers).
    _parents: dict[str, str | None] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )
    _children: dict[str, tuple[str, ...]] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        """Validate the level tree and warn on deprecated options."""
        level_names = [lvl.name for lvl in self.levels]
        if len(level_names) != len(set(level_names)):
            raise ValueError("Level names must be unique inside a HierarchySpec.")

        object.__setattr__(self, "_parents", self._resolve_parents())
        object.__setattr__(self, "_children", self._resolve_children())

        if self.key_aliases:
            example_target, example_source = next(iter(self.key_aliases.items()))
            warnings.warn(
                "HierarchySpec.key_aliases is deprecated and will be removed in a "
                "future release. Rename the column on the frame instead: "
                f'df.with_columns(pl.col("{example_source}").alias("{example_target}")). '
                "Synthesized key columns are stripped from the per-level tables, so "
                "normalize()/denormalize() cannot round-trip a hierarchy that relies "
                "on them.",
                DeprecationWarning,
                stacklevel=3,
            )

    def _resolve_parents(self) -> dict[str, str | None]:
        """
        Resolve each level's parent, inferring a linear chain when none declare one.

        Returns:
            Mapping of level name → parent name (``None`` for the root).

        Raises:
            ValueError: If the tree is malformed — a partially declared spec,
                an unknown or self-referencing parent, several roots, a level
                declared before its parent, or a cycle.
        """
        if not self.levels:
            return {}

        declared = [lvl for lvl in self.levels[1:] if lvl.parent is not None]
        if declared and len(declared) != len(self.levels) - 1:
            missing = [lvl.name for lvl in self.levels[1:] if lvl.parent is None]
            raise ValueError(
                f"Level(s) {missing} do not declare a parent while others do. "
                "Parent declaration is all-or-nothing: either no level sets "
                "parent= (the hierarchy is then read as a linear chain in "
                "declaration order) or every non-root level sets it. Inferring "
                "the rest from declaration order would silently attach a "
                "branch to whichever level happened to be declared last."
            )

        if self.levels[0].parent is not None:
            raise ValueError(
                f"Root level '{self.levels[0].name}' declares parent "
                f"'{self.levels[0].parent}' but must not have one. The root is "
                "the first level declared."
            )

        parents: dict[str, str | None] = {self.levels[0].name: None}

        if not declared:
            # Linear chain: each level's parent is the one declared before it.
            for previous, level in zip(self.levels[:-1], self.levels[1:]):
                parents[level.name] = previous.name
            return parents

        known = {lvl.name for lvl in self.levels}
        seen = {self.levels[0].name}

        for level in self.levels[1:]:
            parent = level.parent
            assert parent is not None  # guaranteed by the all-or-nothing check above
            if parent == level.name:
                raise ValueError(f"Level '{level.name}' cannot be its own parent.")
            if parent not in known:
                raise ValueError(
                    f"Level '{level.name}' names unknown parent '{parent}'. "
                    f"Known levels: {sorted(known)}."
                )
            if parent not in seen:
                # Declaring a child before its parent would also admit cycles,
                # since every parent here is already known to exist.
                raise ValueError(
                    f"Level '{level.name}' is declared before its parent "
                    f"'{parent}'. Declare parents before their children so the "
                    "hierarchy reads root → leaf."
                )
            parents[level.name] = parent
            seen.add(level.name)

        return parents

    def _resolve_children(self) -> dict[str, tuple[str, ...]]:
        """Invert the parent map, preserving declaration order among siblings."""
        children: dict[str, list[str]] = {lvl.name: [] for lvl in self.levels}
        for level in self.levels:
            parent = self._parents[level.name]
            if parent is not None:
                children[parent].append(level.name)
        return {name: tuple(kids) for name, kids in children.items()}

    def parent_of(self, level_name: str) -> str | None:
        """
        The parent of ``level_name``, or ``None`` if it is the root.

        Args:
            level_name: The level whose parent to look up.

        Returns:
            The parent level's name, or ``None`` for the root.

        Raises:
            KeyError: If the level is not found.
        """
        self._require_level(level_name)
        return self._parents[level_name]

    def children_of(self, level_name: str) -> list[str]:
        """
        The immediate children of ``level_name``, in declaration order.

        A chain gives at most one child; a branching level gives several.

        Args:
            level_name: The level whose children to look up.

        Returns:
            List of child level names (empty for a leaf).

        Raises:
            KeyError: If the level is not found.
        """
        self._require_level(level_name)
        return list(self._children[level_name])

    def ancestors_of(self, level_name: str) -> list[str]:
        """
        The chain of ancestors above ``level_name``, ordered root → parent.

        Unique even in a branching tree — every level has exactly one path back
        to the root, which is why column paths and key propagation are
        unambiguous regardless of hierarchy shape.

        Args:
            level_name: The level whose ancestors to look up.

        Returns:
            List of ancestor names, root first. Empty for the root.

        Raises:
            KeyError: If the level is not found.
        """
        self._require_level(level_name)
        chain: list[str] = []
        current = self._parents[level_name]
        while current is not None:
            chain.append(current)
            current = self._parents[current]
        chain.reverse()
        return chain

    def axis_of(self, level_name: str) -> list[str]:
        """
        The root → ``level_name`` chain, inclusive.

        This is the *axis* that :meth:`HierarchicalPacker.pack` and
        :meth:`HierarchicalPacker.unpack` traverse for a given target level.

        Args:
            level_name: The level terminating the axis.

        Returns:
            List of level names from the root down to ``level_name``.

        Raises:
            KeyError: If the level is not found.
        """
        return [*self.ancestors_of(level_name), level_name]

    def descendants_of(self, level_name: str) -> list[str]:
        """
        Every level below ``level_name``, in declaration (topological) order.

        Spans all branches of the subtree, not just one axis.

        Args:
            level_name: The level whose subtree to list.

        Returns:
            List of descendant names, parents before children. Empty for a leaf.

        Raises:
            KeyError: If the level is not found.
        """
        self._require_level(level_name)
        subtree = {level_name}
        found: list[str] = []
        for level in self.levels:
            parent = self._parents[level.name]
            if parent in subtree:
                subtree.add(level.name)
                found.append(level.name)
        return found

    def is_ancestor_of(self, level_name: str, other: str) -> bool:
        """
        Whether ``level_name`` lies on the root → ``other`` axis, above ``other``.

        Args:
            level_name: The candidate ancestor.
            other: The candidate descendant.

        Returns:
            ``True`` if ``level_name`` is a strict ancestor of ``other``.

        Raises:
            KeyError: If either level is not found.
        """
        self._require_level(level_name)
        return level_name in self.ancestors_of(other)

    @property
    def root(self) -> str:
        """The name of the single root level."""
        return self.levels[0].name

    @property
    def leaves(self) -> list[str]:
        """Names of every level with no children, in declaration order."""
        return [lvl.name for lvl in self.levels if not self._children[lvl.name]]

    def topological_levels(self) -> list[str]:
        """Every level name, parents before children (declaration order)."""
        return [lvl.name for lvl in self.levels]

    def reverse_topological_levels(self) -> list[str]:
        """Every level name, children before parents."""
        return [lvl.name for lvl in reversed(self.levels)]

    def _require_level(self, level_name: str) -> None:
        """Raise a descriptive KeyError if ``level_name`` is not in the hierarchy."""
        if level_name not in self._parents:
            raise KeyError(
                f"Level '{level_name}' not found in hierarchy. "
                f"Known levels: {[lvl.name for lvl in self.levels]}."
            )

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
            key_aliases: **Deprecated** — see :class:`HierarchySpec`. Rename the
                column on the frame instead. Passing a non-empty mapping emits a
                ``DeprecationWarning``.

        Returns:
            A new HierarchySpec instance.

        Raises:
            ValueError: If parent_keys don't match parent's id_fields in count.
        """
        spec = cls(levels=list(levels), key_aliases=key_aliases or {})

        # Validate parent_keys compatibility against the *resolved* parent, which
        # is the level declared before this one only in an unbranched chain.
        by_name = spec.levels_by_name
        for level in levels:
            parent_name = spec.parent_of(level.name)
            if parent_name is None:
                if level.parent_keys:
                    raise ValueError(
                        f"Root level '{level.name}' should not have parent_keys defined."
                    )
                continue
            if not level.parent_keys:
                continue
            parent_id_count = len(by_name[parent_name].id_fields)
            if len(level.parent_keys) != parent_id_count:
                raise ValueError(
                    f"Level '{level.name}' has {len(level.parent_keys)} parent_keys "
                    f"but parent '{parent_name}' has {parent_id_count} id_fields. "
                    "These must match."
                )

        return spec

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
        Get the single child level below the given level.

        Args:
            level_name: The name of the current level.

        Returns:
            The child LevelSpec, or None if this is a leaf level.

        Raises:
            ValueError: If the level has several children, where "next" has no
                single answer. Use :meth:`children_of` instead.
        """
        children = self.children_of(level_name)
        if not children:
            return None
        if len(children) > 1:
            raise ValueError(
                f"Level '{level_name}' has {len(children)} child branches "
                f"{children}, so 'the next level' is ambiguous. Use "
                "children_of() to get them all, or name the branch you mean."
            )
        return self.levels_by_name[children[0]]


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
    # Introspection Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_schema(schema_or_frame: SchemaInput) -> pl.Schema:
        """Extract a ``pl.Schema`` from a Schema, DataFrame, or LazyFrame."""
        if isinstance(schema_or_frame, pl.LazyFrame):
            return schema_or_frame.collect_schema()
        if isinstance(schema_or_frame, pl.DataFrame):
            return schema_or_frame.schema
        return schema_or_frame

    @property
    def level_names(self) -> list[str]:
        """
        Return all level names ordered from root (coarsest) to leaf (finest).

        Returns:
            List of level name strings.

        Examples:
            >>> packer.level_names
            ['country', 'city', 'street']
        """
        return [m.name for m in self._levels_meta]

    @property
    def root_level(self) -> str:
        """
        Return the name of the coarsest (root) level.

        Returns:
            Root level name.

        Examples:
            >>> packer.root_level
            'country'
        """
        return self._levels_meta[0].name

    @property
    def leaf_level(self) -> str:
        """
        Return the name of the single finest (leaf) level.

        Returns:
            Leaf level name.

        Raises:
            ValueError: If the hierarchy branches and therefore has several
                leaves. Use :attr:`leaf_levels` instead.

        Examples:
            >>> packer.leaf_level
            'street'
        """
        leaves = self.spec.leaves
        if len(leaves) > 1:
            raise ValueError(
                f"Hierarchy has {len(leaves)} leaf levels {leaves}, one per branch, "
                "so 'the leaf level' is ambiguous. Use leaf_levels, or name the "
                "branch you mean."
            )
        return leaves[0]

    @property
    def leaf_levels(self) -> list[str]:
        """
        Return the name of every leaf level — one per branch of the hierarchy.

        Returns:
            List of leaf level names in declaration order. A single-element list
            for an unbranched chain.

        Examples:
            >>> packer.leaf_levels
            ['building', 'service']
        """
        return self.spec.leaves

    @property
    def axes(self) -> list[list[str]]:
        """
        Return every root → leaf axis in the hierarchy.

        An *axis* is the chain :meth:`pack` and :meth:`unpack` traverse for a
        given target level. An unbranched hierarchy has exactly one.

        Returns:
            List of level-name chains, root first, one per leaf level.

        Examples:
            >>> packer.axes
            [['country', 'city', 'street', 'building'], ['country', 'city', 'service']]
        """
        return [self.spec.axis_of(leaf) for leaf in self.spec.leaves]

    def get_axis(self, level: str) -> list[str]:
        """
        Return the root → ``level`` chain, inclusive.

        This is the sequence of levels :meth:`pack` folds and :meth:`unpack`
        explodes when ``level`` is the target. Branches hanging off it are left
        packed.

        Args:
            level: The level terminating the axis.

        Returns:
            List of level names from the root down to ``level``.

        Raises:
            KeyError: If ``level`` is not found in the hierarchy.

        Examples:
            >>> packer.get_axis('building')
            ['country', 'city', 'street', 'building']
            >>> packer.get_axis('service')
            ['country', 'city', 'service']
        """
        return self.spec.axis_of(level)

    def get_ancestor_levels(self, level: str) -> list[str]:
        """
        Return all ancestor level names above ``level``, ordered root → parent.

        Even in a branching hierarchy a level has exactly one path back to the
        root, so this chain is always unambiguous.

        Args:
            level: The level whose ancestors to retrieve.

        Returns:
            List of ancestor level names. Empty list if ``level`` is the root.

        Raises:
            KeyError: If ``level`` is not found in the hierarchy.

        Examples:
            >>> packer.get_ancestor_levels('street')
            ['country', 'city']
            >>> packer.get_ancestor_levels('country')
            []
        """
        return self.spec.ancestors_of(level)

    def get_descendant_levels(self, level: str) -> list[str]:
        """
        Return all descendant level names below ``level``, parents before children.

        Spans every branch of the subtree, not just one axis.

        Args:
            level: The level whose descendants to retrieve.

        Returns:
            List of descendant level names. Empty list if ``level`` is a leaf.

        Raises:
            KeyError: If ``level`` is not found in the hierarchy.

        Examples:
            >>> packer.get_descendant_levels('country')
            ['city', 'street', 'building', 'service']
            >>> packer.get_descendant_levels('building')
            []
        """
        return self.spec.descendants_of(level)

    def get_child_levels(self, level: str) -> list[str]:
        """
        Return the immediate children of ``level``, in declaration order.

        Returns several names where the hierarchy branches, one in a chain, and
        none at a leaf.

        Args:
            level: The level whose children to retrieve.

        Returns:
            List of child level names.

        Raises:
            KeyError: If ``level`` is not found in the hierarchy.

        Examples:
            >>> packer.get_child_levels('city')
            ['street', 'service']
        """
        return self.spec.children_of(level)

    def get_level_keys(
        self,
        level: str,
        *,
        include_ancestors: bool = False,
        form: Literal["short", "long"] = "short",
    ) -> list[str]:
        """
        Return the identifying key column names for ``level``.

        Args:
            level: The level whose keys to retrieve.
            include_ancestors: If ``True``, also include all ancestor key columns
                before the level's own keys. Forces ``form="long"`` to avoid
                ambiguity between same-named keys at different levels.
            form: Output form for column names.
                - ``"short"``: unqualified field name only (e.g. ``"id"``).
                - ``"long"``: fully qualified path (e.g. ``"country.city.id"``).
                Defaults to ``"short"``.

        Returns:
            List of key column name strings.

        Raises:
            KeyError: If ``level`` is not found in the hierarchy.

        Examples:
            >>> packer.get_level_keys('city')
            ['id', 'name']
            >>> packer.get_level_keys('city', form='long')
            ['country.city.id', 'country.city.name']
            >>> packer.get_level_keys('city', include_ancestors=True)
            ['country.code', 'country.city.id', 'country.city.name']
        """
        meta = self._levels_meta[self.spec.index_of(level)]
        if include_ancestors:
            # Always use long form to avoid ambiguity when ancestor keys are included,
            # since multiple levels may share the same short key name (e.g. "id").
            return list(meta.ancestor_keys) + list(meta.id_columns)
        if form == "long":
            return list(meta.id_columns)
        # short form: strip the level prefix from each qualified id column
        return [col[len(meta.prefix) :] for col in meta.id_columns]

    def get_level_fields(
        self,
        level: str,
        schema_or_frame: SchemaInput,
        *,
        form: Literal["short", "long"] = "short",
    ) -> list[str]:
        """
        Return all column/field names that belong to ``level`` in the given schema.

        Works with both flat (fully unpacked) and packed schemas.  In a flat
        schema every hierarchy column is a top-level column with a dotted prefix;
        in a packed schema the level's own fields live inside a
        ``List[Struct]`` or ``Struct`` column.

        Args:
            level: The level whose fields to extract.
            schema_or_frame: A ``pl.Schema``, ``pl.DataFrame``, or ``pl.LazyFrame``
                whose schema is inspected.
            form: Output form for field names.
                - ``"short"``: unqualified field name only (e.g. ``"name"``).
                - ``"long"``: fully qualified dotted path
                  (e.g. ``"country.city.street.name"``).
                Defaults to ``"short"``.

        Returns:
            List of field name strings for the requested level.

        Raises:
            KeyError: If ``level`` is not found in the hierarchy.

        Examples:
            >>> # Flat schema
            >>> packer.get_level_fields('city', flat_df)
            ['id', 'name', 'population']
            >>> packer.get_level_fields('city', flat_df, form='long')
            ['country.city.id', 'country.city.name', 'country.city.population']
            >>> # Packed schema
            >>> city_packed = packer.pack(flat_df, 'city')
            >>> packer.get_level_fields('city', city_packed)
            ['id', 'name', 'population']
        """
        schema = self._extract_schema(schema_or_frame)
        meta = self._levels_meta[self.spec.index_of(level)]

        # Collect names of immediate child levels to exclude their columns
        child_level_names = set(meta.children)

        # ---- Packed case: the level's path is a column in the schema ----
        if meta.path in schema:
            dtype = schema[meta.path]
            # Unwrap List wrapper if present
            inner = dtype.inner if isinstance(dtype, pl.List) else dtype
            if isinstance(inner, pl.Struct):
                fields: list[str] = []
                for f in inner.fields:
                    # Exclude sub-hierarchy fields (child level structs/lists)
                    if f.name in child_level_names:
                        continue
                    if form == "long":
                        fields.append(f"{meta.prefix}{f.name}")
                    else:
                        fields.append(f.name)
                return fields

        # ---- Flat case: level columns are top-level with prefix ----
        if not meta.prefix:
            # Root level — should not normally appear without a prefix
            return []

        result: list[str] = []
        for col in schema.keys():
            if not col.startswith(meta.prefix):
                continue
            remainder = col[len(meta.prefix) :]
            # Exclude columns that belong to child levels
            if any(
                remainder == n or remainder.startswith(n + self.separator)
                for n in child_level_names
            ):
                continue
            if form == "long":
                result.append(col)
            else:
                result.append(remainder)
        return result

    def infer_current_level(self, schema_or_frame: SchemaInput, *, axis: str | None = None) -> str:
        """
        Infer which hierarchy level each row currently represents.

        Row granularity is set by the **deepest level whose own columns are
        still flat**: everything below it has been folded into nested columns,
        so one row is one entity at that level.

        Note that this reports **row granularity**, which is one level coarser
        than the argument to :meth:`pack`: ``pack(frame, "city")`` nests the
        city level into a list column, so the resulting rows are at ``country``
        granularity.

        Args:
            schema_or_frame: A ``pl.Schema``, ``pl.DataFrame``, or ``pl.LazyFrame``
                to inspect.
            axis: Optional level naming the axis to measure along. Only the root
                → ``axis`` chain is considered, which resolves the ambiguity when
                a branching hierarchy has flat columns from two branches at once.

        Returns:
            The name of the level each row currently represents.

        Raises:
            KeyError: If ``axis`` is not found in the hierarchy.
            ValueError: If the schema does not match any recognisable hierarchy
                state, or if flat columns from two different branches are present
                and no ``axis`` was given to disambiguate.

        Examples:
            >>> packer.infer_current_level(flat_df)
            'apartment'
            >>> packer.infer_current_level(packer.pack(flat_df, 'street'))
            'city'
            >>> packer.infer_current_level(nested, axis='service')
            'country'
        """
        schema = self._extract_schema(schema_or_frame)

        candidates = (
            [self._levels_meta[self.spec.index_of(name)] for name in self.spec.axis_of(axis)]
            if axis is not None
            else self._levels_meta
        )
        flat = [meta for meta in candidates if self._own_level_columns(meta, schema)]

        if not flat:
            # Fully packed: the whole hierarchy sits inside the root's column, so
            # each row is one root entity.
            root = self._levels_meta[0]
            if root.path in schema or any(col.startswith(root.prefix) for col in schema.keys()):
                return root.name
            raise ValueError(
                "Cannot infer current level: the schema does not match any recognisable "
                f"hierarchy state. Schema columns: {list(schema.keys())}"
            )

        # The deepest flat level. In a chain that is simply the last one; when the
        # hierarchy branches, two flat levels on different branches describe two
        # different granularities and no single answer is right.
        deepest = max(flat, key=lambda meta: len(self.spec.ancestors_of(meta.name)))
        conflicting = [
            meta.name
            for meta in flat
            if meta.name != deepest.name and not self.spec.is_ancestor_of(meta.name, deepest.name)
        ]
        if conflicting:
            raise ValueError(
                f"Cannot infer current level: levels {[deepest.name, *conflicting]} are all "
                "flat but lie on different branches, so the frame mixes granularities. "
                "Pass axis='<level>' to measure along one branch."
            )
        return deepest.name

    def get_level_schema(
        self,
        level: str,
        schema_or_frame: SchemaInput,
    ) -> dict[str, pl.DataType]:
        """
        Return a mapping of field name → data type for all fields at ``level``.

        Works with both flat and packed schemas (see :meth:`get_level_fields`).

        Args:
            level: The level whose field types to retrieve.
            schema_or_frame: A ``pl.Schema``, ``pl.DataFrame``, or ``pl.LazyFrame``
                whose schema is inspected.

        Returns:
            Dictionary mapping short field names to their ``pl.DataType``.

        Raises:
            KeyError: If ``level`` is not found in the hierarchy.

        Examples:
            >>> packer.get_level_schema('city', flat_df)
            {'id': String, 'name': String, 'population': Int64}
        """
        schema = self._extract_schema(schema_or_frame)
        meta = self._levels_meta[self.spec.index_of(level)]

        child_level_names = set(meta.children)

        # Packed case
        if meta.path in schema:
            dtype = schema[meta.path]
            inner = dtype.inner if isinstance(dtype, pl.List) else dtype
            if isinstance(inner, pl.Struct):
                return {
                    f.name: f.dtype  # type: ignore[misc]
                    for f in inner.fields
                    if f.name not in child_level_names
                }

        # Flat case
        if not meta.prefix:
            return {}

        result: dict[str, pl.DataType] = {}
        for col, dtype in schema.items():
            if not col.startswith(meta.prefix):
                continue
            remainder = col[len(meta.prefix) :]
            if any(
                remainder == n or remainder.startswith(n + self.separator)
                for n in child_level_names
            ):
                continue
            result[remainder] = dtype
        return result

    def describe(self) -> str:
        """
        Return a human-readable summary of the hierarchy structure.

        Returns:
            Multi-line string describing all levels, their paths, keys, and
            ancestor keys.

        Examples:
            >>> print(packer.describe())
            HierarchicalPacker (separator=".")
              Levels (3):
                0. country  (root)
                   Path: "country"
                   Keys: code
                1. city
                   Path: "country.city"
                   Keys: id, name
                   Ancestor keys: country.code
                2. street  (leaf)
                   Path: "country.city.street"
                   Keys: name
                   Ancestor keys: country.code, country.city.id, country.city.name
        """
        n = len(self._levels_meta)
        lines: list[str] = [f'HierarchicalPacker (separator="{self.separator}")']
        lines.append(f"  Levels ({n}):")
        for meta in self._levels_meta:
            tags: list[str] = []
            if meta.parent is None:
                tags.append("root")
            if not meta.children:
                tags.append("leaf")
            tag_str = f"  ({', '.join(tags)})" if tags else ""
            lines.append(f"    {meta.index}. {meta.name}{tag_str}")
            lines.append(f'       Path: "{meta.path}"')
            if len(meta.children) > 1:
                # Only worth showing where it is not implied by the ordering.
                lines.append(f"       Branches: {', '.join(meta.children)}")
            keys_str = (
                ", ".join(col[len(meta.prefix) :] for col in meta.id_columns)
                if meta.id_columns
                else "(none)"
            )
            lines.append(f"       Keys: {keys_str}")
            if meta.ancestor_keys:
                lines.append(f"       Ancestor keys: {', '.join(meta.ancestor_keys)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Hierarchy Discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _discover_from_struct(
        dtype: pl.Struct,
        parent_path: tuple[str, ...],
        levels: dict[tuple[str, ...], set[str]],
        packed_paths: set[tuple[str, ...]],
    ) -> None:
        """
        Recursively discover levels inside a ``Struct`` dtype.

        For each struct field whose dtype is ``List[Struct]`` or ``Struct``
        (with further nested fields), a child level is registered.  Other
        fields are recorded as data fields of ``parent_path``.

        Args:
            dtype: The Struct data type to inspect.
            parent_path: Tuple of path components leading to this struct.
            levels: Mutable dict mapping level path tuples to their field name sets.
            packed_paths: Mutable set tracking which paths were found inside packed columns.
        """
        for struct_field in dtype.fields:
            inner = struct_field.dtype
            # Unwrap List wrapper
            inner_unwrapped = inner.inner if isinstance(inner, pl.List) else inner

            if isinstance(inner_unwrapped, pl.Struct) and inner_unwrapped.fields:
                # This field represents a child level
                child_path = parent_path + (struct_field.name,)
                if child_path not in levels:
                    levels[child_path] = set()
                packed_paths.add(child_path)
                HierarchicalPacker._discover_from_struct(
                    inner_unwrapped, child_path, levels, packed_paths
                )
            else:
                # Scalar or non-hierarchical field at the current level
                levels[parent_path].add(struct_field.name)

    @staticmethod
    def discover_levels(
        schema_or_frame: SchemaInput,
        *,
        separator: str = DEFAULT_SEPARATOR,
        escape_char: str = DEFAULT_ESCAPE_CHAR,
    ) -> list[DiscoveredLevel]:
        """
        Infer hierarchy levels from a schema without a pre-existing spec.

        Examines column names (splitting by ``separator``) and nested
        ``List[Struct]`` / ``Struct`` types to determine what hierarchy levels
        the data contains.

        This is a **static method** — no :class:`HierarchicalPacker` instance
        is needed.

        Args:
            schema_or_frame: A ``pl.Schema``, ``pl.DataFrame``, or ``pl.LazyFrame``
                to inspect.
            separator: Separator character between path components.
                Defaults to ``"."``.
            escape_char: Character used to escape literal separators in field
                names.  Defaults to ``"\\\\"``.

        Returns:
            List of :class:`DiscoveredLevel` objects sorted by depth then name.

        Examples:
            >>> df = pl.DataFrame({
            ...     "country.code": ["US"],
            ...     "country.city.id": ["NYC"],
            ...     "country.city.name": ["New York"],
            ... })
            >>> levels = HierarchicalPacker.discover_levels(df)
            >>> [lvl.name for lvl in levels]
            ['country', 'city']
        """
        schema = HierarchicalPacker._extract_schema(schema_or_frame)

        # level path tuple → set of field names at that level
        levels: dict[tuple[str, ...], set[str]] = {}
        # Paths discovered inside packed columns
        packed_paths: set[tuple[str, ...]] = set()

        for col_name, col_dtype in schema.items():
            parts = _split_path_static(col_name, separator, escape_char)

            # Check if this column is a packed nested type
            inner = col_dtype.inner if isinstance(col_dtype, pl.List) else col_dtype
            if isinstance(inner, pl.Struct) and inner.fields:
                # The column path IS a level, and the struct contains child data
                level_path = tuple(parts)
                if level_path not in levels:
                    levels[level_path] = set()
                packed_paths.add(level_path)
                HierarchicalPacker._discover_from_struct(inner, level_path, levels, packed_paths)
            elif len(parts) >= 2:
                # Flat scalar column: components except last form the level path
                level_path = tuple(parts[:-1])
                field_name = parts[-1]
                if level_path not in levels:
                    levels[level_path] = set()
                levels[level_path].add(field_name)
            # else: single-component scalar column — not hierarchical, skip

        # Ensure all intermediate paths exist (e.g. if we only saw
        # "country.city.street.name", the "country" level should exist too)
        all_paths = list(levels.keys())
        for path in all_paths:
            for i in range(1, len(path)):
                prefix = path[:i]
                if prefix not in levels:
                    levels[prefix] = set()

        if not levels:
            return []

        # Build DiscoveredLevel objects sorted by depth then name
        result: list[DiscoveredLevel] = []
        for path_tuple in sorted(levels.keys(), key=lambda p: (len(p), p)):
            name = path_tuple[-1]
            depth = len(path_tuple) - 1
            full_path = separator.join(path_tuple)
            parent = path_tuple[-2] if len(path_tuple) > 1 else None
            is_packed = path_tuple in packed_paths
            result.append(
                DiscoveredLevel(
                    name=name,
                    depth=depth,
                    path=full_path,
                    fields=tuple(sorted(levels[path_tuple])),
                    parent=parent,
                    is_packed=is_packed,
                )
            )

        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def pack(
        self,
        frame: FrameT,
        at_level: str = _MISSING_LEVEL,
        *,
        extra_columns: ExtraColumnsMode = "preserve",
        parent_strategy: ParentStrategy = "aggregate",
        **_legacy: Any,
    ) -> FrameT:
        """
        Pack so that each row is one ``at_level`` entity.

        Every level *below* ``at_level`` is folded into nested
        ``List[Struct]`` columns; ``at_level``'s own columns stay flat. This is
        the exact inverse of :meth:`unpack`, which explodes down to the same
        granularity::

            packer.pack(flat, "city")       # one row per city
            packer.unpack(nested, "city")   # one row per city

        and it makes :meth:`infer_current_level` a true inverse::

            packer.infer_current_level(packer.pack(df, level)) == level

        Packing to a leaf level folds nothing: a leaf-granularity frame is
        already flat.

        Args:
            frame: The DataFrame or LazyFrame to pack.
            at_level: The level each row should represent. Everything below it
                is nested; its own columns and its ancestors' stay flat.
            extra_columns: How to handle columns that don't belong to the hierarchy:
                - ``"preserve"``: Keep extra columns if they have uniform values
                  within each group (default). Raises error if values differ.
                - ``"drop"``: Silently drop extra columns.
                - ``"error"``: Raise an error if any extra columns are present.
            parent_strategy: How to carry the root level's own attribute columns:
                - ``"aggregate"`` (default): collapse them through the pack
                  ``group_by`` along with everything else. Best in the common case.
                - ``"split_join"``: pull the root attributes into a small dimension
                  table (unique per root key) and reattach them after packing only
                  the structural and child columns. Equivalent in contents, and far
                  cheaper *when the root attributes are heavy relative to the child
                  data* (e.g. large per-entity blobs/embeddings replicated across
                  every leaf row); for child-dominated data it adds join overhead
                  for no gain. See ``benchmarks/README.md`` for measured trade-offs.

        Returns:
            Packed frame with nested structures, same type as input.

        Raises:
            KeyError: If the level is not found in the hierarchy.
            HierarchyValidationError: If validation is enabled and data integrity
                issues are detected, or if extra_columns="error" and extra columns
                are present.

        Note:
            ``validate_on_pack`` only applies to **eager** input. The uniformity
            check needs to execute the query, so running it on a LazyFrame would
            break the lazy contract; it is therefore skipped and the returned plan
            stays unexecuted. Call :meth:`validate` explicitly if you need the
            check on a lazy pipeline.

            Packing is a ``group_by`` that collects children into lists. Polars'
            streaming engine has no native list-collecting aggregation, so this
            node falls back to the in-memory engine and peak memory scales with
            the whole dataset. Use :meth:`pack_streaming` for inputs that do not
            fit in memory.
        """
        _reject_legacy_level_kwarg("pack", _legacy, at_level)

        if parent_strategy == "split_join":
            return self._pack_split_join(frame, at_level, extra_columns=extra_columns)

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

        # Validation requires eager .collect(); skip it when frame is lazy to preserve lazy semantics.
        effective_validate = self.validate_on_pack and isinstance(frame, pl.DataFrame)

        # Fold everything strictly below ``at_level``, children before parents so
        # that every descendant is already a List[Struct] column by the time its
        # parent's struct is built. ``at_level`` itself is never folded — that is
        # what leaves its rows at the requested granularity. In a branching
        # hierarchy a level's branches all share its prefix, so
        # ``_pack_single_level`` collects them together; branches absent from
        # ``frame`` contribute no columns and are skipped there at no cost.
        for name in self._fold_order(at_level):
            lf, schema = self._pack_single_level(
                lf, self.spec.index_of(name), schema, validate=effective_validate
            )

        if added_cols:
            lf = lf.drop(*added_cols, strict=False)

        lf = self._drop_internal_columns(lf)

        # Settle on one canonical column order. Left alone, the order depends on
        # how the input happened to be laid out: a branch already packed in
        # ``frame`` is carried through in its old position, while a branch folded
        # here is appended. Packing the same data along two different axes would
        # then give two different column orders, and ``denormalize`` — which
        # assembles from per-level tables and has no input order to inherit —
        # could not match either. Projection is cheap; ambiguity is not.
        lf = lf.select(self._pack_column_order(lf.collect_schema(), at_level))
        return self._match_frame_type(lf, frame)

    def _root_attribute_columns(self, schema: pl.Schema) -> list[str]:
        """
        Columns owned by the root level itself: under the root prefix but not a
        root id column and not part of any child branch (or an internal column).
        """
        root = self._levels_meta[0]
        children = [self._levels_meta[self.spec.index_of(name)] for name in root.children]
        id_columns = set(root.id_columns)

        attrs: list[str] = []
        for col in schema.keys():
            if not col.startswith(root.prefix) or col in id_columns:
                continue
            if any(col.startswith(child.prefix) or col == child.path for child in children):
                continue
            if col == ROW_ID_COLUMN:
                continue
            attrs.append(col)
        return attrs

    def _pack_split_join(
        self, frame: FrameT, at_level: str, *, extra_columns: ExtraColumnsMode
    ) -> FrameT:
        """
        Pack while reattaching root-level attributes via a join instead of carrying
        them through the aggregation. See :meth:`pack` (``parent_strategy``).

        The root's columns are never folded, so its attributes always come back
        as plain columns and a single row join covers every target level.
        """
        lf, _added, schema = self._prepare_frame(frame)
        root = self._levels_meta[0]
        root_keys: list[str] = list(root.id_columns)
        attr_cols: list[str] = self._root_attribute_columns(schema)

        # Nothing to split off → fall back to the standard aggregation pack.
        if not root_keys or not attr_cols:
            return self.pack(frame, at_level, extra_columns=extra_columns)

        dim = lf.select([*root_keys, *attr_cols]).unique(subset=root_keys)
        structural = lf.drop(*attr_cols)
        packed = self._to_lazy(self.pack(structural, at_level, extra_columns=extra_columns))
        # The join appends the root attributes; put them back where the plain
        # aggregation pack leaves them, so the two strategies stay interchangeable.
        result = packed.join(dim, on=root_keys, how="left")
        result = result.select(self._pack_column_order(result.collect_schema(), at_level))

        return self._match_frame_type(result, frame)

    def unpack(self, frame: FrameT, at_level: str = _MISSING_LEVEL, **_legacy: Any) -> FrameT:
        """
        Unpack nested list-of-struct columns until each row is one ``at_level``
        entity, mirroring :func:`explode` + :func:`unnest` per level.

        The exact inverse of :meth:`pack` at the same level.

        Only the **axis** leading to ``at_level`` — the root → ``at_level`` chain
        — is exploded. Sibling branches hanging off that chain stay packed as
        ``List[Struct]`` columns, replicated onto each emitted row. A flat frame
        can only carry one granularity, so exploding a second branch alongside
        the first would mean a cartesian product of two unrelated child sets;
        leaving it nested keeps the frame lossless and re-packable instead.
        Unpack to a level on the other branch to explode that one::

            packer.unpack(nested, "building")  # street axis; service stays packed
            packer.unpack(nested, "service")   # service axis; street stays packed

        Args:
            frame: The DataFrame or LazyFrame to unpack.
            at_level: The level each row should represent. Its axis is the chain
                that gets exploded.

        Returns:
            Unpacked frame with flattened columns, same type as input.

        Raises:
            KeyError: If the level is not found in the hierarchy.
            TypeError: If the pre-0.8.0 ``to_level`` keyword is passed.
        """
        _reject_legacy_level_kwarg("unpack", _legacy, at_level)

        lf = self._to_lazy(frame)
        schema = lf.collect_schema()

        for name in self.spec.axis_of(at_level):
            level = self._levels_meta[self.spec.index_of(name)]
            if level.path not in schema:
                continue
            lf, schema = self._explode_and_unnest(lf, level, schema)

        lf = self._drop_internal_columns(lf)
        return self._match_frame_type(lf, frame)

    def pack_streaming(
        self,
        source: pl.LazyFrame | pl.DataFrame | str | Path,
        at_level: str = _MISSING_LEVEL,
        *,
        partitions: int = 16,
        tmp_dir: str | Path | None = None,
        defer: bool = True,
        extra_columns: ExtraColumnsMode = "preserve",
        partition_strategy: PartitionStrategy = "balanced",
        **_legacy: Any,
    ) -> pl.LazyFrame:
        """
        Memory-bounded :meth:`pack` for datasets too large to pack in one shot.

        ``pack`` relies on a ``group_by`` that collects children into lists. The
        streaming engine has no native list-collecting aggregation, so that node
        falls back to the in-memory engine and peak memory scales with the full
        dataset. ``pack_streaming`` instead buckets the input by the **root-level
        key** (so every entity's rows stay together), packs each bucket
        independently while sinking the result to Parquet, and returns a single
        :class:`polars.LazyFrame` over the packed output. Peak memory is therefore
        bounded by one bucket rather than the whole dataset.

        Bucketing itself is a single streaming pass: the input is written once to
        a partitioned Parquet staging area (via ``pl.PartitionBy``) and each
        bucket is then read back exactly once. On Polars versions without
        partitioned sinks this degrades to one filtered pass per bucket, which is
        correct but re-reads the source ``partitions`` times.

        Note that a global ``sort`` cannot be used to group entities instead: sort
        is itself an in-memory fallback under the streaming engine, so it would
        cost exactly the memory this method exists to bound. ``"balanced"`` sorts
        only the per-entity row counts, which is a native streaming aggregation
        whose state is proportional to the *number of entities*, not rows.

        Args:
            source: Input at the finest granularity. May be a DataFrame, a
                LazyFrame, or a path/glob to Parquet file(s) (scanned lazily).
            at_level: The level each row of the result should represent.
            partitions: Target number of root-key buckets. More buckets means
                lower peak memory and more temporary files. Must be >= 1; ``1``
                skips bucketing entirely. Under ``"balanced"`` this is a target
                rather than an exact count — see ``partition_strategy``.
            tmp_dir: Directory for the intermediate per-bucket Parquet files.
                Defaults to a fresh :func:`tempfile.mkdtemp` directory; the caller
                owns cleanup. Transient staging files are written under a
                ``_stage`` subdirectory and removed automatically. Reusing a
                directory across calls with a different ``partitions`` count may
                leave stale files — prefer a fresh dir.
            defer: When ``True`` (default), the bucketing/sinking is wrapped in
                :func:`polars.defer` so nothing executes until the returned
                LazyFrame is collected, keeping the call chain lazy. Note that the
                packed result is materialized at the defer boundary. When
                ``False``, the buckets are packed and sunk eagerly and a
                :func:`polars.scan_parquet` handle is returned, so downstream
                operations stream straight from disk (safest when the packed
                result itself does not fit in memory).
            extra_columns: How to handle columns outside the hierarchy. See
                :meth:`pack`.
            partition_strategy: How root keys are assigned to buckets.
                - ``"balanced"`` (default): count rows per entity in one extra
                  streaming pass, then cut the key-ordered entities into
                  contiguous buckets of roughly ``total_rows / partitions`` rows.
                  Balances *rows*, which is what bounds peak memory, and keeps the
                  buckets in ascending key order so the result is sorted by root
                  key. The realised bucket count floats around ``partitions``: an
                  entity is never split, so a bucket closes early rather than
                  overflow, and more buckets than requested may be produced.
                - ``"hash"``: assign by ``hash(root_key) % partitions``. One pass
                  cheaper and gives exactly ``partitions`` buckets, but it
                  balances entities rather than rows, so an uneven entity-size
                  distribution produces an uneven peak. Top-level row order is
                  not guaranteed.

        Returns:
            A LazyFrame over the packed result. Under ``"balanced"`` the rows are
            ordered by root key; under ``"hash"`` top-level row order is not
            guaranteed. Child-list order follows the same rules as :meth:`pack`
            in both cases.

        Raises:
            ValueError: If ``partitions`` < 1 or ``partition_strategy`` is unknown.
            HierarchyValidationError: If the root level defines no id fields to
                partition on.
        """
        if partitions < 1:
            raise ValueError(f"partitions must be >= 1, got {partitions}.")

        _reject_legacy_level_kwarg("pack_streaming", _legacy, at_level)

        if partition_strategy not in ("balanced", "hash"):
            raise ValueError(
                f"Invalid partition_strategy: {partition_strategy!r}. "
                "Must be 'balanced' or 'hash'."
            )

        source_lf = (
            pl.scan_parquet(source) if isinstance(source, (str, Path)) else self._to_lazy(source)
        )

        root_keys = list(self._levels_meta[0].id_columns)
        if not root_keys:
            raise HierarchyValidationError(
                "pack_streaming requires the root level to define id_fields to " "partition on.",
                level=self._levels_meta[0].name,
            )

        # Materialize computed key columns so the root key exists for bucketing.
        prepared, _added, _schema = self._prepare_frame(source_lf)

        # Expected output schema is needed up-front for pl.defer; collecting a
        # LazyFrame schema is cheap (metadata only, no data movement).
        expected_schema = self.pack(
            source_lf, at_level, extra_columns=extra_columns
        ).collect_schema()

        out_dir = (
            Path(tmp_dir)
            if tmp_dir is not None
            else Path(tempfile.mkdtemp(prefix="nexpresso_pack_"))
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        sort_output = partition_strategy == "balanced"

        def _pack_bucket(bucket_source: pl.LazyFrame, index: int) -> None:
            packed = self.pack(bucket_source, at_level, extra_columns=extra_columns)
            if sort_output:
                # Buckets are contiguous ascending key ranges, so sorting within
                # each one makes the concatenation globally sorted. The bucket is
                # already materialized by the pack, so this stays bounded.
                sort_by = self._root_sort_exprs(packed.collect_schema())
                if sort_by:
                    packed = packed.sort(sort_by)
            packed.sink_parquet(out_dir / f"part_{index:05d}.parquet")

        def _bucketed(source_lf: pl.LazyFrame) -> tuple[pl.LazyFrame, int]:
            """Attach BUCKET_COLUMN to *source_lf*; return it and the bucket count."""
            if partition_strategy == "hash":
                expr = (pl.struct(root_keys).hash() % partitions).alias(BUCKET_COLUMN)
                return source_lf.with_columns(expr), partitions

            bucket_map = self._balanced_bucket_map(source_lf, root_keys, partitions)
            # Buckets are numbered contiguously from 0, so the height of the
            # distinct set is the count.
            n_buckets = bucket_map.select(pl.col(BUCKET_COLUMN).n_unique()).item()
            # ``nulls_equal`` matters: ``group_by`` treats null as its own group, so
            # the map has a row for it, but a normal join does not match null to
            # null. Without this those rows get a *null* bucket, which
            # ``PartitionBy`` writes as a ``__HIVE_DEFAULT_PARTITION__`` directory
            # that is not an integer bucket id.
            joined = source_lf.join(bucket_map.lazy(), on=root_keys, how="left", nulls_equal=True)
            return joined, int(n_buckets)

        def _pack_whole_source() -> list[Path]:
            """Single part covering everything — used when there is nothing to split.

            Also the empty-input path: a source with no rows yields no buckets at
            all, and ``scan_parquet`` needs at least one source, so a single
            (empty) part keeps the result well-formed with the right schema, the
            way eager ``pack`` already behaves.
            """
            _pack_bucket(prepared, 0)
            return [out_dir / "part_00000.parquet"]

        def _run_partitions() -> list[Path]:
            if partitions == 1:
                return _pack_whole_source()

            bucketed, n_buckets = _bucketed(prepared)
            if n_buckets == 0:
                return _pack_whole_source()

            # Single streaming pass writes every bucket to its own directory, so
            # the source is read once instead of once per bucket.
            stage_dir = out_dir / "_stage"
            bucketed.sink_parquet(
                pl.PartitionBy(stage_dir, key=BUCKET_COLUMN, include_key=False),
                mkdir=True,
            )
            try:
                # Scan whole bucket *directories*: a bucket may be spread over
                # several files, and all rows for a root key must be packed
                # together. Order numerically by the bucket value — the directory
                # names sort lexicographically ("...=10" before "...=2"), which
                # would scramble the key ranges.
                bucket_dirs = _sorted_bucket_dirs(stage_dir)
                for i, bucket_dir in enumerate(bucket_dirs):
                    _pack_bucket(pl.scan_parquet(bucket_dir / "**/*.parquet"), i)
            finally:
                shutil.rmtree(stage_dir, ignore_errors=True)

            if not bucket_dirs:
                return _pack_whole_source()

            return [out_dir / f"part_{i:05d}.parquet" for i in range(len(bucket_dirs))]

        if defer:

            def _materialize() -> pl.DataFrame:
                parts = _run_partitions()
                return pl.scan_parquet(parts).collect(engine="streaming")

            return pl.defer(_materialize, schema=expected_schema)

        return pl.scan_parquet(_run_partitions())

    def _balanced_bucket_map(
        self, source_lf: pl.LazyFrame, root_keys: list[str], partitions: int
    ) -> pl.DataFrame:
        """
        Assign each root key to a contiguous, roughly row-balanced bucket.

        Counting rows per entity is a *reducing* aggregation, which the streaming
        engine runs natively — its state is proportional to the number of
        entities, not the number of rows — so this stays memory-bounded where a
        global ``sort`` of the data would not.

        Entities are then walked in key order and packed greedily into buckets of
        about ``total_rows / partitions`` rows. An entity is never split, so a
        bucket closes early rather than overflow; the bucket count is therefore
        driven by the target size and may exceed ``partitions``. Capping it would
        push every leftover entity into the final bucket and defeat the point.

        Args:
            source_lf: The prepared source frame.
            root_keys: Root-level key columns.
            partitions: Target bucket count.

        Returns:
            A small DataFrame of ``root_keys`` plus a ``BUCKET_COLUMN``, one row
            per distinct root key, ordered by key.
        """
        counts = (
            source_lf.group_by(root_keys)
            .agg(pl.len().alias("__hier_n"))
            .collect(engine="streaming")
            .sort(root_keys)
        )
        if counts.is_empty():
            return counts.select(*root_keys).with_columns(
                pl.lit(0, dtype=pl.UInt32).alias(BUCKET_COLUMN)
            )

        total = int(counts["__hier_n"].sum())
        target = max(1, math.ceil(total / partitions))

        assignment: list[int] = []
        bucket = 0
        filled = 0
        for size in counts["__hier_n"]:
            if filled and filled + size > target:
                bucket += 1
                filled = 0
            assignment.append(bucket)
            filled += size

        return counts.select(*root_keys).with_columns(
            pl.Series(BUCKET_COLUMN, assignment, dtype=pl.UInt32)
        )

    def _root_sort_exprs(self, schema: pl.Schema) -> list[pl.Expr]:
        """
        Expressions that sort a packed frame by its root key.

        The root key is a set of top-level columns until the frame is packed all
        the way to the root, at which point it lives inside the root struct.

        Args:
            schema: Schema of the packed frame.

        Returns:
            Sort expressions, or an empty list if the root key is not reachable.
        """
        root = self._levels_meta[0]
        if root.id_columns and all(col in schema for col in root.id_columns):
            return [pl.col(col) for col in root.id_columns]

        if root.path in schema:
            dtype = schema[root.path]
            inner = dtype.inner if isinstance(dtype, pl.List) else dtype
            if isinstance(inner, pl.Struct):
                fields = {f.name for f in inner.fields}
                short = [col[len(root.prefix) :] for col in root.id_columns]
                if short and all(name in fields for name in short):
                    return [pl.col(root.path).struct.field(name) for name in short]

        return []

    def unpack_streaming(
        self,
        source: pl.LazyFrame | pl.DataFrame | str | Path,
        at_level: str = _MISSING_LEVEL,
        *,
        sink_path: str | Path | None = None,
        **_legacy: Any,
    ) -> pl.LazyFrame:
        """
        Streaming-friendly :meth:`unpack` returning a :class:`polars.LazyFrame`.

        ``unpack`` is ``explode`` + ``unnest``, which the streaming engine already
        runs with bounded memory; the only issue is that callers usually pass
        eager frames and get a materialized result. This helper accepts a Parquet
        path (scanned lazily) or a frame and keeps the pipeline lazy so it can be
        composed with downstream operations or sunk straight to disk.

        Args:
            source: Packed input. A DataFrame, LazyFrame, or path/glob to Parquet
                file(s) (scanned lazily).
            at_level: The level each row of the result should represent.
            sink_path: Optional Parquet path. When given, the unpacked result is
                streamed to it via ``sink_parquet`` and a fresh scan over that
                file is returned (true disk-to-disk, nothing materialized).

        Returns:
            A LazyFrame over the unpacked result.
        """
        _reject_legacy_level_kwarg("unpack_streaming", _legacy, at_level)

        source_lf = (
            pl.scan_parquet(source) if isinstance(source, (str, Path)) else self._to_lazy(source)
        )

        unpacked = self.unpack(source_lf, at_level)

        if sink_path is not None:
            unpacked.sink_parquet(sink_path)
            return pl.scan_parquet(sink_path)

        return unpacked

    def _own_level_columns(self, meta: LevelMetadata, schema: pl.Schema) -> list[str]:
        """
        Columns in ``schema`` that belong to ``meta`` itself.

        A column is owned by a level when it sits under the level's prefix but
        does **not** belong to any child branch — i.e. neither a child's packed
        column (``child.path``) nor anything under a child's prefix. Every
        branch is excluded, so a level with two children keeps only what is
        genuinely its own. Internal bookkeeping columns are excluded.
        """
        children = [self._levels_meta[self.spec.index_of(name)] for name in meta.children]
        own: list[str] = []
        for col in schema.keys():
            if col == ROW_ID_COLUMN or not meta.prefix or not col.startswith(meta.prefix):
                continue
            if any(col == child.path or col.startswith(child.prefix) for child in children):
                continue
            own.append(col)
        return own

    def split_levels(self, frame: FrameT) -> dict[str, FrameT]:
        """
        Split a packed frame into standalone tables—one per hierarchy level.

        Each emitted table is *level-local*: it contains the level's own
        columns (its id fields and attributes) plus the **key** columns of its
        ancestors, which act as foreign keys back to the coarser tables.
        Attributes belonging to a coarser level are **not** duplicated into the
        finer tables, and descendant columns are never included. This is the
        normalized ("third normal form"-ish) shape that :meth:`denormalize` and
        :meth:`build_from_tables` expect.

        For a ``country → city → street`` hierarchy the result is::

            country : country.code, country.name
            city    : country.code, country.city.id, country.city.population
            street  : country.code, country.city.id,
                      country.city.street.name, country.city.street.length

        Levels that are still flat in ``frame`` (i.e. coarser than the frame's
        current granularity, such as ``country`` in a frame packed to ``city``)
        are emitted too, deduplicated on their key columns, so no attribute is
        silently dropped.

        A branching hierarchy needs no special handling here: every level is its
        own table regardless of shape, so a ``city`` with both ``street`` and
        ``service`` children simply emits three tables, each keyed back to
        ``city``. This is the shape :class:`~nexpresso.HierarchyView` queries,
        and the one that represents a branching hierarchy without duplication.

        Args:
            frame: The packed DataFrame or LazyFrame to split.

        Returns:
            Dictionary mapping level names to their respective tables, ordered
            root → leaf. Each table has the same type as ``frame``.

        Note:
            The returned plans all branch off the same upstream pipeline. With a
            LazyFrame input, collect them together rather than one at a time so
            the shared work runs once::

                tables = packer.split_levels(lazy_packed)
                frames = dict(zip(tables, pl.collect_all(list(tables.values()))))

            Eager input already does this internally.

            On Polars >= 1.41, setting ``POLARS_ALLOW_NESTED_CSPE=1`` in the
            environment speeds this up a further 1.5-1.8x. Each level's plan is
            built on the previous level's, so the shared subplans are *nested*;
            the default common-subplan elimination only dedupes one level deep
            and re-runs the rest. See ``docs/concepts/lazy-and-streaming.md``.
        """
        lf, added_cols, schema = self._prepare_frame(frame)

        outputs: dict[str, pl.LazyFrame] = {}
        added = set(added_cols)

        # One working frame per level, unpacked to that level's own granularity.
        # Each level continues from its *parent's* frame, so sibling branches each
        # explode independently off the shared ancestor plan rather than one
        # branch inheriting the other's explode. In an unbranched chain this is
        # exactly the progressive unpack it replaces.
        frames: dict[str, tuple[pl.LazyFrame, pl.Schema]] = {}

        for level in self._levels_meta:
            base_lf, base_schema = frames[level.parent] if level.parent else (lf, schema)

            unpacked_here = level.path in base_schema
            if unpacked_here:
                current = self.unpack(base_lf, level.name)
                current_schema = current.collect_schema()
            else:
                current, current_schema = base_lf, base_schema
            frames[level.name] = (current, current_schema)

            if not unpacked_here and not any(
                col.startswith(level.prefix) for col in current_schema.keys()
            ):
                # Neither packed nor flat in this frame — nothing to emit.
                continue

            # Ancestor *keys* only (foreign keys), never ancestor attributes.
            ancestor_keys = [
                col for col in level.ancestor_keys if col in current_schema and col not in added
            ]
            own_cols = [
                col for col in self._own_level_columns(level, current_schema) if col not in added
            ]
            keep = ancestor_keys + own_cols
            if not keep:
                continue

            output_table = current.select(keep)

            # Drop rows that only exist because of a null/absent child branch.
            id_subset = [col for col in level.id_columns if col in keep]
            if level.children:
                null_guard = ancestor_keys + id_subset
            else:
                null_guard = [col for col in level.required_columns if col in keep]
            if null_guard:
                output_table = output_table.drop_nulls(subset=null_guard)

            if not self._is_at_granularity(level, current_schema):
                # Rows are at a finer granularity than this level, so the level's
                # own values repeat; collapse them to one row per entity.
                #
                # The surviving row order becomes this level's order inside its
                # parent's child list when the tables are denormalized again, so
                # under ``preserve_child_order`` the dedup has to be stable —
                # ``keep="any"`` without ``maintain_order`` is free to reorder,
                # and does so differently across Polars versions. Callers who
                # turned child order off keep the cheaper unordered form.
                key_subset = ancestor_keys + id_subset
                unique_kwargs: dict[str, Any] = (
                    {"keep": "first", "maintain_order": True}
                    if self.preserve_child_order
                    else {"keep": "any"}
                )
                output_table = (
                    output_table.unique(subset=key_subset, **unique_kwargs)
                    if key_subset
                    else output_table.unique(**unique_kwargs)
                )

            outputs[level.name] = output_table

        if isinstance(frame, pl.DataFrame):
            # Every level's plan branches off the same progressive-unpack chain, so
            # collecting them one by one re-executes that shared work per level.
            # ``collect_all`` runs them as a single graph instead.
            names = list(outputs)
            collected = pl.collect_all([outputs[name] for name in names])
            return dict(zip(names, collected))  # type: ignore[return-value]
        return outputs  # type: ignore[return-value]

    def normalize(
        self, frame: FrameT, *, at_level: str | None = None, **_legacy: Any
    ) -> dict[str, FrameT]:
        """
        Convenience wrapper that packs to ``at_level`` and splits into normalized
        per-level tables.

        The emitted tables have :meth:`split_levels`' level-local shape: each
        level's own columns plus its ancestors' key columns as foreign keys, with
        no coarser attributes duplicated into the finer tables.

        Args:
            frame: The DataFrame or LazyFrame to normalize.
            at_level: Level to pack to first (defaults to the root). Only affects
                how much work the pack does; the emitted tables are the same
                either way.

        Returns:
            Dictionary mapping level names to their respective normalized tables,
            ordered root → leaf.

        Raises:
            TypeError: If the pre-0.8.0 ``root_level`` keyword is passed.
        """
        _reject_legacy_level_kwarg("normalize", _legacy)

        target = at_level or self._levels_meta[0].name
        packed = self.pack(frame, target)
        return self.split_levels(packed)

    def denormalize(
        self,
        tables: Mapping[str, pl.LazyFrame | pl.DataFrame],
        *,
        at_level: str | None = None,
        **_legacy: Any,
    ) -> pl.LazyFrame | pl.DataFrame:
        """
        Reconstruct nested columns by progressively attaching child tables to
        their parents. The input should be a mapping produced by
        :meth:`normalize`.

        This is a true inverse of :meth:`normalize` — for every level ``L``::

            packer.denormalize(
                packer.normalize(df, at_level=L), at_level=L
            ) == packer.pack(df, L)

        Child levels are attached leaf → root as list-of-struct columns, and when
        ``at_level`` is finer than the root the ancestors' attribute columns are
        joined back on (the upward pass only ever attaches descendants). Like
        :meth:`pack`, the result has one row per ``at_level`` entity with that
        level's own columns flat.

        Args:
            tables: Mapping of level name to table, as produced by
                :meth:`normalize` / :meth:`split_levels`.
            at_level: The level each row should represent (defaults to root).

        Returns:
            Denormalized frame with nested structures. Matches the type of the
            table supplied for ``at_level``.

        Raises:
            HierarchyValidationError: If ``tables`` is empty, the root table is
                absent, a table needed to reach ``at_level`` is missing, or a
                supplied table lacks its own key columns.
            TypeError: If the pre-0.8.0 ``target_level`` keyword is passed.
        """
        _reject_legacy_level_kwarg("denormalize", _legacy)

        if not tables:
            raise HierarchyValidationError(
                "Expected at least one table to denormalize.",
                details={"tables_provided": 0},
            )

        target_name = at_level or self._levels_meta[0].name
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

        # Every level's table must carry its own key columns — they are what the
        # upward pass joins on. Check up front so a missing key surfaces as a
        # named error instead of a bare Polars ColumnNotFoundError from deep
        # inside a join plan.
        self._validate_table_keys(prepared_tables)

        # Snapshot the per-level tables before the upward pass mutates them; the
        # ancestor attributes are re-attached from these when the target level is
        # finer than the root (see below).
        source_tables = dict(prepared_tables)

        # Each level as it stands with its branches attached but before its own
        # fold — see the snapshot inside the loop.
        branch_tables: dict[str, pl.LazyFrame] = {}

        # Propagate child structures upward, children before parents. A parent
        # with several branches accumulates one List[Struct] column per branch
        # before its own fold, since each iteration joins a single child on.
        for level in reversed(self._levels_meta[1:]):
            level_idx = level.index
            assert level.parent is not None  # non-root levels always have one
            parent_name = level.parent

            child_lf = prepared_tables.get(level.name)
            if child_lf is None:
                # A level on the target's own axis is required; a level off it
                # belongs to a branch the caller simply did not supply.
                on_target_axis = level.name == target_name or self.spec.is_ancestor_of(
                    level.name, target_name
                )
                if on_target_axis:
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

            # Children are folded first, so ``child_lf`` already carries this
            # level's own branches as packed columns. Snapshot it before it is
            # replaced by its packed form: a target below a branch point needs
            # its ancestors' *other* branches, and this is the only point at
            # which they exist as top-level columns.
            branch_tables[level.name] = child_lf

            child_schema = child_lf.collect_schema()
            # Never validate here: the uniformity check runs an eager ``collect()``,
            # which would silently break the lazy contract of ``denormalize``.
            # Per-level tables are normalized by construction anyway.
            child_packed, _ = self._pack_single_level(
                child_lf, level_idx, child_schema, validate=False
            )
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

        # The target's rows must be at *its own* granularity, so take the form it
        # had before its own fold: branches attached, own columns still flat. The
        # root is never folded by the loop, so its ``prepared_tables`` entry is
        # already in that form.
        branch_tables.setdefault(root_name, prepared_tables[root_name])
        result = branch_tables.get(target_name)
        if result is None:
            raise HierarchyValidationError(
                f"Missing table for level '{target_name}'.",
                level=target_name,
            )

        # The upward pass only ever attaches *descendants*, so a target below the
        # root carries just its own columns plus its ancestors' key columns. Join
        # the ancestors' attributes back on so the result matches
        # ``pack(flat, at_level)``: tables from :meth:`split_levels` are
        # normalized and no longer duplicate coarser attributes into finer tables.
        if target_idx > 0:
            result = self._attach_ancestor_attributes(
                result, target_name, source_tables, branch_tables
            )

        added_aliases = alias_map.get(target_name, ())
        if added_aliases:
            result = result.drop(*added_aliases, strict=False)

        # The joins above append ancestor attributes at the end, and the upward
        # pass appends each branch as it is folded. Restore the column order
        # ``pack`` produces so the two are equal, not merely equivalent.
        result = result.select(self._pack_column_order(result.collect_schema(), target_name))

        result = self._drop_internal_columns(result)

        # Match output type to the target table's input type
        target_table = tables[target_name]
        if isinstance(target_table, pl.DataFrame):
            return result.collect()
        return result

    def _pack_column_order(self, schema: pl.Schema, at_level: str) -> list[str]:
        """
        Column order that :meth:`pack` produces for ``at_level`` granularity.

        The canonical order is a depth-first walk of the tree: each level's own
        columns, then its children in declaration order. That is the order a flat
        frame naturally holds, so :meth:`unpack` already produces it, and pinning
        :meth:`pack` and :meth:`denormalize` to the same walk means all three
        agree column-for-column at a given granularity.

        It deliberately does *not* mirror ``group_by``'s habit of emitting its
        keys first. Hoisting them would make a packed frame's layout depend on
        which level happened to fold last, and would leave ``pack(df, L)`` and
        ``unpack(nested, L)`` — two routes to one granularity — disagreeing on
        column order for no reason a caller could predict.

        Args:
            schema: Schema of the assembled frame.
            at_level: The level each row represents.

        Returns:
            Column names in canonical order. Any column that does not belong to
            the hierarchy keeps its relative position at the end.
        """
        target = self._levels_meta[self.spec.index_of(at_level)]
        present = list(schema.names())
        remaining = set(present)

        ordered: list[str] = []

        def take(col: str) -> None:
            if col in remaining:
                remaining.discard(col)
                ordered.append(col)

        # Descending the target's axis recurses; a child off the axis is a packed
        # column and is emitted in place.
        axis = self.spec.axis_of(target.name)
        axis_position = {name: i for i, name in enumerate(axis)}

        def walk(name: str) -> None:
            meta = self._levels_meta[self.spec.index_of(name)]
            for col in self._own_level_columns(meta, schema):
                take(col)
            for child in meta.children:
                if axis_position.get(child) == axis_position[name] + 1:
                    walk(child)
                else:
                    take(self._levels_meta[self.spec.index_of(child)].path)

        walk(axis[0])

        # Anything left (extra non-hierarchy columns) keeps its original order.
        ordered.extend(col for col in present if col in remaining)
        return ordered

    def _validate_table_keys(self, prepared_tables: Mapping[str, pl.LazyFrame]) -> None:
        """
        Check that each supplied level table carries that level's own key columns.

        Those keys are what :meth:`denormalize` joins children to parents on, so a
        missing one otherwise fails much later as an opaque Polars
        ``ColumnNotFoundError`` inside a join plan. Levels with no table are
        skipped — the caller may legitimately omit levels finer than the target.

        Args:
            prepared_tables: Per-level tables, already key/alias-prepared.

        Raises:
            HierarchyValidationError: If a supplied table is missing key columns.
        """
        for meta in self._levels_meta:
            lf = prepared_tables.get(meta.name)
            if lf is None or not meta.id_columns:
                continue

            available = set(lf.collect_schema().names())
            missing = [col for col in meta.id_columns if col not in available]
            if missing:
                raise HierarchyValidationError(
                    f"Table for level '{meta.name}' is missing its key column(s) "
                    f"{missing}. Key columns identify each row at a level and are "
                    "what child tables join to. Add them to the table, or rename an "
                    "existing column to match "
                    f"(e.g. df.with_columns(pl.col(<source>).alias({missing[0]!r}))).",
                    level=meta.name,
                    details={"missing_columns": missing, "available": sorted(available)},
                )

    def _attach_ancestor_attributes(
        self,
        result: pl.LazyFrame,
        at_level: str,
        source_tables: Mapping[str, pl.LazyFrame],
        nested_tables: Mapping[str, pl.LazyFrame],
    ) -> pl.LazyFrame:
        """
        Join each ancestor level's own attributes and off-axis branches onto ``result``.

        ``result`` is at ``at_level`` granularity and already carries every
        ancestor *key* column, which is what the joins key on.

        Two kinds of column come down. An ancestor's plain **attributes** are
        read from the caller's tables. An ancestor's **off-axis branches** —
        children that are not on the target's axis, such as ``service`` when
        the target is ``building`` — are read from the post-upward-pass tables,
        where they already exist as packed ``List[Struct]`` columns. Without
        them the result would silently lose a whole branch that :meth:`pack`
        carries along.

        Args:
            result: Frame at the target level's granularity.
            at_level: The level each row represents.
            source_tables: Per-level tables as provided by the caller.
            nested_tables: Per-level tables after children were folded into
                their parents.

        Returns:
            ``result`` widened with the ancestors' attribute and branch columns.
        """
        result_schema = result.collect_schema()

        axis = set(self.spec.axis_of(at_level))

        for name in self.spec.ancestors_of(at_level):
            anc = self._levels_meta[self.spec.index_of(name)]
            anc_lf = source_tables.get(anc.name)
            if anc_lf is None:
                continue

            anc_schema = anc_lf.collect_schema()
            join_keys = [
                key
                for key in (*anc.ancestor_keys, *anc.id_columns)
                if key in anc_schema and key in result_schema
            ]
            if not join_keys:
                continue

            attrs = [
                col
                for col in self._own_level_columns(anc, anc_schema)
                if col not in join_keys and col not in result_schema
            ]
            if attrs:
                result = result.join(anc_lf.select([*join_keys, *attrs]), on=join_keys, how="left")
                result_schema = result.collect_schema()

            branch_lf = nested_tables.get(anc.name)
            if branch_lf is None:
                continue
            branch_schema = branch_lf.collect_schema()
            branches = [
                self._levels_meta[self.spec.index_of(child)].path
                for child in anc.children
                if child not in axis
            ]
            branches = [
                col for col in branches if col in branch_schema and col not in result_schema
            ]
            if not branches:
                continue
            if not all(key in branch_schema for key in join_keys):
                continue

            result = result.join(
                branch_lf.select([*join_keys, *branches]), on=join_keys, how="left"
            )
            result_schema = result.collect_schema()

        return result

    def build_from_tables(
        self,
        tables: Mapping[str, pl.LazyFrame | pl.DataFrame],
        *,
        at_level: str | None = None,
        join_type: Literal["left", "inner"] = "left",
        **_legacy: Any,
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
            at_level: The level each row of the result should represent
                (default: root level).
            join_type: How to join child tables to parents ("left" or "inner").

        Returns:
            Nested frame at ``at_level`` granularity.

        Raises:
            HierarchyValidationError: If required tables or columns are missing.
            TypeError: If the pre-0.8.0 ``target_level`` keyword is passed.

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
        _reject_legacy_level_kwarg("build_from_tables", _legacy)

        if not tables:
            raise HierarchyValidationError(
                "Expected at least one table to build from.",
                details={"tables_provided": 0},
            )

        target_name = at_level or self._levels_meta[0].name

        # Check that we have all required levels
        # The target's own axis must be complete; branches off it are optional and
        # simply do not appear in the result when their table is not supplied.
        for name in self.spec.axis_of(target_name):
            if name not in tables:
                raise HierarchyValidationError(
                    f"Missing table for level '{name}'.",
                    level=name,
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

        # Join tables children-first so a parent has every branch attached before
        # it is itself joined onto its own parent.
        for level in reversed(self._levels_meta[1:]):
            level_idx = level.index
            level_spec = self.spec.levels[level_idx]

            if level.name not in prepared_tables:
                continue

            assert level.parent is not None  # non-root levels always have one
            parent_name = level.parent
            parent_meta = self._levels_meta[self.spec.index_of(parent_name)]

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
            # Resolve the schema once: called inline below it would re-resolve the
            # plan for every entry in the mapping.
            mapping_schema = lf.collect_schema()
            # Rename columns first according to mapping
            rename_exprs = [
                pl.col(raw_col).alias(target_col)
                for raw_col, target_col in column_mapping.items()
                if raw_col in mapping_schema
            ]
            if rename_exprs:
                # Select all columns, applying renames
                all_cols = mapping_schema.keys()
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

        # Count nulls for every key column in a single pass. Collecting per column
        # would re-execute the whole upstream plan once per key.
        checks: list[tuple[LevelMetadata, str, str]] = []
        exprs: list[pl.Expr] = []
        for meta in levels_to_check:
            for key_col in meta.id_columns:
                if key_col not in schema:
                    continue
                alias = f"__null_count_{len(checks)}"
                checks.append((meta, key_col, alias))
                exprs.append(pl.col(key_col).is_null().sum().alias(alias))

        if not exprs:
            return errors

        counts = lf.select(exprs).collect().row(0, named=True)

        for meta, key_col, alias in checks:
            null_count = counts[alias]
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

    def validate_schema(
        self,
        schema_or_frame: SchemaInput,
        *,
        expected_level: str | None = None,
    ) -> SchemaValidationResult:
        """
        Validate whether this packer's hierarchy spec is compatible with a schema.

        Performs structural validation (column existence, type compatibility)
        without inspecting actual data values.  For data validation (null keys,
        uniformity), use :meth:`validate`.

        Args:
            schema_or_frame: A ``pl.Schema``, ``pl.DataFrame``, or ``pl.LazyFrame``
                to validate against.
            expected_level: If provided, also verify that the schema represents
                data at this specific packing level.  If ``None``, the level is
                inferred automatically.

        Returns:
            A :class:`SchemaValidationResult` with detailed compatibility info.

        Examples:
            >>> result = packer.validate_schema(df)
            >>> if not result.is_compatible:
            ...     for err in result.errors:
            ...         print(err)
        """
        schema = self._extract_schema(schema_or_frame)
        errors: list[str] = []
        warnings: list[str] = []
        present: list[str] = []
        missing: list[str] = []

        # Infer current level
        inferred_level: str | None = None
        try:
            inferred_level = self.infer_current_level(schema)
        except ValueError:
            warnings.append(
                "Could not infer current packing level from schema. "
                f"Schema columns: {list(schema.keys())}"
            )

        # Check expected level
        if expected_level is not None and inferred_level is not None:
            if expected_level != inferred_level:
                errors.append(
                    f"Expected data at level '{expected_level}' but inferred "
                    f"level is '{inferred_level}'."
                )

        # Validate each level
        for meta in self._levels_meta:
            level_found = False

            # --- Check flat columns ---
            flat_id_found: list[str] = []
            flat_id_missing: list[str] = []
            for id_col in meta.id_columns:
                if id_col in schema:
                    flat_id_found.append(id_col)
                    # Type check: should be a scalar, not nested
                    col_dtype = schema[id_col]
                    if isinstance(col_dtype, (pl.List, pl.Struct, pl.Array)):
                        errors.append(
                            f"[Level: {meta.name}] Key column '{id_col}' has "
                            f"type {col_dtype} but expected a scalar type."
                        )
                else:
                    flat_id_missing.append(id_col)

            if flat_id_found:
                level_found = True
                if flat_id_missing:
                    warnings.append(
                        f"[Level: {meta.name}] Some key columns missing from flat schema: "
                        f"{flat_id_missing}. Found: {flat_id_found}."
                    )

            # --- Check packed column ---
            if meta.path in schema:
                col_dtype = schema[meta.path]
                inner = col_dtype.inner if isinstance(col_dtype, pl.List) else col_dtype
                if isinstance(inner, pl.Struct) and inner.fields:
                    level_found = True
                    # Check struct contains expected id fields (short names)
                    struct_field_names = {f.name for f in inner.fields}
                    short_ids = [col[len(meta.prefix) :] for col in meta.id_columns]
                    missing_ids = [sid for sid in short_ids if sid not in struct_field_names]
                    if missing_ids:
                        errors.append(
                            f"[Level: {meta.name}] Packed column '{meta.path}' "
                            f"is missing expected key fields: {missing_ids}. "
                            f"Struct fields: {sorted(struct_field_names)}."
                        )
                elif not flat_id_found:
                    # Path exists but is not a Struct — unexpected type
                    warnings.append(
                        f"[Level: {meta.name}] Column '{meta.path}' exists but "
                        f"has type {col_dtype}, expected List[Struct] or Struct."
                    )

            if level_found:
                present.append(meta.name)
            else:
                missing.append(meta.name)

        # If no levels found at all, that's definitely incompatible
        if not present:
            errors.append(
                "No hierarchy levels found in schema. "
                f"Expected columns with prefix patterns like: "
                f"{[m.prefix for m in self._levels_meta[:3]]}..."
            )

        return SchemaValidationResult(
            is_compatible=len(errors) == 0,
            inferred_level=inferred_level,
            present_levels=present,
            missing_levels=missing,
            errors=errors,
            warnings=warnings,
        )

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

    # Mapping from aggregation name to a callable that transforms a List[T] expr.
    _LIST_AGGREGATIONS: dict[PromoteAggregation, Callable[[pl.Expr], pl.Expr]] = {
        "list": lambda e: e,
        "set": lambda e: e.list.eval(pl.element().drop_nulls().unique()),
        "sum": lambda e: e.list.sum(),
        "mean": lambda e: e.list.mean(),
        "min": lambda e: e.list.min(),
        "max": lambda e: e.list.max(),
        "first": lambda e: e.list.first(),
        "last": lambda e: e.list.last(),
        "count": lambda e: e.list.len(),
        "single": lambda e: e.list.eval(pl.element().drop_nulls().unique()).list.first(),
    }

    #: Group-by counterparts of :data:`_LIST_AGGREGATIONS`, for callers that
    #: aggregate a child *table* instead of a nested list column (notably
    #: :class:`~nexpresso.HierarchyView.promote`). Kept adjacent so the two
    #: stay in step — they must agree on null handling, or the packed and
    #: normalized paths quietly disagree.
    GROUP_AGGREGATIONS: dict[PromoteAggregation, Callable[[pl.Expr], pl.Expr]] = {
        "list": lambda e: e,
        "set": lambda e: e.drop_nulls().unique(),
        "sum": lambda e: e.sum(),
        "mean": lambda e: e.mean(),
        "min": lambda e: e.min(),
        "max": lambda e: e.max(),
        "first": lambda e: e.first(),
        "last": lambda e: e.last(),
        "count": lambda e: e.len(),
        "single": lambda e: e.drop_nulls().unique().first(),
    }

    # Like _LIST_AGGREGATIONS but used at intermediate levels when traversing more
    # than one hop.  The only difference is "count": at intermediate hops we sum
    # the per-child counts rather than re-counting the outer list length.
    _INTERMEDIATE_AGGREGATIONS: dict[PromoteAggregation, Callable[[pl.Expr], pl.Expr]] = {
        "list": lambda e: e,
        "set": lambda e: e.list.eval(pl.element().drop_nulls().unique()),
        "sum": lambda e: e.list.sum(),
        "mean": lambda e: e.list.mean(),
        "min": lambda e: e.list.min(),
        "max": lambda e: e.list.max(),
        "first": lambda e: e.list.first(),
        "last": lambda e: e.list.last(),
        "count": lambda e: e.list.sum(),  # sum inner counts rather than re-counting
        "single": lambda e: e.list.eval(pl.element().drop_nulls().unique()).list.first(),
    }

    def attribute_expr(
        self,
        attribute: str,
        from_level: str,
        to_level: str,
        agg: PromoteAggregation = "list",
    ) -> pl.Expr:
        """
        Return a Polars expression that computes an aggregated attribute on a
        frame already packed at ``to_level`` granularity.

        The expression can be passed directly to standard Polars operations
        (``filter``, ``with_columns``, ``sort``, arithmetic, …) so the full
        Polars expression algebra is available without any bespoke wrappers::

            packed = packer.pack(flat_df, "city")
            expr   = packer.attribute_expr("id", "city", "country", "count")

            packed.filter(expr > 10)                      # filter
            packed.with_columns(expr.alias("city_count")) # annotate
            packed.sort(expr, descending=True)            # sort
            (expr / packer.attribute_expr("revenue", "city", "country", "sum"))
                                                          # arithmetic

        **Same-level (trivial) case** — when ``from_level == to_level`` the
        attribute is already a scalar column at that granularity; ``agg`` is
        not applied and the expression is simply ``pl.col(attribute)``.

        **Cross-level** — navigates the nested list-of-struct structure produced
        by :meth:`pack`, applying the aggregation at each level.  For depths
        greater than one hop the aggregation cascades: ``"count"`` sums inner
        counts correctly; ``"mean"`` gives a mean-of-means approximation for
        unequal group sizes.

        Args:
            attribute: Unqualified field name at ``from_level``
                (e.g. ``"population"``).
            from_level: Level where the attribute lives.  Must be ``to_level``
                or a descendant.
            to_level: Level at which the expression is evaluated.  The frame
                passed to Polars must be packed at this granularity.
            agg: How to aggregate when ``from_level != to_level``:
                ``"list"`` | ``"set"`` | ``"sum"`` | ``"mean"`` | ``"min"`` |
                ``"max"`` | ``"first"`` | ``"last"`` | ``"count"`` | ``"single"``.

        Returns:
            A ``pl.Expr`` ready for use with any Polars DataFrame / LazyFrame
            operation.

        Raises:
            KeyError: If either level is not found.
            ValueError: If ``from_level`` is neither ``to_level`` nor one of its
                descendants — including a level on a sibling branch, which is
                not reachable by descending through ``to_level``'s nested
                columns.

        Examples:
            >>> packed = packer.pack(flat_df, "city")
            >>> expr = packer.attribute_expr("id", "city", "country", "count")
            >>> packed.filter(expr > 10)
        """
        to_meta = self._levels_meta[self.spec.index_of(to_level)]

        # Same-level: the attribute is already a direct scalar column.
        if from_level == to_level:
            return pl.col(f"{to_meta.prefix}{self._escape_field(attribute)}")

        if not self.spec.is_ancestor_of(to_level, from_level):
            reason = (
                "it is coarser"
                if self.spec.is_ancestor_of(from_level, to_level)
                else "it sits on a different branch"
            )
            raise ValueError(
                f"from_level '{from_level}' must be '{to_level}' or one of its "
                f"descendants, but {reason}. Attributes can only be derived by "
                f"descending into '{to_level}'; its descendants are "
                f"{self.spec.descendants_of(to_level)}."
            )

        # Cross-level: build the expression by navigating the nested structure
        # from the innermost level (from_level) outward to to_level. The route is
        # the segment of from_level's axis below to_level — unique even when the
        # hierarchy branches, since a level has only one path back to the root.
        #
        # traverse[0] = immediate child of to_level (outermost nested column)
        # traverse[-1] = from_level (innermost)
        axis = self.spec.axis_of(from_level)
        traverse = [self.spec.index_of(name) for name in axis[axis.index(to_level) + 1 :]]
        n_hops = len(traverse)

        final_agg = self._LIST_AGGREGATIONS[agg]
        intermediate_agg = self._INTERMEDIATE_AGGREGATIONS[agg]

        # Innermost expression: extract the attribute from a from_level element.
        inner: pl.Expr = pl.element().struct.field(attribute)

        # Wrap each intermediate hop (from from_level outward, excluding the
        # outermost column access which is handled below).
        for hop in range(n_hops - 1, 0, -1):
            parent_meta = self._levels_meta[traverse[hop - 1]]
            child_meta = self._levels_meta[traverse[hop]]
            field_in_parent = child_meta.path[len(parent_meta.prefix) :]

            # The innermost hop uses final_agg; all others use intermediate_agg.
            agg_fn = final_agg if (hop == n_hops - 1) else intermediate_agg
            inner = agg_fn(pl.element().struct.field(field_in_parent).list.eval(inner))

        # Outermost: reference the actual column rather than pl.element().
        imm_child_meta = self._levels_meta[traverse[0]]
        outer_agg = final_agg if n_hops == 1 else intermediate_agg
        return outer_agg(pl.col(imm_child_meta.path).list.eval(inner))

    def enrich(
        self,
        frame: FrameT,
        *specs: LevelAttribute,
        at_level: str,
    ) -> FrameT:
        """
        Add multiple cross-level attribute columns to a packed frame in one call.

        Each :class:`LevelAttribute` spec is converted to an expression via
        :meth:`attribute_expr` and applied together with ``with_columns``.

        The frame must already be packed at ``at_level`` granularity (i.e.
        produced by :meth:`pack` or :meth:`build_from_tables`).

        Args:
            frame: Packed frame at ``at_level`` granularity.
            *specs: One or more :class:`LevelAttribute` specs describing the
                attributes to derive and their aggregation strategies.
            at_level: The granularity level of ``frame``.

        Returns:
            Frame with new attribute columns appended, preserving input type.

        Raises:
            KeyError: If any level name is not found in the hierarchy.
            ValueError: If any ``from_level`` is coarser than ``at_level``.

        Examples:
            >>> from nexpresso import LevelAttribute
            >>> packed_country = packer.pack(flat_df, "city")
            >>> result = packer.enrich(
            ...     packed_country,
            ...     LevelAttribute("id", "city", "count", alias="city_count"),
            ...     LevelAttribute("revenue", "city", "sum", alias="total_revenue"),
            ...     at_level="country",
            ... )
        """
        to_meta = self._levels_meta[self.spec.index_of(at_level)]
        exprs = []
        for spec in specs:
            expr = self.attribute_expr(spec.attribute, spec.from_level, at_level, spec.agg)
            col_name = f"{to_meta.prefix}{self._escape_field(spec.alias or spec.attribute)}"
            exprs.append(expr.alias(col_name))
        lf = self._to_lazy(frame).with_columns(exprs)
        return self._match_frame_type(lf, frame)

    def any_child_satisfies(
        self,
        frame: FrameT,
        *,
        from_level: str,
        to_level: str,
        condition: pl.Expr,
    ) -> FrameT:
        """
        Filter a packed frame to rows where **at least one** child satisfies a
        condition.

        Entities with no children are filtered out — nothing can satisfy the
        condition. This holds for both an empty and a null child list, and is
        the mirror image of :meth:`all_children_satisfy`, where both cases pass.

        The frame must be packed at ``to_level`` granularity so that
        ``from_level`` data is accessible as a nested list-of-struct column.
        ``from_level`` must be the immediate child of ``to_level``.

        The ``condition`` expression should be written using ``pl.element()``
        to refer to individual child struct elements::

            packer.any_child_satisfies(
                packed_country,
                from_level="city",
                to_level="country",
                condition=pl.element().struct.field("population") > 1_000_000,
            )

        Args:
            frame: Packed frame at ``to_level`` granularity.
            from_level: Immediate child level whose elements are tested.
            to_level: Parent level; one row per entity in the result.
            condition: Boolean expression evaluated per child element using
                ``pl.element()``.

        Returns:
            Filtered frame preserving input type (DataFrame / LazyFrame).

        Raises:
            KeyError: If either level is not found.
            ValueError: If ``from_level`` is not the immediate child of
                ``to_level``.

        Examples:
            >>> packed = packer.pack(flat_df, "city")
            >>> result = packer.any_child_satisfies(
            ...     packed,
            ...     from_level="city",
            ...     to_level="country",
            ...     condition=pl.element().struct.field("population") > 1_000_000,
            ... )
        """
        self._require_immediate_child(from_level, to_level, "existential predicates")
        from_meta = self._levels_meta[self.spec.index_of(from_level)]
        mask = pl.col(from_meta.path).list.eval(condition.cast(pl.UInt8)).list.sum() > 0
        lf = self._to_lazy(frame).filter(mask)
        return self._match_frame_type(lf, frame)

    def all_children_satisfy(
        self,
        frame: FrameT,
        *,
        from_level: str,
        to_level: str,
        condition: pl.Expr,
    ) -> FrameT:
        """
        Filter a packed frame to rows where **every** child satisfies a
        condition.

        Entities with no children pass the filter (vacuous truth). That covers
        both an empty child list and a **null** child list — the latter is what
        a ``left`` join produces for a parent with no matching children.

        A child whose ``condition`` evaluates to *null* (typically a null
        attribute) counts as **not** satisfying it, so the entity is filtered
        out. Unknown is not treated as true; guard the condition explicitly
        (e.g. ``.fill_null(True)``) if you want the opposite.

        The frame must be packed at ``to_level`` granularity so that
        ``from_level`` data is accessible as a nested list-of-struct column.
        ``from_level`` must be the immediate child of ``to_level``.

        The ``condition`` expression should be written using ``pl.element()``
        to refer to individual child struct elements::

            packer.all_children_satisfy(
                packed_country,
                from_level="city",
                to_level="country",
                condition=pl.element().struct.field("population") > 10_000,
            )

        Args:
            frame: Packed frame at ``to_level`` granularity.
            from_level: Immediate child level whose elements are tested.
            to_level: Parent level; one row per entity in the result.
            condition: Boolean expression evaluated per child element using
                ``pl.element()``.

        Returns:
            Filtered frame preserving input type (DataFrame / LazyFrame).

        Raises:
            KeyError: If either level is not found.
            ValueError: If ``from_level`` is not the immediate child of
                ``to_level``.

        Examples:
            >>> packed = packer.pack(flat_df, "city")
            >>> result = packer.all_children_satisfy(
            ...     packed,
            ...     from_level="city",
            ...     to_level="country",
            ...     condition=pl.element().struct.field("population") > 10_000,
            ... )
        """
        self._require_immediate_child(from_level, to_level, "existential predicates")
        from_meta = self._levels_meta[self.spec.index_of(from_level)]
        child_col = pl.col(from_meta.path)
        evaluated = child_col.list.eval(condition.cast(pl.UInt8))
        # A null child list yields null on both sides, and `filter` drops null
        # masks — so vacuous truth needs an explicit branch. An *empty* list
        # already passes (0 == 0). `sum()` skips nulls, so a null condition
        # result makes the count fall short and the entity is filtered out.
        mask = child_col.is_null() | (evaluated.list.sum() == child_col.list.len())
        lf = self._to_lazy(frame).filter(mask)
        return self._match_frame_type(lf, frame)

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

        Examples:
            >>> result = packer.promote_attribute(
            ...     flat_df, "population",
            ...     from_level="city", to_level="country", agg="sum",
            ... )
        """
        self._require_immediate_child(from_level, to_level, "attribute promotion")

        from_meta = self._levels_meta[self.spec.index_of(from_level)]
        to_meta = self._levels_meta[self.spec.index_of(to_level)]

        # Pack to the target granularity, which leaves from_level as a
        # list-of-struct column on each to_level row.
        packed_lf = self._to_lazy(self.pack(frame, to_level))

        # Validate the attribute exists inside the nested struct.
        self._validate_list_struct_field(
            packed_lf.collect_schema(), from_meta.path, attribute, from_level
        )

        # Delegate expression generation to attribute_expr, then alias.
        expr = self.attribute_expr(attribute, from_level, to_level, agg)
        out_col = f"{to_meta.prefix}{self._escape_field(alias or attribute)}"

        result = packed_lf.with_columns(expr.alias(out_col))
        return self._match_frame_type(result, frame)

    @staticmethod
    def _validate_list_struct_field(
        schema: pl.Schema, list_col: str, attribute: str, level_name: str
    ) -> None:
        """Raise ``ValueError`` if *attribute* is not a field of the struct inside *list_col*."""
        if list_col not in schema:
            raise ValueError(
                f"Expected packed column '{list_col}' not found in schema. "
                f"Available columns: {list(schema.keys())}"
            )
        dtype = schema[list_col]
        inner = dtype.inner if isinstance(dtype, pl.List) else dtype
        if not isinstance(inner, pl.Struct):
            raise ValueError(f"Expected struct inside list column '{list_col}', got {inner}.")
        field_names = [f.name for f in inner.fields]
        if attribute not in field_names:
            raise ValueError(
                f"Attribute '{attribute}' not found at level '{level_name}'. "
                f"Available fields: {field_names}"
            )

    # ------------------------------------------------------------------
    # Separator Escaping
    # ------------------------------------------------------------------
    def escape_field(self, name: str) -> str:
        """
        Escape separator characters in a field name.

        Public because collaborators (notably
        :class:`~nexpresso.HierarchyView`) must build column paths that agree
        with the ones this packer emits.

        Args:
            name: The field name to escape.

        Returns:
            The escaped field name.
        """
        return self._escape_field(name)

    def split_path(self, path: str) -> list[str]:
        """
        Split a column path into components, respecting escaped separators.

        Public counterpart of :meth:`_split_path`; see :meth:`escape_field` for
        why. A field literally named ``net.sales`` round-trips as a single
        component rather than splitting into two.

        Args:
            path: The path to split.

        Returns:
            List of path components.
        """
        return self._split_path(path)

    def join_path(self, components: Sequence[str]) -> str:
        """
        Join path components, escaping separators inside each one.

        Public counterpart of :meth:`_join_path`; see :meth:`escape_field`.

        Args:
            components: The path components to join.

        Returns:
            The joined, escaped path.
        """
        return self._join_path(components)

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
        Split a path by this packer's separator, respecting escaped separators.

        Args:
            path: The path to split.

        Returns:
            List of path components.
        """
        return _split_path_static(path, self.separator, self.escape_char)

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
        # with_row_index always prepends a UInt32 column — no need to re-collect schema.
        fields: dict[str, PolarsDataType] = {ROW_ID_COLUMN: pl.UInt32}
        fields.update(schema)
        return lf, pl.Schema(fields)

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

    def _fold_order(self, at_level: str) -> list[str]:
        """
        The order :meth:`pack` folds ``at_level``'s descendants in.

        A post-order walk: every level appears after its own descendants (so a
        child is already a ``List[Struct]`` when its parent folds), and siblings
        follow declaration order. Sibling order is free for correctness — the
        branches are independent — but it fixes where each branch's column lands,
        since each fold appends its new column at the end. Declaration order is
        the one users can predict, and it is what
        :meth:`_pack_column_order` and :meth:`_pack_single_level` both assume.

        Args:
            at_level: The level being packed to. It is not itself included.

        Returns:
            Descendant level names in fold order. Empty for a leaf.
        """
        order: list[str] = []

        def visit(name: str) -> None:
            for child in self.spec.children_of(name):
                visit(child)
            order.append(name)

        for child in self.spec.children_of(at_level):
            visit(child)
        return order

    def _is_at_granularity(self, meta: LevelMetadata, schema: pl.Schema) -> bool:
        """
        Whether a frame with ``schema`` holds exactly one row per ``meta`` entity.

        It does when every level off ``meta``'s own axis is packed away: any
        off-axis level still carrying flat columns fans the frame out below
        ``meta``, making its values repeat. That covers both a **descendant**
        (streets flat in a frame we are reading at city level) and a **sibling
        branch** (streets flat while reading the service level) — the second is
        why this cannot simply ask whether the level was just exploded.

        Args:
            meta: The level whose granularity is in question.
            schema: Schema of the working frame.

        Returns:
            ``True`` if rows are one-per-entity at ``meta``.
        """
        axis = set(self.spec.axis_of(meta.name))
        for other in self._levels_meta:
            if other.name in axis:
                continue
            if self._own_level_columns(other, schema):
                return False
        return True

    def _require_immediate_child(self, from_level: str, to_level: str, operation: str) -> None:
        """
        Check that ``from_level`` is a direct child of ``to_level``.

        Operations that read a single ``List[Struct]`` column need the child to
        sit one hop below the parent. When ``to_level`` branches, the message
        lists every branch so the caller can see which one they meant.

        Args:
            from_level: The candidate child level.
            to_level: The candidate parent level.
            operation: Short description used in the error message.

        Raises:
            KeyError: If either level is not found in the hierarchy.
            ValueError: If ``from_level`` is not an immediate child of ``to_level``.
        """
        parent = self.spec.parent_of(from_level)
        self.spec._require_level(to_level)
        if parent != to_level:
            children = self.spec.children_of(to_level)
            raise ValueError(
                f"from_level '{from_level}' must be the immediate child of "
                f"to_level '{to_level}' for {operation}, but its parent is "
                f"{parent!r}. Immediate children of '{to_level}': {children}."
            )

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

        level_names = self.spec.axis_of(self.spec.levels[level_idx].name)
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
        """
        Build per-level metadata by walking the tree parents-first.

        A level's path and ancestor keys extend its *parent's*, which is the
        same thing as extending the previous level only in an unbranched chain.
        Levels are declared parents-first, so one forward pass suffices.
        """
        metas: list[LevelMetadata] = []
        by_name: dict[str, LevelMetadata] = {}

        for index, level in enumerate(self.spec.levels):
            parent_name = self.spec.parent_of(level.name)
            parent_meta = by_name[parent_name] if parent_name is not None else None

            if parent_meta is None:
                path = level.name
                ancestor_keys: tuple[str, ...] = ()
            else:
                path = f"{parent_meta.path}{self.separator}{level.name}"
                ancestor_keys = (*parent_meta.ancestor_keys, *parent_meta.id_columns)
            prefix = f"{path}{self.separator}" if path else ""

            id_columns, id_exprs = self._resolve_fields(index, level.id_fields)
            required_columns, required_exprs = self._resolve_fields(
                index, level.required_fields or ()
            )

            meta = LevelMetadata(
                index=index,
                name=level.name,
                path=path,
                prefix=prefix,
                ancestor_keys=ancestor_keys,
                id_columns=tuple(id_columns),
                id_exprs=tuple(id_exprs),
                required_columns=tuple(required_columns),
                required_exprs=tuple(required_exprs),
                order_by=tuple(level.order_by or ()),
                parent=parent_name,
                children=tuple(self.spec.children_of(level.name)),
            )
            metas.append(meta)
            by_name[level.name] = meta

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
        self, lf: pl.LazyFrame, level_idx: int, schema: pl.Schema, *, validate: bool = True
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

        # Put the child branches last, in declaration order, so the struct's
        # field order does not depend on which axis the caller unpacked. Without
        # this, packing a service-axis frame and a building-axis frame back to
        # the root would yield city structs with the same fields in different
        # orders. A chain already satisfies this — its single child column is
        # appended by the previous iteration — so the ordering is unchanged there.
        branch_paths = [
            self._levels_meta[self.spec.index_of(child)].path for child in meta.children
        ]
        branch_set = set(branch_paths)
        level_cols = [col for col in level_cols if col not in branch_set] + [
            path for path in branch_paths if path in level_cols
        ]

        group_keys = list(meta.ancestor_keys)

        # Child-list ordering is purely cosmetic: de-duplication and null recovery
        # of parent attributes are handled by ``drop_nulls().first()`` below and do
        # not depend on row order. We therefore avoid a global ``sort`` (a pipeline
        # breaker that prevents streaming) and instead sort the child list *inside*
        # the aggregation only when ordering is explicitly requested.
        order_exprs = list(meta.order_by) if meta.order_by else []

        # ``order_by`` expressions reference this level's (child) columns, which are
        # about to be folded into the struct. Materialize them into temporary
        # top-level columns so they remain available to ``sort_by`` in the agg.
        order_temp_cols: list[str] = []
        if order_exprs and group_keys:
            order_temp_cols = [f"{ORDER_TEMP_COLUMN_PREFIX}{i}" for i in range(len(order_exprs))]
            lf = lf.with_columns(
                [expr.alias(alias) for expr, alias in zip(order_exprs, order_temp_cols)]
            )

        struct_expr = pl.struct(
            [pl.col(col).alias(col[len(meta.prefix) :]) for col in level_cols]
        ).alias(meta.path)

        lf = lf.select(pl.all().exclude(level_cols), struct_expr)
        schema = lf.collect_schema()

        if not group_keys:
            return lf, schema

        has_row_id = ROW_ID_COLUMN in schema
        excluded = set(group_keys) | {meta.path} | set(order_temp_cols)
        if has_row_id:
            excluded.add(ROW_ID_COLUMN)
        remaining_cols = [col for col in schema.keys() if col not in excluded]

        # Validate that grouped values are identical if validation is enabled
        if validate and remaining_cols:
            self._validate_aggregation_uniformity(lf, group_keys, remaining_cols, meta.name)

        agg_exprs = [pl.col(col).drop_nulls().first().alias(col) for col in remaining_cols]

        # Aggregate child structs into a list, sorting within each group when child
        # order is requested. ``group_by`` without ``maintain_order`` lets the
        # streaming engine run the aggregation; top-level row order becomes
        # nondeterministic, which does not affect packed contents.
        sort_by_cols: list[str] = [*order_temp_cols]
        if self.preserve_child_order and has_row_id:
            sort_by_cols.append(ROW_ID_COLUMN)
        child_list = pl.col(meta.path).sort_by(sort_by_cols) if sort_by_cols else pl.col(meta.path)
        agg_exprs.append(child_list)

        # Carry the original row order upward (as the minimum child row id) so that
        # coarser levels can preserve child order without a global sort.
        if self.preserve_child_order and has_row_id:
            agg_exprs.append(pl.col(ROW_ID_COLUMN).min().alias(ROW_ID_COLUMN))

        lf = lf.group_by(group_keys).agg(agg_exprs)
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
        # Check all value columns for uniformity within groups in a single pass.
        # Using a unique alias per column avoids name collisions.
        agg_exprs = [
            pl.col(col).drop_nulls().n_unique().alias(f"__nuniq_{i}")
            for i, col in enumerate(value_cols)
        ]
        # Reduce to a single row of violation counts inside the engine rather than
        # pulling one row per group back into Python.
        result = (
            lf.group_by(group_keys)
            .agg(agg_exprs)
            .select(
                [
                    (pl.col(f"__nuniq_{i}") > 1).sum().alias(f"__nuniq_{i}")
                    for i in range(len(value_cols))
                ]
            )
            .collect()
            .row(0, named=True)
        )

        for i, col in enumerate(value_cols):
            non_uniform_count = result[f"__nuniq_{i}"]
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
        # ``pack`` always produces List, but a caller may hand us data whose level
        # column was cast to a fixed-size Array; ``explode`` handles both.
        if isinstance(dtype, (pl.List, pl.Array)):
            # Polars 2.0 flips this default to False, which would silently drop
            # parents whose child list is empty. Pin it so a childless parent
            # keeps surviving unpack as a single null-child row.
            lf = lf.explode(meta.path, empty_as_null=True)

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

    # Unpack to see the joined data. build_from_tables returns the union type
    # LazyFrame | DataFrame; the inputs here are eager, so narrow it.
    assert isinstance(nested, pl.DataFrame)
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

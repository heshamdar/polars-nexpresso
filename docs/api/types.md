# Types API Reference

This page documents the type definitions and data classes used in Nexpresso.

## Hierarchy Specification

### HierarchySpec

```python
@dataclass(frozen=True)
class HierarchySpec:
    levels: Sequence[LevelSpec]
    key_aliases: Mapping[str, str] = field(default_factory=dict)
```

Collection of `LevelSpec` objects ordered from coarse to fine granularity.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `levels` | `Sequence[LevelSpec]` | Ordered list of level specifications |
| `key_aliases` | `Mapping[str, str]` | Mapping of `{target_column: source_column}` |

**Class Methods:**

#### from_levels

```python
@classmethod
def from_levels(
    cls,
    *levels: LevelSpec,
    key_aliases: Mapping[str, str] | None = None,
) -> HierarchySpec:
```

Build a HierarchySpec from an ordered sequence of LevelSpec objects.

Validates that `parent_keys` in child levels match the count of `id_fields` in parent levels.

**Example:**

```python
spec = HierarchySpec.from_levels(
    LevelSpec(name="region", id_fields=["id"]),
    LevelSpec(name="store", id_fields=["id"], parent_keys=["region_id"]),
)
```

**Properties:**

#### levels_by_name

```python
@property
def levels_by_name(self) -> Mapping[str, LevelSpec]:
```

Get a mapping of level name to LevelSpec.

**Methods:**

#### index_of

```python
def index_of(self, level_name: str) -> int:
```

Get the index of a level by name.

**Raises:** `KeyError` if level not found.

#### level

```python
def level(self, level_name: str) -> LevelSpec:
```

Get a LevelSpec by name.

#### next_level

```python
def next_level(self, level_name: str) -> LevelSpec | None:
```

Get the next (child) level after the given level. Returns `None` for leaf level.

---

### LevelSpec

```python
@dataclass(frozen=True)
class LevelSpec:
    name: str
    id_fields: Sequence[ColumnSelector] = ()
    required_fields: Sequence[ColumnSelector] | None = None
    order_by: Sequence[pl.Expr] | None = None
    parent_keys: Sequence[str] | None = None
```

Declarative description of a hierarchy level.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Logical identifier for the level |
| `id_fields` | `Sequence[ColumnSelector]` | Columns that uniquely identify records |
| `required_fields` | `Sequence[ColumnSelector] \| None` | Columns that must be non-null |
| `order_by` | `Sequence[pl.Expr] \| None` | Expressions for ordering children |
| `parent_keys` | `Sequence[str] \| None` | Foreign keys to parent level |

**Example:**

```python
LevelSpec(
    name="store",
    id_fields=["id"],
    required_fields=["name"],
    order_by=[pl.col("store.opened_date").desc()],
    parent_keys=["region_id"],
)
```

---

## Exceptions

### HierarchyValidationError

```python
class HierarchyValidationError(Exception):
    level: str | None
    details: dict
```

Exception raised when hierarchy validation fails.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `level` | `str \| None` | The hierarchy level where error occurred |
| `details` | `dict` | Additional context about the error |

**Example:**

```python
try:
    packer.pack(df, "level")
except HierarchyValidationError as e:
    print(f"Error at level: {e.level}")
    print(f"Details: {e.details}")
```

---

## Type Aliases

### FrameT

```python
FrameT = TypeVar("FrameT", pl.LazyFrame, pl.DataFrame)
```

Type variable for DataFrame or LazyFrame. Used to preserve input type in return values.

### ColumnSelector

```python
ColumnSelector = str | pl.Expr
```

A column reference, either as a string name or a Polars expression.

### ExtraColumnsMode

```python
ExtraColumnsMode = Literal["preserve", "drop", "error"]
```

Controls handling of non-hierarchy columns during packing:

- `"preserve"` - Keep if values are uniform within groups
- `"drop"` - Silently remove
- `"error"` - Raise exception if present

---

## Internal Types

These types are primarily for internal use but are documented for completeness.

### LevelMetadata

```python
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
```

Internal metadata for a hierarchy level, computed from `LevelSpec`.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `index` | `int` | Zero-based position in hierarchy |
| `name` | `str` | Level identifier |
| `path` | `str` | Dot-separated path (e.g., "country.city") |
| `prefix` | `str` | Column prefix (e.g., "country.city.") |
| `ancestor_keys` | `tuple[str, ...]` | All ancestor key columns |
| `id_columns` | `tuple[str, ...]` | Qualified ID column names |
| `id_exprs` | `tuple[pl.Expr, ...]` | Computed ID expressions |
| `required_columns` | `tuple[str, ...]` | Qualified required column names |
| `required_exprs` | `tuple[pl.Expr, ...]` | Computed required expressions |
| `order_by` | `tuple[pl.Expr, ...]` | Ordering expressions |

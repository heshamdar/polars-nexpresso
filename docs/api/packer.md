# HierarchicalPacker API Reference

This page documents the hierarchical packer API for working with multi-level data.

## Classes

### HierarchicalPacker

```python
class HierarchicalPacker:
    def __init__(
        self,
        spec: HierarchySpec,
        *,
        granularity_separator: str = ".",
        escape_char: str = "\\",
        preserve_child_order: bool = True,
        validate_on_pack: bool = True,
    ) -> None: ...
```

General-purpose helper for packing/unpacking nested hierarchies in Polars.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spec` | `HierarchySpec` | Required | The hierarchy specification |
| `granularity_separator` | `str` | `"."` | Separator for column names |
| `escape_char` | `str` | `"\\"` | Escape character for separator in field names |
| `preserve_child_order` | `bool` | `True` | Maintain row order when packing |
| `validate_on_pack` | `bool` | `True` | Validate data integrity during pack |

---

## Methods

### pack

```python
def pack(
    self,
    frame: FrameT,
    to_level: str,
    *,
    extra_columns: Literal["preserve", "drop", "error"] = "preserve",
    parent_strategy: Literal["aggregate", "split_join"] = "aggregate",
) -> FrameT:
```

Pack flattened columns down to the specified level.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `frame` | `DataFrame \| LazyFrame` | The frame to pack |
| `to_level` | `str` | Target level name |
| `extra_columns` | `Literal["preserve", "drop", "error"]` | How to handle non-hierarchy columns |
| `parent_strategy` | `Literal["aggregate", "split_join"]` | `"aggregate"` (default) carries root attributes through the `group_by`; `"split_join"` reattaches them via a join, far cheaper when root attributes are heavy relative to child data |

**Returns:** Same type as input with nested structures

**Raises:**
- `KeyError` - If level not found
- `HierarchyValidationError` - If validation fails

!!! note "Row order"
    Packing does not perform a global sort, so **top-level row order is not
    guaranteed**. Child-list order is preserved when `preserve_child_order=True`
    (the default) or via a level's `order_by`. De-duplication and null handling
    of parent attributes are independent of order. For very large inputs, see
    [`pack_streaming`](#pack_streaming).

**Example:**

```python
# Pack to country level
packed = packer.pack(flat_df, "country")

# Drop extra columns
packed = packer.pack(flat_df, "country", extra_columns="drop")

# Reattach heavy root attributes via a join instead of the aggregation
packed = packer.pack(flat_df, "country", parent_strategy="split_join")
```

---

### unpack

```python
def unpack(self, frame: FrameT, to_level: str) -> FrameT:
```

Unpack nested structures to the specified level.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `frame` | `DataFrame \| LazyFrame` | The frame to unpack |
| `to_level` | `str` | Target level name |

**Returns:** Same type as input with flattened columns

**Example:**

```python
# Unpack to street level
flat = packer.unpack(packed_df, "street")
```

---

### pack_streaming

```python
def pack_streaming(
    self,
    source: LazyFrame | DataFrame | str | Path,
    to_level: str,
    *,
    partitions: int = 16,
    tmp_dir: str | Path | None = None,
    defer: bool = True,
    extra_columns: Literal["preserve", "drop", "error"] = "preserve",
) -> LazyFrame:
```

Memory-bounded version of [`pack`](#pack). `pack` aggregates with a `group_by`
whose state holds every group in memory, so peak memory scales with the whole
dataset even under the streaming engine. `pack_streaming` buckets the input by
the **root-level key** (keeping each entity's rows together), packs each bucket
independently while sinking it to Parquet, and returns a single `LazyFrame` over
the packed output — bounding peak memory to one bucket.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `DataFrame \| LazyFrame \| str \| Path` | Input at the finest granularity. A path/glob is scanned lazily with `scan_parquet`. |
| `to_level` | `str` | Target level name |
| `partitions` | `int` | Number of root-key buckets. More buckets = lower peak memory and more temporary files. Must be ≥ 1. |
| `tmp_dir` | `str \| Path \| None` | Directory for intermediate Parquet files. Defaults to a fresh temp directory the caller owns. |
| `defer` | `bool` | When `True` (default), wraps the work in `pl.defer` so nothing runs until the result is collected. When `False`, sinks eagerly and returns a `scan_parquet` handle (downstream streams straight from disk). |
| `extra_columns` | `Literal["preserve", "drop", "error"]` | How to handle non-hierarchy columns |

**Returns:** A `LazyFrame` over the packed result. Top-level row order is not
guaranteed; child-list order follows the same rules as [`pack`](#pack).

**Example:**

```python
# Bound peak memory by processing the data in 32 root-key buckets.
packed = packer.pack_streaming(flat_df, "region", partitions=32)

# Returns a LazyFrame, so keep composing lazily:
top = (
    packer.pack_streaming("dump/*.parquet", "region", partitions=64)
    .filter(pl.col("region.id").is_in(active_regions))
    .collect(engine="streaming")
)
```

---

### unpack_streaming

```python
def unpack_streaming(
    self,
    source: LazyFrame | DataFrame | str | Path,
    to_level: str,
    *,
    sink_path: str | Path | None = None,
) -> LazyFrame:
```

Streaming-friendly version of [`unpack`](#unpack) that returns a `LazyFrame`.
`unpack` (explode + unnest) already runs with bounded memory under the streaming
engine; this helper accepts a Parquet path (scanned lazily) or a frame and keeps
the pipeline lazy so it composes with downstream work or sinks straight to disk.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `DataFrame \| LazyFrame \| str \| Path` | Packed input. A path/glob is scanned lazily. |
| `to_level` | `str` | Target level name |
| `sink_path` | `str \| Path \| None` | When given, the result is streamed to this Parquet path via `sink_parquet` and a fresh scan over it is returned. |

**Returns:** A `LazyFrame` over the unpacked result.

**Example:**

```python
# Disk-to-disk: scan a packed Parquet file, unpack, and sink the leaves.
leaves = packer.unpack_streaming("packed.parquet", "store", sink_path="leaves.parquet")
```

---

### normalize

```python
def normalize(
    self,
    frame: FrameT,
    *,
    root_level: str | None = None,
) -> dict[str, FrameT]:
```

Split a frame into separate tables per hierarchy level. Thin wrapper: packs to
`root_level` and hands the result to [`split_levels`](#split_levels), so the
emitted tables have exactly that method's level-local shape.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `frame` | `DataFrame \| LazyFrame` | The frame to normalize |
| `root_level` | `str \| None` | Pack to this level first (default: first level) |

**Returns:** Dictionary mapping level names to tables, ordered root → leaf

**Example:**

```python
tables = packer.normalize(nested_df)

tables["country"].columns
# ['country.code', 'country.name']

tables["city"].columns
# ['country.code', 'country.city.id', 'country.city.population']
#   ^ foreign key   ^ own columns only — no country.name
```

Each table holds its own columns plus its ancestors' **key** columns. Coarser
attributes are not duplicated into finer tables; join them back from the coarser
table when you need them.

---

### denormalize

```python
def denormalize(
    self,
    tables: Mapping[str, DataFrame | LazyFrame],
    *,
    target_level: str | None = None,
) -> DataFrame | LazyFrame:
```

Reconstruct nested structure from per-level tables — the inverse of
[`normalize`](#normalize), and it accepts exactly the level-local shape that
`normalize` / [`split_levels`](#split_levels) emit.

Child levels are attached to their parents as nested list-of-struct columns,
walking leaf → root, and the root's own columns are then folded into the root
struct. When `target_level` is finer than the root, the ancestors' attribute
columns are joined back on afterwards (the upward pass only ever attaches
descendants).

**Round-trip guarantee** — for every level `L`:

```python
packer.denormalize(packer.normalize(df, root_level=L), target_level=L)
    == packer.pack(df, L)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `tables` | `Mapping[str, DataFrame \| LazyFrame]` | Level name to table mapping |
| `target_level` | `str \| None` | Target level (default: root) |

**Returns:** Reconstructed frame with nested structures

**Raises:** `HierarchyValidationError` if the root table is absent, a table
required to reach `target_level` is missing, or a supplied table lacks its own
key columns (those keys are what child tables join on).

**Example:**

```python
tables = packer.normalize(nested_df)

nested = packer.denormalize(tables)   # default: back to the root level
assert nested.equals(packer.pack(flat_df, "country"))   # single `country` struct

streets = packer.denormalize(tables, target_level="street")
assert streets.equals(packer.pack(flat_df, "street"))
```

---

### build_from_tables

```python
def build_from_tables(
    self,
    tables: Mapping[str, DataFrame | LazyFrame],
    *,
    target_level: str | None = None,
    join_type: Literal["left", "inner"] = "left",
) -> DataFrame | LazyFrame:
```

Build nested hierarchy from independent normalized tables.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `tables` | `Mapping[str, DataFrame \| LazyFrame]` | Level name to table mapping |
| `target_level` | `str \| None` | Pack to this level (default: root) |
| `join_type` | `Literal["left", "inner"]` | How to join tables |

**Returns:** Nested frame packed to target level

**Example:**

```python
nested = packer.build_from_tables({
    "region": regions_df,
    "store": stores_df,
    "product": products_df,
})
```

---

### validate

```python
def validate(
    self,
    frame: FrameT,
    *,
    level: str | None = None,
    raise_on_error: bool = True,
) -> list[HierarchyValidationError]:
```

Validate hierarchy constraints on a frame.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `frame` | `DataFrame \| LazyFrame` | Frame to validate |
| `level` | `str \| None` | Specific level to validate (default: all) |
| `raise_on_error` | `bool` | Raise on first error or collect all |

**Returns:** List of validation errors (empty if valid)

**Example:**

```python
# Collect all errors
errors = packer.validate(df, raise_on_error=False)

# Fail fast
packer.validate(df, raise_on_error=True)
```

---

### prepare_level_table

```python
def prepare_level_table(
    self,
    level_name: str,
    data: DataFrame | LazyFrame,
    column_mapping: dict[str, str] | None = None,
) -> DataFrame | LazyFrame:
```

Prepare a raw table for use in `build_from_tables`.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `level_name` | `str` | Target level in hierarchy |
| `data` | `DataFrame \| LazyFrame` | Raw data table |
| `column_mapping` | `dict[str, str] \| None` | Mapping of raw to target column names |

**Returns:** Table with properly prefixed columns

**Example:**

```python
prepared = packer.prepare_level_table(
    "product",
    raw_df,
    column_mapping={"prod_id": "id", "prod_name": "name"}
)
```

---

### get_level_columns

```python
def get_level_columns(self, level: str) -> list[str]:
```

Get all columns belonging to a level.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `level` | `str` | Level name |

**Returns:** List of qualified column names

**Example:**

```python
cols = packer.get_level_columns("city")
# ["country.city.id", "country.city.name", ...]
```

---

### split_levels

```python
def split_levels(self, frame: FrameT) -> dict[str, FrameT]:
```

Split a packed frame into standalone tables per level.

Each table is **level-local**: it holds the level's own columns (id fields and
attributes) plus the *key* columns of its ancestors, which serve as foreign keys
back to the coarser tables. Coarser attributes are not copied into finer tables,
and descendant columns are never included.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `frame` | `DataFrame \| LazyFrame` | Packed frame to split |

**Returns:** Dictionary mapping level names to tables, ordered root → leaf

**Example:**

For a `country → city → street` hierarchy:

```python
tables = packer.split_levels(packer.pack(flat_df, "country"))

tables["country"].columns
# ['country.code', 'country.name']

tables["city"].columns
# ['country.code', 'country.city.id', 'country.city.population']
#   ^ foreign key   ^ own columns

tables["street"].columns
# ['country.code', 'country.city.id',
#  'country.city.street.name', 'country.city.street.length']
```

Levels that are still flat in `frame` (e.g. `country` in a frame packed only to
`city`) get their own deduplicated table too, so no attribute is dropped.
`denormalize` accepts this shape and reattaches ancestor attributes when the
target level is finer than the root.

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
) -> FrameT:
```

Pack flattened columns down to the specified level.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `frame` | `DataFrame \| LazyFrame` | The frame to pack |
| `to_level` | `str` | Target level name |
| `extra_columns` | `Literal["preserve", "drop", "error"]` | How to handle non-hierarchy columns |

**Returns:** Same type as input with nested structures

**Raises:**
- `KeyError` - If level not found
- `HierarchyValidationError` - If validation fails

**Example:**

```python
# Pack to country level
packed = packer.pack(flat_df, "country")

# Drop extra columns
packed = packer.pack(flat_df, "country", extra_columns="drop")
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

### normalize

```python
def normalize(
    self,
    frame: FrameT,
    *,
    root_level: str | None = None,
) -> dict[str, FrameT]:
```

Split a frame into separate tables per hierarchy level.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `frame` | `DataFrame \| LazyFrame` | The frame to normalize |
| `root_level` | `str \| None` | Pack to this level first (default: first level) |

**Returns:** Dictionary mapping level names to tables

**Example:**

```python
tables = packer.normalize(nested_df)
# {"country": country_df, "city": city_df, "street": street_df}
```

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

Reconstruct nested structure from per-level tables.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `tables` | `Mapping[str, DataFrame \| LazyFrame]` | Level name to table mapping |
| `target_level` | `str \| None` | Target level (default: root) |

**Returns:** Reconstructed frame with nested structures

**Example:**

```python
nested = packer.denormalize({"country": ..., "city": ..., "street": ...})
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

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `frame` | `DataFrame \| LazyFrame` | Packed frame to split |

**Returns:** Dictionary mapping level names to tables

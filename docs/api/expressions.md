# Expression API Reference

This page documents the nested expression builder API.

## Functions

### generate_nested_exprs

```python
def generate_nested_exprs(
    fields: dict[str, FieldValue],
    schema: pl.Schema | pl.DataFrame | pl.LazyFrame,
    struct_mode: Literal["select", "with_fields"] = "select",
) -> list[pl.Expr]:
```

Generate Polars expressions for nested data operations.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `fields` | `dict[str, FieldValue]` | Dictionary defining operations on columns/fields |
| `schema` | `pl.Schema \| DataFrame \| LazyFrame` | Schema to use for type inference |
| `struct_mode` | `Literal["select", "with_fields"]` | How to handle struct fields |

**Returns:** `list[pl.Expr]` - Expressions ready for `.select()` or `.with_columns()`

**Example:**

```python
exprs = generate_nested_exprs(
    {"nested": {"a": lambda x: x * 2}},
    df.schema,
    struct_mode="with_fields"
)
result = df.select(exprs)
```

---

### apply_nested_operations

```python
def apply_nested_operations(
    df: FrameT,
    fields: dict[str, FieldValue],
    struct_mode: Literal["select", "with_fields"] = "select",
    use_with_columns: bool = False,
) -> FrameT:
```

Apply nested operations directly to a DataFrame or LazyFrame.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `DataFrame \| LazyFrame` | The frame to operate on |
| `fields` | `dict[str, FieldValue]` | Dictionary defining operations |
| `struct_mode` | `Literal["select", "with_fields"]` | How to handle struct fields |
| `use_with_columns` | `bool` | If True, use `.with_columns()` instead of `.select()` |

**Returns:** Same type as input (DataFrame or LazyFrame)

**Example:**

```python
result = apply_nested_operations(
    df,
    {"order": {"total": pl.field("price") * pl.field("qty")}},
    struct_mode="with_fields"
)
```

---

## Classes

### NestedExpressionBuilder

```python
class NestedExpressionBuilder:
    def __init__(
        self,
        schema: pl.Schema,
        struct_mode: Literal["select", "with_fields"] = "select",
    ) -> None: ...
    
    def build(self, fields: dict[str, FieldValue]) -> list[pl.Expr]: ...
```

Builder class for creating nested Polars expressions.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `schema` | `pl.Schema` | The schema of the DataFrame to work with |
| `struct_mode` | `Literal["select", "with_fields"]` | How to handle struct fields |

**Methods:**

#### build

```python
def build(self, fields: dict[str, FieldValue]) -> list[pl.Expr]:
```

Build a list of Polars expressions from the field specification.

**Example:**

```python
builder = NestedExpressionBuilder(df.schema, struct_mode="with_fields")
exprs = builder.build({"col": {"field": lambda x: x * 2}})
result = df.select(exprs)
```

---

## Type Aliases

### FieldValue

```python
FieldValue = None | dict[str, "FieldValue"] | Callable[[pl.Expr], pl.Expr] | pl.Expr
```

Defines valid values for field specifications:

- `None` - Keep the field unchanged
- `dict` - Recursively process nested structure
- `Callable` - Apply function to field
- `pl.Expr` - Use expression to create/modify field

### StructMode

```python
StructMode = Literal["select", "with_fields"]
```

Controls how unspecified struct fields are handled:

- `"select"` - Only keep specified fields (default)
- `"with_fields"` - Keep all fields, modify specified ones

---

## Exceptions

### ValueError

Raised when:
- Column doesn't exist and value is not `pl.Expr`
- Trying to apply function to non-existent field
- Trying to recurse into non-existent struct field
- Trying to recurse into non-nested type (not Struct, List, or Array)
- Invalid `struct_mode` value

### TypeError

Raised when:
- Field specification type is invalid (not None, dict, Callable, or pl.Expr)

# Struct Modes

When transforming struct fields, Nexpresso provides two modes that control how unspecified fields are handled.

!!! tip "Key Difference"
    - **`select`**: You must use `field: None` to keep fields you don't want to modify
    - **`with_fields`**: Fields **inside structs** are preserved automatically - only specify fields you want to add or modify

!!! warning "Top-level columns"
    The `struct_mode` only affects fields **inside struct columns**. Top-level DataFrame columns must always be included in the dictionary (use `column: None` to keep unchanged).

## The Two Modes

### `select` Mode (Default)

Only keeps fields that are explicitly specified in the dictionary. **You must use `field: None` to keep any field you want to preserve.**

```python
df = pl.DataFrame({
    "data": [{"a": 1, "b": 2, "c": 3}]
})

result = apply_nested_operations(df, {
    "data": {
        "a": None,       # Keep "a" unchanged
        "b": lambda x: x * 2,  # Modify "b"
        # "c" is NOT specified, so it will be DROPPED!
    }
}, struct_mode="select")

# Result: {"a": 1, "b": 4}
# Field "c" is dropped because it wasn't specified!
```

### `with_fields` Mode

Keeps all existing fields automatically. **No need to specify `field: None`** - only specify fields you want to add or modify.

```python
result = apply_nested_operations(df, {
    "data": {
        "a": lambda x: x * 10,  # Modify "a"
        # "b" and "c" are preserved automatically - no need to list them!
    }
}, struct_mode="with_fields")

# Result: {"a": 10, "b": 2, "c": 3}
# All fields preserved, only "a" modified
```

## When to Use Each Mode

### Use `select` When:

- You want to filter down to specific fields
- You're reshaping data for an API response
- You need to remove sensitive fields
- You're creating a projection of the data

```python
# Create a public-facing view of user data
public_fields = apply_nested_operations(users_df, {
    "user": {
        "name": None,
        "avatar_url": None,
        # Exclude: email, phone, password_hash, etc.
    }
}, struct_mode="select")
```

### Use `with_fields` When:

- You're adding calculated fields
- You're modifying existing fields while keeping others
- You don't want to enumerate all fields
- You're enriching data

```python
# Add calculated metrics without losing existing data
enriched = apply_nested_operations(orders_df, {
    "order": {
        "subtotal": pl.field("price") * pl.field("qty"),
        "tax": pl.field("price") * pl.field("qty") * 0.08,
    }
}, struct_mode="with_fields")
```

## Comparison Table

| Aspect | `select` | `with_fields` |
|--------|----------|---------------|
| Unspecified fields | Dropped | Preserved |
| Use case | Filter/project | Enrich/modify |
| Output schema | Explicit | Additive |
| Safety | Won't accidentally keep fields | Won't accidentally lose fields |

## Combining with Nested Structures

The mode applies recursively to all nested levels:

```python
# with_fields preserves at all levels
apply_nested_operations(df, {
    "order": {
        "customer": {
            "loyalty_points": pl.field("purchases") * 10,
        },
        "items": {
            "total": pl.field("price") * pl.field("qty"),
        }
    }
}, struct_mode="with_fields")
```

## Empty Dictionary Behavior

The mode also affects how empty dictionaries are handled:

```python
# select mode with empty dict
{"struct": {}}  # Error: can't create empty struct

# with_fields mode with empty dict
{"struct": {}}  # Returns struct unchanged
```

## Common Mistakes

### Forgetting Fields in Select Mode

```python
# Oops! Forgot to include "customer_id"
apply_nested_operations(orders, {
    "order": {
        "total": None,
        "status": None,
        # Missing: "customer_id" - it will be dropped!
    }
}, struct_mode="select")
```

**Fix**: Use `with_fields` if you want to keep all fields, or explicitly include all needed fields.

### Using Select When You Meant With Fields

```python
# Trying to add a field but using select mode
apply_nested_operations(df, {
    "data": {
        "new_field": pl.field("a") + pl.field("b"),
    }
}, struct_mode="select")

# Result only has "new_field", lost "a" and "b"!
```

**Fix**: Use `struct_mode="with_fields"` when adding fields.

## Best Practices

1. **Be explicit about your intent**: Choose the mode that matches what you're trying to do
2. **Use `with_fields` by default** when unsure - it's safer against data loss
3. **Use `select` deliberately** when you specifically want to filter fields
4. **Document your choice** in comments when it's not obvious

```python
# Deliberately selecting only safe fields for external API
result = apply_nested_operations(data, fields, struct_mode="select")
```

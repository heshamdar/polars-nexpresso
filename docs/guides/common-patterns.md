# Common Patterns

This guide shows recipes for common tasks with Nexpresso.

## Nested Expression Patterns

### Adding Calculated Fields

```python
# Add profit and margin to products
result = apply_nested_operations(df, {
    "products": {
        "name": None,
        "price": None,
        "cost": None,
        "profit": pl.field("price") - pl.field("cost"),
        "margin": (pl.field("price") - pl.field("cost")) / pl.field("price"),
    }
}, struct_mode="with_fields")
```

### Conditional Logic

```python
# Apply tiered discounts
result = apply_nested_operations(df, {
    "customer": {
        "tier": None,
        "discount": pl.when(pl.field("tier") == "Gold")
            .then(0.15)
            .when(pl.field("tier") == "Silver")
            .then(0.10)
            .otherwise(0.05),
    }
}, struct_mode="with_fields")
```

### String Transformations

```python
# Normalize text fields
result = apply_nested_operations(df, {
    "user": {
        "email": lambda x: x.str.to_lowercase(),
        "name": lambda x: x.str.strip_chars(),
        "name_upper": pl.field("name").str.to_uppercase(),
    }
}, struct_mode="with_fields")
```

### Working with Dates

```python
# Extract date components
result = apply_nested_operations(df, {
    "event": {
        "timestamp": None,
        "year": pl.field("timestamp").dt.year(),
        "month": pl.field("timestamp").dt.month(),
        "day_of_week": pl.field("timestamp").dt.weekday(),
    }
}, struct_mode="with_fields")
```

### Aggregating Within Lists

```python
# Note: Use pl.element() inside list.eval() context
result = df.with_columns(
    pl.col("items").list.eval(
        pl.element().struct.field("price")
    ).list.sum().alias("total")
)
```

## Hierarchical Data Patterns

### Pack and Transform, Then Unpack

```python
# 1. Start with flat data
flat = packer.unpack(nested, "product")

# 2. Add calculations
flat = flat.with_columns([
    (pl.col("store.product.price") * pl.col("store.product.qty"))
        .alias("store.product.total")
])

# 3. Pack back up
packed = packer.pack(flat, "store")
```

### Filter at a Specific Level

```python
# Unpack to target level
stores = packer.unpack(nested, "store")

# Apply filter
active_stores = stores.filter(pl.col("region.store.active") == True)

# Pack back to desired level
result = packer.pack(active_stores, "region")
```

### Split, Transform, Rejoin

```python
# Split into separate tables
tables = packer.normalize(nested)

# Transform individual tables
tables["product"] = tables["product"].with_columns([
    (pl.col("store.product.price") * 1.1).alias("store.product.price")
])

# Reconstruct
rebuilt = packer.denormalize(tables)
```

### Handling Missing Children

```python
# Parents without children get empty lists
nested = packer.build_from_tables({
    "parent": parents,
    "child": children,  # Some parents may not have children
}, join_type="left")

# To exclude childless parents:
nested = packer.build_from_tables({
    "parent": parents,
    "child": children,
}, join_type="inner")
```

## Combining Expression Builder and Packer

### Transform, Then Pack

```python
# Transform flat data first
transformed = apply_nested_operations(flat_df, {
    "store.product.price": lambda x: x * 1.1,
}, struct_mode="with_fields")

# Then pack
packed = packer.pack(transformed, "store")
```

### Pack, Transform, Unpack

```python
# Pack to get nested structure
packed = packer.pack(flat_df, "region")

# Transform nested data
result = apply_nested_operations(packed, {
    "region": {
        "store": {
            "product": {
                "profit": pl.field("price") - pl.field("cost"),
            }
        }
    }
}, struct_mode="with_fields")

# Unpack for analysis
flat = packer.unpack(result, "product")
```

## Error Handling Patterns

### Safe Field Access

```python
# Handle potentially missing fields
result = apply_nested_operations(df, {
    "data": {
        "value": None,
        # Use coalesce for defaults
        "safe_value": pl.field("value").fill_null(0),
    }
}, struct_mode="with_fields")
```

### Validation Before Packing

```python
# Validate data integrity
errors = packer.validate(df, raise_on_error=False)
if errors:
    for error in errors:
        print(f"Warning: {error}")

# Or fail fast
packer.validate(df, raise_on_error=True)  # Raises on first error
```

### Handling Extra Columns

```python
# Be explicit about extra columns
try:
    packed = packer.pack(df, "level", extra_columns="error")
except HierarchyValidationError as e:
    print(f"Unexpected columns: {e.details['extra_columns']}")
```

## Performance Patterns

### Use LazyFrames Throughout

```python
# Start lazy
lf = df.lazy()

# All operations return LazyFrame
packed = packer.pack(lf, "store")  # LazyFrame
result = apply_nested_operations(packed, fields)  # LazyFrame

# Collect only at the end
final = result.collect()
```

### Minimize Pack/Unpack Cycles

```python
# Instead of multiple cycles:
# ❌ pack -> unpack -> pack -> unpack

# Do transformations at the right level:
# ✅ unpack to finest needed level, transform, pack once
flat = packer.unpack(nested, "product")
flat = flat.with_columns([...])  # All transformations
packed = packer.pack(flat, "region")
```

### Schema Caching

The HierarchicalPacker internally caches schema information to avoid repeated collection. This happens automatically.

## Type Preservation Pattern

```python
# Input type is preserved
df = pl.DataFrame({...})
lf = pl.LazyFrame({...})

# DataFrame in -> DataFrame out
result_df = packer.pack(df, "level")
assert isinstance(result_df, pl.DataFrame)

# LazyFrame in -> LazyFrame out
result_lf = packer.pack(lf, "level")
assert isinstance(result_lf, pl.LazyFrame)
```

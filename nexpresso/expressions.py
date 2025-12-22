"""
Polars Nexpresso - A helper module for generating Polars expressions to work with nested data structures.

This module provides utilities to easily select, modify, and create columns and nested
fields in Polars DataFrames, particularly for complex nested structures like lists of
structs and deeply nested hierarchies.
"""

from collections.abc import Callable
from functools import lru_cache
from typing import Literal

import polars as pl
from packaging import version
from polars._typing import PolarsDataType
from polars.expr.expr import Expr

from nexpresso.hierarchical_packer import FrameT


@lru_cache(maxsize=1)
def _polars_version() -> version.Version:
    """Get the current Polars version as a parsed Version object."""
    return version.parse(pl.__version__)


def _supports_arr_eval() -> bool:
    """Check if the current Polars version supports arr.eval()."""
    return _polars_version() >= version.parse("1.35.1")


# Type aliases for better readability
FieldValue = None | dict[str, "FieldValue"] | Callable[[pl.Expr], pl.Expr] | pl.Expr

StructMode = Literal["select", "with_fields"]


class NestedExpressionBuilder:
    """
    Builder class for creating nested Polars expressions.

    This class encapsulates the logic for generating expressions that work with
    nested data structures, providing a cleaner and more maintainable API.
    """

    def __init__(
        self,
        schema: pl.Schema,
        struct_mode: StructMode = "select",
    ) -> None:
        """
        Initialize the builder with a schema and mode.

        Args:
            schema: The schema of the DataFrame to work with.
            struct_mode: How to handle struct fields:
                - 'select': Only keep specified fields (default)
                - 'with_fields': Keep all existing fields and add/modify specified ones
        """
        if struct_mode not in ("select", "with_fields"):
            raise ValueError(
                f"Invalid struct_mode: {struct_mode}. " "Must be 'select' or 'with_fields'."
            )
        self._schema = schema
        self._struct_mode = struct_mode

    def build(self, fields: dict[str, FieldValue]) -> list[pl.Expr]:
        """
        Build a list of Polars expressions from the field specification.

        Args:
            fields: A dictionary defining operations on columns/fields.
                - `key`: Column/field name
                - `value`: Operation specification:
                    - `None`: Select field as-is
                    - `dict`: Recursively process nested structure
                    - `Callable`: Apply function to field (e.g., lambda x: x + 1)
                    - `pl.Expr`: Full expression to create/modify field

        Returns:
            List of Polars expressions ready for use in `.select()` or `.with_columns()`

        Raises:
            ValueError: If column doesn't exist or operations are invalid
            TypeError: If field value type is invalid
        """
        expressions = []

        for col_name, field_spec in fields.items():
            expr = self._process_top_level_field(col_name, field_spec)
            expressions.append(expr)

        return expressions

    def _process_top_level_field(
        self,
        col_name: str,
        field_spec: FieldValue,
    ) -> pl.Expr:
        """Process a top-level field specification."""
        base_expr = pl.col(col_name)

        # Handle column creation
        if col_name not in self._schema:
            if not isinstance(field_spec, pl.Expr):
                raise ValueError(
                    f"Column '{col_name}' not found in schema. "
                    "To create a new column, provide a pl.Expr."
                )
            return field_spec.alias(col_name)

        # Handle different field specification types
        if field_spec is None:
            return base_expr
        elif isinstance(field_spec, pl.Expr):
            return field_spec.alias(col_name)
        elif callable(field_spec):
            return field_spec(base_expr).alias(col_name)
        elif isinstance(field_spec, dict):
            col_type: PolarsDataType = self._schema[col_name]
            return self._process_nested_field(col_type, field_spec, base_expr).alias(col_name)
        else:
            raise TypeError(
                f"Invalid field specification type for '{col_name}': "
                f"{type(field_spec)}. Expected None, dict, Callable, or pl.Expr."
            )

    def _process_nested_field(
        self,
        dtype: PolarsDataType,
        field_spec: dict[str, FieldValue],
        base_expr: pl.Expr,
    ) -> pl.Expr:
        """
        Recursively process nested field specifications.

        Handles structs, lists, and arrays with proper type inference.
        """
        # Handle List types (including nested lists)
        if isinstance(dtype, pl.List):
            return self._process_list_field(dtype.inner, field_spec, base_expr)

        # Handle Array types (fixed-size arrays)
        # As of Polars 1.0+, Arrays support arr.eval() for element-wise operations
        if isinstance(dtype, pl.Array):
            if not _supports_arr_eval():
                raise ValueError(
                    f"Array types require Polars >= 1.0.0 for arr.eval() support. "
                    f"Current version: {pl.__version__}. "
                    "Workaround: Convert the Array to a List first using "
                    ".cast(pl.List(inner_type))."
                )
            inner_expr = self._process_nested_field(dtype.inner, field_spec, pl.element())
            # Use arr.eval for arrays (available in Polars 1.0+)
            return base_expr.arr.eval(inner_expr)

        # Handle Struct types
        if isinstance(dtype, pl.Struct):
            return self._process_struct_field(dtype, field_spec, base_expr)

        # If we reach here, we're trying to recurse into a non-nested type
        raise ValueError(
            f"Cannot recurse into field with type {dtype}. "
            "Only Struct, List, and Array types support nested operations."
        )

    def _process_list_field(
        self,
        inner_dtype: PolarsDataType,
        field_spec: dict[str, FieldValue],
        base_expr: pl.Expr,
    ) -> pl.Expr:
        """
        Process operations on list elements.

        Uses list.eval() to apply expressions to each element in the list.
        """
        # Recursively process the inner type
        inner_expr = self._process_nested_field(inner_dtype, field_spec, pl.element())

        return base_expr.list.eval(inner_expr)

    def _process_struct_field(
        self,
        struct_dtype: pl.Struct,
        field_spec: dict[str, FieldValue],
        base_expr: pl.Expr,
    ) -> pl.Expr:
        """
        Process operations on struct fields.

        Handles both 'select' and 'with_fields' modes appropriately.
        """
        schema_map: dict[str, PolarsDataType] = {
            field.name: field.dtype for field in struct_dtype.fields
        }
        # Track transformed expressions so new fields can reference them
        transformed_fields: dict[str, pl.Expr] = {}
        field_exprs_to_use: dict[str, pl.Expr] = {}

        # First pass: build all field expressions (without aliasing yet)
        for field_name, sub_spec in field_spec.items():
            if sub_spec is None:
                # In 'select' mode, None means include the field as-is
                # In 'with_fields' mode, None means keep existing field unchanged
                if self._struct_mode == "select":
                    # Will be handled in final selection
                    field_exprs_to_use[field_name] = base_expr.struct.field(field_name)
                else:
                    # with_fields mode: keep existing field
                    field_exprs_to_use[field_name] = base_expr.struct.field(field_name)
                continue

            field_expr = self._build_field_expression(
                field_name, sub_spec, schema_map, base_expr, transformed_fields
            )

            if field_expr is not None:
                # Store the expression (we'll alias later)
                transformed_fields[field_name] = field_expr
                field_exprs_to_use[field_name] = field_expr

        # Second pass: build final field expressions with aliases
        final_exprs: list[Expr] = []
        for field_name, expr in field_exprs_to_use.items():
            final_exprs.append(expr.alias(field_name))

        # Build the struct appropriately based on mode
        if self._struct_mode == "select":
            # In select mode, only include specified fields
            selected_fields: list[Expr] = []
            if final_exprs:
                struct_with_transforms: Expr = base_expr.struct.with_fields(final_exprs)
                selected_fields.extend(
                    [struct_with_transforms.struct.field(name) for name in field_spec.keys()]
                )
            else:
                # No transformations, just select
                selected_fields.extend([base_expr.struct.field(name) for name in field_spec.keys()])
            return pl.struct(selected_fields)
        else:
            # In with_fields mode, use with_fields() which preserves original field references
            # This ensures pl.field() references the original struct, not transformed fields
            if final_exprs:
                return base_expr.struct.with_fields(final_exprs)
            return base_expr

    def _build_field_expression(
        self,
        field_name: str,
        field_spec: dict[str, FieldValue] | Callable[[pl.Expr], pl.Expr] | pl.Expr,
        schema_map: dict[str, PolarsDataType],
        base_expr: pl.Expr,
        transformed_fields: dict[str, pl.Expr] | None = None,
    ) -> pl.Expr | None:
        """
        Build an expression for a single struct field.

        Returns None if the field should be kept as-is (for with_fields mode).

        Args:
            transformed_fields: Dictionary of already-transformed field expressions
                that can be referenced by pl.field() calls in new field expressions.
        """
        if transformed_fields is None:
            transformed_fields = {}

        field_base_expr = base_expr.struct.field(field_name)

        if isinstance(field_spec, pl.Expr):
            # Return the expression as-is
            # Note: pl.field() references the ORIGINAL struct fields, not transformed ones
            # This is the expected Polars behavior
            return field_spec
        elif callable(field_spec):
            if field_name not in schema_map:
                raise ValueError(
                    f"Cannot apply function to non-existent field '{field_name}'. "
                    "Use pl.Expr to create a new field."
                )
            return field_spec(field_base_expr)
        elif isinstance(field_spec, dict):
            if field_name not in schema_map:
                raise ValueError(f"Cannot recurse into non-existent struct field '{field_name}'.")
            return self._process_nested_field(schema_map[field_name], field_spec, field_base_expr)
        else:
            raise TypeError(f"Invalid field specification for '{field_name}': {type(field_spec)}")


def generate_nested_exprs(
    fields: dict[str, FieldValue],
    schema: pl.Schema | FrameT,
    struct_mode: StructMode = "select",
) -> list[pl.Expr]:
    """
    Generate Polars expressions for nested data operations.

    This is a convenience function that creates a NestedExpressionBuilder
    and builds expressions from the field specification.

    Args:
        fields: Dictionary defining operations on columns/fields.
            Each key is a column/field name, and the value specifies the operation:
            - `None`: Select field as-is
            - `dict`: Recursively process nested structure
            - `Callable`: Apply function to field (e.g., `lambda x: x + 1`)
            - `pl.Expr`: Full expression to create/modify field

        schema: The schema of the DataFrame to work with. Can be a Schema, DataFrame, or LazyFrame.
        If a DataFrame or LazyFrame is provided, the schema will be collected automatically.

        struct_mode: How to handle struct fields:
            - `'select'`: Only keep specified fields (default)
            - `'with_fields'`: Keep all existing fields and add/modify specified ones

    Returns:
        List of Polars expressions ready for use in `.select()` or `.with_columns()`

    Examples:
        >>> df = pl.DataFrame({
        ...     "a": [1, 2, 3],
        ...     "nested": [{"x": 10, "y": 20}, {"x": 11, "y": 21}]
        ... })
        >>>
        >>> # Select and transform nested fields
        >>> exprs = generate_nested_exprs({
        ...     "a": lambda x: x * 2,
        ...     "nested": {
        ...         "x": lambda x: x + 1,
        ...         "new_field": pl.lit(100)
        ...     }
        ... }, df.collect_schema())
        >>> df.select(exprs)

        >>> # Work with lists of structs
        >>> df2 = pl.DataFrame({
        ...     "items": [
        ...         [{"value": 1}, {"value": 2}],
        ...         [{"value": 3}, {"value": 4}]
        ...     ]
        ... })
        >>> exprs = generate_nested_exprs({
        ...     "items": {
        ...         "value": lambda x: x * 2
        ...     }
        ... }, df2.schema)
        >>> df2.select(exprs)
    """

    if isinstance(schema, pl.DataFrame | pl.LazyFrame):
        schema = schema.collect_schema()

    builder = NestedExpressionBuilder(schema, struct_mode)
    return builder.build(fields)


# Convenience class method for direct DataFrame usage
def apply_nested_operations(
    df: FrameT,
    fields: dict[str, FieldValue],
    struct_mode: StructMode = "select",
    use_with_columns: bool = False,
) -> FrameT:
    """
    Apply nested operations directly to a DataFrame or LazyFrame.

    This is a convenience function that combines expression generation
    with DataFrame operation application.

    Args:
        df: The DataFrame or LazyFrame to operate on.
        fields: Dictionary defining operations (see `generate_nested_exprs`).
        struct_mode: How to handle struct fields (see `generate_nested_exprs`).
        use_with_columns: If True, use `.with_columns()` instead of `.select()`.

    Returns:
        DataFrame or LazyFrame with operations applied.

    Examples:
        >>> df = pl.DataFrame({
        ...     "a": [1, 2, 3],
        ...     "nested": [{"x": 10}, {"x": 11}]
        ... })
        >>>
        >>> result = apply_nested_operations(
        ...     df,
        ...     {"nested": {"x": lambda x: x * 2}},
        ...     struct_mode="with_fields"
        ... )
    """
    exprs = generate_nested_exprs(fields, df.collect_schema(), struct_mode)

    if use_with_columns:
        return df.with_columns(exprs)
    else:
        return df.select(exprs)


if __name__ == "__main__":
    # Example usage
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [
                {"x": 10, "y": [{"z": 11}, {"z": 12}]},
                {"x": 12, "y": [{"z": 13}, {"z": 14}]},
                {"x": 14, "y": [{"z": 15}, {"z": 16}]},
            ],
        }
    )

    print("Original DataFrame:")
    print(df)
    print("\nSchema:")
    print(df.collect_schema())
    print("\n" + "=" * 80 + "\n")

    # Example 1: Using the convenience function
    query = {
        "a": lambda x: x.max(),
        "c": {
            "x": lambda x: x.max(),
            "y": {
                "z": lambda x: x.max(),
                "new_field": pl.field("z").min(),
                "new_lit": pl.lit(100),
            },
        },
    }

    print("Example 1: Using generate_nested_exprs with 'with_fields' mode")
    generated_expr = generate_nested_exprs(query, df.collect_schema(), "with_fields")
    result1 = df.select(generated_expr)
    print(result1)
    print("\nSchema:")
    print(result1.collect_schema())
    print("\n" + "=" * 80 + "\n")

    # Example 2: Using the direct application function
    print("Example 2: Using apply_nested_operations")
    result2 = apply_nested_operations(
        df,
        {"c": {"x": lambda x: x * 2}},
        struct_mode="with_fields",
        use_with_columns=True,
    )
    print(result2)
    print("\n" + "=" * 80 + "\n")

    # Example 3: Using 'select' mode
    print("Example 3: Using 'select' mode (only specified fields)")
    result3 = apply_nested_operations(
        df,
        {"c": {"x": lambda x: x * 2}},
        struct_mode="select",
    )
    print(result3)

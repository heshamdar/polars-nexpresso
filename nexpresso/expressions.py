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

# Filter type aliases
# FilterValue: spec for row-level predicates (used with generate_nested_filter_expr)
FilterValue = Callable[[pl.Expr], pl.Expr] | pl.Expr | dict[str, "FilterValue"]

# ElementFilterSpec: spec for element-level predicates (used with filter_nested_elements)
ElementFilterSpec = Callable[[pl.Expr], pl.Expr] | pl.Expr | dict[str, "ElementFilterSpec"]


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


class NestedFilterBuilder:
    """
    Builder class for creating nested Polars filter expressions.

    Converts a nested dictionary specification into Polars boolean expressions for
    row-level filtering or element-level list filtering.

    The callable at each dict path endpoint receives the Polars expression **at that
    structural level**, so users can write any aggregation or predicate they need:

    - Top-level column: callable receives ``pl.col("col")``
    - Struct field via dict: callable receives ``pl.col("col").struct.field("f")``
    - List<Struct> field via dict (row filter): callable receives
      ``pl.col("col").list.eval(pl.element().struct.field("f"))`` — a List expression
      the user can reduce with ``.list.max()``, ``.list.any()``, etc.
    - List<Struct> field via dict (element filter): callable receives
      ``pl.element().struct.field("f")`` and must return an element-level boolean.
    """

    def __init__(self, schema: pl.Schema) -> None:
        """
        Initialize the builder with a DataFrame schema.

        Args:
            schema: The schema of the DataFrame to filter.
        """
        self._schema = schema

    def build_row_filter(self, filter_spec: dict[str, FilterValue]) -> pl.Expr:
        """
        Build a boolean row-level expression from the filter specification.

        Multiple top-level keys are AND-combined into a single expression.

        Args:
            filter_spec: Dict mapping column names to predicates. Each value is:
                - ``Callable``: receives the column/field expression, returns a boolean expr
                - ``pl.Expr``: used directly as a boolean expression
                - ``dict``: navigate into a nested Struct or List<Struct> field

        Returns:
            A boolean ``pl.Expr`` suitable for use with ``.filter()``.

        Raises:
            ValueError: If a column is not found or dict is used on a non-nested type.
            TypeError: If a filter spec value has an unsupported type.

        Examples:
            >>> builder = NestedFilterBuilder(df.schema)
            >>> pred = builder.build_row_filter({
            ...     "population": lambda x: x > 100_000,
            ...     "streets": {"length_km": lambda x: x.list.max() > 1},
            ... })
            >>> df.filter(pred)
        """
        if not filter_spec:
            return pl.lit(True)

        predicates = [self._process_col_predicate(col, spec) for col, spec in filter_spec.items()]
        result = predicates[0]
        for pred in predicates[1:]:
            result = result & pred
        return result

    def build_element_filters(
        self, filter_spec: dict[str, ElementFilterSpec]
    ) -> list[pl.Expr]:
        """
        Build with_columns-style expressions that filter list elements in-place.

        Rows are never dropped — only the elements within each list column are filtered.

        Args:
            filter_spec: Dict mapping column names (must be List type) to element
                predicates. Each value is:
                - ``Callable``: receives element-level expression, returns element bool
                - ``pl.Expr``: used directly as element-level boolean
                - ``dict``: navigate into struct fields of each list element

        Returns:
            List of aliased Polars expressions for use with ``.with_columns()``.

        Raises:
            ValueError: If a column is not a List type or is not found.
            TypeError: If a spec value has an unsupported type.

        Examples:
            >>> builder = NestedFilterBuilder(df.schema)
            >>> exprs = builder.build_element_filters({
            ...     "streets": {"length_km": lambda x: x > 1},
            ... })
            >>> df.with_columns(exprs)
        """
        return [self._process_col_element_filter(col, spec) for col, spec in filter_spec.items()]

    # -------------------------------------------------------------------------
    # Row-filter private helpers
    # -------------------------------------------------------------------------

    def _process_col_predicate(self, col_name: str, spec: FilterValue) -> pl.Expr:
        """Build a row-level boolean predicate for a single top-level column."""
        if col_name not in self._schema:
            available = list(self._schema.names())
            raise ValueError(
                f"Column '{col_name}' not found in schema. "
                f"Available columns: {available}"
            )

        dtype = self._schema[col_name]
        base_expr = pl.col(col_name)

        if callable(spec):
            return spec(base_expr)
        elif isinstance(spec, pl.Expr):
            return spec
        elif isinstance(spec, dict):
            if isinstance(dtype, pl.Struct):
                return self._build_struct_row_predicate(dtype, spec, base_expr)
            elif isinstance(dtype, pl.List):
                return self._build_list_row_predicate(dtype.inner, spec, base_expr)
            else:
                raise ValueError(
                    f"Cannot use a dict filter spec on column '{col_name}' with type "
                    f"{dtype}. Dict specs are only supported for Struct and List types. "
                    "Use a callable or pl.Expr for scalar columns."
                )
        else:
            raise TypeError(
                f"Invalid filter spec type for column '{col_name}': {type(spec)}. "
                "Expected Callable, pl.Expr, or dict."
            )

    def _build_struct_row_predicate(
        self,
        struct_dtype: pl.Struct,
        spec: dict[str, FilterValue],
        base_expr: pl.Expr,
    ) -> pl.Expr:
        """AND-combine row-level predicates for each specified struct field."""
        schema_map: dict[str, PolarsDataType] = {f.name: f.dtype for f in struct_dtype.fields}
        predicates: list[pl.Expr] = []

        for field_name, field_spec in spec.items():
            if field_name not in schema_map:
                available = list(schema_map.keys())
                raise ValueError(
                    f"Field '{field_name}' not found in struct. "
                    f"Available fields: {available}"
                )
            field_expr = base_expr.struct.field(field_name)
            field_dtype = schema_map[field_name]

            if callable(field_spec):
                predicates.append(field_spec(field_expr))
            elif isinstance(field_spec, pl.Expr):
                predicates.append(field_spec)
            elif isinstance(field_spec, dict):
                if isinstance(field_dtype, pl.Struct):
                    predicates.append(
                        self._build_struct_row_predicate(field_dtype, field_spec, field_expr)
                    )
                elif isinstance(field_dtype, pl.List):
                    predicates.append(
                        self._build_list_row_predicate(field_dtype.inner, field_spec, field_expr)
                    )
                else:
                    raise ValueError(
                        f"Cannot use a dict filter spec on field '{field_name}' with type "
                        f"{field_dtype}. Only Struct and List fields support dict specs."
                    )
            else:
                raise TypeError(
                    f"Invalid filter spec type for field '{field_name}': {type(field_spec)}. "
                    "Expected Callable, pl.Expr, or dict."
                )

        if not predicates:
            return pl.lit(True)
        result = predicates[0]
        for pred in predicates[1:]:
            result = result & pred
        return result

    def _build_list_row_predicate(
        self,
        inner_dtype: PolarsDataType,
        spec: dict[str, FilterValue],
        base_expr: pl.Expr,
    ) -> pl.Expr:
        """
        Build row-level predicates for a List<Struct> column.

        For each field key in spec, extracts ``base_expr.list.eval(element.struct.field(f))``
        — a List<T> expression — and passes it to the user's callable so they can apply
        any list aggregation (e.g. ``.list.max() > 1``, ``.list.any()``, etc.).
        Multiple fields are AND-combined.
        """
        if not isinstance(inner_dtype, pl.Struct):
            raise ValueError(
                f"Dict filter spec on a List column requires the list element type to be "
                f"a Struct, but got List<{inner_dtype}>. "
                "Use a top-level callable to write the predicate directly."
            )

        schema_map: dict[str, PolarsDataType] = {f.name: f.dtype for f in inner_dtype.fields}
        predicates: list[pl.Expr] = []

        for field_name, field_spec in spec.items():
            if field_name not in schema_map:
                available = list(schema_map.keys())
                raise ValueError(
                    f"Field '{field_name}' not found in list element struct. "
                    f"Available fields: {available}"
                )
            # Extract the field across all list elements → gives List<T>
            extracted = base_expr.list.eval(pl.element().struct.field(field_name))

            if callable(field_spec):
                predicates.append(field_spec(extracted))
            elif isinstance(field_spec, pl.Expr):
                predicates.append(field_spec)
            else:
                raise TypeError(
                    f"Invalid filter spec type for list element field '{field_name}': "
                    f"{type(field_spec)}. Expected Callable or pl.Expr when navigating "
                    "into a List<Struct> column."
                )

        if not predicates:
            return pl.lit(True)
        result = predicates[0]
        for pred in predicates[1:]:
            result = result & pred
        return result

    # -------------------------------------------------------------------------
    # Element-filter private helpers
    # -------------------------------------------------------------------------

    def _process_col_element_filter(
        self, col_name: str, spec: ElementFilterSpec
    ) -> pl.Expr:
        """Build a with_columns expression that filters elements within a list column."""
        if col_name not in self._schema:
            available = list(self._schema.names())
            raise ValueError(
                f"Column '{col_name}' not found in schema. "
                f"Available columns: {available}"
            )

        dtype = self._schema[col_name]
        if not isinstance(dtype, pl.List):
            raise ValueError(
                f"filter_nested_elements can only be applied to List columns, "
                f"but '{col_name}' has type {dtype}."
            )

        pred_expr = self._build_element_predicate(dtype.inner, spec, pl.element())
        return (
            pl.col(col_name)
            .list.eval(pl.when(pred_expr).then(pl.element()))
            .list.drop_nulls()
            .alias(col_name)
        )

    def _build_element_predicate(
        self,
        inner_dtype: PolarsDataType,
        spec: ElementFilterSpec,
        base_expr: pl.Expr,
    ) -> pl.Expr:
        """Build an element-level boolean predicate (used inside list.eval context)."""
        if callable(spec):
            return spec(base_expr)
        elif isinstance(spec, pl.Expr):
            return spec
        elif isinstance(spec, dict):
            if not isinstance(inner_dtype, pl.Struct):
                raise ValueError(
                    f"Cannot use a dict element filter spec on a List with element type "
                    f"{inner_dtype}. Dict specs require a Struct element type."
                )
            return self._build_element_struct_predicate(inner_dtype, spec, base_expr)
        else:
            raise TypeError(
                f"Invalid element filter spec type: {type(spec)}. "
                "Expected Callable, pl.Expr, or dict."
            )

    def _build_element_struct_predicate(
        self,
        struct_dtype: pl.Struct,
        spec: dict[str, ElementFilterSpec],
        base_expr: pl.Expr,
    ) -> pl.Expr:
        """AND-combine element-level predicates for struct fields (inside list.eval context)."""
        schema_map: dict[str, PolarsDataType] = {f.name: f.dtype for f in struct_dtype.fields}
        predicates: list[pl.Expr] = []

        for field_name, field_spec in spec.items():
            if field_name not in schema_map:
                available = list(schema_map.keys())
                raise ValueError(
                    f"Field '{field_name}' not found in struct element. "
                    f"Available fields: {available}"
                )
            field_expr = base_expr.struct.field(field_name)
            field_dtype = schema_map[field_name]

            if callable(field_spec):
                predicates.append(field_spec(field_expr))
            elif isinstance(field_spec, pl.Expr):
                predicates.append(field_spec)
            elif isinstance(field_spec, dict):
                if isinstance(field_dtype, pl.Struct):
                    predicates.append(
                        self._build_element_struct_predicate(field_dtype, field_spec, field_expr)
                    )
                else:
                    raise ValueError(
                        f"Cannot recurse into field '{field_name}' with type {field_dtype} "
                        "in element filter. Dict specs require a Struct field type."
                    )
            else:
                raise TypeError(
                    f"Invalid element filter spec type for field '{field_name}': "
                    f"{type(field_spec)}. Expected Callable, pl.Expr, or dict."
                )

        if not predicates:
            return pl.lit(True)
        result = predicates[0]
        for pred in predicates[1:]:
            result = result & pred
        return result


def generate_nested_filter_expr(
    filter_spec: dict[str, FilterValue],
    schema: pl.Schema | FrameT,
) -> pl.Expr:
    """
    Generate a boolean Polars expression for row-level filtering using a nested dict spec.

    The callable at each dict path endpoint receives the Polars expression **at that
    structural level**:

    - ``{"col": lambda x: x > 0}`` — x is ``pl.col("col")``
    - ``{"col": {"f": lambda x: x > 0}}`` on a Struct — x is ``pl.col("col").struct.field("f")``
    - ``{"col": {"f": lambda x: ...}}`` on a List<Struct> — x is
      ``pl.col("col").list.eval(pl.element().struct.field("f"))`` (a ``List<T>`` expression).
      Use any list aggregation: ``x.list.max() > 1``, ``x.list.eval(...).list.any()``, etc.

    Multiple top-level keys are AND-combined.

    Args:
        filter_spec: Dict mapping column names to predicates. Each value is:
            - ``Callable[[pl.Expr], pl.Expr]``: receives the expression at this level
            - ``pl.Expr``: used directly as a boolean expression
            - ``dict``: navigate deeper into a Struct or List<Struct> column
        schema: The DataFrame schema, or a DataFrame/LazyFrame to extract the schema from.

    Returns:
        A boolean ``pl.Expr`` for use with ``.filter()``.

    Raises:
        ValueError: If a column is missing or a dict spec is used on a non-nested type.
        TypeError: If a spec value has an unsupported type.

    Examples:
        >>> df = pl.DataFrame({
        ...     "city": ["A", "B"],
        ...     "population": [50_000, 200_000],
        ...     "streets": [
        ...         [{"name": "Main", "length_km": 2.0}],
        ...         [{"name": "Oak", "length_km": 0.4}],
        ...     ],
        ... })
        >>> pred = generate_nested_filter_expr(
        ...     {"population": lambda x: x > 100_000},
        ...     df.schema,
        ... )
        >>> df.filter(pred)
        >>> pred2 = generate_nested_filter_expr(
        ...     {"streets": {"length_km": lambda x: x.list.max() > 1}},
        ...     df.schema,
        ... )
        >>> df.filter(pred2)
    """
    if isinstance(schema, pl.DataFrame | pl.LazyFrame):
        schema = schema.collect_schema()

    builder = NestedFilterBuilder(schema)
    return builder.build_row_filter(filter_spec)


def filter_nested_elements(
    frame: FrameT,
    filter_spec: dict[str, ElementFilterSpec],
) -> FrameT:
    """
    Filter elements within list columns in-place without dropping any rows.

    The callable at each dict path endpoint receives the **element-level** Polars
    expression (inside ``list.eval`` context), and must return an element-level boolean.

    Args:
        frame: The DataFrame or LazyFrame to operate on.
        filter_spec: Dict mapping List column names to element predicates. Each value is:
            - ``Callable[[pl.Expr], pl.Expr]``: receives element-level expression
            - ``pl.Expr``: used directly as element-level boolean
            - ``dict``: navigate into struct fields of each list element

    Returns:
        DataFrame or LazyFrame with filtered list columns (same type as input).
        Rows with all elements filtered out will have an empty list, not null.

    Raises:
        ValueError: If a column is not a List type or is not found.
        TypeError: If a spec value has an unsupported type.

    Examples:
        >>> df = pl.DataFrame({
        ...     "streets": [
        ...         [{"name": "Main", "length_km": 2.0}, {"name": "Oak", "length_km": 0.4}],
        ...         [{"name": "Pine", "length_km": 0.3}],
        ...     ],
        ... })
        >>> result = filter_nested_elements(df, {
        ...     "streets": {"length_km": lambda x: x > 1},
        ... })
        >>> # First row: [{"name": "Main", "length_km": 2.0}]
        >>> # Second row: []
    """
    schema = frame.collect_schema()
    builder = NestedFilterBuilder(schema)
    exprs = builder.build_element_filters(filter_spec)
    return frame.with_columns(exprs)


def apply_nested_filter(
    frame: FrameT,
    filter_spec: dict[str, FilterValue],
) -> FrameT:
    """
    Apply a nested filter spec to a DataFrame or LazyFrame, keeping only matching rows.

    Convenience wrapper around ``generate_nested_filter_expr`` + ``.filter()``.

    Args:
        frame: The DataFrame or LazyFrame to filter.
        filter_spec: Dict mapping column names to predicates (see
            ``generate_nested_filter_expr`` for full spec).

    Returns:
        Filtered DataFrame or LazyFrame (same type as input).

    Examples:
        >>> result = apply_nested_filter(df, {
        ...     "population": lambda x: x > 100_000,
        ...     "streets": {"length_km": lambda x: x.list.max() > 1},
        ... })
    """
    pred = generate_nested_filter_expr(filter_spec, frame)
    return frame.filter(pred)


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

import polars as pl

# MEMORY_EFFICIENT_PARQUET_WRITING_ROW_GROUP_SIZE = 2048 * 2


def convert_polars_schema(
    schema: dict | pl.Schema | pl.LazyFrame | pl.DataFrame,
) -> dict:
    """
    Convert a Polars schema into a nested Python representation.

    Nested container types are unwrapped recursively so the result mirrors the
    shape of the data rather than Polars' dtype objects:

    - ``List(inner)``       → ``[converted_inner]`` (a 1-element **list**)
    - ``Array(inner, size)``→ ``(converted_inner, size)`` (a 2-**tuple**)
    - ``Struct(fields)``    → ``{field_name: converted_dtype, ...}``
    - anything else         → the ``pl.DataType`` object unchanged

    The list/tuple split is what distinguishes a variable-length ``List`` from a
    fixed-size ``Array`` and preserves the array's size.

    Args:
        schema: A ``pl.Schema``, a plain ``dict`` of name → dtype, or a
            DataFrame / LazyFrame whose schema is collected first.

    Returns:
        Dict mapping column name to its nested representation.

    Examples:
        >>> convert_polars_schema({"a": pl.List(pl.Int64)})
        {'a': [Int64]}
        >>> convert_polars_schema({"a": pl.Array(pl.Int64, 3)})
        {'a': (Int64, 3)}
        >>> convert_polars_schema({"a": pl.Struct({"x": pl.Int64})})
        {'a': {'x': Int64}}

    Note:
        **Changed behaviour:** ``Array`` used to render as ``[inner]``, identical
        to ``List``, silently dropping the size. It is now a ``(inner, size)``
        tuple. Code that matched on the old shape needs updating.
    """

    def convert_dtype(dtype: object) -> object:
        """Recursively convert a single Polars dtype."""
        # Array is checked first: it is a sibling of List, not a subclass, but
        # ordering the checks keeps the intent obvious.
        if isinstance(dtype, pl.Array):
            return (convert_dtype(dtype.inner), dtype.size)
        if isinstance(dtype, pl.List):
            return [convert_dtype(dtype.inner)]
        if isinstance(dtype, pl.Struct):
            return {field.name: convert_dtype(field.dtype) for field in dtype.fields}
        # Scalar / parameterised leaf types (Int64, Enum, Categorical, Datetime …)
        # are returned as-is.
        return dtype

    # if the schema is a lazyframe or dataframe, we collect the schema
    if isinstance(schema, (pl.LazyFrame, pl.DataFrame)):
        schema = schema.collect_schema()

    return {col: convert_dtype(dtype) for col, dtype in schema.items()}


def unnest_rename(df: pl.LazyFrame, col: str, separator: str = ".") -> pl.LazyFrame:

    unnest_expr = pl.col(col).name.prefix_fields(f"{col}{separator}")
    return df.with_columns(unnest_expr).unnest(col)


def unnest_all(df: pl.LazyFrame, separator=".") -> pl.LazyFrame:
    struct_cols = [k for k, v in df.collect_schema().items() if isinstance(v, pl.Struct)]

    if len(struct_cols) == 0:
        return df

    for struct_col in struct_cols:
        df = unnest_rename(df, struct_col, separator=separator)

    return unnest_all(df, separator=separator)

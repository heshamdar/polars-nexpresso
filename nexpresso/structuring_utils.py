import polars as pl

# MEMORY_EFFICIENT_PARQUET_WRITING_ROW_GROUP_SIZE = 2048 * 2


def convert_polars_schema(
    schema: dict | pl.Schema | pl.LazyFrame | pl.DataFrame,
) -> dict:
    """
    Convert a Polars schema (a dict mapping column names to dtype objects) into a nested
    dictionary representation using introspection.

    If the user provides a lazyframe or dataframe, the schema is collected first.
    """

    def convert_dtype(dtype) -> object:
        """
        Recursively convert a Polars dtype into a nested Python representation.

        For a list type, the inner type is retrieved via dtype.__dict__['inner'].
        For a struct type, iterate over dtype.__dict__['fields'] (each field has a name and dtype).
        For other types, return the string representation.
        """
        # Check if the dtype has a __dict__ attribute (most do)
        if hasattr(dtype, "__dict__"):
            attr = dtype.__dict__
            # List type: get the inner type and return as a one-element list
            if "inner" in attr:
                return [convert_dtype(attr["inner"])]
            # Struct type: iterate over the fields (each field has a name and dtype)
            elif "fields" in attr:
                return {field.name: convert_dtype(field.dtype) for field in attr["fields"]}
        # Fallback for basic types: return the string representation
        return dtype

    # if the schema is a lazyframe or dataframe, we collect the schema
    if isinstance(schema, (pl.LazyFrame, pl.DataFrame)):
        schema = schema.collect_schema()

    return {col: convert_dtype(dtype) for col, dtype in schema.items()}


def unnest_rename(df: pl.LazyFrame, col: str, separator: str = ".") -> pl.LazyFrame:

    unnest_expr = pl.col(col).name.prefix_fields(f"{col}{separator}")
    return df.with_columns(unnest_expr).unnest(col)


def unnest_all(df: pl.LazyFrame, separator=".") -> pl.LazyFrame:
    struct_cols = [k for k, v in df.collect_schema().items() if v.base_type() == pl.Struct]

    if len(struct_cols) == 0:
        return df

    for struct_col in struct_cols:
        df = unnest_rename(df, struct_col, separator=separator)

    return unnest_all(df, separator=separator)

"""
Test suite for polars-nexpresso functionality.

Tests cover:
- Creating new columns/fields with pl.col or pl.field
- Selecting fields within a struct
- Adding fields within a struct
- Editing fields with the same name
- Creating new fields with new names based on other fields
- Error handling and edge cases
- Arrays and nested lists
- LazyFrame support
"""

import polars as pl
import pytest

from nexpresso import NestedExpressionBuilder, apply_nested_operations, generate_nested_exprs
from tests.conftest import requires_arr_eval


def test_create_new_top_level_column():
    """Test creating a new top-level column using pl.col."""
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        }
    )

    fields = {
        "a": None,  # Keep existing
        "c": pl.col("a") + pl.col("b"),  # Create new column
    }

    exprs = generate_nested_exprs(fields, df.schema)
    result = df.select(exprs)

    assert "c" in result.columns
    assert result["c"].to_list() == [5, 7, 9]


def test_create_new_field_in_struct():
    """Test creating a new field within a struct using pl.field."""
    df = pl.DataFrame(
        {
            "struct_col": [
                {"x": 10, "y": 20},
                {"x": 11, "y": 21},
                {"x": 12, "y": 22},
            ]
        }
    )

    fields = {
        "struct_col": {
            "x": None,  # Keep existing
            "y": None,  # Keep existing
            "sum": pl.field("x") + pl.field("y"),  # New field based on other fields
            "double_x": pl.field("x") * 2,  # New field based on x
        }
    }

    exprs = generate_nested_exprs(fields, df.schema, struct_mode="with_fields")
    result = df.select(exprs)

    struct_val = result["struct_col"][0]
    assert "sum" in struct_val
    assert "double_x" in struct_val
    assert struct_val["sum"] == 30
    assert struct_val["double_x"] == 20


def test_select_fields_in_struct():
    """Test selecting only specific fields from a struct."""
    df = pl.DataFrame(
        {
            "struct_col": [
                {"x": 10, "y": 20, "z": 30},
                {"x": 11, "y": 21, "z": 31},
                {"x": 12, "y": 22, "z": 32},
            ]
        }
    )

    fields = {
        "struct_col": {
            "x": None,  # Select only x
            "z": None,  # Select only z (y is excluded)
        }
    }

    exprs = generate_nested_exprs(fields, df.schema, struct_mode="select")
    result = df.select(exprs)

    # Check that only x and z exist
    struct_val = result["struct_col"][0]
    assert "x" in struct_val
    assert "z" in struct_val
    assert "y" not in struct_val


def test_edit_field_same_name():
    """Test editing a field while keeping the same name."""
    df = pl.DataFrame(
        {
            "struct_col": [
                {"x": 10, "y": 20},
                {"x": 11, "y": 21},
                {"x": 12, "y": 22},
            ]
        }
    )

    fields = {
        "struct_col": {
            "x": lambda x: x * 2,  # Edit x by doubling it
            "y": None,  # Keep y as-is
        }
    }

    exprs = generate_nested_exprs(fields, df.schema, struct_mode="with_fields")
    result = df.select(exprs)

    # Check that x is doubled
    assert result["struct_col"][0]["x"] == 20
    assert result["struct_col"][1]["x"] == 22
    assert result["struct_col"][2]["x"] == 24


def test_create_new_field_based_on_existing():
    """Test creating a new field with a different name based on existing fields."""
    df = pl.DataFrame(
        {
            "struct_col": [
                {"price": 10.0, "quantity": 2},
                {"price": 15.0, "quantity": 3},
                {"price": 20.0, "quantity": 4},
            ]
        }
    )

    fields = {
        "struct_col": {
            "price": None,  # Keep existing
            "quantity": None,  # Keep existing
            "total": pl.field("price") * pl.field("quantity"),  # New field
            "discounted_price": pl.field("price") * 0.9,  # New field
        }
    }

    exprs = generate_nested_exprs(fields, df.schema, struct_mode="with_fields")
    result = df.select(exprs)

    # Check new fields
    first = result["struct_col"][0]
    assert "total" in first
    assert "discounted_price" in first
    assert first["total"] == 20.0
    assert first["discounted_price"] == 9.0


def test_nested_list_of_structs():
    """Test working with lists of structs.

    Note: pl.field() references the ORIGINAL struct fields, not transformed ones.
    This is the expected behavior - transformations apply to the field itself,
    but pl.field() always references the original value.
    """
    df = pl.DataFrame(
        {
            "items": [
                [{"value": 1, "count": 2}, {"value": 3, "count": 4}],
                [{"value": 5, "count": 6}, {"value": 7, "count": 8}],
            ]
        }
    )

    # pl.field() references the ORIGINAL struct fields
    fields = {
        "items": {
            "value": lambda x: x * 2,  # Transform value: 1 -> 2
            "count": None,  # Keep count as-is
            # pl.field("value") references the ORIGINAL value (1), not transformed (2)
            # So total = original_value (1) * count (2) = 2
            "total": pl.field("value") * pl.field("count"),  # New field based on original values
        }
    }

    exprs = generate_nested_exprs(fields, df.schema, struct_mode="with_fields")
    result = df.select(exprs)

    # Check first item of first list
    first_item = result["items"][0][0]
    assert first_item["value"] == 2  # Doubled from 1
    assert first_item["count"] == 2  # Unchanged
    # pl.field() references original value (1) * count (2) = 2
    assert first_item["total"] == 2  # original value (1) * count (2)


def test_deeply_nested_structure():
    """Test deeply nested structures (struct within struct)."""
    df = pl.DataFrame(
        {
            "outer": [
                {"inner": {"x": 1, "y": 2}, "z": 3},
                {"inner": {"x": 4, "y": 5}, "z": 6},
            ]
        }
    )

    fields = {
        "outer": {
            "inner": {
                "x": lambda x: x * 2,  # Edit inner.x
                "y": None,  # Keep inner.y
                "sum": pl.field("x") + pl.field("y"),  # New field in inner
            },
            "z": None,  # Keep outer.z
            "product": pl.field("inner").struct.field("x") * pl.field("z"),  # New field in outer
        }
    }

    exprs = generate_nested_exprs(fields, df.schema, struct_mode="with_fields")
    result = df.select(exprs)

    # Check nested structure
    first = result["outer"][0]
    assert first["inner"]["x"] == 2  # Doubled from 1
    assert first["inner"]["y"] == 2  # Unchanged
    # pl.field() references original values: original x (1) + y (2) = 3
    assert first["inner"]["sum"] == 3  # original x (1) + y (2)
    assert first["z"] == 3  # Unchanged
    # pl.field() references original values: original inner.x (1) * z (3) = 3
    assert first["product"] == 3  # original inner.x (1) * z (3)


def test_apply_nested_operations_convenience():
    """Test the convenience function apply_nested_operations."""
    df = pl.DataFrame(
        {
            "data": [
                {"a": 1, "b": 2},
                {"a": 3, "b": 4},
            ]
        }
    )

    # Using with_columns to add new fields
    result = apply_nested_operations(
        df,
        {"data": {"c": pl.field("a") + pl.field("b")}},
        struct_mode="with_fields",
        use_with_columns=True,
    )

    assert result["data"][0]["c"] == 3  # type: ignore[index]


def test_complex_real_world_example():
    """Test a complex real-world scenario."""
    df = pl.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "orders": [
                [
                    {"item": "apple", "price": 1.0, "quantity": 5},
                    {"item": "banana", "price": 0.5, "quantity": 10},
                ],
                [
                    {"item": "orange", "price": 1.5, "quantity": 3},
                ],
                [
                    {"item": "grape", "price": 2.0, "quantity": 2},
                    {"item": "apple", "price": 1.0, "quantity": 4},
                ],
            ],
            "profile": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
                {"name": "Charlie", "age": 35},
            ],
        }
    )

    fields = {
        "customer_id": None,  # Keep as-is
        "orders": {
            "item": None,  # Keep item name
            "price": None,  # Keep price
            "quantity": None,  # Keep quantity
            "subtotal": pl.field("price") * pl.field("quantity"),  # New: calculate subtotal
            "discounted_price": pl.field("price") * 0.9,  # New: apply 10% discount
        },
        "profile": {
            "name": None,  # Keep name
            "age": lambda x: x + 1,  # Edit: increment age (for next year)
            "is_senior": pl.field("age") >= 30,  # New: flag for senior customers
        },
    }

    exprs = generate_nested_exprs(fields, df.schema, struct_mode="with_fields")
    result = df.select(exprs)

    # Verify some calculations
    first_order = result["orders"][0][0]
    assert first_order["subtotal"] == 5.0  # 1.0 * 5
    assert first_order["discounted_price"] == 0.9  # 1.0 * 0.9

    first_profile = result["profile"][0]
    assert first_profile["age"] == 31  # 30 + 1
    assert first_profile["is_senior"] is True  # 31 >= 30


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_invalid_struct_mode():
    """Test that invalid struct_mode raises ValueError."""
    df = pl.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(ValueError, match="Invalid struct_mode"):
        generate_nested_exprs({"a": None}, df.schema, struct_mode="invalid_mode")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Invalid struct_mode"):
        NestedExpressionBuilder(df.schema, struct_mode="bad_mode")  # type: ignore[arg-type]


def test_create_column_without_expr():
    """Test that creating a column without pl.Expr raises ValueError."""
    df = pl.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(ValueError, match="Column 'new_col' not found in schema"):
        generate_nested_exprs({"new_col": None}, df.schema)

    with pytest.raises(ValueError, match="Column 'new_col' not found in schema"):
        generate_nested_exprs({"new_col": lambda x: x * 2}, df.schema)

    with pytest.raises(ValueError, match="Column 'new_col' not found in schema"):
        generate_nested_exprs({"new_col": {"x": None}}, df.schema)


def test_apply_function_to_nonexistent_field():
    """Test that applying function to non-existent field raises ValueError."""
    df = pl.DataFrame(
        {
            "struct_col": [
                {"x": 10, "y": 20},
                {"x": 11, "y": 21},
            ]
        }
    )

    with pytest.raises(ValueError, match="Cannot apply function to non-existent field 'z'"):
        generate_nested_exprs(
            {"struct_col": {"z": lambda x: x * 2}}, df.schema, struct_mode="with_fields"
        )


def test_recurse_into_nonexistent_struct_field():
    """Test that recursing into non-existent struct field raises ValueError."""
    df = pl.DataFrame(
        {
            "struct_col": [
                {"x": 10, "y": 20},
                {"x": 11, "y": 21},
            ]
        }
    )

    with pytest.raises(ValueError, match="Cannot recurse into non-existent struct field 'z'"):
        generate_nested_exprs(
            {"struct_col": {"z": {"inner": None}}}, df.schema, struct_mode="with_fields"
        )


def test_recurse_into_non_nested_type():
    """Test that recursing into non-nested type raises ValueError."""
    df = pl.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(
        ValueError,
        match="Cannot recurse into field with type Int64.*Only Struct, List, and Array types support nested operations",
    ):
        generate_nested_exprs({"a": {"inner": None}}, df.schema)


def test_invalid_field_specification_type():
    """Test that invalid field specification type raises TypeError."""
    df = pl.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(
        TypeError,
        match="Invalid field specification type for 'a'.*Expected None, dict, Callable, or pl.Expr",
    ):
        generate_nested_exprs({"a": "invalid"}, df.schema)  # type: ignore[arg-type]

    with pytest.raises(
        TypeError,
        match="Invalid field specification type for 'a'.*Expected None, dict, Callable, or pl.Expr",
    ):
        generate_nested_exprs({"a": 123}, df.schema)  # type: ignore[arg-type]

    df_struct = pl.DataFrame(
        {
            "struct_col": [
                {"x": 10},
                {"x": 11},
            ]
        }
    )

    with pytest.raises(TypeError, match="Invalid field specification for 'x'"):
        generate_nested_exprs(
            {"struct_col": {"x": "invalid"}}, df_struct.schema, struct_mode="with_fields"  # type: ignore[arg-type]
        )


# ============================================================================
# Edge Cases and Additional Functionality
# ============================================================================


def test_list_of_primitives():
    """Test working with lists of primitives (not structs)."""
    df = pl.DataFrame({"numbers": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]})

    # Can't recurse into primitives - this should raise an error
    with pytest.raises(ValueError, match="Cannot recurse into field with type"):
        generate_nested_exprs({"numbers": {"inner": None}}, df.schema)

    # But we can transform the list itself
    fields = {"numbers": lambda x: x.list.eval(pl.element() * 2)}
    exprs = generate_nested_exprs(fields, df.schema)
    result = df.select(exprs)

    assert result["numbers"][0].to_list() == [2, 4, 6]
    assert result["numbers"][1].to_list() == [8, 10, 12]


def test_nested_lists():
    """Test working with nested lists (lists of lists)."""
    df = pl.DataFrame(
        {
            "nested_lists": [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
            ]
        }
    )

    # Transform inner list elements
    fields = {"nested_lists": lambda x: x.list.eval(pl.element().list.eval(pl.element() * 2))}
    exprs = generate_nested_exprs(fields, df.schema)
    result = df.select(exprs)

    assert result["nested_lists"][0][0].to_list() == [2, 4]
    assert result["nested_lists"][0][1].to_list() == [6, 8]


def test_list_of_lists_of_structs():
    """Test working with lists of lists of structs."""
    df = pl.DataFrame(
        {
            "nested_items": [
                [
                    [{"value": 1}, {"value": 2}],
                    [{"value": 3}, {"value": 4}],
                ],
                [
                    [{"value": 5}, {"value": 6}],
                ],
            ]
        }
    )

    fields = {
        "nested_items": {
            "value": lambda x: x * 2,
        }
    }

    exprs = generate_nested_exprs(fields, df.schema, struct_mode="with_fields")
    result = df.select(exprs)

    assert result["nested_items"][0][0][0]["value"] == 2
    assert result["nested_items"][0][0][1]["value"] == 4
    assert result["nested_items"][0][1][0]["value"] == 6


@requires_arr_eval
def test_array_type():
    """Test that Array types work with arr.eval() in Polars.

    As of Polars 1.0+, Arrays support arr.eval() for element-wise operations.
    This test verifies that nested operations work correctly on Array types.
    """
    df = pl.DataFrame(
        {
            "items": [
                [{"value": 1, "count": 2}, {"value": 3, "count": 4}],
                [{"value": 5, "count": 6}, {"value": 7, "count": 8}],
            ]
        },
        schema={"items": pl.Array(pl.Struct({"value": pl.Int64, "count": pl.Int64}), 2)},
    )

    fields = {
        "items": {
            "value": lambda x: x * 2,
            "count": None,
        }
    }

    # Arrays now support arr.eval() in Polars 1.0+
    exprs = generate_nested_exprs(fields, df.schema, struct_mode="with_fields")
    result = df.select(exprs)

    # Verify the transformations worked
    assert result["items"][0][0]["value"] == 2  # 1 * 2
    assert result["items"][0][0]["count"] == 2  # unchanged
    assert result["items"][0][1]["value"] == 6  # 3 * 2
    assert result["items"][0][1]["count"] == 4  # unchanged
    assert result["items"][1][0]["value"] == 10  # 5 * 2
    assert result["items"][1][1]["value"] == 14  # 7 * 2


def test_empty_field_spec_select_mode():
    """Test empty field_spec dict in select mode.

    Note: Polars doesn't allow creating empty structs, so this will raise an error.
    This test documents this limitation.
    """
    df = pl.DataFrame(
        {
            "struct_col": [
                {"x": 10, "y": 20},
                {"x": 11, "y": 21},
            ]
        }
    )

    # Empty dict should raise an error because Polars doesn't allow empty structs
    # The error is raised during expression generation, not during selection
    fields = {"struct_col": {}}

    with pytest.raises(ValueError, match="expected at least 1 expression"):
        exprs = generate_nested_exprs(fields, df.schema, struct_mode="select")
        df.select(exprs)


def test_empty_field_spec_with_fields_mode():
    """Test empty field_spec dict in with_fields mode."""
    df = pl.DataFrame(
        {
            "struct_col": [
                {"x": 10, "y": 20},
                {"x": 11, "y": 21},
            ]
        }
    )

    # Empty dict should keep all fields unchanged
    fields = {"struct_col": {}}
    exprs = generate_nested_exprs(fields, df.schema, struct_mode="with_fields")
    result = df.select(exprs)

    struct_val = result["struct_col"][0]
    assert struct_val["x"] == 10
    assert struct_val["y"] == 20


def test_select_mode_with_transformations():
    """Test select mode with field transformations."""
    df = pl.DataFrame(
        {
            "struct_col": [
                {"x": 10, "y": 20, "z": 30},
                {"x": 11, "y": 21, "z": 31},
            ]
        }
    )

    # Transform x and select only x and y
    fields = {
        "struct_col": {
            "x": lambda x: x * 2,  # Transform x
            "y": None,  # Keep y as-is
            # z is excluded
        }
    }

    exprs = generate_nested_exprs(fields, df.schema, struct_mode="select")
    result = df.select(exprs)

    struct_val = result["struct_col"][0]
    assert struct_val["x"] == 20  # Doubled
    assert struct_val["y"] == 20  # Unchanged
    assert "z" not in struct_val  # Excluded


def test_multiple_top_level_columns():
    """Test multiple top-level columns with different operations."""
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "nested": [
                {"x": 10, "y": 20},
                {"x": 11, "y": 21},
                {"x": 12, "y": 22},
            ],
        }
    )

    fields = {
        "a": lambda x: x * 2,  # Transform a
        "b": None,  # Keep b as-is
        "nested": {"x": lambda x: x + 1, "y": None},  # Transform nested.x
        "new_col": pl.col("a") + pl.col("b"),  # Create new column
    }

    exprs = generate_nested_exprs(fields, df.schema, struct_mode="with_fields")
    result = df.select(exprs)

    assert result["a"].to_list() == [2, 4, 6]
    assert result["b"].to_list() == [4, 5, 6]
    assert result["new_col"].to_list() == [5, 7, 9]
    assert result["nested"][0]["x"] == 11
    assert result["nested"][0]["y"] == 20


def test_lazyframe_support():
    """Test that LazyFrame is supported."""
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "nested": [
                {"x": 10, "y": 20},
                {"x": 11, "y": 21},
                {"x": 12, "y": 22},
            ],
        }
    )

    lazy_df = df.lazy()

    result = apply_nested_operations(
        lazy_df,
        {"nested": {"x": lambda x: x * 2}},
        struct_mode="with_fields",
    )

    assert isinstance(result, pl.LazyFrame)

    collected = result.collect()
    assert collected["nested"][0]["x"] == 20
    assert collected["nested"][0]["y"] == 20


def test_nested_expression_builder_direct():
    """Test using NestedExpressionBuilder directly."""
    df = pl.DataFrame(
        {
            "struct_col": [
                {"x": 10, "y": 20},
                {"x": 11, "y": 21},
            ]
        }
    )

    builder = NestedExpressionBuilder(df.schema, struct_mode="with_fields")
    exprs = builder.build({"struct_col": {"x": lambda x: x * 2}})
    result = df.select(exprs)

    assert result["struct_col"][0]["x"] == 20
    assert result["struct_col"][0]["y"] == 20


def test_list_with_callable_transformation():
    """Test list transformation using callable at top level."""
    df = pl.DataFrame(
        {
            "items": [
                [{"value": 1}, {"value": 2}],
                [{"value": 3}, {"value": 4}],
            ]
        }
    )

    # Transform the entire list column
    fields = {"items": lambda x: x.list.reverse()}
    exprs = generate_nested_exprs(fields, df.schema)
    result = df.select(exprs)

    assert result["items"][0][0]["value"] == 2
    assert result["items"][0][1]["value"] == 1


def test_expr_with_alias():
    """Test that pl.Expr with alias works correctly."""
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        }
    )

    # Create new column using pl.Expr
    fields = {"c": (pl.col("a") + pl.col("b")).alias("sum")}
    exprs = generate_nested_exprs(fields, df.schema)
    result = df.select(exprs)

    assert result["c"].to_list() == [5, 7, 9]
    # Note: The alias in the expression is overridden by the column name


def test_struct_with_pl_expr_field():
    """Test using pl.Expr directly for struct field creation."""
    df = pl.DataFrame(
        {
            "struct_col": [
                {"x": 10, "y": 20},
                {"x": 11, "y": 21},
            ]
        }
    )

    fields = {
        "struct_col": {
            "x": None,
            "y": None,
            "sum": pl.field("x") + pl.field("y"),  # Direct pl.Expr
            "multiplied": (pl.field("x") * pl.field("y")).alias("product"),  # With alias
        }
    }

    exprs = generate_nested_exprs(fields, df.schema, struct_mode="with_fields")
    result = df.select(exprs)

    assert result["struct_col"][0]["sum"] == 30
    assert result["struct_col"][0]["multiplied"] == 200

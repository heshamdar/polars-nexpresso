"""
Comprehensive examples showcasing polars-nexpresso functionality.

This file demonstrates various use cases and features of the helper, including:
- Creating new columns and fields
- Working with nested structs and lists
- Selecting vs modifying fields (select vs with_fields modes)
- Transforming nested data structures
- Conditional transformations
- Real-world scenarios

Key Concepts:
- pl.field() references ORIGINAL struct fields, not transformed ones
- Transformations apply to the field itself (e.g., lambda x: x * 2)
- Operations don't accumulate - multiple references get the same original value
- Use struct_mode="select" to keep only specified fields
- Use struct_mode="with_fields" to keep all fields and add/modify some

To run: python examples.py
"""

import polars as pl

from nexpresso import apply_nested_operations, generate_nested_exprs

pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(100)
pl.Config.set_tbl_width_chars(100)
pl.Config.set_fmt_str_lengths(100)


def example_1_basic_column_operations():
    """Example 1: Basic column operations at the top level."""
    print("=" * 80)
    print("Example 1: Basic Column Operations")
    print("=" * 80)

    df = pl.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "salary": [50000, 60000, 70000],
        }
    )

    print("Original DataFrame:")
    print(df)
    print()

    # Keep existing columns, create new ones
    fields = {
        "name": None,  # Keep as-is
        "age": None,  # Keep as-is
        "salary": lambda x: x * 1.1,  # Apply 10% raise
        "bonus": pl.col("salary") * 0.1,  # Calculate bonus based on original salary
        "total_comp": pl.col("salary") * 1.1
        + pl.col("salary") * 0.1,  # Total compensation
    }

    exprs = generate_nested_exprs(fields, df.schema)
    result = df.select(exprs)

    print("After transformations:")
    print(result)
    print()


def example_2_nested_struct_operations():
    """Example 2: Working with nested structs - creating and modifying fields."""
    print("=" * 80)
    print("Example 2: Nested Struct Operations")
    print("=" * 80)

    df = pl.DataFrame(
        {
            "employee": [
                {"name": "Alice", "address": {"city": "NYC", "zip": "10001"}},
                {"name": "Bob", "address": {"city": "LA", "zip": "90001"}},
                {"name": "Charlie", "address": {"city": "Chicago", "zip": "60601"}},
            ]
        }
    )

    print("Original DataFrame:")
    print(df)
    print()

    # Modify nested fields and create new ones
    fields = {
        "employee": {
            "name": None,  # Keep name
            "address": {
                "city": None,  # Keep city
                "zip": None,  # Keep zip
                "state": pl.lit("Unknown"),  # Add new field
            },
            "full_address": pl.field("address").struct.field("city")
            + pl.lit(", ")
            + pl.field("address").struct.field("zip"),  # Create computed field
        }
    }

    exprs = generate_nested_exprs(fields, df.schema, struct_mode="with_fields")
    result = df.select(exprs)

    print("After adding nested fields:")
    print(result)
    print()


def example_3_select_mode():
    """Example 3: Using select mode to keep only specified fields."""
    print("=" * 80)
    print("Example 3: Select Mode - Keeping Only Specified Fields")
    print("=" * 80)

    df = pl.DataFrame(
        {
            "product": [
                {"name": "Widget", "price": 10.0, "cost": 5.0, "stock": 100},
                {"name": "Gadget", "price": 20.0, "cost": 8.0, "stock": 50},
                {"name": "Thing", "price": 15.0, "cost": 7.0, "stock": 200},
            ]
        }
    )

    print("Original DataFrame:")
    print(df)
    print()

    # Select mode: only keep specified fields
    fields = {
        "product": {
            "name": None,  # Keep name
            "price": lambda x: x * 1.2,  # Increase price by 20%
            # cost and stock are excluded - they won't appear in result
        }
    }

    exprs = generate_nested_exprs(fields, df.schema, struct_mode="select")
    result = df.select(exprs)

    print("After select (only name and modified price):")
    print(result)
    print()


def example_4_with_fields_mode():
    """Example 4: Using with_fields mode to add/modify while keeping all fields."""
    print("=" * 80)
    print("Example 4: With Fields Mode - Modifying While Keeping All Fields")
    print("=" * 80)

    df = pl.DataFrame(
        {
            "product": [
                {"name": "Widget", "price": 10.0, "cost": 5.0, "stock": 100},
                {"name": "Gadget", "price": 20.0, "cost": 8.0, "stock": 50},
                {"name": "Thing", "price": 15.0, "cost": 7.0, "stock": 200},
            ]
        }
    )

    print("Original DataFrame:")
    print(df)
    print()

    # With fields mode: keep all fields, add/modify some
    fields = {
        "product": {
            "price": lambda x: x * 1.2,  # Modify price
            "profit": pl.field("price") * 1.2
            - pl.field("cost"),  # New field using original values
            "margin": (pl.field("price") * 1.2 - pl.field("cost"))
            / (pl.field("price") * 1.2),  # New field
            # name, cost, stock are kept as-is
        }
    }

    exprs = generate_nested_exprs(fields, df.schema, struct_mode="with_fields")
    result = df.select(exprs)

    print("After with_fields (all fields kept, new ones added):")
    print(result)
    print()


def example_5_lists_of_structs():
    """Example 5: Working with lists of structs."""
    print("=" * 80)
    print("Example 5: Lists of Structs")
    print("=" * 80)

    df = pl.DataFrame(
        {
            "order_id": [1, 2, 3],
            "items": [
                [
                    {"product": "Apple", "quantity": 5, "price": 1.0},
                    {"product": "Banana", "quantity": 10, "price": 0.5},
                ],
                [
                    {"product": "Orange", "quantity": 3, "price": 1.5},
                ],
                [
                    {"product": "Grape", "quantity": 2, "price": 2.0},
                    {"product": "Apple", "quantity": 4, "price": 1.0},
                ],
            ],
        }
    )

    print("Original DataFrame:")
    print(df)
    print()

    # Transform items in the list
    fields = {
        "order_id": None,  # Keep order_id
        "items": {
            "product": None,  # Keep product name
            "quantity": lambda x: x * 2,  # Double quantity
            "price": None,  # Keep price
            # Note: pl.field() references original values, not transformed
            "subtotal": pl.field("quantity")
            * pl.field("price"),  # Original qty * price
            "discounted_price": pl.field("price") * 0.9,  # 10% discount
        },
    }

    exprs = generate_nested_exprs(fields, df.schema, struct_mode="with_fields")
    result = df.select(exprs)

    print("After transforming list items:")
    print(result)
    print()


def example_6_deeply_nested():
    """Example 6: Deeply nested structures."""
    print("=" * 80)
    print("Example 6: Deeply Nested Structures")
    print("=" * 80)

    df = pl.DataFrame(
        {
            "company": [
                {
                    "name": "TechCorp",
                    "departments": [
                        {
                            "name": "Engineering",
                            "employees": [
                                {"name": "Alice", "salary": 100000},
                                {"name": "Bob", "salary": 120000},
                            ],
                        },
                        {
                            "name": "Sales",
                            "employees": [
                                {"name": "Charlie", "salary": 80000},
                            ],
                        },
                    ],
                }
            ]
        }
    )

    print("Original DataFrame:")
    print(df)
    print()

    # Work with deeply nested structures
    fields = {
        "company": {
            "name": None,
            "departments": {
                "name": None,
                "employees": {
                    "name": None,
                    "salary": lambda x: x * 1.1,  # 10% raise
                    "bonus": pl.field("salary") * 0.1,  # Bonus based on original salary
                },
            },
        }
    }

    exprs = generate_nested_exprs(fields, df.schema, struct_mode="with_fields")
    result = df.select(exprs)

    print("After applying raises and calculating bonuses:")
    print(result)
    print()


def example_7_conditional_transformations():
    """Example 7: Conditional transformations using expressions."""
    print("=" * 80)
    print("Example 7: Conditional Transformations")
    print("=" * 80)

    df = pl.DataFrame(
        {
            "customer": [
                {"name": "Alice", "age": 25, "spending": 1000},
                {"name": "Bob", "age": 65, "spending": 500},
                {"name": "Charlie", "age": 30, "spending": 2000},
            ]
        }
    )

    print("Original DataFrame:")
    print(df)
    print()

    # Apply conditional logic
    fields = {
        "customer": {
            "name": None,
            "age": None,
            "spending": None,
            "discount": pl.when(pl.field("age") >= 65)
            .then(0.15)
            .when(pl.field("spending") > 1500)
            .then(0.10)
            .otherwise(0.0),  # Conditional discount
            "final_amount": pl.field("spending")
            * (
                1
                - pl.when(pl.field("age") >= 65)
                .then(0.15)
                .when(pl.field("spending") > 1500)
                .then(0.10)
                .otherwise(0.0)
            ),
            "customer_type": pl.when(pl.field("age") >= 65)
            .then(pl.lit("Senior"))
            .when(pl.field("spending") > 1500)
            .then(pl.lit("VIP"))
            .otherwise(pl.lit("Regular")),
        }
    }

    exprs = generate_nested_exprs(fields, df.schema, struct_mode="with_fields")
    result = df.select(exprs)

    print("After applying conditional logic:")
    print(result)
    print()


def example_8_convenience_function():
    """Example 8: Using the convenience function apply_nested_operations."""
    print("=" * 80)
    print("Example 8: Using apply_nested_operations Convenience Function")
    print("=" * 80)

    df = pl.DataFrame(
        {
            "data": [
                {"value": 10, "multiplier": 2},
                {"value": 20, "multiplier": 3},
                {"value": 30, "multiplier": 4},
            ]
        }
    )

    print("Original DataFrame:")
    print(df)
    print()

    # Using the convenience function with with_columns
    result = apply_nested_operations(
        df,
        {
            "data": {
                "value": lambda x: x * 2,
                "multiplier": None,
                "result": pl.field("value") * pl.field("multiplier"),
            }
        },
        struct_mode="with_fields",
        use_with_columns=True,  # Use with_columns instead of select
    )

    print("After applying operations (using with_columns):")
    print(result)
    print()

    # Using select mode
    result2 = apply_nested_operations(
        df,
        {
            "data": {
                "value": lambda x: x * 2,
                "result": pl.field("value") * pl.field("multiplier"),
            }
        },
        struct_mode="select",  # Only keep specified fields
    )

    print("After applying operations (select mode):")
    print(result2)
    print()


def example_9_complex_real_world():
    """Example 9: Complex real-world scenario."""
    print("=" * 80)
    print("Example 9: Complex Real-World Scenario")
    print("=" * 80)

    # E-commerce order processing
    df = pl.DataFrame(
        {
            "order_id": [1001, 1002, 1003],
            "customer": [
                {"id": 1, "name": "Alice", "tier": "Gold"},
                {"id": 2, "name": "Bob", "tier": "Silver"},
                {"id": 3, "name": "Charlie", "tier": "Gold"},
            ],
            "items": [
                [
                    {"sku": "A001", "name": "Laptop", "price": 1000.0, "qty": 1},
                    {"sku": "A002", "name": "Mouse", "price": 25.0, "qty": 2},
                ],
                [
                    {"sku": "B001", "name": "Keyboard", "price": 75.0, "qty": 1},
                ],
                [
                    {"sku": "A001", "name": "Laptop", "price": 1000.0, "qty": 2},
                    {"sku": "A003", "name": "Monitor", "price": 300.0, "qty": 1},
                ],
            ],
            "shipping": [
                {"method": "Express", "cost": 20.0},
                {"method": "Standard", "cost": 10.0},
                {"method": "Express", "cost": 20.0},
            ],
        }
    )

    print("Original DataFrame:")
    print(df)
    print()

    # Complex transformations
    fields = {
        "order_id": None,
        "customer": {
            "id": None,
            "name": None,
            "tier": None,
            "tier_discount": pl.when(pl.field("tier") == "Gold")
            .then(0.10)
            .when(pl.field("tier") == "Silver")
            .then(0.05)
            .otherwise(0.0),
        },
        "items": {
            "sku": None,
            "name": None,
            "price": None,
            "qty": None,
            "line_total": pl.field("price") * pl.field("qty"),
            "discounted_price": pl.field("price")
            * (1 - pl.when(pl.field("price") > 500).then(0.05).otherwise(0.0)),
        },
        "shipping": {
            "method": None,
            "cost": lambda x: x * 1.1,  # Add 10% handling fee
        },
    }

    exprs = generate_nested_exprs(fields, df.schema, struct_mode="with_fields")
    result = df.select(exprs)

    print("After complex transformations:")
    print(result)
    print()


def example_10_aggregations_in_nested():
    """Example 10: Using aggregations within nested structures."""
    print("=" * 80)
    print("Example 10: Aggregations in Nested Structures")
    print("=" * 80)

    df = pl.DataFrame(
        {
            "id": [1, 2],
            "team": [
                {
                    "name": "Team A",
                    "members": [
                        {"name": "Alice", "score": 85},
                        {"name": "Bob", "score": 90},
                        {"name": "Charlie", "score": 75},
                    ],
                },
                {
                    "name": "Team B",
                    "members": [
                        {"name": "Diana", "score": 95},
                        {"name": "Eve", "score": 88},
                    ],
                },
            ],
        }
    )

    print("Original DataFrame:")
    print(df)
    print()

    # Add aggregations
    fields = {
        "team": {
            "members": {
                "above_average": pl.field("score")
                > pl.field("score").mean(),  # Compare to list mean
            },
        }
    }

    exprs = generate_nested_exprs(fields, df.schema, struct_mode="with_fields")
    result = df.with_columns(exprs)

    print("After adding aggregations:")
    print(result)
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("POLARS-NEXPRESSO - COMPREHENSIVE EXAMPLES")
    print("=" * 80 + "\n")

    example_1_basic_column_operations()
    example_2_nested_struct_operations()
    example_3_select_mode()
    example_4_with_fields_mode()
    example_5_lists_of_structs()
    example_6_deeply_nested()
    example_7_conditional_transformations()
    example_8_convenience_function()
    example_9_complex_real_world()
    example_10_aggregations_in_nested()

    print("=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""Tests for structuring_utils module."""

import polars as pl

from nexpresso.structuring_utils import convert_polars_schema, unnest_all, unnest_rename


class TestUnnestRename:
    """Tests for the unnest_rename function."""

    def test_unnest_single_struct_column(self) -> None:
        """Test unnesting a single struct column with default dot notation."""
        df = pl.LazyFrame(
            {
                "id": [1, 2, 3],
                "person": [
                    {"name": "Alice", "age": 30},
                    {"name": "Bob", "age": 25},
                    {"name": "Charlie", "age": 35},
                ],
            }
        )

        result = unnest_rename(df, "person").collect()

        assert "person" not in result.columns
        assert "person.name" in result.columns
        assert "person.age" in result.columns
        assert result["person.name"].to_list() == ["Alice", "Bob", "Charlie"]
        assert result["person.age"].to_list() == [30, 25, 35]

    def test_unnest_with_custom_separator(self) -> None:
        """Test unnesting with a custom separator."""
        df = pl.LazyFrame(
            {
                "id": [1, 2],
                "person": [
                    {"name": "Alice", "age": 30},
                    {"name": "Bob", "age": 25},
                ],
            }
        )

        result = unnest_rename(df, "person", separator="_").collect()

        assert "person" not in result.columns
        assert "person_name" in result.columns
        assert "person_age" in result.columns
        assert result["person_name"].to_list() == ["Alice", "Bob"]
        assert result["person_age"].to_list() == [30, 25]

    def test_unnest_preserves_non_struct_columns(self) -> None:
        """Test that non-struct columns are preserved during unnesting."""
        df = pl.LazyFrame(
            {
                "id": [1, 2, 3],
                "value": [10, 20, 30],
                "person": [
                    {"name": "Alice", "age": 30},
                    {"name": "Bob", "age": 25},
                    {"name": "Charlie", "age": 35},
                ],
            }
        )

        result = unnest_rename(df, "person").collect()

        assert "id" in result.columns
        assert "value" in result.columns
        assert result["id"].to_list() == [1, 2, 3]
        assert result["value"].to_list() == [10, 20, 30]

    def test_unnest_with_nested_struct(self) -> None:
        """Test unnesting a struct that contains another struct."""
        df = pl.LazyFrame(
            {
                "id": [1, 2],
                "person": [
                    {
                        "name": "Alice",
                        "address": {"street": "123 Main", "city": "NYC"},
                    },
                    {
                        "name": "Bob",
                        "address": {"street": "456 Oak", "city": "LA"},
                    },
                ],
            }
        )

        result = unnest_rename(df, "person").collect()

        assert "person" not in result.columns
        assert "person.name" in result.columns
        assert "person.address" in result.columns
        # The nested struct should still be a struct after first unnest
        assert isinstance(result["person.address"].dtype, pl.Struct)
        assert result["person.name"].to_list() == ["Alice", "Bob"]


class TestUnnestAll:
    """Tests for the unnest_all function."""

    def test_unnest_all_single_level_struct(self) -> None:
        """Test unnesting all struct columns at a single level."""
        df = pl.LazyFrame(
            {
                "id": [1, 2, 3],
                "person": [
                    {"name": "Alice", "age": 30},
                    {"name": "Bob", "age": 25},
                    {"name": "Charlie", "age": 35},
                ],
            }
        )

        result = unnest_all(df).collect()

        assert "person" not in result.columns
        assert "person.name" in result.columns
        assert "person.age" in result.columns
        assert result["person.name"].to_list() == ["Alice", "Bob", "Charlie"]
        assert result["person.age"].to_list() == [30, 25, 35]

    def test_unnest_all_multiple_struct_columns(self) -> None:
        """Test unnesting multiple struct columns."""
        df = pl.LazyFrame(
            {
                "id": [1, 2],
                "person": [
                    {"name": "Alice", "age": 30},
                    {"name": "Bob", "age": 25},
                ],
                "company": [
                    {"name": "Acme", "revenue": 1000},
                    {"name": "Beta", "revenue": 2000},
                ],
            }
        )

        result = unnest_all(df).collect()

        assert "person" not in result.columns
        assert "company" not in result.columns
        assert "person.name" in result.columns
        assert "person.age" in result.columns
        assert "company.name" in result.columns
        assert "company.revenue" in result.columns
        assert result["person.name"].to_list() == ["Alice", "Bob"]
        assert result["company.name"].to_list() == ["Acme", "Beta"]

    def test_unnest_all_deeply_nested_structs(self) -> None:
        """Test unnesting structs of structs (multiple levels of nesting)."""
        df = pl.LazyFrame(
            {
                "id": [1, 2],
                "person": [
                    {
                        "name": "Alice",
                        "address": {
                            "street": "123 Main",
                            "location": {"city": "NYC", "zip": "10001"},
                        },
                    },
                    {
                        "name": "Bob",
                        "address": {
                            "street": "456 Oak",
                            "location": {"city": "LA", "zip": "90001"},
                        },
                    },
                ],
            }
        )

        result = unnest_all(df).collect()

        # All structs should be unnested
        assert "person" not in result.columns
        assert "person.address" not in result.columns
        assert "person.address.location" not in result.columns

        # Check that all nested fields are flattened
        assert "person.name" in result.columns
        assert "person.address.street" in result.columns
        assert "person.address.location.city" in result.columns
        assert "person.address.location.zip" in result.columns

        # Verify values
        assert result["person.name"].to_list() == ["Alice", "Bob"]
        assert result["person.address.street"].to_list() == ["123 Main", "456 Oak"]
        assert result["person.address.location.city"].to_list() == ["NYC", "LA"]
        assert result["person.address.location.zip"].to_list() == ["10001", "90001"]

    def test_unnest_all_very_deeply_nested_structs(self) -> None:
        """Test unnesting very deeply nested structs (4+ levels)."""
        df = pl.LazyFrame(
            {
                "id": [1],
                "level1": [
                    {
                        "field1": "a",
                        "level2": {
                            "field2": "b",
                            "level3": {
                                "field3": "c",
                                "level4": {
                                    "field4": "d",
                                    "level5": {"field5": "e"},
                                },
                            },
                        },
                    }
                ],
            }
        )

        result = unnest_all(df).collect()

        # Verify all levels are unnested
        assert "level1" not in result.columns
        assert "level1.level2" not in result.columns
        assert "level1.level2.level3" not in result.columns
        assert "level1.level2.level3.level4" not in result.columns
        assert "level1.level2.level3.level4.level5" not in result.columns

        # Check flattened fields
        assert "level1.field1" in result.columns
        assert "level1.level2.field2" in result.columns
        assert "level1.level2.level3.field3" in result.columns
        assert "level1.level2.level3.level4.field4" in result.columns
        assert "level1.level2.level3.level4.level5.field5" in result.columns

        # Verify values
        assert result["level1.field1"].to_list() == ["a"]
        assert result["level1.level2.field2"].to_list() == ["b"]
        assert result["level1.level2.level3.field3"].to_list() == ["c"]
        assert result["level1.level2.level3.level4.field4"].to_list() == ["d"]
        assert result["level1.level2.level3.level4.level5.field5"].to_list() == ["e"]

    def test_unnest_all_with_custom_separator(self) -> None:
        """Test unnesting all structs with a custom separator."""
        df = pl.LazyFrame(
            {
                "id": [1, 2],
                "person": [
                    {
                        "name": "Alice",
                        "address": {"street": "123 Main", "city": "NYC"},
                    },
                    {
                        "name": "Bob",
                        "address": {"street": "456 Oak", "city": "LA"},
                    },
                ],
            }
        )

        result = unnest_all(df, separator="_").collect()

        assert "person" not in result.columns
        assert "person_address" not in result.columns
        assert "person_name" in result.columns
        assert "person_address_street" in result.columns
        assert "person_address_city" in result.columns
        assert result["person_name"].to_list() == ["Alice", "Bob"]
        assert result["person_address_street"].to_list() == ["123 Main", "456 Oak"]

    def test_unnest_all_no_struct_columns(self) -> None:
        """Test that unnest_all returns the dataframe unchanged when there are no structs."""
        df = pl.LazyFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "age": [30, 25, 35],
            }
        )

        result = unnest_all(df).collect()

        # Should be unchanged
        assert result.columns == ["id", "name", "age"]
        assert result["id"].to_list() == [1, 2, 3]
        assert result["name"].to_list() == ["Alice", "Bob", "Charlie"]
        assert result["age"].to_list() == [30, 25, 35]

    def test_unnest_all_mixed_struct_and_non_struct(self) -> None:
        """Test unnesting when dataframe has both struct and non-struct columns."""
        df = pl.LazyFrame(
            {
                "id": [1, 2],
                "value": [10, 20],
                "person": [
                    {"name": "Alice", "age": 30},
                    {"name": "Bob", "age": 25},
                ],
                "status": ["active", "inactive"],
            }
        )

        result = unnest_all(df).collect()

        # Non-struct columns should be preserved
        assert "id" in result.columns
        assert "value" in result.columns
        assert "status" in result.columns

        # Struct columns should be unnested
        assert "person" not in result.columns
        assert "person.name" in result.columns
        assert "person.age" in result.columns

        # Verify values
        assert result["id"].to_list() == [1, 2]
        assert result["value"].to_list() == [10, 20]
        assert result["status"].to_list() == ["active", "inactive"]
        assert result["person.name"].to_list() == ["Alice", "Bob"]

    def test_unnest_all_preserves_row_order(self) -> None:
        """Test that row order is preserved during unnesting."""
        df = pl.LazyFrame(
            {
                "id": [3, 1, 2],
                "person": [
                    {"name": "Charlie", "age": 35},
                    {"name": "Alice", "age": 30},
                    {"name": "Bob", "age": 25},
                ],
            }
        )

        result = unnest_all(df).collect()

        # Order should be preserved
        assert result["id"].to_list() == [3, 1, 2]
        assert result["person.name"].to_list() == ["Charlie", "Alice", "Bob"]
        assert result["person.age"].to_list() == [35, 30, 25]

    def test_unnest_all_with_null_values(self) -> None:
        """Test unnesting structs that contain null values."""
        df = pl.LazyFrame(
            {
                "id": [1, 2, 3],
                "person": [
                    {"name": "Alice", "age": 30},
                    {"name": None, "age": None},
                    {"name": "Charlie", "age": 35},
                ],
            }
        )

        result = unnest_all(df).collect()

        assert "person.name" in result.columns
        assert "person.age" in result.columns
        assert result["person.name"].to_list() == ["Alice", None, "Charlie"]
        assert result["person.age"].to_list() == [30, None, 35]

    def test_unnest_all_with_empty_struct(self) -> None:
        """Test unnesting an empty struct (struct with no fields)."""
        # Create an empty struct - this is a bit tricky in Polars
        # We'll create a struct with fields and then test edge cases
        df = pl.LazyFrame(
            {
                "id": [1],
                "person": [{"name": "Alice"}],
            }
        )

        result = unnest_all(df).collect()

        assert "person.name" in result.columns
        assert result["person.name"].to_list() == ["Alice"]

    def test_unnest_all_complex_mixed_nesting(self) -> None:
        """Test unnesting a complex structure with multiple nested structs at different levels."""
        df = pl.LazyFrame(
            {
                "id": [1, 2],
                "user": [
                    {
                        "name": "Alice",
                        "profile": {"bio": "Developer", "settings": {"theme": "dark"}},
                        "contact": {"email": "alice@example.com"},
                    },
                    {
                        "name": "Bob",
                        "profile": {"bio": "Designer", "settings": {"theme": "light"}},
                        "contact": {"email": "bob@example.com"},
                    },
                ],
            }
        )

        result = unnest_all(df).collect()

        # All structs should be unnested
        assert "user" not in result.columns
        assert "user.profile" not in result.columns
        assert "user.profile.settings" not in result.columns
        assert "user.contact" not in result.columns

        # Check flattened fields
        assert "user.name" in result.columns
        assert "user.profile.bio" in result.columns
        assert "user.profile.settings.theme" in result.columns
        assert "user.contact.email" in result.columns

        # Verify values
        assert result["user.name"].to_list() == ["Alice", "Bob"]
        assert result["user.profile.bio"].to_list() == ["Developer", "Designer"]
        assert result["user.profile.settings.theme"].to_list() == ["dark", "light"]
        assert result["user.contact.email"].to_list() == [
            "alice@example.com",
            "bob@example.com",
        ]

    def test_unnest_all_returns_lazyframe(self) -> None:
        """Test that unnest_all returns a LazyFrame."""
        df = pl.LazyFrame(
            {
                "id": [1, 2],
                "person": [
                    {"name": "Alice", "age": 30},
                    {"name": "Bob", "age": 25},
                ],
            }
        )

        result = unnest_all(df)

        assert isinstance(result, pl.LazyFrame)
        assert not isinstance(result, pl.DataFrame)

    def test_unnest_all_with_single_row(self) -> None:
        """Test unnesting with a single row."""
        df = pl.LazyFrame(
            {
                "id": [1],
                "person": [{"name": "Alice", "age": 30}],
            }
        )

        result = unnest_all(df).collect()

        assert result.height == 1
        assert result["person.name"].to_list() == ["Alice"]
        assert result["person.age"].to_list() == [30]


class TestConvertPolarsSchema:
    """Tests for the convert_polars_schema function."""

    def test_scalar_types_pass_through(self) -> None:
        """Non-nested dtypes are returned unchanged, as dtype objects."""
        result = convert_polars_schema({"a": pl.Int64, "b": pl.String, "c": pl.Boolean})

        assert result == {"a": pl.Int64, "b": pl.String, "c": pl.Boolean}

    def test_list_becomes_single_element_list(self) -> None:
        """List(inner) is rendered as [inner]."""
        assert convert_polars_schema({"a": pl.List(pl.Int64)}) == {"a": [pl.Int64]}

    def test_nested_list(self) -> None:
        """Nesting is applied recursively."""
        assert convert_polars_schema({"a": pl.List(pl.List(pl.Int64))}) == {"a": [[pl.Int64]]}

    def test_list_of_struct(self) -> None:
        """A struct inside a list is expanded into a dict."""
        schema = {"items": pl.List(pl.Struct({"name": pl.String, "qty": pl.Int64}))}

        assert convert_polars_schema(schema) == {"items": [{"name": pl.String, "qty": pl.Int64}]}

    def test_struct_with_mixed_fields(self) -> None:
        """Struct fields are converted recursively."""
        schema = {
            "data": pl.Struct(
                {
                    "id": pl.Int64,
                    "tags": pl.List(pl.String),
                    "meta": pl.Struct({"score": pl.Float64}),
                }
            )
        }

        assert convert_polars_schema(schema) == {
            "data": {
                "id": pl.Int64,
                "tags": [pl.String],
                "meta": {"score": pl.Float64},
            }
        }

    def test_array_preserves_size(self) -> None:
        """Array(inner, size) is a (inner, size) tuple, not a list."""
        result = convert_polars_schema({"a": pl.Array(pl.Int64, 3)})

        assert result == {"a": (pl.Int64, 3)}

    def test_array_is_distinguishable_from_list(self) -> None:
        """The list/tuple split is what separates List from Array."""
        result = convert_polars_schema({"lst": pl.List(pl.Int64), "arr": pl.Array(pl.Int64, 3)})

        assert isinstance(result["lst"], list)
        assert isinstance(result["arr"], tuple)
        assert result["lst"] != result["arr"]

    def test_nested_array(self) -> None:
        """Array nesting keeps each level's size."""
        schema = {"a": pl.Array(pl.Array(pl.Int64, 2), 3)}

        assert convert_polars_schema(schema) == {"a": ((pl.Int64, 2), 3)}

    def test_array_of_struct(self) -> None:
        """A struct inside an array is expanded, size retained."""
        schema = {"a": pl.Array(pl.Struct({"x": pl.Int64}), 2)}

        assert convert_polars_schema(schema) == {"a": ({"x": pl.Int64}, 2)}

    def test_enum_and_categorical_pass_through(self) -> None:
        """Parameterised leaf types are returned as-is, not unwrapped."""
        enum_dtype = pl.Enum(["x", "y"])
        result = convert_polars_schema({"e": enum_dtype, "c": pl.Categorical()})

        assert result["e"] == enum_dtype
        assert result["c"] == pl.Categorical()

    def test_accepts_dataframe(self) -> None:
        """A DataFrame's schema is collected automatically."""
        df = pl.DataFrame({"id": [1], "items": [[{"name": "a"}]]})

        assert convert_polars_schema(df) == {"id": pl.Int64, "items": [{"name": pl.String}]}

    def test_accepts_lazyframe(self) -> None:
        """A LazyFrame's schema is collected automatically."""
        lf = pl.LazyFrame({"id": [1], "items": [[{"name": "a"}]]})

        assert convert_polars_schema(lf) == {"id": pl.Int64, "items": [{"name": pl.String}]}

    def test_accepts_polars_schema_object(self) -> None:
        """A pl.Schema instance works the same as a plain dict."""
        schema = pl.Schema({"a": pl.List(pl.Int64), "b": pl.Int64})

        assert convert_polars_schema(schema) == {"a": [pl.Int64], "b": pl.Int64}

    def test_empty_schema(self) -> None:
        """An empty schema converts to an empty dict."""
        assert convert_polars_schema({}) == {}

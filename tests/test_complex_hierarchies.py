"""
Tests for complex hierarchy scenarios.

This module tests edge cases and complex scenarios including:
- Non-hierarchy columns (list-of-struct, struct, primitive at root and nested levels)
- Deeply nested hierarchies (5+ levels)
- Null handling edge cases
- Name collision scenarios
- Mixed nested structures
- Empty and boundary cases
- build_from_tables edge cases
"""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from nexpresso import (
    HierarchicalPacker,
    HierarchySpec,
    HierarchyValidationError,
    LevelSpec,
    apply_nested_operations,
)

# =============================================================================
# Non-Hierarchy Columns Tests
# =============================================================================


class TestNonHierarchyColumns:
    """Tests for handling columns that are not part of the defined hierarchy."""

    def test_root_level_struct_column_preserved(self) -> None:
        """Test that a struct column at root level is preserved during pack."""
        spec = HierarchySpec(
            levels=[
                LevelSpec(name="parent", id_fields=["id"]),
                LevelSpec(name="child", id_fields=["id"]),
            ]
        )
        packer = HierarchicalPacker(spec, validate_on_pack=False)

        df = pl.DataFrame(
            {
                "parent.id": ["p1", "p1"],
                "parent.child.id": ["c1", "c2"],
                "parent.child.value": [10, 20],
                "metadata": [{"source": "api", "version": 1}, {"source": "api", "version": 1}],
            }
        )

        # Default behavior: preserve extra columns
        packed = packer.pack(df, "parent")

        # metadata should be preserved
        assert "metadata" in packed.columns
        # Should have 1 row (packed to parent level)
        assert packed.height == 1
        assert packed["metadata"][0]["source"] == "api"

    def test_root_level_list_of_struct_preserved(self) -> None:
        """Test that a list-of-struct column at root level is preserved."""
        spec = HierarchySpec(
            levels=[
                LevelSpec(name="parent", id_fields=["id"]),
                LevelSpec(name="child", id_fields=["id"]),
            ]
        )
        packer = HierarchicalPacker(spec, validate_on_pack=False)

        df = pl.DataFrame(
            {
                "parent.id": ["p1", "p1"],
                "parent.child.id": ["c1", "c2"],
                "tags": [
                    [{"name": "retail"}, {"name": "urban"}],
                    [{"name": "retail"}, {"name": "urban"}],  # Same as above for uniformity
                ],
            }
        )

        packed = packer.pack(df, "parent")
        assert "tags" in packed.columns
        assert packed.height == 1

    def test_extra_columns_error_mode(self) -> None:
        """Test that extra_columns='error' raises when extra columns exist."""
        spec = HierarchySpec(
            levels=[
                LevelSpec(name="parent", id_fields=["id"]),
            ]
        )
        packer = HierarchicalPacker(spec)

        df = pl.DataFrame(
            {
                "parent.id": ["p1"],
                "extra_col": ["value"],
            }
        )

        with pytest.raises(HierarchyValidationError, match="not part of the hierarchy"):
            packer.pack(df, "parent", extra_columns="error")

    def test_extra_columns_drop_mode(self) -> None:
        """Test that extra_columns='drop' removes extra columns silently."""
        spec = HierarchySpec(
            levels=[
                LevelSpec(name="parent", id_fields=["id"]),
            ]
        )
        packer = HierarchicalPacker(spec)

        df = pl.DataFrame(
            {
                "parent.id": ["p1"],
                "parent.name": ["Parent 1"],
                "extra_col": ["value"],
            }
        )

        packed = packer.pack(df, "parent", extra_columns="drop")
        assert "extra_col" not in packed.columns
        assert "parent" in packed.columns

    def test_extra_columns_preserve_mode_with_uniform_values(self) -> None:
        """Test that extra_columns='preserve' keeps uniform extra columns."""
        spec = HierarchySpec(
            levels=[
                LevelSpec(name="parent", id_fields=["id"]),
                LevelSpec(name="child", id_fields=["id"]),
            ]
        )
        packer = HierarchicalPacker(spec, validate_on_pack=True)

        df = pl.DataFrame(
            {
                "parent.id": ["p1", "p1"],
                "parent.child.id": ["c1", "c2"],
                "version": [1, 1],  # Uniform - should be preserved
            }
        )

        packed = packer.pack(df, "parent", extra_columns="preserve")
        assert "version" in packed.columns
        assert packed["version"][0] == 1

    def test_nested_list_of_struct_within_hierarchy_level_preserved(self) -> None:
        """Test that list-of-struct columns within a hierarchy level are preserved."""
        spec = HierarchySpec(
            levels=[
                LevelSpec(name="store", id_fields=["id"]),
                LevelSpec(name="product", id_fields=["sku"]),
            ]
        )
        packer = HierarchicalPacker(spec)

        # store.contacts is a list-of-struct, but NOT part of the hierarchy
        # It's an attribute of the store level
        df = pl.DataFrame(
            {
                "store.id": ["s1", "s1"],
                "store.name": ["Store 1", "Store 1"],
                "store.contacts": [
                    [{"name": "John", "phone": "123"}],
                    [{"name": "John", "phone": "123"}],  # Uniform
                ],
                "store.product.sku": ["SKU001", "SKU002"],
                "store.product.price": [10.0, 20.0],
            }
        )

        packed = packer.pack(df, "store")
        unpacked = packer.unpack(packed, "product")

        # contacts should still be present
        assert "store.contacts" in unpacked.columns

    def test_no_extra_columns_no_error(self) -> None:
        """Test that pack works when there are no extra columns."""
        spec = HierarchySpec(
            levels=[
                LevelSpec(name="parent", id_fields=["id"]),
            ]
        )
        packer = HierarchicalPacker(spec)

        df = pl.DataFrame(
            {
                "parent.id": ["p1"],
                "parent.name": ["Parent 1"],
            }
        )

        # Should not raise even with extra_columns="error"
        packed = packer.pack(df, "parent", extra_columns="error")
        assert "parent" in packed.columns


# =============================================================================
# Deeply Nested Hierarchies Tests
# =============================================================================


class TestDeeplyNestedHierarchies:
    """Tests for hierarchies with 5+ levels."""

    @pytest.fixture
    def six_level_spec(self) -> HierarchySpec:
        """Create a 6-level hierarchy specification."""
        return HierarchySpec(
            levels=[
                LevelSpec(name="continent", id_fields=["code"]),
                LevelSpec(name="country", id_fields=["code"]),
                LevelSpec(name="region", id_fields=["id"]),
                LevelSpec(name="city", id_fields=["id"]),
                LevelSpec(name="district", id_fields=["id"]),
                LevelSpec(name="street", id_fields=["name"]),
            ]
        )

    @pytest.fixture
    def six_level_data(self) -> pl.DataFrame:
        """Create test data for 6-level hierarchy."""
        return pl.DataFrame(
            {
                "continent.code": ["NA", "NA", "NA"],
                "continent.name": ["North America", "North America", "North America"],
                "continent.country.code": ["US", "US", "US"],
                "continent.country.name": ["United States", "United States", "United States"],
                "continent.country.region.id": ["west", "west", "east"],
                "continent.country.region.name": ["West Coast", "West Coast", "East Coast"],
                "continent.country.region.city.id": ["LA", "LA", "NYC"],
                "continent.country.region.city.name": [
                    "Los Angeles",
                    "Los Angeles",
                    "New York",
                ],
                "continent.country.region.city.district.id": ["hollywood", "dtla", "manhattan"],
                "continent.country.region.city.district.name": [
                    "Hollywood",
                    "Downtown LA",
                    "Manhattan",
                ],
                "continent.country.region.city.district.street.name": [
                    "Sunset Blvd",
                    "Main St",
                    "Broadway",
                ],
                "continent.country.region.city.district.street.length_km": [10.0, 5.0, 21.0],
            }
        )

    def test_pack_to_root_level(
        self, six_level_spec: HierarchySpec, six_level_data: pl.DataFrame
    ) -> None:
        """Test packing a 6-level hierarchy to the root."""
        packer = HierarchicalPacker(six_level_spec)
        packed = packer.pack(six_level_data, "continent")

        assert packed.height == 1
        assert "continent" in packed.columns

    def test_pack_unpack_roundtrip_deep(
        self, six_level_spec: HierarchySpec, six_level_data: pl.DataFrame
    ) -> None:
        """Test pack/unpack roundtrip for deep hierarchy."""
        packer = HierarchicalPacker(six_level_spec)

        packed = packer.pack(six_level_data, "continent")
        unpacked = packer.unpack(packed, "street")

        assert_frame_equal(unpacked, six_level_data)

    def test_partial_pack_unpack(
        self, six_level_spec: HierarchySpec, six_level_data: pl.DataFrame
    ) -> None:
        """Test packing to middle level and unpacking to deeper level."""
        packer = HierarchicalPacker(six_level_spec)

        # Pack to region level (level 2) - aggregates to country level
        # Since there's only 1 country (US), this gives 1 row with regions nested
        region_level = packer.pack(six_level_data, "region")
        assert region_level.height == 1  # 1 country with nested regions
        assert "continent.country.region" in region_level.columns

        # Unpack to region level to get 2 rows (west, east)
        unpacked_regions = packer.unpack(region_level, "region")
        assert unpacked_regions.height == 2  # 2 unique regions

        # Unpack back to street level
        street_level = packer.unpack(region_level, "street")
        assert street_level.height == 3  # Original 3 streets

        # Column order may differ after pack/unpack, but data should be the same
        assert_frame_equal(
            street_level.select(sorted(street_level.columns)),
            six_level_data.select(sorted(six_level_data.columns)),
        )


# =============================================================================
# Null Handling Edge Cases
# =============================================================================


class TestNullHandling:
    """Tests for null value handling in hierarchies."""

    def test_null_child_entity_preserved(self) -> None:
        """Test that a parent with no children results in null/empty list."""
        spec = HierarchySpec(
            levels=[
                LevelSpec(name="parent", id_fields=["id"]),
                LevelSpec(name="child", id_fields=["id"]),
            ]
        )
        packer = HierarchicalPacker(spec)

        # Parent p2 has null child
        df = pl.DataFrame(
            {
                "parent.id": ["p1", "p2"],
                "parent.name": ["Parent 1", "Parent 2"],
                "parent.child.id": ["c1", None],
                "parent.child.value": [10, None],
            }
        )

        packed = packer.pack(df, "parent")
        assert packed.height == 2

        # Unpack and verify nulls are preserved
        unpacked = packer.unpack(packed, "child")
        null_child = unpacked.filter(pl.col("parent.id") == "p2")
        assert null_child["parent.child.id"][0] is None

    def test_all_children_null_for_parent(self) -> None:
        """Test parent where all potential children are null."""
        spec = HierarchySpec(
            levels=[
                LevelSpec(name="parent", id_fields=["id"]),
                LevelSpec(name="child", id_fields=["id"]),
            ]
        )
        packer = HierarchicalPacker(spec)

        df = pl.DataFrame(
            {
                "parent.id": ["p1", "p1"],
                "parent.name": ["Parent 1", "Parent 1"],
                "parent.child.id": [None, None],
                "parent.child.value": [None, None],
            }
        )

        packed = packer.pack(df, "parent")
        # Should still work, just with null children
        assert packed.height == 1


# =============================================================================
# Name Collision Scenarios
# =============================================================================


class TestNameCollisions:
    """Tests for field name collisions across levels."""

    def test_same_field_name_different_levels(self) -> None:
        """Test that 'name' field at different levels doesn't collide."""
        spec = HierarchySpec(
            levels=[
                LevelSpec(name="company", id_fields=["id"]),
                LevelSpec(name="department", id_fields=["id"]),
                LevelSpec(name="employee", id_fields=["id"]),
            ]
        )
        packer = HierarchicalPacker(spec)

        df = pl.DataFrame(
            {
                "company.id": ["c1", "c1"],
                "company.name": ["Acme Corp", "Acme Corp"],  # company.name
                "company.department.id": ["d1", "d1"],
                "company.department.name": ["Engineering", "Engineering"],  # department.name
                "company.department.employee.id": ["e1", "e2"],
                "company.department.employee.name": ["Alice", "Bob"],  # employee.name
            }
        )

        packed = packer.pack(df, "company")
        unpacked = packer.unpack(packed, "employee")

        # All 'name' fields should be preserved correctly
        assert unpacked["company.name"][0] == "Acme Corp"
        assert unpacked["company.department.name"][0] == "Engineering"
        assert set(unpacked["company.department.employee.name"].to_list()) == {"Alice", "Bob"}


# =============================================================================
# Mixed Nested Structures
# =============================================================================


class TestMixedNestedStructures:
    """Tests for structures with both hierarchy and non-hierarchy nesting."""

    def test_hierarchy_with_nested_list_attribute(self) -> None:
        """Test hierarchy level that has a list attribute (not a child level)."""
        spec = HierarchySpec(
            levels=[
                LevelSpec(name="order", id_fields=["id"]),
                LevelSpec(name="item", id_fields=["sku"]),
            ]
        )
        packer = HierarchicalPacker(spec)

        df = pl.DataFrame(
            {
                "order.id": ["o1", "o1"],
                "order.customer_tags": [["vip", "repeat"], ["vip", "repeat"]],  # List, not struct
                "order.item.sku": ["SKU001", "SKU002"],
                "order.item.qty": [2, 3],
            }
        )

        packed = packer.pack(df, "order")
        assert packed.height == 1
        assert "order" in packed.columns

        unpacked = packer.unpack(packed, "item")
        assert "order.customer_tags" in unpacked.columns

    def test_apply_nested_operations_on_packed_hierarchy(self) -> None:
        """Test using apply_nested_operations on packed hierarchical data."""
        spec = HierarchySpec(
            levels=[
                LevelSpec(name="store", id_fields=["id"]),
                LevelSpec(name="product", id_fields=["sku"]),
            ]
        )
        packer = HierarchicalPacker(spec)

        df = pl.DataFrame(
            {
                "store.id": ["s1", "s1", "s2"],
                "store.name": ["Store 1", "Store 1", "Store 2"],
                "store.product.sku": ["A", "B", "C"],
                "store.product.price": [10.0, 20.0, 15.0],
                "store.product.cost": [5.0, 10.0, 8.0],
            }
        )

        # Pack to store level
        packed = packer.pack(df, "store")

        # Apply transformations on the packed data
        fields = {
            "store": {
                "id": None,
                "name": None,
                "product": {
                    "sku": None,
                    "price": None,
                    "cost": None,
                    "profit": pl.field("price") - pl.field("cost"),
                },
            }
        }

        result = apply_nested_operations(packed, fields, struct_mode="with_fields")

        # Unpack to verify
        unpacked = packer.unpack(result, "product")
        assert "store.product.profit" in unpacked.columns
        assert unpacked["store.product.profit"][0] == 5.0  # 10 - 5


# =============================================================================
# Empty and Boundary Cases
# =============================================================================


class TestBoundaryCases:
    """Tests for empty DataFrames and boundary conditions."""

    def test_single_row_dataframe(self) -> None:
        """Test hierarchy operations on single-row DataFrame."""
        spec = HierarchySpec(
            levels=[
                LevelSpec(name="parent", id_fields=["id"]),
                LevelSpec(name="child", id_fields=["id"]),
            ]
        )
        packer = HierarchicalPacker(spec)

        df = pl.DataFrame(
            {
                "parent.id": ["p1"],
                "parent.child.id": ["c1"],
                "parent.child.value": [100],
            }
        )

        packed = packer.pack(df, "parent")
        assert packed.height == 1

        unpacked = packer.unpack(packed, "child")
        assert unpacked.height == 1
        assert_frame_equal(unpacked, df)

    def test_root_level_only(self) -> None:
        """Test DataFrame with only root level columns (no children)."""
        spec = HierarchySpec(
            levels=[
                LevelSpec(name="entity", id_fields=["id"]),
            ]
        )
        packer = HierarchicalPacker(spec)

        df = pl.DataFrame(
            {
                "entity.id": ["e1", "e2"],
                "entity.name": ["Entity 1", "Entity 2"],
            }
        )

        packed = packer.pack(df, "entity")
        assert packed.height == 2
        assert "entity" in packed.columns

    def test_wide_hierarchy_many_columns_per_level(self) -> None:
        """Test hierarchy with many columns per level."""
        spec = HierarchySpec(
            levels=[
                LevelSpec(name="parent", id_fields=["id"]),
                LevelSpec(name="child", id_fields=["id"]),
            ]
        )
        packer = HierarchicalPacker(spec)

        # Create data with many columns per level
        data = {
            "parent.id": ["p1", "p1"],
            "parent.child.id": ["c1", "c2"],
        }
        # Add 20 parent-level columns
        for i in range(20):
            data[f"parent.attr_{i}"] = [f"val_{i}", f"val_{i}"]
        # Add 20 child-level columns
        for i in range(20):
            data[f"parent.child.field_{i}"] = [i, i + 1]

        df = pl.DataFrame(data)

        packed = packer.pack(df, "parent")
        unpacked = packer.unpack(packed, "child")

        assert unpacked.width == df.width
        assert unpacked.height == df.height


# =============================================================================
# build_from_tables Edge Cases
# =============================================================================


class TestBuildFromTablesEdgeCases:
    """Tests for edge cases in build_from_tables."""

    def test_tables_with_extra_columns(self) -> None:
        """Test that extra columns in tables are preserved."""
        spec = HierarchySpec.from_levels(
            LevelSpec(name="parent", id_fields=["id"]),
            LevelSpec(name="child", id_fields=["id"], parent_keys=["parent_id"]),
        )
        packer = HierarchicalPacker(spec)

        parent_df = pl.DataFrame(
            {
                "id": ["p1", "p2"],
                "name": ["Parent 1", "Parent 2"],
                "extra_parent_col": ["a", "b"],  # Extra column
            }
        )
        child_df = pl.DataFrame(
            {
                "id": ["c1", "c2", "c3"],
                "name": ["Child 1", "Child 2", "Child 3"],
                "parent_id": ["p1", "p1", "p2"],
                "extra_child_col": [1, 2, 3],  # Extra column
            }
        )

        result = packer.build_from_tables({"parent": parent_df, "child": child_df})

        # Unpack to verify extra columns are present
        unpacked = packer.unpack(result, "child")
        assert "parent.extra_parent_col" in unpacked.columns
        assert "parent.child.extra_child_col" in unpacked.columns

    def test_orphan_children_with_left_join(self) -> None:
        """Test children without matching parents (left join behavior)."""
        spec = HierarchySpec.from_levels(
            LevelSpec(name="parent", id_fields=["id"]),
            LevelSpec(name="child", id_fields=["id"], parent_keys=["parent_id"]),
        )
        packer = HierarchicalPacker(spec)

        parent_df = pl.DataFrame({"id": ["p1"], "name": ["Parent 1"]})
        child_df = pl.DataFrame(
            {
                "id": ["c1", "c2"],
                "parent_id": ["p1", "p_unknown"],  # c2 has no matching parent
            }
        )

        # With left join (default), orphan children are dropped
        result = packer.build_from_tables(
            {"parent": parent_df, "child": child_df},
            join_type="left",
        )

        unpacked = packer.unpack(result, "child")
        # Only c1 should be present (c2's parent doesn't exist)
        assert unpacked.height == 1
        assert unpacked["parent.child.id"][0] == "c1"

    def test_three_level_build_from_tables(self) -> None:
        """Test building a 3-level hierarchy from tables."""
        spec = HierarchySpec.from_levels(
            LevelSpec(name="company", id_fields=["id"]),
            LevelSpec(name="dept", id_fields=["id"], parent_keys=["company_id"]),
            LevelSpec(name="employee", id_fields=["id"], parent_keys=["dept_id"]),
        )
        packer = HierarchicalPacker(spec)

        companies = pl.DataFrame({"id": ["c1"], "name": ["Acme"]})
        depts = pl.DataFrame(
            {"id": ["d1", "d2"], "name": ["Eng", "Sales"], "company_id": ["c1", "c1"]}
        )
        employees = pl.DataFrame(
            {
                "id": ["e1", "e2", "e3"],
                "name": ["Alice", "Bob", "Carol"],
                "dept_id": ["d1", "d1", "d2"],
            }
        )

        result = packer.build_from_tables(
            {"company": companies, "dept": depts, "employee": employees}
        )

        assert result.height == 1  # One company

        unpacked = packer.unpack(result, "employee")
        assert unpacked.height == 3  # Three employees


# =============================================================================
# Struct Mode Guidance Tests (for nexpresso integration)
# =============================================================================


class TestStructModeGuidance:
    """Tests verifying struct_mode behavior with helpful error context."""

    def test_select_mode_drops_unspecified_fields(self) -> None:
        """Test that select mode only keeps specified fields."""
        df = pl.DataFrame(
            {
                "data": [
                    {"a": 1, "b": 2, "c": 3},
                    {"a": 4, "b": 5, "c": 6},
                ]
            }
        )

        result = apply_nested_operations(
            df,
            {"data": {"a": None, "b": lambda x: x * 2}},
            struct_mode="select",
        )

        # Only a and b should be present
        struct_val = result["data"][0]
        assert "a" in struct_val
        assert "b" in struct_val
        assert "c" not in struct_val

    def test_with_fields_mode_preserves_all(self) -> None:
        """Test that with_fields mode keeps all original fields."""
        df = pl.DataFrame(
            {
                "data": [
                    {"a": 1, "b": 2, "c": 3},
                    {"a": 4, "b": 5, "c": 6},
                ]
            }
        )

        result = apply_nested_operations(
            df,
            {"data": {"a": lambda x: x * 10}},
            struct_mode="with_fields",
        )

        # All fields should be present
        struct_val = result["data"][0]
        assert struct_val["a"] == 10  # Modified
        assert struct_val["b"] == 2  # Preserved
        assert struct_val["c"] == 3  # Preserved

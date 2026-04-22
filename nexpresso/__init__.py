"""
Polars Nexpresso

A utility library for generating Polars expressions to work with nested data structures.
Easily select, modify, and create columns and nested fields in Polars DataFrames.
"""

from nexpresso.expressions import (
    NestedExpressionBuilder,
    NestedFilterBuilder,
    apply_nested_filter,
    apply_nested_operations,
    filter_nested_elements,
    generate_nested_exprs,
    generate_nested_filter_expr,
)
from nexpresso.hierarchical_packer import (
    DiscoveredLevel,
    HierarchicalPacker,
    HierarchySpec,
    HierarchyValidationError,
    LevelAttribute,
    LevelSpec,
    PromoteAggregation,
    SchemaValidationResult,
)
from nexpresso.structuring_utils import convert_polars_schema, unnest_all, unnest_rename

__version__ = "0.3.1"

__all__ = [
    "__version__",
    # Nested expression builder
    "NestedExpressionBuilder",
    "generate_nested_exprs",
    "apply_nested_operations",
    # Nested filter builder
    "NestedFilterBuilder",
    "generate_nested_filter_expr",
    "filter_nested_elements",
    "apply_nested_filter",
    # Hierarchical packer
    "DiscoveredLevel",
    "HierarchicalPacker",
    "HierarchySpec",
    "HierarchyValidationError",
    "LevelAttribute",
    "LevelSpec",
    "PromoteAggregation",
    "SchemaValidationResult",
    # Structuring utilities
    "convert_polars_schema",
    "unnest_all",
    "unnest_rename",
]

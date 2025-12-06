"""
Polars Nexpresso

A utility library for generating Polars expressions to work with nested data structures.
Easily select, modify, and create columns and nested fields in Polars DataFrames.
"""

from nexpresso.hierarchical_packer import (
    HierarchicalPacker,
    HierarchySpec,
    HierarchyValidationError,
    LevelSpec,
)
from nexpresso.nexpresso import (
    NestedExpressionBuilder,
    apply_nested_operations,
    generate_nested_exprs,
)
from nexpresso.structuring_utils import convert_polars_schema, unnest_all, unnest_rename

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Nested expression builder
    "NestedExpressionBuilder",
    "generate_nested_exprs",
    "apply_nested_operations",
    # Hierarchical packer
    "HierarchicalPacker",
    "HierarchySpec",
    "HierarchyValidationError",
    "LevelSpec",
    # Structuring utilities
    "convert_polars_schema",
    "unnest_all",
    "unnest_rename",
]

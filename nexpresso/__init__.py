"""
Polars Nexpresso

A utility library for generating Polars expressions to work with nested data structures.
Easily select, modify, and create columns and nested fields in Polars DataFrames.
"""

from nexpresso.expressions import (
    NestedExpressionBuilder,
    apply_nested_operations,
    generate_nested_exprs,
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
from nexpresso.hierarchy_protocol import HierarchyOperator, NestedBackend
from nexpresso.level_expr import F, FieldRef, LevelExpr
from nexpresso.normalized_packer import NormalizedPacker
from nexpresso.structuring_utils import convert_polars_schema, unnest_all, unnest_rename

__version__ = "0.3.0"

__all__ = [
    "__version__",
    # Nested expression builder
    "NestedExpressionBuilder",
    "generate_nested_exprs",
    "apply_nested_operations",
    # Hierarchical packer
    "DiscoveredLevel",
    "HierarchicalPacker",
    "HierarchySpec",
    "HierarchyValidationError",
    "LevelAttribute",
    "LevelSpec",
    "PromoteAggregation",
    "SchemaValidationResult",
    # Normalized packer
    "NormalizedPacker",
    # Hierarchy protocol
    "HierarchyOperator",
    "NestedBackend",
    # Level expression DSL
    "F",
    "FieldRef",
    "LevelExpr",
    # Structuring utilities
    "convert_polars_schema",
    "unnest_all",
    "unnest_rename",
]

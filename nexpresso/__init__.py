"""
Polars Nexpresso

A utility library for generating Polars expressions to work with nested data structures.
Easily select, modify, and create columns and nested fields in Polars DataFrames.
"""

from nexpresso.nexpresso import (
    NestedExpressionBuilder,
    apply_nested_operations,
    generate_nested_exprs,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "NestedExpressionBuilder",
    "generate_nested_exprs",
    "apply_nested_operations",
]

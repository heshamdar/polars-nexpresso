"""
Deprecated alias for :mod:`nexpresso.expressions`.

This module used to hold a second, hand-maintained copy of the nested
expression builder. The copy drifted from :mod:`nexpresso.expressions` (it
never picked up the ``arr.eval()`` version gate, for instance), so importing
from here silently gave you different behaviour. It is now a thin re-export of
the canonical implementation.

Import from :mod:`nexpresso` or :mod:`nexpresso.expressions` instead.
"""

from __future__ import annotations

from nexpresso.expressions import (
    FieldValue,
    NestedExpressionBuilder,
    StructMode,
    apply_nested_operations,
    generate_nested_exprs,
)

__all__ = [
    "FieldValue",
    "NestedExpressionBuilder",
    "StructMode",
    "apply_nested_operations",
    "generate_nested_exprs",
]

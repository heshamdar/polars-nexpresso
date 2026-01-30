# CLAUDE.md - AI Assistant Guide for Polars Nexpresso

This document provides comprehensive guidance for AI assistants working with the polars-nexpresso codebase. It covers architecture, conventions, workflows, and best practices.

**Last Updated:** 2026-01-30
**Version:** 0.2.0

---

## Table of Contents

1. [Repository Overview](#repository-overview)
2. [Project Structure](#project-structure)
3. [Core Concepts & Architecture](#core-concepts--architecture)
4. [Development Setup](#development-setup)
5. [Code Conventions](#code-conventions)
6. [Testing Guidelines](#testing-guidelines)
7. [CI/CD Pipeline](#cicd-pipeline)
8. [Common Patterns](#common-patterns)
9. [Things to Avoid](#things-to-avoid)
10. [Tips for AI Assistants](#tips-for-ai-assistants)

---

## Repository Overview

**Project Name:** Polars Nexpresso
**Description:** A utility library for working with nested and hierarchical data in Polars
**Language:** Python 3.10+
**Primary Dependency:** polars >= 1.20.0
**Package Manager:** uv
**License:** MIT

### Main Features

1. **Nested Expression Builder** - Clean, intuitive syntax for transforming deeply nested structs and lists
2. **Hierarchical Packer** - Pack/unpack operations for hierarchical data (similar to pandas MultiIndex)

### Key URLs

- **Repository:** https://github.com/heshamdar/polars-nexpresso
- **Documentation:** https://heshamdar.github.io/polars-nexpresso
- **Package:** https://pypi.org/project/polars-nexpresso/

---

## Project Structure

```
polars-nexpresso/
├── nexpresso/                        # Main package
│   ├── __init__.py                   # Public API exports
│   ├── expressions.py                # Nested expression builder
│   ├── hierarchical_packer.py        # Hierarchical data operations
│   ├── nexpresso.py                  # Legacy module (expressions alias)
│   ├── structuring_utils.py          # Utility functions
│   └── py.typed                      # PEP 561 marker for type checking
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── conftest.py                   # Pytest fixtures & version checks
│   ├── test_complex_hierarchies.py   # Complex hierarchy tests
│   ├── test_hierarchical_packer.py   # Packer tests
│   ├── test_integration.py           # Integration tests
│   ├── test_matrix.py                # Multi-version test runner
│   ├── test_nested_helper.py         # Nested expression tests
│   └── test_structuring_utils.py     # Utility tests
├── docs/                             # MkDocs documentation
│   ├── api/                          # API reference
│   ├── concepts/                     # Conceptual guides
│   ├── guides/                       # How-to guides
│   ├── index.md                      # Documentation home
│   ├── getting-started.md
│   └── examples.md
├── .github/workflows/
│   ├── test.yml                      # CI test workflow
│   └── publish.yml                   # PyPI publish workflow
├── examples.py                       # Comprehensive examples
├── pyproject.toml                    # Project metadata & config
├── pytest.ini                        # Pytest configuration
├── mkdocs.yml                        # MkDocs configuration
├── uv.lock                           # Locked dependencies
├── README.md                         # User-facing documentation
├── SETUP.md                          # Development setup guide
└── LICENSE                           # MIT License
```

### Key Files

| File | Purpose |
|------|---------|
| `nexpresso/expressions.py` | Core nested expression building logic |
| `nexpresso/hierarchical_packer.py` | Hierarchical data pack/unpack operations |
| `nexpresso/__init__.py` | Public API surface (what users import) |
| `tests/conftest.py` | Shared test fixtures and version utilities |
| `tests/test_matrix.py` | Multi-version compatibility testing |
| `pyproject.toml` | Package metadata, dependencies, tool config |
| `examples.py` | Runnable examples demonstrating features |

---

## Core Concepts & Architecture

### Design Patterns

#### 1. Builder Pattern
**`NestedExpressionBuilder`** encapsulates expression generation:
```python
builder = NestedExpressionBuilder(schema, struct_mode="select")
exprs = builder.build(fields)
```
- Single responsibility: transform field specs into Polars expressions
- Recursive processing for nested structures
- Internal methods handle different data types

#### 2. Specification-Driven Design
**`HierarchySpec`** and **`LevelSpec`** use declarative configuration:
```python
spec = HierarchySpec.from_levels(
    LevelSpec(name="region", id_fields=["id"]),
    LevelSpec(name="store", id_fields=["id"], parent_keys=["region_id"]),
)
```
- Separates "what to do" from "how to do it"
- Immutable dataclasses with validation in `__post_init__`
- Composable and reusable

#### 3. Type-Generic Operations
**`FrameT` TypeVar** supports both DataFrame and LazyFrame:
```python
FrameT = TypeVar("FrameT", pl.LazyFrame, pl.DataFrame)

def pack(self, frame: FrameT, to_level: str) -> FrameT:
    # DataFrame in = DataFrame out
    # LazyFrame in = LazyFrame out
```

#### 4. Metadata Caching
**`HierarchicalPacker`** pre-computes metadata in `__init__`:
- `_levels_meta`: Cached level metadata
- `_computed_exprs`: Pre-built expressions
- Avoids repeated computation during operations

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `expressions.py` | Nested data transformation via expression building |
| `hierarchical_packer.py` | Hierarchical data operations (pack/unpack/normalize) |
| `structuring_utils.py` | Utility functions (schema conversion, unnesting) |
| `nexpresso.py` | Legacy module (imports from expressions.py) |

### Public API Surface

```python
# From expressions.py
from nexpresso import (
    NestedExpressionBuilder,      # Builder class
    generate_nested_exprs,         # Generate expressions
    apply_nested_operations,       # Apply to DataFrame
)

# From hierarchical_packer.py
from nexpresso import (
    HierarchicalPacker,            # Main packer class
    HierarchySpec,                 # Hierarchy specification
    LevelSpec,                     # Level specification
    HierarchyValidationError,      # Custom exception
)

# From structuring_utils.py
from nexpresso import (
    convert_polars_schema,         # Schema conversion
    unnest_all,                    # Recursive unnesting
    unnest_rename,                 # Unnest with renaming
)
```

---

## Development Setup

### Prerequisites

- Python 3.10 or higher
- `uv` package manager (recommended) or `pip`

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/heshamdar/polars-nexpresso.git
cd polars-nexpresso

# Install dependencies with uv
uv sync

# Or with pip
pip install -e ".[dev]"
```

### Development Dependencies

Defined in `pyproject.toml` under `[dependency-groups]`:

```toml
[dependency-groups]
dev = [
    "pytest>=8.4.2",
    "pytest-cov>=4.1.0",
    "black>=24.1.0",
    "ruff>=0.10.0",
    "mypy>=1.15.0",
]
docs = [
    "mkdocs>=1.6.0",
    "mkdocs-material>=9.5.0",
    "mkdocstrings[python]>=0.26.0",
    "pymdown-extensions>=10.0.0",
]
```

### Running Tests

```bash
# Run all tests with current Polars version
uv run pytest -v

# Run specific test file
uv run pytest tests/test_nested_helper.py -v

# Run with coverage
uv run pytest --cov=nexpresso --cov-report=html

# Test against multiple Polars versions (recommended before commits)
uv run python tests/test_matrix.py

# Test specific versions
uv run python tests/test_matrix.py --versions 1.20.0 1.30.0 latest

# Stop on first failure
uv run python tests/test_matrix.py --stop-on-failure
```

### Running Examples

```bash
# Run comprehensive examples
uv run python examples.py

# Run module examples
uv run python -m nexpresso.hierarchical_packer
```

### Code Quality Tools

```bash
# Format code with Black
uv run black nexpresso tests

# Lint with Ruff
uv run ruff check nexpresso tests

# Type check with MyPy
uv run mypy nexpresso
```

### Building Documentation

```bash
# Install docs dependencies
uv sync --group docs

# Serve documentation locally
uv run mkdocs serve

# Build documentation
uv run mkdocs build
```

### Building Package

```bash
# Build both wheel and source distribution
uv build

# Build only wheel
uv build --wheel

# Check distribution
uv run twine check dist/*
```

---

## Code Conventions

### Python Style

- **Line Length:** 100 characters (enforced by Black and Ruff)
- **Python Version:** 3.10+ features allowed
- **Type Hints:** Required for all public APIs
- **Docstrings:** Required for all public classes and functions

### Type Annotations

**Always use modern Python 3.10+ syntax:**

```python
# ✅ Good - Modern union syntax
def process(data: str | None) -> list[str] | dict[str, int]:
    pass

# ❌ Bad - Old Union syntax
from typing import Union, Optional, List, Dict
def process(data: Optional[str]) -> Union[List[str], Dict[str, int]]:
    pass
```

**Key type aliases:**

```python
FieldValue = None | dict[str, "FieldValue"] | Callable[[pl.Expr], pl.Expr] | pl.Expr
StructMode = Literal["select", "with_fields"]
FrameT = TypeVar("FrameT", pl.LazyFrame, pl.DataFrame)
ColumnSelector = str | pl.Expr
ExtraColumnsMode = Literal["preserve", "drop", "error"]
```

**Generic return types:**

```python
def pack(self, frame: FrameT, to_level: str) -> FrameT:
    """Returns same type as input (DataFrame or LazyFrame)."""
```

### Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Public classes | PascalCase | `NestedExpressionBuilder`, `HierarchySpec` |
| Public functions | snake_case | `generate_nested_exprs`, `apply_nested_operations` |
| Private methods | _leading_underscore | `_process_struct_field`, `_match_frame_type` |
| Constants | UPPER_SNAKE_CASE | `ROW_ID_COLUMN`, `DEFAULT_SEPARATOR` |
| Type aliases | PascalCase | `FrameT`, `FieldValue`, `StructMode` |
| Module-level vars | snake_case | `__version__`, `__all__` |

### Docstring Format

Use Google-style docstrings:

```python
def generate_nested_exprs(
    fields: dict[str, FieldValue],
    schema: pl.Schema | FrameT,
    struct_mode: StructMode = "select",
) -> list[pl.Expr]:
    """
    Generate Polars expressions for nested data operations.

    This function provides a convenient way to work with nested structures
    by generating expressions from a dictionary specification.

    Args:
        fields: Dictionary defining operations on columns/fields.
            - key: Column/field name
            - value: Operation specification (None, dict, Callable, or pl.Expr)
        schema: DataFrame schema or a DataFrame/LazyFrame to extract schema from.
        struct_mode: How to handle struct fields. Defaults to "select".
            - "select": Only keep specified fields
            - "with_fields": Keep all fields, add/modify specified ones

    Returns:
        List of Polars expressions ready for use in select() or with_columns().

    Raises:
        ValueError: If column doesn't exist or operations are invalid.
        TypeError: If field value type is invalid.

    Examples:
        >>> df = pl.DataFrame({"data": [{"x": 1, "y": 2}]})
        >>> fields = {"data": {"x": lambda x: x * 2}}
        >>> exprs = generate_nested_exprs(fields, df.schema)
        >>> result = df.select(exprs)
    """
```

### Error Handling

**1. Use custom exceptions for domain errors:**

```python
class HierarchyValidationError(Exception):
    """Exception raised when hierarchy validation fails."""
    def __init__(self, message: str, level: str | None = None, details: dict | None = None):
        self.level = level
        self.details = details or {}
        prefix = f"[Level: {level}] " if level else ""
        super().__init__(f"{prefix}{message}")
```

**2. Provide descriptive error messages:**

```python
# ✅ Good - Clear context and actionable guidance
if col_name not in self._schema:
    if not isinstance(field_spec, pl.Expr):
        raise ValueError(
            f"Column '{col_name}' not found in schema. "
            "To create a new column, provide a pl.Expr."
        )

# ❌ Bad - Vague error
if col_name not in self._schema:
    raise ValueError("Column not found")
```

**3. Validate inputs early:**

```python
def __init__(self, schema: pl.Schema, struct_mode: StructMode = "select") -> None:
    if struct_mode not in ("select", "with_fields"):
        raise ValueError(
            f"Invalid struct_mode: {struct_mode}. "
            "Must be 'select' or 'with_fields'."
        )
    # Continue with validated inputs
```

**4. Support error collection:**

```python
def validate(self, frame: FrameT, raise_on_error: bool = True) -> list[str]:
    """
    Validate hierarchy data integrity.

    Args:
        frame: The frame to validate.
        raise_on_error: If True, raise on first error. If False, collect all errors.

    Returns:
        List of error messages (empty if valid).
    """
```

### Comment Guidelines

**When to comment:**

```python
# ✅ Good - Explains WHY, not WHAT
# Use arr.eval() for Polars >= 1.35.1, fall back to list.eval() for older versions
if _supports_arr_eval():
    expr = base_expr.arr.eval(transformed_expr)
else:
    expr = base_expr.list.eval(transformed_expr)

# ✅ Good - Section headers for organization
# =============================================================================
# Pack/Unpack Operations
# =============================================================================

# ❌ Bad - States the obvious
# Increment i by 1
i += 1

# ❌ Bad - Redundant with docstring
# This function validates the schema
def validate_schema(schema):
    """Validate the schema."""
```

**Documentation comments:**

- Module-level docstrings: Describe purpose and main exports
- Class docstrings: Describe responsibility and usage
- Method docstrings: Include Args, Returns, Raises, Examples
- Inline comments: Explain non-obvious logic or workarounds

---

## Testing Guidelines

### Test Organization

```
tests/
├── conftest.py                    # Shared fixtures & utilities
├── test_nested_helper.py          # Nested expression tests
├── test_hierarchical_packer.py    # Hierarchical packer tests
├── test_complex_hierarchies.py    # Complex scenario tests
├── test_integration.py            # Integration tests
├── test_structuring_utils.py      # Utility tests
└── test_matrix.py                 # Multi-version test runner
```

### Test Structure

```python
class TestNestedExpressions:
    """Tests for nested expression building."""

    def test_simple_struct_transformation(self, simple_nested_df):
        """Test basic struct field transformation."""
        fields = {"data": {"value": lambda x: x * 2}}
        result = apply_nested_operations(simple_nested_df, fields)

        expected = pl.DataFrame({
            "id": [1, 2],
            "data": [{"value": 200}, {"value": 400}],
        })
        assert_frame_equal(result, expected)

    def test_missing_column_error(self):
        """Test that missing columns raise clear errors."""
        df = pl.DataFrame({"x": [1, 2]})
        fields = {"missing": None}

        with pytest.raises(ValueError, match="not found in schema"):
            generate_nested_exprs(fields, df.schema)
```

### Fixtures (conftest.py)

**Shared test data:**

```python
@pytest.fixture
def simple_nested_df() -> pl.DataFrame:
    """Create a simple DataFrame with nested structure."""
    return pl.DataFrame({
        "id": [1, 2],
        "data": [
            {"name": "Alice", "value": 100},
            {"name": "Bob", "value": 200},
        ],
    })

@pytest.fixture
def list_of_structs_df() -> pl.DataFrame:
    """Create a DataFrame with list of structs."""
    return pl.DataFrame({
        "id": [1, 2],
        "items": [
            [{"name": "A", "qty": 2}, {"name": "B", "qty": 3}],
            [{"name": "C", "qty": 1}],
        ],
    })
```

### Version-Specific Testing

**Skip tests for unsupported versions:**

```python
from conftest import requires_arr_eval, skip_if_polars_below

@requires_arr_eval
def test_array_operations():
    """Test arr.eval() - requires Polars >= 1.35.1."""
    pass

@skip_if_polars_below("1.30.0")
def test_new_feature():
    """Test feature added in Polars 1.30.0."""
    pass
```

**Version checking utilities:**

```python
from conftest import get_polars_version, polars_version_at_least

def test_conditional_behavior():
    if polars_version_at_least("1.35.1"):
        # Test new behavior
        pass
    else:
        # Test fallback behavior
        pass
```

### Multi-Version Testing

```bash
# Test all default versions (1.20.0, 1.30.0, 1.35.1, latest)
uv run python tests/test_matrix.py

# Custom version list
uv run python tests/test_matrix.py --versions 1.20.0 1.35.1 latest

# Test from minimum version onwards
uv run python tests/test_matrix.py --min-version 1.25.0

# Stop on first failure
uv run python tests/test_matrix.py --stop-on-failure
```

### Assertion Patterns

```python
from polars.testing import assert_frame_equal, assert_series_equal

# ✅ Use Polars testing utilities
assert_frame_equal(result, expected)

# ✅ For order-independent comparison
result_sorted = result.sort(by=["id"])
expected_sorted = expected.sort(by=["id"])
assert_frame_equal(result_sorted, expected_sorted)

# ✅ Check specific columns
assert result["col"].to_list() == [1, 2, 3]

# ✅ Check schema
assert result.schema == expected.schema
```

---

## CI/CD Pipeline

### GitHub Actions Workflows

#### 1. Test Workflow (`.github/workflows/test.yml`)

Triggers on:
- Push to `main`/`master`
- Pull requests to `main`/`master`
- Manual dispatch
- Releases

**Matrix testing:**
```yaml
strategy:
  matrix:
    polars-version: ["1.20.0", "1.30.0", "1.35.1", "latest"]
```

**Steps:**
1. Checkout code
2. Install `uv`
3. Set up Python 3.12
4. Install dependencies (`uv sync --dev`)
5. Run tests for each Polars version

#### 2. Publish Workflow (`.github/workflows/publish.yml`)

Triggers on:
- GitHub releases (automated)
- Manual dispatch (with TestPyPI option)

**Steps:**
1. Run full test suite (same as test.yml)
2. Build package with `uv build`
3. Check package with `twine check`
4. Publish to PyPI (using trusted publishing/OIDC)

### Pre-Commit Checks

Before pushing code, ensure:

```bash
# Run tests
uv run pytest -v

# Multi-version tests
uv run python tests/test_matrix.py --stop-on-failure

# Format code
uv run black nexpresso tests

# Lint
uv run ruff check nexpresso tests

# Type check
uv run mypy nexpresso

# Run examples
uv run python examples.py
```

### Release Checklist

Before creating a release:

- [ ] Update version in `pyproject.toml`
- [ ] Update version in `nexpresso/__init__.py`
- [ ] Update CHANGELOG.md (if exists)
- [ ] Run full test suite: `uv run python tests/test_matrix.py`
- [ ] Run examples: `uv run python examples.py`
- [ ] Build and check: `uv build && uv run twine check dist/*`
- [ ] Update documentation if needed
- [ ] Create git tag: `git tag v0.2.0`
- [ ] Push tag: `git push origin v0.2.0`
- [ ] Create GitHub release (triggers publish workflow)

---

## Common Patterns

### Pattern 1: Type-Preserving Operations

**Always preserve input frame type:**

```python
def pack(self, frame: FrameT, to_level: str) -> FrameT:
    """Pack to coarser granularity, preserving frame type."""
    # Convert to lazy for operations
    lazy = self._to_lazy(frame)

    # Perform operations on LazyFrame
    result = lazy.group_by(...)

    # Return same type as input
    return self._match_frame_type(result, frame)

def _to_lazy(self, frame: FrameT) -> pl.LazyFrame:
    """Convert to LazyFrame if needed."""
    return frame.lazy() if isinstance(frame, pl.DataFrame) else frame

def _match_frame_type(self, lazy: pl.LazyFrame, original: FrameT) -> FrameT:
    """Match the type of the original frame."""
    return lazy.collect() if isinstance(original, pl.DataFrame) else lazy
```

### Pattern 2: Schema Extraction

**Accept multiple schema sources:**

```python
def generate_nested_exprs(
    fields: dict[str, FieldValue],
    schema: pl.Schema | FrameT,
    struct_mode: StructMode = "select",
) -> list[pl.Expr]:
    """Generate expressions accepting schema or frame."""
    # Extract schema if needed
    if isinstance(schema, (pl.DataFrame, pl.LazyFrame)):
        schema = (
            schema.collect_schema()
            if isinstance(schema, pl.LazyFrame)
            else schema.schema
        )

    # Use schema
    builder = NestedExpressionBuilder(schema, struct_mode)
    return builder.build(fields)
```

### Pattern 3: Recursive Processing

**Handle nested structures recursively:**

```python
def _process_nested_field(
    self,
    dtype: PolarsDataType,
    field_spec: FieldValue,
    base_expr: pl.Expr,
) -> pl.Expr:
    """Process a nested field recursively."""
    if isinstance(dtype, pl.List):
        return self._process_list_field(dtype.inner, field_spec, base_expr)
    elif isinstance(dtype, pl.Array):
        return self._process_nested_field(dtype.inner, field_spec, pl.element())
    elif isinstance(dtype, pl.Struct):
        return self._process_struct_field(dtype, field_spec, base_expr)
    else:
        return self._process_leaf_field(field_spec, base_expr)
```

### Pattern 4: Version Compatibility

**Check version and provide fallbacks:**

```python
from functools import lru_cache
from packaging import version

@lru_cache(maxsize=1)
def _polars_version() -> version.Version:
    """Get the current Polars version."""
    return version.parse(pl.__version__)

def _supports_arr_eval() -> bool:
    """Check if arr.eval() is available."""
    return _polars_version() >= version.parse("1.35.1")

def _apply_transformation(self, base_expr: pl.Expr, transform: pl.Expr) -> pl.Expr:
    """Apply transformation with version-appropriate method."""
    if _supports_arr_eval():
        return base_expr.arr.eval(transform)
    else:
        # Fallback for older versions
        return base_expr.list.eval(transform)
```

### Pattern 5: Builder with Validation

**Validate and build in separate phases:**

```python
class NestedExpressionBuilder:
    def __init__(self, schema: pl.Schema, struct_mode: StructMode = "select"):
        # Validate mode immediately
        if struct_mode not in ("select", "with_fields"):
            raise ValueError(f"Invalid struct_mode: {struct_mode}")

        self._schema = schema
        self._struct_mode = struct_mode

    def build(self, fields: dict[str, FieldValue]) -> list[pl.Expr]:
        """Build expressions after validation."""
        expressions = []
        for col_name, field_spec in fields.items():
            expr = self._process_top_level_field(col_name, field_spec)
            expressions.append(expr)
        return expressions
```

### Pattern 6: Immutable Specifications

**Use frozen dataclasses for configuration:**

```python
@dataclass(frozen=True)
class LevelSpec:
    """Immutable level specification."""
    name: str
    id_fields: Sequence[ColumnSelector] = ()
    required_fields: Sequence[ColumnSelector] | None = None
    order_by: Sequence[pl.Expr] | None = None
    parent_keys: Sequence[str] | None = None

@dataclass(frozen=True)
class HierarchySpec:
    """Immutable hierarchy specification."""
    levels: Sequence[LevelSpec]
    key_aliases: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate after initialization."""
        level_names = [lvl.name for lvl in self.levels]
        if len(level_names) != len(set(level_names)):
            raise ValueError("Level names must be unique")
```

---

## Things to Avoid

### Anti-Patterns

❌ **Don't mutate input frames:**
```python
# Bad - modifies input
def bad_transform(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(pl.col("x") * 2)  # ⚠️ May mutate
    return df

# Good - explicit new frame
def good_transform(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col("x") * 2)
```

❌ **Don't use old-style type hints:**
```python
# Bad
from typing import Union, Optional, List, Dict
def process(data: Optional[Dict[str, List[int]]]) -> Union[str, None]:
    pass

# Good
def process(data: dict[str, list[int]] | None) -> str | None:
    pass
```

❌ **Don't collect LazyFrames unnecessarily:**
```python
# Bad - forces evaluation
def bad_process(lf: pl.LazyFrame) -> pl.LazyFrame:
    df = lf.collect()  # ⚠️ Unnecessary collection
    return df.lazy()

# Good - stay lazy
def good_process(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.select(...)
```

❌ **Don't ignore version compatibility:**
```python
# Bad - assumes new API exists
def bad_use_new_feature():
    return df.arr.eval(...)  # ⚠️ Breaks on Polars < 1.35.1

# Good - check version
def good_use_new_feature():
    if _supports_arr_eval():
        return df.arr.eval(...)
    else:
        return df.list.eval(...)  # Fallback
```

❌ **Don't write vague error messages:**
```python
# Bad
if not valid:
    raise ValueError("Invalid input")

# Good
if col_name not in schema:
    raise ValueError(
        f"Column '{col_name}' not found in schema. "
        f"Available columns: {list(schema.keys())}"
    )
```

❌ **Don't use mutable default arguments:**
```python
# Bad - mutable default
def bad_func(items: list = []):  # ⚠️ Shared across calls
    items.append(1)
    return items

# Good - use None and create new
def good_func(items: list | None = None) -> list:
    if items is None:
        items = []
    items = items.copy()  # Don't mutate input
    items.append(1)
    return items
```

❌ **Don't mix eager and lazy operations unpredictably:**
```python
# Bad - inconsistent return type
def bad_process(frame):
    if isinstance(frame, pl.DataFrame):
        return frame.lazy().select(...).collect()
    return frame.select(...)  # LazyFrame

# Good - preserve input type
def good_process(frame: FrameT) -> FrameT:
    lazy = self._to_lazy(frame)
    result = lazy.select(...)
    return self._match_frame_type(result, frame)
```

---

## Tips for AI Assistants

### Understanding the Codebase

1. **Start with public APIs** (`__init__.py`, public methods)
2. **Read module docstrings** for high-level understanding
3. **Check examples.py** for usage patterns
4. **Review tests** to understand expected behavior
5. **Consult SETUP.md** for development workflows

### Making Changes

#### When Adding Features

1. **Determine the right module:**
   - Nested expressions → `expressions.py`
   - Hierarchical operations → `hierarchical_packer.py`
   - Utilities → `structuring_utils.py`

2. **Follow existing patterns:**
   - Use builder pattern for complex object construction
   - Use frozen dataclasses for specifications
   - Preserve frame types (FrameT)
   - Support both eager and lazy frames

3. **Add comprehensive tests:**
   - Unit tests for new functions/methods
   - Integration tests for workflows
   - Version-specific tests if using new Polars APIs
   - Edge cases (nulls, empty data, deeply nested)

4. **Update documentation:**
   - Docstrings with Args/Returns/Raises/Examples
   - Update README.md if public API changes
   - Add examples to examples.py
   - Update docs/ markdown files

5. **Check compatibility:**
   - Test with multiple Polars versions
   - Add version checks for new APIs
   - Update conftest.py skip markers if needed

#### When Fixing Bugs

1. **Add a failing test first** (TDD approach)
2. **Understand the root cause** (read related code)
3. **Make minimal changes** (don't refactor unrelated code)
4. **Verify the fix** (run affected tests)
5. **Check for similar issues** (search codebase)

#### When Refactoring

1. **Ensure tests pass first** (green state)
2. **Make small, incremental changes**
3. **Run tests after each change**
4. **Don't change behavior** (unless intentional)
5. **Update docstrings** if signatures change

### Working with Polars

#### Expression Building

```python
# ✅ Chain methods for clarity
expr = (
    pl.col("data")
    .struct.field("items")
    .list.eval(pl.element().struct.field("price") * 1.1)
)

# ✅ Use pl.col() for columns
pl.col("column_name")

# ✅ Use pl.element() in list context
df.select(pl.col("items").list.eval(pl.element() * 2))

# ✅ Use pl.field() in struct context
pl.col("data").struct.with_fields(
    pl.field("value") * 2
)
```

#### Schema Handling

```python
# ✅ Check column existence
if "col" in df.schema:
    ...

# ✅ Get column type
dtype = df.schema["col"]
if isinstance(dtype, pl.Struct):
    fields = dtype.fields

# ✅ Extract schema from frames
if isinstance(frame, pl.LazyFrame):
    schema = frame.collect_schema()
else:
    schema = frame.schema
```

#### Lazy vs Eager

```python
# ✅ Prefer lazy for operations
def process(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.lazy()
        .select(...)
        .filter(...)
        .collect()
    )

# ✅ Preserve frame type
def process(frame: FrameT) -> FrameT:
    lazy = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    result = lazy.select(...)
    return result.collect() if isinstance(frame, pl.DataFrame) else result
```

### Common Pitfalls

1. **Forgetting to handle None/null values**
   - Always consider null cases in nested structures
   - Use `.drop_nulls()` where appropriate

2. **Not preserving column order**
   - Users expect predictable column order
   - Use explicit column lists when needed

3. **Assuming specific Polars version**
   - Always check version for new APIs
   - Provide fallbacks or clear error messages

4. **Mutating shared state**
   - Use frozen dataclasses for configs
   - Don't modify input frames directly

5. **Over-eager evaluation**
   - Keep operations lazy as long as possible
   - Only collect when necessary

### Debugging Tips

```python
# Print schema for inspection
print(df.schema)

# Check data types
print(df.dtypes)

# Inspect nested structures
print(df.select(pl.col("nested").struct.unnest()))

# Show query plan (LazyFrame)
print(lf.explain())

# Check for nulls
print(df.null_count())
```

### Performance Considerations

1. **Prefer lazy evaluation** - builds optimal query plan
2. **Batch operations** - combine expressions when possible
3. **Avoid repeated schema extraction** - cache and pass through
4. **Use Polars native operations** - avoid Python loops
5. **Pre-compute metadata** - cache expensive computations

---

## Quick Reference

### Most Common Operations

```python
# Generate expressions
from nexpresso import generate_nested_exprs

fields = {"data": {"value": lambda x: x * 2}}
exprs = generate_nested_exprs(fields, df.schema, struct_mode="with_fields")
result = df.select(exprs)

# Apply operations directly
from nexpresso import apply_nested_operations

result = apply_nested_operations(df, fields, struct_mode="with_fields")

# Hierarchical packing
from nexpresso import HierarchicalPacker, HierarchySpec, LevelSpec

spec = HierarchySpec.from_levels(
    LevelSpec(name="region", id_fields=["id"]),
    LevelSpec(name="store", id_fields=["id"], parent_keys=["region_id"]),
)
packer = HierarchicalPacker(spec)

# Pack to coarser level
region_level = packer.pack(store_level, "region")

# Unpack to finer level
store_level = packer.unpack(region_level, "store")

# Build from tables
nested = packer.build_from_tables({
    "region": regions_df,
    "store": stores_df,
})
```

### File Modification Checklist

When modifying code, ensure:

- [ ] Type hints are present and use modern syntax
- [ ] Docstrings follow Google style (Args/Returns/Raises/Examples)
- [ ] Error messages are descriptive and actionable
- [ ] Tests added/updated for changes
- [ ] Version compatibility checked
- [ ] Code formatted with Black (100 char line length)
- [ ] Linting passes (Ruff)
- [ ] Type checking passes (MyPy)
- [ ] Examples work if public API changed
- [ ] Frame type preservation maintained (FrameT)

---

## Additional Resources

- **README.md** - User-facing documentation and examples
- **SETUP.md** - Detailed development setup instructions
- **examples.py** - Comprehensive runnable examples
- **docs/** - Full documentation source (MkDocs)
- **tests/** - Test suite with examples of usage

For questions or issues:
- GitHub Issues: https://github.com/heshamdar/polars-nexpresso/issues
- Documentation: https://heshamdar.github.io/polars-nexpresso

---

**Remember:** When in doubt, follow existing patterns in the codebase. Consistency is more valuable than individual preferences.

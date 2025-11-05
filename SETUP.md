# Setup and Publishing Guide

This document describes how to set up, test, and publish this package.

## Development Setup

1. Install dependencies:
```bash
uv sync
```

2. Install the package in editable mode:
```bash
uv pip install -e .
```

3. Run tests:
```bash
# Run tests with current Polars version
uv run pytest test_nested_helper.py -v

# Or run the original test script
uv run python test_nested_helper.py

# Test against multiple Polars versions (recommended)
uv run python test_matrix.py
```

### Multi-Version Testing

The `test_matrix.py` script tests the library against multiple Polars versions to ensure compatibility:

```bash
# Test all default versions (1.20.0, 1.30.0, 1.35.1, latest)
uv run python test_matrix.py

# Test specific versions
uv run python test_matrix.py --versions 1.0.0 1.15.0 1.30.0

# Test from a minimum version onwards
uv run python test_matrix.py --min-version 1.10.0

# Stop on first failure
uv run python test_matrix.py --stop-on-failure

# Skip specific versions
uv run python test_matrix.py --skip-versions 1.0.0
```

The script uses `uv` to dynamically create isolated environments for each Polars version, ensuring clean test runs. Each version is tested in a separate temporary environment to avoid conflicts.

4. Run examples:
```bash
uv run python examples.py
```

## Building for PyPI

To build the package (without publishing):

```bash
# Build source distribution
uv build --sdist

# Build wheel
uv build --wheel

# Or both
uv build
```

This will create distributions in the `dist/` directory.

## Publishing to PyPI

**Note: Only do this when ready to publish!**

1. Make sure you have PyPI credentials set up:
```bash
pip install twine
```

2. Build the package:
```bash
uv build
```

3. Test the build:
```bash
twine check dist/*
```

4. Upload to TestPyPI (recommended first):
```bash
twine upload --repository testpypi dist/*
```

5. Upload to PyPI:
```bash
twine upload dist/*
```

## Before Publishing Checklist

- [ ] Update version number in `pyproject.toml` and `nexpresso/__init__.py`
- [ ] Update author information in `pyproject.toml`
- [ ] Update GitHub URLs in `pyproject.toml` if different
- [ ] Review and update README.md
- [ ] Run all tests: `uv run pytest test_nested_helper.py -v`
- [ ] Run multi-version tests: `uv run python test_matrix.py`
- [ ] Run examples: `uv run python examples.py`
- [ ] Build and test: `uv build && twine check dist/*`
- [ ] Update CHANGELOG.md if you have one

## Package Structure

```
polars-nexpresso/
├── nexpresso/                     # Main package directory
│   ├── __init__.py                # Package exports
│   └── nexpresso.py               # Core implementation
├── examples.py                     # Example usage
├── test_nested_helper.py          # Test suite
├── pyproject.toml                 # Package configuration
├── README.md                      # Documentation
├── LICENSE                        # MIT License
├── MANIFEST.in                    # Files to include in distribution
└── .gitignore                     # Git ignore rules
```


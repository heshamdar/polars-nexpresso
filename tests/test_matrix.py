#!/usr/bin/env python3
"""
Test matrix runner for polars-nexpresso.

Tests the library against multiple Polars versions using uv to manage
different environments dynamically.

Usage:
    python tests/test_matrix.py                    # Test all versions
    python tests/test_matrix.py --versions 1.20.0 1.30.0  # Test specific versions
    python tests/test_matrix.py --min-version 1.20.0      # Test from minimum version
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Polars versions to test against
DEFAULT_VERSIONS = [
    "1.20.0",  # Minimum supported version
    "1.30.0",  # Intermediate version
    "1.35.1",  # Current minimum in pyproject.toml
    "latest",  # Latest available version
]

# Test files to run
TEST_FILES = [
    "test_nested_helper.py",
    "test_hierarchical_packer.py",
    "test_structuring_utils.py",
    "test_complex_hierarchies.py",
    "test_integration.py",
]


def run_command(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, and stderr."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)


@pytest.mark.skip(
    reason="Run via 'python tests/test_matrix.py' script or parametrize with versions"
)
def test_polars_version(version: str, project_root: Path) -> tuple[bool, str, str]:
    """
    Test the library against a specific Polars version.

    This test is skipped by default when run via pytest.
    To run version matrix tests, use: python tests/test_matrix.py

    Returns:
        (success: bool, output: str, actual_version: str)
    """
    print(f"\n{'='*80}")
    print(f"Testing with Polars {version}")
    print(f"{'='*80}")

    # Create a temporary directory for this test environment
    with tempfile.TemporaryDirectory(prefix=f"polars-{version}-") as tmpdir:
        env_dir = Path(tmpdir)

        # Create a pyproject.toml for this specific version
        polars_version_spec = "polars" if version == "latest" else f"polars=={version}"
        pyproject_content = f"""[project]
name = "polars-nexpresso-test"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "{polars_version_spec}",
    "pytest>=8.4.2",
    "packaging>=21.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["nexpresso"]
"""

        # Write test pyproject.toml
        test_pyproject = env_dir / "pyproject.toml"
        test_pyproject.write_text(pyproject_content)

        # Copy the nexpresso package to the test environment
        import shutil

        shutil.copytree(
            project_root / "nexpresso",
            env_dir / "nexpresso",
            dirs_exist_ok=True,
        )

        # Copy all test files
        tests_dir = env_dir / "tests"
        tests_dir.mkdir(exist_ok=True)

        # Copy conftest.py and __init__.py first (if they exist)
        for special_file in ["conftest.py", "__init__.py"]:
            src = project_root / "tests" / special_file
            if src.exists():
                shutil.copy(src, tests_dir / special_file)

        for test_file in TEST_FILES:
            src = project_root / "tests" / test_file
            if src.exists():
                shutil.copy(src, tests_dir / test_file)

        # Create a uv environment with the specific Polars version
        print(f"Setting up environment for Polars {version}...")

        # Use uv to sync the environment
        sync_cmd = [
            "uv",
            "sync",
            "--project",
            str(env_dir),
        ]

        exit_code, stdout, stderr = run_command(sync_cmd)
        if exit_code != 0:
            error_msg = f"Failed to setup environment for Polars {version}:\n{stderr}"
            print(f"❌ {error_msg}")
            return False, error_msg, version

        # Get the actual installed Polars version
        version_cmd = [
            "uv",
            "run",
            "--project",
            str(env_dir),
            "python",
            "-c",
            "import polars; print(polars.__version__)",
        ]
        exit_code, version_stdout, version_stderr = run_command(version_cmd)
        if exit_code == 0:
            actual_version = version_stdout.strip()
            if version == "latest":
                print(f"Installed Polars version: {actual_version}")
        else:
            # Fallback to requested version if we can't determine actual version
            actual_version = version
            print(f"Warning: Could not determine installed Polars version, using {version}")

        # Run pytest on all test files
        print(f"Running tests with Polars {actual_version}...")
        test_cmd = [
            "uv",
            "run",
            "--project",
            str(env_dir),
            "pytest",
            "tests/",
            "-v",
            "--tb=short",
        ]

        exit_code, stdout, stderr = run_command(test_cmd)

        if exit_code == 0:
            print(f"✅ Polars {actual_version}: All tests passed!")
            return True, stdout, actual_version
        else:
            error_msg = f"Tests failed for Polars {actual_version}:\n{stderr}\n{stdout}"
            print(f"❌ {error_msg}")
            return False, error_msg, actual_version


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test polars-nexpresso against multiple Polars versions"
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        help="Specific Polars versions to test (e.g., --versions 1.0.0 1.15.0)",
    )
    parser.add_argument(
        "--min-version",
        help="Minimum version to test from (will test this and newer)",
    )
    parser.add_argument(
        "--skip-versions",
        nargs="+",
        help="Versions to skip",
        default=[],
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop testing on first failure",
    )

    args = parser.parse_args()

    # Determine which versions to test
    if args.versions:
        versions_to_test = args.versions
    elif args.min_version:
        # Find versions >= min_version
        min_ver = args.min_version
        versions_to_test = [v for v in DEFAULT_VERSIONS if v == "latest" or v >= min_ver]
    else:
        versions_to_test = DEFAULT_VERSIONS

    # Filter out skipped versions
    versions_to_test = [v for v in versions_to_test if v not in args.skip_versions]

    if not versions_to_test:
        print("No versions to test!")
        return 1

    project_root = Path(__file__).parent.parent
    print(f"Testing polars-nexpresso against {len(versions_to_test)} Polars version(s)")
    print(f"Versions: {', '.join(versions_to_test)}")

    results = {}
    for version in versions_to_test:
        success, output, actual_version = test_polars_version(version, project_root)
        results[version] = (success, output, actual_version)

        if not success and args.stop_on_failure:
            print(f"\n❌ Stopping on first failure (Polars {actual_version})")
            break

    # Print summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")

    passed = sum(1 for success, _, _ in results.values() if success)
    failed = len(results) - passed

    for version, (success, _, actual_version) in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"Polars {actual_version:10s}: {status}")

    print(f"\nTotal: {len(results)} | Passed: {passed} | Failed: {failed}")

    if failed > 0:
        print("\nFailed versions:")
        for version, (success, output, actual_version) in results.items():
            if not success:
                print(f"\n  Polars {actual_version}:")
                print(f"  {output[:200]}...")  # Show first 200 chars

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

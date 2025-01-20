import argparse
import json
import re
from pathlib import Path
from typing import Optional, TypedDict, Dict
from packaging.specifiers import SpecifierSet
from packaging.requirements import Requirement
from pypi_simple import PyPISimple, ACCEPT_JSON_ONLY
from packaging.version import Version


class VersionResult(TypedDict):
    solveable: bool
    error: Optional[str]
    error_type: Optional[str]


class PackageResults(TypedDict):
    versions: Dict[str, VersionResult]


# Packages to skip when analyzing dependencies
SKIP_PACKAGES = {
    # Python implementations
    "python",
    "python-dev",
    "python-all",
    "python-all-dev",
    "cpython",
    "pypy",
    "pypy3",
    "jython",
    "ironpython",
    "micropython",
    # Python core packages
    "pip",
    "setuptools",
    "wheel",
    "build",
    "virtualenv",
    "venv",
    # Build tools
    "gcc",
    "clang",
    "cmake",
    "make",
    "ninja",
}


def should_skip_package(package: str) -> bool:
    """Check if a package should be skipped."""
    package = package.lower()

    if package in SKIP_PACKAGES:
        return True

    if package.startswith("_"):
        return True
    return False


def parse_error_requirements(error: str) -> list[tuple[str, str]]:
    """Parse error message to extract package requirements.
    Returns list of (package_name, version_spec) tuples."""

    # Common patterns in conda solver errors
    patterns = [
        # Pattern from the example: "nothing provides jupyter-client <8.0 needed by"
        r"nothing provides ([a-zA-Z0-9\-_]+) (.+) needed by",
        # Add more patterns as we discover them
        r"requires ([a-zA-Z0-9\-_]+) ([<>=!~]+[0-9][a-zA-Z0-9\-_.*]+)",
    ]

    requirements = []
    for pattern in patterns:
        matches = re.finditer(pattern, error)
        for match in matches:
            package = match.group(1)
            version_spec = match.group(2)
            requirements.append((package, version_spec))

    return requirements


def read_requirements(requirements_file: Path) -> dict[str, list[str]]:
    """Read and parse existing requirements file.
    Returns dict of package name to list of exact versions."""
    requirements: dict[str, list[str]] = {}

    if not requirements_file.exists():
        return requirements

    with open(requirements_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            req = Requirement(line)
            if len(req.specifier) != 1:
                raise ValueError(f"Invalid requirement: {line}")

            spec = next(iter(req.specifier))
            if spec.operator != "==":
                raise ValueError(f"Invalid requirement: {line}")

            if req.name not in requirements:
                requirements[req.name] = []
            requirements[req.name].append(spec.version)

    return requirements


def write_requirements(requirements: dict[str, list[str]], output_file: Path) -> None:
    """Write requirements to file in requirements.txt format.
    Writes one line per version for each package."""
    with open(output_file, "w") as f:
        for package, versions in sorted(requirements.items()):
            for version in sorted(versions):
                f.write(f"{package}=={version}\n")
        f.flush()


# Global cache for PyPI versions
PYPI_VERSION_CACHE: dict[str, list[Version]] = {}


def get_pypi_versions(package: str) -> list[Version]:
    """Get all versions for a package from PyPI, using cache if available."""
    if package in PYPI_VERSION_CACHE:
        return PYPI_VERSION_CACHE[package]

    versions: list[Version] = []
    with PyPISimple(accept=ACCEPT_JSON_ONLY) as client:
        try:
            page = client.get_project_page(package)
            for pkg in page.packages:
                if pkg.version:
                    try:
                        version = Version(pkg.version)
                        versions.append(version)
                    except Exception:
                        continue
            # Sort in descending order to get highest matching version
            versions.sort(reverse=True)
            PYPI_VERSION_CACHE[package] = versions
            return versions
        except Exception as e:
            print(f"Error fetching versions for {package}: {e}")
            # Cache empty list to avoid repeated failed requests
            PYPI_VERSION_CACHE[package] = []
            return []


def find_best_matching_version(package: str, spec: SpecifierSet) -> str | None:
    """Find highest version from PyPI that matches the given specifier.
    Excludes prerelease versions (alpha, beta, rc, etc)."""
    versions = get_pypi_versions(package)
    for version in versions:
        # Skip prerelease versions
        if version.is_prerelease:
            continue
        if version in spec:
            return str(version)
    return None


def analyze_package_errors(
    solve_results: Path, packages: list[str] | None = None, output: Path | None = None
) -> None:
    """Analyze errors for packages and determine version requirements."""
    # Load solve results
    with open(solve_results) as f:
        results = json.load(f)

    # Load existing requirements if output file specified
    requirements = read_requirements(output) if output else {}

    # Get list of packages to analyze
    packages_to_analyze = packages if packages else list(results.keys())

    total = len(packages_to_analyze)
    print(f"Analyzing {total} packages...")

    for i, package in enumerate(sorted(packages_to_analyze), 1):
        if package not in results:
            print(f"[{i}/{total}] Package {package} not found in results, skipping")
            continue

        print(f"\n[{i}/{total}] Analysis for {package}:")

        # Collect all version specs for each dependency
        dependency_ranges: dict[str, list[SpecifierSet]] = {}

        for version, result in results[package]["versions"].items():
            if result["solveable"]:
                continue

            if not result["error"]:
                continue

            requirements_list = parse_error_requirements(result["error"])
            for dep_name, version_spec in requirements_list:
                # Skip certain packages
                if should_skip_package(dep_name):
                    continue

                if dep_name not in dependency_ranges:
                    dependency_ranges[dep_name] = []
                try:
                    spec = SpecifierSet(version_spec)
                    dependency_ranges[dep_name].append(spec)
                except Exception as e:
                    print(f"  Error parsing spec {version_spec} for {dep_name}: {e}")

        if not dependency_ranges:
            print("  No dependency requirements found")
            continue

        print("  Required dependencies:")

        # For each dependency, check ranges against existing requirements
        for dep_name, ranges in dependency_ranges.items():
            print(f"    {dep_name}:")
            existing_versions = requirements.get(dep_name, [])
            if existing_versions:
                print(f"      Existing versions: {', '.join(existing_versions)}")

            new_versions = set()
            for range_spec in ranges:
                print(f"      Checking range: {range_spec}")

                # Check if any existing version satisfies this range
                range_satisfied = any(
                    Version(ver) in range_spec for ver in existing_versions
                )

                if not range_satisfied:
                    # Find best matching version from PyPI
                    best_version = find_best_matching_version(dep_name, range_spec)
                    if best_version:
                        print(f"      Adding version: {best_version}")
                        new_versions.add(best_version)
                    else:
                        print(f"      No matching version found for {range_spec}")

            # Add new versions to requirements
            if new_versions:
                if dep_name not in requirements:
                    requirements[dep_name] = []
                requirements[dep_name].extend(sorted(new_versions))
                print(f"      Final versions: {', '.join(requirements[dep_name])}")

        # Write updated requirements after each package
        if output:
            write_requirements(requirements, output)

    if output:
        print(f"\nFinal requirements written to {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze package solve errors to determine version requirements"
    )
    parser.add_argument(
        "-i", "--input", type=Path, help="Path containing the solve.json file"
    )
    parser.add_argument(
        "-p",
        "--package",
        action="append",
        help="Package to analyze (can be repeated, if not specified analyzes all packages)",
        dest="packages",
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="Path to requirements.txt file to update"
    )
    args = parser.parse_args()

    analyze_package_errors(args.input, args.packages, args.output)


if __name__ == "__main__":
    main()

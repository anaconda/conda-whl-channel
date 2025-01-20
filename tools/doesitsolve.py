import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, TypedDict


class VersionResult(TypedDict):
    solveable: bool
    error: Optional[str]
    error_type: Optional[str]


class PackageResults(TypedDict):
    versions: Dict[str, VersionResult]


def load_results(output: Path) -> Dict[str, PackageResults]:
    """Load existing results from JSON file or return empty dict if file doesn't exist."""
    if output.exists():
        with open(output) as f:
            return json.load(f)
    return {}


def save_results(results: Dict[str, PackageResults], output: Path) -> None:
    """Save results to JSON file."""
    with open(output, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.flush()


def get_channel_packages(
    channel: str, filter_packages: list[str] | None = None
) -> Dict[str, list[str]]:
    """Get dict of package names to their versions in the specified conda channel."""
    try:
        cmd = ["conda", "search", "-c", channel, "--override-channels", "--json"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        packages_data = json.loads(result.stdout)

        # Organize by package name -> list of versions
        packages: Dict[str, list[str]] = {}
        for pkg_data in packages_data.values():
            for item in pkg_data:
                name = item["name"]
                # Skip if we're filtering and this package isn't in the filter list
                if filter_packages and name not in filter_packages:
                    continue
                version = item["version"]
                if name not in packages:
                    packages[name] = []
                if version not in packages[name]:
                    packages[name].append(version)

        return packages
    except subprocess.CalledProcessError as e:
        print(f"Error listing packages: {e.stderr}", file=sys.stderr)
        sys.exit(1)


def check_package_solveability(
    package: str, version: str, channel: str, solve_channels: list[str]
) -> VersionResult:
    """Check if a specific package version can be solved using given channels."""
    cmd = [
        "conda",
        "create",
        "--dry-run",
        "--json",
        "-n",
        "test_env",
    ]

    # Add solve channels in order
    for solve_channel in solve_channels:
        cmd.extend(["-c", solve_channel])

    cmd.extend(
        [
            "-c",
            channel,
            "--override-channels",
            f"{package}={version}",
        ]
    )

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {"solveable": True, "error": None, "error_type": None}
    except subprocess.CalledProcessError as e:
        try:
            error_data = json.loads(e.stdout)
            error_msg = error_data.get("message", str(e.stdout))
            error_type = error_data.get("exception_name", type(e).__name__)
        except json.JSONDecodeError:
            print(e.stdout)
            error_msg = str(e.stdout)
            error_type = type(e).__name__
        except Exception as e:
            print(e)
            error_msg = str(e)
            error_type = type(e).__name__
        return {"solveable": False, "error": error_msg, "error_type": error_type}


def get_failed_versions(results: Dict[str, PackageResults]) -> Dict[str, list[str]]:
    """Get dict of package names to list of versions that failed to solve."""
    failed: Dict[str, list[str]] = {}
    for package, pkg_data in results.items():
        failed_versions = [
            version
            for version, result in pkg_data["versions"].items()
            if not result["solveable"]
        ]
        if failed_versions:
            failed[package] = failed_versions
    return failed


def main():
    parser = argparse.ArgumentParser(
        description="Check solveability of conda packages in a channel"
    )
    _ = parser.add_argument(
        "-c", "--channel", required=True, help="Conda channel to analyze"
    )
    _ = parser.add_argument(
        "-o", "--output", type=Path, help="JSON file to store results", required=True
    )
    _ = parser.add_argument(
        "-s",
        "--solve-channel",
        action="append",
        help="Channel to use for solving (can be repeated, defaults to ['defaults'])",
        dest="solve_channels",
    )
    _ = parser.add_argument(
        "-p",
        "--package",
        action="append",
        help="Package to analyze (can be repeated, defaults to all packages)",
        dest="packages",
    )
    _ = parser.add_argument(
        "--skip",
        action="append",
        help="Package to skip (can be repeated)",
        default=[],
    )
    _ = parser.add_argument(
        "--recheck",
        action="store_true",
        help="Re-analyze packages even if they have cached results",
        default=False,
    )
    _ = parser.add_argument(
        "--failed",
        action="store_true",
        help="Only analyze packages that previously failed",
        default=False,
    )
    args = parser.parse_args()

    # Default to ["defaults"] if no solve channels specified
    solve_channels = args.solve_channels or ["defaults"]

    # Load existing results
    results = load_results(args.output)

    # Get all packages and their versions from channel
    print(f"Fetching package list from channel '{args.channel}'...")
    packages = get_channel_packages(args.channel, args.packages)

    # If --failed is specified, filter to only previously failed versions
    if args.failed and results:
        failed_versions = get_failed_versions(results)
        packages = {
            pkg: [ver for ver in versions if ver in failed_versions.get(pkg, [])]
            for pkg, versions in packages.items()
            if pkg in failed_versions
        }
        # Remove packages with no remaining versions
        packages = {pkg: vers for pkg, vers in packages.items() if vers}
        print(f"Filtered to {len(packages)} packages with failed versions")

    # Remove skipped packages (after filtering failed versions)
    if args.skip:
        for pkg in args.skip:
            if pkg in packages:
                del packages[pkg]
                print(f"Skipping package: {pkg}")

    total_packages = len(packages)
    total_versions = sum(len(versions) for versions in packages.values())
    print(
        f"Found {total_packages} packages with {total_versions} total versions to analyze"
    )
    # Check each package version
    pkg_count = 0
    for package, versions in packages.items():
        pkg_count += 1

        if package not in results:
            results[package] = {"versions": {}}

        for version in versions:
            if version in results[package]["versions"] and not args.recheck:
                print(
                    f"[{pkg_count}/{total_packages}] Skipping {package}={version} (already analyzed)"
                )
                continue

            print(f"[{pkg_count}/{total_packages}] Analyzing {package}={version}...")
            result = check_package_solveability(
                package, version, args.channel, solve_channels
            )
            results[package]["versions"][version] = result

            # Save results after each version to maintain progress
            save_results(results, args.output)

            status = "✓" if result["solveable"] else "✗"
            print(f"[{pkg_count}/{total_packages}] {package}={version}: {status}")

    print("\nAnalysis complete!")

    # Calculate statistics
    solved_versions = sum(
        1
        for pkg in results.values()
        for ver in pkg["versions"].values()
        if ver["solveable"]
    )
    solved_packages = sum(
        1
        for pkg in results.values()
        if all(ver["solveable"] for ver in pkg["versions"].values())
    )

    print(f"Fully solveable packages: {solved_packages}/{total_packages}")
    print(f"Solveable versions: {solved_versions}/{total_versions}")


if __name__ == "__main__":
    main()

import argparse
import json
import subprocess
from typing import Dict, List, Optional
from pypi_simple import PyPISimple, ACCEPT_JSON_ONLY
from packaging.version import Version, InvalidVersion

pypi_packages: set[str] | None = None


def parse_channel(channel: str = "defaults") -> Dict[str, str]:
    """
    Parse a conda channel to get a dictionary of package names and their latest versions.

    Args:
        channel (str): Name of the conda channel to parse. Defaults to 'defaults'.

    Returns:
        Dict[str, str]: Dictionary mapping package names to their latest versions
    """
    # Run conda search with JSON output
    cmd = ["conda", "search", "--json", "--channel", channel]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    # Parse JSON output
    packages_data = json.loads(result.stdout)

    # Extract latest version for each package
    latest_versions = {}
    for package_name, package_info in packages_data.items():
        if package_info:  # Check if package has any versions
            # Skip non-"python" packages
            deps = package_info[0].get("depends", ())
            python_dep = [d.startswith("python") for d in deps]
            if not any(python_dep):
                print(f"Non-python package: {package_name}")
                continue
            # Sort versions and get the latest one
            def key_fn(x: dict[str, str]) -> Version:
                v = convert_package_version(x.get("version", "0.0.0"))
                return Version(v) if v else Version("0.0.0")

            latest_version = sorted(package_info, key=key_fn, reverse=True)[0]
            latest_versions[package_name] = latest_version["version"]

    return latest_versions


def pypi_index() -> set[str]:
    global pypi_packages
    if pypi_packages is None:
        client = PyPISimple(accept=ACCEPT_JSON_ONLY)
        pypi_packages = set(client.get_index_page().projects)
    return pypi_packages


def convert_package_name(name: str) -> Optional[str]:
    """
    Convert conda package name to PyPI package name.
    Returns None if package cannot be found on PyPI.

    Args:
        name: The conda package name to convert

    Returns:
        Optional[str]: The PyPI package name if found, None otherwise
    """
    KNOWN_MAPPINGS: dict[str, str] = {
        "scikit-learn": "sklearn",
        "python-dateutil": "dateutil",
    }

    if name in KNOWN_MAPPINGS:
        return KNOWN_MAPPINGS[name]

    if name in pypi_index():
        return name

    if name.startswith("python-"):
        modified_name = name.replace("python-", "python_")
        if modified_name in pypi_index():
            return modified_name

    if name.startswith("py-"):
        modified_name = name[3:]
        if modified_name in pypi_index():
            return modified_name

    print(f"could not find {name} on pypi")
    return None


def convert_package_version(version: str) -> str | None:
    # In a loop, try to parse the version as a string, otherwise strip off the last characters until a .
    while len(version) > 0:
        try:
            _ = Version(version)
            return version
        except InvalidVersion:
            new_version = version.rsplit(".", 1)[0]
            if new_version == version:
                return None
            version = new_version
    return None


def generate_requirements(channel_data: dict[str, str]) -> str:
    packages: set[str] = set()
    for name, version in channel_data.items():
        pypi_name = convert_package_name(name)
        if pypi_name:
            pypi_version = convert_package_version(version)
            if pypi_version:
                packages.add(f"{pypi_name}>{pypi_version}")
    return "\n".join(sorted(packages))


def main():
    parser = argparse.ArgumentParser(
        description="Generate requirements.txt with packages from conda defaults channel"
    )
    _ = parser.add_argument(
        "--channel",
        default="defaults",
        help="Conda channel to parse (default: defaults)",
    )
    _ = parser.add_argument(
        "--output",
        default="requirements.txt",
        help="Output file path (default: requirements.txt)",
    )

    args = parser.parse_args()
    requirements = generate_requirements(parse_channel(args.channel))
    with open(args.output, "w") as f:
        f.write(requirements)


if __name__ == "__main__":
    main()

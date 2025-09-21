import argparse
from pathlib import Path
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import parse_wheel_filename
from typing import Dict, List


def parse_requirements_file(file_path: Path) -> Dict[str, list[SpecifierSet]]:
    """Parse requirements file into dict of package name to list of SpecifierSets."""
    specs: Dict[str, list[SpecifierSet]] = {}
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            req = Requirement(line)
            if req.name not in specs:
                specs[req.name] = []
            specs[req.name].append(req.specifier)
    return specs


def get_matching_wheels(specs: Dict[str, list[SpecifierSet]], wheels_dir: Path) -> List[Path]:
    """Find all wheel files that match the given specifiers.
    
    Args:
        specs: Dictionary mapping package names to lists of SpecifierSets
        wheels_dir: Directory containing wheel files
        
    Returns:
        List of Path objects for matching wheel files
    """
    wheels_to_parse = []
    
    # Loop through every *.whl file in the wheels directory
    for wheel_path in wheels_dir.glob("*.whl"):
        try:
            # Parse the wheel filename to extract package name and version
            name, version, build, tags = parse_wheel_filename(wheel_path.name)
            
            # Check if this package name matches any of our specs
            if name in specs:
                # Check if the version satisfies any of the specifiers for this package
                for specifier_set in specs[name]:
                    if version in specifier_set:
                        wheels_to_parse.append(wheel_path)
                        break  # Found a match, no need to check other specifiers for this package
                        
        except Exception as e:
            # Skip wheels that can't be parsed
            print(f"Warning: Could not parse wheel {wheel_path.name}: {e}")
            continue
    
    return wheels_to_parse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate conda repodata from wheels.")
    parser.add_argument(
        "-r",
        "--requirements",
        type=str,
        required=True,
        help="Requirements.txt file which contains the packages to be generated",
    )
    parser.add_argument(
        "-w",
        "--wheels-dir",
        type=str,
        required=True,
        help="Directory containing wheel files",
    )

    args = parser.parse_args()

    file_path = Path(args.requirements)
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find {file_path}")
    
    wheels_dir = Path(args.wheels_dir)
    if not wheels_dir.exists():
        raise FileNotFoundError(f"Could not find {wheels_dir}")
    
    specs = parse_requirements_file(file_path)
    matching_wheels = get_matching_wheels(specs, wheels_dir)
    
    print(f"Found {len(matching_wheels)} matching wheels:")
    for wheel in matching_wheels:
        print(f"  {wheel}")

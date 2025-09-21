import argparse
from pathlib import Path
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from typing import Dict


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
    
    print("todo")

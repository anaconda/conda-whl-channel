import argparse
import hashlib
import json
import os
import zipfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import parse_wheel_filename, canonicalize_name
from packaging.metadata import Metadata
from typing import Dict, List, Optional, Any
import requests


@lru_cache(maxsize=1)
def get_py_to_conda_mapping() -> Dict[str, str]:
    """
    Download and cache the grayskull mapping from GitHub.
    Returns a dictionary mapping PyPI package names to conda package names.
    """
    url = "https://raw.githubusercontent.com/prefix-dev/parselmouth/refs/heads/main/files/mapping_as_grayskull.json"
    
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    grayskull_mapping = response.json()
    
    # Filter out entries where key equals value and reverse the mapping
    # grayskull mapping is conda -> pypi, we need pypi -> conda
    py_to_conda = {}
    for conda_name, pypi_name in grayskull_mapping.items():
        if conda_name != pypi_name:  # Filter out where key equals value
            py_to_conda[pypi_name] = conda_name
    
    print(f"Loaded {len(py_to_conda)} PyPI to conda mappings from grayskull")
    return py_to_conda


def py_to_conda_name(name: str) -> str:
    canonical_name = canonicalize_name(name)
    mapping = get_py_to_conda_mapping()
    conda_name = mapping.get(canonical_name, canonical_name)
    return conda_name


@dataclass
class CondaPackageMetadata:
    # https://docs.conda.io/projects/conda/en/stable/user-guide/concepts/pkg-specs.html
    build: str = "0"
    build_number: int = 0
    depends: list[str] = field(default_factory=list)
    filename: str = ""
    md5: str | None = None
    name: str = ""
    noarch: str | None = None
    sha256: str | None = None
    size: int = 0
    subdir: str | None = None
    timestamp: int = 0
    version: str = ""

    @property
    def repodata_entry(self) -> dict[Any, Any]:
        # these are required
        entry = {
            "build": self.build,
            "build_number": self.build_number,
            "depends": self.depends,
            "name": self.name,
            "size": self.size,
            "timestamp": self.timestamp,
            "version": self.version,
        }
        for key in ["md5", "noarch", "sha256", "subdir"]:
            value = getattr(self, key)
            if value is not None:
                entry[key] = value
        return entry


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


def _ensure_pure_python(wheels_paths: List[Path]) -> None:
    """Ensure all wheels are pure Python (noarch) packages.
    
    Args:
        wheels_paths: List of Path objects for wheel files to check
        
    Raises:
        SystemExit: If any wheel is not a pure Python package
    """
    for wheel_path in wheels_paths:
        try:
            name, version, build, tags = parse_wheel_filename(wheel_path.name)
            
            # Check if this is a pure Python wheel
            # Pure Python wheels should have 'py2.py3-none-any' or similar patterns
            # that indicate they work on any platform and any Python version
            is_pure_python = False
            
            for tag in tags:
                # Check if the tag indicates pure Python (py2.py3-none-any pattern)
                if (tag.interpreter in ['py2.py3', 'py3'] and 
                    tag.abi == 'none' and 
                    tag.platform == 'any'):
                    is_pure_python = True
                    break
            
            if not is_pure_python:
                print(f"Error: Wheel {wheel_path.name} is not a pure Python package.")
                print(f"  Tags: {[str(tag) for tag in tags]}")
                print("Only pure Python (noarch) packages are supported.")
                raise SystemExit(1)
                
        except Exception as e:
            print(f"Error: Could not parse wheel {wheel_path.name}: {e}")
            raise SystemExit(1)


def extract_wheel_metadata(wheel_path: Path) -> Dict[str, Optional[str]]:
    """Extract metadata from a wheel file using packaging.metadata.from_email.
    
    Args:
        wheel_path: Path to the wheel file
        
    Returns:
        Dictionary containing name, version, requires_dist, and requires_python
    """
    metadata = {
        'name': None,
        'version': None,
        'requires_dist': None,
        'requires_python': None
    }
    
    try:
        with zipfile.ZipFile(wheel_path, 'r') as wheel:
            # Look for METADATA file in the wheel
            metadata_files = [f for f in wheel.namelist() if f.endswith('METADATA')]
            
            if not metadata_files:
                print(f"Warning: No METADATA file found in {wheel_path.name}")
                return metadata
            
            # Read the METADATA file content
            with wheel.open(metadata_files[0]) as metadata_file:
                raw_metadata = metadata_file.read().decode('utf-8')
            
            # Use packaging.metadata.from_email to parse the metadata
            mdata = Metadata.from_email(raw_metadata, validate=False)
            
            metadata['name'] = mdata.name
            metadata['version'] = str(mdata.version) if mdata.version else None
            metadata['requires_dist'] = [str(req) for req in mdata.requires_dist] if mdata.requires_dist else None
            metadata['requires_python'] = str(mdata.requires_python) if mdata.requires_python else None
                        
    except Exception as e:
        print(f"Error reading metadata from {wheel_path.name}: {e}")
    
    return metadata


def get_file_sha_and_size(file_path: Path) -> tuple[str, int]:
    """Get SHA256 hash and size of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    file_size = os.path.getsize(file_path)
    return sha256_hash.hexdigest(), file_size


def repodata_for_package(wheel_path: Path) -> CondaPackageMetadata:
    """Generate conda repodata entry for a wheel package.
    
    Args:
        wheel_path: Path to the wheel file
        
    Returns:
        CondaPackageMetadata object representing the conda package
    """
    # Extract metadata from the wheel
    metadata = extract_wheel_metadata(wheel_path)
    
    # Parse wheel filename to get build info
    name, version, build, tags = parse_wheel_filename(wheel_path.name)
    
    # Convert pip name to conda name
    conda_name = py_to_conda_name(metadata['name'] or name)
    
    # Create build string from wheel filename components
    # Extract python_tag, abi_tag, platform_tag from filename like gen_repo.py does
    filename_parts = wheel_path.name.removesuffix(".whl").rsplit("-", maxsplit=3)
    if len(filename_parts) == 4:
        _, python_tag, abi_tag, platform_tag = filename_parts
        build_string = f"{python_tag}_{abi_tag}_{platform_tag}"
    else:
        # Fallback if parsing fails
        build_string = "py3_0"
    
    # Get file hash and size
    sha256, size = get_file_sha_and_size(wheel_path)
    
    # Create the conda package metadata
    # For now, assume no dependencies as requested
    depends = []
    
    # Add Python dependency if specified in metadata
    if metadata['requires_python']:
        depends.append(f"python {metadata['requires_python']}")
    else:
        depends.append("python")
    
    return CondaPackageMetadata(
        filename=wheel_path.name,
        build=build_string,
        build_number=0,
        depends=depends,
        name=conda_name,
        sha256=sha256,
        version=metadata['version'] or version,
        noarch="python",
        subdir="noarch",
        size=size,
        timestamp=0,
    )


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
    
    # Ensure all wheels are pure Python (noarch) packages
    _ensure_pure_python(matching_wheels)
    
    print(f"Found {len(matching_wheels)} matching wheels:")
    for wheel in matching_wheels:
        print(f"\nWheel: {wheel}")
        repodata = repodata_for_package(wheel)
        print(f"Repodata entry:")
        print(json.dumps(repodata.repodata_entry, indent=2))

import argparse
import codecs
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
from packaging.markers import Marker
from typing import Dict, List, Optional, Any
import requests
import markerpry


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


# Environment definitions for marker evaluation
GENERIC_ENV = {
    "implementation_name": ("cpython", ),
    "platform_python_implementation": ("CPython", ),
}

PLATFORM_SPECIFIC_ENV = {
    "osx-arm64": {
        "implementation_name": ("cpython", ),
        "os_name": ("posix", ),
        "platform_system": ("Darwin", ),
        "platform_python_implementation": ("CPython", ),
        "platform_machine": ("arm64", ),
        "sys_platform": ("darwin", ),
    },
}

# Metapackage storage
META_PKGS = {}
META_SHA_256 = None
META_SIZE = None
HASH_LENGTH = 8


def get_file_sha_and_size(file_path: Path) -> tuple[str, int]:
    """Get SHA256 hash and size of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    file_size = os.path.getsize(file_path)
    return sha256_hash.hexdigest(), file_size


def register_metapackages(pkg_name, pos_deps, neg_deps):
    """Register metapackages for conditional dependencies."""
    global META_SHA_256, META_SIZE
    if META_SHA_256 is None or META_SIZE is None:
        project_path = Path(__file__).parent.parent / "sample-1.0-0.tar.bz2"
        if not project_path.exists():
            raise FileNotFoundError(f"Could not find {project_path}")
        META_SHA_256, META_SIZE = get_file_sha_and_size(project_path)
    short_hash = META_SHA_256[:8]
    pos_pkg_name = f"{pkg_name}-1.1-true_1_{short_hash}.tar.bz2"
    META_PKGS[pos_pkg_name] = {
        "build": f"true_1_{short_hash}",
        "build_number": 1,
        "depends": pos_deps,
        "name": pkg_name,
        "noarch": "generic",
        "sha256": META_SHA_256,
        "size": META_SIZE,
        "subdir": "noarch",
        "timestamp": 0,
        "version": "1.1",
    }
    neg_pkg_name = f"{pkg_name}-0.1-false_0_{short_hash}.tar.bz2"
    META_PKGS[neg_pkg_name] = {
        "build": f"false_0_{short_hash}",
        "build_number": 0,
        "depends": neg_deps,
        "name": pkg_name,
        "noarch": "generic",
        "sha256": META_SHA_256,
        "size": META_SIZE,
        "subdir": "noarch",
        "timestamp": 0,
        "version": "0.1",
    }


def make_metapkgs(
        conda_deps: list[str],
        marker: Marker,
        platform: str,
        requested_extra: Optional[str],
    ) -> list[str]:
    """Create metapackages for conditional dependencies."""
    tree = markerpry.parse_marker(marker)
    if requested_extra is not None:
        tree = tree.evaluate({"extra": (requested_extra, )})
    if "extra" in tree:
        return []
    
    # evaluate with CPython environment
    tree = tree.evaluate(GENERIC_ENV)
    if tree.resolved:
        if tree:
            return conda_deps
        return []
    
    # handle platform specific evaluation when possible
    if (
        "os_name" in tree
        or "platform_system" in tree
        or "sys_platform" in tree
        or "platform_machine" in tree
    ):
        if platform == "noarch":
            raise ValueError(f"dependency {conda_deps} is arch-specific but platform is noarch")
        env = PLATFORM_SPECIFIC_ENV.get(platform, {})
        tree = tree.evaluate(env)
        if tree.resolved:
            if not tree:
                return []
            else:
                # No metapackage is needed, we can just treat this as a normal dependency
                return conda_deps
    
    # create meta-packages to encode conditional
    dep_hash_str = "&".join(d for d in conda_deps)
    dep_hash = hashlib.sha1(dep_hash_str.encode()).hexdigest()[:HASH_LENGTH]
    marker_hash = hashlib.sha1(str(marker).encode()).hexdigest()[:HASH_LENGTH]
    meta_pkg_name = f"_c_{dep_hash}_{marker_hash}"
    
    MIRROR_OP = {
        ">": "<=",
        "<": ">=",
        ">=": "<",
        "<=": ">",
        "==": "!=",
        "!=": "==",
    }
    
    if isinstance(tree, markerpry.OperatorNode):
        right, left = tree.right, tree.left
        if (
            isinstance(right, markerpry.ExpressionNode)
            and right.lhs == "python_version"
            and isinstance(left, markerpry.ExpressionNode)
            and left.lhs == "python_version"
        ):
            op1, value1 = left.comparator, left.rhs
            op2, value2 = right.comparator, right.rhs
            nop1 = MIRROR_OP[str(op1)]
            nop2 = MIRROR_OP[str(op2)]
            if tree.operator == "and":
                positive_deps = [
                    f"python {op1}{value1}",
                    f"python {op2}{value2}",
                    *conda_deps,
                ]
                negative_deps = [f"python {nop1}{value1}", f"python {nop2}{value2}"]
                register_metapackages(meta_pkg_name, positive_deps, negative_deps)
                return [meta_pkg_name]
            elif tree.operator == "or":
                return make_metapkgs(
                    conda_deps,
                    Marker(str(left)),
                    platform,
                    requested_extra,
                ) + make_metapkgs(
                    conda_deps,
                    Marker(str(right)),
                    platform,
                    requested_extra,
                )
        else:
            # Use De Morgan's laws to express:
            # OR will produce two packages, one for each clause
            # AND will produce a single package that depends on both clauses
            raise NotImplementedError(f"complex marker: {tree}")
    assert isinstance(tree, markerpry.ExpressionNode)
    variable, op, value = tree.lhs, tree.comparator, tree.rhs
    op = str(op)
    nop = MIRROR_OP[op]
    if str(variable) == "python_version" or str(variable) == "python_full_version":
        positive_dep = f"python {op}{value}"
        negative_dep = f"python {nop}{value}"
        positive_deps = [positive_dep, *conda_deps]
        negative_deps = [negative_dep]
        register_metapackages(meta_pkg_name, positive_deps, negative_deps)
        return [meta_pkg_name]
    else:
        raise NotImplementedError(f"unsupported marker: {marker}")


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
            # Handle empty or invalid requires_python gracefully
            if mdata.requires_python and str(mdata.requires_python).strip() and str(mdata.requires_python) != '<empty>':
                metadata['requires_python'] = str(mdata.requires_python)
            else:
                metadata['requires_python'] = None
                        
    except Exception as e:
        print(f"Error reading metadata from {wheel_path.name}: {e}")
    
    return metadata


def empty_repodata(subdir: str):
    """Create an empty repodata structure for a subdirectory."""
    return {
        "info": {"subdir": subdir},
        "packages": {},
        "packages.conda": {},
        "removed": [],
        "repodata_version": 1,
    }


def write_as_json_to_file(file_path: str, obj: Any):
    """Write an object as JSON to a file, matching conda's format."""
    with codecs.open(file_path, mode="wb", encoding="utf-8") as fo:
        json_str = json.dumps(
            obj,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
        fo.write(json_str)


def write_repodata(repodata: Dict[str, Dict[Any, Any]], repo_base_path: str):
    """Write repodata to files for each platform."""
    for platform, subdir_data in repodata.items():
        os.makedirs(f"{repo_base_path}/{platform}", exist_ok=True)
        write_as_json_to_file(f"{repo_base_path}/{platform}/repodata.json", subdir_data)


def parse_requires_dist(requires_dist: List[str], platform: str) -> List[str]:
    """Parse requires_dist dependencies and convert to conda dependencies.
    
    Args:
        requires_dist: List of requirement strings from wheel metadata
        platform: Target platform for dependency resolution
        
    Returns:
        List of conda dependency strings
    """
    conda_deps = []
    
    for req_str in requires_dist:
        try:
            req = Requirement(req_str)
            
            # TODO: Handle dependencies with extras with recursive wheel parsing
            # if req.extras:
                # print(f"Warning: Dependency has extras, using base name only: {req_str}")
            
            # Convert PyPI name to conda name
            conda_name = py_to_conda_name(req.name)
            
            # Create conda dependency string
            if req.specifier:
                conda_dep = f"{conda_name} {req.specifier}".strip()
            else:
                conda_dep = conda_name
            
            # Handle conditional dependencies with markers
            if req.marker:
                try:
                    conda_deps.extend(make_metapkgs([conda_dep], req.marker, platform, None))
                except Exception as e:
                    print(f"Warning: Could not process conditional dependency '{req_str}': {e}")
                    continue
            else:
                conda_deps.append(conda_dep)
            
        except Exception as e:
            print(f"Warning: Could not parse dependency '{req_str}': {e}")
            continue
    
    return conda_deps


def repodata_for_package(wheel_path: Path, platform: str) -> CondaPackageMetadata:
    """Generate conda repodata entry for a wheel package.
    
    Args:
        wheel_path: Path to the wheel file
        platform: Target platform for dependency resolution
        
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
    depends = []
    
    # Add Python dependency if specified in metadata
    if metadata['requires_python']:
        depends.append(f"python {metadata['requires_python']}")
    else:
        depends.append("python")
    
    # Parse and add other dependencies
    if metadata['requires_dist']:
        conda_deps = parse_requires_dist(metadata['requires_dist'], platform)
        depends.extend(conda_deps)
    
    return CondaPackageMetadata(
        filename=wheel_path.name,
        build=build_string,
        build_number=0,
        depends=depends,
        name=conda_name,
        sha256=sha256,
        version=metadata['version'] or version,
        noarch="python",
        subdir=platform,
        size=size,
        timestamp=0,
    )


def has_platform_specific_dependencies(requires_dist: List[str]) -> bool:
    """Check if a package has conditional dependencies that require platform-specific evaluation.
    
    Args:
        requires_dist: List of requirement strings from wheel metadata
        
    Returns:
        True if the package has platform-specific conditional dependencies
    """
    for req_str in requires_dist:
        try:
            req = Requirement(req_str)
            if req.marker:
                # Parse the marker to check for platform-specific conditions
                tree = markerpry.parse_marker(req.marker)
                if (
                    "os_name" in tree
                    or "platform_system" in tree
                    or "sys_platform" in tree
                    or "platform_machine" in tree
                ):
                    return True
        except Exception:
            continue
    return False


def separate_wheels_by_platform(wheels: List[Path]) -> Dict[str, List[Path]]:
    """Separate wheels into noarch and platform-specific lists based on conditional dependencies.
    
    Args:
        wheels: List of wheel files to process
        
    Returns:
        Dictionary mapping platform names to lists of wheel files
    """
    noarch_wheels = []
    platform_wheels = []
    
    for wheel_path in wheels:
        try:
            metadata = extract_wheel_metadata(wheel_path)
            if metadata['requires_dist'] and has_platform_specific_dependencies(metadata['requires_dist']):
                platform_wheels.append(wheel_path)
            else:
                noarch_wheels.append(wheel_path)
        except Exception as e:
            print(f"Warning: Could not process wheel {wheel_path.name}: {e}")
            # Default to noarch if we can't determine
            noarch_wheels.append(wheel_path)
    
    return {
        "noarch": noarch_wheels,
        "osx-arm64": platform_wheels
    }


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
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./repo",
        help="Output directory for repodata files (default: ./repo)",
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
    
    print(f"Found {len(matching_wheels)} matching wheels")
    
    # Separate wheels by platform based on conditional dependencies
    wheels_by_platform = separate_wheels_by_platform(matching_wheels)
    
    print(f"Noarch wheels: {len(wheels_by_platform['noarch'])}")
    print(f"Platform-specific wheels: {len(wheels_by_platform['osx-arm64'])}")
    
    # Create repodata structure
    repodata = {}
    
    # Process each platform
    for platform, wheels in wheels_by_platform.items():
        if not wheels:
            continue
            
        print(f"Processing {len(wheels)} wheels for platform: {platform}")
        
        # Create empty repodata for the platform
        subdir_data = empty_repodata(platform)
        packages = {}
        
        # Process each wheel and add to packages
        for i, wheel in enumerate(wheels, 1):
            if i % 100 == 0 or i == len(wheels):
                print(f"Progress: {i}/{len(wheels)} wheels processed for {platform}")
            try:
                conda_pkg = repodata_for_package(wheel, platform)
                packages[conda_pkg.filename] = conda_pkg.repodata_entry
            except Exception as e:
                print(f"Error processing {wheel.name} for {platform}: {e}")
                continue
        
        # Add metapackages for this platform
        # Metapackages are noarch, so include them in all platforms
        platform_metapackages = {k: v for k, v in META_PKGS.items() if v["subdir"] == platform or v["subdir"] == "noarch"}
        packages.update(platform_metapackages)
        
        # Add packages to repodata
        subdir_data["packages"] = packages
        repodata[platform] = subdir_data
    
    # Write repodata to files
    print(f"Writing repodata to {args.output_dir}")
    write_repodata(repodata, args.output_dir)
    
    total_packages = sum(len(data["packages"]) for data in repodata.values())
    print(f"Successfully generated repodata for {total_packages} packages across {len(repodata)} platforms")

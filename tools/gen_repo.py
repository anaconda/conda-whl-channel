import codecs
import json
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
import hashlib

from pypi_simple import PyPISimple, ACCEPT_JSON_ONLY

from packaging.requirements import Requirement
from packaging.metadata import parse_email, Metadata
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name


#if TYPE_CHECKING:
from typing import Optional, Dict, Any
from pypi_simple import DistributionPackage
from packaging.markers import Marker


HASH_LENGTH = 7  # git --short default


logger = logging.getLogger()


def empty_repodata(subdir: str):
    return {
        "info": {"subdir": subdir},
        "packages": {},
        "packages.conda": {},
        "removed": [],
        "repodata_version": 1,
    }

PY_TO_CONDA_NAME = {}

MIRROR_OP = {
    ">": "<=",
    "<": ">=",
    ">=": "<",
    "<=": ">",
    "==": "!=",
    "!=": "==",
}


META_SHA_256: Optional[str] = "31f456d67411418bce044e4be791e4f4490dff81065b9d83c8a42d6327fdacd0"
META_SIZE: Optional[int] = 157
META_PKGS = {}


def get_file_sha_and_size(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    file_size = os.path.getsize(file_path)
    return sha256_hash.hexdigest(), file_size


def register_metapackages(pkg_name, pos_deps, neg_deps):
    global META_SHA_256, META_SIZE
    if META_SHA_256 is None or META_SIZE is None:
        project_path = Path(__file__).parent.parent / 'sample-1.0-0.tar.bz2'
        if not project_path.exists():
            raise FileNotFoundError(f"Could not find {project_path}")
        META_SHA_256, META_SIZE = get_file_sha_and_size(project_path)
    pos_pkg_name = f"{pkg_name}-1.1-true_1.tar.bz2"
    META_PKGS[pos_pkg_name] = {
      "build": "true_1",
      "build_number": 1,
      "depends": pos_deps,
      "name": pkg_name,
      "noarch": "generic",
      "sha256": META_SHA_256,
      "size": META_SIZE,
      "subdir": "noarch",
      "timestamp": 0,
      "version": "1.1"
    }
    neg_pkg_name = f"{pkg_name}-0.1-false_0.tar.bz2"
    META_PKGS[neg_pkg_name] = {
      "build": "false_0",
      "build_number": 0,
      "depends": neg_deps,
      "name": pkg_name,
      "noarch": "generic",
      "sha256": META_SHA_256,
      "size": META_SIZE,
      "subdir": "noarch",
      "timestamp": 0,
      "version": "0.1"
    }


def make_metapkgs(conda_dep: str, marker: Marker) -> str:
    dep_hash = hashlib.sha1(conda_dep.encode()).hexdigest()[:HASH_LENGTH]
    marker_hash = hashlib.sha1(str(marker).encode()).hexdigest()[:HASH_LENGTH]
    meta_pkg_name = f"_c_{dep_hash}_{marker_hash}"
    markers = marker._markers
    if len(markers) != 1 or not isinstance(markers[0], tuple) or len(markers[0]) != 3:
        # User De Morgan's laws to express:
        # OR will produce two packages, one for each clause
        # AND will produce a single package that depends on both clauses
        raise NotImplementedError("complex marker")
    variable, op, value = markers[0]
    op = str(op)
    nop = MIRROR_OP[op]
    if str(variable) == "python_version":
        positive_dep = f"python {op}{value}"
        negative_dep = f"python {nop}{value}"
        positive_deps = [positive_dep, conda_dep]
        negative_deps = [negative_dep]
        register_metapackages(meta_pkg_name, positive_deps, negative_deps)
        return meta_pkg_name
    elif str(variable) == "platform_system":
        positive_dep = f"python {op}{value}"
        negative_dep = f"python {nop}{value}"
        positive_deps = [positive_dep, conda_dep]
        negative_deps = [negative_dep]
        register_metapackages(meta_pkg_name, positive_deps, negative_deps)

    else:
        breakpoint()


# inline version of
#from conda.gateways.disk.create import write_as_json_to_file
def write_as_json_to_file(file_path: str, obj: Any):
    with codecs.open(file_path, mode="wb", encoding="utf-8") as fo:
        json_str = json.dumps(
            obj, indent=2, sort_keys=True, separators=(",", ": "),
        )
        fo.write(json_str)


def py_to_conda_name(name: str) -> str:
    canonical_name = canonicalize_name(name)
    conda_name = PY_TO_CONDA_NAME.get(canonical_name, canonical_name)
    return conda_name


def py_to_conda_req(req: Requirement, seen_py_names: set[str]) -> str:
    conda_name = py_to_conda_name(req.name)
    conda_dep = f"{conda_name} {req.specifier}".strip()
    if req.marker:
        markers = req.marker._markers

        if any(isinstance(m, tuple) and str(m[0]) == "extra" for m in markers):
            logger.debug(f"skipping extra: {req}")
            return None
        # TODO handle evaluation when possible (non-universal wheels)
        conda_dep = make_metapkgs(conda_dep, req.marker)
        #logger.warning(f"including req with marker: '{req}'")
    if req.extras:
        pass
        breakpoint()
    seen_py_names.add(canonicalize_name(req.name))
    return conda_dep

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



class ProjectGenerator:

    def __init__(self, client: PyPISimple, project:str, platforms: set[str], spec: SpecifierSet | None):
        self.client = client
        self.project = project
        self.platforms = platforms
        self.spec = spec or SpecifierSet()
        self.seen_py_names = set()

    @property
    def conda_pkgs(self) -> list[CondaPackageMetadata]:
        project_page = self.client.get_project_page(project=self.project)
        conda_pkgs = []
        for pkg in project_page.packages:
            if pkg.package_type != "wheel":
                continue
            if pkg.version not in self.spec:
                continue
            support_platforms, depends_from_filename = self.info_from_filename(pkg.filename)
            if support_platforms.isdisjoint(self.platforms):
                logger.debug(f"Ignoring wheel for unsupported platform: {pkg.filename}")
                continue
            conda_pkg = self.repodata_for_pkg(pkg, support_platforms, depends_from_filename)
            conda_pkgs.extend(conda_pkg)
        return conda_pkgs

    def info_from_filename(self, filename: str) -> tuple[set[str], list[str]]:
        # https://peps.python.org/pep-0427/#file-name-convention
        _, python_tag, abi_tag, platform_tag = filename.removesuffix(".whl").rsplit("-", maxsplit=3)
        if abi_tag == "none" and platform_tag == "any":
            supported_platforms = set(["noarch"])
            depends_from_filename = []
            return supported_platforms, depends_from_filename

        supported_platforms = set()
        if "macosx" in platform_tag:
            if "universal2" in platform_tag:
                #supported_platforms.add("osx-arm64")
                supported_platforms.add("osx-64")
            if "universal" in platform_tag:
                supported_platforms.add("osx-64")
            if "x86_64" in platform_tag:
                supported_platforms.add("osx-64")
            if "arm64" in platform_tag:
                supported_platforms.add("osx-arm64")
        if "manylinux" in platform_tag:
            if "x86_64" in platform_tag:
                supported_platforms.add("linux-64")
            if "aarch64" in platform_tag:
                supported_platforms.add("linux-aarch64")
        if "win" in platform_tag:
            if "amd64" in platform_tag:
                supported_platforms.add("win-64")
            if "win32" in platform_tag:
                supported_platforms.add("win-32")

        if abi_tag == "abi3":
            assert python_tag.startswith("cp3")
            minor = python_tag[3:]
            depends_from_filename = [f"python >=3.{minor}"]
            return supported_platforms, depends_from_filename

        if abi_tag.endswith("t"):
            # TODO support free-threading
            return set(), []

        # TODO python 2 and ABI thats that have suffix (m, d)
        assert abi_tag.startswith("cp3")
        minor = int(abi_tag[3:])
        depends_from_filename = [f"python >=3.{minor},<3.{minor+1}.0a0"]
        return supported_platforms, depends_from_filename

    def repodata_for_pkg(
            self,
            pkg: DistributionPackage,
            supported_platforms: set[str],
            depends_from_filename: list[str],
        ) -> list[CondaPackageMetadata]:
        raw_metadata = self.client.get_package_metadata(pkg)
        depends = list(depends_from_filename)
        mdata = Metadata.from_email(raw_metadata, validate=False)
        if mdata.requires_python:
            py_dep = "python " + str(mdata.requires_python)
        else:
            py_dep = "python"
        depends.append(py_dep)
        if mdata.requires_dist:
            for py_req in mdata.requires_dist:
                conda_dep = py_to_conda_req(py_req, self.seen_py_names)
                if conda_dep:
                    depends.append(conda_dep)
        _, python_tag, abi_tag, platform_tag = pkg.filename.removesuffix(".whl").rsplit("-", maxsplit=3)
        build = f"{python_tag}_{abi_tag}_{platform_tag}"
        pkgs = []
        for subdir in supported_platforms:
            rdata = CondaPackageMetadata(
                filename=pkg.filename,
                build=build,
                depends=depends,
                name=py_to_conda_name(mdata.name),
                sha256=pkg.digests["sha256"],
                version=str(mdata.version),
                noarch="python",
                subdir=subdir,
                size=pkg.size,
            )
            pkgs.append(rdata)
        return pkgs





def create_repodata(
        projects: list[str],
        specs: dict[str, SpecifierSet],
        platforms: list[str],
        proc_dependencies: bool = False,
    ) -> Dict[str, Dict[Any, Any]]:
    client = PyPISimple(accept=ACCEPT_JSON_ONLY)
    conda_pkgs: list[CondaPackageMetadata] = []

    to_do_projects = set(projects)
    done_projects = set()

    while to_do_projects:
        project = to_do_projects.pop()
        done_projects.add(project)
        logger.info(f"creating repodata entries for project: {project}")
        spec = specs.get(project)
        gen = ProjectGenerator(client, project, platforms, spec)
        conda_pkgs.extend(gen.conda_pkgs)

        logger.debug(f"projects visited: {gen.seen_py_names}")
        if proc_dependencies:
            for to_add in gen.seen_py_names:
                if to_add not in done_projects:
                    to_do_projects.add(to_add)
        logger.debug(f"projects to visit: {to_do_projects}")

    repodata = {}
    for platform in platforms:
        packages = {k: v for k, v in META_PKGS.items() if v["subdir"] == platform}
        packages.update({pkg.filename: pkg.repodata_entry for pkg in conda_pkgs if pkg.subdir == platform})
        subdir_data = empty_repodata(platform)
        subdir_data["packages"] = packages
        subdir_data["info"]["subdir"] = platform
        repodata[platform] = subdir_data
    return repodata


def write_repodata(repodata: Dict[str, Dict[Any, Any]], repo_base_path: str):
    for platform, subdir_data in repodata.items():
        os.makedirs(f"{repo_base_path}/{platform}", exist_ok=True)
        write_as_json_to_file(f"{repo_base_path}/{platform}/repodata.json", subdir_data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    projects = [
        "flask",
        "imagesize"
    ]
    specs = {
        "imagesize": SpecifierSet(">1.4"),
        "markupsafe": SpecifierSet(">3.0.1"),
        "flask": SpecifierSet(">3.0.4"),
        'click': SpecifierSet("==8.1.7"),
        'jinja2': SpecifierSet("==3.1.4"),
        'blinker': SpecifierSet("==1.9.0"),
        'importlib-metadata': SpecifierSet("==8.5.0"),
        "werkzeug": SpecifierSet(">3.1.0"),
        'itsdangerous': SpecifierSet("==2.2.0"),
        "zipp": SpecifierSet("==3.21.0"),
        "typing-extensions": SpecifierSet("==4.12.2"),
    }
    platforms = set(("osx-arm64", "noarch"))
    repodata = create_repodata(projects, specs, platforms, proc_dependencies=True)
    write_repodata(repodata, "./repo")

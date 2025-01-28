import codecs
import json
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
import hashlib
import argparse

import requests_cache
from pypi_simple import PyPISimple, ACCEPT_JSON_ONLY

from packaging.requirements import Requirement
from packaging.metadata import parse_email, Metadata
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

import markerpry


# if TYPE_CHECKING:
from typing import Optional, Dict, Any
from pypi_simple import DistributionPackage
from packaging.markers import Marker
from requests import Session


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


PY_TO_CONDA_NAME = {
    "async-generator": "async_generator",
    "backports-functools-lru-cache": "backports.functools_lru_cache",
    "backports-lzma": "backports.lzma",
    "backports-os": "backports.os",
    "backports-shutil-get-terminal-size": "backports.shutil_get_terminal_size",
    "backports-shutil-which": "backports.shutil_which",
    "backports-tarfile": "backports.tarfile",
    "backports-tempfile": "backports.tempfile",
    "backports-weakref": "backports.weakref",
    "backports-zoneinfo": "backports.zoneinfo",
    "backports-abc": "backports_abc",
    "boolean-py": "boolean.py",
    "category-encoders": "category_encoders",
    "clr-loader": "clr_loader",
    "cmake-setuptools": "cmake_setuptools",
    "cx-oracle": "cx_oracle",
    "dash-cytoscape": "dash_cytoscape",
    "essential-generators": "essential_generators",
    "et-xmlfile": "et_xmlfile",
    "factory-boy": "factory_boy",
    "flask-cors": "flask_cors",
    "func-timeout": "func_timeout",
    "huggingface-hub": "huggingface_hub",
    "idna-ssl": "idna_ssl",
    "importlib-resources": "importlib_resources",
    "interface-meta": "interface_meta",
    "ipython-genutils": "ipython_genutils",
    "jaraco-classes": "jaraco.classes",
    "jaraco-collections": "jaraco.collections",
    "jaraco-context": "jaraco.context",
    "jaraco-functools": "jaraco.functools",
    "jaraco-itertools": "jaraco.itertools",
    "jaraco-test": "jaraco.test",
    "jaraco-text": "jaraco.text",
    "jupyter-bokeh": "jupyter_bokeh",
    "jupyter-client": "jupyter_client",
    "jupyter-console": "jupyter_console",
    "jupyter-core": "jupyter_core",
    "jupyter-dashboards-bundlers": "jupyter_dashboards_bundlers",
    "jupyter-events": "jupyter_events",
    "jupyter-kernel-gateway": "jupyter_kernel_gateway",
    "jupyter-server": "jupyter_server",
    "jupyter-server-fileid": "jupyter_server_fileid",
    "jupyter-server-terminals": "jupyter_server_terminals",
    "jupyter-server-ydoc": "jupyter_server_ydoc",
    "jupyter-telemetry": "jupyter_telemetry",
    "jupyter-ydoc": "jupyter_ydoc",
    "jupyterlab-code-formatter": "jupyterlab_code_formatter",
    "jupyterlab-launcher": "jupyterlab_launcher",
    "jupyterlab-pygments": "jupyterlab_pygments",
    "jupyterlab-server": "jupyterlab_server",
    "jupyterlab-widgets": "jupyterlab_widgets",
    "keyrings-alt": "keyrings.alt",
    "korean-lunar-calendar": "korean_lunar_calendar",
    "lazy-loader": "lazy_loader",
    "line-profiler": "line_profiler",
    "medspacy-quickumls": "medspacy_quickumls",
    "memory-profiler": "memory_profiler",
    "ml-dtypes": "ml_dtypes",
    "multi-key-dict": "multi_key_dict",
    "mypy-extensions": "mypy_extensions",
    "opentracing-instrumentation": "opentracing_instrumentation",
    "opt-einsum": "opt_einsum",
    "parse-type": "parse_type",
    "path-py": "path.py",
    "plaster-pastedeploy": "plaster_pastedeploy",
    "posix-ipc": "posix_ipc",
    "prometheus-client": "prometheus_client",
    "prometheus-flask-exporter": "prometheus_flask_exporter",
    "prompt-toolkit": "prompt_toolkit",
    "pure-eval": "pure_eval",
    "lief": "py-lief",
    "mxnet": "py-mxnet",
    "xgboost": "py-xgboost",
    "xgboost-cpu": "py-xgboost-cpu",
    "pyproject-hooks": "pyproject_hooks",
    "pyramid-debugtoolbar": "pyramid_debugtoolbar",
    "pyramid-jinja2": "pyramid_jinja2",
    "pyramid-mako": "pyramid_mako",
    "pyramid-tm": "pyramid_tm",
    "blosc": "python-blosc",
    "chromedriver-binary": "python-chromedriver-binary",
    "dateutil": "python-dateutil",
    "fastjsonschema": "python-fastjsonschema",
    "flatbuffers": "python-flatbuffers",
    "gil": "python-gil",
    "graphviz": "python-graphviz",
    "kaleido": "python-kaleido",
    "leveldb": "python-leveldb",
    "libarchive-c": "python-libarchive-c",
    "tzdata": "python-tzdata",
    "xxhash": "python-xxhash",
    "zstd": "python-zstd",
    "python-app": "python.app",
    "python-http-client": "python_http_client",
    "pyviz-comms": "pyviz_comms",
    "querystring-parser": "querystring_parser",
    "readme-renderer": "readme_renderer",
    "repoze-lru": "repoze.lru",
    "requests-download": "requests_download",
    "requests-ntlm": "requests_ntlm",
    "ruamel-yaml": "ruamel_yaml",
    "ruamel-yaml-clib": "ruamel.yaml.clib",
    "ruamel-yaml-jinja2": "ruamel.yaml.jinja2",
    "sarif-om": "sarif_om",
    "semantic-version": "semantic_version",
    "service-identity": "service_identity",
    "setuptools-scm": "setuptools_scm",
    "setuptools-scm-git-archive": "setuptools_scm_git_archive",
    "slack-sdk": "slack_sdk",
    "smart-open": "smart_open",
    "sphinx-rtd-theme": "sphinx_rtd_theme",
    "stack-data": "stack_data",
    "thrift-sasl": "thrift_sasl",
    "typing-extensions": "typing_extensions",
    "typing-inspect": "typing_inspect",
    "vega-datasets": "vega_datasets",
    "win32-setctime": "win32_setctime",
    "zc-lockfile": "zc.lockfile",
    "zope-component": "zope.component",
    "zope-deprecation": "zope.deprecation",
    "zope-event": "zope.event",
    "zope-interface": "zope.interface",
    "zope-sqlalchemy": "zope.sqlalchemy",
}

MIRROR_OP = {
    ">": "<=",
    "<": ">=",
    ">=": "<",
    "<=": ">",
    "==": "!=",
    "!=": "==",
}


META_SHA_256: Optional[str] = None
META_SIZE: Optional[int] = None
META_PKGS = {}


class UnsupportedPythonInterpreter(Exception):
    pass


class ArchSpecificDependency(Exception):
    pass


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


PLATFORM_SPECIFIC_ENV = {
    "win-64": { "os_name": ("nt", ) },
    "win-32": { "os_name": ("nt", ) },
    "linux-64": { "os_name": ("posix", ) },
    "linux-aarch64": { "os_name": ("posix", ) },
    "osx-64": { "os_name": ("posix", ) },
    "osx-arm64": { "os_name": ("posix", ) },
}

def _eval_with_state(tree: markerpry.Node, env) -> tuple[Optional[bool], markerpry.Node]:
    result = tree.evaluate(env)
    if isinstance(result, markerpry.BooleanNode):
        return result.state, result
    return None, result


def make_metapkgs(conda_dep: str, marker: Marker, platform: str) -> list[str]:
    tree = markerpry.parse(str(marker))
    # TODO handle evaluation when possible (non-universal wheels)
    if tree.contains("os_name"):
        if platform == "noarch":
            raise ArchSpecificDependency(
                f"dependency {conda_dep} is arch-specific but platform is noarch"
            )
        env = PLATFORM_SPECIFIC_ENV.get(platform, {})
        state, tree = _eval_with_state(tree, env)
        assert state is not None
        if not state:
            logger.debug(
                f"Skipping dependency {conda_dep} because of os_name marker on platform {platform}"
            )
            return []
        else:
            # No metapackage is needed, we can just treat this as a normal dependency
            return [conda_dep]
    dep_hash = hashlib.sha1(conda_dep.encode()).hexdigest()[:HASH_LENGTH]
    marker_hash = hashlib.sha1(str(marker).encode()).hexdigest()[:HASH_LENGTH]
    meta_pkg_name = f"_c_{dep_hash}_{marker_hash}"
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
                    conda_dep,
                ]
                negative_deps = [f"python {nop1}{value1}", f"python {nop2}{value2}"]
                register_metapackages(meta_pkg_name, positive_deps, negative_deps)
                return [meta_pkg_name]
            elif tree.operator == "or":
                return make_metapkgs(
                    conda_dep,
                    Marker(str(left)),
                    platform,
                ) + make_metapkgs(
                    conda_dep,
                    Marker(str(right)),
                    platform,
                )
        else:
            # Use De Morgan's laws to express:
            # OR will produce two packages, one for each clause
            # AND will produce a single package that depends on both clauses
            raise NotImplementedError(f"complex marker: {tree}")

    markers = marker._markers
    if len(markers) != 1 or not isinstance(markers[0], tuple) or len(markers[0]) != 3:
        # TODO figure out why this still hits
        raise NotImplementedError(f"complex marker: {markers}")
    assert isinstance(tree, markerpry.ExpressionNode)
    variable, op, value = tree.lhs, tree.comparator, tree.rhs
    op = str(op)
    nop = MIRROR_OP[op]
    if str(variable) == "python_version" or str(variable) == "python_full_version":
        positive_dep = f"python {op}{value}"
        negative_dep = f"python {nop}{value}"
        positive_deps = [positive_dep, conda_dep]
        negative_deps = [negative_dep]
        register_metapackages(meta_pkg_name, positive_deps, negative_deps)
        return [meta_pkg_name]
    elif str(variable) == "platform_system":
        positive_dep = f"python {op}{value}"
        negative_dep = f"python {nop}{value}"
        positive_deps = [positive_dep, conda_dep]
        negative_deps = [negative_dep]
        register_metapackages(meta_pkg_name, positive_deps, negative_deps)
        return [meta_pkg_name]
    else:
        raise NotImplementedError(f"unsupported marker: {marker}")


# inline version of
# from conda.gateways.disk.create import write_as_json_to_file
def write_as_json_to_file(file_path: str, obj: Any):
    with codecs.open(file_path, mode="wb", encoding="utf-8") as fo:
        json_str = json.dumps(
            obj,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
        fo.write(json_str)


def py_to_conda_name(name: str) -> str:
    canonical_name = canonicalize_name(name)
    conda_name = PY_TO_CONDA_NAME.get(canonical_name, canonical_name)
    return conda_name


def py_to_conda_reqs(
    req: Requirement, seen_py_names: set[str], platform: str
) -> list[str]:
    conda_name = py_to_conda_name(req.name)
    conda_deps = [f"{conda_name} {req.specifier}".strip()]
    if req.marker:
        markers = req.marker._markers
        if any(isinstance(m, tuple) and str(m[0]) == "extra" for m in markers):
            logger.debug(f"skipping extra: {req}")
            return []
        conda_deps = make_metapkgs(conda_deps[0], req.marker, platform)
        # logger.warning(f"including req with marker: '{req}'")
    if req.extras:
        raise NotImplementedError(f"extras not supported: {req}")
    seen_py_names.add(canonicalize_name(req.name))
    return conda_deps


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

    def __init__(
        self,
        client: PyPISimple,
        project: str,
        platforms: set[str],
        spec: SpecifierSet | None,
    ):
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
            try:
                support_platforms, depends_from_filename = self.info_from_filename(
                    pkg.filename
                )
            except UnsupportedPythonInterpreter as e:
                logger.debug(f"unsupported python interpreter: {e}")
                continue
            if support_platforms.isdisjoint(self.platforms):
                logger.debug(f"Ignoring wheel for unsupported platform: {pkg.filename}")
                continue
            try:
                conda_pkg = self.repodata_for_pkg(
                    pkg, support_platforms, depends_from_filename
                )
                conda_pkgs.extend(conda_pkg)
            except ArchSpecificDependency as e:
                if len(support_platforms) == 1 and "noarch" in support_platforms:
                    # The wheel is marked as noarch, but it has a dependency that is arch specific
                    # Pretend like we have a wheel for each platform, and generate
                    # the corresponding conda packages
                    conda_pkg = self.repodata_for_pkg(
                        pkg,
                        {
                            "osx-64",
                            "osx-arm64",
                            "linux-64",
                            "linux-aarch64",
                            "win-64",
                            "win-32",
                        },
                        depends_from_filename,
                    )
                    conda_pkgs.extend(conda_pkg)
                else:
                    raise e

        return conda_pkgs

    def info_from_filename(self, filename: str) -> tuple[set[str], list[str]]:
        # https://peps.python.org/pep-0427/#file-name-convention
        _, python_tag, abi_tag, platform_tag = filename.removesuffix(".whl").rsplit(
            "-", maxsplit=3
        )
        if abi_tag == "none" and platform_tag == "any":
            supported_platforms = set[str](["noarch"])
            depends_from_filename: list[str] = []
            return supported_platforms, depends_from_filename

        supported_platforms = set[str]()
        if "macosx" in platform_tag:
            if "universal2" in platform_tag:
                # supported_platforms.add("osx-arm64")
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
        if abi_tag.lower() == "none":
            if python_tag == ("py3"):
                depends_from_filename = ["python >=3.0"]
                return supported_platforms, depends_from_filename
            elif python_tag.startswith("cp3"):
                minor = python_tag[3:]
                depends_from_filename = [f"python >=3.{minor}"]
                return supported_platforms, depends_from_filename
            else:
                raise NotImplementedError(
                    f"unsupported py_tag for none abi: {python_tag}"
                )

        if not abi_tag.startswith("cp3"):
            raise UnsupportedPythonInterpreter(f"unsupported abi: {abi_tag}")
        try:
            minor = abi_tag[3:]
            minor = "".join(c for c in minor if c.isdigit())
            minor = int(minor)
        except ValueError as e:
            print(f"could not parse {abi_tag} as int")
            raise e
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
        _, python_tag, abi_tag, platform_tag = pkg.filename.removesuffix(".whl").rsplit(
            "-", maxsplit=3
        )
        build = f"{python_tag}_{abi_tag}_{platform_tag}"
        pkgs = []

        for subdir in supported_platforms:
            # Process dependencies for this specific platform
            depends = list(depends_from_filename)
            if mdata.requires_python:
                py_dep = "python " + str(mdata.requires_python)
            else:
                py_dep = "python"
            depends.append(py_dep)

            if mdata.requires_dist:
                for py_req in mdata.requires_dist:
                    conda_deps = py_to_conda_reqs(py_req, self.seen_py_names, subdir)
                    depends.extend(conda_deps)

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
    specs: dict[str, list[SpecifierSet]],
    platforms: set[str],
    proc_dependencies: bool = False,
    session: Optional[Session] = None,
) -> Dict[str, Dict[Any, Any]]:
    client = PyPISimple(accept=ACCEPT_JSON_ONLY, session=session)
    conda_pkgs: list[CondaPackageMetadata] = []

    to_do_projects = set(specs.keys())
    done_projects: set[str] = set()

    while to_do_projects:
        project = to_do_projects.pop()
        done_projects.add(project)
        try:
            logger.info(f"creating repodata entries for project: {project}")
            # Process each specifier set for the project
            for spec in specs[project]:
                gen = ProjectGenerator(client, project, platforms, spec)
                conda_pkgs.extend(gen.conda_pkgs)

                logger.debug(f"projects visited: {gen.seen_py_names}")
                if proc_dependencies:
                    for to_add in gen.seen_py_names:
                        if to_add not in done_projects:
                            to_do_projects.add(to_add)
                logger.debug(f"projects to visit: {to_do_projects}")
        except Exception as e:
            logger.error(f"error processing project {project}: {e}")

    repodata = {}
    for platform in platforms:
        packages = {k: v for k, v in META_PKGS.items() if v["subdir"] == platform}
        packages.update(
            {
                pkg.filename: pkg.repodata_entry
                for pkg in conda_pkgs
                if pkg.subdir == platform
            }
        )
        subdir_data = empty_repodata(platform)
        subdir_data["packages"] = packages
        subdir_data["info"]["subdir"] = platform
        repodata[platform] = subdir_data
    return repodata


def write_repodata(repodata: Dict[str, Dict[Any, Any]], repo_base_path: str):
    for platform, subdir_data in repodata.items():
        os.makedirs(f"{repo_base_path}/{platform}", exist_ok=True)
        write_as_json_to_file(f"{repo_base_path}/{platform}/repodata.json", subdir_data)


def parse_requirements_file(file_path: Path) -> Dict[str, list[SpecifierSet]]:
    """Parse requirements file into dict of package name to list of SpecifierSets."""
    specs: Dict[str, list[SpecifierSet]] = {}
    with open(file_path, "r") as f:
        for line in f:
            req = Requirement(line.strip())
            if req.name not in specs:
                specs[req.name] = []
            specs[req.name].append(req.specifier)
    return specs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Generate repository data.")
    parser.add_argument(
        "-r",
        "--requirements",
        type=str,
        required=True,
        help="Requirements.txt file which contains the packages to be generated",
    )
    parser.add_argument(
        "--recurse",
        action="store_true",
        help="Recurse down through dependencies",
        default=False,
    )
    parser.add_argument(
        "-p",
        "--platform",
        type=str,
        nargs="+",
        default=["osx-arm64", "noarch"],
        help="List of platforms to generate repodata for",
    )
    parser.add_argument(
        "-c",
        "--cache",
        action="store_true",
        default=False,
        help="Cache (and use) PyPI responses",
    )

    args = parser.parse_args()

    file_path = Path(args.requirements)
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find {file_path}")
    specs = parse_requirements_file(file_path)
    if args.cache:
        session = requests_cache.CachedSession("pypi_cache")
    else:
        session = None
    repodata = create_repodata(
        specs,
        set(args.platform),
        proc_dependencies=args.recurse,
        session=session,
    )
    write_repodata(repodata, "./repo")

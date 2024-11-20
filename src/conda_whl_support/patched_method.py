import sys
from logging import getLogger
from json import JSONDecodeError
from os.path import join, basename, getsize

from conda.common.constants import TRACE
from conda.common.url import has_platform
from conda.base.context import context
from conda.gateways.disk.create import write_as_json_to_file, extract_tarball
from conda.gateways.disk.delete import rm_rf
from conda.gateways.disk.read import lexists, read_index_json, compute_sum
from conda.models.records import PackageRecord, PackageCacheRecord, Channel
from conda.models.records import MatchSpec

log = getLogger(__name__)


def execute(self, progress_update_callback=None):
    # I hate inline imports, but I guess it's ok since we're importing from the conda.core
    # The alternative is passing the the classes to ExtractPackageAction __init__
    from conda.core.package_cache_data import PackageCacheData

    log.log(
        TRACE, "extracting %s => %s", self.source_full_path, self.target_full_path
    )

    if lexists(self.target_full_path):
        rm_rf(self.target_full_path)

    # NEW
    if self.source_full_path.endswith(".whl"):
        from . import extract_whl
        extract_whl.extract_whl_as_conda_pkg(
            self.source_full_path,
            self.target_full_path,
        )
        prec: PackageRecord = self.record_or_spec
        index_json = {
            "arch": None,
            "build": prec.build,
            "build_number": prec.build_number,
            "depends": prec.depends,
            "license": prec.license,
            "name": prec.name,
            "noarch": "python",
            "platform": None,
            "subdir": "noarch",
            "timestamp": prec.timestamp,
            "version": prec.version,

        }
        index_json_path = join(self.target_full_path, "info", "index.json")
        write_as_json_to_file(index_json_path, index_json)
    else:
        extract_tarball(
            self.source_full_path,
            self.target_full_path,
            progress_update_callback=progress_update_callback,
        )
    # END NEW


    try:
        raw_index_json = read_index_json(self.target_full_path)
    except (OSError, JSONDecodeError, FileNotFoundError):
        # At this point, we can assume the package tarball is bad.
        # Remove everything and move on.
        print(
            f"ERROR: Encountered corrupt package tarball at {self.source_full_path}. Conda has "
            "left it in place. Please report this to the maintainers "
            "of the package."
        )
        sys.exit(1)

    if isinstance(self.record_or_spec, MatchSpec):
        url = self.record_or_spec.get_raw_value("url")
        assert url
        channel = (
            Channel(url)
            if has_platform(url, context.known_subdirs)
            else Channel(None)
        )
        fn = basename(url)
        sha256 = self.sha256 or compute_sum(self.source_full_path, "sha256")
        size = getsize(self.source_full_path)
        if self.size is not None:
            assert size == self.size, (size, self.size)
        md5 = self.md5 or compute_sum(self.source_full_path, "md5")
        repodata_record = PackageRecord.from_objects(
            raw_index_json,
            url=url,
            channel=channel,
            fn=fn,
            sha256=sha256,
            size=size,
            md5=md5,
        )
    else:
        repodata_record = PackageRecord.from_objects(
            self.record_or_spec, raw_index_json
        )

    repodata_record_path = join(
        self.target_full_path, "info", "repodata_record.json"
    )
    write_as_json_to_file(repodata_record_path, repodata_record)

    target_package_cache = PackageCacheData(self.target_pkgs_dir)
    package_cache_record = PackageCacheRecord.from_objects(
        repodata_record,
        package_tarball_full_path=self.source_full_path,
        extracted_package_dir=self.target_full_path,
    )
    target_package_cache.insert(package_cache_record)

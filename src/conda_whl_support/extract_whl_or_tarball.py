from os.path import join

from conda.models.records import PackageRecord
from conda.gateways.disk.create import write_as_json_to_file


def extract_tarball_or_whl(
        extract_tarball,
        source_full_path: str,
        target_full_path=None,
        progress_update_callback=None,
):
    if source_full_path.endswith(".whl"):
        import inspect
        prec: PackageRecord = inspect.currentframe().f_back.f_locals["self"].record_or_spec
        return extract_whl(
            source_full_path,
            prec,
            target_full_path=target_full_path,
            progress_update_callback=progress_update_callback
        )
    else:
        return extract_tarball(
            source_full_path,
            target_full_path=target_full_path,
            progress_update_callback=progress_update_callback
        )

def extract_whl(
        source_full_path: str,
        prec: PackageRecord,
        target_full_path=None,
        progress_update_callback=None,
) -> None:
    from . import extract_whl
    extract_whl.extract_whl_as_conda_pkg(
        source_full_path,
        target_full_path,
    )
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
    index_json_path = join(target_full_path, "info", "index.json")
    write_as_json_to_file(index_json_path, index_json)

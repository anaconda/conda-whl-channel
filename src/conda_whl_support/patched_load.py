import json
from logging import getLogger

from conda.common.serialize import json_load
from conda.models.records import PrefixRecord
from conda.exceptions import CorruptedEnvironmentError

log = getLogger(__name__)

def _load_single_record(self, prefix_record_json_path):
    log.debug("loading prefix record %s", prefix_record_json_path)
    with open(prefix_record_json_path) as fh:
        try:
            json_data = json_load(fh.read())
        except (UnicodeDecodeError, json.JSONDecodeError):
            # UnicodeDecodeError: catch horribly corrupt files
            # JSONDecodeError: catch bad json format files
            raise CorruptedEnvironmentError(
                self.prefix_path, prefix_record_json_path
            )
        # TODO: consider, at least in memory, storing prefix_record_json_path as part
        #       of PrefixRecord
        prefix_record = PrefixRecord(**json_data)
        self._PrefixData__prefix_records[prefix_record.name] = prefix_record

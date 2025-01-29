import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Dict, Optional, TypedDict, List


class WheelResult(TypedDict):
    """Result of analyzing a wheel."""
    package: str
    plain_dependencies: list[str]
    resolvable_dependencies: list[str]
    unresolvable_dependencies: list[str]
    error: Optional[str]



def read_results(output: Path) -> Dict[str, WheelResult]:
    """Load existing results from JSON file or return empty dict if file doesn't exist."""
    if not output.exists():
        return {}
        
    with open(output) as f:
        return json.load(f)


def write_results(results: Dict[str, WheelResult], output: Path) -> None:
    """Write results to JSON file."""
    with open(output, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.flush()


@dataclass
class WheelJob:
    """A job to analyze a marker."""
    package: str
    marker: str


@dataclass
class JobResult:
    """Result of analyzing a marker, including logs."""
    package: str
    marker: str
    result: WheelResult
    logs: List[str]


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Python package markers"
    )
    _ = parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="JSON file to store results"
    )
    args = parser.parse_args()

    # Load existing results
    results = read_results(args.output)

    # TODO: Create jobs from postgres data
    jobs: List[WheelJob] = []

    # TODO: Process jobs and collect results
    
    # Save final results
    write_results(results, args.output)


if __name__ == "__main__":
    main()

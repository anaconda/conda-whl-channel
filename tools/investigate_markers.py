import argparse
import json
from pathlib import Path
from typing import Dict, Optional, TypedDict
import psycopg2
from psycopg2.extras import DictCursor
import markerpry
import traceback
from packaging.requirements import Requirement, InvalidRequirement


class WheelResult(TypedDict):
    """Result of analyzing a wheel."""
    id: str
    package: str
    plain_dependencies: list[str]
    resolvable_dependencies: list[str]
    unneeded_dependencies: list[str]
    complex_dependencies: list[str]
    error: Optional[str]


def get_connection():
    """Get a connection to the postgres database."""
    return psycopg2.connect(
        dbname="pypi_scraper",
        user="postgres",
        password="postgres",
        host="localhost",
        port=5432
    )


def process_row(row: dict, logs: list[str]) -> WheelResult:
    """Process a single row from the database."""
    result = WheelResult(
        id=row['id'],
        package=row['package_name'],
        plain_dependencies=[],
        resolvable_dependencies=[],
        complex_dependencies=[],
        unneeded_dependencies=[],
        error=None
    )
    
    needs = row['dependencies']['needs']
    for need in needs:
        try:
            req = Requirement(need)
            if req.marker is None:
                result['plain_dependencies'].append(need)
            else:
                tree = markerpry.parse(str(req.marker))
                if isinstance(tree, markerpry.ExpressionNode):
                    result['plain_dependencies'].append(need)
                else:
                    environment: markerpry.Environment = {
                        'implementation_name': ['cpython'],
                        'implementation_version': ['3.10.13'],
                        'os_name': ['posix'],
                        'platform_machine': ['arm64'],
                        'platform_python_implementation': ['CPython'],
                        'platform_system': ['Darwin'],
                        'python_full_version': ['3.10.13'],
                        'python_version': ['3.10'],
                        'sys_platform': ['darwin'],
                    }
                    solved = tree.evaluate(environment)
                    if isinstance(solved, markerpry.BooleanNode):
                        if solved:
                            result['resolvable_dependencies'].append(need)
                        else:
                            result['unneeded_dependencies'].append(need)
                    elif isinstance(solved, markerpry.ExpressionNode):
                            result['resolvable_dependencies'].append(need)
                    else:
                        result['complex_dependencies'].append(need)
        except Exception as e:
            result['complex_dependencies'].append(need)
            if result['error'] is None:
                result['error'] = str(e)
            else:
                result['error'] += f"\n{str(e)}"
            logs.append(f"Error processing need {need}: {str(e)}")
            logs.append(traceback.format_exc())
    
    return result


def get_total_rows() -> int:
    """Get total number of rows in the FILE table that match our criteria."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) 
                FROM FILE JOIN PACKAGE ON FILE.package_id = PACKAGE.id
                WHERE dependencies IS NOT NULL 
                    AND dependencies != 'null'::jsonb 
                    AND data->>'filename' LIKE '%.whl'
                    AND dependencies->>'needs' IS NOT NULL
                    AND jsonb_array_length(dependencies->'needs') > 0
            """)
            return cur.fetchone()[0]


def process_batch(offset: int, limit: int) -> tuple[Dict[str, WheelResult], list[str]]:
    """Process a batch of rows from the FILE table."""
    logs = []
    results = {}
    
    logs.append(f"Processing rows {offset} to {offset + limit}")
    
    with get_connection() as conn:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(
                # Don't do string interpolation in prod
                # as you'll have a nice sql injection to deal with later
                f"""SELECT 
                    file.id,
                    file.dependencies,
                    package.name as package_name
                FROM FILE JOIN PACKAGE ON FILE.package_id = PACKAGE.id
                WHERE dependencies IS NOT NULL 
                    AND dependencies != 'null'::jsonb 
                    AND data->>'filename' LIKE '%.whl'
                    AND dependencies->>'needs' IS NOT NULL
                    AND jsonb_array_length(dependencies->'needs') > 0
                ORDER BY file.id
                LIMIT {limit} OFFSET {offset}""",
            )
            
            for row in cur:
                result = process_row(row, logs)
                results[result['id']] = result
                
    return results, logs


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
    _ = parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of rows to process in each batch"
    )
    args = parser.parse_args()

    # Load existing results
    results = {}
    if args.output.exists():
        with open(args.output) as f:
            results = json.load(f)

    # Get total number of rows
    total_rows = get_total_rows()
    print(f"Found {total_rows} total rows to process")

    # Process batches
    processed = 0
    for offset in range(0, total_rows, args.batch_size):
        batch_results, logs = process_batch(offset, args.batch_size)
        processed += len(batch_results)
        
        # Print logs
        print(f"\n[{processed}/{total_rows}] Processing batch:")
        for log in logs:
            print(f"  {log}")
        
        # Update results
        results.update(batch_results)
        
        # Save periodically
        if processed % 1000 == 0:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, sort_keys=True)
                f.flush()
                
        break
    
    # Final save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.flush()
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()

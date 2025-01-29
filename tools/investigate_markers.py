import argparse
import json
from pathlib import Path
from typing import Dict, Optional, TypedDict
import psycopg2
from psycopg2.extras import DictCursor
import markerpry
import traceback
from packaging.requirements import Requirement, InvalidRequirement
from multiprocessing import Pool
import time


class WheelResult(TypedDict):
    """Result of analyzing a wheel."""
    id: str
    filename: str
    package: str
    num_plain_dependencies: int
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
        filename=row['filename'],
        package=row['package_name'],
        num_plain_dependencies=0,
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
                result['num_plain_dependencies'] += 1
            else:
                tree = markerpry.parse(str(req.marker))
                if isinstance(tree, markerpry.ExpressionNode):
                    result['num_plain_dependencies'] += 1
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
                result['error'] = f"need: {need} - {str(e)}"
            else:
                result['error'] += f"\nneed: {need} - {str(e)}"
    
    return result


def setup_temp_table() -> int:
    """Create temporary table with filtered rows and return total count."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Drop temp table if it exists
            cur.execute("DROP TABLE IF EXISTS wheel_deps")
            
            # Create and populate permanent table
            cur.execute("""
                CREATE TABLE wheel_deps AS
                SELECT 
                    file.id,
                    file.dependencies,
                    package.name as package_name,
                    file.data->>'filename' as filename
                FROM FILE JOIN PACKAGE ON FILE.package_id = PACKAGE.id
                WHERE dependencies IS NOT NULL 
                    AND dependencies != 'null'::jsonb 
                    AND data->>'filename' LIKE '%.whl'
                    AND dependencies->>'needs' IS NOT NULL
                    AND jsonb_array_length(dependencies->'needs') > 0
                ORDER BY file.id
            """)
            
            # Add index for faster access
            cur.execute("CREATE INDEX ON wheel_deps (id)")
            
            # Get total count
            cur.execute("SELECT COUNT(*) FROM wheel_deps")
            return cur.fetchone()[0]


def process_batch(batch: tuple[int, int]) -> tuple[Dict[str, WheelResult], list[str]]:
    """Process a batch of rows using the temp table."""
    offset, limit = batch
    logs = []
    results = {}
    
    logs.append(f"Starting batch offset={offset} limit={limit} at {time.strftime('%H:%M:%S')}")
    
    with get_connection() as conn:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(
                "SELECT * FROM wheel_deps ORDER BY id LIMIT %s OFFSET %s",
                (limit, offset)
            )
            
            row_count = 0
            for row in cur:
                row_count += 1
                if row_count == 1:
                    logs.append(f"First row id: {row['id']}")
                result = process_row(row, logs)
                results[result['id']] = result
            
            logs.append(f"Completed {row_count} rows at {time.strftime('%H:%M:%S')}")
                
    return results, logs


def cleanup_temp_table():
    """Drop the temporary table."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS wheel_deps")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Python package markers"
    )
    _ = parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Text file to store results, one JSON object per line"
    )
    _ = parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of rows to process in each batch"
    )
    _ = parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=20,
        help="Number of parallel jobs (default: 20)",
    )
    args = parser.parse_args()

    try:
        # Truncate output file
        with open(args.output, "w") as f:
            f.truncate(0)

        # Setup temp table and get total rows
        print("Setting up temporary table...")
        total_rows = setup_temp_table()
        print(f"Found {total_rows} total rows to process")
        print(f"Using {args.jobs} workers with batch size {args.batch_size}")

        # Create list of batches to process
        batches = [
            (offset, args.batch_size)
            for offset in range(0, total_rows, args.batch_size)
        ]
        total_batches = len(batches)
        
        # Process batches in parallel but write in order
        processed = 0
        with Pool(args.jobs) as pool:
            # Start all jobs
            pending = {
                i: pool.apply_async(process_batch, (batch,))
                for i, batch in enumerate(batches)
            }
            
            # Process results in order
            for i in range(len(batches)):
                batch_results, logs = pending[i].get()
                processed += 1
                
                # Print batch logs
                print(f"\n[{processed}/{total_batches}] Processing batch:")
                for log in logs:
                    print(f"  {log}")
                
                # Append results to file
                with open(args.output, "a") as f:
                    for result in batch_results.values():
                        # Double-encode error messages to preserve \n
                        if result['error']:
                            result['error'] = json.dumps(result['error'])[1:-1]
                        json_line = json.dumps(result)
                        f.write(json_line + "\n")
                        f.flush()
        
        print("\nProcessing complete!")

    finally:
        # Clean up temp table
        cleanup_temp_table()


if __name__ == "__main__":
    main()

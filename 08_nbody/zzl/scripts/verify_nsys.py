"""Require actual CUDA kernel and runtime rows in a Nsight Systems SQLite export."""
import argparse
import json
import re
from pathlib import Path
import sqlite3
from statistics import median


def verify_report(path: Path) -> dict[str, int]:
    """Read without modifying the export; reject empty or incomplete captures."""
    counts = {}
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_RUNTIME"):
            if table not in tables:
                raise ValueError(f"Missing CUDA data table: {table}")
            # Table names are constants, never derived from the input database.
            count = connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            if count == 0:
                raise ValueError(f"Empty CUDA data table: {table}")
            counts[table] = count
    finally:
        connection.close()
    return counts


def acceleration_summary(path: Path, expected_kernel: str | None = None) -> dict[str, int | float]:
    """Summarize the recorded acceleration kernel without changing the export."""
    if expected_kernel not in (None, "tiled", "naive"):
        raise ValueError("Expected kernel must be tiled or naive")
    verify_report(path)
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT k.start, k.end, names.value "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL AS k "
            "JOIN StringIds AS names ON k.demangledName = names.id "
            "WHERE names.value LIKE '%acceleration_%'"
        ).fetchall()
    except sqlite3.Error as error:
        raise ValueError(f"Cannot read kernel names: {error}") from error
    finally:
        connection.close()
    pattern = re.compile(r"(?<![\w])acceleration_(tiled|naive)(?=\s*(?:<|\(|$))")
    matched = [(start, end, match.group(1)) for start, end, name in rows
               if (match := pattern.search(name))]
    variants = {variant for _, _, variant in matched}
    if len(variants) > 1 or (variants and expected_kernel and variants != {expected_kernel}):
        raise ValueError("Captured acceleration kernels do not match the requested kernel")
    rows = [(start, end) for start, end, _ in matched]
    if any(type(start) is not int or type(end) is not int or end <= start
           for start, end in rows):
        raise ValueError("Invalid acceleration kernel timestamps: expected end > start")
    durations = [end - start for start, end in rows]
    if not durations:
        raise ValueError("No acceleration CUDA kernels were captured")
    durations.sort()
    return {
        "instances": len(durations),
        "total_ns": sum(durations),
        "average_ns": sum(durations) / len(durations),
        "median_ns": median(durations),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    args = parser.parse_args()
    try:
        counts = verify_report(args.sqlite)
    except (OSError, sqlite3.Error, ValueError) as error:
        parser.exit(1, f"Profile verification failed: {error}\n")
    print(json.dumps({"cuda_data_present": True, "rows": counts}, indent=2))


if __name__ == "__main__":
    main()

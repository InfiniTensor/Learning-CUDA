#!/usr/bin/env python3
"""Merge quant_dequant JSON records into one analysis-friendly CSV file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for path in args.inputs:
        record = json.loads(path.read_text(encoding="utf-8"))
        record["source_file"] = path.name
        rows.append(record)
    if not rows:
        raise ValueError("at least one JSON input is required")
    fieldnames = ["source_file"] + [key for key in rows[0] if key != "source_file"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

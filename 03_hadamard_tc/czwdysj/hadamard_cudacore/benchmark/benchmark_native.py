#!/usr/bin/env python3
"""Repeat the native C++ CUDA Event benchmark and retain each independent trial."""
from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path

from benchmark_fht import DEFAULT_BATCHES, DEFAULT_DIMS, FIELDS, REPRESENTATIVE_SHAPES, summarize, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", choices=("representative", "full"), default="representative")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    return parser.parse_args()


def shape_grid(matrix: str) -> tuple[tuple[int, int], ...]:
    if matrix == "representative": return REPRESENTATIVE_SHAPES
    return tuple((batch, dim) for batch in DEFAULT_BATCHES for dim in DEFAULT_DIMS)


def main() -> int:
    args = parse_args()
    if not args.binary.is_file(): raise SystemExit(f"missing native benchmark: {args.binary}")
    raw: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="hadamard_native_") as temp_dir:
        temp_csv = Path(temp_dir) / "one_case.csv"
        for dtype in ("fp32", "fp16", "bf16"):
            for batch, dim in shape_grid(args.matrix):
                print(f"native Shape [{batch}, {dim}] {dtype.upper()}")
                for trial in range(1, args.trials + 1):
                    command = [str(args.binary), "--kernel", "all", "--batch", str(batch), "--dim", str(dim), "--dtype", dtype, "--warmup", str(args.warmup), "--repeat", str(args.repeat), "--csv", str(temp_csv)]
                    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
                    with temp_csv.open(newline="", encoding="utf-8") as stream:
                        for source in csv.DictReader(stream):
                            version = source["implementation"].removeprefix("hadamard_")
                            row = {field: "" for field in FIELDS}
                            row.update({"measurement_context": "native_cpp", "trial": trial, "order_index": "", "implementation": version, "kernel_version": version, "batch": batch, "head_dim": dim, "dtype": dtype.upper(), "latency_us": source["latency_us"], "correct": source["correct"], "max_abs_error": source["max_abs_error"], "max_relative_error": source["max_relative_error"]})
                            raw.append(row)
    write_csv(args.raw_csv, raw)
    summary = summarize(raw)
    for row in summary: row["implementation"] = f"hadamard_{row['implementation']}"
    write_csv(args.summary_csv, summary)
    print(f"native raw: {args.raw_csv}\nnative summary: {args.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

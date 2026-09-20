#!/usr/bin/env python3
"""Merge benchmark summaries and produce a concise human-readable report."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from benchmark_fht import FIELDS, write_csv


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream: return list(csv.DictReader(stream))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--same-summary", type=Path, required=True)
    parser.add_argument("--native-summary", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def cell(row: dict[str, str] | None, dao: float | None = None) -> str:
    if row is None: return "n/a"
    latency = float(row["latency_us"])
    if dao is None: return f"{latency:.3f}"
    ratio = row.get("speed_vs_dao", "")
    return f"{latency:.3f} ({float(ratio):.3f}x)" if ratio else f"{latency:.3f} (invalid)"


def main() -> int:
    args = parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    same, native = read_csv(args.same_summary), read_csv(args.native_summary)
    write_csv(args.summary, [{field: row.get(field, "") for field in FIELDS} for row in native + same])
    by_case: dict[tuple[str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in same:
        by_case[(row["dtype"], row["batch"], row["head_dim"])][row["implementation"]] = row
    lines = ["# Hadamard version benchmark report", "", "## Environment", "", f"- GPU: {metadata['gpu']} (SM {'.'.join(map(str, metadata['compute_capability']))})", f"- PyTorch: {metadata['torch']} / CUDA {metadata['torch_cuda']}", f"- Dao FHT commit: {metadata.get('dao_commit', 'unknown')}", f"- Matrix: {metadata['matrix']}; warmup={metadata['warmup']}, repeat={metadata['repeat']}, trials={metadata['trials']}", "- Contract: `y = x @ H(D) / sqrt(D)`; table ratios are `Dao latency / current latency`.", "- `UNSTABLE` means `(max-min)/median > 5%`; raw values remain available.", "", "## Aggregate versus Dao", "", "| dtype | version | geometric mean speed_vs_dao | faster cases | valid cases |", "|---|---:|---:|---:|---:|"]
    aggregate: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (dtype, _, _), case in by_case.items():
        for version in ("hadamard_v0", "hadamard_v1", "hadamard_v2"):
            row = case.get(version)
            if row and row.get("comparison_valid") == "true" and row.get("speed_vs_dao"):
                aggregate[(dtype, version)].append(float(row["speed_vs_dao"]))
    for dtype in ("FP16", "BF16", "FP32"):
        dtype_case_count = sum(1 for key in by_case if key[0] == dtype)
        for version in ("hadamard_v0", "hadamard_v1", "hadamard_v2"):
            values = aggregate[(dtype, version)]
            if values:
                geo = math.exp(sum(math.log(value) for value in values) / len(values))
                lines.append(f"| {dtype} | {version} | {geo:.3f}x | {sum(value > 1 for value in values)}/{len(values)} | {len(values)}/{dtype_case_count} |")
    for dtype in ("FP16", "BF16", "FP32"):
        lines += ["", f"## {dtype} representative cases", "", "| batch | D | v0 us (vs Dao) | v1 us (vs Dao) | v2 us (vs Dao) | Dao us | clone us | status |", "|---:|---:|---:|---:|---:|---:|---:|---|"]
        cases = sorted((key for key in by_case if key[0] == dtype), key=lambda key: (int(key[1]), int(key[2])))
        for _, batch, dim in cases:
            case, dao = by_case[(dtype, batch, dim)], by_case[(dtype, batch, dim)].get("dao_fht")
            valid = all(case.get(version, {}).get("comparison_valid") == "true" for version in ("hadamard_v0", "hadamard_v1", "hadamard_v2"))
            unstable = any(case.get(version, {}).get("stability") == "UNSTABLE" for version in ("hadamard_v0", "hadamard_v1", "hadamard_v2", "dao_fht", "torch_clone"))
            status = "valid" if valid else "COMPARISON INVALID"
            if unstable: status += "; UNSTABLE"
            lines.append(f"| {batch} | {dim} | {cell(case.get('hadamard_v0'), float(dao['latency_us']) if dao else None)} | {cell(case.get('hadamard_v1'), float(dao['latency_us']) if dao else None)} | {cell(case.get('hadamard_v2'), float(dao['latency_us']) if dao else None)} | {cell(dao)} | {cell(case.get('torch_clone'))} | {status} |")
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"combined summary: {args.summary}\nreport: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

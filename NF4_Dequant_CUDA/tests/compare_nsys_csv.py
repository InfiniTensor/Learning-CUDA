#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(s: str) -> float:
    try:
        return float(s)
    except Exception:
        return 0.0


def _collect(stem: Path) -> dict[str, float]:
    api = _read_csv(stem.parent / f"{stem.name}_cuda_api_sum.csv")
    kern = _read_csv(stem.parent / f"{stem.name}_cuda_gpu_kern_sum.csv")
    mem_t = _read_csv(stem.parent / f"{stem.name}_cuda_gpu_mem_time_sum.csv")

    out = {
        "cudaMemcpy_api_total_ns": 0.0,
        "cudaMemcpyAsync_api_total_ns": 0.0,
        "cudaMalloc_api_total_ns": 0.0,
        "cudaMallocHost_api_total_ns": 0.0,
        "cudaFreeHost_api_total_ns": 0.0,
        "kernel_avg_ns": 0.0,
        "kernel_total_ns": 0.0,
        "d2h_total_ns": 0.0,
        "h2d_total_ns": 0.0,
    }

    for row in api:
        name = row.get("Name", "")
        total = _to_float(row.get("Total Time (ns)", "0"))
        if name == "cudaMemcpy":
            out["cudaMemcpy_api_total_ns"] += total
        elif name == "cudaMemcpyAsync":
            out["cudaMemcpyAsync_api_total_ns"] += total
        elif name == "cudaMalloc":
            out["cudaMalloc_api_total_ns"] = total
        elif name == "cudaMallocHost":
            out["cudaMallocHost_api_total_ns"] = total
        elif name == "cudaFreeHost":
            out["cudaFreeHost_api_total_ns"] = total

    # Report memcpy API as sync+async total to keep comparisons fair.
    out["cudaMemcpy_api_total_ns"] += out["cudaMemcpyAsync_api_total_ns"]

    if kern:
        out["kernel_avg_ns"] = _to_float(kern[0].get("Avg (ns)", "0"))
        out["kernel_total_ns"] = _to_float(kern[0].get("Total Time (ns)", "0"))

    for row in mem_t:
        op = row.get("Operation", "")
        total = _to_float(row.get("Total Time (ns)", "0"))
        if "Device-to-Host" in op:
            out["d2h_total_ns"] = total
        elif "Host-to-Device" in op:
            out["h2d_total_ns"] = total

    return out


def _pct(new: float, base: float) -> float:
    if base == 0.0:
        return 0.0
    return (new - base) / base * 100.0


def _parse_stem_list(raw: str) -> list[Path]:
    stems: list[Path] = []
    for part in raw.split(","):
        s = part.strip()
        if s:
            stems.append(Path(s))
    return stems


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare two Nsight Systems CSV report stems.")
    ap.add_argument("--base-stem", type=Path, help="e.g. tests/data/nsys_run_pageable")
    ap.add_argument("--new-stem", type=Path, help="e.g. tests/data/nsys_run_pinned")
    ap.add_argument(
        "--base-stems",
        type=str,
        help="Comma-separated stems for multi-round median compare.",
    )
    ap.add_argument(
        "--new-stems",
        type=str,
        help="Comma-separated stems for multi-round median compare.",
    )
    args = ap.parse_args()

    keys = [
        "kernel_avg_ns",
        "d2h_total_ns",
        "h2d_total_ns",
        "cudaMemcpy_api_total_ns",
        "cudaMemcpyAsync_api_total_ns",
        "cudaMalloc_api_total_ns",
        "cudaMallocHost_api_total_ns",
        "cudaFreeHost_api_total_ns",
    ]

    if args.base_stems is not None or args.new_stems is not None:
        if args.base_stems is None or args.new_stems is None:
            raise ValueError("Both --base-stems and --new-stems must be provided together.")
        base_stems = _parse_stem_list(args.base_stems)
        new_stems = _parse_stem_list(args.new_stems)
        if len(base_stems) == 0 or len(new_stems) == 0:
            raise ValueError("Empty stem list provided.")
        if len(base_stems) != len(new_stems):
            raise ValueError("base/new stem list lengths do not match.")

        base_runs = [_collect(s) for s in base_stems]
        new_runs = [_collect(s) for s in new_stems]

        print(f"rounds,{len(base_runs)}")
        print("metric,base_median,new_median,delta_pct")
        for k in keys:
            b_med = statistics.median(r[k] for r in base_runs)
            n_med = statistics.median(r[k] for r in new_runs)
            print(f"{k},{b_med:.3f},{n_med:.3f},{_pct(n_med, b_med):.2f}")
        return 0

    if args.base_stem is None or args.new_stem is None:
        raise ValueError("Provide either --base-stem/--new-stem or --base-stems/--new-stems.")

    base = _collect(args.base_stem)
    new = _collect(args.new_stem)
    print("metric,base,new,delta_pct")
    for k in keys:
        b = base[k]
        n = new[k]
        print(f"{k},{b:.3f},{n:.3f},{_pct(n, b):.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

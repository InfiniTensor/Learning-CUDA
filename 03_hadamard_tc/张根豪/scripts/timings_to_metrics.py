#!/usr/bin/env python3
"""Build Roofline inputs from CUDA-event timings when HW counters are busy."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("parse_ncu", ROOT / "scripts" / "parse_ncu.py")
PARSE_NCU = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PARSE_NCU)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timings", type=Path, required=True)
    parser.add_argument("--device-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = json.loads(args.device_json.read_text())
    fp32_peak, memory_peak = PARSE_NCU.theoretical_peaks(device)
    with args.timings.open(newline="") as source_file:
        source_rows = list(csv.DictReader(source_file))
    if not source_rows:
        raise SystemExit("timings CSV is empty")
    rows = []
    for source in source_rows:
        tokens = int(source["total_tokens"])
        head_dim = int(source["head_dim"])
        duration_s = float(source["avg_kernel_ms"]) * 1e-3
        work = float(tokens * head_dim * int(math.log2(head_dim)))
        minimum_io = float(tokens * head_dim * 4)
        rows.append({
            "gpu_name": device["name"],
            "compute_capability": f'{device["compute_capability_major"]}.{device["compute_capability_minor"]}',
            "dtype": source["dtype"],
            "head_dim": head_dim,
            "total_tokens": tokens,
            "kernel_duration_us": f"{duration_s * 1e6:.9g}",
            "algorithmic_gflops": f"{work / duration_s / 1e9:.9g}",
            "arithmetic_intensity_flop_per_byte": f"{work / minimum_io:.9g}",
            "dram_bytes": "",
            "minimum_io_bytes": f"{minimum_io:.9g}",
            "intensity_byte_source": "minimum_io_fallback",
            "effective_io_bandwidth_gbs": f"{minimum_io / duration_s / 1e9:.9g}",
            "measured_dram_bandwidth_gbs": "",
            "achieved_occupancy_pct": "",
            "sm_throughput_pct": "",
            "dram_throughput_pct": "",
            "registers_per_thread": "",
            "shared_bank_conflicts": "",
            "barrier_stall_pct": "",
            "long_scoreboard_stall_pct": "",
            "theoretical_fp32_peak_tflops": f"{fp32_peak:.9g}",
            "theoretical_memory_peak_gbs": f"{memory_peak:.9g}",
            "source_file": args.timings.name,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} timing-derived rows to {args.output}")


if __name__ == "__main__":
    main()

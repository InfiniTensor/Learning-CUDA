#!/usr/bin/env python3
"""Extract a compact table from Nsight Compute's wide raw CSV export."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRICS = {
    "duration_us": "gpu__time_duration.sum",
    "sm_throughput_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_throughput_pct": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_bandwidth_gbs": "dram__bytes.sum.per_second",
    "achieved_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "l2_hit_rate_pct": "lts__t_sector_hit_rate.pct",
    "registers_per_thread": "launch__registers_per_thread",
    "static_shared": "launch__shared_mem_per_block_static",
    "waves_per_sm": "launch__waves_per_multiprocessor",
    "stall_barrier": "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "stall_long_scoreboard": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "stall_short_scoreboard": "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "stall_wait": "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
}


def read_wide(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    rows = list(csv.reader(path.open(newline="")))
    if len(rows) != 3:
        raise SystemExit(f"expected header/unit/value rows in {path}, got {len(rows)}")
    return dict(zip(rows[0], rows[1])), dict(zip(rows[0], rows[2]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = []
    for implementation in ("optimized", "fused_int4"):
        path = args.input_dir / f"{implementation}.csv"
        units, values = read_wide(path)
        row: dict[str, str | float] = {
            "implementation": implementation,
            "kernel_name": values["Kernel Name"],
        }
        for field, metric in METRICS.items():
            row[field] = float(values[metric])
        shared_unit = units[METRICS["static_shared"]]
        if shared_unit == "Kbyte/block":
            row["static_shared"] = float(row["static_shared"]) * 1000.0
        duration_seconds = float(row["duration_us"]) * 1e-6
        row["derived_dram_bytes"] = (
            float(row["dram_bandwidth_gbs"]) * 1e9 * duration_seconds
        )
        output.append(row)

    fields = list(output[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields,
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

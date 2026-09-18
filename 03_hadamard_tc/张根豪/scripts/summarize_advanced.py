#!/usr/bin/env python3
"""Turn the optimization benchmark CSV into a compact reproducibility note."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def yes(value: str) -> str:
    return "PASS" if value.lower() == "true" else "FAIL"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment", type=Path)
    args = parser.parse_args()

    with args.input.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise SystemExit("advanced result CSV is empty")

    indexed = {
        (row["dtype"], int(row["head_dim"]), row["implementation"]): row
        for row in rows
    }
    keys = sorted({(row["dtype"], int(row["head_dim"])) for row in rows})
    required = {"baseline", "optimized", "unfused_int4", "fused_int4"}
    for dtype, dim in keys:
        found = {impl for dt, hd, impl in indexed if dt == dtype and hd == dim}
        missing = required - found
        if missing:
            raise SystemExit(f"{dtype}/d={dim} missing rows: {sorted(missing)}")

    first = rows[0]
    lines = [
        "# Optimized FHT and fused INT4 results",
        "",
        f"Shape: `{first['batch_size']} x {first['seq_len']} x "
        f"{first['num_heads']}` = `{first['total_tokens']}` tokens; "
        f"warmup `{first['warmup_runs']}`, measured `{first['bench_runs']}` runs.",
        "",
        "## 9.1 optimized Hadamard",
        "",
        "| dtype | head_dim | baseline us | optimized us | speedup | bit-exact |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dtype, dim in keys:
        baseline = indexed[(dtype, dim, "baseline")]
        optimized = indexed[(dtype, dim, "optimized")]
        lines.append(
            f"| {dtype.upper()} | {dim} | {float(baseline['avg_ms']) * 1000:.2f} | "
            f"{float(optimized['avg_ms']) * 1000:.2f} | "
            f"{float(optimized['speedup_vs_baseline']):.2f}x | "
            f"{yes(optimized['optimized_vs_baseline_bit_exact'])} |"
        )

    lines += [
        "",
        "## 9.2 fused per-token symmetric INT4",
        "",
        "| dtype | head_dim | unfused us | fused us | fusion speedup | compression | MAE | exact checks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dtype, dim in keys:
        unfused = indexed[(dtype, dim, "unfused_int4")]
        fused = indexed[(dtype, dim, "fused_int4")]
        exact = (
            yes(fused["fused_vs_unfused_bit_exact"]) == "PASS"
            and yes(fused["gpu_vs_cpu_quant_bit_exact"]) == "PASS"
        )
        lines.append(
            f"| {dtype.upper()} | {dim} | {float(unfused['avg_ms']) * 1000:.2f} | "
            f"{float(fused['avg_ms']) * 1000:.2f} | "
            f"{float(fused['fused_speedup_vs_unfused']):.2f}x | "
            f"{float(fused['compression_ratio']):.2f}x | "
            f"{float(fused['quant_mae']):.4g} | {'PASS' if exact else 'FAIL'} |"
        )

    lines += [
        "",
        "`exact checks` requires both fused-vs-unfused and GPU-vs-CPU INT4 "
        "packed bytes/scales to be bit-exact.",
    ]
    full_checks = args.output.parent / "full_shape_checks.log"
    if full_checks.exists():
        check_count = sum(
            "optimized_vs_baseline=BIT-EXACT" in line
            and "fused_vs_unfused=BIT-EXACT" in line
            and "gpu_vs_cpu_quant=BIT-EXACT" in line
            for line in full_checks.read_text().splitlines()
        )
        lines += [
            "",
            f"Extended smoke matrix: **{check_count}/{check_count} PASS** across "
            "FP16/BF16, normalized/non-normalized, and d=32/64/128/256/512/1024. "
            "See `full_shape_checks.log`.",
        ]

    nsys_csv = args.output.parent / "nsys_summary" / "nsys_kernel_summary.csv"
    if nsys_csv.exists():
        with nsys_csv.open(newline="") as source:
            nsys_rows = list(csv.DictReader(source))
        lines += [
            "",
            "## Nsight Systems representative trace",
            "",
            "FP16, d=128, 16384 tokens (5 warmup + 20 measured runs):",
            "",
            "| kernel | calls | avg us | grid | block | registers/thread | static shared B | resource-limit occupancy |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in nsys_rows:
            lines.append(
                f"| {row['kernel_name']} | {row['calls']} | "
                f"{float(row['avg_us']):.2f} | {row['grid']} | {row['block']} | "
                f"{row['registers_per_thread']} | {row['static_shared_bytes']} | "
                f"{float(row['resource_limit_occupancy_pct']):.0f}% |"
            )
        lines += [
            "",
            "Occupancy is a launch-resource upper bound derived from Systems metadata, "
            "not counter-measured achieved occupancy.",
        ]

    ncu_logs = sorted((args.output.parent / "ncu").glob("*.log"))
    if ncu_logs:
        unavailable = sum("driver resource was unavailable" in path.read_text()
                          for path in ncu_logs)
        lines += [
            "",
            f"Nsight Compute: {unavailable}/{len(ncu_logs)} attempts could not acquire "
            "performance counters because a driver resource (typically DCGM) was busy; "
            "the raw logs are retained under `ncu/`.",
        ]

    figure = args.output.parent / "figures" / "advanced_performance.png"
    if figure.exists():
        lines += ["", "![Optimization and fused INT4 plots](figures/advanced_performance.png)"]
    if args.environment and args.environment.exists():
        environment = args.environment.read_text().strip()
        lines += ["", "## Environment", "", "```text", environment, "```"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

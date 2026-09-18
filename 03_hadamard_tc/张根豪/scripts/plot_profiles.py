#!/usr/bin/env python3
"""Create Roofline and profiler-summary figures from ncu_metrics.csv."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def value(row: dict[str, str], key: str) -> float:
    text = row.get(key, "")
    return float(text) if text else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.metrics.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise SystemExit("metrics CSV is empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gpu = rows[0]["gpu_name"]
    fallback = all(row.get("intensity_byte_source") == "minimum_io_fallback" for row in rows)
    compute_peak = value(rows[0], "theoretical_fp32_peak_tflops") * 1000
    memory_peak = value(rows[0], "theoretical_memory_peak_gbs")
    intensities = [value(row, "arithmetic_intensity_flop_per_byte") for row in rows]
    xmin = max(0.05, min(intensities) / 3)
    xmax = max(32.0, max(intensities) * 8, compute_peak / memory_peak * 4)
    xs = np.logspace(math.log10(xmin), math.log10(xmax), 500)
    memory_roof = memory_peak * xs
    roof = np.minimum(memory_roof, compute_peak)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.loglog(xs, roof, color="#222222", linewidth=2.2,
              label=f"Roof: min({memory_peak:.0f} GB/s × AI, {compute_peak/1000:.1f} TFLOP/s)")
    ax.loglog(xs[memory_roof < compute_peak], memory_roof[memory_roof < compute_peak],
              color="#5c5c5c", linestyle="--", linewidth=1)
    colors = {"fp16": "#1976d2", "bf16": "#d84315"}
    markers = {64: "o", 128: "s", 256: "^"}
    for row in rows:
        dtype = row["dtype"]
        hd = int(row["head_dim"])
        ai = value(row, "arithmetic_intensity_flop_per_byte")
        perf = value(row, "algorithmic_gflops")
        face = colors.get(dtype) if dtype == "fp16" else "none"
        ax.scatter(ai, perf, s=82, marker=markers.get(hd, "o"), facecolor=face,
                   edgecolor=colors.get(dtype), linewidth=1.6, zorder=3)
    for hd in sorted({int(row["head_dim"]) for row in rows}):
        subset = [row for row in rows if int(row["head_dim"]) == hd]
        ai = sum(value(row, "arithmetic_intensity_flop_per_byte") for row in subset) / len(subset)
        perf = max(value(row, "algorithmic_gflops") for row in subset)
        offsets = {64: (-14, 11), 128: (-25, 13), 256: (25, 13)}
        ax.annotate(f"d={hd}", (ai, perf), xytext=offsets.get(hd, (4, 8)),
                    ha="center", textcoords="offset points", fontsize=8)
    byte_label = "minimum global I/O byte" if fallback else "measured DRAM byte"
    ax.set_xlabel(f"Arithmetic intensity (algorithmic FLOP / {byte_label})")
    ax.set_ylabel("Algorithmic throughput (GFLOP/s)")
    ax.set_title(f"Fast Hadamard CUDA Roofline — {gpu}")
    ax.grid(True, which="both", alpha=0.22)
    legend_items = [
        Line2D([0], [0], color="#222222", linewidth=2.2,
               label=f"Roof: min({memory_peak:.0f} GB/s × AI, {compute_peak/1000:.1f} TFLOP/s)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["fp16"],
               markeredgecolor=colors["fp16"], label="FP16"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor=colors["bf16"], markeredgewidth=1.6, label="BF16"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "roofline.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    if fallback:
        panels = [
            ("kernel_duration_us", "Kernel duration (μs)"),
            ("effective_io_bandwidth_gbs", "Effective minimum-I/O bandwidth (GB/s)"),
            ("algorithmic_gflops", "Algorithmic throughput (GFLOP/s)"),
            ("roofline_efficiency_pct", "Fraction of Roofline bound (%)"),
        ]
    else:
        panels = [
            ("kernel_duration_us", "Kernel duration (μs)"),
            ("achieved_occupancy_pct", "Achieved occupancy (%)"),
            ("dram_throughput_pct", "DRAM throughput (% peak)"),
            ("barrier_stall_pct", "Barrier stall (% active warp)"),
        ]
    for ax, (key, title) in zip(axes.flat, panels):
        plotted = False
        for dtype in ("fp16", "bf16"):
            subset = sorted((row for row in rows if row["dtype"] == dtype),
                            key=lambda row: int(row["head_dim"]))
            x = [int(row["head_dim"]) for row in subset]
            if key == "roofline_efficiency_pct":
                y = [100 * value(row, "algorithmic_gflops") /
                     min(compute_peak, memory_peak * value(row, "arithmetic_intensity_flop_per_byte"))
                     for row in subset]
            else:
                y = [value(row, key) for row in subset]
            if any(not math.isnan(v) for v in y):
                ax.plot(x, y, marker="o", linewidth=2, label=dtype.upper(),
                        color=colors[dtype])
                plotted = True
        ax.set_title(title)
        ax.set_xticks(sorted({int(row["head_dim"]) for row in rows}))
        ax.grid(True, alpha=0.25)
        if not plotted:
            ax.text(0.5, 0.5, "metric unavailable", ha="center", va="center",
                    transform=ax.transAxes, color="#777777")
    axes[1, 0].set_xlabel("head_dim")
    axes[1, 1].set_xlabel("head_dim")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, loc="upper left")
    summary_source = "CUDA Event (NCU counters unavailable)" if fallback else "Nsight Compute"
    fig.suptitle(f"{summary_source} summary — {gpu}", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.output_dir / "profiler_metrics.png", dpi=180)
    plt.close(fig)
    print(f"wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()

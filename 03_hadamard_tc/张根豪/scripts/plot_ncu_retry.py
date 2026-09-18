#!/usr/bin/env python3
"""Visualize the compact optimized/fused NCU hardware-counter summary."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.input.open(newline="")))
    labels = ["optimized FHT", "fused FHT+INT4"]
    colors = ["#1976d2", "#00897b"]
    x = np.arange(2)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))

    axes[0, 0].bar(labels, [float(r["duration_us"]) for r in rows], color=colors)
    axes[0, 0].set_ylabel("Kernel duration (μs)")
    axes[0, 0].set_title("NCU replay kernel duration")

    width = 0.34
    axes[0, 1].bar(x - width / 2, [float(r["sm_throughput_pct"]) for r in rows],
                   width, label="SM", color="#7b1fa2")
    axes[0, 1].bar(x + width / 2, [float(r["dram_throughput_pct"]) for r in rows],
                   width, label="DRAM", color="#f9a825")
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_ylabel("Peak sustained (%)")
    axes[0, 1].set_title("Compute and DRAM throughput")
    axes[0, 1].legend()

    axes[1, 0].bar(x - width / 2,
                   [float(r["achieved_occupancy_pct"]) for r in rows], width,
                   label="achieved occupancy", color="#3949ab")
    axes[1, 0].bar(x + width / 2, [float(r["l2_hit_rate_pct"]) for r in rows],
                   width, label="L2 sector hit rate", color="#43a047")
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set_ylabel("Percent")
    axes[1, 0].set_title("Occupancy and L2 hit rate")
    axes[1, 0].legend(fontsize=8)

    stall_fields = ["stall_barrier", "stall_long_scoreboard",
                    "stall_short_scoreboard", "stall_wait"]
    stall_labels = ["barrier", "long scoreboard", "short scoreboard", "wait"]
    for index, (field, label) in enumerate(zip(stall_fields, stall_labels)):
        axes[1, 1].bar(x + (index - 1.5) * 0.18,
                       [float(r[field]) for r in rows], 0.18, label=label)
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].set_ylabel("Warp-stall ratio (inst)")
    axes[1, 1].set_title("Selected warp issue stalls")
    axes[1, 1].legend(fontsize=7)

    for ax in axes.flat:
        ax.grid(True, axis="y", alpha=0.22)
        ax.tick_params(axis="x", rotation=10)
    fig.suptitle("H200 Nsight Compute — FP16, d=128, 16384 tokens")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Visualize kernel stability and CUDA API costs from an NSYS SQLite export."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    db = sqlite3.connect(args.sqlite)
    kernels = list(db.execute(
        """select k.end-k.start, s.value from CUPTI_ACTIVITY_KIND_KERNEL k
           join StringIds s on s.id=k.shortName order by k.start"""
    ))
    apis = list(db.execute(
        """select s.value, count(*), sum(r.end-r.start)/1e6
           from CUPTI_ACTIVITY_KIND_RUNTIME r
           join StringIds s on s.id=r.nameId group by s.value
           order by sum(r.end-r.start) desc limit 10"""
    ))
    gpu = db.execute("select name from TARGET_INFO_GPU limit 1").fetchone()
    if not kernels:
        raise SystemExit("Nsight trace has no CUDA kernels")

    durations = [row[0] / 1e3 for row in kernels]
    indices = list(range(len(durations)))
    colors = ["#6a1b9a" if i == 0 else "#f9a825" if i <= 5 else "#1976d2"
              for i in indices]

    fig = plt.figure(figsize=(11, 7.5))
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.05))
    ax_sequence = fig.add_subplot(grid[0, :])
    ax_hist = fig.add_subplot(grid[1, 0])
    ax_api = fig.add_subplot(grid[1, 1])

    ax_sequence.scatter(indices, durations, c=colors, s=24)
    ax_sequence.plot(indices, durations, color="#777777", linewidth=0.7, alpha=0.6)
    ax_sequence.axvline(0.5, color="#999999", linestyle="--", linewidth=0.8)
    ax_sequence.axvline(5.5, color="#999999", linestyle="--", linewidth=0.8)
    ax_sequence.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#6a1b9a",
               markeredgecolor="none", label="dispatch check"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#f9a825",
               markeredgecolor="none", label="warmup"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1976d2",
               markeredgecolor="none", label="measured"),
    ], loc="upper right", ncol=3, fontsize=8)
    ax_sequence.set_xlabel("Kernel launch index")
    ax_sequence.set_ylabel("GPU kernel duration (μs)")
    ax_sequence.set_title("Kernel duration across the captured launch sequence")
    ax_sequence.grid(True, alpha=0.22)

    ax_hist.hist(durations, bins=min(14, max(5, len(durations) // 4)), color="#1976d2",
                 edgecolor="white")
    ax_hist.axvline(sum(durations) / len(durations), color="#d84315", linewidth=2,
                    label=f"mean {sum(durations) / len(durations):.2f} μs")
    ax_hist.set_xlabel("GPU kernel duration (μs)")
    ax_hist.set_ylabel("Launch count")
    ax_hist.set_title("Kernel-duration distribution")
    ax_hist.legend()
    ax_hist.grid(True, axis="y", alpha=0.22)

    api_names = [name.replace("_v3020", "").replace("_v7000", "").replace("_v5050", "")
                 for name, _, _ in reversed(apis)]
    totals = [total for _, _, total in reversed(apis)]
    bars = ax_api.barh(api_names, totals, color="#00897b")
    ax_api.set_xscale("log")
    ax_api.set_xlim(min(totals) * 0.8, max(totals) * 1.45)
    ax_api.set_xlabel("Total CPU API time (ms, log scale)")
    ax_api.set_title("Top CUDA Runtime API totals")
    ax_api.grid(True, axis="x", alpha=0.22)
    for bar, total in zip(bars, totals):
        ax_api.text(total * 1.06, bar.get_y() + bar.get_height() / 2, f"{total:.3g}",
                    va="center", fontsize=7)

    gpu_name = gpu[0] if gpu else "unknown GPU"
    fig.suptitle(f"Nsight Systems trace summary — {gpu_name}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

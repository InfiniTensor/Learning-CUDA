#!/usr/bin/env python3
"""Render measured medians; the benchmark itself has no plotting dependency."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
summary = json.loads((root / "reports/summary.json").read_text())
families = summary["comparisons"]
labels = [f["family"].capitalize() for f in families]
colors = ["#197b89", "#9b5de5", "#f5a442"]
variants = [("cuda_thread_graph", "CUDA (thread + graphs)"),
            ("cpu_dinic", "CPU Dinic"), ("cpu_preflow", "CPU preflow")]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
x = np.arange(len(families))
for ax, metric, title in zip(axes, ["total_query_ms", "process_until_log_ms"],
                           ["10 queries: resident graph", "Cold run: input + setup + queries + output"]):
    for i, ((variant, label), color) in enumerate(zip(variants, colors)):
        values = [f["medians"][variant][metric] for f in families]
        bars = ax.bar(x + (i - 1) * 0.25, values, 0.23, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=3)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Wall time (ms), lower is better")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.margins(y=0.16)
axes[0].legend(loc="upper left", fontsize=9)
fig.suptitle(f"Trade network max flow | N={summary['nodes']:,}, M={summary['edges']:,} | median of 3 runs", fontsize=13)
fig.tight_layout()
fig.savefig(root / "reports/performance.png", dpi=170)
fig.savefig(root / "reports/performance.svg")
plt.close(fig)

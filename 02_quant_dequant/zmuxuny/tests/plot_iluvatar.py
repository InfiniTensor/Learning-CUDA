"""Render standalone MR-V100 performance figures from saved interleaved trials."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
HAD = ROOT.parent.name == "03_hadamard_tc"
r = json.loads((ROOT / "results/iluvatar/comparison.json").read_text())["records"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})
colors = ["#8b97a6", "#366f9d", "#18877c"]
if HAD:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for ax, rows in zip(axes, [8192, 65536]):
        records = [v for v in r if v["rows"] == rows and v["cols"] == 1024 and v["format"] == "mxfp8"]
        x = np.arange(len(records))
        for j, (phase, key, label) in enumerate([
            ("before", "hadamard_ms", "Initial butterfly"),
            ("after", "hadamard_ms", "Tuned butterfly"),
            ("after", "factorized_tc_ms", "CoreX matrix")]):
            values = [v["median"][phase][key] * 1000 for v in records]
            bars = ax.bar(x + (j-1)*.24, values, .23, color=colors[j], label=label)
            ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        ax.set_xticks(x, [v["dtype"].upper() for v in records])
        ax.set_title(f"{rows:,} x 1024 ({rows * 1024 * 2 / 2**20:.0f} MiB input)")
        ax.set_ylabel("Hadamard time (us)")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("MR-V100: native matrix decomposition and tuned butterfly")
else:
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    records = [v for v in r if v["rows"] >= 32768]
    x = np.arange(len(records))
    for j, phase in enumerate(["before", "after"]):
        values = [v["median"][phase]["quant_ms"] * 1000 for v in records]
        bars = ax.bar(x + (j-.5)*.34, values, .32, color=colors[j*2],
                      label="Initial port" if j==0 else "MR-V100 tuned")
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.set_xticks(x, [v["dtype"].upper()+"\n"+v["format"].upper() for v in records])
    ax.set_ylabel("Full quantization time (us)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    ax.set_title("MR-V100: software quantization, 128 MiB input")
fig.savefig(ROOT / "results/iluvatar/performance.png", dpi=180)
plt.close(fig)

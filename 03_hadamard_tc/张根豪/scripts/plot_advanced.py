#!/usr/bin/env python3
"""Visualize baseline optimization and fused INT4 results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise SystemExit("advanced result CSV is empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    colors = {"fp16": "#1976d2", "bf16": "#d84315"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    for dtype in ("fp16", "bf16"):
        subset = [row for row in rows if row["dtype"] == dtype]
        dims = sorted({int(row["head_dim"]) for row in subset})
        by_key = {(int(row["head_dim"]), row["implementation"]): row for row in subset}
        optimized_speedup = [float(by_key[(dim, "optimized")]["speedup_vs_baseline"])
                             for dim in dims]
        fused_speedup = [float(by_key[(dim, "fused_int4")]["fused_speedup_vs_unfused"])
                         for dim in dims]
        quant_mae = [float(by_key[(dim, "fused_int4")]["quant_mae"]) for dim in dims]
        axes[0, 0].plot(dims, optimized_speedup, marker="o", linewidth=2,
                        color=colors[dtype], label=dtype.upper())
        axes[0, 1].plot(dims, fused_speedup, marker="o", linewidth=2,
                        color=colors[dtype], label=dtype.upper())
        axes[1, 1].plot(dims, quant_mae, marker="o", linewidth=2,
                        color=colors[dtype], label=dtype.upper())

    implementations = ["baseline", "optimized", "unfused_int4", "fused_int4"]
    labels = ["baseline", "optimized", "unfused INT4", "fused INT4"]
    fp16_128 = {(row["implementation"]): row for row in rows
                if row["dtype"] == "fp16" and int(row["head_dim"]) == 128}
    latencies = [float(fp16_128[name]["avg_ms"]) * 1000 for name in implementations]
    axes[1, 0].bar(labels, latencies,
                   color=["#757575", "#1976d2", "#f9a825", "#00897b"])
    axes[1, 0].tick_params(axis="x", rotation=16)

    panels = [
        (axes[0, 0], "Optimized FHT speedup vs baseline", "Speedup (×)"),
        (axes[0, 1], "Fused INT4 speedup vs unfused pipeline", "Speedup (×)"),
        (axes[1, 0], "FP16 d=128 end-to-end latency", "Latency (μs)"),
        (axes[1, 1], "Per-token INT4 quantization MAE", "MAE"),
    ]
    for ax, title, ylabel in panels:
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
        ax.set_xlabel("head_dim")
        ax.set_xticks([64, 128, 256])
        ax.legend()
    fig.suptitle("Fast Hadamard optimization and fused INT4")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot simulation and cold-process end-to-end time from benchmark_io.py."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.summary_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("summary CSV is empty")

    labels = [row["case"].replace("record_", "") for row in rows]
    simulation = [float(row["simulation_median_ms"]) for row in rows]
    end_to_end = [float(row["end_to_end_median_ms"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(x - width / 2, simulation, width, label="simulation")
    ax.bar(x + width / 2, end_to_end, width, label="cold end-to-end")
    ax.set_xticks(x, labels)
    ax.set_ylabel("median time (ms)")
    ax.set_title("4096 particles × 1000 steps: trajectory output cost")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()

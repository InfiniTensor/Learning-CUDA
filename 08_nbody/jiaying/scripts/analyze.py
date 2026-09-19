#!/usr/bin/env python3
"""Summarize measured results and draw the report's reproducible figures."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from trajectory import load_trajectory


def main():
    reports = Path("reports")
    rows = list(csv.DictReader((reports / "benchmark.csv").open()))
    medians = {}
    for n in (4096, 16384, 65536):
        for kernel, block in (("naive", 128), ("tiled", 128), ("tiled", 256)):
            values = [float(row["mean_step_ms"]) for row in rows if row["case"].startswith(f"scale_{n}_{kernel}_{block}_")]
            medians[f"{n}_{kernel}_{block}"] = float(np.median(values))
    cpu = json.loads((reports / "metrics/required_cpu_4096_1000.json").read_text())
    gpu = json.loads((reports / "metrics/required_cuda_4096_1000.json").read_text())
    large = json.loads((reports / "metrics/advanced_cuda_65536.json").read_text())
    cp, _, _ = load_trajectory("output/required_cpu_4096_1000.bin")
    gp, _, _ = load_trajectory("output/required_cuda_4096_1000.bin")
    lp, _, _ = load_trajectory("output/advanced_cuda_65536.bin")
    max_error = float(np.max(np.abs(cp.astype(np.float64) - gp)))
    assert max_error < 2e-4, max_error
    for start in range(0, len(lp), 1024):
        assert np.all(np.isfinite(lp[start:start + 1024]))
    assert tuple(lp.shape) == (65536, 101, 3), lp.shape
    assert gpu["diagnostics"]["relative_energy_error"] < 1e-4
    assert gpu["diagnostics"]["normalized_momentum_error"] < 1e-5
    summary = {"median_step_ms": medians, "compute_speedup": cpu["compute_ms"] / gpu["compute_ms"],
               "simulation_wall_speedup": cpu["simulation_wall_ms"] / gpu["simulation_wall_ms"],
               "process_wall_speedup": cpu["process_wall_ms"] / gpu["process_wall_ms"],
               "all_recorded_cpu_gpu_max_position_error": max_error,
               "advanced_shape": list(lp.shape), "advanced_finite": True,
               "advanced_compute_ms": large["compute_ms"], "advanced_process_ms": large["process_wall_ms"],
               "advanced_trajectory_bytes": Path("output/advanced_cuda_65536.bin").stat().st_size}
    (reports / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    ns = [4096, 16384, 65536]
    for kernel, block, color in (("naive", 128, "#ad6b45"), ("tiled", 128, "#0d8ca7"), ("tiled", 256, "#8069a7")):
        axes[0].plot(ns, [medians[f"{n}_{kernel}_{block}"] for n in ns], "o-", label=f"{kernel}, block {block}", color=color)
    axes[0].set_xscale("log", base=2); axes[0].set_yscale("log")
    axes[0].set_xticks(ns, [f"{n:,}" for n in ns]); axes[0].set_xlabel("Particles")
    axes[0].set_ylabel("Mean step time (ms, median of 3 runs)")
    axes[0].set_title("Direct gravity kernel variants"); axes[0].legend()
    bars = axes[1].bar(["CPU / 1 thread", "CUDA"], [cpu["process_wall_ms"] / 1000, gpu["process_wall_ms"] / 1000], color=["#ad6b45", "#0d8ca7"])
    axes[1].bar_label(bars, fmt="%.3f s", padding=4)
    axes[1].set_ylim(0, cpu["process_wall_ms"] / 1000 * 1.18)
    axes[1].set_ylabel("Complete process wall time (s)")
    axes[1].set_title(f"4096 bodies × 1000 steps / {summary['process_wall_speedup']:.1f}× speedup")
    fig.savefig(reports / "performance.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for path, label, color in (("binary", "Leapfrog", "#0d8ca7"), ("binary_euler", "Explicit Euler", "#ad6b45")):
        p, time, _ = load_trajectory(f"output/{path}.bin")
        radius_error = np.abs(np.linalg.norm(p[0] - p[1], axis=1) - 1)
        ax.semilogy(time, np.maximum(radius_error, 1e-8), label=label, color=color)
    ax.set_xlabel("Time (normalized units)"); ax.set_ylabel("Absolute separation error")
    ax.set_title("Circular binary: dt = 0.005, 4000 steps"); ax.legend()
    fig.savefig(reports / "orbit_accuracy.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reproducible CUDA/CPU timing, ablations, mandatory and advanced scale runs."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from generate import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=Path("build/nbody"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--large-steps", type=int, default=1000)
    parser.add_argument("--cpu-steps", type=int, default=1000)
    args = parser.parse_args()
    if min(args.repeats, args.large_steps, args.cpu_steps) < 1:
        parser.error("Counts must be positive")
    exe = args.exe.resolve()
    data_dir = Path("data"); data_dir.mkdir(exist_ok=True)
    output = Path("output"); output.mkdir(exist_ok=True)
    reports = Path("reports"); reports.mkdir(exist_ok=True)
    raw = reports / "metrics"; raw.mkdir(exist_ok=True)
    os.environ["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    rows = []
    for n in (4096, 16384, 65536):
        np.savetxt(data_dir / f"cluster_{n}.txt", generate("cluster", n), fmt="%.9g")

    def run(name, n, steps, backend="cuda", kernel="tiled", block=128, trajectory=False, diagnostics=False, threads=1):
        config = output / f"{name}.txt"
        config.write_text(f"dt=0.001\nnum_steps={steps}\nrecord_interval=10\nG=1\nsoftening=0.03\nintegrator=leapfrog\n")
        cmd = [str(exe), "--input", str(data_dir / f"cluster_{n}.txt"), "--config", str(config), "--backend", backend,
               "--kernel", kernel, "--block-size", str(block), "--cpu-threads", str(threads), "--log", str(raw / f"{name}.json")]
        if trajectory:
            cmd += ["--output", str(output / f"{name}.bin")]
        else:
            cmd += ["--no-trajectory"]
        if not diagnostics:
            cmd += ["--no-diagnostics"]
        begin = time.perf_counter()
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        process_ms = (time.perf_counter() - begin) * 1000
        metrics = json.loads((raw / f"{name}.json").read_text())
        metrics["process_wall_ms"] = process_ms
        (raw / f"{name}.json").write_text(json.dumps(metrics, indent=2) + "\n")
        row = {"case": name, "particles": n, "steps": steps, "backend": backend, "kernel": kernel,
               "block_size": block, "cpu_threads": threads, "trajectory": trajectory,
               "compute_ms": metrics["compute_ms"], "simulation_wall_ms": metrics["simulation_wall_ms"],
               "process_wall_ms": process_ms, "mean_step_ms": metrics["mean_step_ms"],
               "particle_steps_per_second": metrics["particle_steps_per_second"], "device_allocated_bytes": metrics["device_allocated_bytes"]}
        rows.append(row)
        with (reports / "benchmark.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
        print(f"{name}: compute={row['compute_ms']:.3f}ms, process={process_ms:.3f}ms", flush=True)

    run("warmup", 4096, 50)
    for n in (4096, 16384, 65536):
        for kernel, block in (("naive", 128), ("tiled", 128), ("tiled", 256)):
            for repeat in range(args.repeats):
                run(f"scale_{n}_{kernel}_{block}_{repeat}", n, 100, kernel=kernel, block=block)
    run("required_cuda_4096_1000", 4096, 1000, trajectory=True, diagnostics=True)
    run(f"required_cpu_4096_{args.cpu_steps}", 4096, args.cpu_steps, backend="cpu", trajectory=True, diagnostics=True)
    run("advanced_cuda_65536", 65536, args.large_steps, trajectory=True)
    # Same no-output workload, optional multithreaded CPU reference.
    run("cpu_single_4096_100", 4096, 100, backend="cpu")
    run("cpu_parallel_4096_100", 4096, 100, backend="cpu", threads=12)
    print(f"Saved {len(rows)} runs to reports/benchmark.csv")


if __name__ == "__main__":
    main()

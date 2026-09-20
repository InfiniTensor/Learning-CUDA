"""Reproducible serial benchmark runner; never overlaps CPU/GPU benchmark jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generate import make_particles, write_config
from trajectory import load_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "benchmark")
    parser.add_argument("--evidence", type=Path, default=ROOT / "evidence" / "benchmark.json")
    parser.add_argument("--skip-large", action="store_true")
    args = parser.parse_args()
    exe = args.exe.resolve()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = {"platform": platform.platform(), "python": platform.python_version(),
              "device": json.loads(subprocess.check_output([str(exe), "--device-info"], text=True)),
              "runs": [], "summaries": {}, "executable_sha256": hashlib.sha256(exe.read_bytes()).hexdigest()}
    args.evidence.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        args.evidence.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def run(label, n=4096, steps=1000, backend="cuda", kernel="tiled", block=256,
            threads=1, trajectory=False, compare=False):
        input_path = root / f"cluster_{n}.txt"
        if not input_path.exists():
            np.savetxt(input_path, make_particles("cluster", n), fmt="%.9g", header="x y z vx vy vz mass")
        config_path = root / f"steps_{steps}.cfg"
        if not config_path.exists():
            write_config(config_path, dt=0.001, num_steps=steps, record_interval=max(1, steps // 100), softening=0.02)
        command = [str(exe), "--input", str(input_path), "--config", str(config_path), "--backend", backend,
                   "--kernel", kernel, "--block-size", str(block), "--cpu-threads", str(threads),
                   "--log", str(root / f"{label}.json"), "--diagnostics", "auto"]
        if trajectory:
            command += ["--output", str(root / f"{label}.bin"), "--final-state", str(root / f"{label}_final.csv")]
        else:
            command += ["--no-trajectory"]
        if compare:
            command += ["--compare-cpu"]
        print(f"Running {label}: {n} x {steps}", flush=True)
        started = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True, timeout=1800)
        wall = time.perf_counter() - started
        (root / f"{label}.txt").write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        entry = json.loads((root / f"{label}.json").read_text())
        entry.update(label=label, command=command, process_wall_seconds=wall)
        if trajectory:
            data, metadata = load_trajectory(root / f"{label}.bin")
            entry["trajectory_verified_shape"] = list(data.shape)
            entry["trajectory_bytes"] = (root / f"{label}.bin").stat().st_size
            entry["trajectory_sha256"] = hashlib.sha256((root / f"{label}.bin").read_bytes()).hexdigest()
            del data
        report["runs"].append(entry)
        persist()
        print(completed.stdout.strip(), flush=True)
        return entry

    # Short tuning runs choose a competitive OpenMP baseline on this host.
    cpu_tuning = [run(f"cpu_tune_{threads}", steps=60, backend="cpu", threads=threads) for threads in (1, 4, 8, 16, 24)]
    best_threads = min(cpu_tuning, key=lambda x: x["simulation_seconds"])["cpu_threads"]
    report["summaries"]["selected_cpu_threads"] = best_threads
    kernel_runs = {}
    for kernel, block in (("naive", 256), ("tiled", 128), ("tiled", 256), ("tiled", 512)):
        label = f"{kernel}_{block}"
        kernel_runs[label] = [run(f"{label}_repeat_{repeat}", kernel=kernel, block=block) for repeat in range(1, 4)]
    report["summaries"]["kernel_median_seconds"] = {label: statistics.median(x["simulation_seconds"] for x in rows)
                                                    for label, rows in kernel_runs.items()}
    run("baseline_4096", trajectory=True, compare=True, threads=best_threads)
    if not args.skip_large:
        run("advanced_65536", n=65536, trajectory=True)
        run("large_tail_65537", n=65537, steps=20)
        run("scale_100000", n=100000, steps=20)
    persist()
    print(args.evidence)


if __name__ == "__main__":
    main()

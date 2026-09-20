#!/usr/bin/env python3
"""Reproducible experiments. Every plotted point retains its config and raw JSON."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exe", type=Path, default=Path("build/option_pricing"))
    p.add_argument("--output", type=Path, default=Path("results"))
    p.add_argument("--large-paths", type=int, default=10_000_000)
    p.add_argument("--repeats", type=int, default=3)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    raw = a.output / "raw"
    raw.mkdir(exist_ok=True)
    exe = str(a.exe.resolve())
    rows = []
    def run(group, kind="european_call", vr="none", n=100000, steps=256, seed=1234, backend="gpu", block=256, rng="curand", repeat=0):
        name = f"{group}_{kind}_{vr}_{n}_{steps}_{seed}_{block}_{rng}_{repeat}"
        op, sp, out = raw / (name + "_option.cfg"), raw / (name + "_simulation.cfg"), raw / name
        op.write_text(f"option_type={kind}\nspot=100\nstrike=100\nrisk_free_rate=.03\nvolatility=.2\nmaturity=1\nbarrier=130\n")
        sp.write_text(f"num_paths={n}\nnum_steps={steps}\nseed={seed}\nrng={rng}\nvariance_reduction={vr}\nblock_size={block}\n")
        subprocess.run([exe, "--option", str(op), "--simulation", str(sp), "--backend", backend, "--output", str(out)], check=True)
        d = json.loads(out.with_suffix(".json").read_text())
        perf = json.loads(Path(str(out) + "_performance.json").read_text())
        g = d["gpu"]
        row = dict(group=group, kind=kind, variance_reduction=vr, paths=n, steps=steps, seed=seed,
                   block_size=block, rng=rng, repeat=repeat, price=g["price"], standard_error=g["standard_error"],
                   absolute_error=g["absolute_error"], variance_ratio=g["raw_to_adjusted_variance_ratio"],
                   gpu_ms=perf["gpu"]["total_ms"], kernel_ms=perf["gpu"]["simulation_ms"],
                   reduction_ms=perf["gpu"]["reduction_ms"], context_ms=perf["gpu"]["context_setup_ms"],
                   cpu_ms=perf.get("cpu", {}).get("total_ms"), speedup=perf.get("speedup"),
                   cold_speedup=perf.get("cold_speedup"), cpu_gpu_difference=perf.get("cpu_gpu_price_difference"),
                   paths_per_second=perf["gpu"]["paths_per_second"], workspace_bytes=perf["gpu"]["workspace_bytes"])
        rows.append(row)
        with (a.output / "benchmark.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        return row
    for kind in ("european_call", "european_put", "asian_call", "barrier_call"):
        for vr in ("none", "antithetic", "control_variate"):
            for repeat in range(a.repeats):
                run("performance", kind, vr, n=1_000_000, backend="both", repeat=repeat)
    for n in (1000, 10000, 100000, 1000000, a.large_paths):
        for seed in (11, 29, 47, 71, 97, 131, 173, 211):
            run("convergence", n=n, seed=seed)
    for kind in ("european_call", "asian_call", "barrier_call"):
        run("large", kind, n=a.large_paths, backend="both")
    for block in (64, 128, 256, 512):
        for rng in ("curand", "custom"):
            for repeat in range(a.repeats):
                run("tuning", "asian_call", n=1_000_000, block=block, rng=rng, repeat=repeat)
    metadata = dict(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"), repetitions=a.repeats,
                    large_paths=a.large_paths, rows=len(rows), commands={})
    for name, cmd in dict(os=["uname", "-a"], cpu=["lscpu"], cuda=["nvcc", "--version"],
                          gpu=["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"]).items():
        metadata["commands"][name] = subprocess.run(cmd, capture_output=True, text=True).stdout
    (a.output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(rows)} experiment runs to {a.output}")


if __name__ == "__main__":
    main()

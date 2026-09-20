"""Interleaved full-pipeline comparison with explicit dtype and binary provenance.

All binaries see exactly the same input. Alternate order, retain every trial,
check packed-file hashes, and keep profiling separate from timing.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import numpy as np
from reference import write_tensor

ROOT = Path(__file__).resolve().parents[1]
HADAMARD = ROOT.parent.name == "03_hadamard_tc"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--native", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--suite", choices=["small", "large", "all"], default="all")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--baseline-label", default="first optimized 4090 D submission (2026-09-19)")
    a = p.parse_args()
    binaries = {
        k: getattr(a, k).resolve()
        for k in ("before", "after", "native")
        if getattr(a, k)
    }
    small = (
        [
            (r, d, ty)
            for ty in [1, 2]
            for r, d in [
                (32, 64),
                (8192, 64),
                (8192, 128),
                (8192, 256),
                (8192, 512),
                (8192, 1024),
            ]
        ]
        if HADAMARD
        else [(r, 1024, ty) for ty in [0, 1] for r in [32, 1024]]
    )
    large = (
        [(65536, 1024, ty) for ty in [1, 2]]
        if HADAMARD
        else [(32768, 1024, 0), (65536, 1024, 1)]
    )
    shapes = (small if a.suite != "large" else []) + (
        large if a.suite != "small" else []
    )
    report = {
        "baseline": a.baseline_label,
        "seed": 42,
        "distribution": "normal",
        "quant_output_dtype": "fp32",
        "timing": "device runtime events, warmup, all GPU steps; excludes host reference and transfers",
        "binaries": {
            k: {"path": str(v), "sha256": digest(v)} for k, v in binaries.items()
        },
        "records": [],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cuda-tuning-") as tmp:
        t = Path(tmp)
        for rows, cols, dtype in shapes:
            write_tensor(
                t / "input",
                np.random.default_rng(42).normal(size=(rows, cols)).astype(np.float32),
                dtype,
            )
            repeats = 30 if rows * cols > 16_000_000 else 100
            for fmt in ["mxfp8", "nvfp4"]:
                trials = {k: [] for k in binaries}
                hashes = {k: [] for k in binaries}
                for trial in range(a.trials):
                    labels = list(binaries)
                    labels = (
                        labels[trial % len(labels) :] + labels[: trial % len(labels)]
                    )
                    for label in labels:
                        cmd = [
                            binaries[label],
                            "--input",
                            t / "input",
                            "--output",
                            t / "out",
                            "--packed",
                            t / "packed",
                            "--log",
                            t / "log",
                            "--repeats",
                            str(repeats),
                        ]
                        if HADAMARD:
                            cmd += [
                                "--format",
                                fmt,
                                "--factorized_tc_packed",
                                t / "mma_packed",
                            ]
                            if label != "before":
                                cmd += ["--materialized_compare", "1"]
                        else:
                            (t / "config").write_text(
                                f'format = "{fmt}"\nscale_mode = "block"\noutput_type = "fp32"\nrounding = "nearest"\n'
                            )
                            cmd += ["--config", t / "config"]
                        proc = subprocess.run(
                            list(map(str, cmd)), capture_output=True, text=True
                        )
                        if proc.returncode:
                            raise RuntimeError(proc.stdout + proc.stderr)
                        metrics = json.loads((t / "log").read_text())
                        trials[label].append(metrics)
                        hashes[label].append(
                            {
                                "packed": digest(t / "packed"),
                                **(
                                    {"mma_packed": digest(t / "mma_packed")}
                                    if HADAMARD
                                    else {}
                                ),
                            }
                        )
                        print(rows, cols, dtype, fmt, label, trial, flush=True)
                reference = hashes["before"][0]
                assert all(h == reference for hs in hashes.values() for h in hs), (
                    rows,
                    cols,
                    dtype,
                    fmt,
                    hashes,
                )
                median = {
                    label: {
                        k: statistics.median(v[k] for v in values)
                        for k, value in values[0].items()
                        if isinstance(value, (int, float))
                    }
                    for label, values in trials.items()
                }
                report["records"].append(
                    {
                        "rows": rows,
                        "cols": cols,
                        "dtype": ["fp32", "fp16", "bf16"][dtype],
                        "format": fmt,
                        "repeats": repeats,
                        "median": median,
                        "trials": trials,
                        "packed_sha256": hashes,
                        "packed_equal": True,
                    }
                )
                a.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Measure standalone/fused/WMMA kernels and rotation's reconstruction error."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from generate import generate
from reference import (
    cast,
    dequantize,
    quantize,
    read_packed,
    read_tensor,
    uniform,
    write_tensor,
)


def inverse_rotate(x):
    x = np.array(x, dtype=np.float64, copy=True)
    rows, d = x.shape
    step = 1
    while step < d:
        v = x.reshape(rows, -1, 2, step)
        lo = v[:, :, 0, :].copy()
        hi = v[:, :, 1, :].copy()
        v[:, :, 0, :] = lo + hi
        v[:, :, 1, :] = lo - hi
        step *= 2
    x /= np.sqrt(d)
    x *= np.where(uniform(np.arange(d), 7) < 0.5, -1.0, 1.0)
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--binary", type=Path, default=ROOT / "build/hadamard")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results")
    p.add_argument("--materialized-compare", action="store_true")
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with tempfile.TemporaryDirectory(prefix="had-bench-") as tmp:
        t = Path(tmp)
        for dtype in [1, 2]:
            for rows, d in [
                (32, 64),
                (8192, 64),
                (8192, 128),
                (8192, 256),
                (8192, 512),
                (8192, 1024),
            ]:
                x = generate(rows, d, "normal")
                write_tensor(t / "input", x, dtype)
                for fmt in ["mxfp8", "nvfp4"]:
                    trials = []
                    for _ in range(a.trials):
                        subprocess.run(
                            [
                                str(a.binary.resolve()),
                                "--materialized_compare",
                                str(int(a.materialized_compare)),
                                "--input",
                                str(t / "input"),
                                "--output",
                                str(t / "out"),
                                "--packed",
                                str(t / "packed"),
                                "--log",
                                str(t / "log"),
                                "--format",
                                fmt,
                                "--repeats",
                                str(a.repeats),
                                "--batch",
                                "1",
                                "--seq",
                                str(rows // 8),
                                "--heads",
                                "8",
                            ],
                            check=True,
                            capture_output=True,
                        )
                        trials.append(json.loads((t / "log").read_text()))
                    median = {
                        k: statistics.median(v[k] for v in trials)
                        for k in trials[0]
                        if isinstance(trials[0][k], (int, float))
                    }
                    records.append(
                        {
                            "rows": rows,
                            "dim": d,
                            "dtype": ["fp32", "fp16", "bf16"][dtype],
                            "format": fmt,
                            "median": median,
                            "trials": trials,
                        }
                    )
                    print(
                        dtype,
                        rows,
                        d,
                        fmt,
                        "had_ms",
                        median["hadamard_ms"],
                        "fusion",
                        median["fusion_speedup"],
                        flush=True,
                    )
        quality = []
        for dist in ["normal", "outliers"]:
            x = cast(generate(1024, 128, dist), 1)
            write_tensor(t / "input", x, 1)
            for fmt in [0, 1]:
                subprocess.run(
                    [
                        str(a.binary.resolve()),
                                "--materialized_compare",
                                str(int(a.materialized_compare)),
                        "--input",
                        str(t / "input"),
                        "--output",
                        str(t / "out"),
                        "--packed",
                        str(t / "packed"),
                        "--log",
                        str(t / "log"),
                        "--format",
                        ["mxfp8", "nvfp4"][fmt],
                        "--random_sign",
                        "1",
                        "--repeats",
                        "10",
                    ],
                    check=True,
                    capture_output=True,
                )
                rotated = dequantize(read_packed(t / "packed"))
                reconstructed = inverse_rotate(rotated)
                direct = dequantize(quantize(x, fmt, 32 if fmt == 0 else 16))
                quality.append(
                    {
                        "distribution": dist,
                        "format": ["mxfp8", "nvfp4"][fmt],
                        "direct_mse": float(
                            np.mean((x.astype(np.float64) - direct) ** 2)
                        ),
                        "rotate_quantize_inverse_mse": float(
                            np.mean((x.astype(np.float64) - reconstructed) ** 2)
                        ),
                    }
                )
    (a.output_dir / "benchmark.json").write_text(json.dumps(records, indent=2) + "\n")
    (a.output_dir / "rotation_quality.json").write_text(
        json.dumps(quality, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

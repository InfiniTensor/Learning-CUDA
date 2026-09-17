#!/usr/bin/env python3
"""Generate reproducible uniform, normal, or outlier matrices (or import .npy)."""
import argparse
from pathlib import Path
import numpy as np
from tensor_io import write_tensor


def generate(rows, cols, distribution, seed):
    rng = np.random.default_rng(seed)
    if distribution == "uniform":
        return rng.uniform(-1, 1, (rows, cols)).astype(np.float32)
    a = rng.standard_normal((rows, cols), dtype=np.float32)
    if distribution == "outliers":
        indices = rng.choice(a.size, size=max(1, a.size // 1000), replace=False)
        a.flat[indices] *= 100
    return a


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output")
    p.add_argument("--rows", type=int, default=1024)
    p.add_argument("--cols", type=int, default=1024)
    p.add_argument("--distribution", choices=["uniform", "normal", "outliers"], default="normal")
    p.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--npy", help="Import a 2D NumPy array instead of generating one")
    args = p.parse_args()
    values = np.load(args.npy, allow_pickle=False) if args.npy else generate(args.rows, args.cols, args.distribution, args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_tensor(args.output, values, args.dtype)
    print(f"Wrote {values.shape} {args.dtype}: {args.output}")

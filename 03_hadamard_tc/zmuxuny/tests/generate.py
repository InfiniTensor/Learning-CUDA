"""Create deterministic tensor input without PyTorch or new GPU instructions."""

import argparse
import numpy as np
from reference import write_tensor


def generate(rows, cols, distribution, seed=42):
    rng = np.random.default_rng(seed)
    if distribution == "uniform":
        x = rng.uniform(-1, 1, (rows, cols))
    else:
        x = rng.normal(0, 1, (rows, cols))
        if distribution == "outliers":
            mask = rng.random(x.shape) < 0.001
            x[mask] *= 50
    return x.astype(np.float32)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=1024)
    p.add_argument("--cols", type=int, default=1024)
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument(
        "--distribution", choices=["uniform", "normal", "outliers"], default="normal"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    write_tensor(
        a.output,
        generate(a.rows, a.cols, a.distribution, a.seed),
        ["fp32", "fp16", "bf16"].index(a.dtype),
    )

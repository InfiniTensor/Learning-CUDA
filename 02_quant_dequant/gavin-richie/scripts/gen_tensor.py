#!/usr/bin/env python3
"""Generate test tensor files in the project's binary format.

Usage:
  gen_tensor.py OUT.bin ROWS COLS DTYPE DISTRIBUTION [SEED]
  DTYPE in {fp32, fp16}; DISTRIBUTION in {random, normal, outlier}

Format: int64 rows, int64 cols, char dtype[4], row-major values.
"""
import struct
import sys

import numpy as np


def main():
    out, rows, cols, dtype, dist = sys.argv[1:6]
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 42
    rows, cols = int(rows), int(cols)
    rng = np.random.default_rng(seed)

    if dist == "random":
        data = rng.uniform(-10.0, 10.0, size=rows * cols).astype(np.float32)
    elif dist == "normal":
        data = rng.normal(0.0, 1.0, size=rows * cols).astype(np.float32)
    elif dist == "outlier":
        # mostly small values with sparse large outliers, like LLM activations
        data = rng.normal(0.0, 0.01, size=rows * cols).astype(np.float32)
        idx = rng.choice(rows * cols, size=max(1, rows * cols // 512),
                         replace=False)
        data[idx] = rng.uniform(-500, 500, size=len(idx)).astype(np.float32)
    else:
        sys.exit(f"unknown distribution: {dist}")

    with open(out, "wb") as f:
        f.write(struct.pack("<qq", rows, cols))
        f.write(dtype.encode())
        if dtype == "fp32":
            f.write(data.tobytes())
        elif dtype == "fp16":
            f.write(data.astype(np.float16).tobytes())
        else:
            sys.exit(f"unknown dtype: {dtype}")
    print(f"wrote {out}: {rows}x{cols} {dtype} {dist} (seed={seed})")


if __name__ == "__main__":
    main()

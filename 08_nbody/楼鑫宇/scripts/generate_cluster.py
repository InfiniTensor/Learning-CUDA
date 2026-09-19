#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.n < 2:
        parser.error("n must be at least 2")

    rng = np.random.default_rng(args.seed)
    direction = rng.normal(size=(args.n, 3))
    norms = np.linalg.norm(direction, axis=1, keepdims=True)
    while np.any(norms == 0.0):
        zero_rows = norms[:, 0] == 0.0
        direction[zero_rows] = rng.normal(size=(np.count_nonzero(zero_rows), 3))
        norms = np.linalg.norm(direction, axis=1, keepdims=True)
    direction /= norms
    position = direction * rng.random((args.n, 1)) ** (1.0 / 3.0)
    position -= position.mean(axis=0)
    velocity = rng.normal(scale=0.2, size=(args.n, 3))
    velocity -= velocity.mean(axis=0)
    mass = np.full((args.n, 1), 1.0 / args.n)
    rows = np.hstack((position, velocity, mass))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, rows, fmt="%.17g")
    print(f"wrote {args.n} particles to {output}")


if __name__ == "__main__":
    main()

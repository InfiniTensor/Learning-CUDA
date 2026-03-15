#!/usr/bin/env python3
"""Generate particles.txt with N particles (default 1000) for scale testing."""
import argparse
import random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num", type=int, default=1000, help="Number of particles")
    ap.add_argument("-o", "--output", default="data/particles_1k.txt", help="Output path")
    args = ap.parse_args()
    random.seed(42)
    with open(args.output, "w") as f:
        for i in range(args.num):
            x = random.uniform(-0.01, 0.01)
            y = random.uniform(-0.01, 0.01)
            z = random.uniform(-0.01, 0.01)
            vx = random.uniform(-1e6, 1e6)
            vy = random.uniform(-1e6, 1e6)
            vz = random.uniform(-0.5e6, 0.5e6)
            q = 1.6e-19 if i % 2 == 0 else -1.6e-19
            m = 9.1e-31
            f.write(f"{x} {y} {z} {vx} {vy} {vz} {q} {m}\n")
    print(f"Wrote {args.num} particles to {args.output}")

if __name__ == "__main__":
    main()

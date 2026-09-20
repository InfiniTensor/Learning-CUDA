#!/usr/bin/env python3
"""生成可复现的随机粒子，并把质心速度调到接近零。"""

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 N-body 随机输入")
    parser.add_argument("count", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--position-scale", type=float, default=1.0)
    parser.add_argument("--velocity-scale", type=float, default=0.05)
    parser.add_argument("--mass-min", type=float, default=0.5)
    parser.add_argument("--mass-max", type=float, default=1.5)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("count 必须为正")
    if args.mass_min <= 0 or args.mass_max < args.mass_min:
        parser.error("质量范围不合法")

    rng = np.random.default_rng(args.seed)
    position = rng.normal(0.0, args.position_scale, size=(args.count, 3))
    velocity = rng.normal(0.0, args.velocity_scale, size=(args.count, 3))
    mass = rng.uniform(args.mass_min, args.mass_max, size=args.count)

    # 去掉质心速度，这样初始总动量接近 0，守恒量更容易观察。
    center_velocity = (velocity * mass[:, None]).sum(axis=0) / mass.sum()
    velocity -= center_velocity
    table = np.column_stack((position, velocity, mass))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, table, fmt="%.9e", header="x y z vx vy vz mass")
    print(f"wrote {args.count} particles to {args.output.resolve()}")


if __name__ == "__main__":
    main()

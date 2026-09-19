#!/usr/bin/env python3
"""Deterministic normalized-unit initial conditions; total cluster mass = 1."""
import argparse
from pathlib import Path
import numpy as np


def generate(kind, n=4096, seed=2026, softening=0.01):
    rng = np.random.default_rng(seed)
    if kind in ("binary", "ellipse"):
        speed = np.sqrt(0.5 / (1 + softening**2)**1.5)
        if kind == "ellipse":
            speed *= np.sqrt(0.5)
        return np.array([[-0.5, 0, 0, 0, -speed, 0, 1], [0.5, 0, 0, 0, speed, 0, 1]])
    if n < 2:
        raise ValueError("cluster/ring/collision requires at least two particles")
    data = np.zeros((n, 7))
    if kind == "ring":
        data[0, 6] = 100
        theta = rng.uniform(0, 2 * np.pi, n - 1)
        radius = rng.uniform(1, 2.5, n - 1)
        data[1:, 0] = radius * np.cos(theta)
        data[1:, 1] = radius * np.sin(theta)
        data[1:, 2] = rng.normal(0, 0.02, n - 1)
        speed = np.sqrt(100 / radius)
        data[1:, 3] = -speed * np.sin(theta)
        data[1:, 4] = speed * np.cos(theta)
        data[1:, 6] = 1 / n
        data[-1] = [3, 0, 0, 0, np.sqrt(100 / 3), 0, 5]
    else:
        # Plummer spatial law and rejection-sampled isotropic equilibrium velocities.
        u = rng.uniform(0.001, 0.995, n)
        radius = 1 / np.sqrt(u**(-2 / 3) - 1)
        direction = rng.normal(size=(n, 3))
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        data[:, :3] = radius[:, None] * direction
        q = np.zeros(n)
        pending = np.arange(n)
        while len(pending):
            proposal = rng.random(len(pending))
            accepted = 0.1 * rng.random(len(pending)) < proposal**2 * (1 - proposal**2)**3.5
            q[pending[accepted]] = proposal[accepted]
            pending = pending[~accepted]
        direction = rng.normal(size=(n, 3))
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        data[:, 3:6] = direction * (q * np.sqrt(2) * (1 + radius**2)**(-0.25))[:, None]
        data[:, 6] = 1 / n
        if kind == "collision":
            data[:, :3] *= 0.35
            side = np.where(np.arange(n) < n // 2, -1, 1)
            data[:, 0] += 1.5 * side
            data[:, 3:6] *= 0.3
            data[:, 3] -= 0.45 * side
    data[:, :3] -= np.average(data[:, :3], axis=0, weights=data[:, 6])
    data[:, 3:6] -= np.average(data[:, 3:6], axis=0, weights=data[:, 6])
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scenario", choices=["cluster", "binary", "ellipse", "ring", "collision"], default="cluster")
    parser.add_argument("--particles", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--softening", type=float, default=0.01)
    args = parser.parse_args()
    data = generate(args.scenario, args.particles, args.seed, args.softening)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, data, fmt="%.9g", header="x y z vx vy vz mass")
    print(f"Wrote {len(data)} particles: {args.output}")


if __name__ == "__main__":
    main()

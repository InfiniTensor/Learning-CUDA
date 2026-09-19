#!/usr/bin/env python3
"""Generate the reproducible asteroid-belt perturbation demonstration."""

import argparse
from pathlib import Path

import numpy as np


def circular_speed(radius: np.ndarray, central_mass: float, softening: float):
    return np.sqrt(
        central_mass * radius * radius
        / np.power(radius * radius + softening * softening, 1.5)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perturber", choices=["on", "off"], default="on")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.n < 1:
        parser.error("n must be positive")

    central_mass = 1000.0
    belt_total_mass = 1.0
    softening = 0.02
    rng = np.random.default_rng(args.seed)

    angles = rng.uniform(0.0, 2.0 * np.pi, args.n)
    radii = np.sqrt(rng.uniform(5.0**2, 10.0**2, args.n))
    heights = rng.normal(0.0, 0.025, args.n)
    positions = np.column_stack(
        (radii * np.cos(angles), radii * np.sin(angles), heights)
    )
    speeds = circular_speed(radii, central_mass, softening)
    velocities = np.column_stack(
        (-speeds * np.sin(angles), speeds * np.cos(angles), np.zeros(args.n))
    )
    masses = np.full((args.n, 1), belt_total_mass / args.n)
    belt = np.hstack((positions, velocities, masses))

    rows = [np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, central_mass]]), belt]
    if args.perturber == "on":
        perturber_radius = np.sqrt(65.0)
        perturber_speed = 0.8 * float(
            circular_speed(np.array(perturber_radius), central_mass, softening)
        )
        rows.append(
            np.array([[8.0, 0.0, 1.0, 0.0, perturber_speed, 0.0, 1.0]])
        )

    particles = np.vstack(rows)
    total_mass = np.sum(particles[:, 6])
    center = np.sum(particles[:, :3] * particles[:, 6:7], axis=0) / total_mass
    center_velocity = (
        np.sum(particles[:, 3:6] * particles[:, 6:7], axis=0) / total_mass
    )
    particles[:, :3] -= center
    particles[:, 3:6] -= center_velocity

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, particles, fmt="%.17g")
    print(
        f"wrote {particles.shape[0]} particles to {args.output} "
        f"(perturber={args.perturber})"
    )


if __name__ == "__main__":
    main()

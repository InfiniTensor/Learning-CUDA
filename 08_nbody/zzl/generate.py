"""Reproducible initial conditions in normalized units (G=1)."""
import argparse
from pathlib import Path
import numpy as np


def make_particles(scene, n=4096, seed=42, softening=0.02):
    rng = np.random.default_rng(seed)
    if scene in {"binary", "ellipse"}:
        # Equal masses at separation 1. Circular speed includes Plummer softening.
        speed = np.sqrt(0.5 / (1 + softening**2)**1.5)
        if scene == "ellipse":
            speed *= 0.75
        return np.array([[-0.5, 0, 0, 0, -speed, 0, 1], [0.5, 0, 0, 0, speed, 0, 1]])
    if n < 3:
        raise ValueError("This scene needs at least three particles")
    data = np.zeros((n, 7))
    if scene == "cluster":
        directions = rng.normal(size=(n, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        data[:, :3] = directions * np.cbrt(rng.random(n))[:, None]
        data[:, 3:6] = rng.normal(0, 0.30, size=(n, 3))
        data[:, 6] = 1 / n
        data[:, :3] -= data[:, :3].mean(axis=0)
        data[:, 3:6] -= data[:, 3:6].mean(axis=0)
    elif scene in {"orbits", "orbits-control"}:
        data[0, 6] = 1.0
        data[1, 0] = 1.5
        data[1, 4] = np.sqrt(1.0 / 1.5)
        data[1, 6] = 0.025 if scene == "orbits" else 0.0
        r = rng.uniform(0.7, 2.3, n - 2)
        theta = rng.uniform(0, 2 * np.pi, n - 2)
        data[2:, 0] = r * np.cos(theta)
        data[2:, 1] = r * np.sin(theta)
        data[2:, 2] = rng.normal(0, 0.015, n - 2)
        speed = np.sqrt(r * r / (r * r + softening**2)**1.5)
        data[2:, 3] = -speed * np.sin(theta)
        data[2:, 4] = speed * np.cos(theta)
        data[2:, 6] = 1e-7
        # Same primary initial position/velocity in control and perturbed runs.
    elif scene == "encounter":
        sign = np.where(np.arange(n) < n // 2, -1.0, 1.0)
        data[:, :3] = rng.normal(0, 0.09, size=(n, 3))
        data[:, 0] += sign * 0.8
        data[:, 1] += sign * 0.07
        data[:, 3] = -sign * 0.5
        data[:, 6] = 0.002 / n
    else:
        raise ValueError(f"Unknown scene: {scene}")
    return data


def write_config(path, dt=0.001, num_steps=1000, record_interval=10, softening=0.02,
                 G=1.0, integrator="leapfrog"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(f'dt = {dt}\nnum_steps = {num_steps}\nrecord_interval = {record_interval}\n'
                     f'G = {G}\nsoftening = {softening}\nintegrator = "{integrator}"\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=["binary", "ellipse", "cluster", "orbits", "orbits-control", "encounter"], default="cluster")
    parser.add_argument("--particles", type=int, default=4096,
                        help="ignored by the binary and ellipse scenes, which use two particles")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--record-interval", type=int, default=10)
    parser.add_argument("--softening", type=float, default=0.02)
    parser.add_argument("--integrator", choices=["euler", "leapfrog"], default="leapfrog")
    args = parser.parse_args()
    if args.particles <= 0 or args.steps < 0 or args.record_interval <= 0 or args.dt <= 0 or args.softening <= 0:
        parser.error("Invalid particle count or simulation parameter")
    if args.scene not in {"binary", "ellipse"} and args.particles < 3:
        parser.error("This scene needs at least three particles")
    # Mirror read_config: the simulator needs dt and softening^2 to be normal float32 values,
    # so reject configurations here instead of emitting a file the simulator refuses.
    if not np.isfinite(np.float32(args.dt)) or np.float32(args.dt) == 0:
        parser.error("--dt must be representable as a positive float32")
    softened = np.float32(args.softening) ** 2
    if not np.isfinite(softened) or abs(softened) < np.finfo(np.float32).tiny:
        parser.error("--softening must have a normal float32 square")
    if args.output.exists() or (args.config and args.config.exists()):
        parser.error("Choose new output paths; existing files are preserved")
    data = make_particles(args.scene, args.particles, args.seed, args.softening)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        np.savetxt(stream, data, fmt="%.9g", header="x y z vx vy vz mass")
    if args.config:
        write_config(args.config, args.dt, args.steps, args.record_interval, args.softening, integrator=args.integrator)
    print(f"Wrote {len(data)} particles: {args.output}")


if __name__ == "__main__":
    main()

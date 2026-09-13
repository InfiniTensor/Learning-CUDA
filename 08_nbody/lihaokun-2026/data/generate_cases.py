"""Generate deterministic N-body input cases for benchmarking and validation.

Usage:
    python generate_cases.py
    python generate_cases.py --case benchmark_4096 --output-dir data

The generated particle files use: x y z vx vy vz mass.

Physical units for the galaxy cases:
    position: light-years (ly), velocity: ly/Myr, mass: solar masses (Msun)
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform
import random

ROOT = Path(__file__).parent

# 1 km/s is approximately 3.3356 ly/Myr.  The values below describe a
# Milky-Way-like stellar disk, not a precision astrophysical catalogue.
GALAXY_G = 0.1559  # ly^3 / (Msun * Myr^2)
GALAXY_RADIUS = 50000.0
GALAXY_HALF_THICKNESS = 1500.0
CENTRAL_BLACK_HOLE_MASS = 4.3e6


def write_case(name, particles, params, output_dir=ROOT):
    (output_dir / f"{name}_particles.txt").write_text(
        "\n".join("{:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g}".format(*p) for p in particles) + "\n",
        encoding="utf-8",
    )
    (output_dir / f"{name}_params.txt").write_text(
        "\n".join(f"{key} = {value}" for key, value in params.items()) + "\n",
        encoding="utf-8",
    )


def population_counts(count):
    """Return deterministic counts for the five hierarchical object types."""
    compact = max(2, round(count * 0.002))
    stars = max(1, round(count * 0.01))
    planets = max(1, round(count * 0.15))
    black_holes = 1
    neutron_white = compact
    asteroids = count - black_holes - neutron_white - stars - planets
    if asteroids < 1:
        raise ValueError("count is too small for the object population")
    return {
        "black_hole": black_holes,
        "neutron_star_white_dwarf": neutron_white,
        "star": stars,
        "planet": planets,
        "asteroid": asteroids,
    }


def random_mass(object_type, rng):
    """Sample astrophysical masses in solar masses (Msun)."""
    ranges = {
        "black_hole": (4.3e6, 4.3e6),
        "neutron_star_white_dwarf": (0.5, 2.3),
        "star": (0.08, 100.0),
        "planet": (1.0e-8, 1.0e-3),
        "asteroid": (1.0e-15, 1.0e-8),
    }
    low, high = ranges[object_type]
    return low if low == high else 10.0 ** rng.uniform(math.log10(low), math.log10(high))


def write_random_population(name, count, params, seed, output_dir=ROOT):
    """Stream a reproducible Milky-Way-scale random stellar population.

    The disk has a 100,000 ly diameter and a 3,000 ly thickness.  Position
    samples are random in the disk volume. Velocities have random radial,
    vertical and tangential components around the circular velocity generated
    by the central black hole. A full Milky-Way rotation curve would require
    an additional stellar/dark-matter potential, which is not an explicit
    particle in this direct N-body benchmark.
    """
    rng = random.Random(seed)
    path = output_dir / f"{name}_particles.txt"
    metadata_path = output_dir / f"{name}_objects.txt"
    counts = population_counts(count)
    object_types = [kind for kind, number in counts.items() for _ in range(number)]
    rng.shuffle(object_types)
    masses = []
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for index, object_type in enumerate(object_types):
            mass = random_mass(object_type, rng)
            if object_type == "black_hole":
                # The Milky Way's central black hole is the one intentional
                # non-random position; this avoids an arbitrary off-centre
                # 4.3-million-solar-mass attractor.
                values = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, mass)
            else:
                radius = 100.0 + (GALAXY_RADIUS - 100.0) * math.sqrt(rng.random())
                angle = rng.uniform(0.0, 2.0 * math.pi)
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                z = rng.uniform(-GALAXY_HALF_THICKNESS, GALAXY_HALF_THICKNESS)
                circular = math.sqrt(GALAXY_G * CENTRAL_BLACK_HOLE_MASS / radius)
                radial = rng.gauss(0.0, 0.30 * circular)
                tangential = circular * rng.uniform(0.80, 1.20)
                vertical = rng.gauss(0.0, 0.15 * circular)
                vx = radial * math.cos(angle) - tangential * math.sin(angle)
                vy = radial * math.sin(angle) + tangential * math.cos(angle)
                values = (x, y, z, vx, vy, vertical, mass)
            masses.append(mass)
            output.write("{:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g}\n".format(*values))
    with metadata_path.open("w", encoding="utf-8", newline="\n") as metadata:
        metadata.write("index type mass\n")
        for index, object_type in enumerate(object_types):
            metadata.write(f"{index} {object_type} {masses[index]:.9g}\n")
    with (output_dir / f"{name}_params.txt").open("w", encoding="utf-8", newline="\n") as output:
        output.write("\n".join(f"{key} = {value}" for key, value in params.items()) + "\n")
    manifest = {
        "case": name, "particle_count": count, "seed": seed,
        "python_version": platform.python_version(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "units": {"position": "ly", "velocity": "ly/Myr", "mass": "Msun", "time": "Myr"},
        "population_counts": counts, "parameters": params,
        "expected_records": int(params["num_steps"]) // int(params["record_interval"]) + 1,
        "sha256": {f"{name}_{suffix}.txt": hashlib.sha256(
            (output_dir / f"{name}_{suffix}.txt").read_bytes()).hexdigest()
            for suffix in ("particles", "params", "objects")},
    }
    (output_dir / f"{name}_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Generated {name}: particles={count}, seed={seed}, records={manifest['expected_records']}")


def two_body():
    # Equal masses in a circular orbit around their center of mass.
    speed = math.sqrt(0.5)
    return [(-0.5, 0, 0, 0, -speed, 0, 1), (0.5, 0, 0, 0, speed, 0, 1)]


def solar_system():
    # Normalized Sun + four planets with different orbital inclinations.
    particles = [(0, 0, 0, 0, 0, 0, 1000.0)]
    planets = [(0.4, 0.01, -8), (0.7, 0.02, 4), (1.0, 0.03, 12), (1.5, 0.04, -16)]
    for radius, mass, inclination_degrees in planets:
        inclination = math.radians(inclination_degrees)
        speed = math.sqrt(1000.0 / radius)
        particles.append((radius, 0, 0, 0, speed * math.cos(inclination),
                          speed * math.sin(inclination), mass))
    return particles


def cluster(count=64, seed=7):
    rng = random.Random(seed)
    particles = []
    for _ in range(count):
        # Gaussian compact cluster with small random velocities.
        x, y, z = (rng.gauss(0, 1) for _ in range(3))
        vx, vy, vz = (rng.gauss(0, 0.08) for _ in range(3))
        particles.append((x, y, z, vx, vy, vz, 1.0 / count))
    return particles


def plummer_sphere(count=256, seed=23):
    # Isotropic 3D cloud with tangential swirl for an obvious 3D animation.
    rng = random.Random(seed)
    particles = []
    for _ in range(count):
        u = max(rng.random(), 1e-6)
        radius = min((u ** (-2.0 / 3.0) - 1.0) ** -0.5, 8.0)
        cos_theta = 2 * rng.random() - 1
        sin_theta = math.sqrt(1 - cos_theta * cos_theta)
        phi = 2 * math.pi * rng.random()
        x = radius * sin_theta * math.cos(phi)
        y = radius * sin_theta * math.sin(phi)
        z = radius * cos_theta
        vx = -0.12 * y + rng.gauss(0, 0.025)
        vy = 0.12 * x + rng.gauss(0, 0.025)
        vz = rng.gauss(0, 0.04)
        particles.append((x, y, z, vx, vy, vz, 1.0 / count))
    return particles


def disk(count=1024, seed=11):
    rng = random.Random(seed)
    particles = [(0, 0, 0, 0, 0, 0, 1000.0)]
    for _ in range(count):
        radius = 0.5 + 4.5 * math.sqrt(rng.random())
        angle = 2 * math.pi * rng.random()
        x, y = radius * math.cos(angle), radius * math.sin(angle)
        velocity = math.sqrt(1000.0 / radius)
        particles.append((x, y, rng.gauss(0, 0.01), -velocity * math.sin(angle),
                          velocity * math.cos(angle), rng.gauss(0, 0.01), 1e-3))
    return particles


COMMON = {"dt": "1e-3", "record_interval": "1", "G": "1.0", "softening": "1e-4", "integrator": "leapfrog"}
SMALL_CASES = {
    "two_body": (two_body, {**COMMON, "num_steps": "1000"}),
    "solar_system": (solar_system, {**COMMON, "dt": "1e-4", "num_steps": "1000"}),
    "cluster_64": (cluster, {**COMMON, "dt": "2e-4", "num_steps": "1000"}),
    "plummer_256": (plummer_sphere, {**COMMON, "dt": "2e-3", "num_steps": "1000", "softening": "2e-2"}),
    "disk_1024": (disk, {**COMMON, "dt": "1e-4", "num_steps": "1000"}),
}
BENCHMARKS = {"benchmark_4096": (4096, 41), "benchmark_65536": (65536, 43)}
BENCHMARK_PARAMS = {"dt": "0.01", "record_interval": "1", "G": str(GALAXY_G),
                    "num_steps": "1000", "softening": "1.0", "integrator": "leapfrog"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", *SMALL_CASES, *BENCHMARKS], default="all",
                        help="Generate only this case; default: all (legacy behavior).")
    parser.add_argument("--output-dir", type=Path, default=ROOT,
                        help="Destination directory; default: directory containing this script.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = [*SMALL_CASES, *BENCHMARKS] if args.case == "all" else [args.case]
    for name in names:
        if name in BENCHMARKS:
            count, seed = BENCHMARKS[name]
            write_random_population(name, count, BENCHMARK_PARAMS, seed, args.output_dir)
        else:
            factory, params = SMALL_CASES[name]
            write_case(name, factory(), params, args.output_dir)
            print(f"Generated {name}")
    print("Output directory:", args.output_dir.resolve())


if __name__ == "__main__":
    main()

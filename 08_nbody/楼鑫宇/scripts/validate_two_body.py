#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir")
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    metadata = json.loads((result_dir / "metadata.json").read_text(encoding="utf-8"))

    with (result_dir / "trajectory.bin").open("rb") as stream:
        header = np.fromfile(stream, dtype="<i4", count=2)
    if header.size != 2:
        raise ValueError("trajectory header is incomplete")
    particle_count, record_count = map(int, header)
    if particle_count != 2 or record_count != metadata["record_count"]:
        raise ValueError("unexpected two-body trajectory shape")
    expected_size = 8 + 12 * particle_count * record_count
    if (result_dir / "trajectory.bin").stat().st_size != expected_size:
        raise ValueError("trajectory file size mismatch")
    trajectory = np.memmap(
        result_dir / "trajectory.bin", dtype="<f4", mode="r", offset=8,
        shape=(particle_count, record_count, 3))
    separation = np.linalg.norm(trajectory[1] - trajectory[0], axis=1)
    midpoint = 0.5 * (trajectory[0] + trajectory[1])

    with (result_dir / "diagnostics.csv").open(newline="", encoding="utf-8") as stream:
        diagnostics = list(csv.DictReader(stream))
    initial_energy = float(diagnostics[0]["total_energy"])
    final_energy = float(diagnostics[-1]["total_energy"])
    energy_scale = float(diagnostics[0]["kinetic"]) + abs(float(diagnostics[0]["potential"]))
    report = {
        "max_relative_separation_error": float(np.max(np.abs(separation - separation[0])) / separation[0]),
        "max_center_of_mass_distance": float(np.max(np.linalg.norm(midpoint, axis=1))),
        "relative_energy_error": abs(final_energy - initial_energy) / max(energy_scale, 1e-30),
        "all_coordinates_finite": bool(np.isfinite(trajectory).all()),
    }
    print(json.dumps(report, indent=2))
    if not report["all_coordinates_finite"]:
        raise SystemExit("non-finite coordinate detected")


if __name__ == "__main__":
    main()

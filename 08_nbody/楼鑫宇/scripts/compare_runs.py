#!/usr/bin/env python3
"""Compare one or more N-body result directories against a reference run."""

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def load_final_state(result_dir: Path):
    state = np.loadtxt(result_dir / "final_state.csv", delimiter=",", skiprows=1)
    if state.ndim == 1:
        state = state.reshape(1, -1)
    return state


def load_trajectory(result_dir: Path):
    with (result_dir / "trajectory.bin").open("rb") as stream:
        header = stream.read(8)
        if len(header) != 8:
            raise ValueError(f"invalid trajectory header: {result_dir}")
        particle_count, record_count = struct.unpack("<ii", header)
        trajectory = np.fromfile(stream, dtype="<f4")
    expected = particle_count * record_count * 3
    if trajectory.size != expected:
        raise ValueError(
            f"trajectory size mismatch in {result_dir}: "
            f"expected {expected}, got {trajectory.size}"
        )
    return trajectory.reshape(particle_count, record_count, 3)


def vector_metrics(reference, candidate):
    difference = candidate - reference
    particle_error = np.linalg.norm(difference, axis=-1)
    denominator = max(float(np.linalg.norm(reference)), 1.0e-12)
    return {
        "global_relative_l2": float(np.linalg.norm(difference) / denominator),
        "rms_absolute": float(np.sqrt(np.mean(particle_error**2))),
        "p50_absolute": float(np.percentile(particle_error, 50)),
        "p95_absolute": float(np.percentile(particle_error, 95)),
        "p99_absolute": float(np.percentile(particle_error, 99)),
        "max_absolute": float(np.max(particle_error)),
    }


def compare(reference_dir: Path, candidate_dir: Path):
    reference_state = load_final_state(reference_dir)
    candidate_state = load_final_state(candidate_dir)
    if reference_state.shape != candidate_state.shape:
        raise ValueError("final-state shapes differ")
    if not np.array_equal(reference_state[:, 0], candidate_state[:, 0]):
        raise ValueError("particle IDs differ")

    reference_metadata = json.loads(
        (reference_dir / "metadata.json").read_text(encoding="utf-8")
    )
    candidate_metadata = json.loads(
        (candidate_dir / "metadata.json").read_text(encoding="utf-8")
    )
    for key in ("input_file", "particle_count", "config"):
        if reference_metadata[key] != candidate_metadata[key]:
            raise ValueError(f"metadata {key} differs")
    if reference_metadata["recorded_steps"] != candidate_metadata["recorded_steps"]:
        raise ValueError("recorded steps differ")
    if reference_metadata["record_enabled"] != candidate_metadata["record_enabled"]:
        raise ValueError("recording modes differ")

    trajectory_metrics = None
    trajectory_finite = True
    if reference_metadata["record_enabled"]:
        reference_trajectory = load_trajectory(reference_dir)
        candidate_trajectory = load_trajectory(candidate_dir)
        if reference_trajectory.shape != candidate_trajectory.shape:
            raise ValueError("trajectory shapes differ")
        trajectory_difference = candidate_trajectory - reference_trajectory
        frame_denominator = np.maximum(
            np.linalg.norm(reference_trajectory, axis=(0, 2)), 1.0e-12
        )
        frame_relative_l2 = (
            np.linalg.norm(trajectory_difference, axis=(0, 2)) / frame_denominator
        )
        trajectory_metrics = {
            **vector_metrics(reference_trajectory, candidate_trajectory),
            "max_frame_relative_l2": float(np.max(frame_relative_l2)),
            "final_frame_relative_l2": float(frame_relative_l2[-1]),
        }
        trajectory_finite = bool(np.isfinite(candidate_trajectory).all())
    candidate_performance = json.loads(
        (candidate_dir / "performance.json").read_text(encoding="utf-8")
    )
    return {
        "candidate": str(candidate_dir),
        "all_values_finite": bool(
            np.isfinite(candidate_state).all()
            and trajectory_finite
        ),
        "position": vector_metrics(reference_state[:, 1:4], candidate_state[:, 1:4]),
        "velocity": vector_metrics(reference_state[:, 4:7], candidate_state[:, 4:7]),
        "trajectory": trajectory_metrics,
        "relative_energy_error": candidate_performance.get("relative_energy_error"),
        "absolute_momentum_error": candidate_performance.get("absolute_momentum_error"),
        "force_total_ms": candidate_performance["force_total_ms"],
        "simulation_wall_ms": candidate_performance["simulation_wall_ms"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    args = parser.parse_args()
    report = [compare(args.reference, candidate) for candidate in args.candidates]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

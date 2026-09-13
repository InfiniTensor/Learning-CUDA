"""Independent float64 CPU checks of CUDA-generated position trajectories.

Small systems: independent all-body velocity-Verlet/semi-implicit Euler replay.
Large systems: selected bodies replayed in the *recorded* all-body field, plus
global file/initial-state/COM checks. PASS_SAMPLED is deliberately not full proof.
Only NumPy is needed. A failed check exits nonzero and still writes its report.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from visualize import load_trajectory


def read_params(path):
    values = {"dt": "1e-3", "num_steps": "1000", "record_interval": "100",
              "G": "1", "softening": "1e-4", "integrator": "leapfrog"}
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"\'')
    for key in ("num_steps", "record_interval"):
        values[key] = int(values[key])
    # Match parameters/initial data parsed as float32 by nbody.cu, then use
    # independent float64 arithmetic in the reference calculation.
    for key in ("dt", "G", "softening"):
        values[key] = float(np.float32(values[key]))
    if (not np.isfinite([values[k] for k in ("dt", "G", "softening")]).all()
            or values["dt"] <= 0 or values["G"] < 0 or values["softening"] < 0
            or values["num_steps"] < 0 or values["record_interval"] < 1
            or values["integrator"] not in ("leapfrog", "euler")):
        raise ValueError("invalid simulation parameters")
    return values


def force64(targets, sources, masses, gravitational_constant, softening, ids):
    """Direct CPU sum, float64 sqrt and accumulation, explicitly exclude self."""
    result = np.empty_like(targets, dtype=np.float64)
    for start in range(0, len(targets), 16):
        stop = min(start + 16, len(targets))
        delta = sources[None] - targets[start:stop, None]
        r2 = np.einsum("ijk,ijk->ij", delta, delta) + softening**2
        r2[np.arange(stop - start), ids[start:stop]] = np.inf
        scale = gravitational_constant * masses[None] / (r2 * np.sqrt(r2))
        result[start:stop] = np.einsum("ij,ijk->ik", scale, delta)
    return result


def replay(initial, params, observed=None, ids=None):
    """Yield reference positions at every recorded step, without CUDA/PyTorch."""
    if ids is None:
        ids = np.arange(len(initial))
    masses = initial[:, 6]
    position, velocity = initial[ids, :3].copy(), initial[ids, 3:6].copy()
    dt, g, eps = (params[k] for k in ("dt", "G", "softening"))
    if observed is not None and params["record_interval"] != 1:
        raise ValueError("sampled replay requires record_interval=1; intermediate source fields are missing")
    sources = initial[:, :3]
    acceleration = force64(position, sources, masses, g, eps, ids)
    yield 0, position.copy()
    for step in range(1, params["num_steps"] + 1):
        if params["integrator"] == "leapfrog":
            position += dt * velocity + 0.5 * dt**2 * acceleration
            sources = np.asarray(observed[step], dtype=np.float64) if observed is not None else position
            following = force64(position, sources, masses, g, eps, ids)
            velocity += 0.5 * dt * (acceleration + following)
            acceleration = following
        else:
            velocity += dt * acceleration
            position += dt * velocity
            sources = np.asarray(observed[step], dtype=np.float64) if observed is not None else position
            acceleration = force64(position, sources, masses, g, eps, ids)
        if step % params["record_interval"] == 0:
            yield step // params["record_interval"], position.copy()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def energy64(position, velocity, masses, params):
    energy = 0.5 * np.sum(masses[:, None] * velocity**2)
    for i in range(len(masses) - 1):
        delta = position[i + 1:] - position[i]
        distance = np.sqrt(np.einsum("ij,ij->i", delta, delta) + params["softening"]**2)
        energy -= params["G"] * masses[i] * np.sum(masses[i + 1:] / distance)
    return float(energy)


def audit(trajectory_path, particles_path, params_path, sample_count=16,
          reference_limit=512, atol=2e-5, motion_rtol=2e-3):
    begin = time.perf_counter()
    params = read_params(params_path)
    initial = np.loadtxt(particles_path, dtype=np.float32, ndmin=2).astype(np.float64)
    if initial.shape[1] != 7 or not np.isfinite(initial).all() or np.any(initial[:, 6] <= 0):
        raise ValueError("initial particles must be finite x,y,z,vx,vy,vz,positive_mass")
    trajectory = load_trajectory(trajectory_path)
    n = len(initial)
    expected = (params["num_steps"] // params["record_interval"] + 1, n, 3)
    checks = []

    def check(name, passed, **details):
        checks.append(dict(name=name, passed=bool(passed), **details))

    report = {"schema": "nbody-validation-v1", "trajectory": str(trajectory_path),
              "trajectory_sha256": file_sha256(trajectory_path),
              "particles_sha256": file_sha256(particles_path),
              "params_sha256": file_sha256(params_path), "params": params,
              "particles": n, "records": len(trajectory), "checks": checks}
    check("shape", trajectory.shape == expected, expected=list(expected), actual=list(trajectory.shape))
    if trajectory.shape != expected:
        report.update(status="FAIL", scope="file validation", elapsed_sec=time.perf_counter() - begin)
        trajectory._mmap.close()
        return report
    check("initial_coordinates", np.array_equal(trajectory[0], initial[:, :3]))
    finite = True
    masses = initial[:, 6]
    center0 = np.average(initial[:, :3], weights=masses, axis=0)
    center_velocity = np.average(initial[:, 3:6], weights=masses, axis=0)
    max_com_error = 0.0
    max_displacement = np.zeros(n)
    static_records = 0
    previous = None
    for start in range(0, len(trajectory), 32):
        block = np.array(trajectory[start:start + 32], dtype=np.float64, copy=True)
        finite &= bool(np.isfinite(block).all())
        if not finite:
            break
        max_displacement = np.maximum(max_displacement, np.linalg.norm(block - initial[None, :, :3], axis=-1).max(axis=0))
        centers = np.einsum("fij,i->fj", block, masses) / masses.sum()
        times = np.arange(start, start + len(block)) * params["record_interval"] * params["dt"]
        max_com_error = max(max_com_error, float(np.linalg.norm(centers - center0 - times[:, None] * center_velocity, axis=1).max()))
        if previous is not None:
            static_records += int(np.array_equal(block[0], previous))
        static_records += int(np.all(np.diff(block, axis=0) == 0, axis=(1, 2)).sum())
        previous = block[-1].copy()
    check("finite_coordinates_all_records", finite)
    if not finite:
        report.update(status="FAIL", scope="file validation", elapsed_sec=time.perf_counter() - begin)
        trajectory._mmap.close()
        return report
    report["motion"] = {"max_displacement": float(max_displacement.max()),
                         "median_max_displacement": float(np.median(max_displacement)),
                         "fully_static_transitions": static_records,
                         "fastest_particle_id": int(np.argmax(max_displacement))}
    # COM in physical units, allowing float32 storage quantization.
    scale = max(float(np.linalg.norm(initial[:, :3] - center0, axis=1).max()), 1.0)
    com_tolerance = max(atol, scale * 2e-6)
    check("center_of_mass_linear_motion", max_com_error <= com_tolerance,
          max_error=max_com_error, tolerance=com_tolerance)
    full = n <= reference_limit
    if full:
        ids = np.arange(n)
    else:
        # Include the heaviest and fastest body, then a deterministic random sample.
        chosen = [int(np.argmax(masses)), int(np.argmax(max_displacement))]
        chosen += np.random.default_rng(73).permutation(n).tolist()
        ids = np.array(list(dict.fromkeys(chosen))[:min(sample_count, n)])
    report.update(scope="independent all-body CPU replay" if full else
                  "sampled CPU replay conditioned on recorded source positions",
                  reference_particle_ids=ids.tolist(), atol=atol, motion_rtol=motion_rtol,
                  limitations=[] if full else [
                      "Unselected particle trajectories are not independently replayed.",
                      "Recorded source fields may share errors; this is not a full all-body correctness proof.",
                      "Global exact energy is not checked at this scale."])
    if not full and params["record_interval"] != 1:
        check("sampled_replay_available", False, reason="requires record_interval=1")
    else:
        worst_ratio, worst_error, worst_record, worst_particle = 0.0, 0.0, 0, int(ids[0])
        for frame, reference in replay(initial, params, None if full else trajectory, ids):
            if not np.isfinite(reference).all():
                check("cpu_reference_finite", False, record=frame)
                break
            observed = np.asarray(trajectory[frame, ids], dtype=np.float64)
            # Relative to motion, not huge absolute world coordinates. Allow
            # 4 float32 storage ULPs; report tolerance explicitly for review.
            ulp = np.linalg.norm(np.abs(np.spacing(reference.astype(np.float32))).astype(np.float64), axis=1)
            motion = np.linalg.norm(reference - initial[ids, :3], axis=1)
            tolerance = atol + motion_rtol * motion + 4 * ulp
            error = np.linalg.norm(observed - reference, axis=1)
            ratios = error / tolerance
            at = int(np.argmax(ratios))
            if ratios[at] > worst_ratio:
                worst_ratio, worst_record, worst_particle = float(ratios[at]), frame, int(ids[at])
            worst_error = max(worst_error, float(error.max()))
            if frame % 200 == 0:
                print(f"Reference {frame + 1}/{len(trajectory)} | checked bodies {len(ids)}/{n}", flush=True)
        check("cpu_trajectory_replay", worst_ratio <= 1, max_error=worst_error,
              worst_error_to_tolerance_ratio=worst_ratio, worst_record=worst_record,
              worst_particle_id=worst_particle)
    if full and len(trajectory) >= 3:
        # Velocities are not in BIN. Reconstruct at interior records and state
        # explicitly that this diagnostic has O(record_dt^2) truncation error.
        energies = []
        record_dt = params["dt"] * params["record_interval"]
        sample_frames = np.unique(np.linspace(1, len(trajectory) - 2, min(25, len(trajectory) - 2), dtype=int))
        for frame in sample_frames:
            velocity = (trajectory[frame + 1].astype(np.float64) - trajectory[frame - 1]) / (2 * record_dt)
            energies.append(energy64(trajectory[frame].astype(np.float64), velocity, masses, params))
        denominator = max(abs(energies[0]), 1e-30)
        report["energy_diagnostic"] = {"method": "exact pairs; velocities reconstructed by centered differences",
                                       "relative_span": float(np.ptp(energies) / denominator),
                                       "frames": sample_frames.tolist(),
                                       "limitation": "O(record_dt^2) velocity error; reported, not an automatic conservation pass"}
    passed = all(item["passed"] for item in checks)
    report.update(status=("PASS" if full else "PASS_SAMPLED") if passed else "FAIL",
                  elapsed_sec=time.perf_counter() - begin)
    trajectory._mmap.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory")
    parser.add_argument("--particles", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--sample-particles", type=int, default=16)
    parser.add_argument("--reference-limit", type=int, default=512)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--motion-rtol", type=float, default=2e-3)
    args = parser.parse_args()
    if args.sample_particles < 1 or args.reference_limit < 1 or not np.isfinite([args.atol, args.motion_rtol]).all() or min(args.atol, args.motion_rtol) <= 0:
        parser.error("sample counts and tolerances must be finite and positive")
    try:
        report = audit(args.trajectory, args.particles, args.params, args.sample_particles,
                       args.reference_limit, args.atol, args.motion_rtol)
    except (ValueError, OSError) as exc:
        report = {"schema": "nbody-validation-v1", "status": "FAIL", "error": str(exc)}
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(report["status"], path)
    raise SystemExit(1 if report["status"] == "FAIL" else 0)


if __name__ == "__main__":
    main()

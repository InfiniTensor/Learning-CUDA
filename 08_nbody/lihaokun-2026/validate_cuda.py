"""Run the real CUDA executable against analytic and independent CPU references."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from verify_trajectory import audit, file_sha256
from visualize import load_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", default="./nbody_a800")
    parser.add_argument("--output-dir", type=Path, default=Path("results/cuda_validation"))
    args = parser.parse_args()
    executable = str(Path(args.executable).resolve())
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    reports = []

    def run(name, bodies, dt, steps, g=1, eps=.02, integrator="leapfrog"):
        particle_file, parameter_file = root / f"{name}_particles.txt", root / f"{name}_params.txt"
        np.savetxt(particle_file, bodies, fmt="%.12g")
        parameter_file.write_text(f"dt = {dt}\nnum_steps = {steps}\nrecord_interval = 1\nG = {g}\nsoftening = {eps}\nintegrator = {integrator}\n")
        trajectory_file = root / f"{name}.bin"
        subprocess.run([executable, str(particle_file), str(parameter_file), str(trajectory_file),
                        str(root / f"{name}.performance.log")], check=True)
        report = audit(trajectory_file, particle_file, parameter_file, atol=2e-5, motion_rtol=5e-4)
        report["case"] = name
        reports.append(report)
        return trajectory_file, report

    # A float32 position accumulator loses these sub-ULP steps entirely.
    ballistic = np.array([[50000, -30000, 1000, .001, -.002, .003, 1],
                          [-40000, 20000, -700, -.004, .001, -.002, 2]])
    for integrator in ("leapfrog", "euler"):
        path, report = run(f"ballistic_{integrator}", ballistic, .01, 1000, g=0, integrator=integrator)
        data = load_trajectory(path)
        first = ballistic.astype(np.float32).astype(np.float64)
        expected = first[None, :, :3] + np.arange(1001)[:, None, None] * float(np.float32(.01)) * first[None, :, 3:6]
        # Binary output should agree with analytic positions to <= 2 storage ULPs.
        tolerance = 2 * np.abs(np.spacing(expected.astype(np.float32))).astype(np.float64) + 1e-7
        passed = bool(np.all(np.abs(data.astype(np.float64) - expected) <= tolerance))
        report["checks"].append({"name": "analytic_ballistic_sub_ulp", "passed": passed})
        data._mmap.close()

    eps = .02
    omega = np.sqrt(2 / (1 + eps**2)**1.5)
    circular = np.array([[-.5, 0, 0, 0, -.5 * omega, 0, 1],
                         [.5, 0, 0, 0, .5 * omega, 0, 1]])
    errors = []
    for label, dt, steps in (("coarse", .01, 1000), ("fine", .005, 2000)):
        path, report = run(f"circular_{label}", circular, dt, steps, eps=eps)
        data = load_trajectory(path)
        times = np.arange(steps + 1) * float(np.float32(dt))
        analytic = np.column_stack((-.5 * np.cos(omega * times), -.5 * np.sin(omega * times), times * 0))
        error = float(np.linalg.norm(data[:, 0] - analytic, axis=1).max())
        radius_error = float(np.max(np.abs(np.linalg.norm(data[:, 0], axis=1) - .5)))
        report["checks"].append({"name": "softened_circular_analytic", "passed": error < .002 and radius_error < .001,
                                  "max_position_error": error, "max_radius_error": radius_error})
        span = report["energy_diagnostic"]["relative_span"]
        report["checks"].append({"name": "circular_energy_from_dense_position_records",
                                  "passed": span < 1e-3, "relative_span": span,
                                  "limitation": "velocities reconstructed with centered differences"})
        errors.append(error)
        data._mmap.close()
    reports[-1]["checks"].append({"name": "second_order_convergence", "passed": errors[0] / max(errors[1], 1e-30) >= 2.5,
                                  "coarse_over_fine_error": errors[0] / max(errors[1], 1e-30)})

    # Unequal masses + 3D + partially occupied final shared-memory tile (257).
    for count, integrator in ((17, "euler"), (257, "leapfrog")):
        rng = np.random.default_rng(112 + count)
        bodies = np.column_stack((rng.normal(size=(count, 3)), rng.normal(0, .08, (count, 3)),
                                  rng.uniform(.5, 1.5, count) / count))
        run(f"random_{count}_{integrator}", bodies, .001, 100, eps=.1, integrator=integrator)

    for report in reports:
        report["status"] = "PASS" if all(c["passed"] for c in report["checks"]) else "FAIL"
        (root / f"{report['case']}.validation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        print(report["case"], report["status"])
    passed = all(r["status"] == "PASS" for r in reports)
    summary = {"status": "PASS" if passed else "FAIL", "executable": executable,
               "executable_sha256": file_sha256(executable),
               "cases": [{"case": r["case"], "status": r["status"]} for r in reports],
               "scope": "small-system analytic, convergence, ballistic and independent all-body CPU tests; not proof for arbitrary N"}
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()

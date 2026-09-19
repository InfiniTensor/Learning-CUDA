#!/usr/bin/env python3
"""Physics, independent FP64 oracle, boundary sizes and malformed-input checks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.generate import generate
from scripts.trajectory import load_trajectory, close_pairs


def reference(data, dt, steps, softening, integrator, G=1):
    # First match parsed FP32 inputs; perform all reference evolution in FP64.
    s = data.astype(np.float32).astype(np.float64).copy()
    dt, softening, G = map(lambda v: float(np.float32(v)), (dt, softening, G))
    def acceleration():
        displacement = s[None, :, :3] - s[:, None, :3]
        d2 = np.sum(displacement**2, axis=2) + softening**2
        np.fill_diagonal(d2, np.inf)
        return G * np.sum(displacement * (s[None, :, 6] * d2**(-1.5))[:, :, None], axis=1)
    a = acceleration()
    for _ in range(steps):
        if integrator == "leapfrog":
            s[:, 3:6] += dt * 0.5 * a
            s[:, :3] += dt * s[:, 3:6]
            a = acceleration()
            s[:, 3:6] += dt * 0.5 * a
        else:
            a = acceleration()
            s[:, :3] += dt * s[:, 3:6]
            s[:, 3:6] += dt * a
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    exe, output = args.exe.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    results = []

    def run(name, data, steps=12, dt=0.001, eps=0.03, integrator="leapfrog", backend="cuda", extra=(), interval=5, G=1):
        directory = output / name; directory.mkdir(exist_ok=True)
        np.savetxt(directory / "input.txt", data, fmt="%.9g")
        (directory / "config.txt").write_text(f"dt={dt}\nnum_steps={steps}\nrecord_interval={interval}\nG={G}\nsoftening={eps}\nintegrator=\"{integrator}\"\n")
        cmd = [str(exe), "--input", str(directory / "input.txt"), "--config", str(directory / "config.txt"), "--output", str(directory / "trajectory.bin"), "--log", str(directory / "metrics.json"), "--final", str(directory / "final.txt"), "--backend", backend, *extra]
        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode:
            raise AssertionError(f"{name}: {completed.stderr}")
        final = np.loadtxt(directory / "final.txt", ndmin=2)
        points, times, _ = load_trajectory(directory / "trajectory.bin")
        expected_steps = list(range(0, steps + 1, interval))
        if expected_steps[-1] != steps: expected_steps.append(steps)
        assert points.shape == (len(data), len(expected_steps), 3), name
        np.testing.assert_allclose(times, np.array(expected_steps) * np.float32(dt), rtol=1e-7)
        np.testing.assert_allclose(points[:, 0], data.astype(np.float32)[:, :3], atol=1e-7)
        np.testing.assert_allclose(points[:, -1], final[:, :3], rtol=2e-7, atol=1e-7)
        metrics = json.loads((directory / "metrics.json").read_text())
        return final, points, metrics, cmd

    rng = np.random.default_rng(77)
    for n in (1, 2, 31, 127, 128, 129, 255, 256, 257):
        data = np.column_stack((rng.normal(size=(n, 3)), rng.normal(0, 0.1, (n, 3)), rng.uniform(0.5, 1.5, n) / n))
        for integrator in ("euler", "leapfrog"):
            name = f"reference_{n}_{integrator}"
            final, _, _, _ = run(name, data, integrator=integrator)
            ref = reference(data, 0.001, 12, 0.03, integrator)
            error = float(np.max(np.abs(final[:, :6] - ref[:, :6])))
            np.testing.assert_allclose(final[:, :6], ref[:, :6], rtol=3e-5, atol=3e-6, err_msg=name)
            results.append({"case": name, "max_abs_error": error})
    for backend, extra in (("cpu", ()), ("cpu", ("--cpu-threads", "2")), ("cuda", ("--kernel", "naive")), ("cuda", ("--block-size", "256"))):
        name = f"variant_{backend}_{'_'.join(extra)}"
        final, _, _, _ = run(name, data, backend=backend, extra=extra)
        np.testing.assert_allclose(final[:, :6], reference(data, 0.001, 12, 0.03, "leapfrog")[:, :6], rtol=3e-5, atol=3e-6)
        results.append({"case": name, "passed": True})
    final, _, _, cmd = run("zero_steps", data[:3], steps=0)
    np.testing.assert_allclose(final, data[:3].astype(np.float32), rtol=1e-8, atol=1e-8)
    ballistic = np.array([[0, 0, 0, 1, -2, 3, 2.]])
    final, _, _, _ = run("ballistic", ballistic, steps=100, dt=0.01, G=0)
    np.testing.assert_allclose(final[0, :3], [1, -2, 3], atol=4e-6)
    binary = generate("binary", softening=0.01)
    _, points, metrics, _ = run("circular_orbit", binary, steps=7000, dt=0.002, eps=0.01, interval=20)
    separation = np.linalg.norm(points[0] - points[1], axis=1)
    radius_error = float(np.max(np.abs(separation - 1)))
    assert radius_error < 2e-4, radius_error
    assert metrics["diagnostics"]["relative_energy_error"] < 3e-5, metrics
    results.append({"case": "circular_orbit_3_periods", "radius_error": radius_error, **metrics["diagnostics"]})
    # Step halving on a non-circular orbit should show second-order convergence.
    ellipse = generate("ellipse", softening=0.01)
    oracle = reference(ellipse, 0.000125, 8000, 0.01, "leapfrog")
    errors = []
    for dt, steps in ((0.02, 50), (0.01, 100)):
        final, _, _, _ = run(f"ellipse_{steps}", ellipse, dt=dt, steps=steps, eps=0.01)
        errors.append(float(np.linalg.norm(final[:, :3] - oracle[:, :3])))
    assert 3 < errors[0] / errors[1] < 5.2, errors
    results.append({"case": "leapfrog_second_order", "position_errors": errors, "ratio": errors[0] / errors[1]})
    cluster = generate("cluster", 256)
    _, _, metrics, _ = run("cluster_conservation", cluster, steps=1000)
    assert metrics["diagnostics"]["relative_energy_error"] < 5e-4, metrics
    assert metrics["diagnostics"]["normalized_momentum_error"] < 1e-5, metrics
    results.append({"case": "cluster_conservation", **metrics["diagnostics"]})
    coincident = np.array([[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 2.]])
    final, _, _, _ = run("coincident_softening", coincident)
    assert np.isfinite(final).all() and np.all(final[:, :6] == 0)
    _, _, _, comparison_cmd = run("comparison", cluster[:31])
    comparison = subprocess.run(comparison_cmd + ["--no-trajectory", "--compare-cpu"], capture_output=True, text=True)
    assert comparison.returncode == 0, comparison.stderr
    comparison_metrics = json.loads(comparison.stdout)
    assert comparison_metrics["cpu_comparison"]["max_position_abs_error"] < 1e-5
    assert comparison_metrics["trajectory_records"] == 0
    alias_cmd = comparison_cmd.copy()
    alias_cmd[alias_cmd.index("--output") + 1] = alias_cmd[alias_cmd.index("--input") + 1]
    original = Path(alias_cmd[alias_cmd.index("--input") + 1]).read_bytes()
    assert subprocess.run(alias_cmd, capture_output=True).returncode != 0
    assert Path(alias_cmd[alias_cmd.index("--input") + 1]).read_bytes() == original
    results.append({"case": "cpu_comparison_and_output_alias_protection", "passed": True})
    for bad in ("dt=nan", "dt=0", "softening=0", "softening=1e-30", "record_interval=0", "num_steps=-1", "num_steps=1.5", "integrator=invalid", "typo=1", "dt=0.1\ndt=0.2"):
        Path(cmd[cmd.index("--config") + 1]).write_text(bad)
        failed = subprocess.run(cmd, capture_output=True, text=True)
        assert failed.returncode != 0 and "error:" in failed.stderr, bad
    Path(cmd[cmd.index("--config") + 1]).write_text("num_steps=0")
    for bad in ("", "0 0 0 0 0 0 -1", "nan 0 0 0 0 0 1", "0 0 0 0 0 0 1 extra", "0 0 0"):
        Path(cmd[cmd.index("--input") + 1]).write_text(bad)
        assert subprocess.run(cmd, capture_output=True).returncode != 0, bad
    # Compare the spatial hash to a brute-force count, including negative cells.
    coords = rng.normal(size=(80, 3))
    marked, count = close_pairs(coords, 0.5)
    distance = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
    mask = (distance < 0.5) & ~np.eye(len(coords), dtype=bool)
    assert count == int(mask.sum() // 2) and np.array_equal(marked, mask.any(axis=1))
    (output / "truncated.bin").write_bytes(np.array([2, 2], dtype="<i4").tobytes())
    try:
        load_trajectory(output / "truncated.bin")
        raise AssertionError("Accepted truncated trajectory")
    except ValueError:
        pass
    results.append({"case": "zero_step_ballistic_overlap_invalid_inputs_binary_reader_proximity", "passed": True})
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"PASS: {len(results)} checks/groups (including 15 invalid-input cases)")


if __name__ == "__main__":
    main()

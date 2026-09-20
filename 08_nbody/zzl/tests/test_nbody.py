"""Executable-level numerical, parsing, binary-layout and visualization acceptance tests."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generate import make_particles, write_config
from trajectory import close_pairs, load_trajectory

EXE = None
CUDA = False
MEASUREMENTS = {}


class Acceptance(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="nbody-test-")
        self.root = Path(self.temp.name)
        self.counter = 0

    def tearDown(self):
        # Matplotlib has callback cycles; release their memmaps before Windows removes fixtures.
        gc.collect()
        self.assertEqual(self.root.resolve().parent, Path(tempfile.gettempdir()).resolve())
        self.assertTrue(self.root.name.startswith("nbody-test-"))
        self.temp.cleanup()

    def run_case(self, particles, backend="cpu", steps=10, dt=0.001, interval=3,
                 softening=0.02, G=1, integrator="leapfrog", extra=(), raw_config=None,
                 expect_success=True, raw_input=None):
        self.counter += 1
        directory = self.root / str(self.counter)
        directory.mkdir()
        input_path = directory / "particles.txt"
        if raw_input is not None:
            input_path.write_text(raw_input, encoding="utf-8")
        else:
            np.savetxt(input_path, particles, fmt="%.12g")
        config_path = directory / "config.txt"
        if raw_config is not None:
            config_path.write_text(raw_config, encoding="utf-8")
        else:
            write_config(config_path, dt, steps, interval, softening, G, integrator)
        command = [str(EXE), "--input", str(input_path), "--config", str(config_path),
                   "--backend", backend, "--output", str(directory / "trajectory.bin"),
                   "--log", str(directory / "performance.json"), "--final-state", str(directory / "state.csv"),
                   "--cpu-threads", "2", *extra]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if not expect_success:
            self.assertNotEqual(result.returncode, 0, result.stdout)
            return result
        self.assertEqual(result.returncode, 0, result.stderr)
        state = np.loadtxt(directory / "state.csv", delimiter=",", skiprows=1, ndmin=2)[:, 1:]
        log = json.loads((directory / "performance.json").read_text())
        self.assertTrue(np.isfinite(state).all())
        return state, log, directory, command

    def test_binary_layout_initial_and_nondivisible_final(self):
        data = np.array([[1, 2, 3, 0.1, 0.2, 0.3, 1], [-2, 4, -1, -0.2, 0, 0.1, 2]])
        _, log, directory, _ = self.run_case(data, G=0, steps=10, dt=0.1, interval=3)
        for key in ("trajectory_spool_seconds", "trajectory_finalize_seconds",
                    "final_state_write_seconds"):
            self.assertIn(key, log)
            self.assertGreaterEqual(log[key], 0)
        self.assertGreaterEqual(
            log["final_output_seconds"],
            log["trajectory_finalize_seconds"] + log["final_state_write_seconds"])
        positions, meta = load_trajectory(directory / "trajectory.bin")
        self.assertEqual(meta["steps"], [0, 3, 6, 9, 10])
        expected = data[:, None, :3] + data[:, None, 3:6] * (np.array(meta["steps"])[None, :, None] * 0.1)
        np.testing.assert_allclose(positions, expected, atol=2e-6)
        payload = (directory / "trajectory.bin").read_bytes()
        self.assertEqual(struct.unpack("<ii", payload[:8]), (2, 5))
        self.assertEqual(len(payload), 8 + 2 * 5 * 12)
        np.testing.assert_allclose(struct.unpack("<fff", payload[8 + 5 * 12:8 + 6 * 12]), data[1, :3])
        del positions
        self.assertFalse(list(directory.glob("*.tmp")))

    def test_trajectory_transpose_across_buffer_boundary(self):
        # 30000 x 101 frames exceeds 32 MiB; exercise both particle chunks.
        data = np.zeros((30000, 7))
        data[:, 0] = np.arange(30000) * 0.125
        data[:, 4] = (np.arange(30000) % 8) * 0.125
        data[:, 5] = -(np.arange(30000) % 4) * 0.25
        data[:, 6] = 1
        _, _, directory, _ = self.run_case(data, G=0, steps=100, dt=0.25, interval=1,
                                           extra=("--diagnostics", "momentum"))
        positions, meta = load_trajectory(directory / "trajectory.bin")
        np.testing.assert_allclose(positions[:, :, 0],
                                   np.broadcast_to(data[:, None, 0], (30000, 101)), atol=1e-5)
        for axis in (1, 2):
            expected = data[:, None, axis+3] * np.array(meta["steps"])[None, :] * 0.25
            np.testing.assert_array_equal(positions[:, :, axis], expected)
        del positions

    def test_trajectory_varied_record_counts(self):
        # Distinct exactly representable values expose particle/frame/xyz swaps.
        ids = np.arange(19)
        data = np.column_stack((ids, -ids, ids * 2, (ids % 3) / 8,
                                -(ids % 5) / 4, (ids % 7) / 2, np.ones(19)))
        for records in (31, 32, 33, 64, 65):
            with self.subTest(records=records):
                _, _, directory, _ = self.run_case(
                    data, G=0, steps=records-1, dt=0.25, interval=1)
                positions, meta = load_trajectory(directory / "trajectory.bin")
                times = np.arange(records) * 0.25
                expected = data[:, None, :3] + times[None, :, None] * data[:, None, 3:6]
                np.testing.assert_array_equal(positions, expected)
                self.assertEqual(meta["steps"], list(range(records)))
                del positions

    def test_zero_steps_and_single_particle_ballistic(self):
        data = np.array([[2, 3, 4, -0.2, 0.1, 0.3, 1]])
        for backend in (["cpu", "cuda"] if CUDA else ["cpu"]):
            with self.subTest(backend=backend):
                state, log, directory, _ = self.run_case(data, backend, steps=0)
                np.testing.assert_allclose(state, data, atol=1e-7)
                self.assertEqual(log["records"], 1)
                state, _, _, _ = self.run_case(data, backend, steps=40, dt=0.01)
                np.testing.assert_allclose(state[:, :3], data[:, :3] + 0.4 * data[:, 3:6], atol=1e-5)

    def test_zero_gravity_extreme_finite_coordinates(self):
        data = np.array([[3e38, 0, 0, 0, 1, 0, 1],
                         [-3e38, 0, 0, 0, -1, 0, 1]])
        for backend in (["cpu", "cuda"] if CUDA else ["cpu"]):
            for integrator in ("euler", "leapfrog"):
                for kernel in ("naive", "tiled") if backend == "cuda" else ("tiled",):
                    with self.subTest(backend=backend, integrator=integrator, kernel=kernel):
                        state, log, _, _ = self.run_case(
                            data, backend, steps=4, dt=0.25, G=0,
                            integrator=integrator, extra=("--kernel", kernel))
                        expected = data.copy()
                        expected[:, 1] = data[:, 4]
                        np.testing.assert_allclose(state, expected, rtol=1e-7, atol=1e-7)
                        self.assertTrue(log["energy_computed"])
                        self.assertAlmostEqual(log["initial_energy"], 1.0, places=12)
                        self.assertAlmostEqual(log["final_energy"], 1.0, places=12)
                        self.assertEqual(log["relative_energy_error"], 0)

    def test_graph_execution_matches_direct_with_output_boundaries(self):
        if not CUDA:
            self.skipTest("CUDA device unavailable")
        data = make_particles("cluster", 257, seed=81)
        for integrator in ("euler", "leapfrog"):
            for kernel in ("naive", "tiled"):
                for steps, interval, record in ((0, 1, True), (1, 1, True),
                                                (65, 7, True), (67, 100, True), (200, 100, True),
                                                (65, 7, False)):
                    with self.subTest(integrator=integrator, kernel=kernel,
                                      steps=steps, interval=interval, record=record):
                        extra = ("--kernel", kernel) + (() if record else ("--no-trajectory",))
                        direct, _, direct_dir, _ = self.run_case(
                            data, "cuda", steps=steps, interval=interval, integrator=integrator,
                            extra=extra + ("--cuda-execution", "direct"))
                        actual, log, graph_dir, _ = self.run_case(
                            data, "cuda", steps=steps, interval=interval, integrator=integrator,
                            extra=extra + ("--cuda-execution", "graph"))
                        np.testing.assert_array_equal(actual, direct)
                        if record:
                            self.assertEqual((graph_dir / "trajectory.bin").read_bytes(),
                                             (direct_dir / "trajectory.bin").read_bytes())
                        self.assertEqual(log["graph_launches"] * log["graph_batch_steps"]
                                         + log["direct_steps"], steps)
                        self.assertEqual(log["cuda_execution"], "graph" if steps else "direct")
                        self.assertGreaterEqual(log["simulation_host_seconds"], 0)
                        self.assertLessEqual(log["simulation_host_seconds"],
                                             log["backend_wall_seconds"] + 1e-9)
                        if steps:
                            self.assertGreater(log["graph_setup_seconds"], 0)

    def test_graph_auto_policy_boundaries(self):
        if not CUDA:
            self.skipTest("CUDA device unavailable")
        for n, steps, interval, expected in ((256, 128, 4, "graph"),
                                              (257, 128, 4, "direct"),
                                              (256, 127, 4, "direct"),
                                              (256, 128, 1, "direct")):
            with self.subTest(n=n, steps=steps, interval=interval):
                _, log, _, _ = self.run_case(make_particles("cluster", n), "cuda",
                                              steps=steps, interval=interval)
                self.assertEqual(log["cuda_execution"], expected)

    def test_invalid_execution_mode(self):
        self.run_case(make_particles("binary"), expect_success=False,
                      extra=("--cuda-execution", "graph"))
        if CUDA:
            self.run_case(make_particles("binary"), "cuda", expect_success=False,
                          extra=("--cuda-execution", "invalid"))

    def test_recording_does_not_change_final_state(self):
        data = make_particles("cluster", 17, seed=52)
        for backend in (["cpu", "cuda"] if CUDA else ["cpu"]):
            for integrator in ("euler", "leapfrog"):
                kernels = ("naive", "tiled") if backend == "cuda" else ("tiled",)
                for kernel in kernels:
                    with self.subTest(backend=backend, integrator=integrator, kernel=kernel):
                        recorded, _, _, _ = self.run_case(
                            data, backend, steps=11, integrator=integrator,
                            extra=("--kernel", kernel))
                        unrecorded, log, directory, _ = self.run_case(
                            data, backend, steps=11, integrator=integrator,
                            extra=("--kernel", kernel, "--no-trajectory"))
                        np.testing.assert_array_equal(unrecorded, recorded)
                        self.assertEqual(log["records"], 0)
                        self.assertFalse((directory / "trajectory.bin").exists())

    def test_softened_force_and_explicit_euler_against_numpy(self):
        data = make_particles("cluster", 17, seed=6)
        eps, dt, gravity = 0.13, 0.017, 1.3
        delta = data[None, :, :3] - data[:, None, :3]
        r2 = np.sum(delta**2, axis=2) + eps**2
        accel = np.sum(gravity * delta * data[None, :, 6, None] / r2[:, :, None]**1.5, axis=1)
        for backend in (["cpu", "cuda"] if CUDA else ["cpu"]):
            with self.subTest(backend=backend):
                state, _, _, _ = self.run_case(data, backend, steps=1, dt=dt, softening=eps, G=gravity, integrator="euler")
                np.testing.assert_allclose(state[:, :3], data[:, :3] + dt * data[:, 3:6], atol=2e-7)
                np.testing.assert_allclose(state[:, 3:6], data[:, 3:6] + dt * accel, atol=3e-7)

    def test_circular_orbit_long_term(self):
        eps, dt, steps = 0.001, 0.003, 20000
        data = make_particles("binary", softening=eps)
        for backend in (["cpu", "cuda"] if CUDA else ["cpu"]):
            state, log, directory, _ = self.run_case(data, backend, steps=steps, dt=dt, interval=100, softening=eps)
            history, _ = load_trajectory(directory / "trajectory.bin")
            separation = np.linalg.norm(history[1] - history[0], axis=1)
            radial_error = float(np.max(np.abs(separation - 1)))
            self.assertLess(radial_error, 4e-4)
            self.assertLess(log["relative_energy_error"], 8e-5)
            self.assertLess(log["absolute_momentum_error"], 1e-7)
            MEASUREMENTS[f"circular_{backend}"] = {"steps": steps, "dt": dt, "max_separation_error": radial_error,
                                                     "relative_energy_error": log["relative_energy_error"]}
            del history

    def test_elliptical_bound_orbit(self):
        data = make_particles("ellipse", softening=0.001)
        backend = "cuda" if CUDA else "cpu"
        _, log, directory, _ = self.run_case(data, backend, steps=10000, dt=0.001, interval=25, softening=0.001)
        positions, _ = load_trajectory(directory / "trajectory.bin")
        radii = np.linalg.norm(positions[1] - positions[0], axis=1)
        self.assertLess(float(radii.min()), 0.45)
        self.assertGreater(float(radii.min()), 0.3)
        self.assertLess(float(radii.max()), 1.002)
        self.assertLess(log["relative_energy_error"], 1e-4)
        MEASUREMENTS["ellipse"] = {"min_separation": float(radii.min()), "max_separation": float(radii.max()),
                                     "relative_energy_error": log["relative_energy_error"]}
        del positions

    def test_leapfrog_second_order_convergence(self):
        eps = 0.01
        data = make_particles("binary", softening=eps)
        omega = math.sqrt(2 / (1 + eps**2)**1.5)
        rotation = np.array([[math.cos(omega), -math.sin(omega)], [math.sin(omega), math.cos(omega)]])
        exact = data[:, :2] @ rotation.T
        errors = []
        for dt, steps in [(0.04, 25), (0.02, 50)]:
            state, _, _, _ = self.run_case(data, steps=steps, dt=dt, softening=eps)
            errors.append(float(np.linalg.norm(state[:, :2] - exact)))
        self.assertLess(errors[1] / errors[0], 0.3)
        self.assertGreater(errors[1] / errors[0], 0.2)
        MEASUREMENTS["second_order"] = {"coarse_error": errors[0], "fine_error": errors[1], "ratio": errors[1] / errors[0]}

    def test_multibody_conservation(self):
        data = make_particles("cluster", 256, seed=123)
        _, log, _, _ = self.run_case(data, "cuda" if CUDA else "cpu", steps=2000, dt=0.0005, interval=200)
        self.assertLess(log["relative_energy_error"], 5e-4)
        self.assertLess(log["normalized_momentum_error"], 1e-5)
        self.assertLess(log["center_of_mass_ballistic_error"], 1e-5)
        MEASUREMENTS["cluster_256"] = log

    def test_cpu_cuda_three_level_convergence(self):
        # Exact circular solution of the softened two-body equations, T = 1.
        eps = 0.01
        data = make_particles("binary", softening=eps)
        omega = math.sqrt(2 / (1 + eps**2)**1.5)
        rotation = np.array([[math.cos(omega), -math.sin(omega)],
                             [math.sin(omega), math.cos(omega)]])
        exact = data[:, :2] @ rotation.T
        variants = [("cpu", ())]
        if CUDA:
            variants += [("cuda", ("--kernel", k, "--block-size", str(b)))
                         for k, b in (("naive", 256), ("tiled", 128),
                                      ("tiled", 256), ("tiled", 512))]
        results = {}
        for backend, extra in variants:
            for integrator in ("euler", "leapfrog"):
                label = f"{backend}-{extra}-{integrator}"
                with self.subTest(variant=label):
                    errors = []
                    for dt, steps in ((0.04, 25), (0.02, 50), (0.01, 100)):
                        state, _, _, _ = self.run_case(
                            data, backend, dt=dt, steps=steps, softening=eps,
                            integrator=integrator, extra=extra + ("--no-trajectory",))
                        errors.append(float(np.linalg.norm(state[:, :2] - exact)))
                    orders = [math.log2(errors[i] / errors[i+1]) for i in range(2)]
                    lower, upper = (0.8, 1.2) if integrator == "euler" else (1.8, 2.2)
                    for order in orders:
                        self.assertGreater(order, lower)
                        self.assertLess(order, upper)
                    results[label] = {"errors": errors, "observed_orders": orders}
        MEASUREMENTS["three_level_convergence"] = results

    def test_sampled_elliptic_energy_envelope(self):
        # Rerun from the SAME initial state to each checkpoint. This preserves
        # the real integrator state and needs no velocity estimates from frames.
        eps = 0.01
        data = make_particles("ellipse", softening=eps)

        def energy(state):
            mass = state[:, 6]
            kinetic = 0.5 * np.sum(mass[:, None] * state[:, 3:6]**2)
            distance2 = np.sum((state[1, :3] - state[0, :3])**2)
            return float(kinetic - mass[0]*mass[1] / math.sqrt(distance2 + eps**2))

        initial_energy = energy(data)
        results = {}
        for backend in (["cpu", "cuda"] if CUDA else ["cpu"]):
            envelopes = []
            samples = []
            for dt in (0.01, 0.005):
                drift = []
                for time in (0.5, 1.0, 1.5, 2.0, 4.0, 8.0, 12.0, 20.0):
                    state, log, _, _ = self.run_case(
                        data, backend, steps=round(time/dt), dt=dt, softening=eps,
                        extra=("--no-trajectory",))
                    value = abs((energy(state) - initial_energy) / initial_energy)
                    self.assertAlmostEqual(value, log["relative_energy_error"], delta=2e-6)
                    self.assertLess(log["normalized_momentum_error"], 1e-6)
                    drift.append(value)
                envelopes.append(max(drift))
                samples.append({"dt": dt, "relative_energy_errors": drift})
            self.assertLess(envelopes[0], 0.003)
            self.assertLess(envelopes[1], envelopes[0] * 0.4)
            results[backend] = {"sample_times": [0.5, 1, 1.5, 2, 4, 8, 12, 20],
                                "samples": samples, "maximum_sampled_errors": envelopes}
        MEASUREMENTS["sampled_elliptic_energy"] = results

    def test_cuda_tail_blocks_naive_tiled_and_integrators(self):
        if not CUDA:
            self.skipTest("CUDA device unavailable")
        worst = 0
        for n in (1, 17, 127, 128, 129, 255, 256, 257, 511, 512, 513):
            data = make_particles("cluster", max(n, 3))[:n]
            for integrator in ("euler", "leapfrog"):
                cpu, _, _, _ = self.run_case(data, steps=7, integrator=integrator)
                for kernel, block in (("tiled", 128), ("tiled", 256), ("tiled", 512), ("naive", 256)):
                    with self.subTest(n=n, kernel=kernel, block=block, integrator=integrator):
                        gpu, _, _, _ = self.run_case(data, "cuda", steps=7, integrator=integrator,
                                                      extra=("--kernel", kernel, "--block-size", str(block)))
                        np.testing.assert_allclose(gpu, cpu, atol=2e-6, rtol=3e-5)
                        worst = max(worst, float(np.abs(gpu - cpu).max()))
        MEASUREMENTS["gpu_cpu_tail_blocks"] = {"max_absolute_state_difference": worst, "gpu_cases": 88}

    def test_coincident_particles_and_massless_probe(self):
        data = np.array([[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0]])
        for backend in (["cpu", "cuda"] if CUDA else ["cpu"]):
            state, _, _, _ = self.run_case(data, backend, steps=3, dt=0.01)
            self.assertLess(state[2, 3], 0)
            np.testing.assert_array_equal(state[:2, :6], 0)

    def test_invalid_particle_inputs(self):
        for text in ("", "1 2 3\n", "0 0 0 0 0 0 -1\n", "0 0 0 0 0 0 0\n", "nan 0 0 0 0 0 1\n",
                     "1e50 0 0 0 0 0 1\n", "0 0 0 0 0 0 1 extra\n", "1x 0 0 0 0 0 1\n"):
            with self.subTest(text=text):
                self.run_case(None, raw_input=text, expect_success=False)

    def test_invalid_config_inputs(self):
        for text in ("dt=0", "num_steps=-1", "num_steps=1.5", "record_interval=0", "softening=0",
                     "softening=1e-30", "softening=1e30", "dt=nan", "G=-1", "unknown=2",
                     "integrator=rk4", "dt=0.1\ndt=0.2", "dt 0.1", "num_steps=2147483647", "dt=1x"):
            with self.subTest(text=text):
                self.run_case(make_particles("binary"), raw_config=text, expect_success=False)

    def test_bom_comments_and_quotes(self):
        state, _, _, _ = self.run_case(None, raw_input="\ufeff# header\n0 0 0 1 0 0 1 # one particle\n",
                                          raw_config="\ufeffdt = 0.01 # comment\nnum_steps=2\nintegrator='leapfrog'\n")
        self.assertAlmostEqual(state[0, 0], 0.02, places=6)

    def test_preserve_existing_outputs(self):
        _, _, directory, command = self.run_case(make_particles("binary"))
        original = (directory / "trajectory.bin").read_bytes()
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(original, (directory / "trajectory.bin").read_bytes())

    def test_colliding_output_paths(self):
        # All destinations are new; collision must be caught before any output is created.
        source = self.root / "input.txt"
        source.write_text("0 0 0 0 0 0 1\n")
        config = self.root / "run.cfg"
        write_config(config, num_steps=0)
        first = self.root / "result.csv"
        second = self.root / ("RESULT.CSV" if os.name == "nt" else "result.csv")
        command = [str(EXE), "--input", str(source), "--config", str(config), "--backend", "cpu",
                   "--no-trajectory", "--log", str(first), "--final-state", str(second)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(first.exists())

    def test_corrupt_binary_and_metadata(self):
        path = self.root / "bad.bin"
        for content in (b"", struct.pack("<ii", -1, 1), struct.pack("<ii", 3, 4), struct.pack("<ii", 1, 1) + struct.pack("<fff", math.nan, 0, 0)):
            path.write_bytes(content)
            with self.assertRaises(ValueError):
                load_trajectory(path)
        path.write_bytes(struct.pack("<ii", 1, 1) + struct.pack("<fff", 0, 0, 0))
        Path(str(path) + ".json").write_text('{"format":"wrong"}')
        with self.assertRaises(ValueError):
            load_trajectory(path)

    def test_spatial_hash_against_brute_force_and_cap(self):
        points = np.random.default_rng(91).normal(size=(100, 3))
        actual, truncated = close_pairs(points, 0.5)
        expected = {(i, j) for i in range(len(points)) for j in range(i + 1, len(points))
                    if np.linalg.norm(points[i] - points[j]) < 0.5}
        self.assertFalse(truncated)
        self.assertEqual({(a, b) for a, b, _ in actual}, expected)
        pairs, truncated = close_pairs(np.zeros((20, 3)), 0.1, max_pairs=5)
        self.assertTrue(truncated)
        self.assertEqual(len(pairs), 5)
        # Projected overlap is not a close pair when z distance is large.
        self.assertEqual(close_pairs([[0, 0, 0], [0, 0, 10]], 1)[0], [])

    def test_animation_renders_2d_3d_and_exports_gif(self):
        os.environ["MPLBACKEND"] = "Agg"
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        from PIL import Image
        from visualize import create_animation
        _, _, directory, _ = self.run_case(make_particles("binary"), steps=4, interval=1, dt=0.03)
        for dimensions in (2, 3):
            fig, anim, update = create_animation(directory / "trajectory.bin", dimensions=dimensions, risk_distance=1.1)
            self.assertIsInstance(anim, FuncAnimation)
            update(4)
            fig.canvas.draw()
            target = directory / f"preview_{dimensions}.gif"
            anim.save(target, writer=PillowWriter(fps=5), dpi=40)
            with Image.open(target) as picture:
                self.assertGreaterEqual(picture.n_frames, 4)
            plt.close(fig)


def main():
    global EXE, CUDA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    EXE = args.exe.resolve()
    probe = subprocess.run([str(EXE), "--device-info"], capture_output=True, text=True)
    CUDA = probe.returncode == 0
    if args.require_cuda and not CUDA:
        raise SystemExit("CUDA required but unavailable: " + probe.stderr)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Acceptance))
    report = {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "skipped": len(result.skipped), "cuda_available": CUDA,
              # Bind the report to the binary actually exercised, so the packaging
              # gate can reject an acceptance record from a different build.
              "executable_sha256": hashlib.sha256(EXE.read_bytes()).hexdigest(),
              "measurements": MEASUREMENTS,
              "failure_details": [str(case) + "\n" + detail for case, detail in result.failures + result.errors]}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()

"""CPU-only tests of validation failure detection, analytic motion and focus geometry."""
import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from raster_animation import focus_path
from verify_trajectory import audit, force64, replay


def write_bin(path, trajectory):
    with open(path, "wb") as output:
        output.write(struct.pack("<ii", trajectory.shape[1], len(trajectory)))
        np.asarray(trajectory.transpose(1, 0, 2), dtype="<f4").tofile(output)


class PhysicsTests(unittest.TestCase):
    def case(self, directory, bodies, trajectory, g=0, dt=.01):
        root = Path(directory)
        np.savetxt(root / "particles.txt", bodies, fmt="%.12g")
        (root / "params.txt").write_text(f"dt={dt}\nnum_steps={len(trajectory)-1}\nrecord_interval=1\nG={g}\nsoftening=.02\nintegrator=leapfrog\n")
        write_bin(root / "trajectory.bin", trajectory)
        return [root / "trajectory.bin", root / "particles.txt", root / "params.txt"]

    def test_force_matches_two_body_and_no_self_force(self):
        positions = np.array([[-.5, 0., 0.], [.5, 0., 0.]])
        force = force64(positions, positions, np.ones(2), 1, .02, np.arange(2))
        expected = 1 / (1 + .02**2)**1.5
        np.testing.assert_allclose(force, [[expected, 0, 0], [-expected, 0, 0]])

    def test_ballistic_and_corrupted_trajectory(self):
        bodies = np.array([[50000, -30000, 1000, .001, -.002, .003, 1],
                           [-40000, 20000, -700, -.004, .001, -.002, 2]], dtype=np.float32).astype(np.float64)
        analytic = bodies[None, :, :3] + np.arange(1001)[:, None, None] * float(np.float32(.01)) * bodies[None, :, 3:6]
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            paths = self.case(directory, bodies, analytic)
            report = audit(*paths)
            self.assertEqual(report["status"], "PASS")
            json.dumps(report, allow_nan=False)
            old = np.empty_like(analytic, dtype=np.float32)
            old[0] = bodies[:, :3]
            for step in range(1, len(old)):
                old[step] = old[step - 1] + np.float32(.01) * bodies[:, 3:6].astype(np.float32)
            write_bin(paths[0], old)
            report = audit(*paths)
            self.assertEqual(report["status"], "FAIL")
            self.assertFalse(next(c for c in report["checks"] if c["name"] == "cpu_trajectory_replay")["passed"])

    def test_initial_mismatch_and_nonfinite_fail(self):
        bodies = np.array([[0., 0, 0, 1, 0, 0, 1]])
        trajectory = np.array([[[0., 0, 0]], [[.01, 0, 0]]])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            paths = self.case(directory, bodies, trajectory)
            trajectory[0, 0, 0] = 1
            write_bin(paths[0], trajectory)
            self.assertEqual(audit(*paths)["status"], "FAIL")
            trajectory[1, 0, 0] = np.nan
            write_bin(paths[0], trajectory)
            self.assertEqual(audit(*paths)["status"], "FAIL")

    def test_sampled_status_is_not_full_pass(self):
        rng = np.random.default_rng(7)
        bodies = np.column_stack((rng.normal(size=(20, 3)), rng.normal(size=(20, 3)), np.ones(20))).astype(np.float32).astype(np.float64)
        trajectory = bodies[None, :, :3] + np.arange(21)[:, None, None] * float(np.float32(.01)) * bodies[None, :, 3:6]
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            paths = self.case(directory, bodies, trajectory)
            report = audit(*paths, reference_limit=4, sample_count=5)
            self.assertEqual(report["status"], "PASS_SAMPLED")
            self.assertEqual(len(report["reference_particle_ids"]), 5)
            self.assertTrue(report["limitations"])

    def test_circular_reference_converges_against_analytic(self):
        omega = np.sqrt(2 / (1 + .02**2)**1.5)
        initial = np.array([[-.5, 0, 0, 0, -.5*omega, 0, 1], [.5, 0, 0, 0, .5*omega, 0, 1]])
        errors = []
        for dt, steps in ((.02, 200), (.01, 400)):
            params = dict(dt=dt, num_steps=steps, record_interval=1, G=1, softening=.02, integrator="leapfrog")
            values = np.stack([p for _, p in replay(initial, params)])
            times = np.arange(steps + 1) * dt
            analytic = np.column_stack((-.5 * np.cos(omega * times), -.5 * np.sin(omega * times), times * 0))
            errors.append(np.linalg.norm(values[:, 0] - analytic, axis=1).max())
        self.assertGreater(errors[0] / errors[1], 3.8)

    def test_focus_keeps_true_subpixel_motion(self):
        trajectory = np.array([[[50000, 1, 0], [2, 0, 0]], [[50000.25, 1, 0], [2, 1, 0]]])
        particle, path = focus_path(trajectory, 0)
        self.assertEqual(particle, 0)
        np.testing.assert_array_equal(path, [[0, 0, 0], [.25, 0, 0]])
        self.assertEqual(focus_path(trajectory)[0], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""CPU regressions plus optional real CUDA parity and animation export tests.

python3 test_visualization.py                 # skips unavailable dependencies
python3 test_visualization.py --require-cuda  # A800: missing CUDA is a failure
"""
import importlib.util
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from raster_animation import (BACKGROUND, FrameBuffer, Rasterizer, camera_limits,
                              frame_indices, projection_matrix, select_tracers)
from visualize import load_object_types, load_trajectory
from data.generate_orbit_demo import generate


def write_bin(path, values):
    with open(path, "wb") as output:
        output.write(struct.pack("<ii", values.shape[1], values.shape[0]))
        np.asarray(values.transpose(1, 0, 2), dtype="<f4").tofile(output)


def cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class DataTests(unittest.TestCase):
    def test_particle_major_and_chunk_boundaries(self):
        values = np.arange(9 * 7 * 3, dtype=np.float32).reshape(9, 7, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.bin"
            write_bin(path, values)
            loaded = load_trajectory(path)
            self.assertIsInstance(loaded, np.memmap)
            self.assertFalse(loaded.flags.writeable)
            np.testing.assert_array_equal(loaded, values)
            buffer = FrameBuffer(loaded, 3)
            for frame in (0, 2, 3, 8, 1):
                np.testing.assert_array_equal(buffer.get(frame), values[frame])
                self.assertLessEqual(len(buffer.buffer), 3)
            del buffer
            loaded._mmap.close()

    def test_bad_file_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.bin"
            for content in (b"", struct.pack("<ii", -1, 2), struct.pack("<ii", 2, 3)):
                path.write_bytes(content)
                with self.assertRaises(ValueError):
                    load_trajectory(path)
            path.write_text("index type mass\n0 star 1\n0 star 1\n")
            with self.assertRaises(ValueError):
                load_object_types(path, 2)

    def test_frame_schedule(self):
        np.testing.assert_array_equal(frame_indices(10, 4), [0, 4, 8, 9])
        np.testing.assert_array_equal(frame_indices(1, 4), [0])
        preview = frame_indices(1001, 4, 20)
        self.assertEqual((preview[0], preview[-1], len(preview)), (0, 1000, 20))
        self.assertTrue(np.all(np.diff(preview) > 0))

    def test_projection_camera_and_tracers(self):
        xy = projection_matrix("2d", z_scale=35)
        np.testing.assert_array_equal(np.array([[1, 2, 999]]) @ xy, [[1, 2]])
        np.testing.assert_allclose(projection_matrix("3d").T @ projection_matrix("3d"), np.eye(2), atol=1e-6)
        data = np.random.default_rng(7).normal(size=(4, 25, 3)).astype(np.float32)
        low, high = camera_limits(data, xy, 160, 90, radius=2, center=[0, 0, 0])
        np.testing.assert_allclose((high - low) / [160, 90], [4 / 90, 4 / 90])
        ids = select_tracers(data, xy, low, high, 8)
        self.assertEqual(len(np.unique(ids)), 8)

    def test_demo_count_mass_momentum(self):
        data = generate(4096)
        self.assertEqual(data.shape, (4096, 7))
        self.assertTrue(np.isfinite(data).all())
        self.assertAlmostEqual(data[:, 6].sum(), 1.001)
        np.testing.assert_allclose((data[:, 3:6] * data[:, 6:7]).sum(axis=0), 0, atol=1e-17)


class RasterTests(unittest.TestCase):
    def make_renderer(self, colors, device=None):
        return Rasterizer(projection_matrix("2d"), np.array([-1, -1], dtype=np.float32),
                          np.array([1, 1], dtype=np.float32), colors, 64, 48, device=device)

    def test_all_65536_particles_contribute(self):
        points = np.zeros((65536, 3), dtype=np.float32)
        renderer = self.make_renderer(np.tile([1, 0.5, 0.2], (len(points), 1)))
        output = renderer.render(points)
        self.assertAlmostEqual(float(renderer.canvas[:-1, 3].sum()), len(points))
        self.assertEqual(output.shape, (48, 64, 3))
        self.assertEqual(output.dtype, np.uint8)
        self.assertTrue(np.any(output != np.rint(BACKGROUND * 255)))

    def test_invalid_and_offscreen_do_not_wrap(self):
        points = np.array([[3, 0, 0], [np.nan, 0, 0], [1e30, -1e30, 0]], dtype=np.float32)
        renderer = self.make_renderer(np.ones((3, 3)))
        output = renderer.render(points)
        self.assertEqual(renderer.canvas[:-1, 3].sum(), 0)
        np.testing.assert_array_equal(output, np.broadcast_to(np.rint(BACKGROUND * 255).astype(np.uint8), output.shape))

    @unittest.skipUnless(cuda_available(), "CUDA-enabled PyTorch unavailable")
    def test_cuda_numpy_parity_and_chunk_upload(self):
        import torch
        rng = np.random.default_rng(9)
        points = rng.uniform(-1.4, 1.4, (4096, 3)).astype(np.float32)
        points[0] = np.nan
        colors = rng.random((len(points), 3), dtype=np.float32)
        cpu = self.make_renderer(colors)
        gpu = self.make_renderer(colors, "cuda:0")
        np.testing.assert_allclose(cpu.render(points).astype(float), gpu.render(points).astype(float), atol=1)
        data = np.stack((points, points * 0.5, points * 0.3))
        buffer = FrameBuffer(data, 2, "cuda:0")
        self.assertTrue(buffer.get(0).is_cuda)
        np.testing.assert_allclose(buffer.get(2).cpu().numpy(), data[2])
        self.assertLessEqual(len(buffer.buffer), 2)
        print("CUDA parity checked on", torch.cuda.get_device_name(0))


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib unavailable")
class ExportTests(unittest.TestCase):
    def test_funcanimation_export_all_cpu_modes(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            phase = np.linspace(0, np.pi, 7)
            data = np.zeros((7, 2, 3), dtype=np.float32)
            data[:, 0, 0], data[:, 0, 1] = np.cos(phase), np.sin(phase)
            data[:, 1] = -data[:, 0]
            write_bin(directory / "test.bin", data)
            for backend, dimension in (("numpy", "2d"), ("numpy", "3d"),
                                       ("matplotlib", "2d"), ("matplotlib", "3d")):
                output = directory / f"{backend}_{dimension}.gif"
                command = [sys.executable, str(Path(__file__).with_name("visualize.py")),
                           str(directory / "test.bin"), "--backend", backend,
                           "--dimension", dimension, "--output", str(output),
                           "--max-frames", "4", "--width", "640", "--height", "480",
                           "--raster-width", "128", "--raster-height", "96", "--fps", "10", "--trail", "4"]
                subprocess.run(command, check=True, capture_output=True,
                               env={**os.environ, "MPLBACKEND": "Agg"})
                with Image.open(output) as image:
                    self.assertEqual(image.n_frames, 4)
                    self.assertEqual(image.size, (640, 480))
                    image.seek(0)
                    first = np.asarray(image.convert("RGB"))
                    image.seek(3)
                    self.assertTrue(np.any(first != np.asarray(image.convert("RGB"))))
                self.assertIn("func_animation=True", output.with_suffix(".visualization.log").read_text())


if __name__ == "__main__":
    if "--require-cuda" in sys.argv:
        sys.argv.remove("--require-cuda")
        if not cuda_available():
            raise SystemExit("FAIL: --require-cuda was requested but CUDA-enabled PyTorch is unavailable")
    unittest.main(verbosity=2)

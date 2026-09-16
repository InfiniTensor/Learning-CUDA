#!/usr/bin/env python3

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_ivf_sweep.py"
SPEC = importlib.util.spec_from_file_location("benchmark_ivf_sweep", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BenchmarkIvfSweepTests(unittest.TestCase):
    def test_reads_config_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "params.conf"
            path.write_text(
                "# comment\nbatch_size = 64\ndistance_mode = bucket_major # note\n"
            )
            self.assertEqual(
                MODULE.read_params(path),
                {"batch_size": "64", "distance_mode": "bucket_major"},
            )

    def test_extracts_latency_pair(self):
        p50, p99 = MODULE.extract_pair(
            "batch P50/P99: 12.25/18.75 ms",
            r"batch P50/P99: ([0-9.eE+-]+)/([0-9.eE+-]+) ms",
        )
        self.assertEqual((p50, p99), (12.25, 18.75))

    def test_recall(self):
        mean, minimum = MODULE.recall([[1, 2], [3, 4]], [[2, 1], [3, 8]])
        self.assertEqual((mean, minimum), (0.75, 0.5))

    def test_quality_metrics(self):
        metrics = MODULE.quality_metrics(
            [[(1, 3.0), (2, 2.0)], [(3, 1.0), (4, 0.0)]],
            [[(2, 2.5), (1, 1.5)], [(3, 0.75), (8, -0.25)]],
        )
        self.assertEqual(metrics, (0.75, 0.5, 0.375, 0.5))


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_adaptive_nprobe.py"
SPEC = importlib.util.spec_from_file_location("benchmark_adaptive_nprobe", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BenchmarkAdaptiveNprobeTests(unittest.TestCase):
    def test_extracts_actual_probe_triple(self):
        values = MODULE.extract_triple(
            "实际 nprobe min/avg/max: 64/137.25/224",
            r"实际 nprobe min/avg/max: ([0-9.eE+-]+)/([0-9.eE+-]+)/([0-9.eE+-]+)",
        )
        self.assertEqual(values, (64.0, 137.25, 224.0))

    def test_recall(self):
        mean, minimum = MODULE.recall([[1, 2], [3, 4]], [[1, 9], [4, 3]])
        self.assertEqual((mean, minimum), (0.75, 0.5))


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_exact_breakdown", ROOT / "scripts" / "benchmark_exact_breakdown.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BenchmarkExactBreakdownTests(unittest.TestCase):
    def test_parses_breakdown_and_shares(self):
        text = """\
向量库: 1000000 x 384
查询数: 1000
top_k: 50
batch_size: 64
GPU 缓冲区: 1005.2 MiB
平均常驻查询时间: 300.0 ms
run P50: 299.0 ms
run P99: 305.0 ms
常驻 QPS: 3333.3
剖析距离-only kernel: 100.0 ms
剖析局部 Top-K 估算: 150.0 ms
剖析融合 kernel: 250.0 ms
剖析最终归并 kernel: 10.0 ms
"""
        row = MODULE.parse_profile_output(text)
        self.assertEqual(row["top_k"], 50)
        self.assertEqual(row["kernel_pipeline_ms"], 260.0)
        self.assertAlmostEqual(row["distance_share"], 100.0 / 260.0)
        self.assertAlmostEqual(row["local_topk_share"], 150.0 / 260.0)

    def test_compares_reference_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.txt"
            second = root / "second.txt"
            first.write_text("0 7 1.25\n0 8 1.0\n")
            second.write_text("0 7 1.25001\n0 8 1.0\n")
            MODULE.compare_results(first, second, 2e-5)

    def test_rejects_candidate_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.txt"
            second = root / "second.txt"
            first.write_text("0 7 1.25\n")
            second.write_text("0 8 1.25\n")
            with self.assertRaises(ValueError):
                MODULE.compare_results(first, second, 1e-5)

    def test_rejects_rank_order_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.txt"
            second = root / "second.txt"
            first.write_text("0 7 1.25\n0 8 1.0\n")
            second.write_text("0 8 1.0\n0 7 1.25\n")
            with self.assertRaises(ValueError):
                MODULE.compare_results(first, second, 1e-5)


if __name__ == "__main__":
    unittest.main()

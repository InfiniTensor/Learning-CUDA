#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "benchmark_exact_query_tile", ROOT / "scripts" / "benchmark_exact_query_tile.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BenchmarkExactQueryTileTests(unittest.TestCase):
    def test_parses_variant(self):
        label, configs = MODULE.parse_variant(
            f"tile2=10:{ROOT / 'README.md'},50:{ROOT / 'CMakeLists.txt'}"
        )
        self.assertEqual(label, "tile2")
        self.assertEqual([top_k for top_k, _ in configs], [10, 50])

    def test_rejects_unsafe_label(self):
        with self.assertRaises(ValueError):
            MODULE.parse_variant(f"../tile=10:{ROOT / 'README.md'}")

    def test_parses_search_output(self):
        text = """\
向量库: 1000000 x 128
查询数: 1000
top_k: 10
batch_size: 64
exact_query_tile: 4
exact_fused_layout: block_tile
exact_fused_blocks: 128
GPU 缓冲区: 488.4 MiB
平均常驻查询时间: 50.0 ms
run P50: 49.0 ms
run P99: 52.0 ms
常驻 QPS: 20000.0
平均融合距离+局部 Top-K kernel: 48.0 ms
平均最终 Top-K kernel: 0.2 ms
"""
        row = MODULE.parse_search_output(text, "tile4")
        self.assertEqual(row["exact_query_tile"], 4)
        self.assertEqual(row["exact_fused_layout"], "block_tile")
        self.assertEqual(row["exact_fused_blocks"], 128)
        self.assertEqual(row["qps"], 20000.0)
        self.assertEqual(row["fused_kernel_ms"], 48.0)


if __name__ == "__main__":
    unittest.main()

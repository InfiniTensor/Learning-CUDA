#!/usr/bin/env python3

import importlib.util
import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


EXACT = load_module("benchmark_exact_stage6", "benchmark_exact.py")
RUNNER = load_module("run_real_benchmarks_stage6", "run_real_benchmarks.py")
PREPARE = load_module("prepare_real_datasets_stage6", "prepare_real_datasets.py")


class Stage6ToolTests(unittest.TestCase):
    def test_partitioned_result_falls_back_on_recall_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fields = ["nprobe", "recall_at_k", "qps"]
            for name, recall, qps in (
                ("partitioned.csv", 0.89, 1200),
                ("full.csv", 0.90, 1000),
            ):
                with (root / name).open("w", newline="") as output:
                    writer = csv.DictWriter(output, fieldnames=fields)
                    writer.writeheader()
                    writer.writerow({"nprobe": 64, "recall_at_k": recall, "qps": qps})
            decision = RUNNER.choose_partitioned_result(
                root / "partitioned.csv",
                root / "full.csv",
                root / "effective.csv",
                maximum_recall_loss=1e-4,
            )
            self.assertEqual(decision["selected_mode"], "full_heap")
            self.assertEqual(
                (root / "effective.csv").read_text(),
                (root / "full.csv").read_text(),
            )

    def test_exact_config_parser_and_output_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "k10.conf"
            second = root / "k100.conf"
            first.write_text("top_k = 10\n")
            second.write_text("top_k = 100\n")
            self.assertEqual(
                EXACT.parse_configs(f"10:{first},100:{second}"),
                [(10, first), (100, second)],
            )
        self.assertEqual(EXACT.extract("常驻 QPS: 1234.5", r"常驻 QPS: ([0-9.]+)"), 1234.5)

    def test_build_metadata_parser(self):
        output = """\
迭代/训练样本: 15/100000
建库线程: 16
构建时间: 1234.5 ms
保存时间: 67.8 ms
桶大小 min/avg/max: 10/1000.25/2048
空桶: 3
索引大小: 987654 bytes
"""
        row = RUNNER.parse_build_stats(output, "fixture", 1024)
        self.assertEqual(row["training_iterations"], 15)
        self.assertEqual(row["maximum_bucket"], 2048)
        self.assertEqual(row["index_bytes"], 987654)

    def test_known_public_checksums_are_pinned(self):
        self.assertEqual(len(PREPARE.SIFT_MD5), 32)
        self.assertEqual(len(PREPARE.VIBE_SHA256), 64)


if __name__ == "__main__":
    unittest.main()

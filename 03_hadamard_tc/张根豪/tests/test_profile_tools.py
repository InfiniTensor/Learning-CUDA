#!/usr/bin/env python3
"""CPU-only tests for profiler CSV parsing and Roofline peak calculations."""

from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("parse_ncu", ROOT / "scripts" / "parse_ncu.py")
PARSE_NCU = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PARSE_NCU)


class ProfileToolsTest(unittest.TestCase):
    def test_load_raw_ncu_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fp16_hd128_tokens131072.csv"
            with path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["==PROF== Connected"])
                writer.writerow(["ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value"])
                writer.writerow(["0", "hadamard_kernel", "gpu__time_duration.sum", "nsecond", "100,000"])
                writer.writerow(["0", "hadamard_kernel", "dram__bytes.sum", "Mbyte", "70"])
            metrics = PARSE_NCU.load_metrics(path)
            self.assertEqual(
                PARSE_NCU.metric_scaled(metrics, ["gpu__time_duration.sum"], "seconds"),
                1e-4,
            )
            self.assertEqual(
                PARSE_NCU.metric_scaled(metrics, ["dram__bytes.sum"], "bytes"),
                70e6,
            )

    def test_l40s_theoretical_peaks(self) -> None:
        fp32, memory = PARSE_NCU.theoretical_peaks({
            "compute_capability_major": 8,
            "compute_capability_minor": 9,
            "sm_count": 142,
            "clock_rate_khz": 2520000,
            "memory_clock_rate_khz": 9000000,
            "memory_bus_width_bits": 384,
        })
        self.assertAlmostEqual(fp32, 91.60704)
        self.assertAlmostEqual(memory, 864.0)


if __name__ == "__main__":
    unittest.main()

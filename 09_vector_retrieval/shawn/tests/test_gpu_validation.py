#!/usr/bin/env python3
"""Unit tests for the CPU/GPU result-file parity contract."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_gpu_validation as validation  # noqa: E402


class ResultComparisonTest(unittest.TestCase):
    @staticmethod
    def quality(*, rank: float, score_error: float) -> dict[str, str]:
        return {
            "recall_at_k": "1",
            "rank_id_agreement": str(rank),
            "mean_rank_score_absolute_error": str(score_error),
            "qps": "100",
            "num_vectors": "16",
            "dim": "4",
            "num_queries": "1",
            "batch_size": "1",
            "nlist": "4",
            "nprobe": "2",
        }

    def compare(self, cpu: str, gpu: str, tolerance: float):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cpu_path = root / "cpu.results.txt"
            gpu_path = root / "gpu.results.txt"
            cpu_path.write_text(cpu, encoding="utf-8")
            gpu_path.write_text(gpu, encoding="utf-8")
            return validation.compare_results(cpu_path, gpu_path, tolerance)

    def test_identical_results_pass_every_metric(self) -> None:
        result = self.compare(
            "0 10:1.0 11:1.5\n1 20:2.0 21:2.5\n",
            "0 10:1.0 11:1.5\n1 20:2.0 21:2.5\n",
            1.0e-6,
        )
        self.assertEqual(result.strict_rank_agreement, 1.0)
        self.assertEqual(result.id_set_agreement, 1.0)
        self.assertEqual(result.tolerance_aware_rank_agreement, 1.0)
        self.assertEqual(result.max_score_absolute_error, 0.0)

    def test_near_tie_swap_is_tolerance_aware_but_not_strict(self) -> None:
        result = self.compare(
            "0 10:1.0000000 11:1.0000004\n",
            "0 11:1.0000003 10:1.0000001\n",
            1.0e-6,
        )
        self.assertEqual(result.strict_rank_agreement, 0.0)
        self.assertEqual(result.id_set_agreement, 1.0)
        self.assertEqual(result.tolerance_aware_rank_agreement, 1.0)
        self.assertAlmostEqual(result.max_score_absolute_error, 1.0e-7)

    def test_swap_outside_tolerance_fails_tolerance_aware_order(self) -> None:
        result = self.compare(
            "0 10:1.0 11:1.01\n",
            "0 11:1.01 10:1.0\n",
            1.0e-4,
        )
        self.assertEqual(result.id_set_agreement, 1.0)
        self.assertEqual(result.tolerance_aware_rank_agreement, 0.0)

    def test_equal_scores_do_not_hide_a_different_id_set(self) -> None:
        result = self.compare(
            "0 10:1.0 11:1.0\n",
            "0 10:1.0 12:1.0\n",
            1.0e-4,
        )
        self.assertEqual(result.strict_rank_agreement, 0.5)
        self.assertEqual(result.id_set_agreement, 0.5)
        self.assertEqual(result.tolerance_aware_rank_agreement, 0.5)

    def test_duplicate_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate vector IDs"):
            self.compare("0 10:1.0 10:1.0\n", "0 10:1.0 11:1.0\n", 1.0e-4)

    def test_exact_gate_accepts_only_numerically_equivalent_rank_swap(self) -> None:
        direct = self.compare(
            "0 10:1.0000000 11:1.0000004\n",
            "0 11:1.0000003 10:1.0000001\n",
            1.0e-6,
        )
        row = validation.gate_row(
            mode="exact",
            metric="l2",
            dtype="fp32",
            top_k=2,
            cpu=self.quality(rank=1.0, score_error=0.0),
            gpu=self.quality(rank=0.0, score_error=3.0e-7),
            database_hash="database",
            query_hash="queries",
            score_tolerance=1.0e-6,
            direct=direct,
        )
        self.assertEqual(row["passed"], "true")
        self.assertEqual(row["cpu_gpu_rank_agreement"], "0")
        self.assertEqual(row["cpu_gpu_id_set_agreement"], "1")
        self.assertEqual(row["cpu_gpu_tolerance_aware_rank_agreement"], "1")

    def test_exact_gate_rejects_a_different_id_set(self) -> None:
        direct = self.compare(
            "0 10:1.0 11:1.0\n",
            "0 10:1.0 12:1.0\n",
            1.0e-4,
        )
        row = validation.gate_row(
            mode="exact",
            metric="l2",
            dtype="fp32",
            top_k=2,
            cpu=self.quality(rank=1.0, score_error=0.0),
            gpu=self.quality(rank=1.0, score_error=0.0),
            database_hash="database",
            query_hash="queries",
            score_tolerance=1.0e-4,
            direct=direct,
        )
        self.assertEqual(row["passed"], "false")


if __name__ == "__main__":
    unittest.main()

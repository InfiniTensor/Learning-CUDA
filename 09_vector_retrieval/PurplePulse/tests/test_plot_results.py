#!/usr/bin/env python3

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "plot_results.py"
SPEC = importlib.util.spec_from_file_location("plot_results", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class PlotResultsTests(unittest.TestCase):
    def test_generates_all_figures_from_minimal_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            output = root / "figures"
            results.mkdir()

            quality_fields = ["nprobe", "average_ms", "qps", "recall_at_k"]
            for top_k in (10, 50, 100):
                write_csv(
                    results / f"ivf_k{top_k}.csv",
                    quality_fields,
                    [
                        {
                            "nprobe": 16,
                            "average_ms": 2.0 + top_k / 100.0,
                            "qps": 5000 - top_k,
                            "recall_at_k": 0.9,
                        },
                        {
                            "nprobe": 32,
                            "average_ms": 3.0 + top_k / 100.0,
                            "qps": 4000 - top_k,
                            "recall_at_k": 0.97,
                        },
                    ],
                )

            faiss_fields = [
                "backend",
                "index_type",
                "top_k",
                "nprobe",
                "average_query_ms",
                "qps",
                "recall_at_k",
            ]
            write_csv(
                results / "faiss_gpu.csv",
                faiss_fields,
                [
                    {
                        "backend": "gpu",
                        "index_type": "ivf_flat",
                        "top_k": top_k,
                        "nprobe": nprobe,
                        "average_query_ms": 2.5 if nprobe == 16 else 3.5,
                        "qps": 4500 if nprobe == 16 else 3500,
                        "recall_at_k": 0.89 if nprobe == 16 else 0.96,
                    }
                    for top_k in (10, 50, 100)
                    for nprobe in (16, 32)
                ],
            )

            batch_fields = ["top_k", "batch_size", "qps"]
            write_csv(
                results / "optimized_v1_batch_sweep.csv",
                batch_fields,
                [
                    {"top_k": top_k, "batch_size": batch, "qps": 1000 + batch}
                    for top_k in (10, 50, 100)
                    for batch in (8, 64)
                ],
            )

            paths = MODULE.generate_figures(results, output)

            self.assertEqual(
                {path.name for path in paths},
                {"recall-latency.svg", "recall-qps.svg", "batch-qps.svg"},
            )
            for path in paths:
                contents = path.read_text()
                self.assertIn("<svg", contents)
                self.assertIn("</svg>", contents)


if __name__ == "__main__":
    unittest.main()

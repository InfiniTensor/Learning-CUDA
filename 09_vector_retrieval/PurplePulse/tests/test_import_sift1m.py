#!/usr/bin/env python3

import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


IMPORTER = load_module("import_sift1m", ROOT / "scripts" / "import_sift1m.py")
EVALUATOR = load_module(
    "evaluate_ivecs_groundtruth",
    ROOT / "scripts" / "evaluate_ivecs_groundtruth.py",
)


def write_vecs(path: Path, values: np.ndarray, kind: str) -> None:
    value_dtype = "<f4" if kind == "fvecs" else "<i4"
    with path.open("wb") as output:
        for row in np.asarray(values, dtype=value_dtype):
            output.write(struct.pack("<i", row.size))
            output.write(row.tobytes())


class ImportSift1mTests(unittest.TestCase):
    def test_converts_and_evaluates_texmex_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path = root / "sift_base.fvecs"
            query_path = root / "sift_query.fvecs"
            groundtruth_path = root / "sift_groundtruth.ivecs"
            database_output = root / "database.bin"
            query_output = root / "queries.bin"
            groundtruth_output = root / "groundtruth.ivecs"
            manifest_output = root / "manifest.json"
            base = np.asarray([[0, 0], [1, 0], [0, 2]], dtype=np.float32)
            queries = np.asarray([[0.1, 0], [0, 1.8]], dtype=np.float32)
            groundtruth = np.asarray([[0, 1], [2, 0]], dtype=np.int32)
            write_vecs(base_path, base, "fvecs")
            write_vecs(query_path, queries, "fvecs")
            write_vecs(groundtruth_path, groundtruth, "ivecs")

            manifest = IMPORTER.convert(
                base_path,
                query_path,
                groundtruth_path,
                database_output,
                query_output,
                manifest_output,
                query_limit=1,
                groundtruth_output=groundtruth_output,
            )

            self.assertEqual(manifest["database_rows"], 3)
            self.assertEqual(manifest["query_rows"], 1)
            self.assertEqual(manifest["dim"], 2)
            self.assertEqual(manifest["groundtruth_k"], 2)
            self.assertEqual(database_output.read_bytes()[:8], b"PPVEC001")
            self.assertEqual(query_output.read_bytes()[:8], b"PPQRY001")
            self.assertEqual(
                struct.unpack("<QIII", database_output.read_bytes()[8:28]),
                (3, 2, 1, 1),
            )
            self.assertEqual(
                struct.unpack("<QII", query_output.read_bytes()[8:24]),
                (1, 2, 1),
            )
            np.testing.assert_array_equal(
                np.frombuffer(database_output.read_bytes(), dtype="<f4", offset=28),
                base.reshape(-1),
            )
            saved_manifest = json.loads(manifest_output.read_text())
            self.assertEqual(saved_manifest["metric"], "l2")
            self.assertEqual(len(saved_manifest["inputs"]["base"]["sha256"]), 64)
            self.assertEqual(groundtruth_output.read_bytes(), groundtruth_path.read_bytes())
            self.assertEqual(
                saved_manifest["outputs"]["groundtruth"]["sha256"],
                IMPORTER.sha256_file(groundtruth_output),
            )

            result_path = root / "results.txt"
            result_path.write_text("0 0 0.01\n0 1 0.81\n")
            metrics = EVALUATOR.evaluate(groundtruth_path, result_path)
            self.assertEqual(metrics["recall_at_k"], 1.0)
            self.assertEqual(metrics["exact_query_fraction"], 1.0)
            scored_metrics = EVALUATOR.evaluate(
                groundtruth_path,
                result_path,
                database_output,
                query_output,
            )
            self.assertEqual(scored_metrics["score_equivalent_query_fraction"], 1.0)
            self.assertLess(scored_metrics["max_reported_score_error"], 1e-6)

    def test_rejects_inconsistent_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.fvecs"
            with path.open("wb") as output:
                output.write(struct.pack("<i2f", 2, 1.0, 2.0))
                output.write(struct.pack("<i2f", 3, 3.0, 4.0))
            with self.assertRaisesRegex(ValueError, "inconsistent row dimensions"):
                IMPORTER.inspect_vecs(path, "fvecs")


if __name__ == "__main__":
    unittest.main()

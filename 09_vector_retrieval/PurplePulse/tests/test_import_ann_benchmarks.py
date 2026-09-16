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
SPEC = importlib.util.spec_from_file_location(
    "import_ann_benchmarks", ROOT / "scripts" / "import_ann_benchmarks.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["import_ann_benchmarks"] = MODULE
SPEC.loader.exec_module(MODULE)


try:
    import h5py
except ImportError:
    h5py = None


@unittest.skipIf(h5py is None, "h5py is not installed")
class ImportAnnBenchmarksTests(unittest.TestCase):
    def test_accepts_and_validates_vibe_normalized_vectors(self):
        self.assertEqual(MODULE.normalized_distance("normalized"), "normalized")
        MODULE.validate_unit_norms(
            np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32), "fixture"
        )
        with self.assertRaisesRegex(ValueError, "unit-norm validation failed"):
            MODULE.validate_unit_norms(
                np.asarray([[2.0, 0.0]], dtype=np.float32), "fixture"
            )

    def test_converts_hdf5_and_checks_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sift.hdf5"
            database = root / "database.bin"
            queries = root / "queries.bin"
            groundtruth = root / "groundtruth.ivecs"
            manifest = root / "manifest.json"
            with h5py.File(source, "w") as output:
                output.attrs["distance"] = "euclidean"
                output.create_dataset(
                    "train", data=np.asarray([[0, 0], [1, 0], [0, 2]], dtype=np.float32)
                )
                output.create_dataset(
                    "test", data=np.asarray([[0.1, 0], [0, 1.8]], dtype=np.float32)
                )
                output.create_dataset(
                    "neighbors", data=np.asarray([[0, 1], [2, 0]], dtype=np.int32)
                )
            digest = MODULE.sha256_file(source)

            result = MODULE.convert(
                source,
                database,
                queries,
                groundtruth,
                manifest,
                query_limit=1,
                expected_sha256=digest,
                source_url="https://example.invalid/sift.hdf5",
                dataset_name="fixture-2d",
                expected_dim=2,
            )

            self.assertEqual(result["database_rows"], 3)
            self.assertEqual(result["query_rows"], 1)
            self.assertEqual(result["dim"], 2)
            self.assertEqual(database.read_bytes()[:8], b"PPVEC001")
            self.assertEqual(queries.read_bytes()[:8], b"PPQRY001")
            self.assertEqual(
                struct.unpack("<QIII", database.read_bytes()[8:28]), (3, 2, 1, 1)
            )
            self.assertEqual(
                struct.unpack("<QII", queries.read_bytes()[8:24]), (1, 2, 1)
            )
            self.assertEqual(
                np.frombuffer(groundtruth.read_bytes(), dtype="<i4").tolist(),
                [2, 0, 1],
            )
            saved = json.loads(manifest.read_text())
            self.assertEqual(saved["dataset"], "fixture-2d")
            self.assertEqual(saved["warpfusion_metric"], "l2")
            self.assertEqual(saved["source_sha256"], digest)

            with self.assertRaisesRegex(ValueError, "dimension mismatch"):
                MODULE.convert(
                    source,
                    database,
                    queries,
                    groundtruth,
                    manifest,
                    expected_dim=384,
                )

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                MODULE.convert(
                    source,
                    database,
                    queries,
                    groundtruth,
                    manifest,
                    expected_sha256="0" * 64,
                )


if __name__ == "__main__":
    unittest.main()

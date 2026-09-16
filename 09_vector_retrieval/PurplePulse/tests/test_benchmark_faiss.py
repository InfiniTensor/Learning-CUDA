#!/usr/bin/env python3

import importlib.util
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_faiss.py"
SPEC = importlib.util.spec_from_file_location("benchmark_faiss", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BenchmarkFaissTests(unittest.TestCase):
    def test_reads_custom_vector_files_as_memmap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "database.bin"
            queries = root / "queries.bin"
            values = np.arange(12, dtype="<f4").reshape(3, 4)
            database.write_bytes(
                b"PPVEC001" + struct.pack("<QIII", 3, 4, 1, 2) + values.tobytes()
            )
            queries.write_bytes(
                b"PPQRY001" + struct.pack("<QII", 2, 4, 1) + values[:2].tobytes()
            )

            database_info = MODULE.read_database(database)
            query_info = MODULE.read_queries(queries)

            self.assertEqual((database_info.rows, database_info.dim), (3, 4))
            self.assertEqual(database_info.metric, "inner_product")
            np.testing.assert_array_equal(database_info.mmap(), values)
            np.testing.assert_array_equal(query_info.mmap(), values[:2])

    def test_recall_and_percentile(self):
        reference = np.asarray([[1, 2], [3, 4]], dtype=np.int64)
        actual = np.asarray([[2, 1], [3, 8]], dtype=np.int64)
        mean, minimum = MODULE.recall(reference, actual)
        self.assertEqual(mean, 0.75)
        self.assertEqual(minimum, 0.5)
        self.assertEqual(MODULE.percentile([1.0, 2.0, 3.0], 0.99), 3.0)
        self.assertEqual(MODULE.score_errors(
            np.asarray([[3.0, 2.0], [1.0, 0.0]]),
            np.asarray([[2.5, 1.5], [0.75, -0.25]]),
        ), (0.375, 0.5))

    def test_rejects_truncated_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "database.bin"
            path.write_bytes(b"PPVEC001" + struct.pack("<QIII", 3, 4, 1, 1))
            with self.assertRaisesRegex(ValueError, "文件大小"):
                MODULE.read_database(path)

    def test_reads_result_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.txt"
            path.write_text("0 8 1.0\n0 7 0.5\n1 3 0.2\n1 4 0.1\n")
            ids = MODULE.read_result_ids(path, 2, 2)
            np.testing.assert_array_equal(ids, [[8, 7], [3, 4]])
            ids, scores = MODULE.read_result_arrays(path, 2, 2)
            np.testing.assert_array_equal(ids, [[8, 7], [3, 4]])
            np.testing.assert_allclose(scores, [[1.0, 0.5], [0.2, 0.1]])

    def test_writes_csv_incrementally(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            fields = {
                field.name: 0 for field in MODULE.BenchmarkRow.__dataclass_fields__.values()
            }
            fields.update(implementation="faiss-test", backend="gpu", index_type="flat",
                          metric="l2", dtype="fp32")
            MODULE.write_csv(path, [MODULE.BenchmarkRow(**fields)])
            self.assertIn("faiss-test", path.read_text())

    def test_splits_faiss_gpu_memory_categories(self):
        class Resource:
            @staticmethod
            def getMemoryInfo():
                return {0: {"IVFLists": (2, 2 * 1024 * 1024),
                            "TemporaryMemoryBuffer": (1, 3 * 1024 * 1024)}}

        self.assertEqual(MODULE.gpu_memory_mib(Resource()), (2.0, 3.0))


if __name__ == "__main__":
    unittest.main()

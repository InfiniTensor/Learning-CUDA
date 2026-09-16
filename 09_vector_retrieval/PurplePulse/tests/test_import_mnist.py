#!/usr/bin/env python3

import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "import_mnist.py"
SPEC = importlib.util.spec_from_file_location("import_mnist", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ImportMnistTests(unittest.TestCase):
    def test_converts_images_to_vector_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mnist.npz"
            database = root / "database.bin"
            queries = root / "queries.bin"
            np.savez(
                source,
                x_train=np.arange(24, dtype=np.uint8).reshape(3, 2, 4),
                x_test=np.arange(16, dtype=np.uint8).reshape(2, 2, 4),
            )

            database_shape, query_shape = MODULE.convert(
                source, database, queries, database_size=2, query_count=1
            )

            self.assertEqual(database_shape, (2, 8))
            self.assertEqual(query_shape, (1, 8))
            self.assertEqual(database.read_bytes()[:8], b"PPVEC001")
            self.assertEqual(queries.read_bytes()[:8], b"PPQRY001")
            self.assertEqual(
                struct.unpack("<QIII", database.read_bytes()[8:28]),
                (2, 8, 1, 1),
            )
            first = np.frombuffer(database.read_bytes(), dtype="<f4", offset=28)
            np.testing.assert_allclose(first[:8], np.arange(8) / 255.0)


if __name__ == "__main__":
    unittest.main()

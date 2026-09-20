"""Reader memory bound and corruption checks independent of the simulator."""
import struct
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import trajectory


class ReaderTests(unittest.TestCase):
    def test_metadata_types_and_integer_overflow(self):
        valid = dict(format="nbody-particle-major-v1", endianness="little",
                     dtype="float32", particles=1, records=3, steps=[0, 1, 2], dt=0.1)
        invalid = [None, [], True]
        for key, values in {
            "particles": [True, 1.0], "records": [3.0, "3"],
            "dtype": [None, "float64"],
            "steps": [[0, True, 2], [0, 2**63, 1], [0, 2, 1],
                      [0, -1, 2], [0, 1, 2**31], None, [[0], [1], [2]]],
            "dt": [True, "0.1", [], None, 0, -1, float("nan"),
                   float("inf"), 10**400],
        }.items():
            invalid.extend({**valid, key: value} for value in values)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.bin"
            path.write_bytes(struct.pack("<ii", 1, 3) + bytes(36))
            sidecar = Path(str(path) + ".json")
            for metadata in invalid:
                with self.subTest(metadata=metadata):
                    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
                    with patch.object(trajectory.np, "memmap") as mapping:
                        with self.assertRaises(ValueError):
                            trajectory.load_trajectory(path)
                        mapping.assert_not_called()
            for steps in ([0, 1, 2], [0, 1, 2147483647]):
                sidecar.write_text(json.dumps({**valid, "steps": steps}), encoding="utf-8")
                positions, metadata = trajectory.load_trajectory(path)
                self.assertEqual(metadata["steps"], steps)
                positions._mmap.close()

    def test_invalid_metadata_is_rejected_before_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.bin"
            path.write_bytes(struct.pack("<ii3f", 1, 1, 0, 0, 0))
            Path(str(path) + ".json").write_text('{"format":"wrong"}')
            with patch.object(trajectory.np, "memmap") as mapping:
                with self.assertRaisesRegex(ValueError, "Unsupported trajectory metadata"):
                    trajectory.load_trajectory(path)
                mapping.assert_not_called()

    def test_long_single_particle_history_is_scanned_in_bounded_chunks(self):
        records = trajectory.FINITE_SCAN_BYTES // 12 + 7
        for corrupt in (False, True):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "history.bin"
                values = np.zeros(records * 3, dtype="<f4")
                if corrupt:
                    values[-1] = np.nan
                path.write_bytes(struct.pack("<ii", 1, records) + values.tobytes())
                sizes = []
                original = np.isfinite

                def check(chunk):
                    sizes.append(chunk.nbytes)
                    return original(chunk)

                with patch.object(trajectory.np, "isfinite", side_effect=check):
                    if corrupt:
                        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
                            trajectory.load_trajectory(path)
                    else:
                        positions, meta = trajectory.load_trajectory(path)
                        self.assertEqual(positions.shape, (1, records, 3))
                        self.assertTrue(meta["time_is_record_index"])
                        self.assertEqual(float(positions[0, -1, 2]), 0)
                        positions._mmap.close()
                self.assertGreater(len(sizes), 1)
                self.assertLessEqual(max(sizes), trajectory.FINITE_SCAN_BYTES)
                self.assertEqual(sum(sizes), values.nbytes)


if __name__ == "__main__":
    unittest.main(verbosity=2)

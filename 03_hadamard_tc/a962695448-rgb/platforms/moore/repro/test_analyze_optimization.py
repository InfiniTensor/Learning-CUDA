"""Check that damaged or structurally invalid measurements cannot pass the analysis."""

import csv
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from analyze_optimization import analyze

ROOT = Path(__file__).resolve().parent


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.recorded = Path(self.temporary.name) / "recorded"
        shutil.copytree(ROOT / "recorded", self.recorded)
        self.csv = self.recorded / "paired-round1.csv"

    def change_records(self, change):
        with self.csv.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        fields = list(rows[0])
        change(rows)
        with self.csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        manifest = self.recorded / "optimization.json"
        state = json.loads(manifest.read_text())
        state["artifacts"][self.csv.name]["sha256"] = hashlib.sha256(
            self.csv.read_bytes()
        ).hexdigest()
        manifest.write_text(json.dumps(state))

    def test_original_observations_are_complete_and_accepted(self):
        result = analyze(self.recorded)
        self.assertEqual(result["observations"], 6750)
        self.assertEqual(len(result["comparisons"]), 540)
        self.assertTrue(result["accepted"])

    def test_a_changed_byte_is_rejected(self):
        with self.csv.open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "Checksum"):
            analyze(self.recorded)

    def test_missing_observation_is_rejected_after_rehashing(self):
        self.change_records(lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError, "2250"):
            analyze(self.recorded)

    def test_duplicate_group_is_rejected_after_rehashing(self):
        self.change_records(lambda rows: rows.__setitem__(1, rows[0].copy()))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            analyze(self.recorded)

    def test_wrong_units_are_rejected_after_rehashing(self):
        self.change_records(
            lambda rows: rows[0].__setitem__("kernel_ms", rows[0]["kernel_us"])
        )
        with self.assertRaisesRegex(ValueError, "units"):
            analyze(self.recorded)

    def test_unknown_configuration_is_rejected_after_rehashing(self):
        self.change_records(
            lambda rows: rows[0].__setitem__("method", "unknown_transform")
        )
        with self.assertRaisesRegex(ValueError, "configuration"):
            analyze(self.recorded)


if __name__ == "__main__":
    unittest.main()

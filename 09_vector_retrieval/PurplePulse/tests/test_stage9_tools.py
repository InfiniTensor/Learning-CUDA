#!/usr/bin/env python3

import importlib.util
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


REPRODUCE = load_module("stage9_reproduce", "reproduce.py")
VALIDATE = load_module("stage9_validate", "validate_release.py")


class Stage9ToolTests(unittest.TestCase):
    def test_reproduction_plan_is_project_rooted_and_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            args = REPRODUCE.parse_args([
                "--project-root", str(root),
                "--dataset", "sift1m",
                "--skip-download",
                "--dry-run",
            ])
            state_root, steps = REPRODUCE.build_plan(args)
            names = [step["name"] for step in steps]
            self.assertEqual(
                names,
                ["configure", "build", "test", "prepare",
                 "benchmark:sift1m", "plot:sift1m", "report", "validate"],
            )
            self.assertTrue(str(state_root).startswith(str(root)))
            for step in steps:
                self.assertTrue(str(step["log"]).startswith(str(root)))

    def test_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            state = {"completed": ["configure"], "steps": {}}
            REPRODUCE.write_checkpoint(path, state)
            self.assertEqual(REPRODUCE.load_checkpoint(path), state)

    def test_csv_and_pdf_validation_helpers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "rows.csv"
            csv_path.write_text("name,status\nfixture,passed\n", encoding="utf-8")
            self.assertEqual(
                VALIDATE.validate_csv(csv_path, {"name", "status"}), 1
            )
            pdf_path = root / "report.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n" + b"0" * 2048)
            self.assertGreater(VALIDATE.validate_pdf(pdf_path), 1024)


if __name__ == "__main__":
    unittest.main()

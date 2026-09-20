"""Release checks must reject mixed source, binary and acceptance versions."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import io
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from package import verify_release, write_member, verify_output_paths


class ReleaseTests(unittest.TestCase):
    def test_existing_companion_files_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "release.zip"
            verify_output_paths(output)
            for suffix in (".zip", ".zip.sha256", ".zip.verification.json"):
                path = output.with_suffix(suffix)
                path.write_bytes(b"existing evidence")
                with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, "existing release file"):
                    verify_output_paths(output)
                self.assertEqual(path.read_bytes(), b"existing evidence")
                path.unlink()

    def test_archive_preserves_executable_permissions(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            write_member(archive, "bin/nbody", b"binary")
            write_member(archive, "README.md", b"text")
        with zipfile.ZipFile(buffer) as archive:
            for name, mode in (("bin/nbody", 0o100755), ("README.md", 0o100644)):
                info = archive.getinfo("nbody_cuda/" + name)
                self.assertEqual(info.create_system, 3)
                self.assertEqual(info.external_attr >> 16, mode)
                self.assertEqual(info.compress_type, zipfile.ZIP_DEFLATED)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "evidence").mkdir()
        (self.root / "build-wsl-review").mkdir()
        files = ["CMakeLists.txt", "generate.py", "trajectory.py", "visualize.py",
                 "src/main.cpp", "include/nbody.hpp", "tests/test_nbody.py",
                 "scripts/package.py", "scripts/report.py"]
        hashes = {}
        for name in files:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (self.root / "build-wsl-review/nbody").write_bytes(b"executable")
        self.binary_hash = hashlib.sha256(b"executable").hexdigest()
        self.artifact = {"source_sha256": hashes, "executable_sha256": self.binary_hash}
        self.write("submission-artifact.json", self.artifact)
        self.write("submission-benchmark.json", {"executable_sha256": self.binary_hash})
        self.tests = dict(tests_run=26, failures=0, errors=0, skipped=0, cuda_available=True,
                          executable_sha256=self.binary_hash)
        self.write("submission-tests.json", self.tests)

    def write(self, name, value):
        (self.root / "evidence" / name).write_text(json.dumps(value), encoding="utf-8")

    def test_matching_release(self):
        verify_release(self.root)

    def test_changed_reader(self):
        (self.root / "trajectory.py").write_bytes(b"new reader")
        with self.assertRaisesRegex(ValueError, "Source changed.*trajectory.py"):
            verify_release(self.root)

    def test_new_test_missing_from_acceptance(self):
        (self.root / "tests/new_test.py").write_bytes(b"new test")
        with self.assertRaisesRegex(ValueError, "hashes missing.*new_test.py"):
            verify_release(self.root)

    def test_changed_executable(self):
        (self.root / "build-wsl-review/nbody").write_bytes(b"new executable")
        with self.assertRaisesRegex(ValueError, "Executable changed"):
            verify_release(self.root)

    def test_changed_release_script(self):
        for name in ("package.py", "report.py"):
            with self.subTest(name=name):
                path = self.root / "scripts" / name
                path.write_bytes(b"changed script")
                with self.assertRaisesRegex(ValueError, "Source changed.*" + name):
                    verify_release(self.root)
                path.write_bytes(b"fixture")

    def test_new_script_missing_from_acceptance(self):
        (self.root / "scripts/new.py").write_bytes(b"new script")
        with self.assertRaisesRegex(ValueError, "hashes missing.*scripts/new.py"):
            verify_release(self.root)

    def test_invalid_count_types(self):
        for field in ("tests_run", "failures", "errors", "skipped"):
            for value in (True, False, 0.0, "0", None):
                with self.subTest(field=field, value=value):
                    self.write("submission-tests.json", {**self.tests, field: value})
                    with self.assertRaisesRegex(ValueError, "successful CUDA"):
                        verify_release(self.root)

    def test_wrong_benchmark(self):
        self.write("submission-benchmark.json", {"executable_sha256": "wrong"})
        with self.assertRaisesRegex(ValueError, "Benchmark belongs"):
            verify_release(self.root)

    def test_failed_skipped_or_missing_cuda_acceptance(self):
        for field, value in (("failures", 1), ("errors", 1), ("skipped", 1),
                             ("tests_run", 0), ("cuda_available", False)):
            with self.subTest(field=field):
                self.write("submission-tests.json", {**self.tests, field: value})
                with self.assertRaisesRegex(ValueError, "successful CUDA"):
                    verify_release(self.root)

    def test_acceptance_from_another_executable(self):
        for value in (None, "", "0" * 64, self.binary_hash[:-1] + "0"):
            with self.subTest(value=value):
                self.write("submission-tests.json", {**self.tests, "executable_sha256": value})
                with self.assertRaisesRegex(ValueError, "different executable"):
                    verify_release(self.root)


if __name__ == "__main__":
    unittest.main(verbosity=2)

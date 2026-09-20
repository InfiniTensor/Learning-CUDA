"""A successful profiler process does not guarantee a useful CUDA capture."""
from pathlib import Path
from contextlib import closing
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
import contextlib
import io
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_nsys import acceleration_summary, verify_report
import profile_nsys


class ProfileTests(unittest.TestCase):
    def test_failed_process_keeps_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "failure.log"
            with self.assertRaisesRegex(ValueError, "failure.log"):
                profile_nsys.run_logged([sys.executable, "-c",
                    "import sys; print('capture failed', file=sys.stderr); sys.exit(7)"], log, root)
            self.assertIn(b"capture failed", log.read_bytes())
            before = log.read_bytes()
            with self.assertRaises(FileExistsError):
                profile_nsys.run_logged([sys.executable, "-c", "pass"], log, root)
            self.assertEqual(log.read_bytes(), before)

    def test_capture_refuses_changed_input_and_preserves_existing_results(self):
        for changed in (False, True, "unequal"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                argv = ["profile_nsys.py"]
                for name in ("nsys", "executable", "input", "config"):
                    path = root / name
                    path.write_bytes(b"original")
                    argv.extend(["--" + name, str(path)])
                output = root / "capture"
                argv.extend(["--output-dir", str(output)])

                def capture(command, **kwargs):
                    kernel = command[command.index("--kernel") + 1]
                    database = output / (kernel + ".sqlite")
                    with closing(sqlite3.connect(database)) as connection:
                        connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER)")
                        connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (1)")
                        connection.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
                        connection.execute("INSERT INTO StringIds VALUES (1, ?)",
                                           ("nbody::acceleration_" + kernel + "(int)",))
                        connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
                                           "(start INTEGER, end INTEGER, demangledName INTEGER)")
                        connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (10, 30, 1)")
                        if changed == "unequal" and kernel == "naive":
                            connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (40, 60, 1)")
                        connection.commit()
                    if changed is True:
                        (root / "input").write_bytes(b"changed")

                with patch.object(sys, "argv", argv), patch.object(profile_nsys.subprocess, "run", side_effect=capture), contextlib.redirect_stdout(io.StringIO()):
                    if changed:
                        message = "sample counts differ" if changed == "unequal" else "changed during capture"
                        with self.assertRaisesRegex(ValueError, message):
                            profile_nsys.main()
                        self.assertFalse((output / "summary.json").exists())
                    else:
                        profile_nsys.main()
                        report = json.loads((output / "summary.json").read_text())
                        self.assertEqual(report["naive_over_tiled_average_ratio"], 1.0)
                        for key in ("capture_script", "analysis_script"):
                            source = report["provenance"][key]
                            self.assertEqual(source["sha256"], profile_nsys.sha256(Path(source["path"])))
                        self.assertEqual(report["provenance"]["input"]["sha256"],
                                         profile_nsys.sha256(root / "input"))
                    before = {p.name: p.read_bytes() for p in output.glob("*.sqlite")}
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        profile_nsys.main()
                    self.assertEqual(before, {p.name: p.read_bytes() for p in output.glob("*.sqlite")})

    def test_capture_contents(self):
        for runtime, kernels in ((None, None), (1, None), (1, 0), (0, 1), (2, 3)):
            with self.subTest(runtime=runtime, kernels=kernels), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "capture.sqlite"
                with closing(sqlite3.connect(path)) as connection:
                    for table, count in (("CUPTI_ACTIVITY_KIND_RUNTIME", runtime),
                                         ("CUPTI_ACTIVITY_KIND_KERNEL", kernels)):
                        if count is not None:
                            connection.execute(f'CREATE TABLE "{table}" (start INTEGER)')
                            connection.executemany(f'INSERT INTO "{table}" VALUES (?)',
                                                   [(i,) for i in range(count)])
                    connection.commit()
                before = path.read_bytes()
                if runtime and kernels:
                    self.assertEqual(verify_report(path), {
                        "CUPTI_ACTIVITY_KIND_KERNEL": 3, "CUPTI_ACTIVITY_KIND_RUNTIME": 2})
                else:
                    with self.assertRaises(ValueError):
                        verify_report(path)
                self.assertEqual(path.read_bytes(), before)

    def test_missing_file_is_not_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.sqlite"
            with self.assertRaises(sqlite3.OperationalError):
                verify_report(path)
            self.assertFalse(path.exists())

    def test_acceleration_summary_uses_demangled_kernel_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture.sqlite"
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER)")
                connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (1)")
                connection.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
                connection.executemany("INSERT INTO StringIds VALUES (?, ?)", [
                    (1, "acceleration_tiled"), (2, "other_kernel")])
                connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
                                   "(start INTEGER, end INTEGER, demangledName INTEGER)")
                connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?)", [
                    (10, 30, 1), (40, 70, 1), (80, 85, 2)])
                connection.commit()
            self.assertEqual(acceleration_summary(path), {
                "instances": 2, "total_ns": 50, "average_ns": 25.0, "median_ns": 25.0})
            with self.assertRaisesRegex(ValueError, "requested kernel"):
                acceleration_summary(path, expected_kernel="naive")
            for name in ("accelerationXnoise", "acceleration_tiled_extra", "other_acceleration_tiled"):
                with self.subTest(unrelated=name), closing(sqlite3.connect(path)) as connection:
                    connection.execute("UPDATE StringIds SET value=? WHERE id=2", (name,))
                    connection.commit()
                    self.assertEqual(acceleration_summary(path, "tiled")["instances"], 2)
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("UPDATE StringIds SET value='nbody::acceleration_naive(int)' WHERE id=2")
                connection.commit()
            with self.assertRaisesRegex(ValueError, "requested kernel"):
                acceleration_summary(path)
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("UPDATE StringIds SET value='other_kernel' WHERE id=2")
                connection.commit()
            for start, end in ((10, 10), (20, 10), (None, 10), (1.5, 10)):
                with self.subTest(start=start, end=end):
                    with closing(sqlite3.connect(path)) as connection:
                        connection.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET start=?, end=? "
                                           "WHERE demangledName=1", (start, end))
                        connection.commit()
                    with self.assertRaisesRegex(ValueError, "timestamps"):
                        acceleration_summary(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)

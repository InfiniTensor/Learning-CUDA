"""独立检查文件协议、FP16 输入、配置解析和 CLI 的失败路径。"""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import contextlib
import io
from unittest.mock import patch

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE / "tools"))
from quantize import read_config, run, format_summary

FORMAT = sys.argv.pop(1)
BINARY = Path(sys.argv.pop(1)).resolve()


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lowprecision-tests-")
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def input(self, values, dtype="fp32", name="input"):
        path = self.directory / name
        code = "f" if dtype == "fp32" else "e"
        path.write_bytes(struct.pack("<8sIQQ", b"FP32INP1" if dtype == "fp32" else b"FP16INP1", 1, 1, len(values))
                         + struct.pack("<" + code * len(values), *values))
        return path

    def run_pipeline(self, source, mode="block", rounding="nearest", output="fp32", prefix="result", ok=True):
        q = self.directory / (prefix + ".lpq")
        y = self.directory / (prefix + ".out")
        result = subprocess.run([str(BINARY), str(source), str(q), str(y), mode, rounding, output,
                                 "16" if FORMAT == "nvfp4" else "32", "42", "verify"], text=True, capture_output=True)
        if ok:
            self.assertEqual(result.returncode, 0, result.stderr)
            return q, y, json.loads(result.stdout.splitlines()[-1])
        self.assertNotEqual(result.returncode, 0)

    def test_modes_and_output_roundtrip(self):
        for dtype in ("fp32", "fp16"):
            source = self.input([(-1)**i * i * .125 for i in range(33)], dtype, dtype)
            for mode in ("block", "tensor"):
                for rounding in ("nearest", "stochastic"):
                    for output in ("fp32", "fp16", "bf16"):
                        with self.subTest(dtype=dtype, mode=mode, rounding=rounding, output=output):
                            q, y, record = self.run_pipeline(source, mode, rounding, output)
                            self.assertTrue(record["cpu_quant_match"])
                            self.assertTrue(record["cpu_dequant_match"])
                            header = struct.unpack("<8sIIQQQIII f QQ", q.read_bytes()[:72])
                            self.assertEqual(header[:3], (b"LPQUANT2", 2, 4 if FORMAT == "nvfp4" else 8))
                            self.assertEqual(header[5], 33 if mode == "tensor" else (16 if FORMAT == "nvfp4" else 32))
                            self.assertEqual(header[6:9], (int(mode == "tensor"), int(rounding == "stochastic"), 42))
                            self.assertEqual(len(y.read_bytes()), 36 + 33 * (4 if output == "fp32" else 2))
                            restored = self.directory / "restored"
                            result = subprocess.run([str(BINARY), "--dequant-file", str(q), str(restored), output], capture_output=True)
                            self.assertEqual(result.returncode, 0, result.stderr)
                            self.assertEqual(y.read_bytes(), restored.read_bytes())

    def test_fp16_means_actual_half_values(self):
        raw = self.input([.1, -.3, 1.003, -0.0, 65504], "fp16")
        expanded = self.input(list(struct.unpack("<5e", raw.read_bytes()[28:])), "fp32", "expanded")
        q1, _, _ = self.run_pipeline(raw, prefix="half")
        q2, _, _ = self.run_pipeline(expanded, prefix="float")
        self.assertEqual(q1.read_bytes(), q2.read_bytes())

    def test_empty(self):
        for mode in ("tensor", "block"):
            q, y, record = self.run_pipeline(self.input([]), mode=mode)
            self.assertEqual(q.stat().st_size, 72)
            self.assertEqual(y.stat().st_size, 36)
            self.assertEqual(record["mae"], 0)

    def test_nonfinite_input_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            self.run_pipeline(self.input([value]), ok=False)

    def test_overflow_output_logged(self):
        _, _, result = self.run_pipeline(self.input([1e30]), output="fp16")
        self.assertEqual(result["nonfinite_output"], 1)
        self.assertIsNone(result["mse"])

    def test_bad_input_headers(self):
        good = self.input([1, 2]).read_bytes()
        variants = [good[:20], good + b"x", good[:-1], b"BADMAGIC" + good[8:],
                    good[:8] + struct.pack("<I", 99) + good[12:],
                    struct.pack("<8sIQQ", b"FP32INP1", 1, 2**63, 2**63)]
        for data in variants:
            path = self.directory / "bad"; path.write_bytes(data)
            self.run_pipeline(path, ok=False)

    def test_bad_packed_headers(self):
        q, _, _ = self.run_pipeline(self.input([1, -2, 0]))
        original = q.read_bytes()
        for data in (original[:60], original + b"x", original[:-1], original[:32] + b"\x00" * 8 + original[40:]):
            q.write_bytes(data)
            result = subprocess.run([str(BINARY), "--dequant-file", str(q), str(self.directory / "badout"), "fp32"], capture_output=True)
            self.assertNotEqual(result.returncode, 0)

    def test_invalid_options(self):
        source = self.input([1])
        self.run_pipeline(source, mode="unknown", ok=False)
        self.run_pipeline(source, rounding="unknown", ok=False)
        self.run_pipeline(source, output="unknown", ok=False)
        result = subprocess.run([str(BINARY), "--benchmark", "10", "0"], capture_output=True)
        self.assertNotEqual(result.returncode, 0)

    def test_config_parser(self):
        config = self.directory / "params.toml"
        config.write_text(f'format = "{FORMAT}"\n# comment\nscale_mode="tensor"\nrounding="stochastic"\nseed=99\n')
        self.assertEqual(read_config(config)["seed"], 99)
        for extra in ('unknown=1', 'output_type="int8"', 'seed=-1', 'block_size=13'):
            config.write_text(f'format="{FORMAT}"\n' + extra)
            with self.assertRaises(ValueError):
                read_config(config)

    def test_console_summary_and_json(self):
        cfg = read_config(CORE / "configs" / (FORMAT + ".toml"))
        source = self.input([.125, -.5, 1.0])
        for raw in (False, True):
            prefix = self.directory / ("raw" if raw else "pretty")
            console = io.StringIO()
            with patch("quantize.executable", return_value=BINARY), contextlib.redirect_stdout(console):
                record = run(cfg, source, prefix, console_json=raw)
            log = Path(str(prefix) + ".json")
            self.assertEqual(json.loads(log.read_text()), record)
            if raw:
                self.assertEqual(json.loads(console.getvalue()), record)
            else:
                for label in ("[配置]", "[CPU 对照]", "[误差与压缩率]", "[单次性能（非正式基准）]", "[文件]"):
                    self.assertIn(label, console.getvalue())
                self.assertIn(str(log), console.getvalue())
            exceptional = dict(record, max_abs_error=None, mae=None, mse=None,
                               cpu_quant_match=None, cpu_dequant_match=False, nonfinite_output=1)
            rendered = format_summary(exceptional, log)
            self.assertIn("不可用", rendered)
            self.assertIn("未校验", rendered)
            self.assertIn("失败", rendered)

    def test_automatic_output_roundtrip(self):
        import quantize
        cfg = read_config(CORE / "configs" / (FORMAT + ".toml"))
        with patch.object(quantize, "PROJECT", self.directory), \
             patch("quantize.executable", return_value=BINARY), \
             contextlib.redirect_stdout(io.StringIO()):
            source = quantize.generate_automatic(1, 33, "fp16", "outlier", 42)
            first = run(cfg, source)
            second = run(cfg, source)
            stem = "fp16_" + cfg["output_type"]
            self.assertEqual(Path(first["packed"]).name, stem + ".lpq")
            self.assertEqual(Path(second["packed"]).name, stem + "_2.lpq")
            self.assertEqual(Path(first["packed"]).read_bytes(), Path(second["packed"]).read_bytes())
            self.assertEqual(first["config"], cfg)
            self.assertEqual(first["input_sha256"], quantize.input_sha256(source))
            self.assertTrue(first["cpu_quant_match"])
            self.assertTrue(first["cpu_dequant_match"])


if __name__ == "__main__":
    unittest.main()

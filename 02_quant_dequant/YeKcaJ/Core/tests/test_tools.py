"""不启动 profiler 或正式性能测试，检查迁移后的工具路径和报告生成。"""
import importlib.util
import datetime
import contextlib
import io
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

CORE = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location("core_test_" + name, CORE / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ToolTests(unittest.TestCase):
    def test_run_cli_configuration_and_overrides(self):
        tool = load_tool("quantize")
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "config.toml"
            config.write_text('format = "mxfp8"\nblock_size = 32\noutput_type = "fp16"\n')
            cases = [
                (["--format", "nvfp4", "--output-type", "bf16", "--output-dir", folder,
                  "--save", "weights", "log"], "nvfp4", 16, "bf16"),
                (["--config", str(config)], "mxfp8", 32, "fp16"),
                (["--config", str(config), "--format", "nvfp4", "--output-type", "fp32",
                  "--scale-mode", "tensor", "--rounding", "stochastic", "--seed", "42"],
                 "nvfp4", 16, "fp32"),
            ]
            for arguments, fmt, block, dtype in cases:
                with patch.object(sys, "argv", ["quantize", "run", "--input", "example.fp32"] + arguments), \
                     patch.object(tool, "run") as run:
                    tool.main()
                    cfg = run.call_args.args[0]
                    self.assertEqual((cfg["format"], cfg["block_size"], cfg["output_type"]), (fmt, block, dtype))
                    if "--save" in arguments:
                        self.assertEqual(run.call_args.kwargs["save"], ["weights", "log"])
                        self.assertEqual(run.call_args.kwargs["output_dir"], Path(folder))
                    if "--seed" in arguments:
                        self.assertEqual((cfg["scale_mode"], cfg["rounding"], cfg["seed"]), ("tensor", "stochastic", 42))
            with patch.object(sys, "argv", ["quantize", "run", "--input", "example.fp32"]), \
                 self.assertRaisesRegex(ValueError, "--format"):
                tool.main()
            with patch.object(sys, "argv", ["quantize", "run", "--input", "example.fp32",
                                          "--format", "mxfp8", "--prefix", "x", "--output-dir", folder]), \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                tool.main()

    def test_selected_artifacts_and_repeat_numbering(self):
        tool = load_tool("quantize")
        command_paths = []

        def fake_pipeline(command, **kwargs):
            packed, tensor = map(Path, command[2:4])
            packed.parent.mkdir(parents=True, exist_ok=True)
            tensor.parent.mkdir(parents=True, exist_ok=True)
            packed.write_bytes(b"weights")
            tensor.write_bytes(b"tensor")
            command_paths.append((packed, tensor))
            return subprocess.CompletedProcess(command, 0, json.dumps(dict(cpu_quant_match=True, cpu_dequant_match=True)), "")

        with tempfile.TemporaryDirectory() as folder, patch.object(tool, "executable", return_value=Path("pipeline")), \
             patch.object(tool.subprocess, "run", side_effect=fake_pipeline):
            source = Path(folder) / "input with spaces.bin"
            tool.generate(source, 1, 33, "fp16", "normal", 1234)
            original = source.read_bytes()
            cfg = tool.validate_config(dict(format="nvfp4", output_type="bf16"))
            choices = [list(c) for n in (1, 2, 3) for c in itertools.combinations(("weights", "tensor", "log"), n)]
            for i, selection in enumerate(choices + [["all"]]):
                destination = Path(folder) / str(i)
                expected = set(("weights", "tensor", "log") if selection == ["all"] else selection)
                first = tool.run(cfg, source, output_dir=destination, save=selection, quiet=True)
                self.assertEqual(len(list(destination.iterdir())), len(expected))
                for key, choice in (("packed", "weights"), ("output", "tensor"), ("log", "log")):
                    if choice in expected:
                        self.assertTrue(Path(first[key]).is_file())
                    else:
                        self.assertIsNone(first[key])
                if first["log"]:
                    self.assertEqual(json.loads(Path(first["log"]).read_text()), first)
                # 未选文件只留在临时目录，完成后不能出现失效的临时路径日志。
                for path in command_paths[-1]:
                    if path.parent != destination:
                        self.assertFalse(path.exists())
                snapshot = {p: p.read_bytes() for p in destination.iterdir()}
                second = tool.run(cfg, source, output_dir=destination, save=selection, quiet=True)
                self.assertEqual(len(list(destination.iterdir())), len(expected) * 2)
                for path, data in snapshot.items():
                    self.assertEqual(path.read_bytes(), data)
                for key in ("packed", "output", "log"):
                    if second[key]:
                        self.assertTrue(Path(second[key]).stem.endswith("_2"))
                musa = tool.directory_prefix(source, cfg, "musa", destination)
                self.assertIn("_musa_fp16_bf16", musa.name)
            self.assertEqual(source.read_bytes(), original)
            for selection in ([], ["all", "log"], ["unknown"]):
                with self.assertRaises(ValueError):
                    tool.run(cfg, source, output_dir=Path(folder) / "invalid", save=selection, quiet=True)
            prefix = Path(folder) / "existing"
            packed = prefix.with_suffix(".lpq")
            packed.write_bytes(b"original")
            with self.assertRaisesRegex(ValueError, "output exists"):
                tool.run(cfg, source, prefix=prefix, quiet=True)
            self.assertEqual(packed.read_bytes(), b"original")

    def test_evaluation_inputs_reuse_and_reject_changes(self):
        tool = load_tool("quantize")
        with tempfile.TemporaryDirectory() as folder:
            inputs, manifest = tool.prepare_evaluation_inputs(Path(folder) / "input")
            self.assertEqual(len(manifest["sha256"]), 6)
            before = {name: (inputs / name).stat().st_mtime_ns for name in manifest["sha256"]}
            self.assertEqual(tool.prepare_evaluation_inputs(inputs)[1], manifest)
            self.assertEqual(before, {name: (inputs / name).stat().st_mtime_ns for name in before})
            (inputs / "normal.fp16").unlink()
            tool.prepare_evaluation_inputs(inputs)
            self.assertEqual(tool.input_sha256(inputs / "normal.fp16"), manifest["sha256"]["normal.fp16"])
            (inputs / "uniform.fp32").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                tool.prepare_evaluation_inputs(inputs)
            self.assertEqual((inputs / "uniform.fp32").read_bytes(), b"changed")

    def test_evaluation_artifacts_and_backend_isolation(self):
        tool = load_tool("quantize")

        def fake_run(cfg, source, prefix, backend, quiet):
            prefix.parent.mkdir(parents=True, exist_ok=True)
            packed = Path(str(prefix) + ".lpq")
            output = Path(str(prefix) + "." + cfg["output_type"])
            packed.write_bytes(b"packed")
            output.write_bytes(b"tensor")
            return dict(input=str(source), packed=str(packed), output=str(output), backend=backend,
                        input_sha256=tool.input_sha256(source), config=cfg,
                        cpu_quant_match=True, cpu_dequant_match=True)

        with tempfile.TemporaryDirectory() as folder, patch.object(tool, "PROJECT", Path(folder)), \
             patch.object(tool, "executable"), patch.object(tool, "run", side_effect=fake_run), \
             contextlib.redirect_stdout(io.StringIO()):
            cuda = tool.evaluate("cuda")
            musa = tool.evaluate("musa")
            repeat = tool.evaluate("cuda")
            self.assertEqual((cuda.name, musa.name, repeat.name), ("cuda", "musa", "cuda-2"))
            for directory in (cuda, musa, repeat):
                summary = json.loads((directory / "summary.json").read_text())
                self.assertEqual(len(summary), 144)
                self.assertEqual(json.loads((directory / "status.json").read_text())["state"], "complete")
                self.assertFalse(list(directory.glob("*.fp*")))
                for distribution in ("uniform", "normal", "outlier"):
                    self.assertEqual(len(json.loads((directory / f"{distribution}.json").read_text())), 48)
                for r in summary:
                    self.assertTrue(Path(r["input"]).is_relative_to(Path(folder) / "input/evaluation-v1"))
                    for key in ("packed", "output"):
                        self.assertTrue(Path(r[key]).is_file())
                        self.assertTrue(Path(r[key]).is_relative_to(directory))
            with self.assertRaises(FileExistsError):
                tool.evaluate("cuda", cuda)

    def test_failed_evaluation_keeps_explicit_status(self):
        tool = load_tool("quantize")
        with tempfile.TemporaryDirectory() as folder, patch.object(tool, "PROJECT", Path(folder)), \
             patch.object(tool, "executable"), \
             patch.object(tool, "run", side_effect=ValueError("device failed")), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "device failed"):
                tool.evaluate("cuda")
            directory = Path(folder) / "output/evaluation-v1/cuda"
            self.assertEqual(json.loads((directory / "summary.json").read_text()), [])
            status = json.loads((directory / "status.json").read_text())
            self.assertEqual((status["state"], status["completed"], status["expected"]), ("failed", 0, 144))

    def test_comparison_rejects_changed_conditions(self):
        tool = load_tool("compare_benchmarks")
        env = dict(protocol="fixed-fp32-v1", formal=True,
                   input=dict(dtype="fp32", shape="1 x elements", elements=[1 << 20, 4 << 20, 16 << 20],
                              generator_seed=20260909, sha256=tool.INPUT_HASHES),
                   quantization=dict(scale_mode="block", rounding="nearest", seed=1234,
                                     block_size=dict(mxfp8=32, nvfp4=16)),
                   metric=dict(op="quant_optimized", scope="resident_gpu", statistic="median_ms",
                               p95="nearest-rank", output_types=["fp32", "fp16", "bf16"]),
                   warmups=3, repeats=20)
        tool.validate_pair(env, dict(env))
        for change in ({"input": {"sha256": {"1M": "different"}}}, {"repeats": 5},
                       {"formal": False}, {"metric": {"scope": "host_api"}}):
            with self.assertRaises(ValueError):
                tool.validate_pair(env, dict(env, **change))
            with self.assertRaises(ValueError):
                tool.validate_pair(dict(env, **change), dict(env, **change))

    def test_backend_selection(self):
        tool = load_tool("quantize")
        with patch.object(Path, "is_file", return_value=True):
            self.assertEqual(tool.executable("nvfp4", "musa"), CORE / "build-musa" / "pipeline_nvfp4")
            self.assertEqual(tool.executable("mxfp8"), CORE / "build" / "pipeline_mxfp8")
            with self.assertRaises(ValueError):
                tool.executable("mxfp8", "unknown")

    def test_automatic_inputs_and_result_numbering(self):
        tool = load_tool("quantize")
        with tempfile.TemporaryDirectory() as folder, patch.object(tool, "PROJECT", Path(folder)), \
             patch.object(tool.datetime, "date") as date:
            date.today.return_value = datetime.datetime(2026, 9, 10).date()
            first = tool.generate_automatic(1, 33, "fp16", "normal", 1234)
            original = first.read_bytes()
            second = tool.generate_automatic(1, 33, "fp16", "normal", 1234)
            fp32 = tool.generate_automatic(1, 33, "fp32", "normal", 1234)
            self.assertEqual(first.relative_to(folder).as_posix(), "input/910/1.fp16")
            self.assertEqual(second.name, "2.fp16")
            self.assertEqual(fp32.name, "1.fp32")
            self.assertEqual(original, second.read_bytes())
            metadata = json.loads(first.with_name(first.name + ".json").read_text())
            self.assertEqual(metadata["input_sha256"], tool.input_sha256(first))
            self.assertEqual(metadata["seed"], 1234)
            cfg = {"format": "nvfp4", "output_type": "fp32"}
            one = tool.automatic_prefix(first, cfg, "cuda")
            two = tool.automatic_prefix(first, cfg, "cuda")
            self.assertEqual(one.relative_to(folder).as_posix(),
                             "output/910/1/nvfp4/cuda/fp16_fp32")
            self.assertEqual(two.name, "fp16_fp32")
            self.assertEqual(tool.automatic_prefix(first, {"format": "mxfp8", "output_type": "fp32"}, "musa").name, "fp16_fp32")
            self.assertEqual(first.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                tool.generate(first, 1, 33, "fp16", "normal", 99)
            self.assertEqual(first.read_bytes(), original)
            with self.assertRaises(ValueError):
                tool.automatic_prefix(Path(folder) / "external.fp16", cfg)
            first.write_bytes(fp32.read_bytes())
            with self.assertRaises(ValueError):
                tool.automatic_prefix(first, cfg)

    def test_benchmark_paths_and_manifest(self):
        tool = load_tool("benchmark")
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            fmt = Path(command[0]).name.removeprefix("pipeline_")
            record = dict(format=fmt, op="quant_optimized", scope="resident_gpu",
                          elements=int(command[2]), median_ms=1, p95_ms=1, logical_GBps=1)
            return subprocess.CompletedProcess(command, 0, json.dumps(record) + "\n", "")

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "benchmark"
            with patch.object(sys, "argv", ["benchmark", "--directory", str(output)]), \
                 patch.object(tool, "fixed_inputs", return_value={str(n): "hash" for n in tool.SIZES}), \
                 patch.object(tool, "file_hash", return_value="binary-hash"), \
                 patch.object(tool.subprocess, "check_output", return_value="test environment"), \
                 patch.object(tool.subprocess, "run", side_effect=fake_run):
                tool.main()
            self.assertEqual(len(calls), 6)
            self.assertEqual({Path(c[0]).parent for c in calls}, {CORE / "build"})
            metadata = json.loads((output / "environment.json").read_text())
            for name in ("runtime/workspace.cu", "kernels/codec.cuh", "CMakeLists.txt",
                         "reference/nvfp4/main.cpp", "app/main.cpp"):
                self.assertIn(name, metadata["source_sha256"])
            self.assertEqual(len(json.loads((output / "summary.json").read_text())), 6)
            self.assertTrue((output / "RESULTS.md").is_file())

    def test_fixed_benchmark_rejects_modified_input(self):
        tool = load_tool("benchmark")
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            for n in tool.SIZES:
                (directory / f"normal_{n}.fp32").write_bytes(b"fixed corpus")
            expected = {str(n): tool.file_hash(directory / f"normal_{n}.fp32") for n in tool.SIZES}
            with patch.object(tool, "EXPECTED_INPUT_HASHES", expected):
                first = tool.fixed_inputs(CORE / "build", directory)
                self.assertEqual(tool.fixed_inputs(CORE / "build", directory), first)
                (directory / f"normal_{tool.SIZES[0]}.fp32").write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    tool.fixed_inputs(CORE / "build", directory)
            # 删除本地清单也不能把新输入冒充冻结的 v1 数据。
            (directory / "manifest.json").unlink()
            with self.assertRaises(ValueError):
                tool.fixed_inputs(CORE / "build", directory)

    def test_profile_paths_and_missing_data(self):
        tool = load_tool("profile")
        for valid in (True, False):
            with self.subTest(valid=valid), tempfile.TemporaryDirectory() as folder:
                calls = []

                def fake_run(command, **kwargs):
                    calls.append(command)
                    # 从实现导出的常量派生，测试不再各写一份 kernel 名。
                    text = " ".join(tool.DEFAULT_KERNEL.values()) if valid else "SKIPPED"
                    return subprocess.CompletedProcess(command, 0, text, "")

                output = Path(folder) / "profile"
                with patch.object(sys, "argv", ["profile", "--directory", str(output)]), \
                     patch.object(tool.subprocess, "check_output", return_value="test nsys"), \
                     patch.object(tool.subprocess, "run", side_effect=fake_run):
                    if valid:
                        tool.main()
                    else:
                        with self.assertRaises(RuntimeError):
                            tool.main()
                for call in calls:
                    if call[1] == "profile":
                        self.assertEqual(Path(call[-4]).parent, CORE / "build")
                self.assertTrue((output / "mxfp8_stats.txt").is_file())

    def test_formal_help_entrypoints(self):
        for name in ("quantize", "benchmark", "profile"):
            result = subprocess.run([sys.executable, str(CORE / "tools" / (name + ".py")),
                                     "--help"], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()

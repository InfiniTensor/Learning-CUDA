#!/usr/bin/env python3
"""遍历 configs/ 中的真实 TOML，在同进程 CUDA benchmark 上生成性能汇总。

本脚本与 run_e2e_suite.py 分工明确：前者只负责性能实验，调用
quant_dequant_bench 的同进程 warmup/repeat 循环；后者负责 QDTENSOR -> QDWGT ->
QDTENSOR 的文件级端到端验收。两者都使用提交到 configs/ 的真实配置文件，避免
性能报告与功能验收使用两套数值语义。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Final

# 复用端到端脚本已经验证过的 TOML 解析与字段约束，保证两份 suite 对 configs/ 的
# format、block_size、NVFP4 scale_mode 等判断完全一致。
from run_e2e_suite import AppConfigFile
from run_e2e_suite import _DEFAULT_CONFIG_DIRECTORY
from run_e2e_suite import _load_app_config_file
from run_e2e_suite import _nonnegative_integer
from run_e2e_suite import _positive_integer


_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
_GENERATOR_PATH: Final[Path] = _PROJECT_ROOT / "scripts" / "generate_tensor.py"
_DEFAULT_BENCHMARK_EXECUTABLE: Final[Path] = (
    _PROJECT_ROOT / "build" / "rtx4060-release" / "benchmarks" / "quant_dequant_bench"
)
_DISTRIBUTIONS: Final[tuple[str, ...]] = ("uniform", "normal", "outlier")
_INPUT_DTYPES: Final[tuple[str, ...]] = ("fp16", "fp32")
_FORMATS: Final[tuple[str, ...]] = ("mxfp8", "nvfp4")
_SCALE_MODES: Final[tuple[str, ...]] = ("block", "tensor")
_ROUNDINGS: Final[tuple[str, ...]] = ("nearest", "stochastic")
_OUTPUT_DTYPES: Final[tuple[str, ...]] = ("fp16", "bf16", "fp32")


@dataclass(frozen=True)
class BenchmarkSuiteConfig:
    """描述一次同进程 kernel 性能遍历的公共参数。"""

    executable_path: Path  # 已构建的 quant_dequant_bench 可执行文件。
    config_directory: Path  # 真实 TOML 所在目录，默认项目 configs/。
    output_directory: Path  # 输入、每项 JSON 和汇总 Markdown 的根目录。
    num_rows: int  # 基准输入矩阵行数。
    num_cols: int  # 基准输入矩阵列数；默认故意不整除 32/16。
    input_dtype: str  # 单份基准 QDTENSOR 的物理 FP16 或 FP32 dtype。
    distribution: str  # uniform、normal 或 outlier。
    seed: int  # 生成输入时使用的确定性随机种子。
    num_warmups: int  # 每个 config 的不计入统计同进程 warmup 数。
    num_repeats: int  # 每个 config 的正式同进程 repeat 数。
    app_configs: tuple[AppConfigFile, ...]  # 经 CLI 筛选后实际执行的真实 TOML。


@dataclass(frozen=True)
class TimingSummary:
    """保存 benchmark JSON 中一阶段 kernel 的已验证统计结果。"""

    min_ms: float  # 正式 repeat 的最小 CUDA Event 时间。
    mean_ms: float  # 正式 repeat 的算术均值。
    median_ms: float  # 正式 repeat 的中位数。
    p95_ms: float  # nearest-rank p95。
    max_ms: float  # 正式 repeat 的最大值。
    effective_bandwidth_gbps: float | None  # 以 mean_ms 换算的逻辑有效带宽。


@dataclass(frozen=True)
class BenchmarkResult:
    """保存一份已交叉校验的 benchmark JSON，供汇总 Markdown 使用。"""

    config: AppConfigFile  # 本项实际使用的 configs/ TOML。
    report_path: Path  # 本项 benchmark 写出的 JSON 文件。
    quant: TimingSummary  # 量化阶段统计。
    dequant: TimingSummary  # 反量化阶段统计。


def _run_command(command: list[str], description: str) -> subprocess.CompletedProcess[str]:
    """在项目根目录运行命令，失败时保留完整 stdout/stderr。"""
    completed_process = subprocess.run(
        command,
        cwd=_PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed_process.returncode != 0:
        raise RuntimeError(
            f"{description} 失败，退出码 {completed_process.returncode}。\n"
            f"命令：{' '.join(command)}\n"
            f"stdout：\n{completed_process.stdout}\n"
            f"stderr：\n{completed_process.stderr}"
        )
    return completed_process


def _discover_configs(arguments: argparse.Namespace) -> tuple[AppConfigFile, ...]:
    """读取 configs/ 中真实 TOML，并按格式与输出类型筛选性能实验组合。"""
    config_directory = arguments.config_dir.resolve()
    if not config_directory.is_dir():
        raise RuntimeError(f"配置目录不存在：{config_directory}")

    requested_names = set(arguments.config_names)
    config_files = sorted(config_directory.glob("*.toml"))
    discovered_configs = tuple(_load_app_config_file(config_path) for config_path in config_files)
    discovered_names = {config.name for config in discovered_configs}
    unknown_names = requested_names.difference(discovered_names)
    if unknown_names:
        raise RuntimeError(f"--config-names 含不存在配置：{', '.join(sorted(unknown_names))}")

    selected_configs = tuple(
        config
        for config in discovered_configs
        if (not requested_names or config.name in requested_names)
        and config.quant_format in arguments.formats
        and config.scale_mode in arguments.scale_modes
        and config.rounding in arguments.roundings
        and config.output_dtype in arguments.output_dtypes
    )
    if not selected_configs:
        raise RuntimeError("筛选后没有可执行的 benchmark TOML。")
    return selected_configs


def _generate_input(suite: BenchmarkSuiteConfig) -> Path:
    """生成本次 suite 共享的一份确定性 QDTENSOR 输入并检查生成器 JSON。"""
    input_path = suite.output_directory / "inputs" / (
        f"{suite.distribution}_{suite.input_dtype}_"
        f"{suite.num_rows}x{suite.num_cols}_seed{suite.seed}.qdtensor"
    )
    completed_process = _run_command(
        [
            sys.executable,
            str(_GENERATOR_PATH),
            "--output",
            str(input_path),
            "--rows",
            str(suite.num_rows),
            "--cols",
            str(suite.num_cols),
            "--dtype",
            suite.input_dtype,
            "--distribution",
            suite.distribution,
            "--seed",
            str(suite.seed),
        ],
        "生成 benchmark 输入",
    )
    try:
        generator_report = json.loads(completed_process.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("生成器没有输出合法 JSON。") from error

    expected_fields = (
        suite.num_rows,
        suite.num_cols,
        suite.input_dtype,
        suite.distribution,
        suite.seed,
    )
    actual_fields = (
        generator_report.get("rows"),
        generator_report.get("cols"),
        generator_report.get("dtype"),
        generator_report.get("distribution"),
        generator_report.get("seed"),
    )
    if actual_fields != expected_fields:
        raise RuntimeError("生成器 JSON 与 benchmark 请求的 shape/dtype/分布/seed 不一致。")
    if generator_report.get("file_bytes") != input_path.stat().st_size:
        raise RuntimeError(f"生成器报告的文件大小与实际输入不一致：{input_path}")
    return input_path


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    """验证 JSON object 字段，避免后续 KeyError 给出不清晰诊断。"""
    if not isinstance(value, dict):
        raise RuntimeError(f"benchmark JSON 字段 {field_name} 必须是 object。")
    return value


def _require_finite_nonnegative(value: Any, field_name: str) -> float:
    """验证 JSON 数值是有限非负浮点数，并统一转换为 float。"""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError(f"benchmark JSON 字段 {field_name} 必须是数值。")
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value < 0.0:
        raise RuntimeError(f"benchmark JSON 字段 {field_name} 必须是有限非负数。")
    return numeric_value


def _parse_timing_summary(stage: str, value: Any) -> TimingSummary:
    """读取一阶段 benchmark 时间统计，并验证排序关系和可选带宽。"""
    timing = _require_mapping(value, f"performance.{stage}")
    min_ms = _require_finite_nonnegative(timing.get("min_ms"), f"{stage}.min_ms")
    mean_ms = _require_finite_nonnegative(timing.get("mean_ms"), f"{stage}.mean_ms")
    median_ms = _require_finite_nonnegative(timing.get("median_ms"), f"{stage}.median_ms")
    p95_ms = _require_finite_nonnegative(timing.get("p95_ms"), f"{stage}.p95_ms")
    max_ms = _require_finite_nonnegative(timing.get("max_ms"), f"{stage}.max_ms")
    if not min_ms <= median_ms <= p95_ms <= max_ms:
        raise RuntimeError(f"benchmark JSON 的 {stage} 时间分位数顺序不合法。")
    bandwidth_value = timing.get("effective_bandwidth_gbps_from_mean")
    if mean_ms == 0.0:
        if bandwidth_value is not None:
            raise RuntimeError(f"{stage} mean_ms 为 0 时带宽必须为 null。")
        effective_bandwidth_gbps: float | None = None
    else:
        effective_bandwidth_gbps = _require_finite_nonnegative(
            bandwidth_value,
            f"{stage}.effective_bandwidth_gbps_from_mean",
        )
        if effective_bandwidth_gbps <= 0.0:
            raise RuntimeError(f"{stage} 正时间对应的有效带宽必须为正数。")
    return TimingSummary(
        min_ms=min_ms,
        mean_ms=mean_ms,
        median_ms=median_ms,
        p95_ms=p95_ms,
        max_ms=max_ms,
        effective_bandwidth_gbps=effective_bandwidth_gbps,
    )


def _validate_benchmark_report(
    report_path: Path,
    suite: BenchmarkSuiteConfig,
    app_config: AppConfigFile,
) -> BenchmarkResult:
    """读取一份 benchmark JSON，并交叉验证输入、配置、循环次数与统计字段。"""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"无法读取合法 benchmark JSON：{report_path}") from error

    input_info = _require_mapping(report.get("input"), "input")
    config_info = _require_mapping(report.get("config"), "config")
    benchmark_info = _require_mapping(report.get("benchmark"), "benchmark")
    performance_info = _require_mapping(report.get("performance"), "performance")
    if report.get("schema_version") != 1:
        raise RuntimeError(f"{report_path} 的 schema_version 不是 1。")
    if (input_info.get("rows"), input_info.get("cols"), input_info.get("dtype")) != (
        suite.num_rows,
        suite.num_cols,
        suite.input_dtype,
    ):
        raise RuntimeError(f"{report_path} 的输入 shape/dtype 与请求不一致。")
    if (
        config_info.get("format"),
        config_info.get("block_size"),
        config_info.get("scale_mode"),
        config_info.get("rounding"),
        config_info.get("output_type"),
        config_info.get("target_gpu"),
    ) != (
        app_config.quant_format,
        app_config.block_size,
        app_config.scale_mode,
        app_config.rounding,
        app_config.output_dtype,
        app_config.target_gpu,
    ):
        raise RuntimeError(f"{report_path} 的 config 与 {app_config.path} 不一致。")
    if (benchmark_info.get("warmups"), benchmark_info.get("repeats")) != (
        suite.num_warmups,
        suite.num_repeats,
    ):
        raise RuntimeError(f"{report_path} 的 warmups/repeats 与命令行不一致。")
    timing_scope = benchmark_info.get("timing_scope")
    if not isinstance(timing_scope, str) or "CUDA Event" not in timing_scope:
        raise RuntimeError(f"{report_path} 缺少可识别的 CUDA Event 计时边界。")

    return BenchmarkResult(
        config=app_config,
        report_path=report_path,
        quant=_parse_timing_summary("quant", performance_info.get("quant")),
        dequant=_parse_timing_summary("dequant", performance_info.get("dequant")),
    )


def _run_benchmark_once(
    suite: BenchmarkSuiteConfig,
    input_path: Path,
    app_config: AppConfigFile,
) -> BenchmarkResult:
    """调用一次同进程 benchmark，并对写出的 JSON 做结构化校验。"""
    report_path = suite.output_directory / "runs" / app_config.name / "benchmark.json"
    _run_command(
        [
            str(suite.executable_path),
            "--input",
            str(input_path),
            "--config",
            str(app_config.path),
            "--warmups",
            str(suite.num_warmups),
            "--repeats",
            str(suite.num_repeats),
            "--output",
            str(report_path),
        ],
        f"运行 {app_config.name} benchmark",
    )
    return _validate_benchmark_report(report_path, suite, app_config)


def _format_optional(value: float | None, digits: int = 3) -> str:
    """将可选性能数值转为 Markdown 表格单元格。"""
    return "null" if value is None else f"{value:.{digits}f}"


def _write_summary(summary_path: Path, suite: BenchmarkSuiteConfig, results: list[BenchmarkResult]) -> None:
    """写出便于提交/比较的 benchmark 汇总 Markdown，不覆盖单项原始 JSON。"""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 同进程 CUDA kernel benchmark 汇总",
        "",
        "该表由 `quant_dequant_bench` 产生：每份 TOML 在同一个进程中读取一次输入和配置，",
        "随后执行 warmup 与 repeat。CUDA Event 只统计格式专用 kernel，不含 H2D、D2H、",
        "device 分配、文件 I/O 或 CUDA context 初始化。",
        "",
        f"- 输入：`{suite.num_rows} × {suite.num_cols}`、`{suite.input_dtype}`、"
        f"`{suite.distribution}`、seed `{suite.seed}`。",
        f"- 每项：`warmups = {suite.num_warmups}`、`repeats = {suite.num_repeats}`。",
        f"- 配置数：{len(results)}；每项原始 JSON 位于 `runs/<config>/benchmark.json`。",
        "",
        "| 配置 | quant mean ms | quant median ms | quant p95 ms | quant GB/s | dequant mean ms | dequant median ms | dequant p95 ms | dequant GB/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            "| "
            f"{result.config.name} | {result.quant.mean_ms:.3f} | "
            f"{result.quant.median_ms:.3f} | {result.quant.p95_ms:.3f} | "
            f"{_format_optional(result.quant.effective_bandwidth_gbps)} | "
            f"{result.dequant.mean_ms:.3f} | {result.dequant.median_ms:.3f} | "
            f"{result.dequant.p95_ms:.3f} | "
            f"{_format_optional(result.dequant.effective_bandwidth_gbps)} |"
        )

    quant_means = [result.quant.mean_ms for result in results]
    dequant_means = [result.dequant.mean_ms for result in results]
    lines.extend(
        [
            "",
            "## 跨配置观察",
            "",
            f"- quant mean 的范围是 `{min(quant_means):.3f}--{max(quant_means):.3f} ms`，"
            f"中位配置均值为 `{statistics.median(quant_means):.3f} ms`。",
            f"- dequant mean 的范围是 `{min(dequant_means):.3f}--{max(dequant_means):.3f} ms`，"
            f"中位配置均值为 `{statistics.median(dequant_means):.3f} ms`。",
            "- 不同格式、scale mode 与 rounding 的 kernel 数量不同；该表只比较固定 shape 下的"
            "实际实现时间，不能直接等同于 GPU 硬件峰值带宽。",
            "",
        ]
    )
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def _build_argument_parser() -> argparse.ArgumentParser:
    """构造性能 suite CLI；默认筛选出没有输出 dtype 重复的六份 FP32 TOML。"""
    parser = argparse.ArgumentParser(
        description="遍历 configs/，调用同进程 quant_dequant_bench 并汇总 kernel 性能。",
    )
    parser.add_argument("--executable", type=Path, default=_DEFAULT_BENCHMARK_EXECUTABLE)
    parser.add_argument("--config-dir", type=Path, default=_DEFAULT_CONFIG_DIRECTORY)
    parser.add_argument("--config-names", nargs="*", default=())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "outputs" / "benchmark_suite",
    )
    parser.add_argument("--rows", type=_positive_integer, default=4096)
    parser.add_argument("--cols", type=_positive_integer, default=4097)
    parser.add_argument("--input-dtype", choices=_INPUT_DTYPES, default="fp32")
    parser.add_argument("--distribution", choices=_DISTRIBUTIONS, default="normal")
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--warmups", type=_nonnegative_integer, default=10)
    parser.add_argument("--repeats", type=_positive_integer, default=30)
    parser.add_argument("--formats", nargs="+", choices=_FORMATS, default=_FORMATS)
    parser.add_argument("--scale-modes", nargs="+", choices=_SCALE_MODES, default=_SCALE_MODES)
    parser.add_argument("--roundings", nargs="+", choices=_ROUNDINGS, default=_ROUNDINGS)
    parser.add_argument(
        "--output-dtypes",
        nargs="+",
        choices=_OUTPUT_DTYPES,
        default=("fp32",),
        help=(
            "默认只测 fp32 输出，避免三种输出 TOML 重复测同一个 kernel；"
            "若需完整 18 份 TOML，传 --output-dtypes fp16 bf16 fp32。"
        ),
    )
    return parser


def main() -> int:
    """生成共享输入，执行所有筛选后的真实 TOML benchmark，并写汇总 Markdown。"""
    arguments = _build_argument_parser().parse_args()
    try:
        app_configs = _discover_configs(arguments)
        suite = BenchmarkSuiteConfig(
            executable_path=arguments.executable.resolve(),
            config_directory=arguments.config_dir.resolve(),
            output_directory=arguments.output_dir.resolve(),
            num_rows=arguments.rows,
            num_cols=arguments.cols,
            input_dtype=arguments.input_dtype,
            distribution=arguments.distribution,
            seed=arguments.seed,
            num_warmups=arguments.warmups,
            num_repeats=arguments.repeats,
            app_configs=app_configs,
        )
        if not suite.executable_path.is_file():
            raise RuntimeError(f"找不到 benchmark 可执行文件：{suite.executable_path}")
        if not _GENERATOR_PATH.is_file():
            raise RuntimeError(f"找不到输入生成器：{_GENERATOR_PATH}")

        suite.output_directory.mkdir(parents=True, exist_ok=True)
        input_path = _generate_input(suite)
        results = [
            _run_benchmark_once(suite, input_path, app_config)
            for app_config in suite.app_configs
        ]
        summary_path = suite.output_directory / "summary.md"
        _write_summary(summary_path, suite, results)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"run_benchmark_suite 失败：{error}")
        return 1

    print(
        "同进程 benchmark 配置遍历通过："
        f"{len(results)} 份 TOML，"
        f"warmups={suite.num_warmups}，repeats={suite.num_repeats}。"
    )
    print(f"汇总报告：{summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

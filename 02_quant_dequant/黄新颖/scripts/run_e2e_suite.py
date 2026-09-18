#!/usr/bin/env python3
"""遍历真实 TOML 配置，运行完整 CUDA app，并校验每次产物与 JSON 报告。

默认组合覆盖三种输入分布、FP16/FP32 输入以及 configs/ 中的全部合法 app 配置。
脚本是端到端验收与实验工具；codec、单元素位编码和 CPU/GPU 逐元素对照仍由 CTest 负责。
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


_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
_GENERATOR_PATH: Final[Path] = _PROJECT_ROOT / "scripts" / "generate_tensor.py"
_DEFAULT_CONFIG_DIRECTORY: Final[Path] = _PROJECT_ROOT / "configs"
_FORMAT_TO_BLOCK_SIZE: Final[dict[str, int]] = {"mxfp8": 32, "nvfp4": 16}
_OUTPUT_DTYPE_BYTES: Final[dict[str, int]] = {"fp16": 2, "bf16": 2, "fp32": 4}


@dataclass(frozen=True)
class AppConfigFile:
    """保存从一份真实 app TOML 读取且已通过脚本侧语义校验的字段。"""

    name: str  # 配置文件 stem，同时用于稳定的输出目录名称。
    path: Path  # 实际传递给 quant_dequant --config 的 TOML 路径。
    quant_format: str  # mxfp8 或 nvfp4。
    block_size: int  # 格式固定的 32 或 16。
    scale_mode: str  # MXFP8 的 block/tensor，或 NVFP4 的 block。
    rounding: str  # nearest 或 stochastic。
    stochastic_seed: int  # nearest 固定为 0；stochastic 为配置中显式种子。
    output_dtype: str  # fp16、bf16 或 fp32。
    target_gpu: str  # JSON 中应原样记录的报告元数据。


@dataclass(frozen=True)
class SuiteConfig:
    """描述一次端到端组合遍历的公共运行参数。"""

    executable_path: Path  # 已构建的 quant_dequant app。
    output_directory: Path  # 本次输入、产物和汇总文件的根目录。
    num_rows: int  # 每个生成矩阵的行数。
    num_cols: int  # 每个生成矩阵的列数。
    input_dtypes: tuple[str, ...]  # 本次要生成的 FP16/FP32 物理输入类型。
    distributions: tuple[str, ...]  # uniform、normal、outlier 的子集。
    app_configs: tuple[AppConfigFile, ...]  # 已按 CLI 过滤后的真实 TOML 配置。
    base_seed: int  # 按 distribution/dtype 派生稳定 seed 的基础值。
    num_warmups: int  # 每个组合不计入汇总的完整 app 调用次数。
    num_repeats: int  # 每个组合计入汇总的完整 app 调用次数。


@dataclass(frozen=True)
class RunResult:
    """保存一轮通过报告和文件布局校验的正式 app 运行摘要。"""

    input_dtype: str  # 本轮输入 QDTENSOR 的物理 dtype。
    distribution: str  # 本轮输入分布。
    config_name: str  # 真实 TOML 配置名称。
    repeat_index: int  # 当前组合中的正式重复编号。
    report_path: Path  # 已验证 report.json 的路径。
    quant_kernel_ms: float  # CUDA Event 量化 kernel 时间。
    dequant_kernel_ms: float  # CUDA Event 反量化 kernel 时间。
    max_abs: float  # 最终物理输出文件的最大绝对误差。
    mae: float  # 最终物理输出文件的平均绝对误差。
    mse: float  # 最终物理输出文件的均方误差。


def _positive_integer(value: str) -> int:
    """供 argparse 使用：解析并验证正整数。"""
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("必须是正整数。")
    return parsed_value


def _nonnegative_integer(value: str) -> int:
    """供 argparse 使用：解析并验证零或正整数。"""
    parsed_value = int(value)
    if parsed_value < 0:
        raise argparse.ArgumentTypeError("必须是零或正整数。")
    return parsed_value


def _run_command(command: list[str], description: str) -> subprocess.CompletedProcess[str]:
    """执行子进程；失败时保留命令、stdout 和 stderr 以便定位。"""
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


def _parse_flat_config(config_path: Path) -> dict[str, str]:
    """读取本项目受限的 `key = value` TOML 子集，供 suite 推导期望布局。

    app 仍会调用 C++ `load_app_config()` 做权威语法和语义校验；这里仅解析项目
    提交的样例配置，以便在读取 JSON 时知道本轮应出现的 format、scale 和 dtype。
    """
    fields: dict[str, str] = {}
    try:
        config_lines = config_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError(f"无法读取配置文件：{config_path}") from error

    for line_number, raw_line in enumerate(config_lines, start=1):
        # 提交的样例不在字符串中使用 #；保守删除注释即可避免把说明当作值。
        content = raw_line.split("#", maxsplit=1)[0].strip()
        if not content:
            continue
        if "=" not in content:
            raise RuntimeError(f"{config_path}:{line_number} 不是 key = value。")
        key, raw_value = (part.strip() for part in content.split("=", maxsplit=1))
        if not key or not raw_value or key in fields:
            raise RuntimeError(f"{config_path}:{line_number} 含有空或重复配置字段。")
        # 项目配置允许整体双引号字符串；本脚本不需要支持复杂转义，因为 configs/
        # 中的样例固定使用简单标量。
        if raw_value.startswith('"') and raw_value.endswith('"') and len(raw_value) >= 2:
            raw_value = raw_value[1:-1]
        fields[key] = raw_value
    return fields


def _load_app_config_file(config_path: Path) -> AppConfigFile:
    """将一个真实 TOML 转为已验证的 AppConfigFile。"""
    fields = _parse_flat_config(config_path)
    required_fields = {
        "format",
        "block_size",
        "scale_mode",
        "rounding",
        "output_type",
        "target_gpu",
    }
    missing_fields = sorted(required_fields.difference(fields))
    if missing_fields:
        raise RuntimeError(f"{config_path} 缺少字段：{', '.join(missing_fields)}")

    quant_format = fields["format"]
    scale_mode = fields["scale_mode"]
    rounding = fields["rounding"]
    output_dtype = fields["output_type"]
    target_gpu = fields["target_gpu"]
    try:
        block_size = int(fields["block_size"])
        stochastic_seed = int(fields.get("stochastic_seed", "0"))
    except ValueError as error:
        raise RuntimeError(f"{config_path} 的 block_size 或 stochastic_seed 不是整数。") from error

    if quant_format not in _FORMAT_TO_BLOCK_SIZE:
        raise RuntimeError(f"{config_path} 的 format 不受支持：{quant_format}")
    if block_size != _FORMAT_TO_BLOCK_SIZE[quant_format]:
        raise RuntimeError(f"{config_path} 的 block_size 与 format 不匹配。")
    if scale_mode not in {"block", "tensor"}:
        raise RuntimeError(f"{config_path} 的 scale_mode 不受支持：{scale_mode}")
    if quant_format == "nvfp4" and scale_mode != "block":
        raise RuntimeError(f"{config_path}：NVFP4 只能使用 block scale。")
    if rounding not in {"nearest", "stochastic"}:
        raise RuntimeError(f"{config_path} 的 rounding 不受支持：{rounding}")
    if rounding == "nearest" and stochastic_seed != 0:
        raise RuntimeError(f"{config_path}：nearest 配置的 stochastic_seed 必须为 0。")
    if rounding == "stochastic" and "stochastic_seed" not in fields:
        raise RuntimeError(f"{config_path}：stochastic 配置必须显式提供 stochastic_seed。")
    if output_dtype not in _OUTPUT_DTYPE_BYTES:
        raise RuntimeError(f"{config_path} 的 output_type 不受支持：{output_dtype}")
    if not target_gpu:
        raise RuntimeError(f"{config_path} 的 target_gpu 不能为空。")

    return AppConfigFile(
        name=config_path.stem,
        path=config_path,
        quant_format=quant_format,
        block_size=block_size,
        scale_mode=scale_mode,
        rounding=rounding,
        stochastic_seed=stochastic_seed,
        output_dtype=output_dtype,
        target_gpu=target_gpu,
    )


def _discover_and_filter_configs(arguments: argparse.Namespace) -> tuple[AppConfigFile, ...]:
    """扫描 configs/ 中的 TOML，并按 format/scale/rounding/output 筛选组合。"""
    config_directory = arguments.config_dir.resolve()
    if not config_directory.is_dir():
        raise RuntimeError(f"配置目录不存在：{config_directory}")

    requested_names = set(arguments.config_names)
    config_files = sorted(config_directory.glob("*.toml"))
    discovered_configs = [_load_app_config_file(config_path) for config_path in config_files]
    selected_configs = tuple(
        config
        for config in discovered_configs
        if (not requested_names or config.name in requested_names)
        and config.quant_format in arguments.formats
        and config.scale_mode in arguments.scale_modes
        and config.rounding in arguments.roundings
        and config.output_dtype in arguments.output_dtypes
    )
    unknown_names = requested_names.difference(config.name for config in discovered_configs)
    if unknown_names:
        raise RuntimeError(f"--config-names 含不存在配置：{', '.join(sorted(unknown_names))}")
    if not selected_configs:
        raise RuntimeError("筛选后没有任何可运行的 TOML 配置。")
    return selected_configs


def _generate_input(
    suite: SuiteConfig,
    input_dtype: str,
    distribution: str,
    distribution_index: int,
) -> Path:
    """调用生成器，写入一个 distribution/dtype 唯一且可复现的输入 QDTENSOR。"""
    input_path = suite.output_directory / "inputs" / (
        f"{distribution}_{input_dtype}_{suite.num_rows}x{suite.num_cols}.qdtensor"
    )
    # dtype 偏移避免 FP16 和 FP32 输入意外使用同一 random stream 标识。
    dtype_offset = 0 if input_dtype == "fp16" else 1
    derived_seed = suite.base_seed + distribution_index * 1009 + dtype_offset * 100003
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
            input_dtype,
            "--distribution",
            distribution,
            "--seed",
            str(derived_seed),
        ],
        f"生成 {distribution}/{input_dtype} 输入",
    )
    try:
        generator_report = json.loads(completed_process.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"生成器没有输出合法 JSON：{completed_process.stdout}") from error
    if generator_report.get("file_bytes") != input_path.stat().st_size:
        raise RuntimeError(f"生成器报告的文件大小与实际输入不一致：{input_path}")
    return input_path


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    """验证 JSON 字段是 object，给出比 KeyError 更清楚的报告错误。"""
    if not isinstance(value, dict):
        raise RuntimeError(f"报告字段 {field_name} 必须是 JSON object。")
    return value


def _require_finite_nonnegative(value: Any, field_name: str) -> float:
    """验证 JSON 数值为有限非负数。"""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError(f"报告字段 {field_name} 必须是数值。")
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value < 0.0:
        raise RuntimeError(f"报告字段 {field_name} 必须是有限非负数。")
    return numeric_value


def _validate_optional_bandwidth(kernel_ms: float, bandwidth: Any, field_name: str) -> None:
    """验证 CUDA Event 0 ms 对应 null 带宽，正时间对应有限正带宽。"""
    if kernel_ms == 0.0:
        if bandwidth is not None:
            raise RuntimeError(f"{field_name} 的 kernel 时间为 0 ms 时带宽必须为 null。")
        return
    bandwidth_value = _require_finite_nonnegative(bandwidth, field_name)
    if bandwidth_value <= 0.0:
        raise RuntimeError(f"{field_name} 在正 kernel 时间下必须为正数。")


def _validate_report(
    report_path: Path,
    input_path: Path,
    quantized_path: Path,
    dequantized_path: Path,
    suite: SuiteConfig,
    input_dtype: str,
    distribution: str,
    app_config: AppConfigFile,
) -> RunResult:
    """交叉检查一次 app JSON 与实际 QDTENSOR/QDWGT 文件布局。"""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"无法读取合法 JSON 报告：{report_path}") from error

    input_info = _require_mapping(report.get("input"), "input")
    config_info = _require_mapping(report.get("config"), "config")
    artifacts = _require_mapping(report.get("artifacts"), "artifacts")
    error_info = _require_mapping(report.get("error"), "error")
    compression = _require_mapping(report.get("compression"), "compression")
    performance = _require_mapping(report.get("performance"), "performance")

    if report.get("schema_version") != 1:
        raise RuntimeError(f"{report_path} 的 schema_version 不是 1。")
    if (input_info.get("rows"), input_info.get("cols"), input_info.get("dtype")) != (
        suite.num_rows,
        suite.num_cols,
        input_dtype,
    ):
        raise RuntimeError(f"{report_path} 的输入形状或 dtype 与生成器不一致。")
    # 该字段在报告 schema v1 的当前实现中用于记录随机舍入的可复现状态。缺失通常
    # 表明用户只更新了脚本、却仍在运行修改 metrics.cpp 前构建的旧 app。
    if "stochastic_seed" not in config_info:
        raise RuntimeError(
            f"{report_path} 缺少 config.stochastic_seed；"
            "当前 app 可执行文件可能过旧，请重新构建所选 CMake preset。"
        )
    expected_config_fields = (
        app_config.quant_format,
        app_config.block_size,
        app_config.scale_mode,
        app_config.rounding,
        app_config.stochastic_seed,
        app_config.output_dtype,
        app_config.target_gpu,
    )
    reported_config_fields = (
        config_info.get("format"),
        config_info.get("block_size"),
        config_info.get("scale_mode"),
        config_info.get("rounding"),
        config_info.get("stochastic_seed"),
        config_info.get("output_type"),
        config_info.get("target_gpu"),
    )
    if reported_config_fields != expected_config_fields:
        raise RuntimeError(f"{report_path} 的 config 字段与 {app_config.path} 不一致。")

    num_elements = suite.num_rows * suite.num_cols
    blocks_per_row = (suite.num_cols + app_config.block_size - 1) // app_config.block_size
    expected_local_scale_bytes = (
        1
        if app_config.quant_format == "mxfp8" and app_config.scale_mode == "tensor"
        else suite.num_rows * blocks_per_row
    )
    expected_payload_bytes = (
        num_elements
        if app_config.quant_format == "mxfp8"
        else (num_elements + 1) // 2
    )
    expected_global_scale_bytes = 0 if app_config.quant_format == "mxfp8" else 4
    expected_input_payload_bytes = num_elements * (2 if input_dtype == "fp16" else 4)
    expected_dequantized_file_bytes = (
        64 + num_elements * _OUTPUT_DTYPE_BYTES[app_config.output_dtype]
    )
    expected_logical_bytes = (
        expected_payload_bytes + expected_local_scale_bytes + expected_global_scale_bytes
    )

    # 实际文件大小与 JSON 声明必须一致，避免只验证 app 内存中的中间对象。
    if artifacts.get("payload_bytes") != expected_payload_bytes:
        raise RuntimeError(f"{report_path} 的 payload_bytes 不符合 {app_config.quant_format}。")
    if artifacts.get("local_scale_bytes") != expected_local_scale_bytes:
        raise RuntimeError(f"{report_path} 的 local_scale_bytes 不符合 scale_mode。")
    if artifacts.get("global_scale_bytes") != expected_global_scale_bytes:
        raise RuntimeError(f"{report_path} 的 global_scale_bytes 不符合格式。")
    if artifacts.get("quantized_file_bytes") != quantized_path.stat().st_size:
        raise RuntimeError(f"{report_path} 的 quantized_file_bytes 与真实 QDWGT 不一致。")
    if artifacts.get("dequantized_file_bytes") != dequantized_path.stat().st_size:
        raise RuntimeError(f"{report_path} 的 dequantized_file_bytes 与真实文件不一致。")
    if dequantized_path.stat().st_size != expected_dequantized_file_bytes:
        raise RuntimeError(f"反量化输出大小不符合 output_type：{dequantized_path}")
    if input_path.stat().st_size != 64 + expected_input_payload_bytes:
        raise RuntimeError(f"输入 QDTENSOR 大小不符合生成规格：{input_path}")

    if compression.get("input_payload_bytes") != expected_input_payload_bytes:
        raise RuntimeError(f"{report_path} 的 input_payload_bytes 不正确。")
    if compression.get("logical_quantized_bytes") != expected_logical_bytes:
        raise RuntimeError(f"{report_path} 的 logical_quantized_bytes 不正确。")
    logical_ratio = _require_finite_nonnegative(
        compression.get("logical_compression_ratio"),
        "compression.logical_compression_ratio",
    )
    on_disk_ratio = _require_finite_nonnegative(
        compression.get("on_disk_compression_ratio"),
        "compression.on_disk_compression_ratio",
    )
    if logical_ratio <= 0.0 or on_disk_ratio <= 0.0:
        raise RuntimeError(f"{report_path} 的压缩率必须为正数。")
    if not math.isclose(
        logical_ratio,
        expected_input_payload_bytes / expected_logical_bytes,
        rel_tol=1.0e-12,
    ):
        raise RuntimeError(f"{report_path} 的 logical_compression_ratio 不正确。")
    if not math.isclose(
        on_disk_ratio,
        expected_input_payload_bytes / quantized_path.stat().st_size,
        rel_tol=1.0e-12,
    ):
        raise RuntimeError(f"{report_path} 的 on_disk_compression_ratio 不正确。")

    max_abs = _require_finite_nonnegative(error_info.get("max_abs"), "error.max_abs")
    mae = _require_finite_nonnegative(error_info.get("mae"), "error.mae")
    mse = _require_finite_nonnegative(error_info.get("mse"), "error.mse")
    quant_kernel_ms = _require_finite_nonnegative(
        performance.get("quant_kernel_ms"),
        "performance.quant_kernel_ms",
    )
    dequant_kernel_ms = _require_finite_nonnegative(
        performance.get("dequant_kernel_ms"),
        "performance.dequant_kernel_ms",
    )
    _validate_optional_bandwidth(
        quant_kernel_ms,
        performance.get("quant_effective_bandwidth_gbps"),
        "performance.quant_effective_bandwidth_gbps",
    )
    _validate_optional_bandwidth(
        dequant_kernel_ms,
        performance.get("dequant_effective_bandwidth_gbps"),
        "performance.dequant_effective_bandwidth_gbps",
    )

    return RunResult(
        input_dtype=input_dtype,
        distribution=distribution,
        config_name=app_config.name,
        repeat_index=0,
        report_path=report_path,
        quant_kernel_ms=quant_kernel_ms,
        dequant_kernel_ms=dequant_kernel_ms,
        max_abs=max_abs,
        mae=mae,
        mse=mse,
    )


def _run_app_once(
    suite: SuiteConfig,
    input_path: Path,
    input_dtype: str,
    distribution: str,
    app_config: AppConfigFile,
    phase: str,
    index: int,
) -> RunResult:
    """运行一次真实 app，随后验证 QDWGT、输出 QDTENSOR 与 JSON。"""
    run_directory = (
        suite.output_directory
        / "runs"
        / input_dtype
        / distribution
        / app_config.name
        / f"{phase}_{index:02d}"
    )
    quantized_path = run_directory / "weights.qdwgt"
    dequantized_path = run_directory / "dequantized.qdtensor"
    report_path = run_directory / "report.json"
    run_directory.mkdir(parents=True, exist_ok=True)

    _run_command(
        [
            str(suite.executable_path),
            "--input",
            str(input_path),
            "--config",
            str(app_config.path),
            "--quantized-output",
            str(quantized_path),
            "--dequantized-output",
            str(dequantized_path),
            "--report",
            str(report_path),
        ],
        f"{input_dtype}/{distribution}/{app_config.name} {phase} {index}",
    )
    validated_result = _validate_report(
        report_path,
        input_path,
        quantized_path,
        dequantized_path,
        suite,
        input_dtype,
        distribution,
        app_config,
    )
    return RunResult(
        input_dtype=validated_result.input_dtype,
        distribution=validated_result.distribution,
        config_name=validated_result.config_name,
        repeat_index=index,
        report_path=validated_result.report_path,
        quant_kernel_ms=validated_result.quant_kernel_ms,
        dequant_kernel_ms=validated_result.dequant_kernel_ms,
        max_abs=validated_result.max_abs,
        mae=validated_result.mae,
        mse=validated_result.mse,
    )


def _write_summary(output_path: Path, results: list[RunResult]) -> None:
    """将正式运行按输入 dtype、分布和真实配置聚合为 Markdown 表。"""
    grouped_results: dict[tuple[str, str, str], list[RunResult]] = {}
    for result in results:
        group_key = (result.input_dtype, result.distribution, result.config_name)
        grouped_results.setdefault(group_key, []).append(result)

    lines = [
        "# 端到端配置遍历汇总",
        "",
        "只汇总正式 repeat；warmup 仍会校验全部文件和 JSON，但不计入统计。",
        "",
        "| 输入 dtype | 分布 | 配置 | 重复次数 | quant 平均 ms | dequant 平均 ms | quant 中位 ms | dequant 中位 ms | max_abs | MAE | MSE |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group_key, group in sorted(grouped_results.items()):
        input_dtype, distribution, config_name = group_key
        quant_times = [result.quant_kernel_ms for result in group]
        dequant_times = [result.dequant_kernel_ms for result in group]
        lines.append(
            "| "
            f"{input_dtype} | {distribution} | {config_name} | {len(group)} | "
            f"{statistics.fmean(quant_times):.6f} | "
            f"{statistics.fmean(dequant_times):.6f} | "
            f"{statistics.median(quant_times):.6f} | "
            f"{statistics.median(dequant_times):.6f} | "
            f"{max(result.max_abs for result in group):.9g} | "
            f"{max(result.mae for result in group):.9g} | "
            f"{max(result.mse for result in group):.9g} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_argument_parser() -> argparse.ArgumentParser:
    """定义组合遍历、筛选和重复次数的 CLI。"""
    parser = argparse.ArgumentParser(
        description="遍历 QDTENSOR 输入分布和 configs/ 中的量化配置，运行真实 CUDA app。",
    )
    parser.add_argument(
        "--executable",
        type=Path,
        default=_PROJECT_ROOT / "build" / "rtx4060-release" / "apps" / "quant_dequant",
        help="已构建 app；默认是 rtx4060-release 的 quant_dequant。",
    )
    parser.add_argument("--config-dir", type=Path, default=_DEFAULT_CONFIG_DIRECTORY)
    parser.add_argument(
        "--config-names",
        nargs="*",
        default=(),
        help="可选：只运行这些 TOML stem，例如 mxfp8_tensor_stochastic_fp32。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "outputs" / "e2e_suite",
    )
    parser.add_argument("--rows", type=_positive_integer, default=1024)
    parser.add_argument("--cols", type=_positive_integer, default=1025)
    parser.add_argument("--input-dtypes", nargs="+", choices=("fp16", "fp32"), default=("fp16", "fp32"))
    parser.add_argument(
        "--distributions",
        nargs="+",
        choices=("uniform", "normal", "outlier"),
        default=("uniform", "normal", "outlier"),
    )
    parser.add_argument("--formats", nargs="+", choices=("mxfp8", "nvfp4"), default=("mxfp8", "nvfp4"))
    parser.add_argument("--scale-modes", nargs="+", choices=("block", "tensor"), default=("block", "tensor"))
    parser.add_argument("--roundings", nargs="+", choices=("nearest", "stochastic"), default=("nearest", "stochastic"))
    parser.add_argument("--output-dtypes", nargs="+", choices=("fp16", "bf16", "fp32"), default=("fp16", "bf16", "fp32"))
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument(
        "--warmups",
        type=_nonnegative_integer,
        default=0,
        help=(
            "每个配置额外执行的功能验证次数；每次都是独立 app 进程，"
            "不构成同进程性能 warmup，正式性能请使用 quant_dequant_bench。"
        ),
    )
    parser.add_argument(
        "--repeats",
        type=_positive_integer,
        default=1,
        help=(
            "每个配置的正式端到端功能运行次数；用于 JSON/产物交叉验证，"
            "不是专门 kernel benchmark 的 repeat。"
        ),
    )
    return parser


def main() -> int:
    """执行输入生成、真实配置遍历、报告验证和最终汇总。"""
    arguments = _build_argument_parser().parse_args()
    try:
        app_configs = _discover_and_filter_configs(arguments)
    except RuntimeError as error:
        print(f"run_e2e_suite 失败：{error}")
        return 1

    suite = SuiteConfig(
        executable_path=arguments.executable.resolve(),
        output_directory=arguments.output_dir.resolve(),
        num_rows=arguments.rows,
        num_cols=arguments.cols,
        input_dtypes=tuple(arguments.input_dtypes),
        distributions=tuple(arguments.distributions),
        app_configs=app_configs,
        base_seed=arguments.seed,
        num_warmups=arguments.warmups,
        num_repeats=arguments.repeats,
    )
    if suite.num_warmups != 0:
        print(
            "注意：run_e2e_suite 的 warmup 会重新启动 app，"
            "不会预热后续 repeat 的 CUDA stream/device buffer；"
            "性能测量请改用 quant_dequant_bench。"
        )
    if not suite.executable_path.is_file():
        print(f"run_e2e_suite 失败：找不到 app 可执行文件：{suite.executable_path}")
        return 1
    if not _GENERATOR_PATH.is_file():
        print(f"run_e2e_suite 失败：找不到生成器：{_GENERATOR_PATH}")
        return 1

    suite.output_directory.mkdir(parents=True, exist_ok=True)
    results: list[RunResult] = []
    try:
        for input_dtype in suite.input_dtypes:
            for distribution_index, distribution in enumerate(suite.distributions):
                input_path = _generate_input(suite, input_dtype, distribution, distribution_index)
                for app_config in suite.app_configs:
                    for warmup_index in range(suite.num_warmups):
                        _run_app_once(
                            suite,
                            input_path,
                            input_dtype,
                            distribution,
                            app_config,
                            "warmup",
                            warmup_index,
                        )
                    for repeat_index in range(suite.num_repeats):
                        results.append(
                            _run_app_once(
                                suite,
                                input_path,
                                input_dtype,
                                distribution,
                                app_config,
                                "repeat",
                                repeat_index,
                            )
                        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"run_e2e_suite 失败：{error}")
        return 1

    summary_path = suite.output_directory / "summary.md"
    _write_summary(summary_path, results)
    print(
        "端到端配置遍历通过："
        f"{len(results)} 次正式运行，"
        f"{len(suite.app_configs)} 份 TOML，"
        f"{len(suite.input_dtypes)} 种输入 dtype，"
        f"{len(suite.distributions)} 种分布。"
    )
    print(f"汇总报告：{summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

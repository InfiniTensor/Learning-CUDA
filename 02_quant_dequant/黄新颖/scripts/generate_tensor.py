#!/usr/bin/env python3
"""生成用于量化实验的确定性 QDTENSOR v1 输入矩阵。

脚本只写入 `tensor_role = input` 的 FP16/FP32 QDTENSOR，不依赖 NumPy 或 CUDA。
因此它既可在没有 GPU 的机器上准备实验输入，也可作为端到端脚本的唯一数据源。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import random
import struct
from typing import Final


# QDTENSOR v1 的固定二进制约束；这些值必须与 docs/file_format.md 保持一致。
_TENSOR_MAGIC: Final[bytes] = b"QDTENSOR"
_TENSOR_HEADER_BYTES: Final[int] = 64
_TENSOR_FILE_VERSION: Final[int] = 1
_LITTLE_ENDIAN_MARKER: Final[int] = 1
_TENSOR_ROLE_INPUT: Final[int] = 0
_DTYPE_TO_CODE: Final[dict[str, int]] = {"fp16": 1, "fp32": 3}
_DTYPE_TO_STRUCT_FORMAT: Final[dict[str, str]] = {"fp16": "e", "fp32": "f"}


@dataclass(frozen=True)
class TensorSpec:
    """描述一个将要写成 QDTENSOR 输入文件的物理张量。

    `dtype` 决定 payload 的真实二进制格式；生成阶段先在 Python float 中采样，
    再按该 dtype 窄化并写入。因此 FP16 输入也会真实体现输入物理精度的舍入。
    """

    num_rows: int  # row-major 矩阵行数。
    num_cols: int  # row-major 矩阵列数。
    dtype: str  # QDTENSOR input payload 的物理 dtype：fp16 或 fp32。

    def element_count(self) -> int:
        """返回未溢出的逻辑元素数，并检查文件格式要求的正形状。"""
        if self.num_rows <= 0 or self.num_cols <= 0:
            raise ValueError("rows 和 cols 必须都是正整数。")
        if self.dtype not in _DTYPE_TO_CODE:
            raise ValueError("输入 dtype 只能是 fp16 或 fp32。")
        return self.num_rows * self.num_cols


@dataclass(frozen=True)
class DistributionSpec:
    """描述一种可复现实验数据分布及其可选异常值规则。"""

    name: str  # uniform、normal 或 outlier。
    seed: int  # Python 独立随机数发生器的确定性种子。
    uniform_low: float  # uniform 模式下的闭区间下界近似值。
    uniform_high: float  # uniform 模式下的闭区间上界近似值。
    normal_mean: float  # normal/outlier 基础正态分布的均值。
    normal_stddev: float  # normal/outlier 基础正态分布的标准差。
    outlier_probability: float  # outlier 模式中每元素替换为异常值的概率。
    outlier_magnitude: float  # outlier 模式中异常值的绝对幅值。

    def validate(self) -> None:
        """验证各模式都会使用到的数值范围，拒绝 NaN、Inf 和无效概率。"""
        finite_values = (
            self.uniform_low,
            self.uniform_high,
            self.normal_mean,
            self.normal_stddev,
            self.outlier_probability,
            self.outlier_magnitude,
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("分布参数必须全部是有限浮点数。")
        if self.name not in {"uniform", "normal", "outlier"}:
            raise ValueError("distribution 只能是 uniform、normal 或 outlier。")
        if self.uniform_low >= self.uniform_high:
            raise ValueError("uniform-low 必须小于 uniform-high。")
        if self.normal_stddev <= 0.0:
            raise ValueError("normal-stddev 必须为正数。")
        if not 0.0 <= self.outlier_probability <= 1.0:
            raise ValueError("outlier-probability 必须位于 [0, 1]。")
        if self.outlier_magnitude <= 0.0:
            raise ValueError("outlier-magnitude 必须为正数。")


def _sample_value(
    random_generator: random.Random,
    distribution: DistributionSpec,
) -> float:
    """从配置指定的分布采样一个有限 FP32 候选值。

    outlier 模式先采样基础正态值，再按固定概率替换为正负对称的大幅值。
    这样它既保留大多数常规元素，也能测试 block/tensor amax 被稀疏异常值主导时
    的 scale 和误差行为。
    """
    if distribution.name == "uniform":
        return random_generator.uniform(
            distribution.uniform_low,
            distribution.uniform_high,
        )

    base_value = random_generator.normalvariate(
        distribution.normal_mean,
        distribution.normal_stddev,
    )
    if distribution.name == "normal":
        return base_value

    # outlier 使用独立伯努利判断和随机符号，避免异常值总偏向正数或负数。
    if random_generator.random() < distribution.outlier_probability:
        outlier_sign = -1.0 if random_generator.random() < 0.5 else 1.0
        return outlier_sign * distribution.outlier_magnitude
    return base_value


def generate_values(
    tensor: TensorSpec,
    distribution: DistributionSpec,
) -> list[float]:
    """按 row-major 线性顺序生成一个确定性浮点数组。

    Args:
        tensor: 输出矩阵形状和物理 dtype 描述。
        distribution: 数据分布及随机种子。

    Returns:
        长度恰为 `rows * cols` 的有限 Python float 数组。
    """
    num_elements = tensor.element_count()
    distribution.validate()
    random_generator = random.Random(distribution.seed)

    # 预分配使大矩阵生成避免多次扩容；线性顺序就是 QDTENSOR 的 row-major 顺序。
    values: list[float] = [0.0] * num_elements
    for element_index in range(num_elements):
        sampled_value = _sample_value(random_generator, distribution)
        if not math.isfinite(sampled_value):
            raise ValueError(f"第 {element_index} 个采样值不是有限数。")
        values[element_index] = sampled_value
    return values


def _encode_payload(tensor: TensorSpec, values: list[float]) -> bytes:
    """将采样值按目标输入 dtype 编码为 little-endian 连续 payload。"""
    num_elements = tensor.element_count()
    if len(values) != num_elements:
        raise ValueError("values 长度与 TensorSpec 形状不一致。")

    element_format = _DTYPE_TO_STRUCT_FORMAT[tensor.dtype]
    element_bytes = struct.calcsize(f"<{element_format}")
    payload = bytearray(num_elements * element_bytes)
    for element_index, value in enumerate(values):
        try:
            # struct 的 e/f 格式直接实现 IEEE binary16/binary32 的物理窄化。
            struct.pack_into(
                f"<{element_format}",
                payload,
                element_index * element_bytes,
                value,
            )
        except OverflowError as error:
            raise ValueError(
                f"第 {element_index} 个值 {value} 无法表示为 {tensor.dtype}。"
            ) from error
    return bytes(payload)


def write_input_qdtensor(
    output_path: Path,
    tensor: TensorSpec,
    values: list[float],
) -> None:
    """将数据写为严格的 QDTENSOR v1 输入文件。

    副作用：必要时创建父目录，并覆盖 `output_path`。header 的所有整数和 payload
    均显式使用 little-endian，绝不依赖 Python 或运行主机的 struct 对齐。
    """
    payload = _encode_payload(tensor, values)
    num_elements = tensor.element_count()

    header = bytearray(_TENSOR_HEADER_BYTES)
    header[0:8] = _TENSOR_MAGIC
    # offset 8--14 依次是版本、header 大小、字节序、dtype 和 input role。
    struct.pack_into(
        "<HHBBB",
        header,
        8,
        _TENSOR_FILE_VERSION,
        _TENSOR_HEADER_BYTES,
        _LITTLE_ENDIAN_MARKER,
        _DTYPE_TO_CODE[tensor.dtype],
        _TENSOR_ROLE_INPUT,
    )
    # offset 15 和 56--63 保持为零，保留给 QDTENSOR v1 的 future fields。
    struct.pack_into(
        "<QQQQ",
        header,
        16,
        tensor.num_rows,
        tensor.num_cols,
        num_elements,
        _TENSOR_HEADER_BYTES,
    )
    struct.pack_into("<Q", header, 48, len(payload))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output_file:
        output_file.write(header)
        output_file.write(payload)


def _positive_integer(value: str) -> int:
    """供 argparse 使用：解析并验证正整数。"""
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("必须是正整数。")
    return parsed_value


def _build_argument_parser() -> argparse.ArgumentParser:
    """构造生成器 CLI 的参数定义。"""
    parser = argparse.ArgumentParser(
        description="生成确定性 FP16/FP32 QDTENSOR v1 输入矩阵。",
    )
    parser.add_argument("--output", type=Path, required=True, help="输出 .qdtensor 路径。")
    parser.add_argument("--rows", type=_positive_integer, required=True, help="矩阵行数。")
    parser.add_argument("--cols", type=_positive_integer, required=True, help="矩阵列数。")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp32")
    parser.add_argument(
        "--distribution",
        choices=("uniform", "normal", "outlier"),
        required=True,
        help="uniform、normal 或含稀疏异常值的 outlier。",
    )
    parser.add_argument("--seed", type=int, default=20260917, help="确定性随机种子。")
    parser.add_argument("--uniform-low", type=float, default=-1.0)
    parser.add_argument("--uniform-high", type=float, default=1.0)
    parser.add_argument("--normal-mean", type=float, default=0.0)
    parser.add_argument("--normal-stddev", type=float, default=1.0)
    parser.add_argument("--outlier-probability", type=float, default=0.01)
    parser.add_argument("--outlier-magnitude", type=float, default=32.0)
    return parser


def main() -> int:
    """解析 CLI，生成矩阵，写入文件并输出可供脚本读取的 JSON 元数据。"""
    arguments = _build_argument_parser().parse_args()
    tensor = TensorSpec(arguments.rows, arguments.cols, arguments.dtype)
    distribution = DistributionSpec(
        name=arguments.distribution,
        seed=arguments.seed,
        uniform_low=arguments.uniform_low,
        uniform_high=arguments.uniform_high,
        normal_mean=arguments.normal_mean,
        normal_stddev=arguments.normal_stddev,
        outlier_probability=arguments.outlier_probability,
        outlier_magnitude=arguments.outlier_magnitude,
    )

    try:
        values = generate_values(tensor, distribution)
        write_input_qdtensor(arguments.output, tensor, values)
    except (OSError, ValueError) as error:
        print(f"generate_tensor 失败：{error}")
        return 1

    # stdout 只输出一行 JSON，便于 run_e2e_suite.py 或人工脚本解析和记录。
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "rows": tensor.num_rows,
                "cols": tensor.num_cols,
                "dtype": tensor.dtype,
                "distribution": distribution.name,
                "seed": distribution.seed,
                "element_count": tensor.element_count(),
                "file_bytes": arguments.output.stat().st_size,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

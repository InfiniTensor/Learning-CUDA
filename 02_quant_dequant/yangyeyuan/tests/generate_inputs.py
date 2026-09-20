#!/usr/bin/env python3
"""生成不依赖 NumPy 的张量和配置样例，供 CLI 测试与基准脚本复用。"""

import argparse
import math
import random
import struct
from pathlib import Path


def write_tensor(path: Path, rows: int, cols: int, dtype: str, values):
    """按项目约定写入 ASCII header 和紧随其后的 little-endian 数据。"""
    pack_code = "f" if dtype == "fp32" else "e"
    header = (
        "[header]\n"
        f"num_rows: {rows}\n"
        f"num_cols: {cols}\n"
        f"dtype: {dtype}\n\n"
        "[data]\n"
    ).encode("ascii")
    with path.open("wb") as output:
        output.write(header)
        output.write(struct.pack(f"<{len(values)}{pack_code}", *values))


def write_config(path: Path, fmt: str, output_type: str = "fp32"):
    """写入 canonical block size 的最小 TOML 配置。"""
    block_size = 32 if fmt == "mxfp8" else 16
    path.write_text(
        f'format = "{fmt}"\n'
        f"block_size = {block_size}\n"
        'scale_mode = "block"\n'
        f'output_type = "{output_type}"\n'
        'rounding = "nearest"\n'
        'target_gpu = "T4"\n'
        "seed = 2026\n",
        encoding="utf-8",
    )


def main():
    """生成三种分布、一个 FP16 奇数长度样例以及四种 CLI 配置。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=257)
    parser.add_argument("--cols", type=int, default=513)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(2026)
    count = args.rows * args.cols
    distributions = {
        "uniform": [rng.uniform(-1.0, 1.0) for _ in range(count)],
        "normal": [rng.gauss(0.0, 1.0) for _ in range(count)],
        "outlier": [rng.gauss(0.0, 0.1) for _ in range(count)],
    }
    for index in range(0, count, 4093):
        # 主体较窄、少量值较大的分布，用来观察 block scaling 对异常值的隔离效果。
        distributions["outlier"][index] = 20.0 if index % 2 == 0 else -16.0
    for name, values in distributions.items():
        write_tensor(args.output / f"{name}_fp32.bin", args.rows, args.cols,
                     "fp32", values)
    # 35 个元素用于检查 NVFP4 最后一个 byte 的高 nibble 是否正确补零。
    fp16_values = [math.sin(i * 0.17) * 2.0 for i in range(35)]
    write_tensor(args.output / "small_fp16.bin", 5, 7, "fp16", fp16_values)
    write_config(args.output / "mxfp8_fp32.toml", "mxfp8", "fp32")
    write_config(args.output / "nvfp4_fp32.toml", "nvfp4", "fp32")
    write_config(args.output / "mxfp8_bf16.toml", "mxfp8", "bf16")
    write_config(args.output / "nvfp4_fp16.toml", "nvfp4", "fp16")


if __name__ == "__main__":
    main()

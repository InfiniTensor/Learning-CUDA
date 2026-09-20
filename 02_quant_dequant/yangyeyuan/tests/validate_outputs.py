#!/usr/bin/env python3
"""校验 LPQ header/layout，以及反量化原始输出的字节数和指标字段。"""

import argparse
import json
import struct
from pathlib import Path


HEADER = struct.Struct("<8s6I3Q2I2QfI")


def main():
    """检查量化文件长度、格式编号、packed 大小和 metrics.json 的基本完整性。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--elements", type=int, required=True)
    parser.add_argument("--format", choices=("mxfp8", "nvfp4"), required=True)
    parser.add_argument("--output-bytes", type=int, choices=(2, 4), required=True)
    args = parser.parse_args()

    quantized = (args.run_dir / "quantized.lpq").read_bytes()
    # Struct 格式必须与 main.cu 中的 88-byte packed header 保持同步。
    fields = HEADER.unpack_from(quantized)
    magic, version, header_bytes, fmt = fields[:4]
    packed_bytes, scale_count = fields[12], fields[13]
    expected_packed = args.elements if args.format == "mxfp8" else (args.elements + 1) // 2
    assert magic.rstrip(b"\0") == b"LPQBIN1"
    assert version == 1 and header_bytes == HEADER.size == 88
    assert fmt == (1 if args.format == "mxfp8" else 2)
    assert packed_bytes == expected_packed
    assert len(quantized) == HEADER.size + packed_bytes + scale_count
    assert (args.run_dir / "dequantized.bin").stat().st_size == args.elements * args.output_bytes
    metrics = json.loads((args.run_dir / "metrics.json").read_text())
    for key in ("max_abs_error", "mae", "mse", "compression_ratio",
                "quant_kernel_ms", "dequant_kernel_ms",
                "quant_effective_gbps", "dequant_effective_gbps"):
        assert key in metrics and metrics[key] >= 0
    print(f"validated {args.format}: packed={packed_bytes}, scales={scale_count}")


if __name__ == "__main__":
    main()

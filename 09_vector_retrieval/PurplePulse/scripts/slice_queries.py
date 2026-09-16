#!/usr/bin/env python3
"""从查询文件中复制前 N 条 query，用于较慢的 CPU 正确性验证。"""

import argparse
import struct
from pathlib import Path


DTYPE_SIZES = {1: 4, 2: 2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()

    if args.count <= 0:
        parser.error("count 必须大于 0")

    with Path(args.input).open("rb") as source:
        magic = source.read(8)
        if magic != b"PPQRY001":
            raise SystemExit("输入不是本项目的 query 文件")
        num_queries, dim, dtype = struct.unpack("<QII", source.read(16))
        if dtype not in DTYPE_SIZES:
            raise SystemExit(f"不支持的 dtype 编号: {dtype}")
        if args.count > num_queries:
            raise SystemExit("count 不能大于原文件的 query 数量")
        value_bytes = args.count * dim * DTYPE_SIZES[dtype]
        values = source.read(value_bytes)
        if len(values) != value_bytes:
            raise SystemExit("输入文件数据不完整")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output:
        output.write(b"PPQRY001")
        output.write(struct.pack("<QII", args.count, dim, dtype))
        output.write(values)
    print(f"已写出前 {args.count} 条 query: {output_path}")


if __name__ == "__main__":
    main()

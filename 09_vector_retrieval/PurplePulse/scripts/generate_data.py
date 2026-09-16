#!/usr/bin/env python3
"""生成项目自定义的 FP32/FP16 向量库和查询文件。"""

import argparse
import random
import struct
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None


METRICS = {"l2": 1, "inner_product": 2, "cosine": 3}
FLOAT32 = 1
FLOAT16 = 2
DTYPES = {"fp32": FLOAT32, "fp16": FLOAT16}


def write_floats(file, values, dtype, chunk_size=65536):
    """分块写入，避免百万级数据时一次创建巨大的 bytes 对象。"""
    for start in range(0, len(values), chunk_size):
        chunk = values[start : start + chunk_size]
        format_code = "f" if dtype == "fp32" else "e"
        file.write(struct.pack(f"<{len(chunk)}{format_code}", *chunk))


def make_values(random_generator, count):
    return [random_generator.uniform(-1.0, 1.0) for _ in range(count)]


def write_random_values(file, random_generator, total_count, dtype):
    while total_count:
        count = min(total_count, 1_048_576)
        if np is None:
            write_floats(file, make_values(random_generator, count), dtype)
        else:
            numpy_dtype = "<f4" if dtype == "fp32" else "<f2"
            values = random_generator.uniform(-1.0, 1.0, size=count).astype(
                numpy_dtype
            )
            file.write(values.tobytes())
        total_count -= count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--num-vectors", type=int, required=True)
    parser.add_argument("--num-queries", type=int, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--metric", choices=METRICS, default="l2")
    parser.add_argument("--dtype", choices=DTYPES, default="fp32")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.num_vectors <= 0 or args.num_queries <= 0 or args.dim <= 0:
        parser.error("数量和维度必须大于 0")

    database_path = Path(args.database)
    query_path = Path(args.queries)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    query_path.parent.mkdir(parents=True, exist_ok=True)

    random_generator = (
        random.Random(args.seed) if np is None else np.random.default_rng(args.seed)
    )
    with database_path.open("wb") as file:
        file.write(b"PPVEC001")
        file.write(
            struct.pack(
                "<QIII",
                args.num_vectors,
                args.dim,
                DTYPES[args.dtype],
                METRICS[args.metric],
            )
        )
        write_random_values(
            file, random_generator, args.num_vectors * args.dim, args.dtype
        )

    with query_path.open("wb") as file:
        file.write(b"PPQRY001")
        file.write(
            struct.pack("<QII", args.num_queries, args.dim, DTYPES[args.dtype])
        )
        write_random_values(
            file, random_generator, args.num_queries * args.dim, args.dtype
        )

    print(f"已生成向量库: {database_path}")
    print(f"已生成查询: {query_path}")


if __name__ == "__main__":
    main()

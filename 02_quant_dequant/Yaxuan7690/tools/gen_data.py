#!/usr/bin/env python3
# =============================================================================
#  gen_data.py -- 生成测试矩阵与量化参数文件
#
#  张量文件格式 (小端):
#     int64  num_rows
#     int64  num_cols
#     uint8  dtype_len
#     char   dtype[dtype_len]      # "fp32" / "fp16"
#     <data> num_rows*num_cols 个元素, 行主序
#
#  用法示例:
#     python tools/gen_data.py --out-dir data --name normal_1024 --rows 1024 --cols 1024 \
#            --dist normal --dtype fp32
# =============================================================================
import argparse
import os
import struct
import sys

import numpy as np


def write_tensor(path, arr, dtype_name):
    rows, cols = arr.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<qq", rows, cols))
        db = dtype_name.encode("ascii")
        f.write(struct.pack("<B", len(db)))
        f.write(db)
        f.write(arr.tobytes(order="C"))


def make_matrix(dist, rows, cols, seed, outlier_ratio, outlier_scale):
    rng = np.random.default_rng(seed)
    n = rows * cols
    if dist == "random":
        a = rng.random((rows, cols), dtype=np.float64) * 2.0 - 1.0
    elif dist == "normal":
        a = rng.standard_normal((rows, cols))
    elif dist == "outlier":
        a = rng.standard_normal((rows, cols))
        k = max(1, int(n * outlier_ratio))
        idx = rng.choice(n, size=k, replace=False)
        flat = a.reshape(-1)
        mag = outlier_scale * (1.0 + rng.random(k))
        sign = np.where(rng.random(k) < 0.5, -1.0, 1.0)
        flat[idx] = mag * sign
    elif dist == "uniform_wide":
        a = (rng.random((rows, cols), dtype=np.float64) * 2.0 - 1.0) * 100.0
    else:
        raise SystemExit("unknown dist: %s" % dist)
    return np.ascontiguousarray(a.astype(np.float32))


PARAM_TEMPLATE = """# 量化参数文件
format = "{fmt}"
block_size = {block_size}
scale_mode = "{scale_mode}"
output_type = "{output_type}"
rounding = "{rounding}"
target_gpu = "{gpu}"
"""


def write_params(path, fmt, block_size, scale_mode, output_type, rounding, gpu):
    with open(path, "w", encoding="utf-8") as f:
        f.write(PARAM_TEMPLATE.format(fmt=fmt, block_size=block_size, scale_mode=scale_mode,
                                      output_type=output_type, rounding=rounding, gpu=gpu))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--name", default="matrix")
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--cols", type=int, default=1024)
    ap.add_argument("--dist", default="normal",
                    choices=["random", "normal", "outlier", "uniform_wide"])
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outlier-ratio", type=float, default=0.001)
    ap.add_argument("--outlier-scale", type=float, default=50.0)
    ap.add_argument("--params", action="store_true", help="同时生成一组默认参数文件")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    a = make_matrix(args.dist, args.rows, args.cols, args.seed,
                    args.outlier_ratio, args.outlier_scale)
    if args.dtype == "fp16":
        a = a.astype(np.float16)
    tensor_path = os.path.join(args.out_dir, "%s.tensor" % args.name)
    write_tensor(tensor_path, a, args.dtype)
    amax = float(np.max(np.abs(a.astype(np.float64))))
    print("wrote %s  shape=%dx%d dtype=%s amax=%.6g" %
          (tensor_path, args.rows, args.cols, args.dtype, amax))

    if args.params:
        p = os.path.join(args.out_dir, "%s.params" % args.name)
        write_params(p, "mxfp8", 32, "block", "fp16", "nearest", "RTX 5060 Ti")
        print("wrote", p)


if __name__ == "__main__":
    main()
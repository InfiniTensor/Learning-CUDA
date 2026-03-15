#!/usr/bin/env python3
"""
NF4 反量化正确性验证

将 CUDA kernel 输出与 bitsandbytes 参考输出逐元素对比，
输出 MAE / MaxError / RMSE / 相对 MAE，并以退出码反映结果。
"""

import argparse
import struct
import sys

import numpy as np


def read_shape(filepath):
    with open(filepath, "rb") as f:
        num_rows = struct.unpack("<q", f.read(8))[0]
        num_cols = struct.unpack("<q", f.read(8))[0]
    return int(num_rows), int(num_cols)


def load_output(filepath, rows, cols, compute_type):
    if compute_type == "bf16":
        raw = np.fromfile(filepath, dtype=np.uint16)
        out = (raw.astype(np.uint32) << 16).view(np.float32)
    else:
        out = np.fromfile(filepath, dtype=np.float16).astype(np.float32)
    return out.reshape(rows, cols)


def compare(cuda, ref, threshold=1e-2):
    diff = np.abs(cuda - ref)
    mae = diff.mean()
    max_err = diff.max()
    rmse = np.sqrt((diff ** 2).mean())
    val_range = max(ref.max() - ref.min(), 1e-8)
    relative_mae = mae / val_range

    passed = relative_mae < threshold

    print(f"\n  CUDA kernel  vs  bitsandbytes 参考")
    print(f"  {'─' * 40}")
    print(f"  MAE          : {mae:.8f}")
    print(f"  MaxError     : {max_err:.8f}")
    print(f"  RMSE         : {rmse:.8f}")
    print(f"  相对MAE      : {relative_mae:.8f}  (范围: {val_range:.4f})")
    print(f"  结果         : {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    parser = argparse.ArgumentParser(description="NF4 解量化正确性验证")
    parser.add_argument("--weight_file", required=True)
    parser.add_argument("--ref_file", required=True)
    parser.add_argument("--cuda_file", required=True)
    parser.add_argument("--compute_type", default="bf16", choices=["bf16", "fp16"])
    args = parser.parse_args()

    rows, cols = read_shape(args.weight_file)
    print(f"[verify] shape=({rows}, {cols}), type={args.compute_type}")

    ref = load_output(args.ref_file, rows, cols, args.compute_type)
    cuda = load_output(args.cuda_file, rows, cols, args.compute_type)

    passed = compare(cuda, ref)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

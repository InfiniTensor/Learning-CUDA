#!/usr/bin/env python3
"""PyTorch / NumPy 独立参考实现与性能基线。

用途：
  1) 按定义式 y = scale * (H_n @ x) 生成参考结果（float64 计算，float32 输出），
     其中 H[i][j] = (-1)^popcount(i & j)（Sylvester 构造）——
     这是与 CUDA 端快速算法**完全独立**的实现，用于交叉验证。
  2) 用 PyTorch 在 GPU 上执行同样的矩阵乘，测出 O(n^2) 基线耗时，
     用于计算「FHT 相对 PyTorch 矩阵乘基线的加速比」。

用法:
  python3 tools/torch_ref.py <data_file> <out_ref_file> [--n 128] [--scale 1.0]
                             [--repeats 10] [--no-gpu]
输出:
  写出 <out_ref_file>（与程序一致的张量文件格式），并打印 torch_ms=... 供调用方解析。
"""
import argparse
import sys

import numpy as np


def read_tensor(path):
    with open(path, "rb") as f:
        blob = f.read()
    pos = blob.find(b"[data]")
    if pos < 0:
        raise ValueError("缺少 [data] 段")
    head = blob[:pos].decode("ascii", "ignore")
    start = blob.find(b"\n", pos) + 1
    rows = cols = None
    dtype = "fp32"
    for line in head.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip().lower(), v.strip()
        if k == "rows":
            rows = int(v)
        elif k == "cols":
            cols = int(v)
        elif k == "dtype":
            dtype = v
    n = rows * cols
    if dtype == "fp32":
        arr = np.frombuffer(blob[start:start + n * 4], dtype=np.float32)
    elif dtype == "fp16":
        arr = np.frombuffer(blob[start:start + n * 2], dtype=np.float16).astype(np.float32)
    else:  # bf16
        u = np.frombuffer(blob[start:start + n * 2], dtype=np.uint16).astype(np.uint32)
        arr = (u << 16).view(np.float32)
    return rows, cols, arr.astype(np.float32).reshape(rows, cols).copy()


def write_tensor(path, arr):
    rows, cols = arr.shape
    with open(path, "wb") as f:
        f.write(b"[header]\n")
        f.write(f"rows: {rows}\n".encode())
        f.write(f"cols: {cols}\n".encode())
        f.write(b"layout: row_major\n")
        f.write(b"dtype: fp32\n")
        f.write(b"apply_dim: last\n\n[data]\n")
        f.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes())


def hadamard_matrix(n):
    """H[i][j] = (-1)^popcount(i & j)，Sylvester 构造。"""
    idx = np.arange(n, dtype=np.int64)
    a = idx[:, None] & idx[None, :]
    # popcount 奇偶（用位运算技巧，避免 Python 逐元素循环）
    v = a.astype(np.uint64)
    parity = np.zeros_like(v, dtype=np.uint64)
    for _ in range(32):
        parity ^= (v & 1)
        v >>= 1
    return (1.0 - 2.0 * parity.astype(np.float64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out_ref")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--no-gpu", action="store_true")
    args = ap.parse_args()

    rows, cols, x = read_tensor(args.data)
    if cols != args.n:
        print(f"警告: 数据 cols={cols} 与 --n={args.n} 不一致，按 cols 计算", file=sys.stderr)
        args.n = cols

    H = hadamard_matrix(args.n)
    # ---- 参考结果：float64 计算，避免参考自身的精度损失 ----
    y_ref = (x.astype(np.float64) @ H.T) * args.scale
    write_tensor(args.out_ref, y_ref.astype(np.float32))
    print(f"[ref] rows={rows} cols={cols} n={args.n} scale={args.scale}")
    print(f"[ref] 参考结果已写出: {args.out_ref}")

    # ---- PyTorch GPU 基线（矩阵乘，O(n^2)）----
    torch_ms = float("nan")
    if not args.no_gpu:
        try:
            import torch

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            xt = torch.from_numpy(np.ascontiguousarray(x)).to(dev).float()
            Ht = torch.from_numpy(H).to(dev).float().T.contiguous()
            s = float(args.scale)
            for _ in range(3):   # warmup
                y = (xt @ Ht) * s
            if dev == "cuda":
                torch.cuda.synchronize()
                st, en = torch.cuda.Event(True), torch.cuda.Event(True)
                st.record()
                for _ in range(args.repeats):
                    y = (xt @ Ht) * s
                en.record()
                torch.cuda.synchronize()
                torch_ms = st.elapsed_time(en) / args.repeats
            else:
                import time
                t0 = time.perf_counter()
                for _ in range(args.repeats):
                    y = (xt @ Ht) * s
                torch_ms = (time.perf_counter() - t0) / args.repeats * 1e3
            print(f"[ref] PyTorch({dev}) 矩阵乘基线: {torch_ms:.6f} ms "
                  f"(rows={rows}, n={args.n}, repeats={args.repeats})")
        except Exception as e:  # noqa: BLE001
            print(f"[ref] PyTorch 基线不可用: {e}", file=sys.stderr)
    print(f"torch_ms={torch_ms}")


if __name__ == "__main__":
    main()

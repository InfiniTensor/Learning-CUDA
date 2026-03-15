#!/usr/bin/env python3
"""
NF4 反量化 —— bitsandbytes 性能基准

测量 bitsandbytes dequantize_4bit 的执行时间和带宽，
供 CUDA kernel 实现计算加速比。
"""

import argparse
import statistics

import torch
import bitsandbytes.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="bitsandbytes NF4 解量化性能基准")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--blocksize", type=int, default=64, choices=[64, 128])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sweep", action="store_true",
                        help="测试多种矩阵大小")
    return parser.parse_args()


def bench_one(rows, cols, blocksize, seed, warmup, repeats):
    """对单个配置运行基准测试，返回 (avg_ms, median_ms, min_ms, max_ms, bw_gbps)."""
    torch.manual_seed(seed)
    weight = torch.randn(rows, cols, dtype=torch.float16, device="cuda")
    quant, state = F.quantize_4bit(
        weight, quant_type="nf4", blocksize=blocksize,
        compress_statistics=True,
    )

    n_elements = rows * cols
    total_bytes = n_elements // 2 + n_elements * 2  # packed_in + fp16_out

    # 预热
    for _ in range(warmup):
        _ = F.dequantize_4bit(quant, state)
    torch.cuda.synchronize()

    # 计时
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]

    for i in range(repeats):
        starts[i].record()
        _ = F.dequantize_4bit(quant, state)
        ends[i].record()

    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))

    avg_ms = statistics.mean(times)
    median_ms = statistics.median(times)
    min_ms = times[0]
    max_ms = times[-1]
    bw_gbps = total_bytes / (median_ms * 1e-3) / 1e9

    return avg_ms, median_ms, min_ms, max_ms, bw_gbps


def main():
    args = parse_args()

    # 主测试
    avg, med, mn, mx, bw = bench_one(
        args.rows, args.cols, args.blocksize,
        args.seed, args.warmup, args.repeats,
    )

    print(f"\n  bitsandbytes dequantize_4bit 性能")
    print(f"  {'─' * 44}")
    print(f"  矩阵         : ({args.rows}, {args.cols})")
    print(f"  块大小       : {args.blocksize}")
    print(f"  平均耗时     : {avg:.4f} ms")
    print(f"  中位数耗时   : {med:.4f} ms")
    print(f"  最小耗时     : {mn:.4f} ms")
    print(f"  最大耗时     : {mx:.4f} ms")
    print(f"  有效带宽     : {bw:.2f} GB/s (基于中位数)")

    # 可选：扫描不同矩阵大小
    if args.sweep:
        shapes = [
            (1024, 1024), (2048, 2048), (4096, 4096),
            (4096, 11008), (4096, 14336),
            (1536, 1536), (1536, 8960),
        ]
        blocksizes = [64, 128]

        print(f"\n  {'Shape':>18s} {'BS':>4s} {'Avg(ms)':>9s} {'Med(ms)':>9s} {'BW(GB/s)':>10s}")
        print(f"  {'─' * 54}")

        for r, c in shapes:
            for bs in blocksizes:
                a, m, _, _, b = bench_one(r, c, bs, args.seed, 5, 50)
                print(f"  ({r:>5d}, {c:>5d}) {bs:>4d} {a:>9.4f} {m:>9.4f} {b:>10.2f}")

    print(f"\n[bench_bnb] 完成")


if __name__ == "__main__":
    main()

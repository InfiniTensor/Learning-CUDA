#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NVFP4 融合 kernel 分组边界压力测试。

融合 kernel 用 __shfl_down_sync(width=8) 在 8 个 lane 内归约组最大值
（每 lane 打包 2 个元素，故 8 lane = 16 元素 = 一个 NVFP4 分组）。
若 width 或组首判断写错，scale 会跨组串用。

构造方式：每 16 元素的组基准逐组放大，组内幅值递增，使各组 max 显著不同
且落在组内不同位置。一旦发生串组，编码结果必然偏离 CPU oracle。

直接调用 pipeline 二进制并传入 verify，由内部的 compare_packed
与 CPU oracle 做逐字节比对（data/scales/global 全字段）。
"""
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(os.path.expanduser("~/Learning-CUDA-project/02_quant_dequant/YeKcaJ"))
BINARY = ROOT / "Core/build/pipeline_nvfp4"


def make_input(path, values):
    path.write_bytes(
        struct.pack("<8sIQQ", b"FP32INP1", 1, 1, len(values))
        + struct.pack("<" + "f" * len(values), *values)
    )


def main():
    if not BINARY.exists():
        print(f"ERROR: 找不到 {BINARY}")
        return 1

    # 覆盖：单组/整组/跨 warp/跨块/尾部不满/大尺寸
    sizes = [16, 17, 31, 32, 33, 48, 255, 256, 257, 512, 1000, 4096, 8191, 8192, 8199]

    failures = []
    with tempfile.TemporaryDirectory(prefix="nvfp4-group-") as tmp:
        tmp = Path(tmp)
        for n in sizes:
            values = []
            for i in range(n):
                group = i // 16
                slot = i % 16
                base = 0.25 * (1 + group % 13)
                magnitude = base * (1.0 + slot * 0.5)
                values.append(magnitude if i % 2 == 0 else -magnitude)

            src = tmp / f"in_{n}.fp32"
            make_input(src, values)

            r = subprocess.run(
                [str(BINARY), str(src), str(tmp / f"{n}.lpq"), str(tmp / f"{n}.out"),
                 "block", "nearest", "fp32", "16", "42", "verify"],
                capture_output=True, text=True,
            )
            ok = r.returncode == 0
            if not ok:
                failures.append(n)
            print(f"  n={n:<6} {'PASS' if ok else 'FAIL'}")
            if not ok:
                tail = (r.stderr.strip().splitlines() or ["?"])[-1]
                print(f"    {tail}")

    total = len(sizes)
    print(f"\n分组边界: {total - len(failures)}/{total} 通过")
    if failures:
        print(f"失败规模: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NVFP4 / MXFP8 边界值验证（针对 scale 编码器改动）。

背景：`scale_code()` 的 NVFP4 分支从二分查找改为 `encode_e4m3_nearest()`；
该函数同时服务 MXFP8 的元素编码。`self-test` 依赖的
`CPU*/tests/data/zeros.fp32` 等冻结样例被 git 跟踪但物理文件已缺失，
故全零、极值等边界未被覆盖。本测试自行构造这些输入。

直接调用 pipeline 二进制并传 `verify`，由内部 `compare_packed`
与 CPU oracle 逐字节比对（data/scales/global 全字段）。

用法：python3 edge_value_test.py [nvfp4|mxfp8]   （默认 nvfp4）
"""
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(os.path.expanduser("~/Learning-CUDA-project/02_quant_dequant/YeKcaJ"))

FORMAT = sys.argv[1] if len(sys.argv) > 1 else "nvfp4"
BINARY = ROOT / f"Core/build/pipeline_{FORMAT}"
BLOCK = "16" if FORMAT == "nvfp4" else "32"

FLT_MIN = struct.unpack("<f", struct.pack("<I", 0x00000001))[0]  # 最小正 subnormal
FLT_MAX = struct.unpack("<f", struct.pack("<I", 0x7F7FFFFF))[0]


def make_input(path, values):
    path.write_bytes(
        struct.pack("<8sIQQ", b"FP32INP1", 1, 1, len(values))
        + struct.pack("<" + "f" * len(values), *values)
    )


def cases(g):
    yield "全零", [0.0] * 256
    yield "负零", [-0.0] * 256
    yield "全零奇数长度", [0.0] * 257
    yield "最大值", [FLT_MAX] * 256
    yield "正负最大", [FLT_MAX if i % 2 else -FLT_MAX for i in range(256)]
    yield "最小subnormal", [FLT_MIN] * 256
    yield "正负最小subnormal", [FLT_MIN * (-1) ** i for i in range(256)]
    yield "次正规混合", [FLT_MIN * (1 + i % 7) * (-1) ** i for i in range(256)]
    yield "极小数", [1e-38 * (-1) ** i for i in range(256)]
    yield "极大数", [1e38 * (-1) ** i for i in range(256)]
    yield "单组内一个有效值", [0.0] * g + [1e-30] + [0.0] * (g - 1)
    yield "每组仅首元素有效", [1e-30 if i % g == 0 else 0.0 for i in range(256)]
    yield "每组仅末元素有效", [1e-30 if i % g == g - 1 else 0.0 for i in range(256)]
    yield "跨组幅值递增", [1e-30 * (1 + i // g) for i in range(256)]
    yield "跨组幅值递减", [1e-30 * (1 + (255 - i) // g) for i in range(256)]
    yield "零与最大值交替", [FLT_MAX if i % 2 else 0.0 for i in range(256)]


def main():
    if not BINARY.exists():
        print(f"ERROR: 找不到 {BINARY}")
        return 1

    group = int(BLOCK)
    all_cases = list(cases(group))
    failures = []
    with tempfile.TemporaryDirectory(prefix=f"{FORMAT}-edge-") as tmp:
        tmp = Path(tmp)
        for name, values in all_cases:
            src = tmp / "in.fp32"
            make_input(src, values)
            r = subprocess.run(
                [str(BINARY), str(src), str(tmp / "o.lpq"), str(tmp / "o.out"),
                 "block", "nearest", "fp32", BLOCK, "42", "verify"],
                capture_output=True, text=True,
            )
            ok = r.returncode == 0
            if not ok:
                failures.append(name)
            print(f"  {name:<20} n={len(values):<4} {'PASS' if ok else 'FAIL'}")
            if not ok:
                print("    " + (r.stderr.strip().splitlines() or ["?"])[-1])

    print(f"\n[{FORMAT}] 边界值: {len(all_cases) - len(failures)}/{len(all_cases)} 通过")
    if failures:
        print(f"失败: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

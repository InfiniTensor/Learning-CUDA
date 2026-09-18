#!/usr/bin/env python3
"""test_vs_library.py — 官方验收测试：本项目 CUDA kernel vs fast_hadamard_transform 库。

验收标准（项目要求原文）：
    对 FP16，与参考实现（fast_hadamard_transform 库）绝对误差 < 1e-2；
    对 BF16，绝对误差 < 5e-2。

流程（每个配置）：
  1. 运行 build/hadamard_bench 生成输入并跑我们的 kernel，--dump_dir 落盘
     （--check false：正确性完全以官方库为准，不与内置 CPU 参考混用）；
  2. 加载 input.bin，reshape(-1, head_dim) 后用官方库计算参考输出
     （normalize=true 传 scale=1/sqrt(head_dim)；normalize=false 用库默认 1.0）；
  3. 与 gpu_output.bin 比较：max/mean/rmse/相对误差/逐位一致率；
  4. 按阈值判定 PASS/FAIL，汇总 CSV + 表格。

运行方式（需要官方库环境）：
    ~/hadamard_env/bin/python tests/test_vs_library.py            # 全矩阵
    ~/hadamard_env/bin/python tests/test_vs_library.py small      # 小规模冒烟
"""

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

try:
    import torch
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    print("[test_vs_library] 需要 torch + fast_hadamard_transform。\n"
          "  请先按 README '依赖安装' 一节搭建 ~/hadamard_env，\n"
          "  然后用 ~/hadamard_env/bin/python 运行本脚本。", file=sys.stderr)
    sys.exit(2)

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "build" / "hadamard_bench"
DUMP_ROOT = ROOT / "results" / "dumps_lib"
CSV_PATH = ROOT / "results" / "library_check.csv"

THRESHOLDS = {"fp16": 1e-2, "bf16": 5e-2}

# (batch, seq, heads, 标签)
SIZES_FULL = [(1, 128, 8, "small"), (2, 512, 16, "medium"), (4, 1024, 32, "large")]
SIZES_SMOKE = [(1, 128, 8, "small")]


def run_bench(dtype, head_dim, batch, seq, heads, normalize, dump_dir):
    cmd = [
        str(BENCH),
        "--batch", str(batch), "--seq", str(seq), "--heads", str(heads),
        "--head_dim", str(head_dim), "--dtype", dtype,
        "--normalize", "true" if normalize else "false",
        "--warmup", "2", "--iters", "5",
        "--check", "false",           # 正确性由官方库判定
        "--dump_dir", str(dump_dir),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"hadamard_bench 失败 (rc={proc.returncode}):\n"
                           f"{proc.stdout}\n{proc.stderr}")


def compare_with_library(dump_dir, dtype, head_dim, normalize):
    meta = json.loads((dump_dir / "meta.json").read_text())
    raw_in = (dump_dir / "input.bin").read_bytes()
    raw_gpu = (dump_dir / "gpu_output.bin").read_bytes()

    tdt = torch.float16 if dtype == "fp16" else torch.bfloat16
    # 关键：reshape(-1, head_dim)。库按最后一维变换，1D 会被当成单行。
    x = torch.frombuffer(bytearray(raw_in), dtype=tdt).reshape(-1, head_dim).cuda()
    scale = 1.0 / math.sqrt(head_dim) if normalize else 1.0
    ref = hadamard_transform(x, scale).float().reshape(-1).cpu()

    gpu = torch.frombuffer(bytearray(raw_gpu), dtype=tdt).float()
    assert gpu.numel() == ref.numel() == meta["total_tokens"] * head_dim

    d = (gpu - ref).abs()
    n = d.numel()
    max_abs = d.max().item()
    mean_abs = d.mean().item()
    rmse = d.pow(2).mean().sqrt().item()
    rel = (d / torch.clamp(ref.abs(), min=1.0)).max().item()
    bit_exact = (d == 0).float().mean().item()
    passed = max_abs < THRESHOLDS[dtype]
    return {
        "dtype": dtype, "head_dim": head_dim, "normalize": normalize,
        "batch_size": meta["batch_size"], "seq_len": meta["seq_len"],
        "num_heads": meta["num_heads"], "total_tokens": meta["total_tokens"],
        "max_abs_error": max_abs, "mean_abs_error": mean_abs, "rmse": rmse,
        "max_rel_error": rel, "bit_exact_ratio": bit_exact,
        "threshold": THRESHOLDS[dtype], "pass": passed,
    }


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if not BENCH.exists():
        print(f"[test_vs_library] 找不到 {BENCH}，请先 make", file=sys.stderr)
        return 2

    sizes = SIZES_FULL if mode == "all" else SIZES_SMOKE
    results = []
    for dtype in ("fp16", "bf16"):
        for head_dim in (64, 128, 256):
            for batch, seq, heads, tag in sizes:
                for normalize in (True, False):
                    name = (f"{dtype}_hd{head_dim}_{tag}"
                            f"{'_norm' if normalize else '_unnorm'}")
                    dump_dir = DUMP_ROOT / name
                    print(f"===== {name} =====")
                    run_bench(dtype, head_dim, batch, seq, heads, normalize, dump_dir)
                    r = compare_with_library(dump_dir, dtype, head_dim, normalize)
                    print(
                        f"  max_abs={r['max_abs_error']:.3e} "
                        f"(thr {r['threshold']:.0e}) "
                        f"bit-exact={r['bit_exact_ratio']*100:.2f}% -> "
                        f"{'PASS' if r['pass'] else 'FAIL'}")
                    results.append(r)

    # CSV
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as f:
        # 固定 LF，避免 Linux 仓库中因 csv 模块默认 CRLF 产生无意义 diff。
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()),
                           lineterminator="\n")
        w.writeheader()
        w.writerows(results)

    # 汇总
    n_pass = sum(1 for r in results if r["pass"])
    print(f"\n[test_vs_library] {len(results)} 个配置，{n_pass} 个通过 "
          f"（vs fast_hadamard_transform；CSV: {CSV_PATH}）")
    all_bit_exact = all(r["bit_exact_ratio"] == 1.0 for r in results)
    if all_bit_exact:
        print("[test_vs_library] 全部配置与官方库输出 100% bit-exact")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

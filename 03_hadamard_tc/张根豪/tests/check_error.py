#!/usr/bin/env python3
"""check_error.py — 正确性评估脚本（Week1 交付物）。

用法：
  # 评估一个 dump 目录（含 input.bin / gpu_output.bin / meta.json）
  ~/hadamard_env/bin/python tests/check_error.py results/dumps/fp16_hd128

  # 批量评估多个目录
  ~/hadamard_env/bin/python tests/check_error.py results/dumps/*

  # 同时输出 CSV，便于写报告
  ~/hadamard_env/bin/python tests/check_error.py results/dumps/* --csv results/lib_check.csv

工作流程：
  1. 读取 meta.json 获得形状 / dtype / normalize；
  2. 读取 input.bin（低精度）作为参考实现的输入；
  3. 用**官方 fast_hadamard_transform 库**（唯一参考实现）计算标准答案；
  4. 读取 gpu_output.bin（低精度）解码为 FP32，与标准答案比较；
  5. 输出 max_abs_error / mean_abs_error / rmse / max_rel_error / pass-fail。

判断标准（与项目要求一致）：
  FP16: max_abs_error < 1e-2
  BF16: max_abs_error < 5e-2

依赖：torch + fast_hadamard_transform（安装见 README "依赖安装"；
建议直接用 ~/hadamard_env/bin/python 运行）。
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hadamard_reference import decode_native, reference_from_native

THRESHOLDS = {"fp16": 1e-2, "bf16": 5e-2}


def evaluate_dump_dir(dump_dir: Path, verbose: bool = True) -> dict:
    meta_path = dump_dir / "meta.json"
    input_path = dump_dir / "input.bin"
    gpu_out_path = dump_dir / "gpu_output.bin"
    for p in (meta_path, input_path, gpu_out_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} 不存在；请先用 hadamard_bench --dump_dir 生成")

    meta = json.loads(meta_path.read_text())
    dtype = meta["dtype"]
    head_dim = int(meta["head_dim"])
    normalize = bool(meta["normalize"])
    total_tokens = int(meta["total_tokens"])

    raw_in = input_path.read_bytes()
    raw_out = gpu_out_path.read_bytes()

    # 参考实现：官方 fast_hadamard_transform 库（唯一参考）
    ref, ref_source = reference_from_native(raw_in, dtype, head_dim, normalize)

    # GPU 输出
    gpu = decode_native(raw_out, dtype)

    n = len(ref)
    assert n == len(gpu) == total_tokens * head_dim, "元素数量与 meta.json 不一致"

    max_abs = 0.0
    sum_abs = 0.0
    sum_sq = 0.0
    max_rel = 0.0
    for i in range(n):
        d = float(gpu[i]) - float(ref[i])
        ad = abs(d)
        if ad > max_abs:
            max_abs = ad
        rel = ad / max(1.0, abs(float(ref[i])))  # 平滑相对误差
        if rel > max_rel:
            max_rel = rel
        sum_abs += ad
        sum_sq += d * d
    mean_abs = sum_abs / n
    rmse = math.sqrt(sum_sq / n)

    threshold = THRESHOLDS[dtype]
    passed = max_abs < threshold

    if verbose:
        print(
            f"[check_error] {dump_dir.name}: dtype={dtype} head_dim={head_dim} "
            f"normalize={normalize} tokens={total_tokens} ref={ref_source} | "
            f"max_abs_error={max_abs:.6e} mean_abs_error={mean_abs:.6e} "
            f"rmse={rmse:.6e} max_rel_error={max_rel:.6e} "
            f"threshold={threshold:.1e} -> {'PASS' if passed else 'FAIL'}"
        )

    return {
        "dump_dir": str(dump_dir),
        "reference": ref_source,
        "dtype": dtype,
        "batch_size": meta.get("batch_size"),
        "seq_len": meta.get("seq_len"),
        "num_heads": meta.get("num_heads"),
        "head_dim": head_dim,
        "total_tokens": total_tokens,
        "normalize": normalize,
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "rmse": rmse,
        "max_rel_error": max_rel,
        "threshold": threshold,
        "pass": passed,
    }


def main():
    ap = argparse.ArgumentParser(description="Hadamard CUDA 输出正确性评估")
    ap.add_argument("dump_dirs", nargs="+", help="hadamard_bench --dump_dir 生成的目录")
    ap.add_argument("--csv", default=None, help="可选：结果追加到 CSV")
    args = ap.parse_args()

    results = []
    any_fail = False
    for d in args.dump_dirs:
        try:
            r = evaluate_dump_dir(Path(d))
        except FileNotFoundError as e:
            print(f"[check_error] skip {d}: {e}", file=sys.stderr)
            any_fail = True
            continue
        results.append(r)
        if not r["pass"]:
            any_fail = True

    # 汇总表
    if results:
        print()
        header = ("dump", "dtype", "head_dim", "normalize", "tokens",
                  "max_abs", "mean_abs", "rmse", "max_rel", "thr", "pass")
        print("{:<28} {:<6} {:<9} {:<10} {:<8} {:<12} {:<12} {:<12} {:<12} {:<8} {}".format(*header))
        for r in results:
            print("{:<28} {:<6} {:<9} {:<10} {:<8} {:<12.4e} {:<12.4e} {:<12.4e} {:<12.4e} {:<8.1e} {}".format(
                Path(r["dump_dir"]).name, r["dtype"], r["head_dim"],
                str(r["normalize"]), r["total_tokens"], r["max_abs_error"],
                r["mean_abs_error"], r["rmse"], r["max_rel_error"],
                r["threshold"], "PASS" if r["pass"] else "FAIL"))

    if args.csv and results:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        need_header = not csv_path.exists()
        with csv_path.open("a", newline="") as f:
            # 固定 LF，保证结果文件跨平台提交时行尾稳定。
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()),
                               lineterminator="\n")
            if need_header:
                w.writeheader()
            w.writerows(results)
        print(f"\n[check_error] csv appended to {csv_path}")

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()

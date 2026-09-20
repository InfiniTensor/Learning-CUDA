#!/usr/bin/env python3
"""plot_scaling.py -- 把性能扫参结果画成规模曲线与加速比柱状图。

输入是 `scripts/bench.sh` 产出的两个文件，不重新跑任何模拟：

    results/bench/results.csv          # 每个 (N, block) 组合一行
    results/bench/speedup_n65536.json  # 三档 CPU 基线与加速比（可选）

用法：
    python plot_scaling.py --out scaling.png
    python plot_scaling.py --csv results/bench/results.csv \
                           --speedup results/bench/speedup_n65536.json \
                           --peak-tflops 73.5 --out scaling.png

`--peak-tflops` 是**参考线**，必须由使用者按实际硬件填：4090D 的 FP32 峰值
约 73.5 TFLOP/s，MX330 约 1.22 TFLOP/s。脚本不会替你猜是哪块卡 ——
把别处的峰值画到本机数据上，图上就会出现一条假的"离峰值 5%"。
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

# 三档 CPU 基线的键名与显示名。顺序固定：从最弱到最强，
# 柱状图里 OpenMP 那根才是报告里引用的加速比（见 README 的说明）。
_CPU_KEYS = [("naive", "cpu_naive (1 thread)"),
             ("scalar_o3", "cpu_scalar_o3 (1 thread, vectorized)"),
             ("openmp", "cpu_openmp (multithread + vectorized)")]


def load_csv(path):
    if not os.path.exists(path):
        raise SystemExit(f"{path} 不存在：先跑 `bash scripts/bench.sh`。")
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "n": int(r["n"]),
                    "block": int(r["block"]),
                    "ms_per_step": float(r["ms_per_step"]),
                    "gflops": float(r["gflops"]),
                    "psteps": float(r["particle_steps_per_sec"]),
                    "mem_mib": float(r["mem_mib"]),
                    "kernel": r.get("kernel", "?"),
                    "precision": r.get("precision", "?"),
                })
            except (KeyError, ValueError):
                # 半截行（扫参中途被打断）直接跳过，不要让整张图失败。
                continue
    if not rows:
        raise SystemExit(f"{path} 里没有有效数据行。")
    return rows


def load_speedup(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(
        description="画性能扫参曲线（数据来自 bench.sh 的产物）")
    ap.add_argument("--csv", default=os.path.join(root, "results/bench/results.csv"))
    ap.add_argument("--speedup",
                    default=os.path.join(root, "results/bench/speedup_n65536.json"))
    ap.add_argument("--out", default=None, help="存成图片（png/pdf/svg）")
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--peak-tflops", type=float, default=None,
                    help="本机 FP32 峰值 TFLOP/s（参考线；4090D≈73.5，MX330≈1.22）")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    rows = load_csv(args.csv)
    sp = load_speedup(args.speedup)

    if args.out and not args.show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # N -> block -> row
    by_n = defaultdict(dict)
    for r in rows:
        by_n[r["n"]][r["block"]] = r
    ns = sorted(by_n)
    blocks = sorted({r["block"] for r in rows})

    panels = ["ms_per_step", "gflops"] + (["speedup"] if sp else [])
    ncol = 2 if len(panels) > 1 else 1
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.4 * ncol, 4.0 * nrow),
                             squeeze=False)
    flat = [ax for row in axes for ax in row]

    # --- 面板 1：每步耗时 ---
    ax = flat[0]
    for b in blocks:
        xs = [n for n in ns if b in by_n[n]]
        ys = [by_n[n][b]["ms_per_step"] for n in xs]
        ax.plot(xs, ys, marker="o", ms=4, lw=1.3, label=f"block={b}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N (particles)")
    ax.set_ylabel("ms per step")
    ax.set_title("per-step time")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    # --- 面板 2：吞吐 ---
    ax = flat[1]
    for b in blocks:
        xs = [n for n in ns if b in by_n[n]]
        ys = [by_n[n][b]["gflops"] / 1000.0 for n in xs]
        ax.plot(xs, ys, marker="o", ms=4, lw=1.3, label=f"block={b}")
    if args.peak_tflops:
        ax.axhline(args.peak_tflops, color="k", ls="--", lw=0.9, alpha=0.7)
        ax.text(ns[0], args.peak_tflops, f" FP32 peak {args.peak_tflops} TFLOP/s",
                va="bottom", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N (particles)")
    ax.set_ylabel("TFLOP/s")
    ax.set_title("throughput (20 flop/interaction)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    # --- 面板 3：加速比（来自 json，一个 N 上测的三档 CPU 基线）---
    if sp:
        ax = flat[2]
        names, vals = [], []
        cpu_ms = sp.get("cpu_force_eval_ms", {})
        gpu_ms = sp.get("ms_per_step", 0.0)
        for key, disp in _CPU_KEYS:
            ms = cpu_ms.get(key, -1.0)
            if ms and ms > 0 and gpu_ms > 0:
                names.append(disp)
                vals.append(ms / gpu_ms)
        if vals:
            colors = ["#999999", "#6666cc", "#cc4444"]
            bars = ax.bar(range(len(vals)), vals,
                          color=colors[:len(vals)], width=0.55)
            for rect, v in zip(bars, vals):
                ax.text(rect.get_x() + rect.get_width() / 2, v,
                        f"{v:,.0f}x", ha="center", va="bottom", fontsize=9)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, fontsize=8)
            ax.set_yscale("log")
            ax.set_ylabel("speedup vs CPU (log)")
            ax.set_title(f"GPU speedup at N={sp.get('n', '?')} "
                         f"(official denominator: cpu_openmp)")
            ax.grid(True, axis="y", which="both", alpha=0.25)

    for ax in flat[len(panels):]:
        ax.axis("off")

    fig.suptitle(f"scaling: {rows[0]['kernel']} / {rows[0]['precision']}")
    fig.tight_layout()

    # 文本摘要：报告里的性能主表直接从这里抄，避免手工汇总出错。
    print(f"{'N':>8} {'best blk':>9} {'ms/step':>10} {'TFLOP/s':>9} "
          f"{'p-steps/s':>12} {'mem MiB':>8}")
    best = {}
    for n in ns:
        b = min(by_n[n], key=lambda k: by_n[n][k]["ms_per_step"])
        best[n] = by_n[n][b]
        r = best[n]
        print(f"{n:>8} {b:>9} {r['ms_per_step']:>10.4f} "
              f"{r['gflops'] / 1000.0:>9.2f} {r['psteps']:>12.3e} "
              f"{r['mem_mib']:>8.1f}")
    if sp:
        print(f"\nspeedup at N={sp.get('n')} (denominator cpu_openmp): "
              f"{sp.get('speedup_vs_cpu_openmp', float('nan')):,.1f}x "
              f"with {sp.get('openmp_threads', '?')} threads")

    if args.out:
        fig.savefig(args.out, dpi=args.dpi)
        print(f"\nwrote {args.out}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())

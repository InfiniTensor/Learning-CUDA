#!/usr/bin/env python3
"""Create publication-style plots and measured report tables from raw experiments."""
import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics as stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    a = parser.parse_args()
    with (a.results / "benchmark.csv").open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in r:
            if k not in ("group", "kind", "variance_reduction", "rng"):
                r[k] = float(r[k]) if r[k] else None
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    figdir = a.results / "figures"
    figdir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "figure.dpi": 140,
                         "savefig.dpi": 180, "axes.grid": True, "grid.alpha": .2})
    colors = ["#185a9d", "#da7b27", "#26836e"]
    kinds = ("european_call", "european_put", "asian_call", "barrier_call")
    labels = ("European call", "European put", "Asian call", "Barrier call")
    modes = ("none", "antithetic", "control_variate")
    mode_labels = ("Plain MC", "Antithetic", "Control variate")
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figdir / (name + ".png"), bbox_inches="tight")
        fig.savefig(figdir / (name + ".svg"), bbox_inches="tight")
        plt.close(fig)
    def subset(group, **kw):
        return [r for r in groups[group] if all(r[k] == v for k, v in kw.items())]
    def med(rs, key):
        return stats.median(r[key] for r in rs)

    # A stochastic error should be assessed across seeds, not forced monotonicity.
    conv = []
    for n in sorted({r["paths"] for r in groups["convergence"]}):
        sample = subset("convergence", paths=n)
        rmse = math.sqrt(stats.mean(r["absolute_error"] ** 2 for r in sample))
        se = stats.mean(r["standard_error"] for r in sample)
        covered = sum(r["absolute_error"] <= 1.95996398454 * r["standard_error"] for r in sample)
        conv.append((n, rmse, se, covered, len(sample)))
    x, rmse, se = np.array([c[:3] for c in conv]).T
    slope = float(np.polyfit(np.log(x), np.log(rmse), 1)[0])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.loglog(x, rmse, "o-", label="RMSE over 8 seeds", color=colors[0])
    ax.loglog(x, se, "s-", label="Mean estimated standard error", color=colors[2])
    ax.loglog(x, se[0] * np.sqrt(x[0] / x), "--", color="#999999", label=r"$N^{-1/2}$ reference")
    ax.set(xlabel="Number of paths", ylabel="Price error", title=f"European call convergence | fitted RMSE slope {slope:.3f}")
    ax.legend(frameon=False)
    save(fig, "convergence")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
    index = np.arange(4)
    for j, mode in enumerate(modes):
        data = [subset("performance", kind=kind, variance_reduction=mode) for kind in kinds]
        offset = (j - 1) * .24
        axs[0].bar(index + offset, [med(d, "speedup") for d in data], width=.23, label=mode_labels[j], color=colors[j])
        axs[1].bar(index + offset, [med(d, "gpu_ms") for d in data], width=.23, color=colors[j])
    axs[0].set(ylabel="CPU single-thread / GPU speedup", title="1 million paths | median of 3 runs")
    axs[1].set(ylabel="GPU total compute time (ms)", title="Includes allocation, RNG, Greeks, reduction, copy")
    for ax in axs:
        ax.set_xticks(index, labels, rotation=15)
        ax.grid(axis="x", visible=False)
    axs[0].legend(frameon=False, fontsize=8)
    save(fig, "performance")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for j, mode in enumerate(modes):
        data = [subset("performance", kind=kind, variance_reduction=mode) for kind in kinds]
        ax.bar(index + (j-1)*.24, [d[0]["standard_error"] for d in data], width=.23, label=mode_labels[j], color=colors[j])
    ax.set_xticks(index, labels)
    ax.set(yscale="log", ylabel="Price standard error (log scale)", title="Variance reduction | 1 million physical production paths")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    save(fig, "variance_reduction")

    fig, ax = plt.subplots(figsize=(7, 4.3))
    for j, rng in enumerate(("curand", "custom")):
        blocks = sorted({r["block_size"] for r in groups["tuning"]})
        data = [subset("tuning", rng=rng, block_size=b) for b in blocks]
        ys = np.array([med(d, "gpu_ms") for d in data])
        lower = ys - [min(r["gpu_ms"] for r in d) for d in data]
        upper = [max(r["gpu_ms"] for r in d) for d in data] - ys
        ax.errorbar(np.array(blocks) + (-3 if j == 0 else 3), ys, yerr=[lower, upper],
                    marker="o", capsize=4, label=rng, color=colors[j])
    ax.set(xlabel="Threads per block", ylabel="GPU total compute time (ms)",
           title="Asian call launch tuning | median and min–max, 3 runs")
    ax.set_xticks(blocks)
    ax.legend(frameon=False)
    save(fig, "tuning")

    lines = ["### 百万路径：价格、精度与性能", "", "参数：S0=K=100，r=0.03，σ=0.2，T=1；亚式/障碍 256 步，障碍 B=130，seed=1234。每组合重复 3 次。", "",
             "| 合约 | 策略 | GPU价格 | 标准误 | CPU ms 中位数 | GPU ms 中位数（范围） | 加速比中位数 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for kind in kinds:
        for mode in modes:
            d = subset("performance", kind=kind, variance_reduction=mode)
            lines.append(f"| {kind} | {mode} | {d[0]['price']:.8f} | {d[0]['standard_error']:.6g} | {med(d,'cpu_ms'):.3f} | {med(d,'gpu_ms'):.3f} ({min(r['gpu_ms'] for r in d):.3f}–{max(r['gpu_ms'] for r in d):.3f}) | {med(d,'speedup'):.2f}× |")
    maxdiff = max(r["cpu_gpu_difference"] for r in rows if r["cpu_gpu_difference"] is not None)
    lines += ["", f"全部实测 CPU/GPU 价格最大差异为 **{maxdiff:.3g}**。两后端使用同一随机流，但浮点超越函数和统计归约顺序不同，因此不要求价格逐位相等。", "",
              "![性能](results/figures/performance.png)", "", "### 千万路径容量与冷启动", "",
              "| 合约 | 路径数 | 价格 ± 95% 半宽 | CPU ms | GPU ms | 加速比 | 含初始化加速比 | 主路径/s | 显存工作区 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in groups["large"]:
        lines.append(f"| {r['kind']} | {r['paths']:.0f} | {r['price']:.8f} ± {1.95996398454*r['standard_error']:.6f} | {r['cpu_ms']:.3f} | {r['gpu_ms']:.3f} | {r['speedup']:.2f}× | {r['cold_speedup']:.2f}× | {r['paths_per_second']:.3g} | {r['workspace_bytes']:.0f} B |")
    euro = subset("performance", kind="european_call", variance_reduction="none")
    lines += ["", f"百万路径欧式普通 MC 的含初始化加速比中位数为 **{med(euro,'cold_speedup'):.3f}×**。短任务是否获益必须区分上下文已建立和冷启动；大规模路径依赖任务即使包含初始化仍明显超过单线程 CPU。上表千万路径每合约只测一次，不能把该行当作稳定吞吐的重复统计。", "", "### 抽样收敛", "",
              "| N | 8种子 RMSE | 平均 SE | 95% 区间覆盖数 |", "|---:|---:|---:|---:|"]
    for n, error, stderr, covered, total in conv:
        lines.append(f"| {n:.0f} | {error:.7f} | {stderr:.7f} | {covered}/{total} |")
    lines += ["", f"log(RMSE)–log(N) 的拟合斜率为 **{slope:.3f}**，可与理论 -0.5 对照；仅 8 个种子，覆盖率和斜率存在明显统计不确定性。", "",
              "![收敛](results/figures/convergence.png)", "", "### 方差缩减效果", "",
              "| 合约 | 对偶：普通SE²/对偶SE² | 控制变量：普通SE²/控制SE² | 控制变量独立pilot beta |",
              "|---|---:|---:|---:|"]
    for kind in kinds:
        plain, anti, cv = [subset("performance", kind=kind, variance_reduction=m)[0] for m in modes]
        path = next((a.results / "raw").glob(f"performance_{kind}_control_variate_*_0.json"))
        beta = json.loads(path.read_text())["gpu"]["control_beta"]
        lines.append(f"| {kind} | {(plain['standard_error']/anti['standard_error'])**2:.2f}× | {(plain['standard_error']/cv['standard_error'])**2:.2f}× | {beta:.6f} |")
    asian_cv = subset("performance", kind="asian_call", variance_reduction="control_variate")[0]
    lines += ["", f"亚式几何控制变量的原始/调整后样本方差比为 **{asian_cv['variance_ratio']:.1f}×**。其收益在于更小定价误差；并不意味着相同路径数的 kernel 更快。对偶与控制变量互斥，控制变量多算 8192 条先导路径。", "",
              "![方差缩减](results/figures/variance_reduction.png)", "", "### 线程块和随机数实现比较", "",
              "| block_size | cuRAND 中位数 ms | custom 中位数 ms |", "|---:|---:|---:|"]
    for block in blocks:
        lines.append(f"| {block:.0f} | {med(subset('tuning', rng='curand', block_size=block),'gpu_ms'):.3f} | {med(subset('tuning', rng='custom', block_size=block),'gpu_ms'):.3f} |")
    lines += ["", "![粒度调优](results/figures/tuning.png)", "", "笔记本共享显示 GPU 的调度与频率变化使小任务耗时有波动。图中展示全部范围；默认 256 线程是通用配置，本实验不足以证明它在所有负载下最优。"]
    content = "\n".join(lines)
    (a.results / "summary.md").write_text(content.replace("results/figures/", "figures/"), encoding="utf-8")
    report = a.results.parent / "REPORT.md"
    if report.exists():
        text = report.read_text(encoding="utf-8")
        before = text.split("<!-- RESULTS_START -->")[0]
        after = text.split("<!-- RESULTS_END -->")[1]
        report.write_text(before + "<!-- RESULTS_START -->\n" + content + "\n<!-- RESULTS_END -->" + after, encoding="utf-8")
    (a.results / "analysis.json").write_text(json.dumps(dict(runs=len(rows), convergence_slope=slope,
        max_cpu_gpu_difference=maxdiff, asian_control_variance_ratio=asian_cv["variance_ratio"]), indent=2) + "\n")
    print(json.dumps(dict(runs=len(rows), convergence_slope=slope, max_cpu_gpu_difference=maxdiff), indent=2))


if __name__ == "__main__":
    main()

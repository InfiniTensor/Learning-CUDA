#!/usr/bin/env python3
# make_plots.py - 报告图表生成（读取 data/*.csv → report/*.png）
# 需 matplotlib（Windows 主机运行：python scripts/make_plots.py）
# 配色遵循 dataviz 规范：分类色固定顺序（蓝→橙）、单一坐标轴、细标记、直接标注
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- 设计系统参数（light surface）----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"      # 分类槽位 1
ORANGE = "#eb6834"    # 分类槽位 2
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"],
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "font.size": 10,
})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
REPORT = os.path.join(ROOT, "report")


def parse_convergence(path):
    """解析 convergence.csv → {section: [(N, se, err, z), ...]}"""
    sections = {}
    cur = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("# ---"):
                cur = None
                if line.startswith("# ---"):
                    cur = line.split("---")[1].strip()
                    sections[cur] = []
            elif line.startswith("#") or cur is None:
                continue
            else:
                parts = line.split()
                sections[cur].append((int(parts[0]), float(parts[2]), float(parts[3]),
                                      float(parts[4])))
    return sections


def style_axes(ax):
    ax.grid(True, which="major", alpha=0.8)
    for s in ax.spines.values():
        s.set_linewidth(0.8)


def convergence_plot(sections, names, title, fname, slope_ref=True, annotate=None):
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    colors = [BLUE, ORANGE]
    for i, name in enumerate(names):
        data = sections[name]
        N = [d[0] for d in data]
        err = [d[2] for d in data]
        se = [d[1] for d in data]
        ax.errorbar(N, err, yerr=[1.96 * s for s in se], fmt="o-", color=colors[i],
                    linewidth=2, markersize=5, capsize=3, capthick=1, label=name)
        if annotate and i == len(names) - 1:
            ax.annotate(annotate, xy=(N[-1], err[-1]), xytext=(0.5, 0.9),
                        textcoords="axes fraction", color=INK2)
    if slope_ref:
        # 1/√N 参考线（穿过第一组的第一个点）
        d0 = sections[names[0]][0]
        N = [d[0] for d in sections[names[0]]]
        ref = [d0[2] * (d0[0] / n) ** 0.5 for n in N]
        ax.plot(N, ref, "--", color=MUTED, linewidth=1.2, label="∝ 1/√N 参考")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("路径数 N")
    ax.set_ylabel("|误差|（log 尺度）")
    ax.set_title(title)
    if len(names) > 1 or slope_ref:
        ax.legend()
    style_axes(ax)
    fig.tight_layout()
    out = os.path.join(REPORT, fname)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print("saved", out)


def bench_plot(path, fname):
    """GPU 吞吐横向条形图（对数轴，单一指标 paths/sec）"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    gpu = [r for r in rows if r["engine"] == "gpu" and r["option"].startswith("option_asian")]
    cpu = [r for r in rows if r["engine"] == "cpu" and r["option"].startswith("option_asian")]

    def label(r):
        if r["variance_reduction"] != "none":
            vr = {"antithetic": "对偶", "control_variate": "控制变量"}[r["variance_reduction"]]
            return f"block={r['block']} {vr}"
        return f"block={r['block']} rng={r['rng']}"

    gpu = sorted(gpu, key=lambda r: -float(r["paths_per_sec"]))
    labels = [label(r) for r in gpu]
    values = [float(r["paths_per_sec"]) for r in gpu]

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    y = range(len(labels))
    ax.barh(list(y), values, color=BLUE, height=0.62)
    for yi, v in zip(y, values):
        ax.text(v, yi, f" {v:.2e}", va="center", ha="left", color=INK2, fontsize=9)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlabel("paths/sec（对数轴）")
    ax.set_title("亚式看涨 GPU 吞吐对比（10M 路径 × 256 步，除标注外）")
    cpu_pps = float(cpu[0]["paths_per_sec"]) if cpu else 0
    if cpu_pps > 0:
        ax.axvline(cpu_pps, color=MUTED, linestyle="--", linewidth=1.2)
        ax.text(cpu_pps, len(labels) - 0.35, f" CPU 单线程 {cpu_pps:.1e}",
                color=MUTED, fontsize=9, ha="left")
    style_axes(ax)
    fig.tight_layout()
    out = os.path.join(REPORT, fname)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print("saved", out)


def main():
    conv_path = os.path.join(DATA, "convergence.csv")
    if not os.path.exists(conv_path):
        sys.exit("缺少 data/convergence.csv，先运行 scripts/convergence.sh")
    s = parse_convergence(conv_path)
    convergence_plot(s, ["european_call_vs_BS"], "欧式看涨：误差随路径数收敛（vs Black-Scholes）",
                     "fig1_convergence_european.png")
    convergence_plot(s, ["asian_raw", "asian_cv"],
                     "亚式看涨：控制变量将误差降低 ~37 倍（σ 从 7.7 → 0.21）",
                     "fig2_convergence_asian.png", slope_ref=False)
    convergence_plot(s, ["barrier_raw", "barrier_bgk"],
                     "障碍看涨：离散监控偏差不收敛；BGK 修正后随 N 收敛",
                     "fig3_convergence_barrier.png", slope_ref=False)
    bench_path = os.path.join(DATA, "bench.csv")
    if os.path.exists(bench_path):
        bench_plot(bench_path, "fig4_bench.png")
    print("done")


if __name__ == "__main__":
    main()

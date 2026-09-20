#!/usr/bin/env python3
"""plot_energy.py -- 把运行日志里的诊断量解析成守恒量漂移曲线。

数据来源不是轨迹文件，而是 `nbody_sim` 的 stdout 日志（用 `--diag-interval N`
让它每 N 步打印一行）。这样能量 / 动量 / 角动量三条曲线共享同一批记录点，
不需要额外的输出格式，也不用为了画图再跑一遍模拟。

日志行的格式（`main.cu` 里 `record_diag` 打印）：
    step    500  E=-6.210000e-01  |p|=3.1e-04  |L|=4.4e+01  max|a|=1.2e+02  2T/|W|=0.9850

n_version1..5 的日志行里没有 |L| 这一列，解析器把它当可选字段，
所以新旧日志都能画（缺列的那条曲线自动从图里消失）。

用法：
    # Euler vs Leapfrog 能量漂移对比（报告 §4.2 的那张图）
    python plot_energy.py results/validate/two_body_euler.log \
                         results/validate/two_body_leapfrog.log \
                         --out energy_euler_vs_leapfrog.png

    # 只画能量面板并直接看
    python plot_energy.py run.log --panels energy --show
"""

import argparse
import re
import sys

import numpy as np

# 这一行的列数变过（|L| 是后加的），用可选组兼容两种情况。
# 不要改成 split()：E 与 |p| 之间空格数会随数值宽度变化，定长切分会碎。
_DIAG_RE = re.compile(
    r"^\s*step\s+(?P<step>\d+)\s+"
    r"E=(?P<E>\S+)\s+"
    r"\|p\|=(?P<p>\S+)\s+"
    r"(?:\|L\|=(?P<L>\S+)\s+)?"
    r"max\|a\|=(?P<maxa>\S+)\s+"
    r"2T/\|W\|=(?P<virial>\S+)"
)


def parse_diag_log(path):
    """解析一个日志文件，返回 dict of ndarray。

    没有角动量列时用 NaN 填充，画图时该曲线会被跳过（matplotlib 默认
    不画 NaN 点），比抛异常友好：拿一份旧日志也能出图看能量。
    """
    steps, energy, mom, ang, maxa, virial = [], [], [], [], [], []
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = _DIAG_RE.match(line)
            if not m:
                continue
            steps.append(int(m.group("step")))
            energy.append(float(m.group("E")))
            mom.append(float(m.group("p")))
            ang.append(float(m.group("L")) if m.group("L") else np.nan)
            maxa.append(float(m.group("maxa")))
            virial.append(float(m.group("virial")))

    if not steps:
        raise SystemExit(
            f"{path}: 没找到诊断量行。跑模拟时要加 `--diag-interval N`，"
            "否则程序只在首尾各算一次诊断量。"
        )

    return {
        "path": path,
        "step": np.array(steps, dtype=np.int64),
        "energy": np.array(energy, dtype=np.float64),
        "momentum": np.array(mom, dtype=np.float64),
        "angular": np.array(ang, dtype=np.float64),
        "max_acc": np.array(maxa, dtype=np.float64),
        "virial": np.array(virial, dtype=np.float64),
    }


def relative_drift(values):
    """相对首值的漂移 |v - v0| / |v0|。

    首值为 0 时退回绝对差（否则整条曲线是 NaN）。动量就是这种情况：
    初值经过 ZeroNetMomentum 后理论上是 0，只能看绝对增长。
    """
    if values.size == 0:
        return values
    v0 = values[0]
    if abs(v0) > 0.0:
        return np.abs(values - v0) / abs(v0)
    return np.abs(values - v0)


def main():
    ap = argparse.ArgumentParser(
        description="从 nbody_sim 的日志里画守恒量漂移曲线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("logs", nargs="+", help="一个或多个含诊断量行的日志")
    ap.add_argument("--label", action="append", default=None,
                    help="曲线名（可重复，顺序对应 logs；默认用文件名）")
    ap.add_argument("--panels", default="energy,angular,momentum,virial",
                    help="要画的面板，逗号分隔：energy,angular,momentum,virial,acc")
    ap.add_argument("--out", default=None, help="存成图片（png/pdf/svg）")
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--title", default=None)
    ap.add_argument("--show", action="store_true", help="交互显示（不存文件）")
    args = ap.parse_args()

    if args.out and not args.show:
        # 无显示环境下必须显式选 Agg，否则 savefig 会尝试连 X server。
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [parse_diag_log(p) for p in args.logs]
    labels = args.label or [r["path"].rsplit("/", 1)[-1] for r in runs]
    if len(labels) != len(runs):
        raise SystemExit(f"--label 给了 {len(labels)} 个，logs 有 {len(runs)} 个")

    panels = [p.strip() for p in args.panels.split(",") if p.strip()]
    # 面板 -> (取数函数, 纵轴标签, 是否用对数轴)
    specs = {
        "energy": (lambda r: relative_drift(r["energy"]), "|dE/E0|", True),
        "angular": (lambda r: relative_drift(r["angular"]), "|dL/L0|", True),
        "momentum": (lambda r: r["momentum"], "|p| (absolute)", True),
        "virial": (lambda r: r["virial"], "2T/|W|", False),
        "acc": (lambda r: r["max_acc"], "max|a|", True),
    }
    unknown = [p for p in panels if p not in specs]
    if unknown:
        raise SystemExit(f"未知面板 {unknown}，可选 {sorted(specs)}")

    ncol = 2 if len(panels) > 1 else 1
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.4 * ncol, 3.6 * nrow),
                             squeeze=False)
    flat = [ax for row in axes for ax in row]

    print(f"{'log':<40} {'records':>8} {'|dE/E0|':>12} {'|dL/L0|':>12} "
          f"{'|p|_final':>12} {'virial_final':>13}")
    for run, label in zip(runs, labels):
        de = relative_drift(run["energy"])[-1]
        dl = relative_drift(run["angular"])[-1]
        print(f"{label:<40} {run['step'].size:>8} {de:>12.3e} {dl:>12.3e} "
              f"{run['momentum'][-1]:>12.3e} {run['virial'][-1]:>13.4f}")

    for ax, panel in zip(flat, panels):
        getter, ylabel, use_log = specs[panel]
        for run, label in zip(runs, labels):
            y = getter(run)
            if not np.any(np.isfinite(y)):
                continue  # 该日志没有这一列（例如旧版没有 |L|）
            ax.plot(run["step"], y, marker="o", ms=3, lw=1.2, label=label)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.set_title(panel)
        if use_log:
            ax.set_yscale("log")
        if panel == "virial":
            ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.6)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)

    for ax in flat[len(panels):]:
        ax.axis("off")

    fig.suptitle(args.title or "conservation drifts")
    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=args.dpi)
        print(f"\nwrote {args.out}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())

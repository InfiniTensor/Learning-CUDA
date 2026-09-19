#!/usr/bin/env python3
"""从冻结的逐次 CSV 重绘定价报告插图，不修改实验数据或汇总口径。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "report_assets"
COLORS = {"none": "#767676", "antithetic": "#0F4D92", "control_variate": "#42949E", "rqmc": "#B64342"}
NAMES = {"none": "Plain MC", "antithetic": "Antithetic", "control_variate": "Control variate", "rqmc": "Randomized Halton"}
MARKERS = {"none": "o", "antithetic": "s", "control_variate": "^", "rqmc": "D"}
LINES = {"none": "--", "antithetic": "-", "control_variate": "-.", "rqmc": "-"}
PRODUCTS = {"european_mc": "European call", "asian_mc": "Arithmetic Asian call", "barrier_mc": "Discrete up-and-out call", "european_qmc": "European call: RQMC and plain MC"}


def read_rows(path: Path) -> list[dict]:
    """严格读取非空证据表；缺失值或非有限数不能被画成零。"""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty evidence: {path}")
    return rows


def fingerprint(path: Path) -> dict:
    """记录输入指纹，图表可以独立追溯到随报告提供的 CSV。"""
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def summarize(stage: str, field: str) -> list[dict]:
    """复算每个配置的均值、样本标准差和四分位数，并与原 aggregate 核对。"""
    groups = defaultdict(list)
    for row in read_rows(ASSETS / "evidence" / stage / "summary.csv"):
        groups[(row["variance_reduction"], int(row["num_paths"]))].append(float(row[field]))
    aggregate = {(r["variance_reduction"], int(r["num_paths"])): r for r in read_rows(ASSETS / "evidence" / stage / "aggregate.csv")}
    mean_field = "mean_reported_standard_error" if field == "standard_error" else "mean_absolute_error"
    if set(groups) != set(aggregate):
        raise ValueError(f"Configuration mismatch: {stage}")
    result = []
    for (method, count), values in sorted(groups.items()):
        if len(values) != 10 or any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError(f"Invalid repetition data: {stage}, {method}, {count}")
        mean, sd = statistics.mean(values), statistics.stdev(values)
        if not math.isclose(mean, float(aggregate[(method, count)][mean_field]), rel_tol=1e-10, abs_tol=1e-13):
            raise ValueError(f"Aggregate mean differs: {stage}, {method}, {count}")
        q25, q75 = np.quantile(values, [.25, .75], method="linear")
        result.append({"method": "rqmc" if stage == "european_qmc" else method,
                       "num_paths": count, "replications": len(values), "mean": mean,
                       "sample_sd": sd, "q25": float(q25), "q75": float(q75), "values": values})
    return result


def style() -> dict:
    """采用已安装字体和色盲/灰度可分辨的颜色、线型与标记组合。"""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    family = "Arial" if "Arial" in installed else "DejaVu Sans"
    return {"font.family": family, "font.size": 11.5, "axes.labelsize": 12,
            "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
            "axes.linewidth": .85, "axes.spines.right": False, "axes.spines.top": False,
            "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10,
            "legend.frameon": False, "svg.fonttype": "none", "svg.hashsalt": "cuda-report-task5",
            "pdf.fonttype": 42, "savefig.facecolor": "white"}


def draw(stage: str, field: str, destination: Path) -> dict:
    """显示均值及真实重复波动；绝对差用 IQR，避免对数轴截断负的误差条。"""
    points = summarize(stage, field)
    if stage == "european_qmc":
        points.extend(p for p in summarize("european_mc", field) if p["method"] == "none")
    methods = [m for m in ("none", "antithetic", "control_variate", "rqmc") if any(p["method"] == m for p in points)]
    band_kind = "mean +/- sample SD of reported SE" if field == "standard_error" else "25th-75th percentile of absolute differences"
    with plt.rc_context(style()):
        fig, ax = plt.subplots(figsize=(7.2, 4.3))
        for method in methods:
            part = sorted((p for p in points if p["method"] == method), key=lambda p: p["num_paths"])
            x = np.array([p["num_paths"] for p in part])
            means = np.array([p["mean"] for p in part])
            if field == "standard_error":
                spread = np.array([p["sample_sd"] for p in part])
                low, high = means - spread, means + spread
            else:
                low, high = np.array([p["q25"] for p in part]), np.array([p["q75"] for p in part])
            if np.any(low <= 0):
                raise ValueError("Non-positive uncertainty bound on log axis")
            ax.fill_between(x, low, high, color=COLORS[method], alpha=.13, linewidth=0, zorder=1)
            ax.plot(x, means, label=NAMES[method], color=COLORS[method], linestyle=LINES[method],
                    marker=MARKERS[method], linewidth=1.8, markersize=5.5,
                    markeredgecolor="white", markeredgewidth=.65, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        if stage == "european_qmc":
            # 两种采样的规模网格不同，保留实际 x 值，不插值或移动到同一网格。
            ax.set_xticks([1e4, 1e5, 1e6, 1e7])
        else:
            ax.set_xticks(sorted({p["num_paths"] for p in points}))
        ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=5))
        ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(2, 5)))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.grid(axis="y", which="major", color="#D7DDE3", linewidth=.55)
        ax.tick_params(which="both", direction="out", width=.75)
        ax.set_axisbelow(True)
        ax.set_xlabel("Estimator samples, $N$")
        ax.set_ylabel("Mean reported standard error" if field == "standard_error" else "Mean absolute difference to reference")
        ax.set_title(PRODUCTS[stage], pad=26)
        note = "10 repetitions; band: mean ± 1 SD of SE" if field == "standard_error" else "10 repetitions; band: interquartile range"
        ax.text(0, 1.035, note, transform=ax.transAxes, color="#5B626B", fontsize=9)
        ax.legend(loc="upper right", handlelength=2.8, borderaxespad=.5)
        ax.margins(x=.075, y=.14)
        fig.tight_layout(pad=1.35)
        for ext in ("svg", "png", "pdf"):
            metadata = {"Creator": "redraw_report_figures.py"}
            if ext == "svg":
                metadata["Date"] = None
            elif ext == "pdf":
                metadata.update(CreationDate=None, ModDate=None)
            fig.savefig(destination.with_suffix("."+ext), dpi=300, metadata=metadata)
        plt.close(fig)
    sources = [ASSETS / "evidence" / stage / file for file in ("summary.csv", "aggregate.csv")]
    if stage == "european_qmc":
        sources += [ASSETS / "evidence/european_mc" / file for file in ("summary.csv", "aggregate.csv")]
    return {"figure": destination.name, "field": field, "band": band_kind,
            "meaning": "Descriptive variability over ten runs; not a confidence interval or reference uncertainty.",
            "sources": [fingerprint(p) for p in sources], "points": points}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ASSETS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for stage in PRODUCTS:
        for field, suffix in (("standard_error", "standard_error"), ("absolute_error", "absolute_error")):
            destination = args.output_dir / f"{stage}_{suffix}.svg"
            records.append(draw(stage, field, destination))
            print(f"Rendered {destination.name}")
    (args.output_dir / "figure_statistics.json").write_text(json.dumps({
        "description": "Recomputed from preserved formal CSV, not new GPU measurements.",
        "matplotlib": matplotlib.__version__, "numpy": np.__version__,
        "figures": records}, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

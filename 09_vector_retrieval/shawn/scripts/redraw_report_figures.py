#!/usr/bin/env python3
"""Redraw Task 9 report figures from the committed lightweight evidence only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
import tempfile
from pathlib import Path

# 字体缓存属于运行时临时数据，不写入用户目录或原始实验证据。
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cuda-report-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator, PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "report_assets"
COLORS = {"blue": "#0F4D92", "teal": "#42949E", "warm": "#B64342", "grey": "#767676"}


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取原始 CSV；Nsight 的状态前导行不属于表头，须显式跳过。"""
    text = path.read_text(encoding="utf-8-sig")
    if path.name == "cuda_gpu_kern_sum.csv":
        lines = text.splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith("Time (%),"))
        text = "\n".join(lines[start:])
    return list(csv.DictReader(io.StringIO(text)))


def configure_style() -> str:
    """使用可用无衬线字体，保持矢量文字可编辑及论文尺寸下可读。"""
    available = {item.name for item in font_manager.fontManager.ttflist}
    chosen = "Arial" if "Arial" in available else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": chosen,
        "font.size": 11.5,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelcolor": "#272727",
        "text.color": "#272727",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.85,
        "xtick.color": "#4D4D4D",
        "ytick.color": "#4D4D4D",
        "legend.frameon": False,
        "legend.fontsize": 10.5,
        "lines.linewidth": 1.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
    })
    return chosen


def grouped_statistics(
    rows: list[dict[str, str]], parameter: str, metric: str, expected_runs: int
) -> list[dict[str, object]]:
    """按实验参数重算均值及样本标准差，不将运行间 SD 写成置信区间。"""
    groups: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(int(row[parameter]), []).append(row)
    result = []
    for value, items in sorted(groups.items()):
        items.sort(key=lambda row: int(row["repetition"]))
        assert len(items) == expected_runs, (parameter, value, len(items))
        assert len({row["repetition"] for row in items}) == expected_runs
        assert all(int(row["quality_queries"]) == 20 for row in items)
        observations = [float(row[metric]) for row in items]
        assert all(math.isfinite(number) for number in observations)
        result.append({
            "parameter": value,
            "runs": expected_runs,
            "mean": statistics.mean(observations),
            "sample_sd": statistics.stdev(observations),
            "observations": observations,
        })
    return result


def canvas(title: str, xlabel: str, ylabel: str):
    """将环境与统计口径留给正文图注，扩大图内的数据展示区域。"""
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.16)
    ax.set_title(title, loc="left", pad=13)
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.55)
    ax.tick_params(length=4, width=0.8)
    return fig, ax


def save_figure(fig, destination: Path, stem: str) -> list[str]:
    """同一画布输出可编辑 SVG、300 dpi PNG 与嵌入字体 PDF。"""
    names = []
    for extension in ("svg", "png", "pdf"):
        target = destination / f"{stem}.{extension}"
        metadata = {"Creator": "redraw_report_figures.py"}
        if extension in ("svg", "pdf"):
            metadata["Date" if extension == "svg" else "CreationDate"] = None
        fig.savefig(target, dpi=300, metadata=metadata)
        names.append(target.name)
    plt.close(fig)
    return names


def draw_curve(
    destination: Path,
    stem: str,
    rows: list[dict[str, str]],
    parameter: str,
    metric: str,
    expected_runs: int,
    title: str,
    xlabel: str,
    ylabel: str,
    upper_limit: float,
    exact: bool = False,
    reference: float | None = None,
) -> dict[str, object]:
    stats = grouped_statistics(rows, parameter, metric, expected_runs)
    tail = metric == "p99_latency_ms"
    fig, ax = canvas(title, xlabel, ylabel)
    positions = list(range(len(stats)))
    color, marker, line = (COLORS["blue"], "o", "-") if exact else (COLORS["teal"], "s", "--")
    means = [float(item["mean"]) for item in stats]
    deviations = [float(item["sample_sd"]) for item in stats]
    ax.errorbar(
        positions, means, yerr=deviations, color=color, marker=marker,
        linestyle=line, markersize=5.5, markerfacecolor="white", markeredgewidth=1.5,
        ecolor=color, elinewidth=1.3, capsize=4, capthick=1.3,
        label="Exact" if exact else "IVF-Flat", zorder=3,
    )
    if reference is not None:
        ax.axhline(reference, color=COLORS["grey"], linestyle=":", linewidth=1.5,
                   label="Exact, K = 10 (mean)", zorder=2)
        # 图例置于空闲的数据区上沿，不再占用标题所在行。
        ax.legend(loc="upper left" if tail else "upper right", ncol=2,
                  borderaxespad=0.55, handlelength=2.3)
    ax.set_xticks(positions, [str(item["parameter"]) for item in stats])
    ax.set_xlim(-0.22, len(stats) - 0.78)
    ax.set_ylim(0, upper_limit)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    for x, value in zip(positions, means):
        label = f"{value:,.1f}" if metric == "qps" else f"{value:.3f}"
        # 首个 nprobe 吞吐量标签置于点下方，避免和精确检索参考线相撞。
        below = stem == "ivf_nprobe_qps" and x == 0
        ax.annotate(label, (x, value), xytext=(0, -13 if below else 11),
                    textcoords="offset points", ha="center",
                    va="top" if below else "bottom", fontsize=10, color=color)
    output = save_figure(fig, destination, stem)
    return {
        "files": output, "metric": metric, "parameter": parameter,
        "statistics": stats, "error_bars": "sample standard deviation, ddof=1",
        "sampling_unit": "process run" if not tail else "per-run P99 estimate",
        "x_axis": "categorical evaluated settings; connecting lines guide the eye",
        "y_limits": [0, upper_limit], "reference_mean": reference,
    }


def draw_recall(destination: Path, rows: list[dict[str, str]]) -> dict[str, object]:
    """召回率固定相同查询集，显示完整比例范围，不制造质量置信度。"""
    stats = grouped_statistics(rows, "nprobe", "recall_at_k", 5)
    assert all(item["mean"] == 1 and item["sample_sd"] == 0 for item in stats)
    fig, ax = canvas("IVF recall across probe settings", "Probed lists (nprobe)",
                     "Recall@10")
    positions = list(range(len(stats)))
    ax.plot(positions, [float(item["mean"]) for item in stats], color=COLORS["teal"],
            marker="s", linestyle="--", markersize=5.5, markerfacecolor="white",
            markeredgewidth=1.5, clip_on=False)
    ax.set_xticks(positions, [str(item["parameter"]) for item in stats])
    ax.set_xlim(-0.22, len(stats) - 0.78)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.text(0.5, 0.54, "Recall@10 = 1.00 across probe settings",
            transform=ax.transAxes, ha="center", va="center", color=COLORS["grey"],
            fontsize=12)
    return {"files": save_figure(fig, destination, "ivf_nprobe_recall"),
            "metric": "recall_at_k", "statistics": stats,
            "quality_queries": 20, "error_bars": "none; same quality sample reused",
            "y_limits": [0, 1.05]}


def kernel_statistics(path: Path) -> dict[str, float]:
    """只聚合 GPU 核函数执行时间；不混入 CUDA API、拷贝或主机时间。"""
    score = 0
    sort = 0
    for row in read_csv(path):
        value = int(row["Total Time (ns)"])
        if "score_candidates_kernel" in row["Name"]:
            score += value
        elif "DeviceRadixSort" in row["Name"]:
            sort += value
        else:
            raise ValueError(f"Unclassified kernel: {row['Name']}")
    total = score + sort
    assert total > 0 and score > 0 and sort > 0
    return {"scoring_ms": score / 1e6, "sorting_ms": sort / 1e6,
            "total_ms": total / 1e6, "scoring_percent": score / total * 100,
            "sorting_percent": sort / total * 100}


def draw_kernel_share(destination: Path, evidence: Path) -> dict[str, object]:
    stats = [kernel_statistics(evidence / f"profile_nsys_{mode}" / "cuda_gpu_kern_sum.csv")
             for mode in ("exact", "ivf")]
    fig, ax = canvas("GPU kernel time composition", "Share of GPU kernel time", "")
    fig.subplots_adjust(left=0.17)
    ax.grid(False)
    for index, item in enumerate(stats):
        left = item["scoring_percent"]
        ax.barh(index, left, color=COLORS["blue"], height=0.52,
                edgecolor="white", linewidth=1, label="Scoring" if index == 0 else None)
        ax.barh(index, item["sorting_percent"], left=left, color=COLORS["warm"],
                height=0.52, hatch="///", edgecolor="white", linewidth=1,
                label="CUB sorting" if index == 0 else None)
        ax.text(left / 2, index, f"{left:.1f}%", ha="center", va="center",
                color="white", fontweight="bold", fontsize=11.5)
        ax.text(left + item["sorting_percent"] / 2, index,
                f"{item['sorting_percent']:.1f}%", ha="center", va="center",
                color="white", fontweight="bold", fontsize=10.5)
    ax.set_yticks([0, 1], ["Exact", "IVF-Flat"])
    ax.set_xlim(0, 100)
    ax.set_ylim(1.50, -0.70)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(PercentFormatter(100))
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.legend(loc="upper right", ncol=2, handlelength=1.8, borderaxespad=0.35)
    return {"files": save_figure(fig, destination, "kernel_time_share"),
            "mode_statistics": dict(zip(("exact", "ivf_flat"), stats)),
            "denominator": "sum of GPU kernel execution durations in one mode's trace",
            "error_bars": "none; single trace per mode"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=ASSETS / "evidence")
    parser.add_argument("--output-dir", type=Path, default=ASSETS)
    args = parser.parse_args()
    evidence = args.evidence_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    font = configure_style()
    sources = [Path(name) / "summary.csv" for name in ("exact_topk", "ivf_nprobe", "ivf_batch")]
    sources += [Path(f"profile_nsys_{mode}") / "cuda_gpu_kern_sum.csv" for mode in ("exact", "ivf")]
    exact, nprobe, batch = [read_csv(evidence / item) for item in sources[:3]]
    exact_ref = [row for row in exact if int(row["top_k"]) == 10]
    qps_ref = statistics.mean(float(row["qps"]) for row in exact_ref)
    p99_ref = statistics.mean(float(row["p99_latency_ms"]) for row in exact_ref)
    figures: dict[str, object] = {}
    specifications = [
        ("exact_topk_qps", exact, "top_k", "qps", 3, "Exact search throughput", "Top-K", "Queries / second", 1800, True, None),
        ("exact_topk_p99", exact, "top_k", "p99_latency_ms", 3, "Exact search: per-run P99", "Top-K", "P99 query time (ms)", 0.65, True, None),
        ("ivf_nprobe_qps", nprobe, "nprobe", "qps", 5, "IVF throughput versus probing", "Probed lists (nprobe)", "Queries / second", 1800, False, qps_ref),
        ("ivf_nprobe_p99", nprobe, "nprobe", "p99_latency_ms", 5, "IVF: per-run P99 versus probing", "Probed lists (nprobe)", "P99 segmented query time (ms)", 13.5, False, p99_ref),
        ("ivf_batch_qps", batch, "batch_size", "qps", 5, "IVF throughput versus query grouping", "Query group size", "Queries / second", 800, False, None),
        ("ivf_batch_p99", batch, "batch_size", "p99_latency_ms", 5, "IVF: per-run P99 versus query grouping", "Query group size", "P99 segmented query time (ms)", 2.15, False, None),
    ]
    for spec in specifications:
        stem, *parameters = spec
        figures[stem] = draw_curve(output, stem, *parameters)
    figures["ivf_nprobe_recall"] = draw_recall(output, nprobe)
    figures["kernel_time_share"] = draw_kernel_share(output, evidence)
    manifest = {
        "generator": "scripts/redraw_report_figures.py",
        "figure_size_inches": [7.2, 4.3], "font": font,
        "png_dpi": 300, "svg_text": "editable text", "palette": COLORS,
        "input_identity": [{"path": item.as_posix(),
                            "sha256": hashlib.sha256((evidence / item).read_bytes()).hexdigest()}
                           for item in sources],
        "statistical_notes": "SD uses n-1. P99 SD is between runs, not a within-query tail confidence interval. Repeated quality uses the same 20 queries.",
        "figures": figures,
    }
    (output / "figure_data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(figures)} figures in SVG/PNG/PDF plus figure_data_manifest.json")


if __name__ == "__main__":
    main()

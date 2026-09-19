#!/usr/bin/env python3
"""Plot Task 09 parameter sweeps from aggregate CSV files."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PUBLICATION_STYLE = {
    # DejaVu Sans 随 Matplotlib 提供；放在首位可避免无桌面字体的服务器反复告警。
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 12,
    "axes.linewidth": 1.8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
    "svg.fonttype": "none",
}
BLUE = "#0F4D92"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact", type=Path, required=True)
    parser.add_argument("--ivf-nprobe", type=Path, required=True)
    parser.add_argument("--ivf-batch", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    """读取 aggregate.csv，并拒绝空表或非有限关键指标。"""
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"aggregate contains no rows: {path}")
    return rows


def plot_metric(
    *,
    rows: list[dict[str, str]],
    x_field: str,
    y_field: str,
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    log_x: bool = False,
    bounded_recall: bool = False,
) -> None:
    """从 aggregate 绘制单变量扫描曲线。"""
    points = sorted((float(row[x_field]), float(row[y_field])) for row in rows)
    if any(not (math.isfinite(x) and math.isfinite(y)) for x, y in points):
        raise ValueError(f"non-finite value in {title}")
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]

    with plt.rc_context(PUBLICATION_STYLE):
        figure, axis = plt.subplots(figsize=(7.2, 4.5))
        axis.plot(
            x_values,
            y_values,
            color=BLUE,
            linewidth=2.5,
            marker="o",
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
        if log_x:
            axis.set_xscale("log", base=2)
        axis.set_xticks(x_values)
        axis.set_xticklabels([f"{value:g}" for value in x_values])
        axis.set_title(title, fontsize=15, pad=12)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.65)
        axis.set_axisbelow(True)

        y_low, y_high = min(y_values), max(y_values)
        span = y_high - y_low
        padding = max(span * 0.12, abs(y_high) * 0.02, 1.0e-9)
        if bounded_recall:
            axis.set_ylim(max(0.0, y_low - padding), min(1.0, y_high + padding))
        else:
            axis.set_ylim(max(0.0, y_low - padding), y_high + padding)
        for x_value, y_value in points:
            label = f"{y_value:.4f}" if bounded_recall else f"{y_value:.3g}"
            annotation_offset = -18 if bounded_recall and y_value >= 0.98 else 8
            axis.annotate(
                label,
                (x_value, y_value),
                xytext=(0, annotation_offset),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        figure.tight_layout(pad=1.5)
        figure.savefig(output_path, format="svg", dpi=300, facecolor="white")
        figure.savefig(output_path.with_suffix(".png"), dpi=300, facecolor="white")
        plt.close(figure)


def main() -> None:
    args = parse_args()
    sources = {
        "exact": args.exact.resolve(),
        "ivf_nprobe": args.ivf_nprobe.resolve(),
        "ivf_batch": args.ivf_batch.resolve(),
    }
    for path in sources.values():
        if not path.is_file():
            raise SystemExit(f"aggregate does not exist: {path}")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exact = load_rows(sources["exact"])
    ivf_nprobe = load_rows(sources["ivf_nprobe"])
    ivf_batch = load_rows(sources["ivf_batch"])

    specifications = (
        (
            exact,
            "top_k",
            "mean_qps",
            "Exact CUDA throughput vs Top-K",
            "Top-K",
            "Mean QPS",
            "exact_topk_qps.svg",
            True,
            False,
        ),
        (
            exact,
            "top_k",
            "mean_p99_latency_ms",
            "Exact CUDA tail latency vs Top-K",
            "Top-K",
            "Mean P99 latency (ms/query)",
            "exact_topk_p99.svg",
            True,
            False,
        ),
        (
            ivf_nprobe,
            "nprobe",
            "mean_recall_at_k",
            "IVF-Flat recall vs nprobe",
            "nprobe",
            "Recall@K",
            "ivf_nprobe_recall.svg",
            True,
            True,
        ),
        (
            ivf_nprobe,
            "nprobe",
            "mean_qps",
            "IVF-Flat throughput vs nprobe",
            "nprobe",
            "Mean QPS",
            "ivf_nprobe_qps.svg",
            True,
            False,
        ),
        (
            ivf_nprobe,
            "nprobe",
            "mean_p99_latency_ms",
            "IVF-Flat tail latency vs nprobe",
            "nprobe",
            "Mean P99 latency (ms/query)",
            "ivf_nprobe_p99.svg",
            True,
            False,
        ),
        (
            ivf_batch,
            "batch_size",
            "mean_qps",
            "IVF-Flat throughput vs batch size",
            "Batch size",
            "Mean QPS",
            "ivf_batch_qps.svg",
            True,
            False,
        ),
        (
            ivf_batch,
            "batch_size",
            "mean_p99_latency_ms",
            "IVF-Flat tail latency vs batch size",
            "Batch size",
            "Mean P99 latency (ms/query)",
            "ivf_batch_p99.svg",
            True,
            False,
        ),
    )
    for (
        rows,
        x_field,
        y_field,
        title,
        x_label,
        y_label,
        filename,
        log_x,
        recall,
    ) in specifications:
        plot_metric(
            rows=rows,
            x_field=x_field,
            y_field=y_field,
            title=title,
            x_label=x_label,
            y_label=y_label,
            output_path=args.output_dir / filename,
            log_x=log_x,
            bounded_recall=recall,
        )

    index = [
        "# Task 09 Report Plots\n\n",
        "所有曲线均直接读取对应 `aggregate.csv`；warm-up 不进入统计，"
        "重复实验不挑选最佳值。\n\n",
    ]
    for filename in (specification[6] for specification in specifications):
        index.append(f"## {Path(filename).stem}\n\n![{filename}]({filename})\n\n")
    index.append("## Data sources\n\n")
    index.extend(f"- `{name}`: `{path}`\n" for name, path in sources.items())
    (args.output_dir / "plots_index.md").write_text("".join(index), encoding="utf-8")
    print(f"generated {len(specifications)} SVG plots in {args.output_dir}")


if __name__ == "__main__":
    main()

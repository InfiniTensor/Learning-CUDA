#!/usr/bin/env python3
"""Generate dependency-free SVG convergence plots from sweep aggregates."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from html import escape
from pathlib import Path


COLORS = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregate", action="append", required=True, metavar="LABEL=CSV",
        help="可重复提供，例如 european_mc=outputs/.../aggregate.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"aggregate must use LABEL=CSV syntax: {value}")
    label, path = value.split("=", 1)
    if not label or not path:
        raise ValueError(f"aggregate must use LABEL=CSV syntax: {value}")
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"aggregate does not exist: {resolved}")
    return label, resolved


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def log_ticks(low: float, high: float) -> list[float]:
    """用 1/2/5 decade ticks 覆盖数值范围，兼顾宽路径规模与小误差。"""
    start = math.floor(math.log10(low))
    stop = math.ceil(math.log10(high))
    ticks = []
    for exponent in range(start, stop + 1):
        for multiplier in (1.0, 2.0, 5.0):
            value = multiplier * 10.0**exponent
            if low <= value <= high:
                ticks.append(value)
    return ticks


def tick_label(value: float) -> str:
    if value >= 1_000 or value < 0.01:
        return f"{value:.0e}"
    return f"{value:.3g}"


def render_svg(
    *, label: str, rows: list[dict[str, str]], field: str, y_label: str,
    output_path: Path,
) -> bool:
    """按 variance mode 绘制 log-log 曲线；空值不被伪装成零点。"""
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        try:
            x_value = float(row["num_paths"])
            y_value = float(row[field])
        except (KeyError, ValueError):
            continue
        if x_value > 0.0 and y_value > 0.0 and math.isfinite(y_value):
            series[row.get("variance_reduction", "unknown")].append(
                (x_value, y_value)
            )
    if not series:
        return False
    for points in series.values():
        points.sort()
    all_points = [point for points in series.values() for point in points]
    x_values = [point[0] for point in all_points]
    y_values = [point[1] for point in all_points]
    x_low, x_high = min(x_values), max(x_values)
    y_low, y_high = min(y_values), max(y_values)
    if x_low == x_high:
        x_low /= 2.0
        x_high *= 2.0
    if y_low == y_high:
        y_low /= 2.0
        y_high *= 2.0
    x_padding = 0.08 * (math.log10(x_high) - math.log10(x_low))
    y_padding = 0.12 * (math.log10(y_high) - math.log10(y_low))
    x_log_low = math.log10(x_low) - x_padding
    x_log_high = math.log10(x_high) + x_padding
    y_log_low = math.log10(y_low) - y_padding
    y_log_high = math.log10(y_high) + y_padding

    width, height = 920, 560
    left, right, top, bottom = 105, 35, 65, 82
    plot_width = width - left - right
    plot_height = height - top - bottom

    def x_coord(value: float) -> float:
        return left + (
            (math.log10(value) - x_log_low) / (x_log_high - x_log_low)
        ) * plot_width

    def y_coord(value: float) -> float:
        return top + (
            (y_log_high - math.log10(value)) / (y_log_high - y_log_low)
        ) * plot_height

    title = f"{label}: {y_label} convergence"
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n',
        '<rect width="100%" height="100%" fill="white"/>\n',
        '<style>text{font-family:Arial,sans-serif;fill:#111827}'
        '.grid{stroke:#e5e7eb;stroke-width:1}.axis{stroke:#111827;stroke-width:1.4}'
        '.tick{font-size:12px}.label{font-size:15px}.title{font-size:20px;font-weight:600}'
        '.legend{font-size:13px}</style>\n',
        f'<text class="title" x="{width / 2}" y="32" text-anchor="middle">'
        f'{escape(title)}</text>\n',
    ]
    for tick in log_ticks(10**x_log_low, 10**x_log_high):
        x = x_coord(tick)
        svg.append(
            f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" '
            f'y2="{top + plot_height}"/>\n'
        )
        svg.append(
            f'<text class="tick" x="{x:.2f}" y="{top + plot_height + 23}" '
            f'text-anchor="middle">{escape(tick_label(tick))}</text>\n'
        )
    for tick in log_ticks(10**y_log_low, 10**y_log_high):
        y = y_coord(tick)
        svg.append(
            f'<line class="grid" x1="{left}" y1="{y:.2f}" '
            f'x2="{left + plot_width}" y2="{y:.2f}"/>\n'
        )
        svg.append(
            f'<text class="tick" x="{left - 12}" y="{y + 4:.2f}" '
            f'text-anchor="end">{escape(tick_label(tick))}</text>\n'
        )
    svg.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top + plot_height}" '
            f'x2="{left + plot_width}" y2="{top + plot_height}"/>\n',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" '
            f'y2="{top + plot_height}"/>\n',
            f'<text class="label" x="{left + plot_width / 2}" y="{height - 24}" '
            'text-anchor="middle">Estimator samples (log scale)</text>\n',
            f'<text class="label" x="25" y="{top + plot_height / 2}" '
            f'text-anchor="middle" transform="rotate(-90 25 {top + plot_height / 2})">'
            f'{escape(y_label)} (log scale)</text>\n',
        ]
    )
    for index, (name, points) in enumerate(sorted(series.items())):
        color = COLORS[index % len(COLORS)]
        coordinates = " ".join(
            f"{x_coord(x):.2f},{y_coord(y):.2f}" for x, y in points
        )
        svg.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
            'stroke-width="2.5" stroke-linejoin="round"/>\n'
        )
        for x_value, y_value in points:
            svg.append(
                f'<circle cx="{x_coord(x_value):.2f}" cy="{y_coord(y_value):.2f}" '
                f'r="4" fill="{color}"/>\n'
            )
        legend_x = left + 14 + index * 180
        svg.append(
            f'<line x1="{legend_x}" y1="{top + 18}" x2="{legend_x + 25}" '
            f'y2="{top + 18}" stroke="{color}" stroke-width="3"/>\n'
        )
        svg.append(
            f'<text class="legend" x="{legend_x + 32}" y="{top + 23}">'
            f'{escape(name)}</text>\n'
        )
    svg.append('</svg>\n')
    output_path.write_text("".join(svg), encoding="utf-8")
    return True


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = [
        "# Task 05 Convergence Plots\n\n",
        "曲线由对应 `aggregate.csv` 自动生成；quick preset 仅用于流水线检查。\n\n",
    ]
    created = 0
    for source in args.aggregate:
        label, path = parse_source(source)
        rows = load_rows(path)
        for field, suffix, y_label in (
            ("mean_absolute_error", "absolute_error", "Mean absolute error"),
            ("mean_reported_standard_error", "standard_error", "Mean reported SE"),
        ):
            output = args.output_dir / f"{label}_{suffix}.svg"
            if render_svg(
                label=label, rows=rows, field=field, y_label=y_label,
                output_path=output,
            ):
                created += 1
                index.append(f"## {label}: {y_label}\n\n![{label} {suffix}]({output.name})\n\n")
    if created == 0:
        raise SystemExit("no finite positive aggregate values were available to plot")
    (args.output_dir / "plots_index.md").write_text("".join(index), encoding="utf-8")
    print(f"generated {created} SVG plots in {args.output_dir}")


if __name__ == "__main__":
    main()

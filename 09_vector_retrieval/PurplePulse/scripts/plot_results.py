#!/usr/bin/env python3
"""Generate dependency-free SVG figures from project benchmark CSV files."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


WIDTH = 1180
HEIGHT = 430
COLORS = {"WarpFusion": "#0072B2", "FAISS GPU": "#D55E00"}
K_COLORS = {10: "#0072B2", 50: "#009E73", 100: "#D55E00"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def line(x1: float, y1: float, x2: float, y2: float, **attrs: object) -> str:
    attributes = " ".join(f'{key.rstrip("_").replace("_", "-")}="{esc(value)}"' for key, value in attrs.items())
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" {attributes}/>'


def text(x: float, y: float, value: object, **attrs: object) -> str:
    attributes = " ".join(f'{key.rstrip("_").replace("_", "-")}="{esc(attr)}"' for key, attr in attrs.items())
    return f'<text x="{x:.2f}" y="{y:.2f}" {attributes}>{esc(value)}</text>'


def padded_domain(values: list[float], fraction: float = 0.06) -> tuple[float, float]:
    low, high = min(values), max(values)
    span = high - low
    padding = span * fraction if span else max(abs(high) * fraction, 1.0)
    return low - padding, high + padding


def scale(value: float, domain: tuple[float, float], start: float, end: float) -> float:
    low, high = domain
    return start + (value - low) / (high - low) * (end - start)


def ticks(domain: tuple[float, float], count: int = 5) -> list[float]:
    low, high = domain
    return [low + index * (high - low) / (count - 1) for index in range(count)]


def base_svg(title: str, body: list[str], width: int = WIDTH, height: int = HEIGHT) -> str:
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{esc(title)}</title>',
        '<desc id="desc">Benchmark curves generated from the checked-in raw CSV data.</desc>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#202124} .grid{stroke:#d9dde3;stroke-width:1} .axis{stroke:#51565c;stroke-width:1.2} .tick{font-size:11px} .label{font-size:12px;font-weight:600} .panel-title{font-size:14px;font-weight:600} .figure-title{font-size:18px;font-weight:600} .note{font-size:10px;fill:#5f6368}</style>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        *body,
        '</svg>',
        '',
    ])


def load_quality_series(results_dir: Path) -> dict[int, dict[str, list[dict[str, float]]]]:
    formal_files = {top_k: f"ivf_k{top_k}.csv" for top_k in (10, 50, 100)}
    purple_files = (
        formal_files
        if all((results_dir / filename).exists() for filename in formal_files.values())
        else {top_k: f"optimized_v1_k{top_k}.csv" for top_k in (10, 50, 100)}
    )
    series: dict[int, dict[str, list[dict[str, float]]]] = {}
    for top_k, filename in purple_files.items():
        rows = read_csv(results_dir / filename)
        series[top_k] = {
            "WarpFusion": [
                {
                    "recall": float(row["recall_at_k"]),
                    "latency": float(row["average_ms"]),
                    "qps": float(row["qps"]),
                    "nprobe": float(row["nprobe"]),
                }
                for row in rows
            ]
        }
    faiss_filename = (
        "faiss_gpu.csv"
        if (results_dir / "faiss_gpu.csv").exists()
        else "faiss_gpu_formal_v3.csv"
    )
    faiss_rows = read_csv(results_dir / faiss_filename)
    for top_k in purple_files:
        series[top_k]["FAISS GPU"] = [
            {
                "recall": float(row["recall_at_k"]),
                "latency": float(row["average_query_ms"]),
                "qps": float(row["qps"]),
                "nprobe": float(row["nprobe"]),
            }
            for row in faiss_rows
            if row["backend"] == "gpu"
            and row["index_type"] == "ivf_flat"
            and int(row["top_k"]) == top_k
        ]
        if not series[top_k]["FAISS GPU"]:
            raise ValueError(f"missing FAISS GPU IVF rows for K={top_k}")
    return series


def quality_figure(series: dict[int, dict[str, list[dict[str, float]]]], metric: str,
                   title: str, y_label: str) -> str:
    body = [text(WIDTH / 2, 27, title, text_anchor="middle", class_="figure-title")]
    body.extend([
        line(430, 47, 455, 47, stroke=COLORS["WarpFusion"], stroke_width=3),
        text(462, 51, "WarpFusion", class_="label"),
        line(590, 47, 615, 47, stroke=COLORS["FAISS GPU"], stroke_width=3),
        text(622, 51, "FAISS GPU", class_="label"),
    ])
    panel_width = 350
    panel_gap = 30
    plot_top, plot_bottom = 86, 355
    for panel_index, top_k in enumerate((10, 50, 100)):
        panel_left = 55 + panel_index * (panel_width + panel_gap)
        plot_left, plot_right = panel_left + 48, panel_left + panel_width - 10
        all_points = [point for values in series[top_k].values() for point in values]
        x_domain = padded_domain([point["recall"] for point in all_points], 0.04)
        y_domain = padded_domain([point[metric] for point in all_points], 0.08)
        body.append(text((plot_left + plot_right) / 2, 73, f"K = {top_k}", text_anchor="middle", class_="panel-title"))
        for value in ticks(y_domain):
            y = scale(value, y_domain, plot_bottom, plot_top)
            body.append(line(plot_left, y, plot_right, y, class_="grid"))
            body.append(text(plot_left - 7, y + 4, f"{value:.0f}", text_anchor="end", class_="tick"))
        for value in ticks(x_domain):
            x = scale(value, x_domain, plot_left, plot_right)
            body.append(line(x, plot_top, x, plot_bottom, class_="grid"))
            body.append(text(x, plot_bottom + 17, f"{value:.2f}", text_anchor="middle", class_="tick"))
        body.extend([
            line(plot_left, plot_top, plot_left, plot_bottom, class_="axis"),
            line(plot_left, plot_bottom, plot_right, plot_bottom, class_="axis"),
            text((plot_left + plot_right) / 2, 397, "Recall@K", text_anchor="middle", class_="label"),
        ])
        if panel_index == 0:
            body.append(text(15, (plot_top + plot_bottom) / 2, y_label, text_anchor="middle", class_="label", transform=f"rotate(-90 15 {(plot_top + plot_bottom) / 2})"))
        for name in ("WarpFusion", "FAISS GPU"):
            points = sorted(series[top_k][name], key=lambda item: item["recall"])
            coordinates = [
                (scale(point["recall"], x_domain, plot_left, plot_right),
                 scale(point[metric], y_domain, plot_bottom, plot_top), point)
                for point in points
            ]
            path = " ".join(("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}" for index, (x, y, _) in enumerate(coordinates))
            body.append(f'<path d="{path}" fill="none" stroke="{COLORS[name]}" stroke-width="2.5"/>')
            for x, y, point in coordinates:
                body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{COLORS[name]}" stroke="#ffffff" stroke-width="1.5"/>')
                body.append(text(x + 5, y - 6, int(point["nprobe"]), class_="note"))
    body.append(text(WIDTH - 10, HEIGHT - 8, "Matched batch=64 · point labels: nprobe", text_anchor="end", class_="note"))
    return base_svg(title, body)


def batch_figure(results_dir: Path) -> str:
    rows = read_csv(results_dir / "optimized_v1_batch_sweep.csv")
    body = [text(WIDTH / 2, 28, "Batch size vs throughput", text_anchor="middle", class_="figure-title")]
    plot_left, plot_right, plot_top, plot_bottom = 85, 1135, 78, 350
    x_values = sorted({int(row["batch_size"]) for row in rows})
    qps_values = [float(row["qps"]) for row in rows]
    x_domain = padded_domain([float(value) for value in x_values], 0.03)
    y_domain = padded_domain(qps_values, 0.08)
    for value in ticks(y_domain):
        y = scale(value, y_domain, plot_bottom, plot_top)
        body.append(line(plot_left, y, plot_right, y, class_="grid"))
        body.append(text(plot_left - 9, y + 4, f"{value:.0f}", text_anchor="end", class_="tick"))
    for value in x_values:
        x = scale(value, x_domain, plot_left, plot_right)
        body.append(line(x, plot_top, x, plot_bottom, class_="grid"))
        body.append(text(x, plot_bottom + 18, value, text_anchor="middle", class_="tick"))
    body.extend([
        line(plot_left, plot_top, plot_left, plot_bottom, class_="axis"),
        line(plot_left, plot_bottom, plot_right, plot_bottom, class_="axis"),
        text((plot_left + plot_right) / 2, 397, "Batch size (queries)", text_anchor="middle", class_="label"),
        text(20, (plot_top + plot_bottom) / 2, "Throughput (QPS)", text_anchor="middle", class_="label", transform=f"rotate(-90 20 {(plot_top + plot_bottom) / 2})"),
    ])
    legend_x = 430
    for index, top_k in enumerate((10, 50, 100)):
        color = K_COLORS[top_k]
        body.append(line(legend_x + index * 135, 50, legend_x + 25 + index * 135, 50, stroke=color, stroke_width=3))
        body.append(text(legend_x + 32 + index * 135, 54, f"K={top_k}", class_="label"))
        points = sorted(
            ((int(row["batch_size"]), float(row["qps"])) for row in rows if int(row["top_k"]) == top_k),
            key=lambda item: item[0],
        )
        coordinates = [(scale(x, x_domain, plot_left, plot_right), scale(y, y_domain, plot_bottom, plot_top)) for x, y in points]
        path = " ".join(("M" if item == 0 else "L") + f" {x:.2f} {y:.2f}" for item, (x, y) in enumerate(coordinates))
        body.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, y in coordinates:
            body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>')
    return base_svg("Batch size vs throughput", body)


def generate_figures(
    results_dir: Path,
    output_dir: Path,
    batch_results_dir: Path | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    quality = load_quality_series(results_dir)
    figures = {
        "recall-latency.svg": quality_figure(
            quality, "latency", "Recall–latency frontier", "1000-query latency (ms)"
        ),
        "recall-qps.svg": quality_figure(
            quality, "qps", "Recall–throughput frontier", "Throughput (QPS)"
        ),
    }
    batch_root = batch_results_dir if batch_results_dir is not None else results_dir
    if (batch_root / "optimized_v1_batch_sweep.csv").exists():
        figures["batch-qps.svg"] = batch_figure(batch_root)
    paths = []
    for filename, contents in figures.items():
        path = output_dir / filename
        path.write_text(contents)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/million"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--batch-results-dir", type=Path)
    args = parser.parse_args()
    for path in generate_figures(
        args.results_dir, args.output_dir, args.batch_results_dir
    ):
        print(path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate the Chinese figures used by the final technical report."""

from __future__ import annotations

import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
WIDTH = 1120
HEIGHT = 520

NAVY = "#142B4A"
BLUE = "#2878B5"
ORANGE = "#D95F02"
GREEN = "#2A9D8F"
GOLD = "#E9C46A"
PALE = "#EAF1F7"
GRID = "#D9DDE3"
TEXT = "#18212B"
MUTED = "#65717D"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def attrs(values: dict[str, object]) -> str:
    return " ".join(
        f'{key.rstrip("_").replace("_", "-")}="{esc(value)}"'
        for key, value in values.items()
    )


def text(x: float, y: float, value: object, css: str = "label", **extra: object) -> str:
    return f'<text x="{x:.2f}" y="{y:.2f}" class="{css}" {attrs(extra)}>{esc(value)}</text>'


def line(x1: float, y1: float, x2: float, y2: float, css: str = "grid", **extra: object) -> str:
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'class="{css}" {attrs(extra)}/>'
    )


def rect(x: float, y: float, width: float, height: float, fill: str, **extra: object) -> str:
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
        f'fill="{fill}" {attrs(extra)}/>'
    )


def circle(x: float, y: float, radius: float, fill: str) -> str:
    return (
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}" '
        'stroke="#ffffff" stroke-width="1.5"/>'
    )


def path(points: list[tuple[float, float]], color: str) -> str:
    commands = " ".join(
        f'{"M" if index == 0 else "L"} {x:.2f} {y:.2f}'
        for index, (x, y) in enumerate(points)
    )
    return f'<path d="{commands}" fill="none" stroke="{color}" stroke-width="3"/>'


def svg(title: str, body: list[str], width: int = WIDTH, height: int = HEIGHT) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            f'<title>{esc(title)}</title>',
            '<desc>图中数据由仓库内原始 CSV 自动生成。</desc>',
            '<style>.figure-title{font-size:22px;font-weight:600}.panel-title{font-size:16px;font-weight:600}.label{font-size:14px;font-weight:600}.tick{font-size:12px}.note{font-size:12px}</style>',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            *body,
            '</svg>',
            '',
        ]
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def integer_label(value: float | int) -> str:
    """Format chart counts consistently with report tables."""
    return f"{round(float(value)):,}"


def write(name: str, contents: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / name
    target.write_text(contents, encoding="utf-8")
    return target


def architecture_figure() -> str:
    body = [text(40, 42, "WarpFusion 向量检索系统的组成与数据流", "figure-title")]
    panels = [
        (36, 210, "输入", ["向量库与查询", "检索参数", "公开数据校验和"]),
        (278, 248, "CPU 端准备", ["格式与参数检查", "精确结果参考", "构建并保存 IVF 索引"]),
        (558, 300, "GPU 常驻检索", ["Exact：边算距离边选 Top-K", "IVF：选桶、扫描、合并", "每批只传查询与结果"]),
        (890, 194, "输出证据", ["ID 与分数", "召回率与延迟", "CSV、图表、报告"]),
    ]
    for x, width, heading, items in panels:
        body.append(rect(x, 78, width, 354, PALE if heading == "GPU 常驻检索" else "#F4F5F7"))
        body.append(text(x + 20, 112, heading, "panel-title"))
        for index, item in enumerate(items):
            y = 142 + index * 86
            body.append(rect(x + 20, y, width - 40, 62, "#FFFFFF"))
            body.append(text(x + width / 2, y + 37, item, "label", text_anchor="middle"))
    for x1, x2 in ((246, 278), (526, 558), (858, 890)):
        body.append(line(x1, 255, x2, 255, "axis", stroke=BLUE, stroke_width=2.4))
        body.append(text((x1 + x2) / 2, 245, "→", "label", text_anchor="middle"))
    body.append(line(55, 463, 1065, 463, "axis", stroke=GRID, stroke_width=1.2))
    body.append(text(WIDTH / 2, 494, "每一次优化都要同时通过正确性检查和同输入 A/B 对照", "note", text_anchor="middle"))
    return svg("WarpFusion 系统流程", body)


def exact_fused_figure() -> str:
    rows = read_csv(ROOT / "results/exact_fused/million_compare.csv")
    values = {(int(row["top_k"]), row["mode"]): float(row["qps"]) for row in rows}
    body = [text(WIDTH / 2, 34, "Exact 融合路径与距离矩阵基线的吞吐对比", "figure-title", text_anchor="middle")]
    left, right, top, bottom = 95, 1070, 78, 420
    maximum = 6000.0
    for tick in range(0, 6001, 1000):
        y = bottom - tick / maximum * (bottom - top)
        body.append(line(left, y, right, y))
        body.append(text(left - 12, y + 4, integer_label(tick), "tick", text_anchor="end"))
    body.append(line(left, top, left, bottom, "axis", stroke=MUTED, stroke_width=1.2))
    centers = [260, 560, 860]
    bar_width = 82
    for center, top_k in zip(centers, (10, 50, 100)):
        matrix = values[(top_k, "matrix")]
        fused = values[(top_k, "fused")]
        for offset, value, color, label in ((-52, matrix, MUTED, "距离矩阵"), (52, fused, BLUE, "融合路径")):
            height = value / maximum * (bottom - top)
            body.append(rect(center + offset - bar_width / 2, bottom - height, bar_width, height, color))
            body.append(text(center + offset, bottom - height - 10, integer_label(value), "note", text_anchor="middle"))
        body.append(text(center, 452, f"K={top_k}", "label", text_anchor="middle"))
        body.append(text(center, 482, f"加速比 {fused / matrix:.2f}×", "note", text_anchor="middle"))
    body.extend([
        rect(770, 48, 18, 12, MUTED), text(797, 60, "先写距离矩阵", "note"),
        rect(925, 48, 18, 12, BLUE), text(952, 60, "边计算边选 Top-K", "note"),
    ])
    return svg("Exact 融合路径吞吐", body)


def exact_breakdown_figure() -> str:
    datasets = [
        ("百万条 128 维合成数据", ROOT / "results/stage7_exact_profile/synthetic1m/breakdown.csv"),
        ("67.7 万条 384 维文本向量", ROOT / "results/stage7_exact_profile/yahoo-minilm-384/breakdown.csv"),
    ]
    body = [text(WIDTH / 2, 34, "Exact 融合路径的 GPU 执行时间分解", "figure-title", text_anchor="middle")]
    colors = [BLUE, GOLD, ORANGE]
    labels = ["距离与向量读取", "局部 Top-K", "最终合并"]
    for index, (label, _) in enumerate(zip(labels, colors)):
        x = 370 + index * 180
        body.append(rect(x, 49, 16, 11, colors[index]))
        body.append(text(x + 24, 60, label, "note"))
    for panel_index, (dataset_name, source) in enumerate(datasets):
        rows = read_csv(source)
        x0 = 60 + panel_index * 550
        body.append(text(x0 + 250, 94, dataset_name, "panel-title", text_anchor="middle"))
        for bar_index, row in enumerate(rows):
            x = x0 + 90 + bar_index * 145
            y = 400
            shares = [
                float(row["distance_share"]),
                float(row["local_topk_share"]),
                float(row["final_merge_share"]),
            ]
            for share, color in zip(shares, colors):
                height = share * 270
                body.append(rect(x, y - height, 72, height, color))
                y -= height
            body.append(text(x + 36, 427, f"K={row['top_k']}", "label", text_anchor="middle"))
            body.append(text(x + 36, 118, f"{shares[0] * 100:.1f}%", "note", text_anchor="middle"))
        body.append(line(x0 + 50, 400, x0 + 500, 400, "axis", stroke=MUTED, stroke_width=1.2))
    body.append(text(WIDTH / 2, 482, "柱顶数字表示“距离计算与向量读取”占 GPU 核函数流水的比例", "note", text_anchor="middle"))
    return svg("Exact 耗时分解", body)


def k50_scheduling_figure() -> str:
    rows = read_csv(ROOT / "results/formal_4090d_20260908/k50_bucket_major_sweep.csv")
    body = [text(WIDTH / 2, 34, "K=50 条件下按桶任务调度的性能影响", "figure-title", text_anchor="middle")]
    left, right, top, bottom = 95, 1070, 80, 410
    x_values = [float(row["nprobe"]) for row in rows]
    all_qps = [float(row[key]) for row in rows for key in ("probe_major_qps", "bucket_major_qps")]
    y_min, y_max = 0.0, max(all_qps) * 1.08
    for tick in range(0, 25001, 5000):
        y = bottom - (tick - y_min) / (y_max - y_min) * (bottom - top)
        body.append(line(left, y, right, y))
        body.append(text(left - 10, y + 4, integer_label(tick), "tick", text_anchor="end"))
    def px(value: float) -> float:
        return left + (value - min(x_values)) / (max(x_values) - min(x_values)) * (right - left)
    def py(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
    for value in x_values:
        x = px(value)
        body.append(line(x, top, x, bottom))
        body.append(text(x, 434, int(value), "tick", text_anchor="middle"))
    for key, color, label in (("probe_major_qps", MUTED, "按查询顺序"), ("bucket_major_qps", BLUE, "按桶重排")):
        points = [(px(float(row["nprobe"])), py(float(row[key]))) for row in rows]
        body.append(path(points, color))
        for x, y in points:
            body.append(circle(x, y, 5, color))
        body.append(line(770 if key == "probe_major_qps" else 920, 55, 795 if key == "probe_major_qps" else 945, 55, "axis", stroke=color, stroke_width=3))
        body.append(text(802 if key == "probe_major_qps" else 952, 60, label, "note"))
    last = rows[-1]
    body.append(text(px(float(last["nprobe"])) - 5, py(float(last["bucket_major_qps"])) - 14, f"+{float(last['speedup_percent']):.1f}%", "label", text_anchor="end"))
    body.append(text((left + right) / 2, 474, "搜索桶数 nprobe", "label", text_anchor="middle"))
    return svg("K=50 任务调度", body)


def k100_partitioned_figure() -> str:
    full = read_csv(ROOT / "results/real/sift1m/ivf_k100_full_heap.csv")
    partitioned = read_csv(ROOT / "results/real/sift1m/ivf_k100_partitioned.csv")
    body = [text(WIDTH / 2, 34, "K=100 条件下分区候选结构的性能影响", "figure-title", text_anchor="middle")]
    left, right, top, bottom = 95, 1070, 82, 410
    x_values = [float(row["nprobe"]) for row in full]
    maximum = max(float(row["qps"]) for row in partitioned) * 1.08
    for tick in range(0, 50001, 10000):
        y = bottom - tick / maximum * (bottom - top)
        body.append(line(left, y, right, y))
        body.append(text(left - 10, y + 4, integer_label(tick), "tick", text_anchor="end"))
    def px(value: float) -> float:
        return left + (value - min(x_values)) / (max(x_values) - min(x_values)) * (right - left)
    def py(value: float) -> float:
        return bottom - value / maximum * (bottom - top)
    for value in x_values:
        x = px(value)
        body.append(line(x, top, x, bottom))
        body.append(text(x, 434, int(value), "tick", text_anchor="middle"))
    for rows, color, label in ((full, MUTED, "完整大候选表"), (partitioned, GREEN, "4 个小候选表")):
        points = [(px(float(row["nprobe"])), py(float(row["qps"]))) for row in rows]
        body.append(path(points, color))
        for x, y in points:
            body.append(circle(x, y, 5, color))
        base_x = 780 if rows is full else 940
        body.append(line(base_x, 56, base_x + 25, 56, "axis", stroke=color, stroke_width=3))
        body.append(text(base_x + 32, 61, label, "note"))
    row160_full = next(row for row in full if int(row["nprobe"]) == 160)
    row160_part = next(row for row in partitioned if int(row["nprobe"]) == 160)
    ratio = float(row160_part["qps"]) / float(row160_full["qps"])
    body.append(text(px(160), py(float(row160_part["qps"])) - 16, f"Recall@100 相同；加速比 {ratio:.2f}×", "label", text_anchor="middle"))
    body.append(text((left + right) / 2, 474, "搜索桶数 nprobe", "label", text_anchor="middle"))
    return svg("K=100 分区候选", body)


def real_comparison_figure() -> str:
    datasets = [("SIFT1M（图像特征）", "sift1m"), ("Yahoo MiniLM（文本向量）", "yahoo-minilm-384")]
    body = [text(WIDTH / 2, 34, "公开数据集上的端到端吞吐对比（nprobe=160）", "figure-title", text_anchor="middle")]
    for panel_index, (title, directory) in enumerate(datasets):
        root = ROOT / "results/real" / directory
        faiss = read_csv(root / "faiss_gpu.csv")
        x0 = 55 + panel_index * 550
        body.append(text(x0 + 250, 80, title, "panel-title", text_anchor="middle"))
        warp_values = []
        faiss_values = []
        recalls = []
        for top_k in (10, 50, 100):
            warp = next(row for row in read_csv(root / f"ivf_k{top_k}.csv") if int(row["nprobe"]) == 160)
            baseline = next(
                row for row in faiss
                if row["index_type"] == "ivf_flat" and int(row["top_k"]) == top_k and int(row["nprobe"]) == 160
            )
            warp_values.append(float(warp["qps"]))
            faiss_values.append(float(baseline["qps"]))
            recalls.append((float(warp["recall_at_k"]), float(baseline["recall_at_k"])))
        maximum = max(warp_values + faiss_values) * 1.15
        panel_bottom, panel_top = 405, 112
        for group, top_k in enumerate((10, 50, 100)):
            center = x0 + 95 + group * 155
            for offset, value, color in ((-27, warp_values[group], BLUE), (27, faiss_values[group], ORANGE)):
                height = value / maximum * (panel_bottom - panel_top)
                body.append(rect(center + offset - 22, panel_bottom - height, 44, height, color))
            ratio = warp_values[group] / faiss_values[group]
            body.append(text(center, min(panel_bottom - warp_values[group] / maximum * (panel_bottom - panel_top), panel_bottom - faiss_values[group] / maximum * (panel_bottom - panel_top)) - 10, f"{ratio:.2f}×", "note", text_anchor="middle"))
            body.append(text(center, 432, f"K={top_k}", "label", text_anchor="middle"))
            body.append(text(center, 458, f"召回 {recalls[group][0]:.4f}/{recalls[group][1]:.4f}", "note", text_anchor="middle"))
        body.append(line(x0 + 35, panel_bottom, x0 + 505, panel_bottom, "axis", stroke=MUTED, stroke_width=1.2))
    body.extend([
        rect(780, 49, 16, 11, BLUE), text(804, 60, "WarpFusion", "note"),
        rect(920, 49, 16, 11, ORANGE), text(944, 60, "FAISS GPU", "note"),
        text(WIDTH / 2, 496, "召回数字依次为 WarpFusion / FAISS；两者各自训练聚类中心", "note", text_anchor="middle"),
    ])
    return svg("公开数据对比", body)


def recall_qps_figure() -> str:
    datasets = [
        ("SIFT1M（图像特征）", "sift1m"),
        ("Yahoo MiniLM（文本向量）", "yahoo-minilm-384"),
    ]
    body = [
        text(
            WIDTH / 2,
            34,
            "公开数据集上的召回率与吞吐权衡（K=10）",
            "figure-title",
            text_anchor="middle",
        )
    ]
    panel_top, panel_bottom = 98, 402
    panel_width = 455
    for panel_index, (title, directory) in enumerate(datasets):
        root = ROOT / "results" / "real" / directory
        warp = read_csv(root / "ivf_k10.csv")
        faiss = [
            row
            for row in read_csv(root / "faiss_gpu.csv")
            if row["index_type"] == "ivf_flat" and int(row["top_k"]) == 10
        ]
        series = ((warp, BLUE, "WarpFusion"), (faiss, ORANGE, "FAISS GPU"))
        recalls = [float(row["recall_at_k"]) for rows, _, _ in series for row in rows]
        qps_values = [float(row["qps"]) for rows, _, _ in series for row in rows]
        x_min = max(0.0, min(recalls) - 0.003)
        x_max = min(1.0, max(recalls) + 0.001)
        y_max = max(qps_values) * 1.08
        left = 78 + panel_index * 550
        right = left + panel_width

        body.append(text((left + right) / 2, 76, title, "panel-title", text_anchor="middle"))
        for tick_index in range(5):
            fraction = tick_index / 4
            x = left + fraction * panel_width
            recall = x_min + fraction * (x_max - x_min)
            body.append(line(x, panel_top, x, panel_bottom))
            body.append(text(x, 425, f"{recall * 100:.1f}%", "tick", text_anchor="middle"))
            qps_tick = y_max * fraction
            y = panel_bottom - fraction * (panel_bottom - panel_top)
            body.append(line(left, y, right, y))
            body.append(text(left - 9, y + 4, integer_label(qps_tick), "tick", text_anchor="end"))

        def px(value: float) -> float:
            return left + (value - x_min) / (x_max - x_min) * panel_width

        def py(value: float) -> float:
            return panel_bottom - value / y_max * (panel_bottom - panel_top)

        for rows, color, _ in series:
            points = [(px(float(row["recall_at_k"])), py(float(row["qps"]))) for row in rows]
            body.append(path(points, color))
            for x, y in points:
                body.append(circle(x, y, 4.5, color))
        body.append(text(left + 8, panel_top + 18, "nprobe=32", "note"))
        body.append(text(right - 8, panel_bottom - 10, "nprobe=224", "note", text_anchor="end"))
        body.append(line(left, panel_top, left, panel_bottom, "axis", stroke=MUTED, stroke_width=1.2))
        body.append(line(left, panel_bottom, right, panel_bottom, "axis", stroke=MUTED, stroke_width=1.2))

    body.extend(
        [
            line(770, 52, 795, 52, "axis", stroke=BLUE, stroke_width=3),
            text(802, 57, "WarpFusion", "note"),
            line(920, 52, 945, 52, "axis", stroke=ORANGE, stroke_width=3),
            text(952, 57, "FAISS GPU", "note"),
            text(WIDTH / 2, 458, "横轴为 Recall@10，纵轴为 QPS；每条曲线包含 nprobe=32、64、128、160、192、224", "note", text_anchor="middle"),
            text(WIDTH / 2, 486, "越靠近右上方表示在相同召回率附近具有更高吞吐", "note", text_anchor="middle"),
        ]
    )
    return svg("召回率与吞吐权衡", body)


def generate_report_figures() -> list[Path]:
    figures = {
        "report-architecture.svg": architecture_figure(),
        "report-exact-fused.svg": exact_fused_figure(),
        "report-exact-breakdown.svg": exact_breakdown_figure(),
        "report-k50-scheduling.svg": k50_scheduling_figure(),
        "report-k100-partitioned.svg": k100_partitioned_figure(),
        "report-recall-qps.svg": recall_qps_figure(),
        "report-real-comparison.svg": real_comparison_figure(),
    }
    return [write(name, contents) for name, contents in figures.items()]


if __name__ == "__main__":
    for generated in generate_report_figures():
        print(generated)

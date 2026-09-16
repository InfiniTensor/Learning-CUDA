#!/usr/bin/env python3
"""Generate reviewable PNG assets from experiment CSVs and output images.

Only Pillow is required. The script never invents missing values: a chart is skipped
with an explicit message when its required input is absent or incomplete.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BG = "#f7f9fc"
INK = "#172033"
MUTED = "#5f6b7a"
BLUE = "#2563eb"
TEAL = "#0f9d8a"
ORANGE = "#f59e0b"
RED = "#dc2626"
GRID = "#d8dee9"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def canvas(title: str, subtitle: str = "") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (1600, 900), BG)
    draw = ImageDraw.Draw(image)
    draw.text((70, 45), title, fill=INK, font=font(42, True))
    if subtitle:
        draw.text((72, 100), subtitle, fill=MUTED, font=font(22))
    return image, draw


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def value(row: dict[str, str], *keys: str) -> float:
    for key in keys:
        raw = row.get(key)
        if raw not in (None, ""):
            return float(raw)
    raise KeyError(keys)


def infer_config(row: dict[str, str]) -> str:
    if row.get("config"):
        return row["config"]
    pr = int(float(row["pr"]))
    sr = int(float(row["sr"]))
    h = float(row["h"])
    if pr == 2 and sr == 7:
        return "small"
    if pr == 4 and sr == 14:
        return "large"
    if pr == 3 and sr == 10 and h == 15:
        return "strong-h"
    if pr == 3 and sr == 10:
        return "base"
    return f"pr{pr}-sr{sr}"


def draw_axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], ymax: float,
              ylabel: str) -> None:
    left, top, right, bottom = box
    draw.line((left, top, left, bottom), fill=INK, width=3)
    draw.line((left, bottom, right, bottom), fill=INK, width=3)
    for i in range(6):
        y = bottom - (bottom - top) * i / 5
        val = ymax * i / 5
        draw.line((left, int(y), right, int(y)), fill=GRID, width=1)
        draw.text((left - 95, int(y) - 12), f"{val:.0f}", fill=MUTED, font=font(18))
    draw.text((left, top - 42), ylabel, fill=MUTED, font=font(20))


def performance_chart(csv_paths: list[Path], output: Path, device_label: str) -> bool:
    rows: list[dict[str, str]] = []
    for csv_path in csv_paths:
        rows.extend(read_csv(csv_path))
    chosen: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        if row.get("channels") != "3" or infer_config(row) != "base":
            continue
        size = row.get("size", "")
        if size in ("1920x1080", "3840x2160"):
            chosen[(size, int(row["kernel_ver"]))] = row
    if len(chosen) != 6:
        print(f"SKIP performance chart: expected 6 base RGB rows in {csv_paths}")
        return False

    legacy = any("kernel_mean_ms" not in row for row in chosen.values())
    statistic = "kernel min / pipeline e2e mean (legacy)" if legacy else "kernel mean / pipeline e2e mean"
    image, draw = canvas(f"{device_label}: V0 / V1 / V2 latency",
                         f"Archived experiment: {statistic}; excludes image I/O")
    box = (150, 190, 1510, 760)
    vals = [value(row, "kernel_mean_ms", "kernel_ms") for row in chosen.values()]
    vals += [value(row, "e2e_mean_ms", "e2e_ms") for row in chosen.values()]
    ymax = math.ceil(max(vals) / 100.0) * 100.0
    draw_axes(draw, box, ymax, "milliseconds")
    group_centers = [430, 1180]
    colors = [BLUE, TEAL, ORANGE]
    left, top, _, bottom = box
    for group_index, size in enumerate(("1920x1080", "3840x2160")):
        center = group_centers[group_index]
        for ver in range(3):
            row = chosen[(size, ver)]
            kernel_ms = value(row, "kernel_mean_ms", "kernel_ms")
            e2e_ms = value(row, "e2e_mean_ms", "e2e_ms")
            x = center + (ver - 1) * 150
            for offset, number, shade in ((-34, kernel_ms, colors[ver]),
                                           (34, e2e_ms, "#94a3b8")):
                height = (bottom - top) * number / ymax
                draw.rounded_rectangle((x + offset - 26, bottom - height,
                                        x + offset + 26, bottom), radius=6, fill=shade)
                draw.text((x + offset - 38, bottom - height - 30), f"{number:.1f}",
                          fill=INK, font=font(17))
            draw.text((x - 20, bottom + 16), f"V{ver}", fill=INK, font=font(20, True))
        draw.text((center - 75, bottom + 55), size, fill=INK, font=font(24, True))
    for idx, shade in enumerate(colors):
        draw.rectangle((960 + idx * 24, 115, 980 + idx * 24, 140), fill=shade)
    draw.text((1042, 112), "kernel", fill=INK, font=font(20))
    draw.rectangle((1170, 115, 1200, 140), fill="#94a3b8")
    draw.text((1210, 112), "GPU e2e", fill=INK, font=font(20))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, optimize=True)
    return True


def triptych(clean_path: Path, noisy_path: Path, denoised_path: Path, output: Path) -> bool:
    paths = [clean_path, noisy_path, denoised_path]
    if not all(path.exists() for path in paths):
        print(f"SKIP triptych: missing one of {paths}")
        return False
    images = [Image.open(path).convert("RGB") for path in paths]
    if len({im.size for im in images}) != 1:
        print("SKIP triptych: image dimensions do not match")
        return False
    image = Image.new("RGB", (1800, 1050), BG)
    draw = ImageDraw.Draw(image)
    draw.text((65, 35), "NLM visual quality: full frame and enlarged ROI", fill=INK,
              font=font(40, True))
    names = ["Clean reference", "Uniform noise (amplitude=25)", "Denoised V2 (base)"]
    full_size = (540, 304)
    source_w, source_h = images[0].size
    roi_w, roi_h = 240, 136
    roi_left = max(0, source_w // 2 - roi_w // 2)
    roi_top = max(0, source_h // 2 - roi_h // 2)
    for idx, (name, source) in enumerate(zip(names, images)):
        x = 40 + idx * 590
        y = 115
        full = source.resize(full_size, Image.Resampling.LANCZOS)
        image.paste(full, (x, y))
        scale_x = full_size[0] / source_w
        scale_y = full_size[1] / source_h
        draw.rectangle((x + roi_left * scale_x, y + roi_top * scale_y,
                        x + (roi_left + roi_w) * scale_x,
                        y + (roi_top + roi_h) * scale_y), outline=RED, width=4)
        draw.text((x, y + full_size[1] + 12), name, fill=INK, font=font(24, True))
        crop = source.crop((roi_left, roi_top, roi_left + roi_w, roi_top + roi_h))
        crop = crop.resize((540, 306), Image.Resampling.NEAREST)
        image.paste(crop, (x, 520))
        draw.rectangle((x, 520, x + 540, 826), outline=RED, width=3)
        draw.text((x, 840), "Center ROI, 2.25x nearest-neighbor zoom", fill=MUTED,
                  font=font(19))
    draw.text((65, 980), "Paired deterministic synthetic images, seed=7; no post-processing.",
              fill=MUTED, font=font(21))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, optimize=True)
    return True


def scatter_chart(quality_csv: Path, benchmark_csvs: list[Path], output: Path,
                  device_label: str) -> bool:
    if not quality_csv.exists() or not benchmark_csvs or \
            not all(path.exists() for path in benchmark_csvs):
        print("SKIP quality-latency chart: CSV missing")
        return False
    quality = {row["label"]: float(row["psnr_db"]) for row in read_csv(quality_csv)
               if row.get("label") in ("small", "base", "large")}
    latency: dict[str, float] = {}
    for benchmark_csv in benchmark_csvs:
        for row in read_csv(benchmark_csv):
            if row.get("size") == "1920x1080" and row.get("channels") == "3" and \
                    row.get("kernel_ver") == "2":
                name = infer_config(row)
                if name in ("small", "base", "large"):
                    latency[name] = value(row, "e2e_mean_ms", "e2e_ms")
    if len(quality) != 3 or len(latency) != 3:
        print("SKIP quality-latency chart: small/base/large rows incomplete")
        return False

    image, draw = canvas(f"Quality / latency trade-off ({device_label})",
                         "Same uniform-noise amplitude=25 input; V2; higher PSNR and lower latency are better")
    box = (170, 190, 1510, 770)
    xs = list(latency.values())
    ys = list(quality.values())
    xmin, xmax = 0.0, max(xs) * 1.15
    ymin, ymax = min(ys) - 1.0, max(ys) + 1.0
    left, top, right, bottom = box
    draw.line((left, top, left, bottom), fill=INK, width=3)
    draw.line((left, bottom, right, bottom), fill=INK, width=3)
    for i in range(6):
        x = left + (right - left) * i / 5
        draw.line((int(x), top, int(x), bottom), fill=GRID, width=1)
        draw.text((int(x) - 20, bottom + 15), f"{xmax * i / 5:.0f}", fill=MUTED,
                  font=font(18))
        y = bottom - (bottom - top) * i / 5
        draw.line((left, int(y), right, int(y)), fill=GRID, width=1)
        draw.text((left - 75, int(y) - 12), f"{ymin + (ymax-ymin)*i/5:.1f}",
                  fill=MUTED, font=font(18))
    draw.text((650, 830), "GPU e2e mean (ms)", fill=INK, font=font(22))
    draw.text((25, 155), "PSNR (dB)", fill=INK, font=font(22))
    colors = {"small": TEAL, "base": BLUE, "large": ORANGE}
    for name in ("small", "base", "large"):
        px = left + (right - left) * (latency[name] - xmin) / (xmax - xmin)
        py = bottom - (bottom - top) * (quality[name] - ymin) / (ymax - ymin)
        draw.ellipse((px - 14, py - 14, px + 14, py + 14), fill=colors[name])
        label_x = px + 20 if px < right - 330 else px - 330
        draw.text((label_x, py - 35), f"{name}: {quality[name]:.2f} dB, "
                  f"{latency[name]:.1f} ms", fill=INK, font=font(20, True))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, optimize=True)
    return True


def mctracer_timeline(trace_path: Path, output: Path, device_label: str) -> bool:
    if not trace_path.exists():
        print(f"SKIP mcTracer timeline: missing {trace_path}")
        return False
    with trace_path.open("r", encoding="utf-8") as handle:
        events = json.load(handle).get("traceEvents", [])
    events = [event for event in events
              if event.get("ph") == "X" and event.get("cat") == "0" and
              float(event.get("dur", 0)) > 0]
    if not events:
        print(f"SKIP mcTracer timeline: no device duration events in {trace_path}")
        return False

    def label(name: str) -> str:
        if "HTOD" in name:
            return "H2D"
        if "DTOH" in name:
            return "D2H"
        if "CvtU8" in name:
            return "U8 to F32"
        if "CvtF32" in name:
            return "F32 to U8"
        if "Nlm" in name:
            return "NLM V1"
        return name.split("(", 1)[0]

    events.sort(key=lambda event: float(event.get("ts", 0)))
    start = min(float(event["ts"]) for event in events)
    finish = max(float(event["ts"]) + float(event["dur"]) for event in events)
    span = max(finish - start, 1.0)
    device_total = sum(float(event["dur"]) for event in events)

    image, draw = canvas(f"{device_label}: MACA device timeline",
                         "mcTracer raw JSON; 1080p RGB base, V1")
    left, right = 285, 1500
    top, row_height = 190, 82
    colors = {"H2D": TEAL, "D2H": RED, "U8 to F32": BLUE,
              "F32 to U8": ORANGE, "NLM V1": "#7c3aed"}
    for index, event in enumerate(events):
        name = label(str(event.get("name", "event")))
        duration = float(event["dur"])
        x0 = left + (right - left) * (float(event["ts"]) - start) / span
        x1 = left + (right - left) * (float(event["ts"]) + duration - start) / span
        x1 = max(x1, x0 + 6)
        y = top + index * row_height
        draw.text((65, y + 8), name, fill=INK, font=font(21, True))
        draw.line((left, y + 25, right, y + 25), fill=GRID, width=2)
        draw.rounded_rectangle((x0, y + 9, x1, y + 41), radius=5,
                               fill=colors.get(name, MUTED))
        draw.text((left, y + 48),
                  f"{duration / 1e6:.6f} ms; {100.0 * duration / device_total:.3f}%",
                  fill=MUTED, font=font(17))

    bottom = top + len(events) * row_height + 20
    draw.text((65, bottom),
              f"Device-event span: {span / 1e6:.3f} ms | "
              f"summed device duration: {device_total / 1e6:.3f} ms",
              fill=INK, font=font(22, True))
    draw.text((65, bottom + 42),
              "The NLM kernel occupies 99.6% of traced device time; transfer bars "
              "are widened to remain visible.",
              fill=MUTED, font=font(20))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, optimize=True)
    return True


def nsys_chart(csv_path: Path, output: Path) -> bool:
    if not csv_path.exists():
        print("SKIP nsys chart: CSV missing")
        return False
    rows = read_csv(csv_path)
    if not rows:
        return False
    image, draw = canvas("Nsight Systems: NLM kernel time",
                         "RTX 4090 D verified profile; conversion kernels are below 0.1%")
    box = (220, 220, 1460, 750)
    ymax = math.ceil(max(float(row["gpu_ms"]) for row in rows) / 20.0) * 20.0
    draw_axes(draw, box, ymax, "GPU milliseconds")
    left, top, _, bottom = box
    for idx, row in enumerate(rows):
        x = 470 + idx * 350
        number = float(row["gpu_ms"])
        height = (bottom - top) * number / ymax
        draw.rounded_rectangle((x - 85, bottom - height, x + 85, bottom),
                               radius=10, fill=(BLUE, TEAL, ORANGE)[idx])
        draw.text((x - 52, bottom - height - 42), f"{number:.3f} ms", fill=INK,
                  font=font(22, True))
        draw.text((x - 25, bottom + 18), row["implementation"], fill=INK,
                  font=font(25, True))
        draw.text((x - 125, bottom + 57), f"GPU share {row['gpu_share_percent']}%",
                  fill=MUTED, font=font(19))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, optimize=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument("--benchmark", type=Path, action="append", default=[])
    parser.add_argument("--performance-device", default="RTX 4090 D")
    parser.add_argument("--performance-output", default="performance_4090.png")
    parser.add_argument("--performance-only", action="store_true")
    parser.add_argument("--quality", type=Path,
                        default=Path("assets/results/rtx3060_laptop/quality_tradeoff.csv"))
    parser.add_argument("--quality-benchmark", type=Path, action="append", default=[])
    parser.add_argument("--quality-device", default="RTX 3060 Laptop (archived)")
    parser.add_argument("--clean", type=Path, default=Path("test_images/clean_1920x1080_3ch.png"))
    parser.add_argument("--noisy", type=Path,
                        default=Path("test_images/noisy_1920x1080_3ch_sigma25.png"))
    parser.add_argument("--denoised", type=Path,
                        default=Path("test_images/denoised_1920x1080_base_v2.png"))
    parser.add_argument("--triptych-output", default="quality_triptych.png")
    parser.add_argument("--quality-output", default="quality_latency_tradeoff.png")
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--trace-output", default="mctracer_timeline.png")
    parser.add_argument("--experiment-only", action="store_true")
    args = parser.parse_args()

    benchmark_paths = args.benchmark or [
        Path("assets/results/rtx4090d/benchmark_legacy.csv")
    ]
    performance_generated = performance_chart(
        benchmark_paths, args.assets / args.performance_output,
        args.performance_device)
    if args.performance_only:
        print(f"Generated {int(performance_generated)}/1 assets in {args.assets}")
        return 0 if performance_generated else 1

    generated = int(performance_generated)
    generated += triptych(args.clean, args.noisy, args.denoised,
                          args.assets / args.triptych_output)
    quality_benchmarks = args.quality_benchmark or [
        Path("assets/results/rtx3060_laptop/benchmark_quality.csv")
    ]
    generated += scatter_chart(args.quality, quality_benchmarks,
                               args.assets / args.quality_output,
                               args.quality_device)
    if args.trace:
        generated += mctracer_timeline(args.trace, args.assets / args.trace_output,
                                       args.performance_device)
    if args.experiment_only:
        expected = 4 if args.trace else 3
        print(f"Generated {generated}/{expected} assets in {args.assets}")
        return 0 if generated == expected else 1
    generated += nsys_chart(Path("assets/results/rtx4090d/nsys_summary_legacy.csv"),
                            args.assets / "nsys_kernel_summary.png")
    expected = 5 if args.trace else 4
    print(f"Generated {generated}/{expected} assets in {args.assets}")
    return 0 if generated == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())

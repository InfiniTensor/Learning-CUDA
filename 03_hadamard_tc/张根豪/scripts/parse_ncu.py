#!/usr/bin/env python3
"""Turn Nsight Compute raw CSV exports into one compact, reviewable table.

The parser intentionally accepts metric-name variation across Nsight releases.
It uses the algorithmic FHT work (N*d*log2(d) scalar add/sub operations) for
throughput and Nsight's measured DRAM bytes for arithmetic intensity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


NAME_RE = re.compile(r"(?P<dtype>fp16|bf16)_hd(?P<head_dim>\d+)_tokens(?P<tokens>\d+)")


def number(value: str) -> float | None:
    value = value.strip().replace(",", "").replace("%", "")
    if not value or value.lower() in {"n/a", "nan", "not available"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def scaled(value: str, unit: str, target: str) -> float | None:
    parsed = number(value)
    if parsed is None:
        return None
    u = unit.strip().lower().replace(" ", "")
    if target == "seconds":
        factors = {"second": 1.0, "msecond": 1e-3, "usecond": 1e-6,
                   "nsecond": 1e-9, "s": 1.0, "ms": 1e-3, "us": 1e-6,
                   "ns": 1e-9}
    elif target == "bytes":
        factors = {"byte": 1.0, "kbyte": 1e3, "mbyte": 1e6, "gbyte": 1e9,
                   "bytes": 1.0, "kb": 1e3, "mb": 1e6, "gb": 1e9}
    else:
        factors = {}
    return parsed * factors.get(u, 1.0)


def load_metrics(path: Path) -> dict[str, tuple[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as source:
        rows = list(csv.reader(source))
    header_index = next(
        (i for i, row in enumerate(rows) if "Metric Name" in row and "Metric Value" in row),
        None,
    )
    if header_index is None:
        raise ValueError(f"{path}: no Nsight Compute metric header found")
    header = rows[header_index]
    name_i = header.index("Metric Name")
    value_i = header.index("Metric Value")
    unit_i = header.index("Metric Unit") if "Metric Unit" in header else None
    metrics: dict[str, tuple[str, str]] = {}
    for row in rows[header_index + 1 :]:
        if len(row) <= max(name_i, value_i) or row[name_i] == "Metric Name":
            continue
        name = row[name_i].strip()
        if name:
            metrics[name] = (row[value_i], row[unit_i] if unit_i is not None and len(row) > unit_i else "")
    return metrics


def first(metrics: dict[str, tuple[str, str]], names: list[str]) -> tuple[str, str] | None:
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


def metric_number(metrics: dict[str, tuple[str, str]], names: list[str]) -> float | None:
    item = first(metrics, names)
    return number(item[0]) if item else None


def metric_scaled(metrics: dict[str, tuple[str, str]], names: list[str], target: str) -> float | None:
    item = first(metrics, names)
    return scaled(item[0], item[1], target) if item else None


def fp32_cores_per_sm(major: int, minor: int) -> int:
    # FP32 lanes per SM for the architectures used by this project/cluster.
    table = {(7, 0): 64, (7, 5): 64, (8, 0): 64, (8, 6): 128,
             (8, 7): 128, (8, 9): 128, (9, 0): 128}
    if (major, minor) not in table:
        raise ValueError(f"unknown FP32 lanes/SM for compute capability {major}.{minor}")
    return table[(major, minor)]


def theoretical_peaks(device: dict) -> tuple[float, float]:
    cores = fp32_cores_per_sm(
        int(device["compute_capability_major"]), int(device["compute_capability_minor"])
    )
    fp32_tflops = (
        int(device["sm_count"]) * cores * 2 * float(device["clock_rate_khz"]) * 1e3 / 1e12
    )
    memory_gbs = (
        float(device["memory_clock_rate_khz"]) * 1e3 * 2
        * float(device["memory_bus_width_bits"]) / 8 / 1e9
    )
    return fp32_tflops, memory_gbs


def optional(value: float | None) -> str:
    return "" if value is None else f"{value:.9g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--device-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = json.loads(args.device_json.read_text())
    peak_fp32_tflops, peak_memory_gbs = theoretical_peaks(device)
    output_rows = []
    errors = []
    for path in sorted(args.input_dir.glob("*.csv")):
        match = NAME_RE.search(path.stem)
        if not match:
            continue
        try:
            metrics = load_metrics(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue

        dtype = match.group("dtype")
        head_dim = int(match.group("head_dim"))
        tokens = int(match.group("tokens"))
        duration_s = metric_scaled(metrics, ["gpu__time_duration.sum"], "seconds")
        read_bytes = metric_scaled(metrics, ["dram__bytes_read.sum"], "bytes")
        write_bytes = metric_scaled(metrics, ["dram__bytes_write.sum"], "bytes")
        total_bytes = metric_scaled(metrics, ["dram__bytes.sum"], "bytes")
        if total_bytes is None and read_bytes is not None and write_bytes is not None:
            total_bytes = read_bytes + write_bytes

        min_io_bytes = float(tokens * head_dim * 4)  # one 16-bit read + one 16-bit write
        work_flops = float(tokens * head_dim * int(math.log2(head_dim)))
        ai_bytes = total_bytes if total_bytes and total_bytes > 0 else min_io_bytes
        byte_source = "ncu_dram" if total_bytes and total_bytes > 0 else "minimum_io_fallback"

        shared_ld = metric_number(metrics, [
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"
        ])
        shared_st = metric_number(metrics, [
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"
        ])
        shared_conflicts = None
        if shared_ld is not None or shared_st is not None:
            shared_conflicts = (shared_ld or 0.0) + (shared_st or 0.0)

        output_rows.append({
            "gpu_name": device["name"],
            "compute_capability": f'{device["compute_capability_major"]}.{device["compute_capability_minor"]}',
            "dtype": dtype,
            "head_dim": head_dim,
            "total_tokens": tokens,
            "kernel_duration_us": optional(duration_s * 1e6 if duration_s else None),
            "algorithmic_gflops": optional(work_flops / duration_s / 1e9 if duration_s else None),
            "arithmetic_intensity_flop_per_byte": optional(work_flops / ai_bytes),
            "dram_bytes": optional(total_bytes),
            "minimum_io_bytes": optional(min_io_bytes),
            "intensity_byte_source": byte_source,
            "effective_io_bandwidth_gbs": optional(min_io_bytes / duration_s / 1e9 if duration_s else None),
            "measured_dram_bandwidth_gbs": optional(total_bytes / duration_s / 1e9 if duration_s and total_bytes else None),
            "achieved_occupancy_pct": optional(metric_number(metrics, [
                "sm__warps_active.avg.pct_of_peak_sustained_active",
                "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
            ])),
            "sm_throughput_pct": optional(metric_number(metrics, [
                "sm__throughput.avg.pct_of_peak_sustained_elapsed"
            ])),
            "dram_throughput_pct": optional(metric_number(metrics, [
                "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
                "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            ])),
            "registers_per_thread": optional(metric_number(metrics, ["launch__registers_per_thread"])),
            "shared_bank_conflicts": optional(shared_conflicts),
            "barrier_stall_pct": optional(metric_number(metrics, [
                "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
                "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
            ])),
            "long_scoreboard_stall_pct": optional(metric_number(metrics, [
                "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
                "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
            ])),
            "theoretical_fp32_peak_tflops": f"{peak_fp32_tflops:.9g}",
            "theoretical_memory_peak_gbs": f"{peak_memory_gbs:.9g}",
            "source_file": path.name,
        })

    if not output_rows:
        details = "\n".join(errors) if errors else f"no matching CSV files in {args.input_dir}"
        raise SystemExit(f"no usable Nsight Compute rows: {details}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} rows to {args.output}")
    if errors:
        print("warnings:")
        print("\n".join(errors))


if __name__ == "__main__":
    main()

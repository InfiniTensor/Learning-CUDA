#!/usr/bin/env python3
"""Extract the key hardware counters from the Tensor Core NCU raw CSVs.

`slurm/tc_ncu_h200.slurm` 为 warp / tc_fast / tc_split 各导出一份
`--page raw` CSV。本脚本只挑出用于结构诊断的那几列，输出
ncu_summary.csv 与 ncu_summary.md，避免在报告里手抄 600 多列。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

TARGETS = ["warp", "tc_fast", "tc_split"]

# (NCU 指标名, 表头, 小数位)
METRICS = [
    ("gpu__time_duration.avg", "NCU duration us", 3),
    ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "SM throughput %", 2),
    ("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", "DRAM throughput %", 2),
    ("dram__bytes.sum.per_second", "DRAM GB/s", 1),
    ("sm__warps_active.avg.pct_of_peak_sustained_active", "achieved occupancy %", 2),
    ("lts__t_sector_hit_rate.pct", "L2 hit %", 2),
    ("launch__registers_per_thread", "registers/thread", 0),
    # NCU 这一列的数值以 Kbyte 计（9.728 == 9728 B），与 nsys 导出的字节数一致。
    ("launch__shared_mem_per_block_dynamic", "dynamic shared KB/block", 3),
    ("smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
     "stall long scoreboard", 2),
    ("smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
     "stall short scoreboard", 2),
    ("smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
     "stall mio throttle", 2),
    ("smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
     "stall barrier", 2),
]


def load(path: Path) -> dict[str, str]:
    rows = list(csv.DictReader(path.open(newline="")))
    if len(rows) < 2:
        raise SystemExit(f"{path} 没有数据行（第 0 行是单位行）")
    return rows[1]


def fmt(value: str, digits: int) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncu-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.ncu_dir.parent

    data = {}
    for target in TARGETS:
        path = args.ncu_dir / f"{target}_raw.csv"
        if path.exists():
            data[target] = load(path)
    if not data:
        raise SystemExit(f"{args.ncu_dir} 下没有 *_raw.csv")

    names = list(data)
    with (output_dir / "ncu_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric"] + names)
        for key, label, digits in METRICS:
            writer.writerow([label] + [fmt(data[n].get(key, ""), digits) for n in names])

    lines = [
        "# Tensor Core kernel 的 NCU 硬件计数器",
        "",
        f"kernel: {', '.join(names)}；配置 FP16 / head_dim=128 / 16384 tokens。",
        "计数器用于 kernel 结构诊断，不与无 profiler 的端到端计时混算。",
        "",
        "| 指标 | " + " | ".join(names) + " |",
        "|---|" + "---:|" * len(names),
    ]
    for key, label, digits in METRICS:
        lines.append(
            f"| {label} | " + " | ".join(fmt(data[n].get(key, ""), digits) for n in names) + " |"
        )
    lines.append("")
    (output_dir / "ncu_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[summarize_tc_ncu] wrote {output_dir}/ncu_summary.csv and ncu_summary.md")


if __name__ == "__main__":
    main()

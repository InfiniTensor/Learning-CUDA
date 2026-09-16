#!/usr/bin/env python3
"""扫描 score-mass 自适应 nprobe 参数并输出质量、性能和实际 probe CSV。"""

import argparse
import csv
import re
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-binary", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--exact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--max-nprobe", type=int, required=True)
    parser.add_argument("--min-nprobe", type=int, required=True)
    parser.add_argument("--step", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--execution", choices=("masked", "grouped"),
                        default="masked")
    parser.add_argument("--target-masses", required=True,
                        help="逗号分隔，例如 0.70,0.80,0.90")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    return parser.parse_args()


def read_ids(path: Path) -> list[list[int]]:
    rows: dict[int, list[int]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"{path}:{line_number}: 结果行必须有三列")
        query_id, vector_id = int(fields[0]), int(fields[1])
        rows.setdefault(query_id, []).append(vector_id)
    return [rows[query_id] for query_id in sorted(rows)]


def recall(exact: list[list[int]], approximate: list[list[int]]) -> tuple[float, float]:
    if len(exact) != len(approximate) or not exact:
        raise ValueError("exact 与 approximate 的 query 数不一致或为空")
    recalls = []
    for expected, actual in zip(exact, approximate):
        if len(expected) != len(actual) or not expected:
            raise ValueError("exact 与 approximate 的 K 不一致或为空")
        recalls.append(len(set(expected) & set(actual)) / len(expected))
    return sum(recalls) / len(recalls), min(recalls)


def extract(text: str, pattern: str) -> float:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"无法从 ivf_search 输出解析字段: {pattern}")
    return float(match.group(1))


def extract_pair(text: str, pattern: str) -> tuple[float, float]:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"无法从 ivf_search 输出解析字段: {pattern}")
    return float(match.group(1)), float(match.group(2))


def extract_triple(text: str, pattern: str) -> tuple[float, float, float]:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"无法从 ivf_search 输出解析字段: {pattern}")
    return float(match.group(1)), float(match.group(2)), float(match.group(3))


def main() -> None:
    args = parse_args()
    masses = [float(value) for value in args.target_masses.split(",")]
    if (not masses or any(value <= 0.0 or value > 1.0 for value in masses) or
            args.min_nprobe <= 0 or args.min_nprobe > args.max_nprobe or
            args.step <= 0 or args.temperature <= 0.0):
        raise ValueError("自适应 nprobe 参数不合法")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exact_ids = read_ids(Path(args.exact))
    rows = []
    for target_mass in masses:
        label = f"{target_mass:.4f}".rstrip("0").rstrip(".").replace(".", "p")
        result_path = output_dir / f"mass_{label}.txt"
        counts_path = output_dir / f"mass_{label}_nprobes.txt"
        command = [
            args.search_binary,
            "--index", args.index,
            "--queries", args.queries,
            "--params", args.params,
            "--nprobe", str(args.max_nprobe),
            "--nprobe-policy", "score_mass",
            "--adaptive-execution", args.execution,
            "--adaptive-nprobe-min", str(args.min_nprobe),
            "--adaptive-nprobe-step", str(args.step),
            "--adaptive-target-mass", str(target_mass),
            "--adaptive-temperature", str(args.temperature),
            "--backend", "gpu",
            "--warmup", str(args.warmup),
            "--repeat", str(args.repeat),
            "--output", str(result_path),
            "--probe-counts-output", str(counts_path),
        ]
        completed = subprocess.run(command, check=True, text=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        print(completed.stdout, end="")
        approximate_ids = read_ids(result_path)
        recall_at_k, minimum_recall = recall(exact_ids, approximate_ids)
        actual_min, actual_average, actual_max = extract_triple(
            completed.stdout,
            r"实际 nprobe min/avg/max: ([0-9.eE+-]+)/([0-9.eE+-]+)/([0-9.eE+-]+)",
        )
        batch_p50_ms, batch_p99_ms = extract_pair(
            completed.stdout, r"batch P50/P99: ([0-9.eE+-]+)/([0-9.eE+-]+) ms"
        )
        row = {
            "policy": "score_mass",
            "execution": args.execution,
            "target_mass": target_mass,
            "temperature": args.temperature,
            "min_nprobe": args.min_nprobe,
            "max_nprobe": args.max_nprobe,
            "step": args.step,
            "actual_nprobe_min": actual_min,
            "actual_nprobe_average": actual_average,
            "actual_nprobe_max": actual_max,
            "average_ms": extract(completed.stdout, r"平均查询: ([0-9.eE+-]+) ms"),
            "qps": extract(completed.stdout, r"QPS: ([0-9.eE+-]+)"),
            "batch_p50_ms": batch_p50_ms,
            "batch_p99_ms": batch_p99_ms,
            "recall_at_k": recall_at_k,
            "min_query_recall": minimum_recall,
            "center_ms": extract(
                completed.stdout, r"平均 GPU 中心选择: ([0-9.eE+-]+) ms"
            ),
            "policy_ms": extract(
                completed.stdout, r"平均自适应分层开销: ([0-9.eE+-]+) ms"
            ),
            "scan_ms": extract(
                completed.stdout, r"平均 GPU 桶扫描\+局部 Top-K: ([0-9.eE+-]+) ms"
            ),
            "merge_ms": extract(
                completed.stdout, r"平均 GPU 最终 Top-K 归并: ([0-9.eE+-]+) ms"
            ),
            "gpu_buffer_mib": extract(
                completed.stdout, r"GPU 缓冲区: ([0-9.eE+-]+) MiB"
            ),
        }
        rows.append(row)
        with csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"recall={recall_at_k:.6f}, actual_avg_nprobe={actual_average:.3f}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()

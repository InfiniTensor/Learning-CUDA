#!/usr/bin/env python3
"""Compare matrix-based and fused exact-search modes and write a CSV."""

import argparse
import csv
import re
import subprocess
from pathlib import Path


def extract(text: str, pattern: str) -> float:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"无法解析字段: {pattern}")
    return float(match.group(1))


def read_results(path: Path) -> dict[int, dict[int, float]]:
    queries: dict[int, dict[int, float]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"{path}:{line_number}: 结果行必须有三列")
        query_id, vector_id, score = int(fields[0]), int(fields[1]), float(fields[2])
        queries.setdefault(query_id, {})[vector_id] = score
    return queries


def compare_results(expected_path: Path, actual_path: Path,
                    tolerance: float) -> tuple[float, float]:
    expected = read_results(expected_path)
    actual = read_results(actual_path)
    if expected.keys() != actual.keys():
        raise ValueError("baseline 与 fused 的 query 集合不同")
    maximum_error = 0.0
    error_sum = 0.0
    error_count = 0
    for query_id in expected:
        if expected[query_id].keys() != actual[query_id].keys():
            raise ValueError(f"query {query_id} 的 Top-K 候选集合不同")
        for vector_id, score in expected[query_id].items():
            error = abs(score - actual[query_id][vector_id])
            maximum_error = max(maximum_error, error)
            error_sum += error
            error_count += 1
    if maximum_error > tolerance:
        raise ValueError(
            f"最大分数误差 {maximum_error} 超过容差 {tolerance}"
        )
    return error_sum / error_count if error_count else 0.0, maximum_error


def run_search(args: argparse.Namespace, config: str, output: Path) -> tuple[str, dict]:
    command = [
        args.search_binary,
        "--database", args.database,
        "--queries", args.queries,
        "--params", config,
        "--backend", "gpu",
        "--output", str(output),
        "--warmup", str(args.warmup),
        "--repeat", str(args.repeat),
    ]
    completed = subprocess.run(
        command, check=True, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    text = completed.stdout
    print(text, end="")
    row = {
        "top_k": int(extract(text, r"top_k: (\d+)")),
        "average_ms": extract(text, r"平均常驻查询时间: ([0-9.eE+-]+) ms"),
        "qps": extract(text, r"常驻 QPS: ([0-9.eE+-]+)"),
        "run_p50_ms": extract(text, r"run P50: ([0-9.eE+-]+) ms"),
        "run_p99_ms": extract(text, r"run P99: ([0-9.eE+-]+) ms"),
        "run_samples": int(extract(text, r"run samples: (\d+)")),
        "batch_latency_samples": int(extract(text, r"batch samples: (\d+)")),
        "gpu_buffer_mib": extract(text, r"GPU 缓冲区: ([0-9.eE+-]+) MiB"),
        "initialization_ms": extract(
            text, r"GPU 一次性初始化总时间: ([0-9.eE+-]+) ms"
        ),
        "distance_or_fused_ms": extract(
            text,
            r"平均(?:距离|融合距离\+局部 Top-K) kernel: ([0-9.eE+-]+) ms",
        ),
        "final_topk_ms": extract(
            text, r"平均最终 Top-K kernel: ([0-9.eE+-]+) ms"
        ),
    }
    return text, row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-binary", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--baseline-configs", required=True)
    parser.add_argument("--fused-configs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=5e-5)
    args = parser.parse_args()

    baseline_configs = args.baseline_configs.split(",")
    fused_configs = args.fused_configs.split(",")
    if len(baseline_configs) != len(fused_configs) or not baseline_configs:
        parser.error("baseline-configs 与 fused-configs 必须是一一对应的非空列表")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for baseline_config, fused_config in zip(baseline_configs, fused_configs):
        baseline_path = output_dir / f"baseline_{len(rows) // 2}.txt"
        _, baseline = run_search(args, baseline_config, baseline_path)
        baseline["mode"] = "matrix"
        baseline["mean_absolute_score_error"] = 0.0
        baseline["maximum_score_error"] = 0.0

        fused_path = output_dir / f"fused_{len(rows) // 2}.txt"
        _, fused = run_search(args, fused_config, fused_path)
        if baseline["top_k"] != fused["top_k"]:
            raise ValueError("成对配置的 top_k 不一致")
        fused["mode"] = "fused"
        fused["mean_absolute_score_error"], fused["maximum_score_error"] = compare_results(
            baseline_path, fused_path, args.tolerance
        )
        rows.extend((baseline, fused))

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "top_k", "mode", "average_ms", "qps", "run_p50_ms", "run_p99_ms",
        "run_samples", "batch_latency_samples", "gpu_buffer_mib",
        "initialization_ms", "distance_or_fused_ms", "final_topk_ms",
        "mean_absolute_score_error", "maximum_score_error",
    ]
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()

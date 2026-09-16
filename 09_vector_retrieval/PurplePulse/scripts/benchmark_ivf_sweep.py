#!/usr/bin/env python3
"""运行 IVF nprobe 扫描并把性能、显存和 recall 汇总为 CSV。"""

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
    parser.add_argument("--nlist", type=int, required=True)
    parser.add_argument("--training-samples", type=int, default=0)
    parser.add_argument("--training-iterations", type=int, default=0)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--nprobes", required=True,
                        help="逗号分隔，例如 64,128,192")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    return parser.parse_args()


def read_results(path: Path) -> list[list[tuple[int, float]]]:
    rows: dict[int, list[tuple[int, float]]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"{path}:{line_number}: 结果行必须有三列")
        query_id, vector_id, score = int(fields[0]), int(fields[1]), float(fields[2])
        rows.setdefault(query_id, []).append((vector_id, score))
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


def quality_metrics(
    exact: list[list[tuple[int, float]]],
    approximate: list[list[tuple[int, float]]],
) -> tuple[float, float, float, float]:
    if len(exact) != len(approximate) or not exact:
        raise ValueError("exact 与 approximate 的 query 数不一致或为空")
    exact_ids = [[item[0] for item in query] for query in exact]
    approximate_ids = [[item[0] for item in query] for query in approximate]
    recall_at_k, minimum_recall = recall(exact_ids, approximate_ids)
    errors = []
    for expected, actual in zip(exact, approximate):
        if len(expected) != len(actual) or not expected:
            raise ValueError("exact 与 approximate 的 K 不一致或为空")
        errors.extend(
            abs(expected_item[1] - actual_item[1])
            for expected_item, actual_item in zip(expected, actual)
        )
    return (
        recall_at_k,
        minimum_recall,
        sum(errors) / len(errors),
        max(errors),
    )


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


def read_params(path: Path) -> dict[str, str]:
    params = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), 1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected key = value")
        key, value = (part.strip() for part in line.split("=", 1))
        params[key] = value
    return params


def main() -> None:
    args = parse_args()
    nprobes = [int(value) for value in args.nprobes.split(",")]
    if not nprobes or any(value <= 0 or value > args.nlist for value in nprobes):
        raise ValueError("nprobes 必须位于 1..nlist")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exact_results = read_results(Path(args.exact))
    params = read_params(Path(args.params))
    rows = []
    for nprobe in nprobes:
        result_path = output_dir / f"nlist{args.nlist}_nprobe{nprobe}.txt"
        command = [
            args.search_binary,
            "--index", args.index,
            "--queries", args.queries,
            "--params", args.params,
            "--nlist", str(args.nlist),
            "--nprobe", str(nprobe),
            "--backend", "gpu",
            "--warmup", str(args.warmup),
            "--repeat", str(args.repeat),
            "--output", str(result_path),
        ]
        completed = subprocess.run(command, check=True, text=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        print(completed.stdout, end="")
        approximate_results = read_results(result_path)
        recall_at_k, minimum_recall, mean_score_error, max_score_error = (
            quality_metrics(exact_results, approximate_results)
        )
        top_k = int(extract(completed.stdout, r"nlist/nprobe/top_k: \d+/\d+/(\d+)"))
        run_p50_ms, run_p99_ms = extract_pair(
            completed.stdout, r"run P50/P99: ([0-9.eE+-]+)/([0-9.eE+-]+) ms"
        )
        batch_p50_ms, batch_p99_ms = extract_pair(
            completed.stdout, r"batch P50/P99: ([0-9.eE+-]+)/([0-9.eE+-]+) ms"
        )
        row = {
            "implementation": "WarpFusion",
            "backend": "gpu",
            "num_vectors": int(extract(completed.stdout, r"索引: (\d+) x \d+")),
            "num_queries": len(approximate_results),
            "dim": int(extract(completed.stdout, r"索引: \d+ x (\d+)")),
            "metric": re.search(
                r"dtype/metric: [^/]+/([^\n]+)", completed.stdout
            ).group(1),
            "batch_size": int(params["batch_size"]),
            "distance_mode": params["distance_mode"],
            "nlist": args.nlist,
            "nprobe": nprobe,
            "training_samples": args.training_samples,
            "training_iterations": args.training_iterations,
            "training_seed": args.training_seed,
            "top_k": top_k,
            "average_ms": extract(completed.stdout, r"平均查询: ([0-9.eE+-]+) ms"),
            "qps": extract(completed.stdout, r"QPS: ([0-9.eE+-]+)"),
            "run_p50_ms": run_p50_ms,
            "run_p99_ms": run_p99_ms,
            "batch_p50_ms": batch_p50_ms,
            "batch_p99_ms": batch_p99_ms,
            "recall_at_k": recall_at_k,
            "min_query_recall": minimum_recall,
            "mean_absolute_score_error": mean_score_error,
            "max_absolute_score_error": max_score_error,
            "run_samples": int(extract(
                completed.stdout, r"run samples: (\d+)"
            )),
            "batch_latency_samples": int(extract(
                completed.stdout, r"batch samples: (\d+)"
            )),
            "index_load_ms": extract(completed.stdout, r"索引加载: ([0-9.eE+-]+) ms"),
            "gpu_initialization_ms": extract(
                completed.stdout, r"GPU 初始化: ([0-9.eE+-]+) ms"
            ),
            "index_h2d_ms": extract(completed.stdout, r"索引 H2D: ([0-9.eE+-]+) ms"),
            "gpu_buffer_mib": extract(completed.stdout, r"GPU 缓冲区: ([0-9.eE+-]+) MiB"),
            "center_ms": extract(completed.stdout, r"平均 GPU 中心选择: ([0-9.eE+-]+) ms"),
            "scan_ms": extract(completed.stdout, r"平均 GPU 桶扫描\+局部 Top-K: ([0-9.eE+-]+) ms"),
            "merge_ms": extract(completed.stdout, r"平均 GPU 最终 Top-K 归并: ([0-9.eE+-]+) ms"),
        }
        rows.append(row)
        # 长时间扫描每完成一个 nprobe 就落盘，远端中断时保留已完成结果。
        with csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"recall@{top_k}={recall_at_k:.6f}, "
              f"min_query_recall={minimum_recall:.6f}, "
              f"mean_absolute_score_error={mean_score_error:.9g}, "
              f"max_absolute_score_error={max_score_error:.9g}")

    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()

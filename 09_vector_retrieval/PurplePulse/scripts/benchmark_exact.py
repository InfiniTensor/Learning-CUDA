#!/usr/bin/env python3
"""Benchmark WarpFusion Exact for several K values and write one stable CSV."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path


def extract(text: str, pattern: str) -> float:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"cannot parse vector_search output: {pattern}")
    return float(match.group(1))


def parse_configs(text: str) -> list[tuple[int, Path]]:
    result = []
    for item in text.split(","):
        fields = item.split(":", 1)
        if len(fields) != 2:
            raise ValueError("configs must use K:path,K:path syntax")
        top_k, path = int(fields[0]), Path(fields[1])
        if top_k <= 0 or not path.is_file():
            raise ValueError(f"invalid Exact config: {item}")
        result.append((top_k, path))
    if not result:
        raise ValueError("at least one Exact config is required")
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-binary", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--configs", required=True, help="K:path,K:path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("warmup must be non-negative and repeat must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for expected_k, config in parse_configs(args.configs):
        result_path = args.output_dir / f"exact_k{expected_k}.txt"
        command = [
            args.search_binary,
            "--database", args.database,
            "--queries", args.queries,
            "--params", str(config),
            "--backend", "gpu",
            "--warmup", str(args.warmup),
            "--repeat", str(args.repeat),
            "--output", str(result_path),
        ]
        completed = subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(completed.stdout, end="")
        top_k = int(extract(completed.stdout, r"top_k: (\d+)"))
        if top_k != expected_k:
            raise ValueError(f"config {config} produced K={top_k}, expected {expected_k}")
        row: dict[str, object] = {
            "implementation": "WarpFusion Exact fused",
            "top_k": top_k,
            "batch_size": int(extract(completed.stdout, r"batch_size: (\d+)")),
            "average_query_ms": extract(
                completed.stdout, r"平均常驻查询时间: ([0-9.eE+-]+) ms"
            ),
            "qps": extract(completed.stdout, r"常驻 QPS: ([0-9.eE+-]+)"),
            "run_p50_ms": extract(completed.stdout, r"run P50: ([0-9.eE+-]+) ms"),
            "run_p99_ms": extract(completed.stdout, r"run P99: ([0-9.eE+-]+) ms"),
            "run_samples": int(extract(completed.stdout, r"run samples: (\d+)")),
            "batch_p50_ms": extract(completed.stdout, r"batch P50: ([0-9.eE+-]+) ms"),
            "batch_p99_ms": extract(completed.stdout, r"batch P99: ([0-9.eE+-]+) ms"),
            "batch_latency_samples": int(
                extract(completed.stdout, r"batch samples: (\d+)")
            ),
            "gpu_initialization_ms": extract(
                completed.stdout, r"GPU 一次性初始化总时间: ([0-9.eE+-]+) ms"
            ),
            "database_h2d_ms": extract(
                completed.stdout, r"数据库一次性 H2D: ([0-9.eE+-]+) ms"
            ),
            "gpu_buffer_mib": extract(
                completed.stdout, r"GPU 缓冲区: ([0-9.eE+-]+) MiB"
            ),
            "query_h2d_ms": extract(
                completed.stdout, r"平均查询 H2D: ([0-9.eE+-]+) ms"
            ),
            "distance_and_local_topk_ms": extract(
                completed.stdout,
                r"平均融合距离\+局部 Top-K kernel: ([0-9.eE+-]+) ms",
            ),
            "final_topk_ms": extract(
                completed.stdout, r"平均最终 Top-K kernel: ([0-9.eE+-]+) ms"
            ),
            "result_d2h_ms": extract(
                completed.stdout, r"平均结果 D2H: ([0-9.eE+-]+) ms"
            ),
            "result_path": str(result_path),
        }
        rows.append(row)
        write_csv(args.csv, rows)
    print(f"CSV: {args.csv}")


if __name__ == "__main__":
    main()

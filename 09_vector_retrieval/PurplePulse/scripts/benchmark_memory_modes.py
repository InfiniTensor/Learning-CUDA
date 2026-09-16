#!/usr/bin/env python3
"""Compare fused Agent-memory scoring with a separate GPU rerank baseline."""

from __future__ import annotations

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
    parser.add_argument("--memory-metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--nlist", type=int, required=True)
    parser.add_argument("--nprobe", type=int, required=True)
    parser.add_argument("--rerank-factor", type=int, default=10)
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--importance-weight", type=float, default=0.15)
    parser.add_argument("--recency-weight", type=float, default=0.10)
    parser.add_argument("--time-scale", type=float, default=604800.0)
    parser.add_argument("--now", type=int, default=1_702_592_000)
    parser.add_argument("--min-timestamp", type=int, default=0)
    parser.add_argument("--session-id", default="any")
    parser.add_argument("--source-type", default="any")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    return parser.parse_args()


def extract(text: str, pattern: str) -> float:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"cannot parse ivf_search output: {pattern}")
    return float(match.group(1))


def read_ids(path: Path) -> list[list[int]]:
    rows: dict[int, list[int]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"{path}:{line_number}: expected three columns")
        rows.setdefault(int(fields[0]), []).append(int(fields[1]))
    return [rows[query_id] for query_id in sorted(rows)]


def recall(reference: list[list[int]], actual: list[list[int]]) -> tuple[float, float]:
    if len(reference) != len(actual) or not reference:
        raise ValueError("reference and actual query counts differ or are empty")
    values = []
    for expected, found in zip(reference, actual):
        if not expected:
            raise ValueError("reference contains an empty query result")
        values.append(len(set(expected) & set(found)) / len(expected))
    return sum(values) / len(values), min(values)


def main() -> None:
    args = parse_args()
    if args.nlist <= 0 or not 0 < args.nprobe <= args.nlist:
        raise ValueError("invalid nlist/nprobe")
    if args.rerank_factor <= 0 or args.semantic_weight <= 0:
        raise ValueError("weights and rerank factor must be positive")
    if args.importance_weight < 0 or args.recency_weight < 0 or args.time_scale <= 0:
        raise ValueError("invalid memory scoring weights")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    reference_ids: list[list[int]] | None = None
    for mode in ("fused", "rerank"):
        result_path = output_dir / f"{mode}.txt"
        command = [
            args.search_binary,
            "--index", args.index,
            "--queries", args.queries,
            "--params", args.params,
            "--memory-metadata", args.memory_metadata,
            "--memory-mode", mode,
            "--memory-semantic-weight", str(args.semantic_weight),
            "--memory-importance-weight", str(args.importance_weight),
            "--memory-recency-weight", str(args.recency_weight),
            "--memory-time-scale", str(args.time_scale),
            "--memory-now", str(args.now),
            "--filter-min-timestamp", str(args.min_timestamp),
            "--filter-session-id", args.session_id,
            "--filter-source-type", args.source_type,
            "--memory-rerank-factor", str(args.rerank_factor),
            "--nlist", str(args.nlist),
            "--nprobe", str(args.nprobe),
            "--backend", "gpu",
            "--warmup", str(args.warmup),
            "--repeat", str(args.repeat),
            "--output", str(result_path),
        ]
        completed = subprocess.run(
            command, check=True, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        print(completed.stdout, end="")
        ids = read_ids(result_path)
        if reference_ids is None:
            reference_ids = ids
            recall_at_k, minimum_recall = 1.0, 1.0
        else:
            recall_at_k, minimum_recall = recall(reference_ids, ids)
        batch_match = re.search(
            r"batch P50/P99: ([0-9.eE+-]+)/([0-9.eE+-]+) ms",
            completed.stdout,
        )
        if batch_match is None:
            raise ValueError("cannot parse batch latency")
        rows.append({
            "mode": mode,
            "nlist": args.nlist,
            "nprobe": args.nprobe,
            "rerank_factor": 1 if mode == "fused" else args.rerank_factor,
            "session_id": args.session_id,
            "source_type": args.source_type,
            "average_ms": extract(completed.stdout, r"平均查询: ([0-9.eE+-]+) ms"),
            "qps": extract(completed.stdout, r"QPS: ([0-9.eE+-]+)"),
            "batch_p50_ms": float(batch_match.group(1)),
            "batch_p99_ms": float(batch_match.group(2)),
            "scan_ms": extract(completed.stdout, r"平均 GPU 桶扫描\+局部 Top-K: ([0-9.eE+-]+) ms"),
            "merge_ms": extract(completed.stdout, r"平均 GPU 最终 Top-K 归并: ([0-9.eE+-]+) ms"),
            "rerank_ms": extract(completed.stdout, r"平均记忆独立重排: ([0-9.eE+-]+) ms"),
            "gpu_buffer_mib": extract(completed.stdout, r"GPU 缓冲区: ([0-9.eE+-]+) MiB"),
            "recall_vs_fused": recall_at_k,
            "min_query_recall_vs_fused": minimum_recall,
        })
        with csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{mode}: recall_vs_fused={recall_at_k:.6f}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()

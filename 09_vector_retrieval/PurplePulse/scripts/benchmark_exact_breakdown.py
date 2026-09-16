#!/usr/bin/env python3
"""Profile fused Exact distance, local selection, and final merge by K."""

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


def parse_k_paths(text: str) -> list[tuple[int, Path]]:
    result = []
    for item in text.split(","):
        fields = item.split(":", 1)
        if len(fields) != 2:
            raise ValueError("paths must use K:path,K:path syntax")
        top_k, path = int(fields[0]), Path(fields[1])
        if top_k <= 0 or not path.is_file():
            raise ValueError(f"invalid K:path item: {item}")
        result.append((top_k, path))
    if not result:
        raise ValueError("at least one K:path item is required")
    return result


def read_results(path: Path) -> dict[int, list[tuple[int, float]]]:
    queries: dict[int, list[tuple[int, float]]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"{path}:{line_number}: expected three columns")
        query_id, vector_id, score = int(fields[0]), int(fields[1]), float(fields[2])
        queries.setdefault(query_id, []).append((vector_id, score))
    return queries


def compare_results(expected_path: Path, actual_path: Path, tolerance: float) -> None:
    expected = read_results(expected_path)
    actual = read_results(actual_path)
    if expected.keys() != actual.keys():
        raise ValueError("profile and reference query sets differ")
    for query_id in expected:
        if len(expected[query_id]) != len(actual[query_id]):
            raise ValueError(f"query {query_id} Top-K lengths differ")
        for rank, (expected_item, actual_item) in enumerate(
            zip(expected[query_id], actual[query_id])
        ):
            expected_id, expected_score = expected_item
            actual_id, actual_score = actual_item
            if expected_id != actual_id:
                raise ValueError(
                    f"query {query_id} rank {rank} IDs differ: "
                    f"{expected_id} != {actual_id}"
                )
            error = abs(expected_score - actual_score)
            if error > tolerance:
                raise ValueError(
                    f"query {query_id} rank {rank} score error {error} "
                    f"exceeds {tolerance}"
                )


def parse_profile_output(text: str) -> dict[str, object]:
    fused_ms = extract(text, r"剖析融合 kernel: ([0-9.eE+-]+) ms")
    distance_ms = extract(text, r"剖析距离-only kernel: ([0-9.eE+-]+) ms")
    local_ms = extract(text, r"剖析局部 Top-K 估算: ([0-9.eE+-]+) ms")
    merge_ms = extract(text, r"剖析最终归并 kernel: ([0-9.eE+-]+) ms")
    pipeline_ms = fused_ms + merge_ms
    return {
        "num_vectors": int(extract(text, r"向量库: (\d+) x")),
        "dim": int(extract(text, r"向量库: \d+ x (\d+)")),
        "num_queries": int(extract(text, r"查询数: (\d+)")),
        "top_k": int(extract(text, r"top_k: (\d+)")),
        "batch_size": int(extract(text, r"batch_size: (\d+)")),
        "average_query_ms": extract(
            text, r"平均常驻查询时间: ([0-9.eE+-]+) ms"
        ),
        "qps": extract(text, r"常驻 QPS: ([0-9.eE+-]+)"),
        "run_p50_ms": extract(text, r"run P50: ([0-9.eE+-]+) ms"),
        "run_p99_ms": extract(text, r"run P99: ([0-9.eE+-]+) ms"),
        "gpu_buffer_mib": extract(text, r"GPU 缓冲区: ([0-9.eE+-]+) MiB"),
        "distance_only_ms": distance_ms,
        "local_topk_estimated_ms": local_ms,
        "fused_kernel_ms": fused_ms,
        "final_merge_ms": merge_ms,
        "kernel_pipeline_ms": pipeline_ms,
        "distance_share": distance_ms / pipeline_ms if pipeline_ms else 0.0,
        "local_topk_share": local_ms / pipeline_ms if pipeline_ms else 0.0,
        "final_merge_share": merge_ms / pipeline_ms if pipeline_ms else 0.0,
    }


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
    parser.add_argument("--reference-results", help="optional K:path,K:path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--profile-repeat", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=5e-5)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat <= 0 or args.profile_repeat <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")

    configs = parse_k_paths(args.configs)
    references = (
        dict(parse_k_paths(args.reference_results))
        if args.reference_results
        else {}
    )
    if references and set(references) != {top_k for top_k, _ in configs}:
        raise ValueError("reference-results must cover the same K values as configs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for expected_k, config in configs:
        result_path = args.output_dir / f"exact_profile_k{expected_k}.txt"
        command = [
            args.search_binary,
            "--database", args.database,
            "--queries", args.queries,
            "--params", str(config),
            "--backend", "gpu",
            "--output", str(result_path),
            "--warmup", str(args.warmup),
            "--repeat", str(args.repeat),
            "--profile-breakdown", "1",
            "--profile-repeat", str(args.profile_repeat),
        ]
        completed = subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(completed.stdout, end="")
        row = parse_profile_output(completed.stdout)
        if row["top_k"] != expected_k:
            raise ValueError(
                f"config {config} produced K={row['top_k']}, expected {expected_k}"
            )
        if references:
            compare_results(references[expected_k], result_path, args.tolerance)
        row["result_path"] = str(result_path)
        rows.append(row)
        write_csv(args.csv, rows)
    print(f"CSV: {args.csv}")


if __name__ == "__main__":
    main()

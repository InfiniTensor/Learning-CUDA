#!/usr/bin/env python3
"""Benchmark Exact fused query-tile variants with strict result validation."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path

from benchmark_exact_breakdown import compare_results, extract, parse_k_paths


def parse_variant(text: str) -> tuple[str, list[tuple[int, Path]]]:
    fields = text.split("=", 1)
    if len(fields) != 2 or not re.fullmatch(r"[A-Za-z0-9_-]+", fields[0]):
        raise ValueError("variant must use label=K:path,K:path syntax")
    return fields[0], parse_k_paths(fields[1])


def parse_search_output(text: str, label: str) -> dict[str, object]:
    layout = re.search(r"exact_fused_layout: ([A-Za-z0-9_-]+)", text)
    if layout is None:
        raise ValueError("cannot parse exact_fused_layout")
    return {
        "variant": label,
        "num_vectors": int(extract(text, r"向量库: (\d+) x")),
        "dim": int(extract(text, r"向量库: \d+ x (\d+)")),
        "num_queries": int(extract(text, r"查询数: (\d+)")),
        "top_k": int(extract(text, r"top_k: (\d+)")),
        "batch_size": int(extract(text, r"batch_size: (\d+)")),
        "exact_query_tile": int(extract(text, r"exact_query_tile: (\d+)")),
        "exact_fused_layout": layout.group(1),
        "exact_fused_blocks": int(
            extract(text, r"exact_fused_blocks: (\d+)")
        ),
        "average_query_ms": extract(
            text, r"平均常驻查询时间: ([0-9.eE+-]+) ms"
        ),
        "qps": extract(text, r"常驻 QPS: ([0-9.eE+-]+)"),
        "run_p50_ms": extract(text, r"run P50: ([0-9.eE+-]+) ms"),
        "run_p99_ms": extract(text, r"run P99: ([0-9.eE+-]+) ms"),
        "gpu_buffer_mib": extract(text, r"GPU 缓冲区: ([0-9.eE+-]+) MiB"),
        "fused_kernel_ms": extract(
            text, r"平均融合距离\+局部 Top-K kernel: ([0-9.eE+-]+) ms"
        ),
        "final_merge_ms": extract(
            text, r"平均最终 Top-K kernel: ([0-9.eE+-]+) ms"
        ),
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
    parser.add_argument(
        "--variant", action="append", required=True,
        help="repeat label=K:path,K:path for each A/B variant",
    )
    parser.add_argument("--reference-results", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=5e-5)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("warmup must be non-negative and repeat positive")

    references = dict(parse_k_paths(args.reference_results))
    variants = [parse_variant(item) for item in args.variant]
    expected_keys = set(references)
    if any({top_k for top_k, _ in configs} != expected_keys
           for _, configs in variants):
        raise ValueError("every variant must cover the reference K values")

    rows: list[dict[str, object]] = []
    for label, configs in variants:
        variant_dir = args.output_dir / label
        variant_dir.mkdir(parents=True, exist_ok=True)
        for expected_k, config in configs:
            result_path = variant_dir / f"exact_k{expected_k}.txt"
            command = [
                args.search_binary,
                "--database", args.database,
                "--queries", args.queries,
                "--params", str(config),
                "--backend", "gpu",
                "--output", str(result_path),
                "--warmup", str(args.warmup),
                "--repeat", str(args.repeat),
            ]
            completed = subprocess.run(
                command,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            print(completed.stdout, end="")
            row = parse_search_output(completed.stdout, label)
            if row["top_k"] != expected_k:
                raise ValueError(
                    f"config {config} produced K={row['top_k']}, "
                    f"expected {expected_k}"
                )
            compare_results(
                references[expected_k], result_path, args.tolerance
            )
            row["result_path"] = str(result_path)
            rows.append(row)
            write_csv(args.csv, rows)
    print(f"CSV: {args.csv}")


if __name__ == "__main__":
    main()

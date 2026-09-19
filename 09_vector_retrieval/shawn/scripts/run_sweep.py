#!/usr/bin/env python3
"""Run reproducible exact/IVF sweeps while preserving every raw artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build/vector_search"))
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument(
        "--base-config", type=Path, default=Path("configs/ivf_smoke.txt")
    )
    parser.add_argument("--index", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sweep"))
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 10, 50, 100])
    parser.add_argument("--nprobe", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1, 32, 128])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="允许覆盖同名原始文件，仅供失败 stage 的显式续跑",
    )
    return parser.parse_args()


def read_config(path: Path) -> dict[str, str]:
    """读取 config/performance/quality 共用的 key=value 文本格式。"""
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"')
    return values


def sha256(path: Path) -> str:
    """以流式方式计算大文件哈希，避免百万向量数据再次占用内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def optional_command(command: list[str], cwd: Path) -> str:
    executable = shutil.which(command[0])
    if executable is None:
        return f"unavailable: {command[0]}"
    completed = subprocess.run(
        [executable, *command[1:]],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0:
        return f"failed ({completed.returncode}): {output}"
    return output or "clean"


def write_environment(
    *,
    root: Path,
    args: argparse.Namespace,
    database_hash: str,
    query_hash: str,
    index_hash: str,
) -> None:
    """冻结代码、输入、GPU 和工具链，禁止报告数字失去 provenance。"""
    sections = {
        "utc_start": datetime.now(timezone.utc).isoformat(),
        "git_commit": optional_command(["git", "rev-parse", "HEAD"], root),
        "git_status_porcelain": optional_command(
            ["git", "status", "--porcelain"], root
        ),
        "binary": str(args.binary),
        "database": str(args.database),
        "database_sha256": database_hash,
        "queries": str(args.queries),
        "queries_sha256": query_hash,
        "index": str(args.index) if args.index is not None else "not-used",
        "index_sha256": index_hash or "not-used",
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "nvidia_smi_L": optional_command(["nvidia-smi", "-L"], root),
        "nvidia_smi_inventory": optional_command(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            root,
        ),
        "nvcc_version": optional_command(["nvcc", "--version"], root),
    }
    text = "".join(f"[{name}]\n{value}\n\n" for name, value in sections.items())
    (args.output_dir / "environment.txt").write_text(text, encoding="utf-8")


def run_search(
    *,
    args: argparse.Namespace,
    config_path: Path,
    stem: str,
    destination: Path,
    command_log: Path,
) -> dict[str, str]:
    """执行单次搜索并保存 result/performance/quality/console 四类原始文件。"""
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / f"{stem}.results.txt"
    performance_path = destination / f"{stem}.performance.log"
    quality_path = destination / f"{stem}.quality.log"
    console_path = destination / f"{stem}.console.log"
    command = [
        str(args.binary),
        "search",
        "--database",
        str(args.database),
        "--queries",
        str(args.queries),
        "--config",
        str(config_path),
        "--backend",
        args.backend,
        "--output",
        str(result_path),
        "--performance",
        str(performance_path),
        "--quality",
        str(quality_path),
    ]
    if args.index is not None:
        command.extend(["--index", str(args.index)])
    with command_log.open("a", encoding="utf-8") as stream:
        stream.write(shlex.join(command) + "\n")
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    console = completed.stdout + completed.stderr
    console_path.write_text(console, encoding="utf-8")
    if console:
        print(console, end="" if console.endswith("\n") else "\n", flush=True)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    values = read_config(performance_path)
    values.update(read_config(quality_path))
    return values


def aggregate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """按配置聚合重复值，并显式报告 QPS 标准差与 CPU reference 样本量。"""
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row["top_k"], row["nprobe"], row["batch_size"], row["backend"])
        groups[key].append(row)

    output: list[dict[str, str]] = []
    ordered = sorted(
        groups,
        key=lambda item: (int(item[0]), int(item[1]), int(item[2]), item[3]),
    )
    for key in ordered:
        group = groups[key]

        def values(field: str) -> list[float]:
            return [float(row[field]) for row in group]

        qps = values("qps")
        output.append(
            {
                "top_k": key[0],
                "nprobe": key[1],
                "batch_size": key[2],
                "backend": key[3],
                "repetitions": str(len(group)),
                "quality_queries": group[0]["quality_queries"],
                "mean_qps": f"{statistics.fmean(qps):.17g}",
                "std_qps": f"{statistics.stdev(qps):.17g}" if len(qps) > 1 else "0",
                "median_qps": f"{statistics.median(qps):.17g}",
                "mean_p50_latency_ms": f"{statistics.fmean(values('p50_latency_ms')):.17g}",
                "mean_p99_latency_ms": f"{statistics.fmean(values('p99_latency_ms')):.17g}",
                "mean_recall_at_k": f"{statistics.fmean(values('recall_at_k')):.17g}",
                "mean_rank_id_agreement": f"{statistics.fmean(values('rank_id_agreement')):.17g}",
                "mean_rank_score_absolute_error": f"{statistics.fmean(values('mean_rank_score_absolute_error')):.17g}",
                "mean_candidates_per_query": f"{statistics.fmean(values('mean_candidates_per_query')):.17g}",
                "mean_cpu_reference_qps": f"{statistics.fmean(values('cpu_reference_qps')):.17g}",
                "mean_speedup_vs_cpu_exact": f"{statistics.fmean(values('speedup_vs_cpu_exact')):.17g}",
                "resident_data_and_index_bytes": group[0][
                    "resident_data_and_index_bytes"
                ],
                "estimated_device_working_set_bytes": group[0][
                    "estimated_device_working_set_bytes"
                ],
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    columns = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        if columns:
            writer.writeheader()
            writer.writerows(rows)


def write_report_table(rows: list[dict[str, str]], path: Path) -> None:
    """生成可直接并入 REPORT.md 的统一 exact/IVF 表格。"""
    fields = (
        "top_k",
        "nprobe",
        "batch_size",
        "repetitions",
        "quality_queries",
        "mean_recall_at_k",
        "mean_qps",
        "std_qps",
        "mean_p50_latency_ms",
        "mean_p99_latency_ms",
        "mean_candidates_per_query",
        "mean_cpu_reference_qps",
        "mean_speedup_vs_cpu_exact",
        "estimated_device_working_set_bytes",
    )
    labels = (
        "K",
        "nprobe",
        "Batch",
        "Reps",
        "GT Q",
        "Recall@K",
        "Mean QPS",
        "QPS std",
        "P50 ms",
        "P99 ms",
        "Candidates/query",
        "CPU ref QPS",
        "Speedup",
        "Device bytes",
    )
    lines = [
        "| " + " | ".join(labels) + " |\n",
        "|" + "|".join("---:" for _ in labels) + "|\n",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[field] for field in fields) + " |\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.repetitions <= 0 or args.warmup < 0:
        raise ValueError("repetitions must be positive and warmup non-negative")
    if any(value <= 0 for value in (*args.top_k, *args.nprobe, *args.batch_size)):
        raise ValueError("Top-K, nprobe and batch size values must be positive")

    root = Path(__file__).resolve().parents[1]
    args.binary = args.binary.resolve()
    args.database = args.database.resolve()
    args.queries = args.queries.resolve()
    args.base_config = args.base_config.resolve()
    args.index = args.index.resolve() if args.index is not None else None
    args.output_dir = args.output_dir.resolve()
    for path in (args.binary, args.database, args.queries, args.base_config):
        if not path.is_file():
            raise SystemExit(f"required file does not exist: {path}")

    base = read_config(args.base_config)
    mode = base.get("search_mode", "")
    if mode == "ivf_flat" and (args.index is None or not args.index.is_file()):
        raise SystemExit("ivf_flat sweep requires an existing --index")
    if mode == "exact" and args.index is not None:
        raise SystemExit("exact sweep must not provide --index")
    if mode not in ("exact", "ivf_flat"):
        raise SystemExit(f"unsupported search_mode in base config: {mode!r}")

    if (
        args.output_dir.exists()
        and any(args.output_dir.iterdir())
        and not args.allow_existing
    ):
        raise SystemExit(
            f"output directory is not empty: {args.output_dir}; use a new directory "
            "or pass --allow-existing for an intentional stage retry"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command_log = args.output_dir / "commands.txt"
    command_log.write_text("", encoding="utf-8")
    database_hash = sha256(args.database)
    query_hash = sha256(args.queries)
    index_hash = sha256(args.index) if args.index is not None else ""
    write_environment(
        root=root,
        args=args,
        database_hash=database_hash,
        query_hash=query_hash,
        index_hash=index_hash,
    )

    rows: list[dict[str, str]] = []
    nlist = int(base["nlist"])
    for top_k in args.top_k:
        for nprobe in args.nprobe:
            if nprobe > nlist:
                continue
            for batch_size in args.batch_size:
                config_stem = f"k{top_k}_p{nprobe}_b{batch_size}"
                config = dict(base)
                config.update(
                    top_k=str(top_k), nprobe=str(nprobe), batch_size=str(batch_size)
                )
                config_path = args.output_dir / "configs" / f"{config_stem}.txt"
                config_path.parent.mkdir(parents=True, exist_ok=True)
                config_path.write_text(
                    "".join(f"{key} = {value}\n" for key, value in config.items()),
                    encoding="utf-8",
                )

                # 每个配置独立 warm-up，避免初始化、页缓存和 CUDA context 混入统计。
                for warmup in range(args.warmup):
                    run_search(
                        args=args,
                        config_path=config_path,
                        stem=f"w{warmup:03d}",
                        destination=args.output_dir / "warmup" / config_stem,
                        command_log=command_log,
                    )
                for repetition in range(args.repetitions):
                    values = run_search(
                        args=args,
                        config_path=config_path,
                        stem=f"r{repetition:03d}",
                        destination=args.output_dir / "raw" / config_stem,
                        command_log=command_log,
                    )
                    row = {
                        "top_k": str(top_k),
                        "nprobe": str(nprobe),
                        "batch_size": str(batch_size),
                        "repetition": str(repetition),
                        "database_sha256": database_hash,
                        "queries_sha256": query_hash,
                        "index_sha256": index_hash,
                    }
                    row.update(values)
                    rows.append(row)

    if not rows:
        raise SystemExit("sweep produced no valid configurations")
    summary_columns = [
        "top_k",
        "nprobe",
        "batch_size",
        "repetition",
        "backend",
        "qps",
        "p50_latency_ms",
        "p99_latency_ms",
        "recall_at_k",
        "rank_id_agreement",
        "mean_rank_score_absolute_error",
        "mean_candidates_per_query",
        "peak_candidates_per_query",
        "resident_data_and_index_bytes",
        "backend_working_set_bytes",
        "estimated_device_working_set_bytes",
        "cpu_reference_queries",
        "cpu_reference_qps",
        "sampled_backend_qps",
        "speedup_vs_cpu_exact",
        "quality_queries",
        "database_sha256",
        "queries_sha256",
        "index_sha256",
    ]
    with (args.output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=summary_columns, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    aggregate = aggregate_rows(rows)
    write_csv(args.output_dir / "aggregate.csv", aggregate)
    write_report_table(aggregate, args.output_dir / "report_table.md")
    print(
        f"completed {len(rows)} measured runs after per-config warm-up; "
        f"artifacts: {args.output_dir / 'aggregate.csv'}"
    )


if __name__ == "__main__":
    main()

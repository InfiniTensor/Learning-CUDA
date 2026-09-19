#!/usr/bin/env python3
"""Run exact/IVF CPU-GPU correctness gates on a CUDA server."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import platform
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

METRICS = ("l2", "inner_product", "cosine")
DTYPES = ("fp32", "fp16")


@dataclass(frozen=True)
class ResultComparison:
    """CPU/GPU 逐结果比较；严格排序与数值等价排序分别记录。"""

    strict_rank_agreement: float
    id_set_agreement: float
    tolerance_aware_rank_agreement: float
    mean_score_absolute_error: float
    max_score_absolute_error: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build/vector_search"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/gpu_validation")
    )
    parser.add_argument("--num-vectors", type=int, default=4096)
    parser.add_argument("--num-queries", type=int, default=32)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--clusters", type=int, default=32)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 10, 50, 100])
    parser.add_argument("--batch-size", type=int, default=17)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--score-tolerance", type=float, default=2.0e-4)
    return parser.parse_args()


def optional_command(command: list[str]) -> str:
    """执行环境查询命令；工具缺失时返回 unavailable。"""
    executable = shutil.which(command[0])
    if executable is None:
        return f"unavailable: {command[0]}"
    completed = subprocess.run(
        [executable, *command[1:]], capture_output=True, text=True, check=False
    )
    output = (completed.stdout or completed.stderr).strip()
    return (
        output
        if completed.returncode == 0
        else (f"failed ({completed.returncode}): {output}")
    )


def write_environment(output_dir: Path, root: Path, binary: Path) -> None:
    """保存服务器硬件与工具链信息。"""
    sections = {
        "git commit": optional_command(["git", "-C", str(root), "rev-parse", "HEAD"]),
        "git status --porcelain": optional_command(
            ["git", "-C", str(root), "status", "--porcelain"]
        ),
        "binary": str(binary),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "nvidia-smi -L": optional_command(["nvidia-smi", "-L"]),
        "nvidia-smi inventory": optional_command(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "nvcc --version": optional_command(["nvcc", "--version"]),
        "cmake --version": optional_command(["cmake", "--version"]),
        "c++ --version": optional_command(["c++", "--version"]),
    }
    text = "".join(f"[{name}]\n{value}\n\n" for name, value in sections.items())
    (output_dir / "environment.txt").write_text(text, encoding="utf-8")


def run(command: list[str], command_log: Path) -> None:
    """记录命令并直接执行，不经 shell 解释。"""
    with command_log.open("a", encoding="utf-8") as stream:
        stream.write(shlex.join(command) + "\n")
    subprocess.run(command, check=True)


def read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def read_results(path: Path) -> list[list[tuple[int, float]]]:
    """解析逐 query 的 id:score，严格检查 query 编号和结果形状。"""
    rows: list[list[tuple[int, float]]] = []
    for expected_query, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines()
    ):
        fields = raw_line.split()
        if not fields or int(fields[0]) != expected_query:
            raise ValueError(f"invalid query row in {path}: {raw_line!r}")
        entries: list[tuple[int, float]] = []
        for field in fields[1:]:
            vector_id, score = field.split(":", 1)
            entries.append((int(vector_id), float(score)))
        rows.append(entries)
    if not rows:
        raise ValueError(f"empty result file: {path}")
    return rows


def compare_results(
    cpu_path: Path, gpu_path: Path, score_tolerance: float
) -> ResultComparison:
    """对照 Top-K 集合、严格/容差排序及按相同 ID 对齐的 score。

    CPU 与 CUDA 可能因 FMA 或归约舍入令近同分邻居发生相邻换位。只有两个
    rank 上的 score 差不超过容差且整行 ID 集合完全相同，这种换位才计入
    tolerance-aware agreement；严格 rank agreement 仍单独保留用于诊断。
    """
    if score_tolerance < 0.0:
        raise ValueError("score tolerance must be non-negative")
    cpu = read_results(cpu_path)
    gpu = read_results(gpu_path)
    if len(cpu) != len(gpu) or any(
        len(cpu_row) != len(gpu_row) for cpu_row, gpu_row in zip(cpu, gpu, strict=True)
    ):
        raise ValueError("CPU/GPU result shapes do not match")
    total = 0
    strict_rank_matches = 0
    id_set_matches = 0
    tolerance_aware_rank_matches = 0
    score_errors: list[float] = []
    for cpu_row, gpu_row in zip(cpu, gpu, strict=True):
        cpu_scores_by_id = {vector_id: score for vector_id, score in cpu_row}
        gpu_scores_by_id = {vector_id: score for vector_id, score in gpu_row}
        if len(cpu_scores_by_id) != len(cpu_row) or len(gpu_scores_by_id) != len(
            gpu_row
        ):
            raise ValueError("CPU/GPU result contains duplicate vector IDs")
        cpu_ids = set(cpu_scores_by_id)
        gpu_ids = set(gpu_scores_by_id)
        shared_ids = cpu_ids & gpu_ids
        id_set_matches += len(shared_ids)
        same_id_set = cpu_ids == gpu_ids
        for vector_id in shared_ids:
            cpu_score = cpu_scores_by_id[vector_id]
            gpu_score = gpu_scores_by_id[vector_id]
            if not (math.isfinite(cpu_score) and math.isfinite(gpu_score)):
                raise ValueError("CPU/GPU result contains a non-finite score")
            score_errors.append(abs(cpu_score - gpu_score))
        for (cpu_id, cpu_score), (gpu_id, gpu_score) in zip(
            cpu_row, gpu_row, strict=True
        ):
            total += 1
            if not (math.isfinite(cpu_score) and math.isfinite(gpu_score)):
                raise ValueError("CPU/GPU result contains a non-finite score")
            ids_match = cpu_id == gpu_id
            strict_rank_matches += int(ids_match)
            tolerance_aware_rank_matches += int(
                ids_match
                or (
                    same_id_set
                    and abs(cpu_score - gpu_score) <= score_tolerance
                )
            )
    if total == 0:
        raise ValueError("CPU/GPU result contains no neighbors")
    if not score_errors:
        score_errors = [math.inf]
    return ResultComparison(
        strict_rank_agreement=strict_rank_matches / total,
        id_set_agreement=id_set_matches / total,
        tolerance_aware_rank_agreement=tolerance_aware_rank_matches / total,
        mean_score_absolute_error=sum(score_errors) / len(score_errors),
        max_score_absolute_error=max(score_errors),
    )


def write_config(
    path: Path,
    *,
    mode: str,
    top_k: int,
    batch_size: int,
    nlist: int,
    nprobe: int,
    training_samples: int,
    quality_queries: int,
    seed: int,
) -> None:
    """写出每个验证案例的配置。"""
    values = {
        "top_k": top_k,
        "search_mode": mode,
        "batch_size": batch_size,
        "nlist": nlist,
        "nprobe": nprobe,
        "pq_m": 16,
        "kmeans_iterations": 3,
        "training_samples": training_samples,
        "quality_queries": quality_queries,
        "seed": seed,
    }
    path.write_text(
        "".join(f"{key} = {value}\n" for key, value in values.items()),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run_search(
    *,
    binary: Path,
    database: Path,
    queries: Path,
    config: Path,
    backend: str,
    case_dir: Path,
    command_log: Path,
    index: Path | None = None,
) -> dict[str, str]:
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / f"{backend}.results.txt"
    performance_path = case_dir / f"{backend}.performance.log"
    quality_path = case_dir / f"{backend}.quality.log"
    # 避免本次命令失败后误读同目录上一轮留下的结果。
    for stale_path in (result_path, performance_path, quality_path):
        stale_path.unlink(missing_ok=True)
    command = [
        str(binary),
        "search",
        "--database",
        str(database),
        "--queries",
        str(queries),
        "--config",
        str(config),
        "--backend",
        backend,
        "--output",
        str(result_path),
        "--performance",
        str(performance_path),
        "--quality",
        str(quality_path),
    ]
    if index is not None:
        command.extend(["--index", str(index)])
    run(command, command_log)
    values = read_key_values(performance_path)
    values.update(read_key_values(quality_path))
    return values


def gate_row(
    *,
    mode: str,
    metric: str,
    dtype: str,
    top_k: int,
    cpu: dict[str, str],
    gpu: dict[str, str],
    database_hash: str,
    query_hash: str,
    score_tolerance: float,
    direct: ResultComparison,
) -> dict[str, str]:
    """Exact 使用绝对真值 gate；IVF 比较同一索引上的 CPU/GPU 质量统计。"""
    cpu_recall = float(cpu["recall_at_k"])
    gpu_recall = float(gpu["recall_at_k"])
    cpu_rank = float(cpu["rank_id_agreement"])
    gpu_rank = float(gpu["rank_id_agreement"])
    cpu_score_error = float(cpu["mean_rank_score_absolute_error"])
    gpu_score_error = float(gpu["mean_rank_score_absolute_error"])
    recall_difference = abs(cpu_recall - gpu_recall)
    rank_difference = abs(cpu_rank - gpu_rank)
    score_error_difference = abs(cpu_score_error - gpu_score_error)
    direct_passed = (
        direct.id_set_agreement >= 1.0 - 1.0e-12
        and direct.tolerance_aware_rank_agreement >= 1.0 - 1.0e-12
        and direct.max_score_absolute_error <= score_tolerance
    )
    if mode == "exact":
        passed = (
            gpu_recall >= 1.0 - 1.0e-12
            and gpu_score_error <= score_tolerance
            and direct_passed
        )
    else:
        passed = (
            recall_difference <= 1.0e-12
            and score_error_difference <= score_tolerance
            and direct_passed
        )
    cpu_qps = float(cpu["qps"])
    gpu_qps = float(gpu["qps"])
    return {
        "mode": mode,
        "metric": metric,
        "dtype": dtype,
        "top_k": str(top_k),
        "num_vectors": gpu["num_vectors"],
        "dim": gpu["dim"],
        "num_queries": gpu["num_queries"],
        "batch_size": gpu["batch_size"],
        "nlist": gpu["nlist"],
        "nprobe": gpu["nprobe"],
        "cpu_qps": f"{cpu_qps:.17g}",
        "gpu_qps": f"{gpu_qps:.17g}",
        "speedup": f"{gpu_qps / cpu_qps:.17g}" if cpu_qps > 0.0 else "0",
        "cpu_recall_at_k": f"{cpu_recall:.17g}",
        "gpu_recall_at_k": f"{gpu_recall:.17g}",
        "recall_difference": f"{recall_difference:.17g}",
        "cpu_rank_agreement": f"{cpu_rank:.17g}",
        "gpu_rank_agreement": f"{gpu_rank:.17g}",
        "rank_difference": f"{rank_difference:.17g}",
        "score_error_difference": f"{score_error_difference:.17g}",
        "cpu_gpu_rank_agreement": f"{direct.strict_rank_agreement:.17g}",
        "cpu_gpu_id_set_agreement": f"{direct.id_set_agreement:.17g}",
        "cpu_gpu_tolerance_aware_rank_agreement": (
            f"{direct.tolerance_aware_rank_agreement:.17g}"
        ),
        "cpu_gpu_mean_score_absolute_error": (
            f"{direct.mean_score_absolute_error:.17g}"
        ),
        "cpu_gpu_max_score_absolute_error": f"{direct.max_score_absolute_error:.17g}",
        "score_tolerance": f"{score_tolerance:.17g}",
        "database_sha256": database_hash,
        "queries_sha256": query_hash,
        "passed": str(passed).lower(),
        "error": "",
    }


def execution_failure_row(
    *,
    mode: str,
    metric: str,
    dtype: str,
    top_k: int,
    num_vectors: int,
    dim: int,
    num_queries: int,
    batch_size: int,
    nlist: int,
    nprobe: int,
    score_tolerance: float,
    database_hash: str,
    query_hash: str,
    error: Exception,
) -> dict[str, str]:
    """失败案例保留完整列，使后续成功案例仍可写入同一 CSV。"""
    return {
        "mode": mode,
        "metric": metric,
        "dtype": dtype,
        "top_k": str(top_k),
        "num_vectors": str(num_vectors),
        "dim": str(dim),
        "num_queries": str(num_queries),
        "batch_size": str(batch_size),
        "nlist": str(nlist),
        "nprobe": str(nprobe),
        "cpu_qps": "",
        "gpu_qps": "",
        "speedup": "",
        "cpu_recall_at_k": "",
        "gpu_recall_at_k": "",
        "recall_difference": "",
        "cpu_rank_agreement": "",
        "gpu_rank_agreement": "",
        "rank_difference": "",
        "score_error_difference": "",
        "cpu_gpu_rank_agreement": "",
        "cpu_gpu_id_set_agreement": "",
        "cpu_gpu_tolerance_aware_rank_agreement": "",
        "cpu_gpu_mean_score_absolute_error": "",
        "cpu_gpu_max_score_absolute_error": "",
        "score_tolerance": f"{score_tolerance:.17g}",
        "database_sha256": database_hash,
        "queries_sha256": query_hash,
        "passed": "false",
        "error": str(error).replace("\n", " "),
    }


def write_report_table(rows: list[dict[str, str]], output_path: Path) -> None:
    """生成正确性与性能汇总表。"""
    fields = [
        "mode",
        "metric",
        "dtype",
        "top_k",
        "cpu_qps",
        "gpu_qps",
        "speedup",
        "gpu_recall_at_k",
        "gpu_rank_agreement",
        "cpu_gpu_rank_agreement",
        "cpu_gpu_id_set_agreement",
        "cpu_gpu_tolerance_aware_rank_agreement",
        "cpu_gpu_max_score_absolute_error",
        "passed",
    ]
    labels = [
        "Mode",
        "Metric",
        "Dtype",
        "K",
        "CPU QPS",
        "GPU QPS",
        "Speedup",
        "GPU Recall@K",
        "GPU Rank agree.",
        "CPU/GPU strict rank",
        "CPU/GPU ID set",
        "CPU/GPU tolerance-aware rank",
        "CPU/GPU max score err.",
        "Pass",
    ]
    lines = [
        "| " + " | ".join(labels) + " |\n",
        "|" + "|".join("---" for _ in labels) + "|\n",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[field] for field in fields) + " |\n")
    output_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if (
        min(
            args.num_vectors,
            args.num_queries,
            args.dim,
            args.clusters,
            args.batch_size,
        )
        <= 0
        or args.clusters > args.num_vectors
        or args.score_tolerance < 0.0
        or any(top_k <= 0 or top_k > 100 for top_k in args.top_k)
        or max(args.top_k) > args.num_vectors
    ):
        raise ValueError("invalid validation dimensions, Top-K, or tolerance")

    root = Path(__file__).resolve().parents[1]
    binary = args.binary.resolve()
    if not binary.is_file():
        raise SystemExit(f"binary does not exist: {binary}")
    generator = root / "scripts" / "generate_data.py"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command_log = output_dir / "commands.txt"
    command_log.write_text("", encoding="utf-8")
    summary = output_dir / "summary.csv"
    report_table = output_dir / "report_table.md"
    summary.unlink(missing_ok=True)
    report_table.unlink(missing_ok=True)
    write_environment(output_dir, root, binary)

    rows: list[dict[str, str]] = []
    nlist = min(args.clusters, args.num_vectors)
    nprobe = min(8, nlist)
    training_samples = min(max(2048, nlist), args.num_vectors)
    ivf_top_k = min(10, args.num_vectors)

    for metric_index, metric in enumerate(METRICS):
        for dtype_index, dtype in enumerate(DTYPES):
            case = f"{metric}_{dtype}"
            case_dir = output_dir / case
            data_dir = case_dir / "data"
            case_seed = args.seed + metric_index * 100 + dtype_index
            run(
                [
                    sys.executable,
                    str(generator),
                    "--output-dir",
                    str(data_dir),
                    "--num-vectors",
                    str(args.num_vectors),
                    "--num-queries",
                    str(args.num_queries),
                    "--dim",
                    str(args.dim),
                    "--dtype",
                    dtype,
                    "--metric",
                    metric,
                    "--clusters",
                    str(args.clusters),
                    "--seed",
                    str(case_seed),
                ],
                command_log,
            )
            database = data_dir / "database.bin"
            queries = data_dir / "queries.bin"
            database_hash = sha256(database)
            query_hash = sha256(queries)

            for top_k in args.top_k:
                exact_dir = case_dir / "exact" / f"k{top_k}"
                exact_config = exact_dir / "config.txt"
                exact_dir.mkdir(parents=True, exist_ok=True)
                write_config(
                    exact_config,
                    mode="exact",
                    top_k=top_k,
                    batch_size=args.batch_size,
                    nlist=nlist,
                    nprobe=nprobe,
                    training_samples=training_samples,
                    quality_queries=args.num_queries,
                    seed=case_seed,
                )
                try:
                    cpu = run_search(
                        binary=binary,
                        database=database,
                        queries=queries,
                        config=exact_config,
                        backend="cpu",
                        case_dir=exact_dir,
                        command_log=command_log,
                    )
                    gpu = run_search(
                        binary=binary,
                        database=database,
                        queries=queries,
                        config=exact_config,
                        backend="cuda",
                        case_dir=exact_dir,
                        command_log=command_log,
                    )
                    direct = compare_results(
                        exact_dir / "cpu.results.txt",
                        exact_dir / "cuda.results.txt",
                        args.score_tolerance,
                    )
                    row = gate_row(
                        mode="exact",
                        metric=metric,
                        dtype=dtype,
                        top_k=top_k,
                        cpu=cpu,
                        gpu=gpu,
                        database_hash=database_hash,
                        query_hash=query_hash,
                        score_tolerance=args.score_tolerance,
                        direct=direct,
                    )
                except (
                    subprocess.CalledProcessError,
                    OSError,
                    ValueError,
                    KeyError,
                ) as error:
                    print(f"{case}:exact:k{top_k} failed: {error}", file=sys.stderr)
                    row = execution_failure_row(
                        mode="exact",
                        metric=metric,
                        dtype=dtype,
                        top_k=top_k,
                        num_vectors=args.num_vectors,
                        dim=args.dim,
                        num_queries=args.num_queries,
                        batch_size=args.batch_size,
                        nlist=nlist,
                        nprobe=nprobe,
                        score_tolerance=args.score_tolerance,
                        database_hash=database_hash,
                        query_hash=query_hash,
                        error=error,
                    )
                rows.append(row)

            ivf_dir = case_dir / "ivf"
            ivf_dir.mkdir(parents=True, exist_ok=True)
            ivf_config = ivf_dir / "config.txt"
            ivf_index = ivf_dir / "gpu_built.ivf"
            write_config(
                ivf_config,
                mode="ivf_flat",
                top_k=ivf_top_k,
                batch_size=args.batch_size,
                nlist=nlist,
                nprobe=nprobe,
                training_samples=training_samples,
                quality_queries=args.num_queries,
                seed=case_seed,
            )
            try:
                for backend in ("cpu", "cuda"):
                    for suffix in (
                        "results.txt",
                        "performance.log",
                        "quality.log",
                    ):
                        (ivf_dir / f"{backend}.{suffix}").unlink(missing_ok=True)
                ivf_index.unlink(missing_ok=True)
                build_log = ivf_dir / "cuda.build.log"
                build_log.unlink(missing_ok=True)
                run(
                    [
                        str(binary),
                        "build",
                        "--database",
                        str(database),
                        "--config",
                        str(ivf_config),
                        "--index",
                        str(ivf_index),
                        "--backend",
                        "cuda",
                        "--performance",
                        str(build_log),
                    ],
                    command_log,
                )
                cpu = run_search(
                    binary=binary,
                    database=database,
                    queries=queries,
                    config=ivf_config,
                    backend="cpu",
                    case_dir=ivf_dir,
                    command_log=command_log,
                    index=ivf_index,
                )
                gpu = run_search(
                    binary=binary,
                    database=database,
                    queries=queries,
                    config=ivf_config,
                    backend="cuda",
                    case_dir=ivf_dir,
                    command_log=command_log,
                    index=ivf_index,
                )
                direct = compare_results(
                    ivf_dir / "cpu.results.txt",
                    ivf_dir / "cuda.results.txt",
                    args.score_tolerance,
                )
                row = gate_row(
                    mode="ivf_flat",
                    metric=metric,
                    dtype=dtype,
                    top_k=ivf_top_k,
                    cpu=cpu,
                    gpu=gpu,
                    database_hash=database_hash,
                    query_hash=query_hash,
                    score_tolerance=args.score_tolerance,
                    direct=direct,
                )
            except (
                subprocess.CalledProcessError,
                OSError,
                ValueError,
                KeyError,
            ) as error:
                print(f"{case}:ivf_flat:k{ivf_top_k} failed: {error}", file=sys.stderr)
                row = execution_failure_row(
                    mode="ivf_flat",
                    metric=metric,
                    dtype=dtype,
                    top_k=ivf_top_k,
                    num_vectors=args.num_vectors,
                    dim=args.dim,
                    num_queries=args.num_queries,
                    batch_size=args.batch_size,
                    nlist=nlist,
                    nprobe=nprobe,
                    score_tolerance=args.score_tolerance,
                    database_hash=database_hash,
                    query_hash=query_hash,
                    error=error,
                )
            rows.append(row)

    with summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_report_table(rows, report_table)
    failed = [
        f"{row['mode']}:{row['metric']}:{row['dtype']}:k{row['top_k']}"
        for row in rows
        if row["passed"] != "true"
    ]
    if failed:
        raise SystemExit(
            "CPU/GPU validation failed: "
            + ", ".join(failed)
            + f"; artifacts: {summary}, {output_dir / 'environment.txt'}, "
            + str(report_table)
        )
    print(
        f"all {len(rows)} vector-search CPU/GPU gates passed; "
        f"artifacts: {summary}, {output_dir / 'environment.txt'}, "
        f"{report_table}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the Stage-6 benchmark matrix for one prepared public dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


DATA_DIR_NAMES = {
    "sift1m": "sift1m",
    "yahoo-minilm-384": "yahoo-minilm-384-normalized",
}
TOP_KS = (10, 50, 100)
PROVENANCE_FILES = (
    "src/gpu_ivf_flat.cu",
    "src/gpu_search.cu",
    "src/ivf_flat.cpp",
    "scripts/run_real_benchmarks.py",
    "scripts/benchmark_exact.py",
    "scripts/benchmark_ivf_sweep.py",
    "scripts/benchmark_faiss.py",
)


def run(command: list[str], log_path: Path | None = None) -> str:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, end="")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(completed.stdout, encoding="utf-8")
    return completed.stdout


def capture(command: list[str]) -> str:
    try:
        return subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ).stdout.strip()
    except OSError as error:
        return f"unavailable: {error}"


def extract(text: str, pattern: str) -> float:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"cannot parse command output: {pattern}")
    return float(match.group(1))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_build_stats(text: str, dataset: str, nlist: int) -> dict[str, object]:
    training = re.search(r"迭代/训练样本: (\d+)/(\d+)", text)
    buckets = re.search(
        r"桶大小 min/avg/max: ([0-9.eE+-]+)/([0-9.eE+-]+)/([0-9.eE+-]+)",
        text,
    )
    if training is None or buckets is None:
        raise ValueError("cannot parse IVF build metadata")
    return {
        "dataset": dataset,
        "nlist": nlist,
        "training_iterations": int(training.group(1)),
        "training_samples": int(training.group(2)),
        "threads": int(extract(text, r"建库线程: (\d+)")),
        "build_ms": extract(text, r"构建时间: ([0-9.eE+-]+) ms"),
        "save_ms": extract(text, r"保存时间: ([0-9.eE+-]+) ms"),
        "minimum_bucket": int(float(buckets.group(1))),
        "average_bucket": float(buckets.group(2)),
        "maximum_bucket": int(float(buckets.group(3))),
        "empty_buckets": int(extract(text, r"空桶: (\d+)")),
        "index_bytes": int(extract(text, r"索引大小: (\d+) bytes")),
    }


def write_one_row_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def read_csv_by_nprobe(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    return {int(row["nprobe"]): row for row in rows}


def choose_partitioned_result(
    partitioned_csv: Path,
    full_heap_csv: Path,
    effective_csv: Path,
    maximum_recall_loss: float,
) -> dict[str, object]:
    partitioned = read_csv_by_nprobe(partitioned_csv)
    full_heap = read_csv_by_nprobe(full_heap_csv)
    if partitioned.keys() != full_heap.keys():
        raise ValueError("partitioned and full-heap scans use different nprobe values")
    comparisons = []
    fallback = False
    for nprobe in sorted(partitioned):
        partitioned_recall = float(partitioned[nprobe]["recall_at_k"])
        full_heap_recall = float(full_heap[nprobe]["recall_at_k"])
        recall_loss = full_heap_recall - partitioned_recall
        fallback = fallback or recall_loss > maximum_recall_loss
        comparisons.append(
            {
                "nprobe": nprobe,
                "partitioned_recall": partitioned_recall,
                "full_heap_recall": full_heap_recall,
                "partitioned_qps": float(partitioned[nprobe]["qps"]),
                "full_heap_qps": float(full_heap[nprobe]["qps"]),
                "recall_loss": recall_loss,
            }
        )
    selected = full_heap_csv if fallback else partitioned_csv
    shutil.copyfile(selected, effective_csv)
    return {
        "selected_mode": "full_heap" if fallback else "partitioned",
        "maximum_allowed_recall_loss": maximum_recall_loss,
        "comparisons": comparisons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATA_DIR_NAMES), required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, default=Path("data/real"))
    parser.add_argument("--results-root", type=Path, default=Path("results/real"))
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--nprobes", default="32,64,128,160,192,224")
    parser.add_argument("--training-samples", type=int, default=100000)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--skip-index-build", action="store_true")
    parser.add_argument("--skip-faiss", action="store_true")
    parser.add_argument(
        "--maximum-partition-recall-loss",
        type=float,
        default=1e-4,
        help="Fall back to the complete heap if any nprobe exceeds this loss",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.maximum_partition_recall_loss < 0:
        raise ValueError("maximum-partition-recall-loss must be non-negative")
    os.environ.setdefault("_FAISS_WHEEL_DISABLE_CUDA_PRELOAD", "1")
    cuda_library = "/usr/local/cuda/lib64"
    library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if Path(cuda_library).is_dir() and cuda_library not in library_path.split(":"):
        os.environ["LD_LIBRARY_PATH"] = ":".join(
            item for item in (cuda_library, library_path) if item
        )
    root = args.project_root.resolve()
    data = (root / args.data_root / DATA_DIR_NAMES[args.dataset]).resolve()
    results = (root / args.results_root / args.dataset).resolve()
    build = (root / args.build_dir).resolve()
    scripts = root / "scripts"
    configs = root / "configs"
    results.mkdir(parents=True, exist_ok=True)

    for required in (
        data / "database.bin",
        data / "queries.bin",
        data / "groundtruth.ivecs",
        build / "vector_search",
        build / "ivf_build",
        build / "ivf_search",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    index = data / f"nlist{args.nlist}.ivf"
    if not args.skip_index_build:
        build_text = run(
            [
                str(build / "ivf_build"),
                "--database", str(data / "database.bin"),
                "--output", str(index),
                "--nlist", str(args.nlist),
                "--iterations", str(args.iterations),
                "--training-samples", str(args.training_samples),
                "--threads", str(args.threads),
            ],
            results / "logs" / "ivf_build.log",
        )
        write_one_row_csv(
            results / "build.csv",
            parse_build_stats(build_text, args.dataset, args.nlist),
        )
    elif not index.is_file():
        raise FileNotFoundError(index)

    exact_configs = ",".join(
        f"{top_k}:{configs / f'real_exact_k{top_k}.conf'}"
        for top_k in TOP_KS
    )
    run(
        [
            sys.executable,
            str(scripts / "benchmark_exact.py"),
            "--search-binary", str(build / "vector_search"),
            "--database", str(data / "database.bin"),
            "--queries", str(data / "queries.bin"),
            "--configs", exact_configs,
            "--output-dir", str(results / "exact_outputs"),
            "--csv", str(results / "exact_gpu.csv"),
            "--warmup", str(args.warmup),
            "--repeat", str(args.repeat),
        ],
        results / "logs" / "exact.log",
    )

    validations = []
    for top_k in TOP_KS:
        output = run(
            [
                sys.executable,
                str(scripts / "evaluate_ivecs_groundtruth.py"),
                "--groundtruth", str(data / "groundtruth.ivecs"),
                "--results", str(results / "exact_outputs" / f"exact_k{top_k}.txt"),
                "--database", str(data / "database.bin"),
                "--queries", str(data / "queries.bin"),
                "--json",
            ]
        )
        validations.append(json.loads(output))
    (results / "groundtruth_validation.json").write_text(
        json.dumps(validations, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    def run_ivf(top_k: int, config: Path, suffix: str) -> Path:
        csv_path = results / f"ivf_k{top_k}{suffix}.csv"
        run(
            [
                sys.executable,
                str(scripts / "benchmark_ivf_sweep.py"),
                "--search-binary", str(build / "ivf_search"),
                "--index", str(index),
                "--queries", str(data / "queries.bin"),
                "--params", str(config),
                "--exact", str(results / "exact_outputs" / f"exact_k{top_k}.txt"),
                "--output-dir", str(results / f"ivf_k{top_k}{suffix}_outputs"),
                "--csv", str(csv_path),
                "--nlist", str(args.nlist),
                "--training-samples", str(args.training_samples),
                "--training-iterations", str(args.iterations),
                "--training-seed", "2026",
                "--nprobes", args.nprobes,
                "--warmup", str(args.warmup),
                "--repeat", str(args.repeat),
            ],
            results / "logs" / f"ivf_k{top_k}{suffix}.log",
        )
        return csv_path

    partition_validations = {}
    for top_k in TOP_KS:
        optimized_config = configs / f"real_ivf_nlist1024_k{top_k}.conf"
        if top_k == 10:
            run_ivf(top_k, optimized_config, "")
            continue
        partitioned_csv = run_ivf(top_k, optimized_config, "_partitioned")
        full_heap_csv = run_ivf(
            top_k,
            configs / f"real_ivf_nlist1024_k{top_k}_safe.conf",
            "_full_heap",
        )
        partition_validations[str(top_k)] = choose_partitioned_result(
            partitioned_csv,
            full_heap_csv,
            results / f"ivf_k{top_k}.csv",
            args.maximum_partition_recall_loss,
        )
    (results / "partitioned_topk_validation.json").write_text(
        json.dumps(partition_validations, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if not args.skip_faiss:
        run(
            [
                sys.executable,
                str(scripts / "benchmark_faiss.py"),
                "--database", str(data / "database.bin"),
                "--queries", str(data / "queries.bin"),
                "--output-dir", str(results / "faiss_outputs"),
                "--csv", str(results / "faiss_gpu.csv"),
                "--index-dir", str(data / "faiss_indexes"),
                "--reference-results", str(results / "exact_outputs" / "exact_k100.txt"),
                "--backends", "gpu",
                "--index-types", "flat,ivf_flat",
                "--top-ks", "10,50,100",
                "--nprobes", args.nprobes,
                "--nlist", str(args.nlist),
                "--batch-size", str(args.batch_size),
                "--train-samples", str(args.training_samples),
                "--iterations", str(args.iterations),
                "--seed", "2026",
                "--warmup", str(args.warmup),
                "--repeat", str(args.repeat),
            ],
            results / "logs" / "faiss_gpu.log",
        )
        run(
            [
                sys.executable,
                str(scripts / "plot_results.py"),
                "--results-dir", str(results),
                "--output-dir", str(results / "figures"),
            ]
        )

    environment = {
        "dataset": args.dataset,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": capture(["git", "-C", str(root), "rev-parse", "HEAD"]),
        "source_sha256": {
            name: sha256_file(root / name) for name in PROVENANCE_FILES
        },
        "gpu": capture([
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]),
        "nvcc": capture(["nvcc", "--version"]),
        "python": sys.version,
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (results / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Stage-6 results: {results}")


if __name__ == "__main__":
    main()

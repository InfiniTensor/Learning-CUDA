#!/usr/bin/env python3
"""在本项目自定义数据上运行可复现的 FAISS Flat/IVF-Flat 基准。"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

# 固定 CPU 对照的线程口径；用户可在命令前显式覆盖。
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np


DTYPES = {1: np.dtype("<f4"), 2: np.dtype("<f2")}
METRICS = {1: "l2", 2: "inner_product", 3: "cosine"}


@dataclass(frozen=True)
class VectorFile:
    path: Path
    rows: int
    dim: int
    dtype_code: int
    metric: str | None
    header_bytes: int

    def mmap(self) -> np.memmap:
        return np.memmap(
            self.path,
            dtype=DTYPES[self.dtype_code],
            mode="r",
            offset=self.header_bytes,
            shape=(self.rows, self.dim),
        )


@dataclass
class BenchmarkRow:
    implementation: str
    backend: str
    index_type: str
    metric: str
    num_vectors: int
    num_queries: int
    dim: int
    dtype: str
    cpu_threads: int
    top_k: int
    batch_size: int
    nlist: int
    nprobe: int
    train_samples: int
    training_iterations: int
    training_seed: int
    build_ms: float
    load_ms: float
    gpu_transfer_ms: float
    gpu_index_mib: float
    gpu_temp_mib: float
    average_query_ms: float
    qps: float
    run_p50_ms: float
    run_p99_ms: float
    run_samples: int
    batch_p50_ms: float
    batch_p99_ms: float
    batch_latency_samples: int
    recall_at_k: float
    min_query_recall: float
    mean_absolute_score_error: float
    max_absolute_score_error: float


def _read_header(path: Path, expected_magic: bytes) -> tuple[bytes, int]:
    with path.open("rb") as source:
        magic = source.read(8)
    if magic != expected_magic:
        raise ValueError(f"{path}: magic/version 不正确")
    return magic, path.stat().st_size


def read_database(path: Path) -> VectorFile:
    _, actual_size = _read_header(path, b"PPVEC001")
    with path.open("rb") as source:
        source.seek(8)
        rows, dim, dtype_code, metric_code = struct.unpack("<QIII", source.read(20))
    if rows <= 0 or dim <= 0 or dtype_code not in DTYPES or metric_code not in METRICS:
        raise ValueError(f"{path}: 向量库 header 字段不合法")
    header_bytes = 28
    expected_size = header_bytes + rows * dim * DTYPES[dtype_code].itemsize
    if actual_size != expected_size:
        raise ValueError(f"{path}: 文件大小应为 {expected_size}，实际为 {actual_size}")
    return VectorFile(path, rows, dim, dtype_code, METRICS[metric_code], header_bytes)


def read_queries(path: Path) -> VectorFile:
    _, actual_size = _read_header(path, b"PPQRY001")
    with path.open("rb") as source:
        source.seek(8)
        rows, dim, dtype_code = struct.unpack("<QII", source.read(16))
    if rows <= 0 or dim <= 0 or dtype_code not in DTYPES:
        raise ValueError(f"{path}: query header 字段不合法")
    header_bytes = 24
    expected_size = header_bytes + rows * dim * DTYPES[dtype_code].itemsize
    if actual_size != expected_size:
        raise ValueError(f"{path}: 文件大小应为 {expected_size}，实际为 {actual_size}")
    return VectorFile(path, rows, dim, dtype_code, None, header_bytes)


def parse_positive_csv(text: str, name: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in text.split(",")]
    except ValueError as error:
        raise ValueError(f"{name} 必须是逗号分隔的正整数") from error
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} 必须是逗号分隔的正整数")
    return values


def percentile(values: Iterable[float], fraction: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return 0.0
    return float(np.percentile(array, fraction * 100.0, method="higher"))


def gpu_memory_mib(resource) -> tuple[float, float]:
    """返回 FAISS GPU 的索引数据与临时工作区保留量。"""
    if not hasattr(resource, "getMemoryInfo"):
        return 0.0, 0.0
    index_bytes = 0
    temp_bytes = 0
    for categories in resource.getMemoryInfo().values():
        for name, (_, allocated_bytes) in categories.items():
            if name == "TemporaryMemoryBuffer":
                temp_bytes += allocated_bytes
            else:
                index_bytes += allocated_bytes
    scale = 1024.0 * 1024.0
    return index_bytes / scale, temp_bytes / scale


def recall(reference: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    if reference.shape != actual.shape or reference.ndim != 2 or reference.shape[1] == 0:
        raise ValueError("reference 与 actual 的形状必须相同且 K>0")
    per_query = np.asarray(
        [len(set(expected) & set(found)) / reference.shape[1]
         for expected, found in zip(reference.tolist(), actual.tolist())],
        dtype=np.float64,
    )
    return float(per_query.mean()), float(per_query.min())


def score_errors(reference: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """按名次比较 Top-K 分数，量化近似结果相对 exact 的质量损失。"""
    if reference.shape != actual.shape or reference.size == 0:
        raise ValueError("reference 与 actual 的分数形状必须相同且非空")
    errors = np.abs(
        np.asarray(reference, dtype=np.float64)
        - np.asarray(actual, dtype=np.float64)
    )
    return float(errors.mean()), float(errors.max())


def as_float32(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float32)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = as_float32(values)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("cosine 数据包含零向量")
    return values / norms


def import_faiss():
    try:
        import faiss  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "未安装 FAISS；请在隔离环境中安装 faiss-cpu 或 CUDA 对应的 "
            "faiss-gpu 包"
        ) from error
    return faiss


def make_flat_index(faiss, dim: int, metric: str):
    return faiss.IndexFlatL2(dim) if metric == "l2" else faiss.IndexFlatIP(dim)


def make_ivf_index(faiss, dim: int, metric: str, nlist: int, iterations: int, seed: int):
    metric_id = faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT
    quantizer = make_flat_index(faiss, dim, metric)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, metric_id)
    index.cp.niter = iterations
    index.cp.seed = seed
    return index


def prepare_vectors(database: VectorFile, queries: VectorFile) -> tuple[np.ndarray, np.ndarray]:
    if database.dim != queries.dim:
        raise ValueError("数据库与 query 维度不一致")
    database_values = as_float32(database.mmap())
    query_values = as_float32(queries.mmap())
    if database.metric == "cosine":
        database_values = normalize_rows(database_values)
        query_values = normalize_rows(query_values)
    return database_values, query_values


def search_batches(index, queries: np.ndarray, top_k: int, batch_size: int):
    distances = np.empty((len(queries), top_k), dtype=np.float32)
    ids = np.empty((len(queries), top_k), dtype=np.int64)
    latencies = []
    start_all = time.perf_counter()
    for start in range(0, len(queries), batch_size):
        end = min(start + batch_size, len(queries))
        start_batch = time.perf_counter()
        batch_distances, batch_ids = index.search(queries[start:end], top_k)
        latencies.append((time.perf_counter() - start_batch) * 1000.0)
        distances[start:end] = batch_distances
        ids[start:end] = batch_ids
    total_ms = (time.perf_counter() - start_all) * 1000.0
    return distances, ids, total_ms, latencies


def write_results(path: Path, ids: np.ndarray, scores: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output:
        for query_id, (query_ids, query_scores) in enumerate(zip(ids, scores)):
            for vector_id, score in zip(query_ids, query_scores):
                output.write(f"{query_id} {int(vector_id)} {float(score):.9g}\n")


def write_csv(path: Path, rows: list[BenchmarkRow]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def read_result_arrays(
    path: Path, num_queries: int, minimum_k: int
) -> tuple[np.ndarray, np.ndarray]:
    id_rows: list[list[int]] = [[] for _ in range(num_queries)]
    score_rows: list[list[float]] = [[] for _ in range(num_queries)]
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"{path}:{line_number}: 结果行必须有三列")
            query_id, vector_id, score = int(fields[0]), int(fields[1]), float(fields[2])
            if query_id < 0 or query_id >= num_queries:
                raise ValueError(f"{path}:{line_number}: query_id 越界")
            id_rows[query_id].append(vector_id)
            score_rows[query_id].append(score)
    if any(len(row) < minimum_k for row in id_rows):
        raise ValueError(f"{path}: 每个 query 至少需要 {minimum_k} 个结果")
    ids = np.asarray([row[:minimum_k] for row in id_rows], dtype=np.int64)
    scores = np.asarray([row[:minimum_k] for row in score_rows], dtype=np.float32)
    return ids, scores


def read_result_ids(path: Path, num_queries: int, minimum_k: int) -> np.ndarray:
    """兼容已有调用者；新代码应使用 read_result_arrays 同时读取分数。"""
    return read_result_arrays(path, num_queries, minimum_k)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-results", type=Path,
        help="可选的本项目 exact 结果，用来交叉验证 FAISS Flat",
    )
    parser.add_argument("--backends", default="cpu,gpu", help="cpu,gpu 或二者")
    parser.add_argument("--index-types", default="flat,ivf_flat")
    parser.add_argument("--top-ks", default="10,50,100")
    parser.add_argument("--nprobes", default="128,160,192,224")
    parser.add_argument("--nlist", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=100000)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[list[str], list[str], list[int], list[int]]:
    backends = [item.strip() for item in args.backends.split(",")]
    index_types = [item.strip() for item in args.index_types.split(",")]
    top_ks = parse_positive_csv(args.top_ks, "top-ks")
    nprobes = parse_positive_csv(args.nprobes, "nprobes")
    if not backends or any(item not in {"cpu", "gpu"} for item in backends):
        raise ValueError("backends 只能包含 cpu,gpu")
    if not index_types or any(item not in {"flat", "ivf_flat"} for item in index_types):
        raise ValueError("index-types 只能包含 flat,ivf_flat")
    for name in (
        "nlist", "batch_size", "train_samples", "iterations", "repeat",
        "cpu_threads",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} 必须大于 0")
    if args.warmup < 0 or any(value > args.nlist for value in nprobes):
        raise ValueError("warmup 必须非负，nprobe 必须位于 1..nlist")
    return backends, index_types, top_ks, nprobes


def main() -> None:
    args = parse_args()
    backends, index_types, top_ks, nprobes = validate_args(args)
    faiss = import_faiss()
    faiss.omp_set_num_threads(args.cpu_threads)
    database_info = read_database(args.database)
    query_info = read_queries(args.queries)
    database, queries = prepare_vectors(database_info, query_info)
    if max(top_ks) > database_info.rows:
        raise ValueError("top_k 不能超过数据库向量数")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.index_dir.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    metric = database_info.metric
    assert metric is not None
    metric_suffix = "l2" if metric == "l2" else "ip"

    cpu_indexes = {}
    build_times = {}
    load_times = {}
    build_index_types = list(dict.fromkeys(["flat", *index_types]))
    for index_type in build_index_types:
        training_suffix = ""
        if index_type == "ivf_flat":
            training_suffix = (
                f"_train{min(args.train_samples, database_info.rows)}"
                f"_iter{args.iterations}_seed{args.seed}"
            )
        index_path = args.index_dir / (
            f"faiss_{index_type}_{metric_suffix}_{database_info.rows}x"
            f"{database_info.dim}_nlist{args.nlist}{training_suffix}.index"
        )
        if index_path.exists() and not args.rebuild:
            start = time.perf_counter()
            cpu_index = faiss.read_index(str(index_path))
            load_times[index_type] = (time.perf_counter() - start) * 1000.0
            build_times[index_type] = 0.0
        else:
            start = time.perf_counter()
            if index_type == "flat":
                cpu_index = make_flat_index(faiss, database_info.dim, metric)
            else:
                cpu_index = make_ivf_index(
                    faiss, database_info.dim, metric, args.nlist,
                    args.iterations, args.seed,
                )
                rng = np.random.default_rng(args.seed)
                sample_count = min(args.train_samples, database_info.rows)
                sample_ids = rng.choice(database_info.rows, sample_count, replace=False)
                cpu_index.train(database[sample_ids])
            cpu_index.add(database)
            build_times[index_type] = (time.perf_counter() - start) * 1000.0
            load_times[index_type] = 0.0
            faiss.write_index(cpu_index, str(index_path))
        if cpu_index.ntotal != database_info.rows:
            raise RuntimeError(f"FAISS {index_type} 索引向量数不正确")
        cpu_indexes[index_type] = cpu_index

    max_k = max(top_ks)
    faiss_reference_scores, faiss_reference_ids = cpu_indexes["flat"].search(
        queries, max_k
    )
    if args.reference_results is not None:
        reference_ids, reference_scores = read_result_arrays(
            args.reference_results, query_info.rows, max_k
        )
    else:
        reference_ids, reference_scores = faiss_reference_ids, faiss_reference_scores
    if args.reference_results is not None:
        exact_recall, exact_minimum = recall(reference_ids, faiss_reference_ids)
        exact_mean_error, exact_max_error = score_errors(
            reference_scores, faiss_reference_scores
        )
        print(
            f"FAISS Flat vs project exact recall@{max_k}: "
            f"{exact_recall:.9f}, min={exact_minimum:.9f}, "
            f"mean_score_error={exact_mean_error:.9g}, "
            f"max_score_error={exact_max_error:.9g}"
        )
        if exact_recall < 1.0:
            raise RuntimeError("FAISS Flat 与本项目 exact 的候选 ID 不一致")
    rows: list[BenchmarkRow] = []
    for backend in backends:
        if backend == "gpu" and (
            not hasattr(faiss, "get_num_gpus") or faiss.get_num_gpus() < 1
        ):
            raise RuntimeError("请求了 GPU backend，但 FAISS 没有检测到 GPU")
        for index_type in index_types:
            gpu_transfer_ms = 0.0
            gpu_index_mib = 0.0
            gpu_temp_mib = 0.0
            if backend == "gpu":
                resource = faiss.StandardGpuResources()
                start = time.perf_counter()
                index = faiss.index_cpu_to_gpu(resource, 0, cpu_indexes[index_type])
                gpu_transfer_ms = (time.perf_counter() - start) * 1000.0
                gpu_index_mib, gpu_temp_mib = gpu_memory_mib(resource)
            else:
                index = cpu_indexes[index_type]

            probe_values = [0] if index_type == "flat" else nprobes
            for nprobe in probe_values:
                if index_type == "ivf_flat":
                    index.nprobe = nprobe
                for top_k in top_ks:
                    for _ in range(args.warmup):
                        # 与本项目 CLI 一致：预热覆盖完整 query set，避免
                        # GPU 时钟爬升和 FAISS 延迟初始化进入正式统计。
                        search_batches(index, queries, top_k, args.batch_size)
                    repeat_ms = []
                    all_batch_ms = []
                    last_scores = last_ids = None
                    for _ in range(args.repeat):
                        last_scores, last_ids, total_ms, batch_ms = search_batches(
                            index, queries, top_k, args.batch_size
                        )
                        repeat_ms.append(total_ms)
                        all_batch_ms.extend(batch_ms)
                    assert last_scores is not None and last_ids is not None
                    average_ms = float(np.mean(repeat_ms))
                    reference = reference_ids[:, :top_k]
                    recall_at_k, minimum_recall = recall(reference, last_ids)
                    mean_score_error, max_score_error = score_errors(
                        reference_scores[:, :top_k], last_scores
                    )
                    row = BenchmarkRow(
                        implementation=f"faiss-{getattr(faiss, '__version__', 'unknown')}",
                        backend=backend,
                        index_type=index_type,
                        metric=metric,
                        num_vectors=database_info.rows,
                        num_queries=query_info.rows,
                        dim=database_info.dim,
                        dtype="fp32" if database_info.dtype_code == 1 else "fp16-to-fp32",
                        cpu_threads=args.cpu_threads,
                        top_k=top_k,
                        batch_size=args.batch_size,
                        nlist=0 if index_type == "flat" else args.nlist,
                        nprobe=nprobe,
                        train_samples=0 if index_type == "flat" else min(
                            args.train_samples, database_info.rows
                        ),
                        training_iterations=0 if index_type == "flat" else args.iterations,
                        training_seed=0 if index_type == "flat" else args.seed,
                        build_ms=build_times[index_type],
                        load_ms=load_times[index_type],
                        gpu_transfer_ms=gpu_transfer_ms,
                        gpu_index_mib=gpu_index_mib,
                        gpu_temp_mib=gpu_temp_mib,
                        average_query_ms=average_ms,
                        qps=query_info.rows / (average_ms / 1000.0),
                        run_p50_ms=percentile(repeat_ms, 0.50),
                        run_p99_ms=percentile(repeat_ms, 0.99),
                        run_samples=len(repeat_ms),
                        batch_p50_ms=percentile(all_batch_ms, 0.50),
                        batch_p99_ms=percentile(all_batch_ms, 0.99),
                        batch_latency_samples=len(all_batch_ms),
                        recall_at_k=recall_at_k,
                        min_query_recall=minimum_recall,
                        mean_absolute_score_error=mean_score_error,
                        max_absolute_score_error=max_score_error,
                    )
                    rows.append(row)
                    # 长矩阵每完成一项立即落盘，远端中断时保留已完成结果。
                    write_csv(args.csv, rows)
                    result_path = args.output_dir / (
                        f"faiss_{backend}_{index_type}_k{top_k}_nprobe{nprobe}.txt"
                    )
                    write_results(result_path, last_ids, last_scores)
                    print(json.dumps(asdict(row), ensure_ascii=False))
            if backend == "gpu":
                del index
                del resource
                gc.collect()

    print(f"CSV: {args.csv}")


if __name__ == "__main__":
    main()

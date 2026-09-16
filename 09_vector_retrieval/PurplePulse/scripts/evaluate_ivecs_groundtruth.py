#!/usr/bin/env python3
"""Evaluate WarpFusion text results against TexMex ivecs ground truth."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from import_sift1m import inspect_vecs, record_dtype


def read_result_rows(path: Path) -> tuple[list[list[int]], list[list[float]]]:
    rows: dict[int, list[int]] = {}
    scores: dict[int, list[float]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"{path}:{line_number}: expected three columns")
        query_id = int(parts[0])
        vector_id = int(parts[1])
        score = float(parts[2])
        if query_id < 0 or vector_id < 0:
            raise ValueError(f"{path}:{line_number}: IDs must be non-negative")
        rows.setdefault(query_id, []).append(vector_id)
        scores.setdefault(query_id, []).append(score)
    if not rows or sorted(rows) != list(range(len(rows))):
        raise ValueError("result query IDs must be contiguous and start at zero")
    top_k = len(rows[0])
    if top_k == 0 or any(len(values) != top_k for values in rows.values()):
        raise ValueError("all result queries must have the same positive K")
    return (
        [rows[index] for index in range(len(rows))],
        [scores[index] for index in range(len(rows))],
    )


def read_result_ids(path: Path) -> list[list[int]]:
    return read_result_rows(path)[0]


def read_vectors(path: Path, database: bool):
    with path.open("rb") as source:
        magic = source.read(8)
        expected_magic = b"PPVEC001" if database else b"PPQRY001"
        if magic != expected_magic:
            raise ValueError(f"{path}: invalid vector file magic")
        if database:
            rows, dim, dtype_code, metric_code = struct.unpack("<QIII", source.read(20))
            header_bytes = 28
        else:
            rows, dim, dtype_code = struct.unpack("<QII", source.read(16))
            metric_code = None
            header_bytes = 24
    dtypes = {1: np.dtype("<f4"), 2: np.dtype("<f2")}
    if dtype_code not in dtypes or rows <= 0 or dim <= 0:
        raise ValueError(f"{path}: invalid vector file header")
    expected_bytes = header_bytes + rows * dim * dtypes[dtype_code].itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{path}: truncated or oversized vector file")
    values = np.memmap(
        path,
        dtype=dtypes[dtype_code],
        mode="r",
        offset=header_bytes,
        shape=(rows, dim),
    )
    return values, metric_code


def vector_scores(database: np.ndarray, query: np.ndarray, ids: np.ndarray, metric: int):
    vectors = np.asarray(database[ids], dtype=np.float32)
    query = np.asarray(query, dtype=np.float32)
    if metric == 1:
        differences = vectors - query
        return np.sum(differences * differences, axis=1, dtype=np.float32)
    products = vectors @ query
    if metric == 2:
        return products
    if metric == 3:
        vector_norms = np.linalg.norm(vectors, axis=1)
        query_norm = np.linalg.norm(query)
        return products / (vector_norms * query_norm)
    raise ValueError(f"unsupported metric code: {metric}")


def add_score_equivalence(
    result: dict[str, float | int],
    reference: np.ndarray,
    actual_ids: np.ndarray,
    actual_reported_scores: np.ndarray,
    database_path: Path,
    query_path: Path,
    rtol: float,
    atol: float,
) -> None:
    database, metric = read_vectors(database_path, database=True)
    queries, _ = read_vectors(query_path, database=False)
    if metric is None or len(queries) < len(actual_ids) or database.shape[1] != queries.shape[1]:
        raise ValueError("database/query shape or metric mismatch")
    equivalent_queries = 0
    rank_errors = []
    reported_errors = []
    descending = metric in {2, 3}
    for query_id in range(len(actual_ids)):
        expected_scores = vector_scores(
            database, queries[query_id], reference[query_id], metric
        )
        recomputed_scores = vector_scores(
            database, queries[query_id], actual_ids[query_id], metric
        )
        expected_scores = np.sort(expected_scores)
        recomputed_scores = np.sort(recomputed_scores)
        if descending:
            expected_scores = expected_scores[::-1]
            recomputed_scores = recomputed_scores[::-1]
        errors = np.abs(
            expected_scores.astype(np.float64) - recomputed_scores.astype(np.float64)
        )
        rank_errors.extend(errors.tolist())
        if np.allclose(expected_scores, recomputed_scores, rtol=rtol, atol=atol):
            equivalent_queries += 1
        reported_errors.extend(
            np.abs(
                actual_reported_scores[query_id].astype(np.float64)
                - vector_scores(
                    database, queries[query_id], actual_ids[query_id], metric
                ).astype(np.float64)
            ).tolist()
        )
    result.update(
        {
            "metric": {1: "l2", 2: "inner_product", 3: "cosine"}[metric],
            "score_equivalent_query_fraction": equivalent_queries / len(actual_ids),
            "mean_rank_score_error": float(np.mean(rank_errors)),
            "max_rank_score_error": float(np.max(rank_errors)),
            "mean_reported_score_error": float(np.mean(reported_errors)),
            "max_reported_score_error": float(np.max(reported_errors)),
            "score_rtol": rtol,
            "score_atol": atol,
        }
    )
    del database
    del queries


def evaluate(
    groundtruth_path: Path,
    result_path: Path,
    database_path: Path | None = None,
    query_path: Path | None = None,
    score_rtol: float = 1e-5,
    score_atol: float = 1e-4,
) -> dict[str, float | int | str]:
    actual, actual_scores = read_result_rows(result_path)
    info = inspect_vecs(groundtruth_path, "ivecs")
    if len(actual) > info.rows:
        raise ValueError("results contain more queries than ground truth")
    top_k = len(actual[0])
    if top_k > info.dim:
        raise ValueError(f"result K={top_k} exceeds ground truth K={info.dim}")
    records = np.memmap(
        groundtruth_path,
        dtype=record_dtype(info.dim, "ivecs"),
        mode="r",
        shape=(info.rows,),
    )
    reference = np.asarray(records["values"][: len(actual), :top_k])
    actual_array = np.asarray(actual, dtype=np.int64)
    per_query = np.asarray(
        [
            len(set(expected.tolist()) & set(found.tolist())) / top_k
            for expected, found in zip(reference, actual_array)
        ],
        dtype=np.float64,
    )
    result: dict[str, float | int | str] = {
        "queries": len(actual),
        "top_k": top_k,
        "recall_at_k": float(per_query.mean()),
        "min_query_recall": float(per_query.min()),
        "exact_query_fraction": float(np.mean(per_query == 1.0)),
    }
    if (database_path is None) != (query_path is None):
        raise ValueError("database and queries must be provided together")
    if database_path is not None and query_path is not None:
        add_score_equivalence(
            result,
            reference,
            actual_array,
            np.asarray(actual_scores, dtype=np.float32),
            database_path,
            query_path,
            score_rtol,
            score_atol,
        )
    del records
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--score-rtol", type=float, default=1e-5)
    parser.add_argument("--score-atol", type=float, default=1e-4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = evaluate(
        args.groundtruth,
        args.results,
        args.database,
        args.queries,
        args.score_rtol,
        args.score_atol,
    )
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(
            f"recall@{result['top_k']}={result['recall_at_k']:.6f}, "
            f"min_query_recall={result['min_query_recall']:.6f}, "
            f"exact_query_fraction={result['exact_query_fraction']:.6f}, "
            f"queries={result['queries']}"
        )
        if "score_equivalent_query_fraction" in result:
            print(
                "score_equivalent_query_fraction="
                f"{result['score_equivalent_query_fraction']:.6f}, "
                f"max_rank_score_error={result['max_rank_score_error']:.9g}, "
                f"max_reported_score_error={result['max_reported_score_error']:.9g}"
            )


if __name__ == "__main__":
    main()

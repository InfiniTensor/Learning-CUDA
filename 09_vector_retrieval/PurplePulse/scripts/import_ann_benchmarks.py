#!/usr/bin/env python3
"""Convert an ANN-Benchmarks HDF5 dataset to WarpFusion binary files."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


DATABASE_MAGIC = b"PPVEC001"
QUERY_MAGIC = b"PPQRY001"
FLOAT32 = 1
METRIC_CODES = {
    "euclidean": 1,
    "l2": 1,
    "angular": 3,
    "cosine": 3,
    # VIBE uses this attribute for unit vectors whose L2, cosine, and inner-
    # product rankings are equivalent. WarpFusion records them as cosine so
    # scores and FAISS comparisons have one unambiguous interpretation.
    "normalized": 3,
}


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_distance(value: object) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    distance = str(value).strip().lower()
    if distance not in METRIC_CODES:
        raise ValueError(f"unsupported ANN-Benchmarks distance: {distance}")
    return distance


def write_vector_file(
    dataset,
    destination: Path,
    rows: int,
    metric_code: int | None,
    chunk_rows: int = 16384,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    dim = int(dataset.shape[1])
    with destination.open("wb") as output:
        if metric_code is None:
            output.write(QUERY_MAGIC)
            output.write(struct.pack("<QII", rows, dim, FLOAT32))
        else:
            output.write(DATABASE_MAGIC)
            output.write(struct.pack("<QIII", rows, dim, FLOAT32, metric_code))
        for start in range(0, rows, chunk_rows):
            values = np.ascontiguousarray(
                dataset[start : min(start + chunk_rows, rows)], dtype="<f4"
            )
            if not np.all(np.isfinite(values)):
                raise ValueError("vector data contains NaN or infinity")
            output.write(values.tobytes())


def write_groundtruth_ivecs(
    dataset,
    destination: Path,
    rows: int,
    database_rows: int,
    chunk_rows: int = 16384,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    top_k = int(dataset.shape[1])
    with destination.open("wb") as output:
        for start in range(0, rows, chunk_rows):
            ids = np.ascontiguousarray(
                dataset[start : min(start + chunk_rows, rows)], dtype="<i4"
            )
            if np.any(ids < 0) or np.any(ids >= database_rows):
                raise ValueError("ground truth contains an out-of-range vector ID")
            if np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0):
                raise ValueError("ground truth contains duplicate IDs in one query")
            records = np.empty((len(ids), top_k + 1), dtype="<i4")
            records[:, 0] = top_k
            records[:, 1:] = ids
            output.write(records.tobytes())


def validate_unit_norms(dataset, name: str, chunk_rows: int = 16384) -> None:
    for start in range(0, int(dataset.shape[0]), chunk_rows):
        values = np.asarray(dataset[start : start + chunk_rows], dtype=np.float32)
        norms = np.linalg.norm(values, axis=1)
        if not np.all(np.isfinite(norms)) or not np.allclose(
            norms, 1.0, rtol=5e-4, atol=5e-4
        ):
            maximum_error = float(np.max(np.abs(norms - 1.0)))
            raise ValueError(
                f"{name} declares normalized vectors but unit-norm validation failed; "
                f"maximum absolute norm error={maximum_error}"
            )


def convert(
    input_path: Path,
    database_output: Path,
    query_output: Path,
    groundtruth_output: Path,
    manifest_output: Path,
    query_limit: int | None = None,
    expected_sha256: str | None = None,
    source_url: str | None = None,
    dataset_name: str | None = None,
    expected_dim: int | None = None,
) -> dict[str, object]:
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("h5py is required to import ANN-Benchmarks HDF5 files") from error

    input_sha256 = sha256_file(input_path)
    if expected_sha256 is not None and input_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            f"input SHA-256 mismatch: expected {expected_sha256}, got {input_sha256}"
        )

    with h5py.File(input_path, "r") as source:
        required = {"train", "test", "neighbors"}
        missing = sorted(required - set(source.keys()))
        if missing:
            raise ValueError(f"HDF5 is missing datasets: {missing}")
        train = source["train"]
        test = source["test"]
        neighbors = source["neighbors"]
        if train.ndim != 2 or test.ndim != 2 or neighbors.ndim != 2:
            raise ValueError("train, test, and neighbors must be two-dimensional")
        if train.shape[1] != test.shape[1]:
            raise ValueError("train and test dimensions differ")
        if expected_dim is not None and int(train.shape[1]) != expected_dim:
            raise ValueError(
                f"dataset dimension mismatch: expected {expected_dim}, "
                f"got {int(train.shape[1])}"
            )
        if test.shape[0] != neighbors.shape[0]:
            raise ValueError("test and neighbors row counts differ")
        if query_limit is not None and query_limit <= 0:
            raise ValueError("query_limit must be positive")
        query_rows = int(test.shape[0]) if query_limit is None else min(
            query_limit, int(test.shape[0])
        )
        distance = normalized_distance(source.attrs.get("distance", "euclidean"))
        if distance == "normalized":
            validate_unit_norms(train, "train")
            validate_unit_norms(test, "test")
        metric_code = METRIC_CODES[distance]
        write_vector_file(train, database_output, int(train.shape[0]), metric_code)
        write_vector_file(test, query_output, query_rows, None)
        write_groundtruth_ivecs(
            neighbors, groundtruth_output, query_rows, int(train.shape[0])
        )
        shape = {
            "database_rows": int(train.shape[0]),
            "query_rows": query_rows,
            "dim": int(train.shape[1]),
            "groundtruth_k": int(neighbors.shape[1]),
        }

    outputs = {}
    for name, path in {
        "database": database_output,
        "queries": query_output,
        "groundtruth": groundtruth_output,
    }.items():
        outputs[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest: dict[str, object] = {
        "dataset": dataset_name or input_path.stem,
        "source_url": source_url,
        "source_path": str(input_path),
        "source_bytes": input_path.stat().st_size,
        "source_sha256": input_sha256,
        "distance": distance,
        "warpfusion_metric": "cosine" if metric_code == 3 else "l2",
        **shape,
        "outputs": outputs,
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--database-output", type=Path, required=True)
    parser.add_argument("--query-output", type=Path, required=True)
    parser.add_argument("--groundtruth-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--source-url")
    parser.add_argument(
        "--dataset-name",
        help="Name recorded in the provenance manifest (defaults to the input stem)",
    )
    parser.add_argument("--expected-dim", type=int)
    args = parser.parse_args()
    manifest = convert(
        args.input,
        args.database_output,
        args.query_output,
        args.groundtruth_output,
        args.manifest_output,
        args.query_limit,
        args.expected_sha256,
        args.source_url,
        args.dataset_name,
        args.expected_dim,
    )
    print(
        f"{manifest['dataset']}: {manifest['database_rows']} x {manifest['dim']}, "
        f"queries={manifest['query_rows']}, groundtruth_k={manifest['groundtruth_k']}, "
        f"distance={manifest['distance']}"
    )
    print(f"manifest: {args.manifest_output}")


if __name__ == "__main__":
    main()

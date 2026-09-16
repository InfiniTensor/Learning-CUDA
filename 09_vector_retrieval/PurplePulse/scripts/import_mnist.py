#!/usr/bin/env python3
"""Convert the public Keras MNIST archive to the project vector format."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def images_to_vectors(images: np.ndarray) -> np.ndarray:
    if images.ndim < 2:
        raise ValueError("MNIST image array must have at least two dimensions")
    rows = images.shape[0]
    return np.ascontiguousarray(images.reshape(rows, -1), dtype="<f4") / 255.0


def write_database(path: Path, vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as destination:
        destination.write(b"PPVEC001")
        destination.write(
            struct.pack("<QIII", len(vectors), vectors.shape[1], 1, 1)
        )
        destination.write(vectors.tobytes())


def write_queries(path: Path, vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as destination:
        destination.write(b"PPQRY001")
        destination.write(struct.pack("<QII", len(vectors), vectors.shape[1], 1))
        destination.write(vectors.tobytes())


def convert(
    input_path: Path,
    database_path: Path,
    query_path: Path,
    database_size: int | None = None,
    query_count: int | None = None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    with np.load(input_path) as archive:
        if "x_train" not in archive or "x_test" not in archive:
            raise ValueError("MNIST archive must contain x_train and x_test")
        database = images_to_vectors(archive["x_train"])
        queries = images_to_vectors(archive["x_test"])
    if database.shape[1] != queries.shape[1]:
        raise ValueError("MNIST train and test dimensions differ")
    if database_size is not None:
        database = database[:database_size]
    if query_count is not None:
        queries = queries[:query_count]
    if len(database) == 0 or len(queries) == 0:
        raise ValueError("database and query counts must be positive")
    write_database(database_path, database)
    write_queries(query_path, queries)
    return database.shape, queries.shape


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--database-size", type=int)
    parser.add_argument("--query-count", type=int)
    args = parser.parse_args()
    database_shape, query_shape = convert(
        args.input,
        args.database,
        args.queries,
        args.database_size,
        args.query_count,
    )
    print(f"database: {database_shape[0]} x {database_shape[1]}")
    print(f"queries: {query_shape[0]} x {query_shape[1]}")
    print("dtype/metric: fp32/l2")


if __name__ == "__main__":
    main()

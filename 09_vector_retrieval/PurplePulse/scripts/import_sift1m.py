#!/usr/bin/env python3
"""Convert TexMex SIFT1M fvecs/ivecs files to WarpFusion input files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


OFFICIAL_SOURCE = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
DATABASE_MAGIC = b"PPVEC001"
QUERY_MAGIC = b"PPQRY001"
FLOAT32 = 1
L2 = 1


@dataclass(frozen=True)
class VecsInfo:
    path: str
    kind: str
    rows: int
    dim: int
    bytes: int
    sha256: str


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def record_dtype(dim: int, kind: str) -> np.dtype:
    value_dtype = "<f4" if kind == "fvecs" else "<i4"
    return np.dtype([("dim", "<i4"), ("values", value_dtype, (dim,))])


def inspect_vecs(path: Path, kind: str, chunk_rows: int = 65536) -> VecsInfo:
    if kind not in {"fvecs", "ivecs"}:
        raise ValueError("kind must be fvecs or ivecs")
    if not path.is_file() or path.stat().st_size < 4:
        raise ValueError(f"{path}: missing or too small")
    dim = int(np.fromfile(path, dtype="<i4", count=1)[0])
    if dim <= 0 or dim > 1_000_000:
        raise ValueError(f"{path}: invalid row dimension {dim}")
    record_bytes = (dim + 1) * 4
    size = path.stat().st_size
    if size % record_bytes != 0:
        raise ValueError(
            f"{path}: size {size} is not divisible by record size {record_bytes}"
        )
    rows = size // record_bytes
    if rows == 0:
        raise ValueError(f"{path}: no rows")
    records = np.memmap(path, dtype=record_dtype(dim, kind), mode="r", shape=(rows,))
    for start in range(0, rows, chunk_rows):
        if not np.all(records["dim"][start : start + chunk_rows] == dim):
            raise ValueError(f"{path}: inconsistent row dimensions")
    del records
    return VecsInfo(
        path=str(path),
        kind=kind,
        rows=rows,
        dim=dim,
        bytes=size,
        sha256=sha256_file(path),
    )


def write_warpfusion_vectors(
    source: Path,
    destination: Path,
    info: VecsInfo,
    rows: int,
    database: bool,
    chunk_rows: int = 16384,
) -> None:
    if info.kind != "fvecs":
        raise ValueError("WarpFusion vectors must be imported from fvecs")
    if rows <= 0 or rows > info.rows:
        raise ValueError(f"requested rows must be in [1, {info.rows}]")
    records = np.memmap(
        source,
        dtype=record_dtype(info.dim, "fvecs"),
        mode="r",
        shape=(info.rows,),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        if database:
            output.write(DATABASE_MAGIC)
            output.write(struct.pack("<QIII", rows, info.dim, FLOAT32, L2))
        else:
            output.write(QUERY_MAGIC)
            output.write(struct.pack("<QII", rows, info.dim, FLOAT32))
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            values = np.ascontiguousarray(records["values"][start:end], dtype="<f4")
            output.write(values.tobytes())
    del records


def validate_groundtruth(
    path: Path,
    info: VecsInfo,
    query_rows: int,
    database_rows: int,
) -> None:
    if info.kind != "ivecs":
        raise ValueError("ground truth must use ivecs")
    if info.rows < query_rows:
        raise ValueError(
            f"ground truth has {info.rows} rows but {query_rows} queries were requested"
        )
    records = np.memmap(
        path,
        dtype=record_dtype(info.dim, "ivecs"),
        mode="r",
        shape=(info.rows,),
    )
    values = np.asarray(records["values"][:query_rows])
    if np.any(values < 0) or np.any(values >= database_rows):
        raise ValueError("ground truth contains an out-of-range vector ID")
    if np.any(np.diff(np.sort(values, axis=1), axis=1) == 0):
        raise ValueError("ground truth contains duplicate IDs in one query")
    del records


def convert(
    base_path: Path,
    query_path: Path,
    groundtruth_path: Path,
    database_output: Path,
    query_output: Path,
    manifest_output: Path,
    query_limit: int | None = None,
    source_url: str = OFFICIAL_SOURCE,
    groundtruth_output: Path | None = None,
) -> dict[str, object]:
    base = inspect_vecs(base_path, "fvecs")
    queries = inspect_vecs(query_path, "fvecs")
    groundtruth = inspect_vecs(groundtruth_path, "ivecs")
    if base.dim != queries.dim:
        raise ValueError(f"base dim {base.dim} differs from query dim {queries.dim}")
    if query_limit is not None and query_limit <= 0:
        raise ValueError("query_limit must be positive")
    query_rows = queries.rows if query_limit is None else min(query_limit, queries.rows)
    validate_groundtruth(groundtruth_path, groundtruth, query_rows, base.rows)
    write_warpfusion_vectors(base_path, database_output, base, base.rows, True)
    write_warpfusion_vectors(query_path, query_output, queries, query_rows, False)
    if groundtruth_output is not None:
        groundtruth_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(groundtruth_path, groundtruth_output)

    output_paths = {
        "database": database_output,
        "queries": query_output,
    }
    if groundtruth_output is not None:
        output_paths["groundtruth"] = groundtruth_output

    manifest: dict[str, object] = {
        "dataset": "SIFT1M",
        "source_url": source_url,
        "metric": "l2",
        "database_rows": base.rows,
        "query_rows": query_rows,
        "dim": base.dim,
        "groundtruth_k": groundtruth.dim,
        "inputs": {
            "base": asdict(base),
            "queries": asdict(queries),
            "groundtruth": asdict(groundtruth),
        },
        "outputs": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in output_paths.items()
        },
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--database-output", type=Path, required=True)
    parser.add_argument("--query-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--groundtruth-output", type=Path)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--source-url", default=OFFICIAL_SOURCE)
    args = parser.parse_args()
    manifest = convert(
        args.base,
        args.queries,
        args.groundtruth,
        args.database_output,
        args.query_output,
        args.manifest_output,
        args.query_limit,
        args.source_url,
        args.groundtruth_output,
    )
    print(
        f"SIFT1M: {manifest['database_rows']} x {manifest['dim']}, "
        f"queries={manifest['query_rows']}, groundtruth_k={manifest['groundtruth_k']}"
    )
    print(f"manifest: {args.manifest_output}")


if __name__ == "__main__":
    main()

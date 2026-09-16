#!/usr/bin/env python3
"""Download, verify, and convert the two Stage-6 public datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

from import_ann_benchmarks import convert as convert_hdf5
from import_sift1m import convert as convert_sift


SIFT_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
SIFT_MD5 = "b23d1b3b2ee8469d819b61ca900ef0ed"
VIBE_NAME = "yahoo-minilm-384-normalized"
VIBE_URL = (
    "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/"
    f"{VIBE_NAME}.hdf5"
)
VIBE_MIRROR_URL = (
    "https://hf-mirror.com/datasets/vector-index-bench/vibe/resolve/main/"
    f"{VIBE_NAME}.hdf5"
)
VIBE_SHA256 = "8b0519850249d7ea4aea258eb43ae837dca1bb2f7a85ee0a5145afdb699c4136"


def file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, algorithm: str, expected: str) -> None:
    actual = file_hash(path, algorithm)
    if actual.lower() != expected.lower():
        raise ValueError(
            f"{path}: {algorithm} mismatch: expected {expected}, got {actual}"
        )


def download(url: str, destination: Path, algorithm: str, expected: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verify(destination, algorithm, expected)
        print(f"verified cached download: {destination}")
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "WarpFusion/0.1"})
    downloaded = 0
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
        total = int(response.headers.get("Content-Length", "0"))
        while chunk := response.read(8 * 1024 * 1024):
            out.write(chunk)
            downloaded += len(chunk)
            if total:
                print(
                    f"\r{destination.name}: {downloaded / (1024**2):.1f}/"
                    f"{total / (1024**2):.1f} MiB",
                    end="",
                    flush=True,
                )
    print()
    verify(partial, algorithm, expected)
    partial.replace(destination)
    return destination


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as source:
        for member in source.getmembers():
            target = (destination / member.name).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"unsafe archive member: {member.name}")
        source.extractall(destination)


def prepare_sift(raw_dir: Path, output_dir: Path, skip_download: bool) -> dict:
    archive = raw_dir / "sift1m" / "sift.tar.gz"
    if not skip_download:
        download(SIFT_URL, archive, "md5", SIFT_MD5)
    elif not archive.is_file():
        raise FileNotFoundError(archive)
    else:
        verify(archive, "md5", SIFT_MD5)

    extracted = raw_dir / "sift1m" / "extracted"
    required = {
        "base": extracted / "sift" / "sift_base.fvecs",
        "queries": extracted / "sift" / "sift_query.fvecs",
        "groundtruth": extracted / "sift" / "sift_groundtruth.ivecs",
    }
    if not all(path.is_file() for path in required.values()):
        safe_extract(archive, extracted)

    destination = output_dir / "sift1m"
    manifest = convert_sift(
        required["base"],
        required["queries"],
        required["groundtruth"],
        destination / "database.bin",
        destination / "queries.bin",
        destination / "manifest.json",
        source_url=SIFT_URL,
        groundtruth_output=destination / "groundtruth.ivecs",
    )
    manifest["archive"] = {
        "path": str(archive),
        "bytes": archive.stat().st_size,
        "md5": file_hash(archive, "md5"),
        "sha256": file_hash(archive, "sha256"),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def prepare_vibe(raw_dir: Path, output_dir: Path, skip_download: bool) -> dict:
    source = raw_dir / "vibe" / f"{VIBE_NAME}.hdf5"
    if not skip_download:
        try:
            download(VIBE_URL, source, "sha256", VIBE_SHA256)
        except Exception as primary_error:
            print(
                f"primary Hugging Face download failed ({primary_error}); "
                "trying the checksum-equivalent mirror",
                file=sys.stderr,
            )
            download(VIBE_MIRROR_URL, source, "sha256", VIBE_SHA256)
    elif not source.is_file():
        raise FileNotFoundError(source)
    else:
        verify(source, "sha256", VIBE_SHA256)

    destination = output_dir / VIBE_NAME
    return convert_hdf5(
        source,
        destination / "database.bin",
        destination / "queries.bin",
        destination / "groundtruth.ivecs",
        destination / "manifest.json",
        expected_sha256=VIBE_SHA256,
        source_url=VIBE_URL,
        dataset_name=VIBE_NAME,
        expected_dim=384,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("all", "sift1m", VIBE_NAME),
        default="all",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/real/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Require already-downloaded archives, but still verify their checksums",
    )
    args = parser.parse_args()

    selected = (
        ("sift1m", VIBE_NAME) if args.dataset == "all" else (args.dataset,)
    )
    for name in selected:
        try:
            manifest = (
                prepare_sift(args.raw_dir, args.output_dir, args.skip_download)
                if name == "sift1m"
                else prepare_vibe(args.raw_dir, args.output_dir, args.skip_download)
            )
        except Exception as error:
            print(f"failed to prepare {name}: {error}", file=sys.stderr)
            raise
        print(
            f"prepared {manifest['dataset']}: {manifest['database_rows']} x "
            f"{manifest['dim']}, queries={manifest['query_rows']}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate and validate deterministic QDAT matrices using only the standard library."""

from __future__ import annotations

import argparse
import math
import random
import struct
from pathlib import Path
from typing import Iterable

MAGIC = b"QDAT"
VERSION = 1
HEADER = struct.Struct("<4sHBBQQ")
DTYPES = {"fp16": (1, "e", 2), "fp32": (2, "f", 4)}
DTYPE_IDS = {value[0]: (name, value[1], value[2]) for name, value in DTYPES.items()}
DTYPE_IDS[3] = ("bf16", None, 2)


def write_qdat(path: Path, rows: int, cols: int, dtype: str, values: Iterable[float]) -> None:
    dtype_id, format_char, _ = DTYPES[dtype]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(HEADER.pack(MAGIC, VERSION, dtype_id, 0, rows, cols))
        pack_value = struct.Struct("<" + format_char).pack
        for value in values:
            output.write(pack_value(value))


def inspect_qdat(path: Path) -> None:
    with path.open("rb") as source:
        raw_header = source.read(HEADER.size)
        if len(raw_header) != HEADER.size:
            raise ValueError(f"{path}: truncated QDAT header")
        magic, version, dtype_id, reserved, rows, cols = HEADER.unpack(raw_header)
        if magic != MAGIC or version != VERSION or reserved != 0 or dtype_id not in DTYPE_IDS:
            raise ValueError(f"{path}: invalid QDAT header")
        dtype, format_char, item_size = DTYPE_IDS[dtype_id]
        payload = source.read()

    expected_size = rows * cols * item_size
    if len(payload) != expected_size:
        raise ValueError(f"{path}: expected {expected_size} payload bytes, got {len(payload)}")

    unpack = struct.Struct("<" + format_char).unpack_from if format_char else None
    finite_values = []
    non_finite = 0
    for offset in range(0, len(payload), item_size):
        if dtype == "bf16":
            bf16_bits = struct.unpack_from("<H", payload, offset)[0]
            value = struct.unpack("<f", struct.pack("<I", bf16_bits << 16))[0]
        else:
            value = float(unpack(payload, offset)[0])
        if math.isfinite(value):
            finite_values.append(value)
        else:
            non_finite += 1

    if finite_values:
        value_range = f"[{min(finite_values):.7g}, {max(finite_values):.7g}]"
    else:
        value_range = "n/a"
    print(
        f"{path}: shape={rows}x{cols}, dtype={dtype}, payload={len(payload)} B, "
        f"finite_range={value_range}, non_finite={non_finite}"
    )


def make_values(kind: str, count: int, rng: random.Random, dtype: str) -> list[float]:
    if kind == "uniform":
        return [rng.uniform(-6.0, 6.0) for _ in range(count)]
    if kind == "normal":
        return [rng.gauss(0.0, 1.0) for _ in range(count)]
    if kind == "outlier":
        values = [rng.gauss(0.0, 0.25) for _ in range(count)]
        sentinels = [0.0, -0.0, 2.0**-20, -(2.0**-20), 6.0, -6.0, 448.0, -448.0]
        if dtype == "fp32":
            sentinels += [float("inf"), float("-inf"), float("nan")]
        for index, value in enumerate(sentinels[:count]):
            values[index] = value
        if count > len(sentinels):
            values[-1] = 1.0e4 if dtype == "fp16" else 1.0e20
        return values
    raise ValueError(f"unknown matrix kind: {kind}")


def generate_one(args: argparse.Namespace, kind: str, output: Path) -> None:
    if args.rows <= 0 or args.cols <= 0:
        raise ValueError("rows and cols must be positive")
    rng = random.Random(args.seed)
    values = make_values(kind, args.rows * args.cols, rng, args.dtype)
    write_qdat(output, args.rows, args.cols, args.dtype, values)
    inspect_qdat(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate one QDAT matrix")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--kind", choices=("uniform", "normal", "outlier"), required=True)

    suite = subparsers.add_parser("suite", help="generate uniform, normal, and outlier matrices")
    suite.add_argument("--output-dir", type=Path, required=True)

    inspect = subparsers.add_parser("inspect", help="validate and summarize QDAT files")
    inspect.add_argument("paths", type=Path, nargs="+")

    for command in (generate, suite):
        command.add_argument("--rows", type=int, default=64)
        command.add_argument("--cols", type=int, default=65)
        command.add_argument("--dtype", choices=tuple(DTYPES), default="fp32")
        command.add_argument("--seed", type=int, default=20260827)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "inspect":
        for path in args.paths:
            inspect_qdat(path)
        return
    if args.command == "generate":
        generate_one(args, args.kind, args.output)
        return
    for kind in ("uniform", "normal", "outlier"):
        output = args.output_dir / f"{kind}_{args.rows}x{args.cols}_{args.dtype}.qdat"
        generate_one(args, kind, output)


if __name__ == "__main__":
    main()

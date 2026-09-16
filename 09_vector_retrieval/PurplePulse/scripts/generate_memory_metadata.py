#!/usr/bin/env python3
"""Generate deterministic Agent-memory metadata in the project binary format."""

from __future__ import annotations

import argparse
import random
import struct
from pathlib import Path


MAGIC = b"PPMETA01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-vectors", type=int, required=True)
    parser.add_argument("--sessions", type=int, default=16)
    parser.add_argument("--source-types", type=int, default=4)
    parser.add_argument("--start-timestamp", type=int, default=1_700_000_000)
    parser.add_argument("--timestamp-span", type=int, default=2_592_000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.num_vectors <= 0
        or args.sessions <= 0
        or args.source_types <= 0
        or args.start_timestamp < 0
        or args.timestamp_span < 0
    ):
        raise SystemExit("counts must be positive and timestamps non-negative")
    if args.sessions >= 2**32 or args.source_types >= 2**32:
        raise SystemExit("sessions and source-types must fit uint32")

    rng = random.Random(args.seed)
    timestamps = [
        args.start_timestamp + rng.randrange(args.timestamp_span + 1)
        for _ in range(args.num_vectors)
    ]
    importance = [rng.random() for _ in range(args.num_vectors)]
    sessions = [rng.randrange(args.sessions) for _ in range(args.num_vectors)]
    source_types = [
        rng.randrange(args.source_types) for _ in range(args.num_vectors)
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        output.write(MAGIC)
        output.write(struct.pack("<Q", args.num_vectors))
        output.write(struct.pack(f"<{args.num_vectors}Q", *timestamps))
        output.write(struct.pack(f"<{args.num_vectors}f", *importance))
        output.write(struct.pack(f"<{args.num_vectors}I", *sessions))
        output.write(struct.pack(f"<{args.num_vectors}I", *source_types))

    print(
        f"wrote {args.num_vectors} metadata rows to {args.output} "
        f"({args.sessions} sessions, {args.source_types} source types)"
    )


if __name__ == "__main__":
    main()

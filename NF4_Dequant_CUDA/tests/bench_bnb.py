#!/usr/bin/env python3
"""
Benchmark bitsandbytes NF4 double-quant dequant reference time (ms).

This script measures the same reference path used in compare_with_bnb.py:
1) dequantize absmax_q via bitsandbytes.dequantize_blockwise
2) high-first nibble unpack
3) NF4 table lookup and scale multiply

Output `bnb_time_ms` can be written back to params.txt for speedup reporting.
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np


HEADER_BYTES = 8 + 8 + 4
CODE2_ENTRIES = 256
CODE2_BYTES = CODE2_ENTRIES * 2
OFFSET_BYTES = 4
STEM_RE = re.compile(r"^nf4_r(\d+)_c(\d+)_bs(\d+)_bpg(\d+)$")


def _load_torch_bnb():
    try:
        import torch
        import bitsandbytes.functional as F
    except Exception as exc:
        raise SystemExit(
            "torch + bitsandbytes are required. Install with:\n"
            "  pip install torch bitsandbytes numpy\n"
            f"Import error: {exc}"
        )
    return torch, F


def _read_nf4_binary(path: Path) -> dict[str, Any]:
    blob = path.read_bytes()
    if len(blob) < HEADER_BYTES + CODE2_BYTES + OFFSET_BYTES:
        raise RuntimeError(f"File too small: {path}")

    rows, cols, blocksize = struct.unpack_from("<qqi", blob, 0)
    if rows <= 0 or cols <= 0 or blocksize <= 0:
        raise RuntimeError(f"Invalid header in {path}")

    num_elements = rows * cols
    num_packed = (num_elements + 1) // 2
    num_blocks = (num_elements + blocksize - 1) // blocksize

    payload_bytes = len(blob) - HEADER_BYTES
    fixed_prefix = num_packed + num_blocks
    fixed_tail = CODE2_BYTES + OFFSET_BYTES
    if payload_bytes < fixed_prefix + fixed_tail:
        raise RuntimeError(f"Invalid payload size in {path}")

    absmax2_bytes = payload_bytes - fixed_prefix - fixed_tail
    if absmax2_bytes % 2 != 0:
        raise RuntimeError(f"Invalid absmax2 length in {path}")
    num_groups = absmax2_bytes // 2

    off = HEADER_BYTES
    packed = np.frombuffer(blob, dtype=np.uint8, count=num_packed, offset=off).copy()
    off += num_packed
    absmax_q = np.frombuffer(blob, dtype=np.uint8, count=num_blocks, offset=off).copy()
    off += num_blocks
    absmax2 = np.frombuffer(blob, dtype=np.float16, count=num_groups, offset=off).copy()
    off += absmax2_bytes
    code2 = np.frombuffer(blob, dtype=np.float16, count=CODE2_ENTRIES, offset=off).copy()
    off += CODE2_BYTES
    (offset,) = struct.unpack_from("<f", blob, off)

    return {
        "rows": int(rows),
        "cols": int(cols),
        "blocksize": int(blocksize),
        "num_elements": int(num_elements),
        "num_blocks": int(num_blocks),
        "packed": packed,
        "absmax_q": absmax_q,
        "absmax2": absmax2,
        "code2": code2,
        "offset": float(offset),
    }


def _parse_params_file(path: Path) -> dict[str, Any]:
    params: dict[str, Any] = {
        "blocksize": None,
        "compute_type": "bf16",
        "target_gpu": "T4",
        "blocks_per_group": 256,
        "bnb_time_ms": None,
    }
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key == "blocksize":
            params[key] = int(value)
        elif key == "compute_type":
            params[key] = value
        elif key == "target_gpu":
            params[key] = value
        elif key == "blocks_per_group":
            params[key] = int(value)
        elif key == "bnb_time_ms":
            params[key] = float(value)
    return params


def _write_params_file(path: Path, params: dict[str, Any]) -> None:
    lines = [
        f"blocksize = {int(params['blocksize'])}",
        f"compute_type = \"{params['compute_type']}\"",
        f"target_gpu = \"{params['target_gpu']}\"",
        f"blocks_per_group = {int(params['blocks_per_group'])}",
    ]
    if params.get("bnb_time_ms") is not None:
        lines.append(f"bnb_time_ms = {float(params['bnb_time_ms']):.6f}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _infer_bpg_from_name(weights_bin: Path) -> int | None:
    stem = weights_bin.stem
    # Accept patterns like ".../nf4_r64_c128_bs64_bpg256_weights.bin"
    if stem.endswith("_weights"):
        stem = stem[: -len("_weights")]
    m = STEM_RE.match(stem)
    if m is None:
        return None
    return int(m.group(4))


def _benchmark_bnb(
    parsed: dict[str, Any],
    blocks_per_group: int,
    warmup: int,
    iters: int,
) -> float:
    torch, F = _load_torch_bnb()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable in current Python env (torch.cuda.is_available() == False). "
            "Install CUDA-enabled PyTorch first."
        )

    device = torch.device("cuda")
    n = int(parsed["num_elements"])
    blocksize = int(parsed["blocksize"])

    packed_t = torch.from_numpy(parsed["packed"]).to(device=device, dtype=torch.uint8)
    absmax_q_t = torch.from_numpy(parsed["absmax_q"]).to(device=device, dtype=torch.uint8)
    absmax2_t = torch.from_numpy(parsed["absmax2"].astype(np.float32, copy=False)).to(
        device=device
    )
    code2_t = torch.from_numpy(parsed["code2"].astype(np.float32, copy=False)).to(device=device)
    offset_t = torch.tensor(parsed["offset"], device=device, dtype=torch.float32)
    block_ids = torch.arange(n, device=device, dtype=torch.int64) // blocksize
    nf4_t = F.get_4bit_type("nf4", device=device).to(dtype=torch.float32)

    state2 = F.QuantState(
        absmax=absmax2_t,
        code=code2_t,
        blocksize=blocks_per_group,
        dtype=torch.float32,
    )

    def bnb_ref():
        centered_absmax = F.dequantize_blockwise(
            absmax_q_t,
            quant_state=state2,
            blocksize=blocks_per_group,
        )
        absmax = centered_absmax + offset_t

        nib = torch.empty(n, dtype=torch.uint8, device=device)
        nib[0::2] = packed_t >> 4
        if n > 1:
            nib[1::2] = packed_t[: n // 2] & 0x0F

        scales = absmax[block_ids]
        return nf4_t[nib.long()] * scales

    for _ in range(warmup):
        _ = bnb_ref()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = bnb_ref()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / max(iters, 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark bitsandbytes reference time.")
    parser.add_argument(
        "--weights-bin",
        type=Path,
        required=True,
        help="Input NF4 binary (header + packed + absmax_q + absmax2 + code2 + offset).",
    )
    parser.add_argument(
        "--blocks-per-group",
        type=int,
        default=None,
        help="Double-quant blocks_per_group. Priority: this arg > params.txt > infer from filename > 256.",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=None,
        help="Optional params.txt (used for fallback blocks_per_group and update target).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup iterations.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=200,
        help="Benchmark iterations (averaged).",
    )
    parser.add_argument(
        "--update-params",
        action="store_true",
        help="Write measured bnb_time_ms back to --params file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.iters <= 0:
        print("warmup must be >= 0 and iters must be > 0", file=sys.stderr)
        return 2

    if args.update_params and args.params is None:
        print("--update-params requires --params", file=sys.stderr)
        return 2

    parsed = _read_nf4_binary(args.weights_bin)

    params_data: dict[str, Any] | None = None
    if args.params is not None:
        if not args.params.exists():
            print(f"params file not found: {args.params}", file=sys.stderr)
            return 2
        params_data = _parse_params_file(args.params)

    if args.blocks_per_group is not None:
        bpg = int(args.blocks_per_group)
    elif params_data is not None and params_data.get("blocks_per_group") is not None:
        bpg = int(params_data["blocks_per_group"])
    else:
        inferred = _infer_bpg_from_name(args.weights_bin)
        bpg = int(inferred) if inferred is not None else 256

    if bpg <= 0:
        print(f"invalid blocks_per_group: {bpg}", file=sys.stderr)
        return 2

    try:
        bnb_time_ms = _benchmark_bnb(
            parsed=parsed,
            blocks_per_group=bpg,
            warmup=int(args.warmup),
            iters=int(args.iters),
        )
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    print(
        "Benchmark result: "
        f"rows={parsed['rows']} cols={parsed['cols']} "
        f"blocksize={parsed['blocksize']} blocks_per_group={bpg} "
        f"warmup={args.warmup} iters={args.iters} "
        f"bnb_time_ms={bnb_time_ms:.6f}"
    )

    if args.update_params:
        assert params_data is not None
        if params_data["blocksize"] is None:
            params_data["blocksize"] = int(parsed["blocksize"])
        params_data["bnb_time_ms"] = float(bnb_time_ms)
        _write_params_file(args.params, params_data)
        print(f"Updated params: {args.params} (bnb_time_ms={bnb_time_ms:.6f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

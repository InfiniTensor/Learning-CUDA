#!/usr/bin/env python3
"""
Generate NF4 double-quant test data in the exact binary layout expected by this project:

    [header]
      int64 num_rows
      int64 num_cols
      int32 blocksize
    [payload]
      uint8 packed_weights[(num_elements + 1) // 2]
      uint8 absmax_q[num_blocks]
      fp16  absmax2[num_groups]
      fp16  code2[256]
      fp32  offset
"""

from __future__ import annotations

import argparse
import inspect
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class CaseSpec:
    rows: int
    cols: int
    blocksize: int
    blocks_per_group: int

    @property
    def stem(self) -> str:
        return (
            f"nf4_r{self.rows}_c{self.cols}_"
            f"bs{self.blocksize}_bpg{self.blocks_per_group}"
        )


def _load_bitsandbytes():
    try:
        import bitsandbytes as bnb
        import bitsandbytes.functional as F
    except Exception as exc:
        raise SystemExit(
            "bitsandbytes is required. Install with: pip install bitsandbytes torch numpy\n"
            f"Import error: {exc}"
        )
    return bnb, F


def _call_quantize_blockwise(F, x: Any, blocksize: int):
    sig = inspect.signature(F.quantize_blockwise)
    kwargs = {"blocksize": blocksize}
    if "nested" in sig.parameters:
        kwargs["nested"] = False
    return F.quantize_blockwise(x, **kwargs)


def _to_numpy_u8(x: Any) -> np.ndarray:
    return x.detach().contiguous().view(-1).cpu().numpy().astype(np.uint8, copy=False)


def _to_numpy_f16(x: Any) -> np.ndarray:
    return x.detach().contiguous().view(-1).cpu().numpy().astype(np.float16, copy=False)


def _write_params_file(
    path: Path,
    blocksize: int,
    compute_type: str,
    blocks_per_group: int,
    target_gpu: str = "T4",
    bnb_time_ms: float | None = None,
) -> None:
    lines = [
        f"blocksize = {blocksize}",
        f'compute_type = "{compute_type}"',
        f'target_gpu = "{target_gpu}"',
        f"blocks_per_group = {blocks_per_group}",
    ]
    if bnb_time_ms is not None and bnb_time_ms > 0.0:
        lines.append(f"bnb_time_ms = {bnb_time_ms:.6f}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_dataset(
    rows: int,
    cols: int,
    blocksize: int,
    blocks_per_group: int,
    seed: int,
    compute_type: str,
    out_dir: Path,
    stem: str,
    save_ref_fp32: bool = True,
    write_dual_params: bool = False,
) -> None:
    import torch

    bnb, F = _load_bitsandbytes()

    if blocksize <= 0:
        raise ValueError("blocksize must be > 0")
    if blocks_per_group <= 0:
        raise ValueError("blocks_per_group must be > 0")

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn((rows, cols), generator=g, dtype=torch.float32, device=device)

    # NF4 quantization.
    qweight, qstate = F.quantize_4bit(
        x,
        blocksize=blocksize,
        compress_statistics=False,
        quant_type="nf4",
    )

    # Double quantization of absmax (equivalent to bnb_4bit_use_double_quant=True path).
    absmax = qstate.absmax.float()
    offset = absmax.mean()
    centered_absmax = absmax - offset
    absmax_q, state2 = _call_quantize_blockwise(F, centered_absmax, blocks_per_group)

    packed_weights = _to_numpy_u8(qweight)
    absmax_q_u8 = _to_numpy_u8(absmax_q)
    absmax2_f16 = _to_numpy_f16(state2.absmax.float())
    code2_f16 = _to_numpy_f16(state2.code.float())

    num_elements = rows * cols
    num_packed = (num_elements + 1) // 2
    num_blocks = (num_elements + blocksize - 1) // blocksize
    num_groups = (num_blocks + blocks_per_group - 1) // blocks_per_group

    if packed_weights.size != num_packed:
        raise RuntimeError(
            f"Packed size mismatch: got {packed_weights.size}, expected {num_packed}"
        )
    if absmax_q_u8.size != num_blocks:
        raise RuntimeError(
            f"absmax_q size mismatch: got {absmax_q_u8.size}, expected {num_blocks}"
        )
    if absmax2_f16.size != num_groups:
        raise RuntimeError(
            f"absmax2 size mismatch: got {absmax2_f16.size}, expected {num_groups}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / f"{stem}_weights.bin"
    params_path = out_dir / f"{stem}_params.txt"
    meta_path = out_dir / f"{stem}_meta.json"
    ref_path = out_dir / f"{stem}_ref_fp32.npy"

    with open(bin_path, "wb") as f:
        f.write(struct.pack("<qqi", rows, cols, blocksize))
        f.write(packed_weights.tobytes(order="C"))
        f.write(absmax_q_u8.tobytes(order="C"))
        f.write(absmax2_f16.tobytes(order="C"))
        f.write(code2_f16.tobytes(order="C"))
        f.write(struct.pack("<f", float(offset.item())))

    _write_params_file(
        path=params_path,
        blocksize=blocksize,
        compute_type=compute_type,
        blocks_per_group=blocks_per_group,
        target_gpu="T4",
    )

    if write_dual_params:
        _write_params_file(
            path=out_dir / f"{stem}_params_bf16.txt",
            blocksize=blocksize,
            compute_type="bf16",
            blocks_per_group=blocks_per_group,
            target_gpu="T4",
        )
        _write_params_file(
            path=out_dir / f"{stem}_params_fp16.txt",
            blocksize=blocksize,
            compute_type="fp16",
            blocks_per_group=blocks_per_group,
            target_gpu="T4",
        )

    if save_ref_fp32:
        np.save(ref_path, x.detach().cpu().numpy().astype(np.float32, copy=False))

    meta = {
        "bitsandbytes_version": getattr(bnb, "__version__", "unknown"),
        "rows": rows,
        "cols": cols,
        "blocksize": blocksize,
        "blocks_per_group": blocks_per_group,
        "num_elements": num_elements,
        "num_packed_bytes": int(packed_weights.size),
        "num_blocks": int(num_blocks),
        "num_groups": int(num_groups),
        "absmax2_len": int(absmax2_f16.size),
        "code2_len": int(code2_f16.size),
        "offset": float(offset.item()),
        "save_ref_fp32": bool(save_ref_fp32),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Generated: {bin_path}")
    print(f"Generated: {params_path}")
    if write_dual_params:
        print(f"Generated: {out_dir / f'{stem}_params_bf16.txt'}")
        print(f"Generated: {out_dir / f'{stem}_params_fp16.txt'}")
    if save_ref_fp32:
        print(f"Generated: {ref_path}")
    print(f"Generated: {meta_path}")
    print(f"Device used for data generation: {device}")
    print(
        "Summary: "
        f"elements={num_elements}, packed={packed_weights.size}, "
        f"blocks={num_blocks}, groups={num_groups}, code2={code2_f16.size}"
    )


def _phase5_cases(include_large: bool) -> Iterable[CaseSpec]:
    rows_cols = [(64, 128)]
    if include_large:
        rows_cols.append((4096, 4096))

    for rows, cols in rows_cols:
        for blocksize in (64, 128):
            for blocks_per_group in (128, 256, 512):
                yield CaseSpec(
                    rows=rows,
                    cols=cols,
                    blocksize=blocksize,
                    blocks_per_group=blocks_per_group,
                )


def generate_phase5_suite(
    out_dir: Path,
    seed: int,
    include_large: bool,
    save_ref_fp32: bool,
) -> None:
    for idx, case in enumerate(_phase5_cases(include_large=include_large)):
        # Keep deterministic but unique seeds across cases.
        case_seed = seed + idx * 17
        generate_dataset(
            rows=case.rows,
            cols=case.cols,
            blocksize=case.blocksize,
            blocks_per_group=case.blocks_per_group,
            seed=case_seed,
            compute_type="bf16",
            out_dir=out_dir,
            stem=case.stem,
            save_ref_fp32=save_ref_fp32,
            write_dual_params=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate NF4 double-quant test binaries.")
    parser.add_argument("--rows", type=int, default=64, help="Matrix rows.")
    parser.add_argument("--cols", type=int, default=128, help="Matrix cols.")
    parser.add_argument("--blocksize", type=int, default=64, help="NF4 quant blocksize.")
    parser.add_argument(
        "--blocks-per-group",
        type=int,
        default=256,
        help="Double-quant group size (in blocks).",
    )
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed.")
    parser.add_argument(
        "--compute-type",
        type=str,
        default="bf16",
        choices=("bf16", "fp16"),
        help="Written into params.txt.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tests/data"),
        help="Output directory.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default="nf4",
        help="Output filename stem (single-case mode).",
    )
    parser.add_argument(
        "--skip-ref-fp32",
        action="store_true",
        help="Do not save *_ref_fp32.npy (reduces disk usage).",
    )
    parser.add_argument(
        "--write-dual-params",
        action="store_true",
        help="Also emit *_params_bf16.txt and *_params_fp16.txt.",
    )
    parser.add_argument(
        "--phase5-suite",
        action="store_true",
        help="Generate Phase-5 matrix suite (64x128 + 4096x4096, bs=64/128, bpg=128/256/512).",
    )
    parser.add_argument(
        "--no-large",
        action="store_true",
        help="When --phase5-suite is enabled, skip 4096x4096 cases.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_ref_fp32 = not args.skip_ref_fp32

    if args.phase5_suite:
        generate_phase5_suite(
            out_dir=args.out_dir,
            seed=args.seed,
            include_large=not args.no_large,
            save_ref_fp32=save_ref_fp32,
        )
        return

    generate_dataset(
        rows=args.rows,
        cols=args.cols,
        blocksize=args.blocksize,
        blocks_per_group=args.blocks_per_group,
        seed=args.seed,
        compute_type=args.compute_type,
        out_dir=args.out_dir,
        stem=args.stem,
        save_ref_fp32=save_ref_fp32,
        write_dual_params=args.write_dual_params,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compare CUDA dequant output with a bitsandbytes reference.

Reference construction explicitly uses bitsandbytes.dequantize_blockwise
for the double-quantized absmax path, then applies NF4 table lookup.
"""

from __future__ import annotations

import argparse
import os
import re
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


HEADER_BYTES = 8 + 8 + 4
CODE2_ENTRIES = 256
CODE2_BYTES = CODE2_ENTRIES * 2
OFFSET_BYTES = 4
NF4_STEM_RE = re.compile(r"^nf4_r(\d+)_c(\d+)_bs(\d+)_bpg(\d+)$")


@dataclass(frozen=True)
class Case:
    rows: int
    cols: int
    blocksize: int
    blocks_per_group: int
    compute_type: str
    stem: str


def _load_torch_bnb():
    try:
        import torch
        import bitsandbytes.functional as F
    except Exception as exc:
        raise SystemExit(
            "torch + bitsandbytes are required. Install with: pip install torch bitsandbytes numpy\n"
            f"Import error: {exc}"
        )
    return torch, F


def _default_exe_path(project_root: Path) -> Path:
    win = project_root / "build" / "Release" / "nf4_dequant.exe"
    if win.exists():
        return win
    unix = project_root / "build" / "nf4_dequant"
    return unix


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
        if not line:
            continue
        if "=" not in line:
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


def _read_nf4_binary(path: Path) -> dict[str, Any]:
    blob = path.read_bytes()
    if len(blob) < HEADER_BYTES + CODE2_BYTES + OFFSET_BYTES:
        raise RuntimeError(f"File too small: {path}")

    rows, cols, blocksize = struct.unpack_from("<qqi", blob, 0)
    if rows <= 0 or cols <= 0 or blocksize <= 0:
        raise RuntimeError(f"Invalid header values in {path}")

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
        "num_packed": int(num_packed),
        "num_blocks": int(num_blocks),
        "num_groups": int(num_groups),
        "packed": packed,
        "absmax_q": absmax_q,
        "absmax2": absmax2,
        "code2": code2,
        "offset": float(offset),
    }


def _decode_output(path: Path, num_elements: int, compute_type: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint16, count=num_elements)
    if raw.size != num_elements:
        raise RuntimeError(
            f"Output size mismatch: got {raw.size}, expected {num_elements} ({path})"
        )
    if compute_type == "bf16":
        bits32 = raw.astype(np.uint32) << 16
        return bits32.view(np.float32)
    if compute_type == "fp16":
        return raw.view(np.float16).astype(np.float32)
    raise RuntimeError(f"Unsupported compute_type: {compute_type}")


def _build_bnb_reference(parsed: dict[str, Any], blocks_per_group: int) -> np.ndarray:
    torch, F = _load_torch_bnb()

    absmax_q_t = torch.from_numpy(parsed["absmax_q"]).to(torch.uint8)
    state2 = F.QuantState(
        absmax=torch.from_numpy(parsed["absmax2"].astype(np.float32, copy=False)),
        code=torch.from_numpy(parsed["code2"].astype(np.float32, copy=False)),
        blocksize=blocks_per_group,
        dtype=torch.float32,
    )

    centered_absmax = F.dequantize_blockwise(
        absmax_q_t,
        quant_state=state2,
        blocksize=blocks_per_group,
    )
    absmax = centered_absmax.cpu().numpy().astype(np.float32, copy=False) + np.float32(
        parsed["offset"]
    )

    nf4 = F.get_4bit_type("nf4", device="cpu").cpu().numpy().astype(np.float32)
    packed = parsed["packed"]
    n = parsed["num_elements"]

    nib = np.empty(n, dtype=np.uint8)
    nib[0::2] = packed[: (n + 1) // 2] >> 4
    if n > 1:
        nib[1::2] = (packed[: (n // 2)] & 0x0F)

    block_ids = np.arange(n, dtype=np.int64) // int(parsed["blocksize"])
    scales = absmax[block_ids]
    ref = nf4[nib] * scales
    return ref.astype(np.float32, copy=False)


def _run_exe(exe: Path, weights: Path, params: Path, output: Path, verbose: bool) -> None:
    cmd = [str(exe), str(weights), str(params), str(output)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if verbose:
        if proc.stdout:
            print(proc.stdout.strip())
        if proc.stderr:
            print(proc.stderr.strip(), file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Executable failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}"
        )


def _evaluate_case(
    exe: Path,
    weights: Path,
    params: Path,
    output: Path,
    mae_threshold: float,
    verbose: bool,
) -> tuple[float, float]:
    run_params = _parse_params_file(params)
    parsed = _read_nf4_binary(weights)

    if run_params["blocksize"] is None:
        run_params["blocksize"] = parsed["blocksize"]

    if int(run_params["blocksize"]) != int(parsed["blocksize"]):
        raise RuntimeError(
            f"blocksize mismatch (params={run_params['blocksize']} vs bin={parsed['blocksize']})"
        )

    bpg = int(run_params["blocks_per_group"])
    compute_type = str(run_params["compute_type"]).lower()
    if compute_type not in ("bf16", "fp16"):
        raise RuntimeError(f"Unsupported compute_type in params: {compute_type}")

    _run_exe(exe=exe, weights=weights, params=params, output=output, verbose=verbose)
    out = _decode_output(output, parsed["num_elements"], compute_type=compute_type)
    ref = _build_bnb_reference(parsed, blocks_per_group=bpg)

    mae = float(np.mean(np.abs(out - ref)))
    maxe = float(np.max(np.abs(out - ref)))
    if mae >= mae_threshold:
        raise RuntimeError(
            f"MAE too large: mae={mae:.8f}, threshold={mae_threshold:.8f}, maxe={maxe:.8f}"
        )
    return mae, maxe


def _phase5_suite_cases(include_large: bool) -> list[Case]:
    dims = [(64, 128)]
    if include_large:
        dims.append((4096, 4096))

    out: list[Case] = []
    for rows, cols in dims:
        for blocksize in (64, 128):
            for bpg in (128, 256, 512):
                stem = f"nf4_r{rows}_c{cols}_bs{blocksize}_bpg{bpg}"
                out.append(
                    Case(
                        rows=rows,
                        cols=cols,
                        blocksize=blocksize,
                        blocks_per_group=bpg,
                        compute_type="bf16",
                        stem=stem,
                    )
                )
                out.append(
                    Case(
                        rows=rows,
                        cols=cols,
                        blocksize=blocksize,
                        blocks_per_group=bpg,
                        compute_type="fp16",
                        stem=stem,
                    )
                )
    return out


def _find_base_params(data_dir: Path, stem: str) -> Path:
    candidates = [
        data_dir / f"{stem}_params.txt",
        data_dir / f"{stem}_params_bf16.txt",
        data_dir / f"{stem}_params_fp16.txt",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"No params file found for stem={stem}")


def _suite(
    exe: Path,
    project_root: Path,
    data_dir: Path,
    mae_threshold: float,
    include_large: bool,
    verbose: bool,
    auto_generate_missing: bool,
) -> int:
    cases = _phase5_suite_cases(include_large=include_large)
    passed = 0
    failed = 0

    with tempfile.TemporaryDirectory(prefix="nf4_compare_") as td:
        tmp = Path(td)
        for case in cases:
            weights = data_dir / f"{case.stem}_weights.bin"
            if not weights.exists():
                if auto_generate_missing:
                    try:
                        _generate_case_data(project_root, data_dir, case)
                    except Exception as exc:
                        print(f"[MISSING] {weights} auto-generate failed: {exc}", file=sys.stderr)
                        failed += 1
                        continue
                else:
                    print(f"[MISSING] {weights}", file=sys.stderr)
                    failed += 1
                    continue

            try:
                base_params = _find_base_params(data_dir, case.stem)
                params_dict = _parse_params_file(base_params)
                params_dict["blocksize"] = case.blocksize
                params_dict["compute_type"] = case.compute_type
                params_dict["blocks_per_group"] = case.blocks_per_group

                params_path = tmp / f"{case.stem}_{case.compute_type}.params.txt"
                out_path = tmp / f"{case.stem}_{case.compute_type}.out.bin"
                _write_params_file(params_path, params_dict)

                mae, maxe = _evaluate_case(
                    exe=exe,
                    weights=weights,
                    params=params_path,
                    output=out_path,
                    mae_threshold=mae_threshold,
                    verbose=verbose,
                )
                print(
                    f"[PASS] {case.stem} dtype={case.compute_type} "
                    f"mae={mae:.8f} maxe={maxe:.8f}"
                )
                passed += 1
            except Exception as exc:
                print(
                    f"[FAIL] {case.stem} dtype={case.compute_type} error={exc}",
                    file=sys.stderr,
                )
                failed += 1

    print(f"Suite summary: passed={passed}, failed={failed}, threshold={mae_threshold}")
    return 0 if failed == 0 else 1


def _generate_case_data(project_root: Path, data_dir: Path, case: Case) -> None:
    script = project_root / "tests" / "generate_test_data.py"
    cmd = [
        sys.executable,
        str(script),
        "--rows",
        str(case.rows),
        "--cols",
        str(case.cols),
        "--blocksize",
        str(case.blocksize),
        "--blocks-per-group",
        str(case.blocks_per_group),
        "--out-dir",
        str(data_dir),
        "--stem",
        case.stem,
        "--skip-ref-fp32",
        "--write-dual-params",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    print(f"[GEN ] {case.stem}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare NF4 CUDA output with bitsandbytes.")
    parser.add_argument(
        "--exe",
        type=Path,
        default=None,
        help="Path to nf4_dequant executable. Default: auto-detect build output.",
    )
    parser.add_argument(
        "--mae-threshold",
        type=float,
        default=1e-2,
        help="MAE threshold for pass/fail.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print executable stdout/stderr for each run.",
    )
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Run Phase-5 suite over generated datasets in tests/data.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("tests/data"),
        help="Data directory (for --suite).",
    )
    parser.add_argument(
        "--no-large",
        action="store_true",
        help="When --suite is used, skip 4096x4096 cases.",
    )
    parser.add_argument(
        "--auto-generate-missing",
        action="store_true",
        help="When --suite is used, generate missing datasets automatically.",
    )
    parser.add_argument("--weights-bin", type=Path, help="Single-case weights binary.")
    parser.add_argument("--params", type=Path, help="Single-case params file.")
    parser.add_argument(
        "--out-bin",
        type=Path,
        default=Path("tests/data/_compare_out.bin"),
        help="Single-case output file written by executable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    exe = args.exe or _default_exe_path(project_root)
    exe = exe.resolve()
    if not exe.exists():
        print(f"Executable not found: {exe}", file=sys.stderr)
        return 2

    if args.suite:
        return _suite(
            exe=exe,
            project_root=project_root,
            data_dir=args.data_dir,
            mae_threshold=float(args.mae_threshold),
            include_large=not args.no_large,
            verbose=bool(args.verbose),
            auto_generate_missing=bool(args.auto_generate_missing),
        )

    if args.weights_bin is None or args.params is None:
        print(
            "Single-case mode requires --weights-bin and --params, "
            "or use --suite.",
            file=sys.stderr,
        )
        return 2

    try:
        mae, maxe = _evaluate_case(
            exe=exe,
            weights=args.weights_bin,
            params=args.params,
            output=args.out_bin,
            mae_threshold=float(args.mae_threshold),
            verbose=bool(args.verbose),
        )
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    print(f"[PASS] mae={mae:.8f} maxe={maxe:.8f} threshold={args.mae_threshold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import DEFAULT_LIBRARY, TEST_PATTERNS, VERSION_IDS, HadamardLibrary, run_correctness_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict normalized FWHT version correctness test")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--dim", type=int, nargs="+", default=(128,))
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16", "all"), default="all")
    parser.add_argument("--kernel", choices=("v0", "v1", "v2", "naive", "shared", "optimized", "all"), default="all")
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    args = parser.parse_args()
    if args.batch <= 0 or any(dim < 32 or dim > 8192 or dim & (dim - 1) for dim in args.dim):
        parser.error("batch must be positive and dimensions must be powers of two in [32, 8192]")
    return args


def main() -> int:
    args = parse_args()
    try:
        import torch
        from fast_hadamard_transform import hadamard_transform
    except ImportError as error:
        raise SystemExit("PyTorch and fast_hadamard_transform are required; run ./scripts/bootstrap_env.sh") from error
    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda is unavailable")
    versions = tuple(VERSION_IDS) if args.kernel == "all" else ({"naive": "v0", "shared": "v1", "optimized": "v2"}.get(args.kernel, args.kernel),)
    dtypes = tuple(("fp32", "fp16", "bf16")) if args.dtype == "all" else (args.dtype,)
    library, passed = HadamardLibrary(torch, args.library), True
    print("Contract: y = x @ H(D) / sqrt(D), Sylvester/Walsh order")
    for dtype_name in dtypes:
        for dim in args.dim:
            print(f"\nShape: [{min(args.batch, 4)}, {dim}] dtype: {dtype_name.upper()}")
            for version in versions:
                gate = run_correctness_gate(torch, library, hadamard_transform, version, args.batch, dim, dtype_name, TEST_PATTERNS)
                passed = passed and gate.passed
                print(
                    f"  hadamard_{version}: {'PASS' if gate.passed else 'FAILED'} "
                    f"my/ref={gate.version_vs_reference.max_abs_error:.3e}, "
                    f"dao/ref={gate.dao_vs_reference.max_abs_error:.3e}, "
                    f"my/dao={gate.version_vs_dao.max_abs_error:.3e}"
                )
    print(f"\nOVERALL: {'PASS' if passed else 'FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

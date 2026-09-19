"""Baselines for the Hadamard transform, measured with PyTorch on the same GPU.

Three references are produced for every (rows, dim, dtype) triple:

  cublas      torch.matmul(x, H)  -- the cuBLAS GEMM formulation of the rotation,
              i.e. what QuaRot does in practice when it applies the Hadamard
              matrix as a linear layer.
  ref_fht     Dao-AILab fast_hadamard_transform (the reference implementation
              named by the project brief), if it is importable.
  memcpy      x.clone(): a pure DRAM round trip, the lower bound for any
              out-of-place transform.
  rotate+quant  unfused rotation followed by a torch quantiser (fp8 / int4),
              the pipeline the fused kernel replaces.

The CSV has the same schema as `fhwt --mode matrix` so that tools/analyze.py can
merge both into one table.
"""

import argparse
import csv
import os
import sys
import time

import torch

try:
    from fast_hadamard_transform import hadamard_transform as ref_hadamard
except Exception:  # pragma: no cover - optional dependency
    ref_hadamard = None


def hadamard_matrix(dim, dtype, device):
    """Sylvester Hadamard matrix, rows ordered as the FWHT produces them."""
    h = torch.ones(1, 1, dtype=torch.float64, device=device)
    while h.shape[0] < dim:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h.to(dtype)


def timeit(fn, iters, warmup, reps=3):
    """Best of `reps` bursts of `iters` calls.

    Same estimator as the CUDA driver uses (`fhwt --reps`): this GPU also
    drives the desktop, so a single burst can be perturbed by unrelated GPU
    work.  Both sides of the comparison must be measured the same way.
    """
    best = float("inf")
    for _ in range(max(reps, 1)):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / iters)
    return best


def traffic_gbps(rows, dim, dtype, ms, factor=2.0):
    elem = torch.tensor([], dtype=dtype).element_size()
    return 2.0 * factor / 2.0 * rows * dim * elem / (ms * 1e-3) / 1e9


def quant_fp8(x):
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = amax / 448.0
    q = (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q, scale


def quant_int4(x):
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = amax / 7.0
    q = torch.clamp(torch.round(x / scale), -8, 7).to(torch.int8)
    return q, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/baseline.csv")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dtype_map = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}
    cases = []
    for dim in (64, 128, 256, 512, 1024):
        for rows in (1024, 4096, 16384, 65536, 262144):
            for dt in ("f16", "bf16"):
                cases.append((rows, dim, dt))

    rows_out = []
    print(f"{'rows':>8} {'dim':>5} {'dtype':>5} {'impl':>14} {'ms':>10} {'GB/s':>9}")
    for rows, dim, dt in cases:
        torch_dtype = dtype_map[dt]
        x = torch.randn(rows, dim, device=args.device, dtype=torch_dtype)
        y = torch.empty_like(x)
        h = hadamard_matrix(dim, torch_dtype, args.device)
        scale = dim ** -0.5

        variants = []
        variants.append(("memcpy", lambda: y.copy_(x)))
        variants.append(("cublas", lambda: torch.matmul(x, h * scale, out=y)))
        if ref_hadamard is not None:
            variants.append(("ref_fht", lambda: ref_hadamard(x, scale=scale)))
        variants.append(
            (
                "rotate+quant_fp8",
                lambda: quant_fp8((torch.matmul(x, h * scale))),
            )
        )
        variants.append(
            (
                "rotate+quant_int4",
                lambda: quant_int4((torch.matmul(x, h * scale))),
            )
        )
        for name, fn in variants:
            try:
                ms = timeit(fn, args.iters, args.warmup, args.reps)
            except Exception as exc:  # pragma: no cover
                print(f"  skip {name}: {exc}")
                continue
            gbps = traffic_gbps(rows, dim, torch_dtype, ms)
            print(f"{rows:>8} {dim:>5} {dt:>5} {name:>14} {ms:>10.4f} {gbps:>9.1f}")
            rows_out.append(
                dict(rows=rows, dim=dim, dtype=dt, impl=name, ms=ms, gbps=gbps)
            )

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rows", "dim", "dtype", "impl", "ms", "gbps"])
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()

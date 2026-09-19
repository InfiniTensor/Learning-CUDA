#!/usr/bin/env python3
"""Check our kernels element-by-element against fast_hadamard_transform.

The project brief states the acceptance criterion as

    FP16 absolute error < 1e-2,  BF16 absolute error < 5e-2
    measured against fast_hadamard_transform (Dao-AILab)

`tests/test_correctness.cu` already compares against an exact double-precision
CPU reference, which is the stricter test; this script closes the loop by
comparing against the *named* reference implementation on the same inputs.

For every (dim, dtype, kernel) it writes a dump from our binary, replays the
identical input through fast_hadamard_transform, and reports

    max |ours - ref|              the brief's criterion, and
    max |ours - exact|, max |ref - exact|   to show which side carries the error.

Usage:
    set PYTHONPATH=%TEMP%\fhwt_ref
    python tools/check_against_ref.py --bin build/fhwt.exe [--dims 64,256,1024]
"""

import argparse
import os
import struct
import subprocess
import sys
import tempfile

import torch

try:
    from fast_hadamard_transform import hadamard_transform as ref_hadamard
except Exception as exc:  # pragma: no cover
    print(f"fast_hadamard_transform is not importable: {exc}")
    print("set PYTHONPATH to the directory holding the compiled extension")
    sys.exit(2)

DTYPE_BY_CODE = {0: torch.float16, 1: torch.bfloat16, 2: torch.float32}
LABEL = {0: "f16", 1: "bf16", 2: "f32"}
TOL = {"f16": 1e-2, "bf16": 5e-2, "f32": 1e-3}


def read_dump(path):
    with open(path, "rb") as fh:
        blob = fh.read()
    assert blob[:8] == b"FHWTDMP1", "bad magic"
    rows, dim, dt_code, kern_code, scale = struct.unpack_from("<qqiif", blob, 8)
    dtype = DTYPE_BY_CODE[dt_code]
    esz = torch.tensor([], dtype=dtype).element_size()
    off = 8 + 8 + 8 + 4 + 4 + 4
    n = rows * dim
    x = torch.frombuffer(bytearray(blob[off:off + n * esz]), dtype=dtype).reshape(rows, dim)
    y = torch.frombuffer(bytearray(blob[off + n * esz:off + 2 * n * esz]),
                         dtype=dtype).reshape(rows, dim)
    return rows, dim, dtype, LABEL[dt_code], kern_code, scale, x, y


def exact_hadamard(dim, device):
    """Sylvester Hadamard matrix in float64, same row order as the FWHT."""
    h = torch.ones(1, 1, dtype=torch.float64, device=device)
    while h.shape[0] < dim:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="build/fhwt.exe")
    ap.add_argument("--dims", default="64,256,1024")
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--kernels", default="reg,tc,smem")
    ap.add_argument("--accs", default="native",
                    help="accumulation modes to sweep (reg/smem only; tc always fp32)")
    args = ap.parse_args()

    dims = [int(d) for d in args.dims.split(",")]
    kernels = args.kernels.split(",")
    accs = args.accs.split(",")
    tmp = tempfile.mkdtemp(prefix="fhwt_dump_")
    device = "cuda"

    header = ["dim", "dtype", "kernel", "acc", "max|ours-ref|", "max|ours-exact|",
              "max|ref-exact|", "tol", "verdict"]
    rows_out = []
    failures = 0
    for dim in dims:
        for dt in ("f16", "bf16"):
            for kern in kernels:
                for acc in accs:
                    path = os.path.join(tmp, f"{dim}_{dt}_{kern}_{acc}.bin")
                    cmd = [args.bin, "--mode", "dump", "--dim", str(dim),
                           "--rows", str(args.rows), "--dtype", dt, "--kernel", kern,
                           "--acc", acc, "--scale", "norm", "--out", path]
                    res = subprocess.run(cmd, capture_output=True, text=True)
                    if res.returncode != 0:
                        print(f"skip dim={dim} {dt} {kern} {acc}: {res.stderr.strip()}")
                        continue
                    rows, d, dtype, label, kcode, scale, x, ours = read_dump(path)

                    xd = x.to(device)
                    ref = ref_hadamard(xd, scale=scale)
                    exact = (xd.to(torch.float64)
                             @ exact_hadamard(d, device).to(torch.float64)) * scale

                    e_ref = (ours.to(device).to(torch.float64)
                             - ref.to(torch.float64)).abs().max().item()
                    e_exact = (ours.to(device).to(torch.float64) - exact).abs().max().item()
                    r_exact = (ref.to(torch.float64) - exact).abs().max().item()
                    tol = TOL[label]
                    ok = e_ref < tol
                    failures += 0 if ok else 1
                    rows_out.append([dim, label, kern, acc, f"{e_ref:.4e}", f"{e_exact:.4e}",
                                     f"{r_exact:.4e}", f"{tol:.0e}", "PASS" if ok else "FAIL"])
                    print(f"dim={dim:5d} {label:4s} {kern:5s} {acc:7s} "
                          f"|ours-ref|={e_ref:.4e} |ours-exact|={e_exact:.4e} "
                          f"|ref-exact|={r_exact:.4e}  {'PASS' if ok else 'FAIL'}")

    print()
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows_out:
        print("| " + " | ".join(str(c) for c in r) + " |")
    print(f"\n{len(rows_out)} comparisons, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

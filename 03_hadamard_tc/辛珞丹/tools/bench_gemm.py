#!/usr/bin/env python3
"""Rotate-then-GEMM vs the same GEMM with the rotation folded into its A-tile load.

`--mode gemm` runs both variants through the *same* hand-written GEMM core, so
the only difference is where the Hadamard happens.  That is what makes this a
measurement of the fusion: comparing a hand-written fused kernel against cuBLAS
would mostly measure the quality of the GEMM instead.  (The absolute cuBLAS
number lives in data/baseline.csv.)

The sweep is built around arithmetic intensity.  Fusing the rotation removes
2*M*K*sizeof(T) bytes and nothing else, so it can only pay off while the GEMM is
not compute-bound -- i.e. while N is small next to M.  The shape list walks N up
from 64 to 4096 at fixed M/K to see how the win decays, then varies K and M at
N = 128 (the QuaRot head_dim case).

This box is a desktop, not a lab server, so a single sweep is not reproducible
to better than 10-20% on the small shapes.  The tool therefore repeats the
*whole sweep* `--runs` times and reports the per-shape median, and writes the
min/max of the speedup across those runs into the CSV so the spread is visible
in the table instead of being hidden by one lucky sample.

Writes `data/gemm.csv` (consumed by tools/analyze.py), `data/gemm.log` and
`data/gemm_runs.log` (every raw run).

Usage:
    python tools/bench_gemm.py [--bin build/fhwt.exe] [--out data] [--reps 3] [--runs 3]
"""

import argparse
import os
import re
import subprocess
import sys

# (M, N, K)
SHAPES = [
    # N sweep at M=8192, K=256: the arithmetic-intensity knob.
    (8192, 64, 256),
    (8192, 128, 256),
    (8192, 256, 256),
    (8192, 512, 256),
    (8192, 1024, 256),
    (8192, 4096, 256),
    # K sweep at M=8192, N=128: the QuaRot head_dim range.
    (8192, 128, 128),
    (8192, 128, 512),
    (8192, 128, 1024),
    # M sweep at N=128, K=256.
    (1024, 128, 256),
    (32768, 128, 256),
    (65536, 128, 256),
    # A realistic skinny shape: an output projection for one long sequence.
    (16384, 256, 512),
    # And a fat one, where the GEMM is expected to be compute-bound.
    (4096, 4096, 256),
]

NUM = r"([\d.]+)"
# The trailing "x vs A" belongs to B's line only; it must not capture, or the
# findall() tuples grow a fourth empty element and the unpack below breaks.
PC_RE = re.compile(r"per call\s+:\s+" + NUM + r" ms\s+" + NUM + r" GFLOP/s\s+" + NUM +
                   r" GB/s(?:\s+[\d.]+x vs A)?")
SAVED_RE = re.compile(r"rotation saves\s+:\s+" + NUM + r" MiB per call \(" + NUM + r"%\)")
SAME_RE = re.compile(r"bit-for-bit:\s*(yes|NO)")

HEADER = ("m,n,k,dtype,unfused_ms,fused_ms,speedup,gflops_unfused,gflops_fused,"
          "unfused_gbs,fused_gbs,saved_mib,saved_pct,bitwise,speedup_min,speedup_max,runs")


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def run(binary, m, n, k, dtype, reps):
    cmd = [binary, "--mode", "gemm", "--rows", str(m), "--n", str(n), "--dim", str(k),
           "--dtype", dtype, "--iters", "20", "--warmup", "5", "--reps", str(reps)]
    res = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if res.returncode != 0:
        return None
    txt = res.stdout
    pc = PC_RE.findall(txt)
    saved = SAVED_RE.search(txt)
    same = SAME_RE.search(txt)
    if len(pc) != 2 or not saved or not same:
        return None
    return pc, float(saved.group(1)), float(saved.group(2)), same.group(1) == "yes"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join("build", "fhwt.exe"))
    ap.add_argument("--out", default="data")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--dtype", default="f16")
    args = ap.parse_args()
    if not os.path.exists(args.bin):
        raise SystemExit(f"binary not found: {args.bin}")

    os.makedirs(args.out, exist_ok=True)
    csv_lines = [HEADER]
    log_lines = []
    raw_lines = []
    for m, n, k in SHAPES:
        samples = []
        for r in range(args.runs):
            got = run(args.bin, m, n, k, args.dtype, args.reps)
            if got is None:
                print(f"skip M={m} N={n} K={k}: run failed", file=sys.stderr)
                break
            pc, saved_mib, saved_pct, bitwise = got
            a_ms, a_gf, a_gbs = (float(x) for x in pc[0])
            b_ms, b_gf, b_gbs = (float(x) for x in pc[1])
            samples.append((a_ms, b_ms, a_gf, b_gf, a_gbs, b_gbs, saved_mib, saved_pct,
                            bitwise, a_ms / b_ms))
            raw_lines.append("M=%-6d N=%-5d K=%-5d run%d | A %8.4f ms | B %8.4f ms | %5.2fx"
                             % (m, n, k, r, a_ms, b_ms, a_ms / b_ms))
        if len(samples) != args.runs:
            continue
        a_ms = median([s[0] for s in samples])
        b_ms = median([s[1] for s in samples])
        a_gf = median([s[2] for s in samples])
        b_gf = median([s[3] for s in samples])
        a_gbs = median([s[4] for s in samples])
        b_gbs = median([s[5] for s in samples])
        saved_mib = median([s[6] for s in samples])
        saved_pct = median([s[7] for s in samples])
        bitwise = all(s[8] for s in samples)
        ratios = [s[9] for s in samples]
        speed = a_ms / b_ms
        line = ("M=%-6d N=%-5d K=%-5d | unfused %8.4f ms %7.0f GFLOP/s | fused %8.4f ms "
                "%7.0f GFLOP/s | %5.2fx [%4.2f-%4.2f over %d runs] | saved %5.1f MiB (%4.1f%%) | bitwise=%s"
                % (m, n, k, a_ms, a_gf, b_ms, b_gf, speed, min(ratios), max(ratios),
                   args.runs, saved_mib, saved_pct, "yes" if bitwise else "NO"))
        log_lines.append(line)
        print(line)
        csv_lines.append("%d,%d,%d,%s,%.6g,%.6g,%.4f,%.1f,%.1f,%.1f,%.1f,%.2f,%.2f,%d,%.4f,%.4f,%d"
                         % (m, n, k, args.dtype, a_ms, b_ms, speed, a_gf, b_gf, a_gbs, b_gbs,
                            saved_mib, saved_pct, 1 if bitwise else 0, min(ratios), max(ratios),
                            args.runs))

    for name, lines in (("gemm.csv", csv_lines), ("gemm.log", log_lines),
                        ("gemm_runs.log", raw_lines)):
        with open(os.path.join(args.out, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.join(args.out, 'gemm.csv')}")


if __name__ == "__main__":
    main()
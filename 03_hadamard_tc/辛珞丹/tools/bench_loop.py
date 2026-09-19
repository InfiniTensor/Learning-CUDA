#!/usr/bin/env python3
"""Plain submission vs. graphs: does the *number of launches* matter?

`--mode loop` models a serving pass (L layers, one rotation each) and measures
four ways of submitting it:

    A  one cudaLaunchKernel per transform
    B  one single-step graph replayed per transform
    C  ONE graph holding the whole pass, replayed once, layers chained
    D  ONE graph, same nodes, but the layers forked (GraphSequence::add_parallel)

B is the trap: swapping the launch API without changing how many submissions
happen buys almost nothing (measured ~1.0x, and 0.86x in one run).  C is the
real optimisation, and its capture cost is amortised after a couple of passes.

D asks a second question: once the submissions are off the critical path, does
it still matter that the graph *chains* the layers?  A layer here is a kernel
whose grid fills only part of the machine, and every layer owns its own buffers,
so the chain is an artefact of what a stream capture can express rather than a
real dependency.

D is therefore measured twice, because the answer depends on how many *roots*
the recorded graph has and that is easy to get wrong:

    D        every layer queued with add_parallel(): 32 nodes, 32 roots
    D-sr     layer 0 queued with add() first (--serial-entry): 32 nodes, 1 root

Measured: D is 0.24-0.26x of A, i.e. four times *slower* than the linear chain,
while D-sr is 1.07-1.12x of A, i.e. slightly faster than the chain.  Both
numbers are stable over five sweeps.  The cost is host-side (nsys: the
cudaGraphLaunch of a multi-root graph degrades from ~6 us to 44-500 us as the
same graph is replayed), which is why it does not show up in kernel durations.
See report sections 5.8 / 7.11 / 8.6.  Kept in the table: the negative result is
what identifies the cause.

A and B measure the host submission rate, which a desktop background load
perturbs by several x, so the whole sweep is repeated `--runs` times and the
per-config median is reported along with the min/max of the speedups.

Writes `data/loop.csv` (consumed by tools/analyze.py), `data/loop.log` and
`data/loop_runs.log`.

Usage:
    python tools/bench_loop.py [--bin build/fhwt.exe] [--out data] [--reps 3] [--runs 3]
"""

import argparse
import os
import re
import subprocess
import sys

# (rows, dims, layers, passes, quant, dtype)
CONFIGS = [
    # Working set climbs from L2-resident (rows 512/1024) to far past L2
    # (rows 65536), which is what the C-vs-A gain actually tracks.
    (512, "64,128,256", 32, 20, "none", "f16"),
    (1024, "64,128,256", 32, 20, "none", "f16"),
    (2048, "64,128,256", 32, 10, "none", "f16"),
    (2048, "128,256", 32, 10, "fp8", "f16"),
    (2048, "128,256", 32, 10, "int4", "f16"),
    (4096, "128", 32, 10, "none", "f16"),
    (16384, "256", 32, 10, "none", "f16"),
    (65536, "256", 8, 10, "none", "f16"),
]

NUM = r"([\d.]+)"
ROW_RE = re.compile(
    r"per pass\s+:\s+" + NUM + r" ms.*?per transform\s+" + NUM + r" us\s+" + NUM + r" GB/s"
    + r"(?:\s+[\d.]+x vs A)?")
OK_RE = re.compile(r"bit-identical to plain")
CAP_RE = re.compile(r"capture cost\s+:\s+B\s+" + NUM + r" ms .*?\| C\s+" + NUM + r" ms")
CAPD_RE = re.compile(r"capture cost\s+:\s+B\s+[\d.]+ ms .*?\| C\s+([\d.]+) ms \| D\s+([\d.]+) ms")

HEADER = ("rows,dims,layers,passes,quant,dtype,plain_ms,per_step_ms,sequence_ms,parallel_ms,"
          "plain_us,per_step_us,sequence_us,parallel_us,speedup_b,speedup_c,speedup_d,"
          "capture_b_ms,capture_c_ms,capture_d_ms,runs,speedup_c_min,speedup_c_max,"
          "speedup_d_min,speedup_d_max,"
          "parallel_sr_us,speedup_d_sr,capture_d_sr_ms,speedup_d_sr_min,speedup_d_sr_max")


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def run(binary, rows, dims, layers, passes, quant, dtype, reps, extra=()):
    cmd = [binary, "--mode", "loop", "--rows", str(rows), "--dims", dims,
           "--layers", str(layers), "--iters", str(passes), "--dtype", dtype,
           "--quant", quant, "--reps", str(reps)] + list(extra)
    res = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if res.returncode != 0:
        return None
    txt = res.stdout
    per_pass = [float(m.group(1)) for m in ROW_RE.finditer(txt)]
    cap = CAP_RE.search(txt)
    capd = CAPD_RE.search(txt)
    if len(per_pass) != 4 or not OK_RE.search(txt) or not cap or not capd:
        return None
    return per_pass, float(cap.group(1)), float(cap.group(2)), float(capd.group(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join("build", "fhwt.exe"))
    ap.add_argument("--out", default="data")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    if not os.path.exists(args.bin):
        raise SystemExit(f"binary not found: {args.bin}")

    os.makedirs(args.out, exist_ok=True)
    csv_lines = [HEADER]
    log_lines = []
    raw_lines = []
    for rows, dims, layers, passes, quant, dtype in CONFIGS:
        samples = []
        for r in range(args.runs):
            got = run(args.bin, rows, dims, layers, passes, quant, dtype, args.reps)
            if got is None:
                print(f"skip rows={rows} dims={dims} quant={quant}: run failed", file=sys.stderr)
                break
            ms, cap_b, cap_c, cap_d = got
            calls = float(layers)
            us = [m * 1e3 / calls for m in ms]
            samples.append((ms, cap_b, cap_c, cap_d, us, ms[0] / ms[1], ms[0] / ms[2],
                            ms[0] / ms[3]))
            raw_lines.append("rows=%-7d dims=%-11s quant=%-5s run%d | A %8.3f us B %8.3f us "
                             "(%4.2fx) C %8.3f us (%5.2fx) D %8.3f us (%5.2fx)"
                             % (rows, dims, quant, r, us[0], us[1], ms[0] / ms[1], us[2],
                                ms[0] / ms[2], us[3], ms[0] / ms[3]))
        if len(samples) != args.runs:
            continue
        ms = [median([s[0][i] for s in samples]) for i in range(4)]
        cap_b = median([s[1] for s in samples])
        cap_c = median([s[2] for s in samples])
        cap_d = median([s[3] for s in samples])
        calls = float(layers)
        us = [m * 1e3 / calls for m in ms]
        speed_b, speed_c, speed_d = ms[0] / ms[1], ms[0] / ms[2], ms[0] / ms[3]
        sc = [s[6] for s in samples]
        sd = [s[7] for s in samples]
        line = ("rows=%-7d dims=%-11s layers=%-3d quant=%-5s | A %8.3f us  B %8.3f us (%4.2fx)"
                "  C %8.3f us (%6.2fx [%5.2f-%5.2f])  D %8.3f us (%5.2fx [%4.2f-%4.2f])"
                % (rows, dims, layers, quant, us[0], us[1], speed_b, us[2], speed_c, min(sc),
                   max(sc), us[3], speed_d, min(sd), max(sd)))
        log_lines.append(line)
        print(line)

        # Second sweep: identical pass, but layer 0 is queued with add() instead
        # of add_parallel(), so the forked graph has one root instead of L.  The
        # flag cannot affect A / B / C (they never call add_parallel()), so only
        # D is taken from this sweep.
        sr = []
        for r in range(args.runs):
            got = run(args.bin, rows, dims, layers, passes, quant, dtype, args.reps,
                      extra=("--serial-entry",))
            if got is None:
                break
            sr.append(got)
            raw_lines.append("rows=%-7d dims=%-11s quant=%-5s run%d | single root: D %8.3f us "
                             "(%5.2fx)" % (rows, dims, quant, r, got[0][3] * 1e3 / calls,
                                           got[0][0] / got[0][3]))
        if len(sr) == args.runs:
            ms_sr = median([g[0][3] for g in sr])
            cap_d_sr = median([g[3] for g in sr])
            us_sr = ms_sr * 1e3 / calls
            speed_d_sr = ms[0] / ms_sr
            sd_sr = [g[0][0] / g[0][3] for g in sr]
            line = ("rows=%-7d dims=%-11s layers=%-3d quant=%-5s | D single-root %8.3f us "
                    "(%5.2fx [%4.2f-%4.2f])"
                    % (rows, dims, layers, quant, us_sr, speed_d_sr, min(sd_sr), max(sd_sr)))
            log_lines.append(line)
            print(line)
        else:
            print(f"skip single-root rows={rows} dims={dims} quant={quant}: run failed",
                  file=sys.stderr)
            us_sr = speed_d_sr = cap_d_sr = 0.0
            sd_sr = [0.0, 0.0]
        csv_lines.append("%d,\"%s\",%d,%d,%s,%s,%.6g,%.6g,%.6g,%.6g,%.4f,%.4f,%.4f,%.4f,"
                         "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%.4f,%.4f,%.4f,%.4f,"
                         "%.4f,%.4f,%.4f,%.4f,%.4f"
                         % (rows, dims, layers, passes, quant, dtype, ms[0], ms[1], ms[2], ms[3],
                            us[0], us[1], us[2], us[3], speed_b, speed_c, speed_d, cap_b, cap_c,
                            cap_d, args.runs, min(sc), max(sc), min(sd), max(sd),
                            us_sr, speed_d_sr, cap_d_sr, min(sd_sr), max(sd_sr)))

    for name, lines in (("loop.csv", csv_lines), ("loop.log", log_lines),
                        ("loop_runs.log", raw_lines)):
        with open(os.path.join(args.out, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.join(args.out, 'loop.csv')}")


if __name__ == "__main__":
    main()
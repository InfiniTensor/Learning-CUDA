#!/usr/bin/env python3
"""How many *roots* does the recorded graph have?  That is what costs.

`--mode loop`'s D strategy records one graph holding the whole pass with the
layers forked.  Whether that graph is fast or four times slower than the linear
chain turns out to depend on a single thing: how many of its nodes are roots.
Every layer queued with add_parallel() and nothing serial before it is a root.
The measurement was done in two places, this tool is the one that isolates it:

  * `--layers` is swept 1..32 to show the cost is not proportional to the node
    count (1 node: no cost at all; 2 nodes: the full cost);
  * every configuration is measured twice, once as-is (all roots) and once with
    `--serial-entry`, which queues layer 0 with add() so the fork has a single
    entry point (1 root + 31 forks).

Correctness is unaffected by the flag -- the steps are independent either way --
and the mode checks that every strategy is bit-identical to plain submission.

Both columns are `best of --reps passes` inside one process, and that number is
itself noisy -- the all-roots penalty is stable, the last few percent are not --
so every configuration is measured `--runs` times and the median is reported.

Writes `data/roots.csv` and `data/roots.log`.

Usage:
    python tools/bench_roots.py [--bin build/fhwt.exe] [--out data] [--reps 3] [--runs 5]
"""

import argparse
import os
import re
import subprocess
import sys

# (rows, dims, layers, passes, quant, dtype)
CONFIGS = [(512, "64,128,256", n, 20, "none", "f16")
           for n in (1, 2, 4, 8, 16, 32)]

NUM = r"([\d.]+)"
ROW_RE = re.compile(r"per pass\s+:\s+" + NUM + r" ms")
SHAPE_RE = re.compile(r"D: 1 forked graph launch per pass \((\d+) nodes, (\d+) forked, "
                      r"(\S+) (\S+)(?:, ([^)]*))?\)")
OK_RE = re.compile(r"bit-identical to plain")
HEADER = ("rows,dims,layers,passes,quant,dtype_nodes,nodes,forks,node_kind,plain_ms,"
          "per_step_ms,sequence_ms,parallel_ms,parallel_sr_ms,speedup_d,speedup_d_sr,"
          "sequence_over_parallel_sr")


def run(binary, rows, dims, layers, passes, quant, dtype, reps, extra=()):
    cmd = [binary, "--mode", "loop", "--rows", str(rows), "--dims", dims,
           "--layers", str(layers), "--iters", str(passes), "--dtype", dtype,
           "--quant", quant, "--reps", str(reps)] + list(extra)
    res = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if res.returncode != 0:
        return None
    txt = res.stdout
    ms = [float(m.group(1)) for m in ROW_RE.finditer(txt)]
    shape = SHAPE_RE.search(txt)
    if len(ms) != 4 or not shape or not OK_RE.search(txt):
        return None
    return ms, shape


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join("build", "fhwt.exe"))
    ap.add_argument("--out", default="data")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()
    if not os.path.exists(args.bin):
        raise SystemExit(f"binary not found: {args.bin}")

    os.makedirs(args.out, exist_ok=True)
    csv_lines = [HEADER]
    log_lines = []
    for rows, dims, layers, passes, quant, dtype in CONFIGS:
        # The two variants are interleaved (all-roots, single-root, all-roots, ...)
        # so a drift in the machine hits both equally instead of one column.
        all_ms, sr_ms, shape = [], [], None
        for _ in range(args.runs):
            got = run(args.bin, rows, dims, layers, passes, quant, dtype, args.reps)
            sr = run(args.bin, rows, dims, layers, passes, quant, dtype, args.reps,
                     extra=("--serial-entry",))
            if got is None or sr is None:
                break
            all_ms.append(got[0])
            sr_ms.append(sr[0])
            shape = got[1]
        if len(all_ms) != args.runs:
            print(f"skip layers={layers}: run failed", file=sys.stderr)
            continue
        ms = [median([m[i] for m in all_ms]) for i in range(4)]
        ms_sr = [median([m[i] for m in sr_ms]) for i in range(4)]
        nodes, forks = int(shape.group(1)), int(shape.group(2))
        kind = shape.group(3) + " " + shape.group(4)
        line = ("layers=%-3d %-10s | nodes %2d forks %2d  A %8.3f us  B %8.3f us  "
                "C %8.3f us | D %9.3f us (%.2fx vs C)  D single-root %8.3f us (%.2fx vs C)"
                % (layers, kind, nodes, forks, ms[0] * 1e3 / layers, ms[1] * 1e3 / layers,
                   ms[2] * 1e3 / layers, ms[3] * 1e3 / layers, ms[2] / ms[3],
                   ms_sr[3] * 1e3 / layers, ms[2] / ms_sr[3]))
        log_lines.append(line)
        print(line)
        csv_lines.append("%d,\"%s\",%d,%d,%s,%s,%d,%d,\"%s\",%.6g,%.6g,%.6g,%.6g,%.6g,"
                         "%.4f,%.4f,%.4f"
                         % (rows, dims, layers, passes, quant, dtype, nodes, forks, kind,
                            ms[0], ms[1], ms[2], ms[3], ms_sr[3], ms[2] / ms[3],
                            ms[2] / ms_sr[3], ms[2] / ms_sr[3]))

    for name, lines in (("roots.csv", csv_lines), ("roots.log", log_lines)):
        with open(os.path.join(args.out, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.join(args.out, 'roots.csv')}")


if __name__ == "__main__":
    main()

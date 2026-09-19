#!/usr/bin/env python3
"""How many passes should one graph hold?

7.13 measured a submission cost that grows with the node count: on this driver a
1-node graph costs ~5.2 us to launch and a 48-node one 39-51 us.  A per-layer
fork/join pass of 48 nodes therefore spends about as long being submitted as it
spends running (25 us on the GPU), which caps the win.  The lever left is not
the shape of the graph but the number of submissions: with K passes in one graph,
one `cudaGraphLaunch` covers K x 48 nodes.

`--passes-per-graph K` repeats the whole node list K times inside the graph.  The
K passes perform the identical computation on the same buffers, so the
concatenation is bit-for-bit the same work; what changes is only how many
submissions it takes.

The scan interleaves the K values round-robin (like bench_roots.py) so that a
drift of the host/GPU during the sweep lands on every column instead of on one,
and reports the median of `--runs` rounds.

Writes `data/batch.csv` and `data/batch.log`.

Usage:
    python tools/bench_batch.py [--bin build/fhwt.exe] [--out data] [--reps 3] [--runs 3]
"""

import argparse
import os
import re
import subprocess
import sys

# (rows, dims, layers, tensors, iters, quant, dtype)
CONFIGS = [
    (512, "128", 8, 6, 20, "none", "f16"),           # the 7.13/7.14 nsys shape (25.0 us span)
    (512, "64,128,256", 8, 6, 20, "none", "f16"),    # the 7.13 peak: mixed dims, L2 resident
    (512, "64,128,256", 8, 1, 20, "none", "f16"),    # chain only: no width to win with
    (4096, "128", 8, 6, 20, "none", "f16"),          # working set over L2
    (16384, "256", 8, 3, 20, "none", "f16"),         # DRAM bound: expect nothing
]

# Each value must divide the pass count, otherwise the window's tail is dropped.
PPG = [1, 2, 5, 10, 20]

NUM = r"([\d.]+)"
ROW_RE = re.compile(r"per pass\s+:\s+" + NUM + r" ms")
NODE_RE = re.compile(r"D: 1 (?:forked )?graph launch per (?:pass|\d+ passes) \((\d+) nodes")
CAP_RE = re.compile(r"capture cost\s+:\s+B\s+" + NUM + r" ms.*?\|\s*C\s+" + NUM +
                    r" ms.*?\|\s*D\s+" + NUM + r" ms")
OK_RE = re.compile(r"bit-identical to plain")

HEADER = ("rows,dims,layers,tensors,passes,passes_per_graph,nodes,plain_us,per_step_us,"
          "sequence_us,groups_us,speedup_seq_vs_plain,speedup_groups_vs_plain,"
          "speedup_groups_vs_sequence,capture_seq_ms,capture_par_ms,runs")


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def run(binary, rows, dims, layers, tensors, iters, quant, dtype, reps, ppg):
    cmd = [binary, "--mode", "loop", "--rows", str(rows), "--dims", dims,
           "--layers", str(layers), "--tensors", str(tensors), "--iters", str(iters),
           "--dtype", dtype, "--quant", quant, "--reps", str(reps),
           "--group-entry", "--passes-per-graph", str(ppg)]
    res = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if res.returncode != 0:
        return None
    txt = res.stdout
    ms = [float(m.group(1)) for m in ROW_RE.finditer(txt)]
    nm = NODE_RE.search(txt)
    cap = CAP_RE.search(txt)
    if len(ms) < 4 or not nm or not cap or not OK_RE.search(txt):
        return None
    return (ms, int(nm.group(1)), float(cap.group(2)), float(cap.group(3)))


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
    for rows, dims, layers, tensors, iters, quant, dtype in CONFIGS:
        slots = layers * tensors
        # round-robin over K so host/GPU drift lands on every column
        acc = {k: [] for k in PPG}
        meta = {}
        for _ in range(args.runs):
            for k in PPG:
                if iters % k:
                    continue
                r = run(args.bin, rows, dims, layers, tensors, iters, quant, dtype,
                        args.reps, k)
                if r is not None:
                    acc[k].append(r)
        for k in PPG:
            if iters % k or not acc[k]:
                continue
            ms = [median([x[0][i] for x in acc[k]]) for i in range(4)]
            meta[k] = (ms, acc[k][0][1], median([x[2] for x in acc[k]]),
                       median([x[3] for x in acc[k]]))
        if not meta:
            print(f"skip rows={rows} tensors={tensors}: every run failed", file=sys.stderr)
            continue
        us = lambda m: m * 1e3 / slots
        # meta[k][0][3] is D's *per-pass* time in ms; us() turns it into us/transform.
        base = us(meta[1][0][3]) if 1 in meta else None
        base_c = us(meta[1][0][2]) if 1 in meta else None
        for k in PPG:
            if k not in meta:
                continue
            ms, nodes, cap_c, cap_d = meta[k]
            csv_lines.append("%d,\"%s\",%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                             "%.4f,%.4f,%d"
                             % (rows, dims, layers, tensors, iters, k, nodes, us(ms[0]),
                                us(ms[1]), us(ms[2]), us(ms[3]), ms[0] / ms[2], ms[0] / ms[3],
                                ms[2] / ms[3], cap_c, cap_d, args.runs))
            rel = (" (%5.2fx vs K=1)" % (base / us(ms[3])) if base else "")
            relc = (" (%5.2fx vs K=1)" % (base_c / us(ms[2])) if base_c else "")
            line = ("rows=%-6d %-11s tensors=%-3d K=%-3d | nodes %4d | A %8.3f "
                    "C %7.3f%s D %7.3f%s us | D vs C %5.2fx | capture C %7.3f D %7.3f ms"
                    % (rows, dims, tensors, k, nodes, us(ms[0]), us(ms[2]), relc,
                       us(ms[3]), rel, ms[2] / ms[3], cap_c, cap_d))
            log_lines.append(line)
            print(line)

    for name, lines in (("batch.csv", csv_lines), ("batch.log", log_lines)):
        with open(os.path.join(args.out, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.join(args.out, 'batch.csv')}")


if __name__ == "__main__":
    main()

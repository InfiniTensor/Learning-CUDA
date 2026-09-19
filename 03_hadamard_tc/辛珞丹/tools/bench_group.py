#!/usr/bin/env python3
"""Fork/join *width* inside a layer, and the multi-stream control.

7.11 / 5.8 established two things about a captured pass: one submission is what
pays, and a graph whose nodes are all roots pays a large host-side replay tax.
Both results were measured on the shape a serving loop actually has -- one
rotation per layer, and nothing that orders the layers -- so the forked graph was
node-for-node as wide as it was, not a fork.  This tool measures the shape that
report 10.1 used to leave open:

  * every layer rotates `tensors` independent [rows, dim] activations (Q / K / V
    in a real attention block), and
  * the first tensor of each layer is queued with add(), the rest with
    add_parallel(), so each layer is a fork that the *next layer's* anchor joins.

That keeps the whole graph at a single root (5.8's fix) while giving the
scheduler something to overlap inside a layer.  Three D shapes are compared on
the same node list:

    D all roots     no anchor anywhere                 (the 7.11 default)
    D groups        first tensor of each layer is an anchor   <- the new shape
    E N streams     no graph at all, launches round-robined over N streams

E is the control.  Streams let work run concurrently but do not reduce the number
of submissions, so if the gain came from "the layers are independent" E would be
fast; if it comes from "one submission for the whole pass" E stays near A.

Writes `data/group.csv` and `data/group.log`.

Usage:
    python tools/bench_group.py [--bin build/fhwt.exe] [--out data] [--reps 3] [--runs 3]
"""

import argparse
import os
import re
import subprocess
import sys

# (rows, dims, layers, tensors, passes, quant, dtype)
CONFIGS = [
    (512, "64,128,256", 8, 1, 20, "none", "f16"),
    (512, "64,128,256", 8, 2, 20, "none", "f16"),
    (512, "64,128,256", 8, 3, 20, "none", "f16"),
    (512, "64,128,256", 8, 6, 20, "none", "f16"),
    (512, "64,128,256", 8, 12, 20, "none", "f16"),
    (4096, "128", 8, 1, 10, "none", "f16"),
    (4096, "128", 8, 3, 10, "none", "f16"),
    (4096, "128", 8, 6, 10, "none", "f16"),
    (16384, "256", 8, 1, 10, "none", "f16"),
    (16384, "256", 8, 3, 10, "none", "f16"),
]

NUM = r"([\d.]+)"
ROW_RE = re.compile(r"per pass\s+:\s+" + NUM + r" ms")
D_RE = re.compile(r"D: 1 (?:forked )?graph launch per pass \((\d+) nodes, (\d+) forked")
SR_RE = re.compile(r"D: 1 (?:forked )?graph launch per pass \(.*?, kernel nodes, ([^)]*)\)")
OK_RE = re.compile(r"bit-identical to plain")
OKE_RE = re.compile(r"correctness \(E\)\s+: (\d+)/(\d+) bit-identical")

HEADER = ("rows,dims,layers,tensors,passes,quant,dtype,nodes,forks,plain_us,sequence_us,"
          "all_roots_us,groups_us,speedup_groups_vs_plain,speedup_groups_vs_sequence,"
          "streams,multi_us,speedup_multi_vs_sequence,multi_vs_groups,runs")


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def run(binary, rows, dims, layers, tensors, passes, quant, dtype, reps, extra=()):
    cmd = [binary, "--mode", "loop", "--rows", str(rows), "--dims", dims,
           "--layers", str(layers), "--tensors", str(tensors), "--iters", str(passes),
           "--dtype", dtype, "--quant", quant, "--reps", str(reps)] + list(extra)
    res = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if res.returncode != 0:
        return None
    txt = res.stdout
    ms = [float(m.group(1)) for m in ROW_RE.finditer(txt)]
    d = D_RE.search(txt)
    if len(ms) < 4 or not d or not OK_RE.search(txt):
        return None
    ok_e = OKE_RE.search(txt)
    if ok_e and ok_e.group(1) != ok_e.group(2):
        return None
    return ms, int(d.group(1)), int(d.group(2))


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
    for rows, dims, layers, tensors, passes, quant, dtype in CONFIGS:
        slots = layers * tensors
        flat = {"ms": [], "nodes": 0}
        grp = {"ms": [], "nodes": 0, "forks": 0, "e": []}
        for _ in range(args.runs):
            a = run(args.bin, rows, dims, layers, tensors, passes, quant, dtype, args.reps)
            extra = ("--group-entry",)
            if tensors > 1:
                extra += ("--streams", str(tensors))
            b = run(args.bin, rows, dims, layers, tensors, passes, quant, dtype, args.reps,
                    extra=extra)
            if a is None or b is None:
                break
            flat["ms"].append(a[0])
            flat["nodes"] = a[1]
            grp["ms"].append(b[0])
            grp["nodes"], grp["forks"] = b[1], b[2]
            grp["e"].append(b[0][4] if len(b[0]) > 4 else 0.0)
        if len(flat["ms"]) != args.runs or len(grp["ms"]) != args.runs:
            print(f"skip rows={rows} tensors={tensors}: run failed", file=sys.stderr)
            continue
        fm = [median([m[i] for m in flat["ms"]]) for i in range(4)]
        gm = [median([m[i] for m in grp["ms"]]) for i in range(4)]
        me = median(grp["e"]) if grp["e"] and grp["e"][0] > 0 else 0.0
        us = lambda ms: ms * 1e3 / slots
        line = ("rows=%-6d %-11s tensors=%-3d | nodes %2d  A %8.3f us  C %7.3f us | "
                "D all-roots %8.3f us (%5.2fx vs C)  D groups %7.3f us (%5.2fx vs C)  "
                "E %2d streams %8.3f us (%5.2fx vs C)"
                % (rows, dims, tensors, grp["nodes"], us(fm[0]), us(fm[2]), us(fm[3]),
                   fm[2] / fm[3], us(gm[3]), fm[2] / gm[3],
                   tensors if me else 0, us(me) if me else 0.0, fm[2] / me if me else 0.0))
        log_lines.append(line)
        print(line)
        csv_lines.append("%d,\"%s\",%d,%d,%d,%s,%s,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                         "%d,%.4f,%.4f,%.4f,%d"
                         % (rows, dims, layers, tensors, passes, quant, dtype, grp["nodes"],
                            grp["forks"], us(fm[0]), us(fm[2]), us(fm[3]), us(gm[3]),
                            fm[0] / gm[3], fm[2] / gm[3], tensors, us(me) if me else 0.0,
                            fm[2] / me if me else 0.0, gm[3] / me if me else 0.0, args.runs))

    for name, lines in (("group.csv", csv_lines), ("group.log", log_lines)):
        with open(os.path.join(args.out, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.join(args.out, 'group.csv')}")


if __name__ == "__main__":
    main()

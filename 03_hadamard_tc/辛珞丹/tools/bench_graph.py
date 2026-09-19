#!/usr/bin/env python3
"""Raw cudaLaunchKernel vs. a captured CUDA Graph, on equal-duration bursts.

For every shape the burst length (`--iters`) is chosen so that one timed burst
lasts roughly 20-40 ms.  That keeps the comparison fair on a machine whose GPU
also drives the desktop: a longer burst accumulates more unrelated GPU work,
which would otherwise look like a slowdown of whichever shape happens to need
more iterations.

Writes `data/graph.csv` (consumed by `tools/analyze.py`, section 8 of the
report tables) and `data/graph.log`.

Usage:
    python tools/bench_graph.py [--bin build/fhwt.exe] [--out data] [--tries 3]
"""

import argparse
import os
import subprocess

# (rows, dim, iters) -- iters picked per shape for an equal-duration burst.
SHAPES = [(1024, 64, 1000), (4096, 128, 1000), (16384, 128, 2000), (16384, 256, 2000),
          (65536, 256, 150), (131072, 256, 150), (262144, 256, 50)]

HEADER = "rows,dim,dtype,iters,raw_ms,raw_gbps,graph_ms,graph_gbps,speedup"


def measure(binary, rows, dim, iters, graph):
    """(ms_per_iteration, GB/s) of one burst, or None if the run failed."""
    cmd = [binary, "--mode", "bench", "--kernel", "reg", "--dim", str(dim),
           "--rows", str(rows), "--dtype", "f16", "--iters", str(iters),
           "--warmup", "30", "--reps", "2"]
    if graph:
        cmd.append("--graph")
    res = subprocess.run(cmd, capture_output=True, text=True)
    lines = [ln for ln in res.stdout.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    fields = lines[-1].split()
    try:
        return float(fields[-3]), float(fields[-2])
    except (IndexError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join("build", "fhwt.exe"))
    ap.add_argument("--out", default="data")
    ap.add_argument("--tries", type=int, default=3,
                    help="invocations per point; the fastest is kept")
    args = ap.parse_args()
    if not os.path.exists(args.bin):
        raise SystemExit(f"binary not found: {args.bin}")

    os.makedirs(args.out, exist_ok=True)
    csv_lines = [HEADER]
    log_lines = []
    for rows, dim, iters in SHAPES:
        best = {"raw": None, "graph": None}
        for _ in range(max(1, args.tries)):
            for mode in ("raw", "graph"):
                got = measure(args.bin, rows, dim, iters, mode == "graph")
                if got and (best[mode] is None or got[1] > best[mode][1]):
                    best[mode] = got
        if not best["raw"] or not best["graph"]:
            print(f"skip rows={rows} dim={dim}: measurement failed", file=sys.stderr)
            continue
        speedup = best["raw"][0] / best["graph"][0]
        csv_lines.append("%d,%d,f16,%d,%.6g,%.6g,%.6g,%.6g,%.4f" % (
            rows, dim, iters, best["raw"][0], best["raw"][1],
            best["graph"][0], best["graph"][1], speedup))
        line = ("rows=%-8d dim=%-5d iters=%-5d raw=%8.4fms %8.1fGB/s | "
                "graph=%8.4fms %8.1fGB/s | speedup=%.2fx" % (
                    rows, dim, iters, best["raw"][0], best["raw"][1],
                    best["graph"][0], best["graph"][1], speedup))
        log_lines.append(line)
        print(line)

    for name, lines in (("graph.csv", csv_lines), ("graph.log", log_lines)):
        with open(os.path.join(args.out, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.join(args.out, 'graph.csv')}")


if __name__ == "__main__":
    import sys
    main()
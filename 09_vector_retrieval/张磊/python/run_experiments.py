#!/usr/bin/env python3
"""Driver that sweeps nprobe / batch_size and summarizes perf + quality logs.

The C++ vsearch binary must already be built. This script only orchestrates
processes and aggregates the plain-text logs produced by `vsearch bench`.

Example:
    python run_experiments.py --vsearch build/vsearch
"""

import argparse
import csv
import os
import subprocess
import sys


def parse_kv_log(path):
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or "=" not in line:
                continue
            k, v = line.strip().split("=", 1)
            rows[k] = v
    return rows


def parse_perf(path):
    """Reads the space separated perf log and returns list of dicts."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        header = None
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            tok = line.split()
            if header is None:
                header = tok
                continue
            row = dict(zip(header, tok))
            rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vsearch", default="build/vsearch")
    p.add_argument("--data", default="data")
    p.add_argument("--vectors", default="vectors.bin")
    p.add_argument("--queries", default="queries.bin")
    p.add_argument("--params", default="params.txt")
    p.add_argument("--nprobe-list", default="1,2,4,8,16,32,64")
    p.add_argument("--batch-list", default="32,128,512")
    p.add_argument("--mode", default="ivf_flat", choices=["ivf_flat", "ivf_pq"])
    p.add_argument("--ref-query-limit", type=int, default=100)
    p.add_argument("--summary", default="data/experiment_summary.csv")
    p.add_argument("--out-dir", default="outputs")
    args = p.parse_args()

    nprobes = [int(x) for x in args.nprobe_list.split(",") if x]
    batches = [int(x) for x in args.batch_list.split(",") if x]
    os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)

    summary = []
    quality_rows = []
    for nprobe in nprobes:
        for batch in batches:
            tag = f"{args.mode}_np{nprobe}_b{batch}"
            out_dir = os.path.join(args.out_dir, "sweep", tag)
            os.makedirs(out_dir, exist_ok=True)
            index = os.path.join(out_dir, "index.idx")
            result = os.path.join(out_dir, "result.txt")
            perf = os.path.join(out_dir, "perf.log")
            quality = os.path.join(out_dir, "quality.log")
            cmd = [
                args.vsearch, "bench",
                "--vectors=" + os.path.join(args.data, args.vectors),
                "--queries=" + os.path.join(args.data, args.queries),
                "--params=" + os.path.join(args.data, args.params),
                f"--search_mode={args.mode}",
                f"--nprobe={nprobe}",
                f"--batch_size={batch}",
                f"--ref_query_limit={args.ref_query_limit}",
                "--force_rebuild=1",
                f"--index={index}",
                f"--result_path={result}",
                f"--perf_log_path={perf}",
                f"--quality_log_path={quality}",
            ]
            print(" ".join(cmd))
            r = subprocess.run(cmd, check=True)
            if r.returncode != 0:
                sys.exit(1)

            rows = parse_perf(perf)
            approx = [x for x in rows if x.get("mode") == args.mode]
            exact = [x for x in rows if x.get("mode") == "exact_gpu"]
            q = parse_kv_log(quality)
            row = {
                "mode": args.mode,
                "nprobe": nprobe,
                "batch": batch,
                "qps": approx[0].get("qps") if approx else "",
                "p50_ms": approx[0].get("p50_ms") if approx else "",
                "p99_ms": approx[0].get("p99_ms") if approx else "",
                "exact_qps": exact[0].get("qps") if exact else "",
                "recall_at_k": q.get("recall_at_k", ""),
                "avg_distance_error": q.get("avg_distance_error", ""),
                "perf_log": perf,
            }
            summary.append(row)
            quality_rows.append(row)

    with open(args.summary, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print("wrote", args.summary)
    print(f"total configurations: {len(summary)}")


if __name__ == "__main__":
    main()

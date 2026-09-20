#!/usr/bin/env python3
"""Collect measured benchmark JSON files into a compact Markdown table."""

import json
import pathlib
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: summarize_results.py RESULTS_DIR OUTPUT.md", file=sys.stderr)
        return 2
    results_dir = pathlib.Path(sys.argv[1])
    output = pathlib.Path(sys.argv[2])
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "measured":
            continue
        ds = data["dataset"]
        exact = data["exact"]
        ivf = data["ivf_flat"]
        rows.append(
            [
                path.name,
                data.get("device", ""),
                ds["n"],
                ds["nq"],
                ds["dim"],
                ds["k"],
                f"{exact['qps']:.3f}",
                f"{exact['p50_ms']:.3f}",
                f"{exact['p99_ms']:.3f}",
                f"{exact['cpu_gpu_recall_at_k']:.6f}",
                f"{exact['speedup_vs_cpu']:.3f}",
                ivf["nprobe"],
                f"{ivf['qps']:.3f}",
                f"{ivf['recall_at_k']:.6f}",
                f"{ivf['mean_rank_score_error']:.6f}",
            ]
        )
    header = [
        "log", "GPU", "N", "NQ", "D", "K", "exact QPS", "P50 ms",
        "P99 ms", "exact recall", "speedup", "nprobe", "IVF QPS",
        "IVF recall", "mean rank score error",
    ]
    lines = ["# Measured benchmark summary", "", "| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    if not rows:
        lines.extend(["", "No measured benchmark JSON files were found."])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output} with {len(rows)} measured rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


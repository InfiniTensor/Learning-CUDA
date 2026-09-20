#!/usr/bin/env python3
"""Compare GPU schedules/launch methods with two single-thread CPU baselines."""
import argparse
import csv
import json
import statistics
import subprocess
from pathlib import Path
from generate import generate, write_csr


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", type=Path, default=Path("build/trade_flow"))
    p.add_argument("--nodes", type=int, default=100020)
    p.add_argument("--edges", type=int, default=1000000)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--families", nargs="+", default=["layered", "balanced", "random"])
    p.add_argument("--directory", type=Path, default=Path("."))
    a = p.parse_args()
    exe = a.exe.resolve()
    data, reports = a.directory / "data", a.directory / "reports"
    logs = reports / "metrics"
    logs.mkdir(parents=True, exist_ok=True)
    variants = [("cuda_warp_graph", ["--engine", "cuda", "--schedule", "warp"]),
                ("cuda_thread_graph", ["--engine", "cuda", "--schedule", "thread"]),
                ("cuda_warp_direct", ["--engine", "cuda", "--schedule", "warp", "--no-graphs"]),
                ("cuda_thread_direct", ["--engine", "cuda", "--schedule", "thread", "--no-graphs"]),
                ("cpu_dinic", ["--engine", "cpu"]),
                ("cpu_preflow", ["--engine", "cpu-pr"])]
    rows, comparisons = [], []
    for family in a.families:
        stem = data / f"{family}_{a.nodes}_{a.edges}"
        edges, queries = generate(family, a.nodes, a.edges, 2026)
        write_csr(stem.with_suffix(".bin"), a.nodes, edges)
        stem.with_suffix(".queries").write_text("".join(f"{s} {t}\n" for s, t in queries))
        del edges
        common = [str(exe), "--graph", str(stem.with_suffix(".bin")),
                  "--queries", str(stem.with_suffix(".queries"))]
        # Full-scale validation, including residual cut certificate. Its time
        # is separately logged and not used as a benchmark repetition.
        check_log = logs / f"{family}_validation.json"
        check_out = logs / f"{family}_validation.txt"
        subprocess.run(common + ["--check", "--output", str(check_out), "--log", str(check_log)], check=True, timeout=300)
        expected = check_out.read_text()
        print(f"{family}: full-scale certificate and CPU equality passed", flush=True)
        samples = {name: [] for name, _ in variants}
        # Rotate order between repetitions to reduce warmup/thermal bias.
        for repeat in range(a.repeats):
            order = variants[repeat:] + variants[:repeat]
            for name, options in order:
                path = logs / f"{family}_{name}_{repeat}.json"
                out = logs / f"{family}_{name}_{repeat}.txt"
                subprocess.run(common + options + ["--output", str(out), "--log", str(path)], check=True, timeout=300)
                assert out.read_text() == expected, (family, name, repeat)
                metrics = json.loads(path.read_text())
                samples[name].append(metrics)
                row = {"family": family, "variant": name, "repeat": repeat}
                for key in ["nodes", "edges", "query_count", "input_read_ms", "preprocess_ms", "ttfq_ms", "first_query_ms",
                            "total_query_ms", "tpq_ms", "process_until_log_ms", "device_buffer_bytes"]:
                    row[key] = metrics[key]
                rows.append(row)
                with (reports / "benchmark.csv").open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
        medians = {name: {key: statistics.median(x[key] for x in values)
                          for key in ["input_read_ms", "preprocess_ms", "ttfq_ms", "first_query_ms", "total_query_ms", "tpq_ms", "process_until_log_ms"]}
                   for name, values in samples.items()}
        best_cpu = min(medians["cpu_dinic"]["total_query_ms"], medians["cpu_preflow"]["total_query_ms"])
        comparison = {"family": family, "medians": medians,
                      "warp_speedup_vs_dinic": medians["cpu_dinic"]["total_query_ms"] / medians["cuda_warp_graph"]["total_query_ms"],
                      "warp_speedup_vs_best_cpu": best_cpu / medians["cuda_warp_graph"]["total_query_ms"],
                      "default_speedup_vs_dinic": medians["cpu_dinic"]["total_query_ms"] / medians["cuda_thread_graph"]["total_query_ms"],
                      "default_speedup_vs_best_cpu": best_cpu / medians["cuda_thread_graph"]["total_query_ms"],
                      "default_process_speedup_vs_best_cpu": min(medians["cpu_dinic"]["process_until_log_ms"], medians["cpu_preflow"]["process_until_log_ms"]) / medians["cuda_thread_graph"]["process_until_log_ms"]}
        comparisons.append(comparison)
        print(json.dumps(comparison, indent=2), flush=True)
        (reports / "summary.json").write_text(json.dumps({"nodes": a.nodes, "edges": a.edges, "query_count": 10,
            "repeats": a.repeats, "seed": 2026, "comparisons": comparisons}, indent=2) + "\n")


if __name__ == "__main__":
    main()

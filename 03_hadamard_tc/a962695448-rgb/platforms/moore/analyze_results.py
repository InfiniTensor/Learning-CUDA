#!/usr/bin/env python3
"""Independently verify the three original MUSA timing CSVs and summarize them."""

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def analyze(source):
    report = json.loads((source / "run_summary.json").read_text())
    if report["status"] != "PASS" or not report["validation"]["full_matrix"]:
        raise ValueError("A successful full validation report is required")
    medians = {}
    expected_keys = None
    total = 0
    for round_index in (1, 2, 3):
        path = source / f"benchmark_round{round_index}.csv"
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != report["artifacts"][path.name]["sha256"]:
            raise ValueError("CSV checksum mismatch: " + path.name)
        with path.open(newline="") as stream:
            records = list(csv.DictReader(stream))
        if len(records) != 1350:
            raise ValueError("Expected 1,350 observations in each round")
        groups = collections.defaultdict(list)
        seen = set()
        for record in records:
            rows, dim = int(record["rows"]), int(record["dim"])
            key = (record["dtype"], rows, dim, record["method"])
            group = int(record["group"])
            if (*key, group) in seen or group not in range(5):
                raise ValueError("Duplicate or invalid timing group")
            seen.add((*key, group))
            if int(record["batch"]) * int(record["seq"]) * int(record["heads"]) != rows:
                raise ValueError("Invalid four-dimensional shape")
            if int(record["repeats"]) != 100 or int(record["seed"]) != 2909 or float(record["scale"]) != 1:
                raise ValueError("Unexpected measurement protocol")
            us, ms = float(record["kernel_us"]), float(record["kernel_ms"])
            if not math.isfinite(us) or us <= 0 or not math.isclose(ms, us / 1000, rel_tol=1e-9):
                raise ValueError("Invalid event time or unit conversion")
            groups[key].append(us)
        if len(groups) != 270 or any(len(values) != 5 for values in groups.values()):
            raise ValueError("Incomplete configuration matrix")
        if expected_keys is None:
            expected_keys = set(groups)
        if set(groups) != expected_keys:
            raise ValueError("Configuration sets differ across rounds")
        medians[round_index] = {key: statistics.median(values) for key, values in groups.items()}
        total += len(records)
    shapes = sorted({key[:3] for key in expected_keys})
    comparisons, summary = [], []
    for candidate in ("optimized", "shuffle32"):
        for operation in ("transform", "split", "fused"):
            ratios, stable_faster, stable_five_percent = [], 0, 0
            for shape in shapes:
                reductions = []
                for round_index in (1, 2, 3):
                    baseline = medians[round_index][(*shape, "baseline_" + operation)]
                    value = medians[round_index][(*shape, candidate + "_" + operation)]
                    ratio, reduction = baseline / value, 100 * (1 - value / baseline)
                    ratios.append(ratio)
                    reductions.append(reduction)
                    comparisons.append(dict(round=round_index, dtype=shape[0], rows=shape[1],
                                            dim=shape[2], operation=operation, candidate=candidate,
                                            baseline_us=baseline, candidate_us=value,
                                            baseline_over_candidate=ratio, time_reduction_percent=reduction))
                stable_faster += all(value > 0 for value in reductions)
                stable_five_percent += all(value >= 5 for value in reductions)
            summary.append(dict(candidate=candidate, operation=operation, configurations=len(shapes),
                                minimum_speedup=min(ratios), maximum_speedup=max(ratios),
                                faster_in_every_round=stable_faster,
                                at_least_five_percent_in_every_round=stable_five_percent))
    representatives = []
    for shape in (("fp16", 16384, 256), ("bf16", 16384, 256), ("fp16", 1, 64), ("bf16", 17, 256)):
        times = {}
        for method in ("baseline_transform", "optimized_transform", "shuffle32_transform",
                       "baseline_fused", "optimized_fused", "shuffle32_fused"):
            values = [medians[round_index][(*shape, method)] for round_index in (1, 2, 3)]
            times[method] = {"round_medians_us": values, "median_of_round_medians_us": statistics.median(values)}
        representatives.append({"dtype": shape[0], "rows": shape[1], "dim": shape[2], "times": times})
    return {"observations": total, "rounds": 3, "configurations_per_round": 270,
            "method": "Median of five group-average event timings per configuration, in each of three independent benchmark processes",
            "limits": "Warm-input MUSA event intervals; no CPU, cross-device, application-latency or statistical-significance claim",
            "summary": summary, "representatives": representatives, "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    result = analyze(args.source)
    args.output.mkdir(parents=True)
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output / "paired_comparisons.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["comparisons"][0]))
        writer.writeheader()
        writer.writerows(result["comparisons"])
    print(json.dumps({"verified_observations": result["observations"], "output": str(args.output)}))


if __name__ == "__main__":
    main()

"""Verify the recorded paired experiment and recompute every old/new comparison."""

import argparse
import collections
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

DTYPES = ("fp16", "bf16")
ROWS = (1, 17, 257, 4096, 16384)
DIMS = (64, 128, 256)
OPERATIONS = ("transform", "split", "fused")
METHODS = (
    "baseline",
    "control_optimized",
    "optimized",
    "control_shuffle32",
    "shuffle32",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_medians(recorded):
    state = json.loads((recorded / "optimization.json").read_text())
    require(state["status"] == "PASS", "The original experiment did not pass.")
    expected = {
        (dtype, rows, dim, method + "_" + operation)
        for dtype in DTYPES
        for rows in ROWS
        for dim in DIMS
        for method in METHODS
        for operation in OPERATIONS
    }
    medians, count = {}, 0
    for round_index in (1, 2, 3):
        path = recorded / f"paired-round{round_index}.csv"
        require(
            hashlib.sha256(path.read_bytes()).hexdigest()
            == state["artifacts"][path.name]["sha256"],
            f"Checksum mismatch: {path.name}",
        )
        with path.open(newline="") as stream:
            records = list(csv.DictReader(stream))
        require(len(records) == 2250, f"Expected 2250 observations: {path.name}")
        groups, seen = collections.defaultdict(list), set()
        for record in records:
            key = (
                record["dtype"],
                int(record["rows"]),
                int(record["dim"]),
                record["method"],
            )
            group = int(record["group"])
            require(
                (*key, group) not in seen and group in range(5),
                f"Duplicate or invalid timing group: {path.name}",
            )
            seen.add((*key, group))
            require(
                int(record["batch"]) * int(record["seq"]) * int(record["heads"])
                == key[1],
                "Inconsistent row count.",
            )
            require(
                int(record["repeats"]) == 100
                and int(record["seed"]) == 2909
                and float(record["scale"]) == 1,
                "Unexpected experiment parameters.",
            )
            value, ms = float(record["kernel_us"]), float(record["kernel_ms"])
            require(
                value > 0
                and math.isfinite(value)
                and math.isclose(ms, value / 1000, rel_tol=1e-9),
                "Invalid time or inconsistent microsecond/millisecond units.",
            )
            groups[key].append(value)
        require(
            set(groups) == expected, "Incomplete or unexpected configuration matrix."
        )
        require(all(len(values) == 5 for values in groups.values()), "Missing groups.")
        medians[round_index] = {
            key: statistics.median(values) for key, values in groups.items()
        }
        count += len(records)
    return medians, count


def analyze(recorded):
    medians, count = read_medians(recorded)
    shapes = sorted(
        (dtype, rows, dim) for dtype in DTYPES for rows in ROWS for dim in DIMS
    )
    comparisons, summary, representatives = [], [], []
    for method in ("optimized", "shuffle32"):
        for operation in OPERATIONS:
            ratios, stable, stable_five = [], 0, 0
            for shape in shapes:
                reductions = []
                for round_index in (1, 2, 3):
                    old = medians[round_index][
                        (*shape, "control_" + method + "_" + operation)
                    ]
                    new = medians[round_index][(*shape, method + "_" + operation)]
                    ratio, reduction = old / new, 100 * (1 - new / old)
                    ratios.append(ratio)
                    reductions.append(reduction)
                    comparisons.append(
                        {
                            "round": round_index,
                            "dtype": shape[0],
                            "rows": shape[1],
                            "dim": shape[2],
                            "method": method,
                            "operation": operation,
                            "old_us": old,
                            "new_us": new,
                            "old_over_new": ratio,
                            "time_reduction_percent": reduction,
                        }
                    )
                stable += all(value > 0 for value in reductions)
                stable_five += all(value >= 5 for value in reductions)
            summary.append(
                {
                    "method": method,
                    "operation": operation,
                    "configurations": len(shapes),
                    "minimum_speedup": min(ratios),
                    "maximum_speedup": max(ratios),
                    "faster_every_round": stable,
                    "at_least_five_percent_every_round": stable_five,
                }
            )
    for shape in (
        ("fp16", 16384, 256),
        ("bf16", 16384, 256),
        ("fp16", 1, 64),
        ("bf16", 17, 256),
    ):
        item = {"dtype": shape[0], "rows": shape[1], "dim": shape[2], "timings": {}}
        for method in ("optimized", "shuffle32"):
            for operation in OPERATIONS:
                old = [
                    medians[n][(*shape, "control_" + method + "_" + operation)]
                    for n in (1, 2, 3)
                ]
                new = [
                    medians[n][(*shape, method + "_" + operation)] for n in (1, 2, 3)
                ]
                item["timings"][method + "_" + operation] = {
                    "old_round_medians_us": old,
                    "new_round_medians_us": new,
                    "old_us": statistics.median(old),
                    "new_us": statistics.median(new),
                    "old_over_new": statistics.median(old) / statistics.median(new),
                }
        representatives.append(item)
    accepted = all(
        item["at_least_five_percent_every_round"] == 30
        for item in summary
        if item["method"] == "optimized" and item["operation"] in ("split", "fused")
    )
    return {
        "status": "VERIFIED",
        "accepted": accepted,
        "observations": count,
        "rounds": 3,
        "configurations_per_round": 450,
        "summary": summary,
        "representatives": representatives,
        "measurement": "Median of 5 group-average event intervals; 100 calls per group; paired old/new methods",
        "scope": "Same S4000, warm inputs, scalar scale=1; no application-level or cross-device inference",
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recorded", type=Path, default=Path(__file__).resolve().parent / "recorded"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory."
    )
    args = parser.parse_args()
    result = analyze(args.recorded)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output / "paired-comparisons.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["comparisons"][0]))
        writer.writeheader()
        writer.writerows(result["comparisons"])
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "accepted", "observations", "summary")
            },
            indent=2,
        )
    )
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

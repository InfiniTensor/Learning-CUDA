#!/usr/bin/env python3
"""比较多个天数 benchmark.csv；保留负例，输出可复算 JSON 与逐方法汇总 CSV。"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys


KEYS = ("dtype", "batch", "seq", "heads", "dim", "scale")
OPERATIONS = ("transform", "split", "fused")
CONDITIONS = ("rows", "repeats", "seed", "input_read_only", "input_working_set_bytes")
REQUIRED = set(KEYS + CONDITIONS + ("method", "group", "order", "kernel_us", "logical_io_bytes", "logical_GBs"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integer(value, name, minimum=1):
    number = int(value)
    if number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return number


def number(value, name):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def stats(values):
    average = statistics.mean(values)
    median = statistics.median(values)
    return {"count": len(values), "median_us": median, "mean_us": average,
            "minimum_us": min(values), "maximum_us": max(values),
            "population_stddev_us": statistics.pstdev(values),
            "cv_percent": statistics.pstdev(values) / average * 100,
            "range_over_median_percent": (max(values) - min(values)) / median * 100}


def load_run(path, run_index):
    buckets = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not REQUIRED <= set(reader.fieldnames or ()):
            raise ValueError(f"{path.name}: missing columns {sorted(REQUIRED - set(reader.fieldnames or ()))}")
        for line, row in enumerate(reader, start=2):
            try:
                item = dict(row)
                for name in ("batch", "seq", "heads", "dim", "rows", "repeats", "input_working_set_bytes", "logical_io_bytes"):
                    item[name] = integer(row[name], name)
                for name in ("group", "order", "seed"):
                    item[name] = integer(row[name], name, 0)
                for name in ("scale", "kernel_us", "logical_GBs"):
                    item[name] = number(row[name], name)
                if item["dtype"] not in ("fp16", "bf16") or not re.fullmatch(r"[a-z][a-z0-9_]*_(transform|split|fused)", item["method"]):
                    raise ValueError("unsupported dtype or method")
                if item["input_read_only"] not in ("true", "false"):
                    raise ValueError("input_read_only must be true or false")
                if item["rows"] != item["batch"] * item["seq"] * item["heads"]:
                    raise ValueError("rows differs from batch*seq*heads")
                if item["dim"] > 256 or item["dim"] & (item["dim"] - 1):
                    raise ValueError("dim must be a power of two <=256")
                if item["input_working_set_bytes"] != item["rows"] * item["dim"] * 2:
                    raise ValueError("input working set does not match FP16/BF16 tensor size")
                key = tuple(item[name] for name in KEYS)
                groups = buckets.setdefault((key, item["method"]), {})
                if item["group"] in groups:
                    raise ValueError("duplicate shape/method/group; do not concatenate repeated runs into one CSV")
                groups[item["group"]] = item
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"{path.name}:{line}: {error}") from error
    if not buckets:
        raise ValueError(f"{path.name}: no samples")
    methods = {method for key, method in buckets}
    prefixes = {method.rsplit("_", 1)[0] for method in methods}
    if "baseline" not in prefixes or len(prefixes) < 2:
        raise ValueError(f"{path.name}: require baseline and at least one candidate implementation")
    if methods != {prefix + "_" + op for prefix in prefixes for op in OPERATIONS}:
        raise ValueError(f"{path.name}: every implementation must include transform/split/fused")
    keys = {key for key, method in buckets}
    for key in keys:
        if {method for case, method in buckets if case == key} != methods:
            raise ValueError(f"{path.name}: incomplete method matrix for {key}")
        reference = buckets[(key, "baseline_transform")]
        if set(reference) != set(range(len(reference))):
            raise ValueError(f"{path.name}: group IDs must be contiguous from zero for {key}")
        condition = tuple(reference[0][name] for name in CONDITIONS)
        for method in methods:
            groups = buckets[(key, method)]
            if set(groups) != set(reference):
                raise ValueError(f"{path.name}: methods have different group coverage for {key}")
            if any(tuple(item[name] for name in CONDITIONS) != condition for item in groups.values()):
                raise ValueError(f"{path.name}: mixed measurement conditions for {key}")
            if len({item["logical_io_bytes"] for item in groups.values()}) != 1:
                raise ValueError(f"{path.name}: changing logical I/O estimate for {key}/{method}")
        for group in reference:
            if {buckets[(key, method)][group]["order"] for method in methods} != set(range(len(methods))):
                raise ValueError(f"{path.name}: duplicated/missing method order in group {group}")
    return {"run": run_index, "file": path.name, "sha256": digest(path), "bytes": path.stat().st_size,
            "raw_rows": sum(len(values) for values in buckets.values()), "shape_dtype_scale_cases": len(keys),
            "buckets": buckets}


def analyze(paths):
    if len(paths) < 2:
        raise ValueError("provide at least two separate run CSV files")
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("the same input path was supplied twice")
    runs = [load_run(path, index) for index, path in enumerate(paths, start=1)]
    reference = runs[0]["buckets"]
    for run in runs[1:]:
        if set(run["buckets"]) != set(reference):
            raise ValueError(f"run {run['run']}: shape/dtype/scale/method matrix differs; compare matching runs")
        for key, groups in reference.items():
            peer = run["buckets"][key]
            if set(peer) != set(groups):
                raise ValueError(f"run {run['run']}: number of groups differs for {key}")
            names = CONDITIONS + ("logical_io_bytes",)
            if tuple(groups[0][name] for name in names) != tuple(peer[0][name] for name in names):
                raise ValueError(f"run {run['run']}: seed/repeats/working-set conditions differ for {key}")
    methods, by_key = [], {}
    for (key, method) in sorted(reference):
        entry = dict(zip(KEYS, key), method=method)
        first = reference[(key, method)][0]
        entry["conditions"] = {name: first[name] for name in CONDITIONS + ("logical_io_bytes",)}
        entry["per_run"] = []
        for run in runs:
            samples = [sample for _, sample in sorted(run["buckets"][(key, method)].items())]
            values = [sample["kernel_us"] for sample in samples]
            entry["per_run"].append({"run": run["run"], **stats(values),
                                     "samples": [{name: s[name] for name in ("group", "order", "kernel_us", "logical_GBs")} for s in samples]})
        entry["across_run_medians"] = stats([value["median_us"] for value in entry["per_run"]])
        methods.append(entry)
        by_key[(key, method)] = entry
    comparisons = []
    for key, candidate_method in sorted(reference):
        prefix, operation = candidate_method.rsplit("_", 1)
        if prefix != "baseline":
            base = by_key[(key, "baseline_" + operation)]
            opt = by_key[(key, candidate_method)]
            paired = []
            for a, b in zip(base["per_run"], opt["per_run"]):
                paired.append({"run": a["run"], "baseline_median_us": a["median_us"],
                               "candidate_median_us": b["median_us"],
                               "baseline_over_candidate": a["median_us"] / b["median_us"],
                               "time_reduction_percent": 100 * (1 - b["median_us"] / a["median_us"])})
            gains = [p["time_reduction_percent"] for p in paired]
            comparison = dict(zip(KEYS, key), candidate=prefix, method=candidate_method, operation=operation, per_run=paired,
                              minimum_time_reduction_percent=min(gains), maximum_time_reduction_percent=max(gains),
                              stable_candidate_every_run_at_least_5_percent=min(gains) >= 5,
                              every_run_faster=min(gains) > 0, any_run_slowdown=min(gains) < 0,
                              every_run_slowdown=max(gains) < 0, any_run_regression_over_3_percent=min(gains) < -3,
                              every_run_regression_over_3_percent=max(gains) < -3)
            comparisons.append(comparison)
            opt["comparison_to_baseline"] = comparison
    counters = {name: sum(c[name] for c in comparisons) for name in (
        "stable_candidate_every_run_at_least_5_percent", "every_run_faster", "any_run_slowdown",
        "every_run_slowdown", "any_run_regression_over_3_percent", "every_run_regression_over_3_percent")}
    return {
        "status": "ANALYSIS_COMPLETE", "correctness_status": "NOT_INFERRED_FROM_TIMING_CSV",
        "inputs": [{k: v for k, v in run.items() if k != "buckets"} for run in runs],
        "methodology": {
            "metric": "kernel_us is the backend device event interval divided by repeats, then median across groups per run. The backend API is identified in the run manifest, not inferred from CSV. Split methods include transform and quantization kernel launches.",
            "excluded_from_interval": ["allocation", "H2D/D2H", "warmup", "validation"],
            "scope_limits": "Not host end-to-end or isolated single-kernel timing. Event interval may include device idle gaps between host launches; same seeded read-only input is reused, with warm-cache effects.",
            "logical_GBs": "Reported logical tensor-I/O estimate only; not measured physical memory bandwidth.",
            "comparison": "Each non-baseline implementation paired with baseline by dtype/full shape/scale/operation within each run. Time reduction (%) = 100*(1-candidate_median/baseline_median). Candidate only if >=5% in EVERY run. Different method matrices cannot be mixed.",
            "variation": "Within-run population stddev/CV describe group means. Across-run stddev/CV describe run medians. Neither is a confidence interval; repeats are not counted as new independent test cases.",
            "missing_evidence": "CSV does not encode hardware identity, compiler/driver versions, clock/temperature, source revision, process independence or final validation status. Verify accompanying run manifests before making platform or correctness claims.",
        },
        "counts": {"runs": len(runs), "total_raw_rows": sum(r["raw_rows"] for r in runs),
                   "distinct_shape_dtype_scale_cases": runs[0]["shape_dtype_scale_cases"],
                   "distinct_method_cases": len(methods), "paired_candidate_operation_cases": len(comparisons), **counters},
        "counts_by_candidate": {candidate: {
            "paired_operation_cases": sum(c["candidate"] == candidate for c in comparisons),
            **{name: sum(c[name] for c in comparisons if c["candidate"] == candidate) for name in counters},
        } for candidate in sorted({c["candidate"] for c in comparisons})},
        "all_method_statistics": methods, "all_comparisons_including_slowdowns": comparisons,
    }


def write_outputs(report, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rows = []
    for method in report["all_method_statistics"]:
        row = {key: method[key] for key in KEYS + ("method",)}
        row.update(method["conditions"])
        for value in method["per_run"]:
            prefix = f"run{value['run']}_"
            row.update({prefix + key: item for key, item in value.items() if key not in ("run", "samples")})
        row.update({"across_run_medians_" + key: value for key, value in method["across_run_medians"].items()})
        comparison = method.get("comparison_to_baseline")
        row["comparison_role"] = "candidate_vs_matching_baseline" if comparison else "baseline"
        for run in range(1, report["counts"]["runs"] + 1):
            for name in ("baseline_over_candidate", "time_reduction_percent"):
                row[f"run{run}_{name}"] = comparison["per_run"][run - 1][name] if comparison else ""
        for key in ("minimum_time_reduction_percent", "maximum_time_reduction_percent", "stable_candidate_every_run_at_least_5_percent",
                    "every_run_faster", "any_run_slowdown", "every_run_slowdown", "any_run_regression_over_3_percent", "every_run_regression_over_3_percent"):
            row[key] = comparison[key] if comparison else ""
        rows.append(row)
    with (output / "method_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path, help="按时间顺序传入独立运行的 CSV")
    parser.add_argument("--output", required=True, type=Path, help="新的输出目录，禁止覆盖已有结果")
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("output already exists; select a fresh directory")
        report = analyze(args.csv)
        report["analyzer_sha256"] = digest(Path(__file__))
        write_outputs(report, args.output)
        print(json.dumps({"status": report["status"], "counts": report["counts"], "output": str(args.output)}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print("ANALYSIS_FAILED:", error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

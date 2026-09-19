#!/usr/bin/env python3
"""Run convergence and variance-reduction sweeps without external packages."""

from __future__ import annotations

import argparse
import csv
import math
import shlex
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build/derivatives_pricer"))
    parser.add_argument("--option", type=Path, default=Path("configs/european_call.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sweep"))
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--paths", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    parser.add_argument(
        "--variance", nargs="+", choices=("none", "antithetic", "control_variate"),
        default=["none", "antithetic", "control_variate"]
    )
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--master-seed", type=int, default=20260824)
    parser.add_argument(
        "--seed", type=int, default=None,
        help="兼容旧命令：提供时覆盖 --master-seed",
    )
    parser.add_argument("--replications", type=int, default=1)
    parser.add_argument(
        "--rng", choices=("curand", "custom", "halton"), default="curand"
    )
    parser.add_argument("--qmc-replications", type=int, default=8)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--compute-greeks", action="store_true")
    parser.add_argument(
        "--reference-paths", type=int, default=200_000,
        help="Asian/Barrier 固定 CPU reference 的路径数；0 表示不计算",
    )
    parser.add_argument(
        "--cpu-benchmark-paths", type=int, default=0,
        help="CUDA 吞吐对比使用的同配置 CPU 样本数；0 表示不运行性能基线",
    )
    return parser.parse_args()


MASK64 = (1 << 64) - 1


def derive_replication_seed(master_seed: int, replication_id: int) -> int:
    """用 SplitMix64 派生可复现的 64 位 replication seed。"""
    value = (master_seed + 0x9E3779B97F4A7C15 * (replication_id + 1)) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def load_key_values(path: Path) -> dict[str, str]:
    """读取程序生成的 key=value 日志，保留原始字符串精度。"""
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def write_aggregate(
    rows: list[dict[str, str]], output_path: Path
) -> list[dict[str, str]]:
    """按配置聚合独立 replication，区分单次 SE 与跨 seed 经验波动。"""
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["num_paths"], row["variance_reduction"], row.get("backend", ""),
            row.get("rng", ""), row.get("num_steps", ""),
        )
        groups[key].append(row)

    aggregate_rows: list[dict[str, str]] = []
    for key, group in groups.items():
        def values(field: str) -> list[float]:
            return [float(row[field]) for row in group]

        prices = values("price")
        runtimes = values("total_runtime_ms")
        standard_errors = values("standard_error")
        absolute_errors = [
            value for value in values("absolute_error") if math.isfinite(value)
        ]
        speedups = [
            value for value in values("throughput_speedup_vs_cpu_single_thread")
            if math.isfinite(value) and value > 0.0
        ]
        cpu_benchmark_times = [
            value for value in values("cpu_benchmark_ms")
            if math.isfinite(value) and value > 0.0
        ]
        cpu_throughputs = [
            value for value in values("cpu_paths_per_second")
            if math.isfinite(value) and value > 0.0
        ]
        empirical_sd = statistics.stdev(prices) if len(prices) > 1 else 0.0
        mean_standard_error = statistics.fmean(standard_errors)
        mean_price = statistics.fmean(prices)
        calibration_ratio = (
            empirical_sd / mean_standard_error
            if len(prices) > 1 and mean_standard_error > 0.0
            else None
        )
        aggregate_rows.append(
            {
                "num_paths": key[0],
                "variance_reduction": key[1],
                "backend": key[2],
                "rng": key[3],
                "num_steps": key[4],
                "replications": str(len(group)),
                "mean_price": f"{mean_price:.17g}",
                "empirical_price_sd": f"{empirical_sd:.17g}",
                "mean_reported_standard_error": f"{mean_standard_error:.17g}",
                "se_calibration_ratio": (
                    f"{calibration_ratio:.17g}"
                    if calibration_ratio is not None else ""
                ),
                "relative_standard_error": (
                    f"{mean_standard_error / abs(mean_price):.17g}"
                    if mean_price != 0.0 else ""
                ),
                "mean_absolute_error":
                    f"{statistics.fmean(absolute_errors):.17g}"
                    if absolute_errors else "",
                "mean_total_runtime_ms": f"{statistics.fmean(runtimes):.17g}",
                "median_total_runtime_ms": f"{statistics.median(runtimes):.17g}",
                "mean_simulation_kernel_ms":
                    f"{statistics.fmean(values('simulation_kernel_ms')):.17g}",
                "mean_reduction_ms":
                    f"{statistics.fmean(values('reduction_ms')):.17g}",
                "mean_kernel_paths_per_second":
                    f"{statistics.fmean(values('kernel_paths_per_second')):.17g}",
                "mean_end_to_end_paths_per_second":
                    f"{statistics.fmean(values('end_to_end_paths_per_second')):.17g}",
                "cpu_benchmark_paths": group[0].get("cpu_benchmark_paths", ""),
                "mean_cpu_benchmark_ms": (
                    f"{statistics.fmean(cpu_benchmark_times):.17g}"
                    if cpu_benchmark_times else ""
                ),
                "mean_cpu_paths_per_second": (
                    f"{statistics.fmean(cpu_throughputs):.17g}"
                    if cpu_throughputs else ""
                ),
                "mean_throughput_speedup_vs_cpu_single_thread":
                    f"{statistics.fmean(speedups):.17g}" if speedups else "",
                "mean_se_squared_times_runtime_ms":
                    f"{statistics.fmean(se * se * runtime for se, runtime in zip(standard_errors, runtimes)):.17g}",
                "reference_price": (
                    group[0].get("reference_price", "")
                    if math.isfinite(float(group[0].get("reference_price", "nan")))
                    else ""
                ),
                "reference_method": (
                    group[0].get("reference_method", "")
                    if math.isfinite(float(group[0].get("reference_price", "nan")))
                    else ""
                ),
            }
        )

    columns = list(aggregate_rows[0]) if aggregate_rows else []
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        if columns:
            writer.writeheader()
            writer.writerows(aggregate_rows)
    return aggregate_rows


def write_report_table(rows: list[dict[str, str]], output_path: Path) -> None:
    """生成 Markdown 聚合表；完整精度保留在 aggregate.csv。"""
    columns = [
        ("num_paths", "Paths"),
        ("num_steps", "Steps"),
        ("variance_reduction", "Variance"),
        ("rng", "RNG"),
        ("replications", "Reps"),
        ("mean_price", "Mean price"),
        ("empirical_price_sd", "Empirical SD"),
        ("mean_reported_standard_error", "Mean SE"),
        ("se_calibration_ratio", "SD/SE"),
        ("mean_absolute_error", "Mean abs. error"),
        ("median_total_runtime_ms", "Median ms"),
        ("mean_simulation_kernel_ms", "Kernel ms"),
        ("mean_reduction_ms", "Reduction ms"),
        ("mean_end_to_end_paths_per_second", "Mean paths/s"),
        ("cpu_benchmark_paths", "CPU paths"),
        ("mean_cpu_benchmark_ms", "CPU ms"),
        ("mean_throughput_speedup_vs_cpu_single_thread", "CPU speedup"),
    ]
    header = "| " + " | ".join(label for _, label in columns) + " |\n"
    divider = "|" + "|".join("---" for _ in columns) + "|\n"
    lines = [header, divider]
    for row in rows:
        values = [row.get(key, "").replace("|", "\\|") for key, _ in columns]
        lines.append("| " + " | ".join(values) + " |\n")
    output_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.replications <= 0 or any(paths < 2 for paths in args.paths):
        raise ValueError("replications must be positive and every path count must be >= 2")
    master_seed = args.seed if args.seed is not None else args.master_seed
    if master_seed < 0 or master_seed > MASK64:
        raise ValueError("master seed must fit in uint64")
    if args.reference_paths == 1 or args.reference_paths < 0:
        raise ValueError("reference paths must be 0 or >= 2")
    if args.cpu_benchmark_paths == 1 or args.cpu_benchmark_paths < 0:
        raise ValueError("CPU benchmark paths must be 0 or >= 2")
    if args.qmc_replications < 2 or args.num_gpus <= 0:
        raise ValueError("qmc replications must be >=2 and num_gpus must be positive")
    if args.rng == "halton" and any(
        paths % args.qmc_replications != 0 for paths in args.paths
    ):
        raise ValueError("every Halton path count must divide qmc_replications")
    args.binary = args.binary.resolve()
    args.option = args.option.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.binary.is_file():
        raise FileNotFoundError(f"binary does not exist: {args.binary}")
    if not args.option.is_file():
        raise FileNotFoundError(f"option config does not exist: {args.option}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
    )
    (args.output_dir / "option_config.txt").write_text(
        args.option.read_text(encoding="utf-8"), encoding="utf-8"
    )
    rows: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="pricing_sweep_") as temporary:
        config_path = Path(temporary) / "simulation.txt"
        option_values = load_key_values(args.option)
        option_type = option_values.get("option_type", "").strip("\"'")
        model = option_values.get("model", "black_scholes").strip("\"'")
        path_dependent = (
            model not in {"black_scholes", "gbm"}
            or option_type in {"asian_call", "barrier_call"}
        )
        fixed_reference: dict[str, str] | None = None
        reference_seed = derive_replication_seed(master_seed, 0x524546)
        if path_dependent and args.reference_paths >= 2:
            # 固定 reference 只计算一次；被测组合均使用另一组 replication seeds。
            reference_variance = (
                "none" if option_type in {"american_call", "american_put"}
                else "control_variate"
            )
            config_path.write_text(
                f"num_paths = {args.reference_paths}\nnum_steps = {args.steps}\n"
                f"seed = {reference_seed}\nrng = curand\n"
                f"variance_reduction = {reference_variance}\n"
                f"qmc_replications = {args.qmc_replications}\n"
                "num_gpus = 1\ncompute_greeks = false\n",
                encoding="utf-8",
            )
            reference_result = args.output_dir / "reference.result.txt"
            reference_performance = args.output_dir / "reference.performance.log"
            subprocess.run(
                [
                    str(args.binary), "--option", str(args.option),
                    "--simulation", str(config_path), "--backend", "cpu",
                    "--reference-paths", "0", "--cpu-benchmark-paths", "0",
                    "--output", str(reference_result),
                    "--performance", str(reference_performance),
                ],
                check=True,
            )
            fixed_reference = load_key_values(reference_result)
        for replication_id in range(args.replications):
            replication_seed = derive_replication_seed(master_seed, replication_id)
            # 同一 replication 的所有方法共享 seed，实现 common random numbers；
            # 不同 path count 从同一随机流前缀开始，便于绘制嵌套收敛曲线。
            for paths in args.paths:
                for variance in args.variance:
                    config_path.write_text(
                        f"num_paths = {paths}\nnum_steps = {args.steps}\n"
                        f"seed = {replication_seed}\nrng = {args.rng}\n"
                        f"variance_reduction = {variance}\n"
                        f"qmc_replications = {args.qmc_replications}\n"
                        f"num_gpus = {args.num_gpus}\n"
                        f"compute_greeks = {'true' if args.compute_greeks else 'false'}\n",
                        encoding="utf-8",
                    )
                    stem = f"rep_{replication_id:03d}_paths_{paths}_{variance}"
                    result_path = args.output_dir / f"{stem}.result.txt"
                    performance_path = args.output_dir / f"{stem}.performance.log"
                    subprocess.run(
                        [
                            str(args.binary), "--option", str(args.option),
                            "--simulation", str(config_path), "--backend", args.backend,
                            "--reference-paths", "0",
                            "--cpu-benchmark-paths", str(args.cpu_benchmark_paths),
                            "--output", str(result_path), "--performance",
                            str(performance_path),
                        ],
                        check=True,
                    )
                    row = {
                        "master_seed": str(master_seed),
                        "replication_id": str(replication_id),
                        "replication_seed": str(replication_seed),
                        "num_paths": str(paths),
                        "variance_reduction": variance,
                    }
                    row.update(load_key_values(result_path))
                    row.update(load_key_values(performance_path))
                    if fixed_reference is not None:
                        reference_price = float(fixed_reference["price"])
                        row["reference_price"] = fixed_reference["price"]
                        row["reference_method"] = (
                            f"fixed_cpu_control_variate_{args.reference_paths}_samples"
                        )
                        row["reference_seed"] = str(reference_seed)
                        row["reference_standard_error"] = fixed_reference["standard_error"]
                        row["reference_confidence_95_low"] = fixed_reference["confidence_95_low"]
                        row["reference_confidence_95_high"] = fixed_reference["confidence_95_high"]
                        row["absolute_error"] = f"{abs(float(row['price']) - reference_price):.17g}"
                    rows.append(row)

    columns = [
        "master_seed", "replication_id", "replication_seed", "num_paths",
        "model", "num_steps", "variance_reduction", "rng", "rng_requested", "backend", "price",
        "reference_price", "reference_method", "reference_seed",
        "reference_standard_error", "reference_confidence_95_low",
        "reference_confidence_95_high", "absolute_error", "standard_error", "confidence_95_low",
        "confidence_95_high", "delta", "gamma", "vega", "greeks_runtime_ms",
        "total_runtime_ms", "simulation_kernel_ms",
        "reduction_ms", "physical_paths", "kernel_paths_per_second",
        "end_to_end_paths_per_second", "throughput_speedup_vs_cpu_single_thread",
        "cpu_benchmark_paths", "cpu_benchmark_ms", "cpu_paths_per_second",
    ]
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    aggregate_rows = write_aggregate(rows, args.output_dir / "aggregate.csv")
    write_report_table(aggregate_rows, args.output_dir / "report_table.md")


if __name__ == "__main__":
    main()

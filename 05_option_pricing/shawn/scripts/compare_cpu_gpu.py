#!/usr/bin/env python3
"""Run one configuration on CPU/GPU and enforce numerical comparison gates."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build/derivatives_pricer"))
    parser.add_argument("--option", type=Path, required=True)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--z-score", type=float, default=5.0)
    parser.add_argument("--price-absolute-tolerance", type=float, default=0.02)
    parser.add_argument("--greek-relative-tolerance", type=float, default=0.10)
    parser.add_argument("--greek-absolute-tolerance", type=float, default=0.01)
    return parser.parse_args()


def read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if "=" in raw_line:
            key, value = raw_line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def run_backend(args: argparse.Namespace, backend: str) -> dict[str, str]:
    """执行同一输入，仅切换 backend，确保比较边界一致。"""
    result = args.output_dir / f"{backend}.result.txt"
    performance = args.output_dir / f"{backend}.performance.log"
    subprocess.run(
        [
            str(args.binary), "--option", str(args.option),
            "--simulation", str(args.simulation), "--backend", backend,
            "--reference-paths", "0", "--output", str(result),
            "--performance", str(performance),
        ],
        check=True,
    )
    values = read_key_values(result)
    values.update(read_key_values(performance))
    return values


def finite_value(values: dict[str, str], key: str) -> float | None:
    """读取可选 reference；nan 表示该模型没有独立解析/树基准。"""
    try:
        value = float(values[key])
    except (KeyError, ValueError):
        return None
    return value if math.isfinite(value) else None


def reference_gate(
    *,
    cpu: dict[str, str],
    gpu: dict[str, str],
    key: str,
    cpu_value: float,
    gpu_value: float,
    cpu_tolerance: float,
    gpu_tolerance: float,
) -> tuple[dict[str, str], bool]:
    """同时约束两端与同一独立 reference，避免共同偏差通过 parity gate。"""
    cpu_reference = finite_value(cpu, key)
    gpu_reference = finite_value(gpu, key)
    if cpu_reference is None and gpu_reference is None:
        return {
            "reference": "",
            "reference_method": "not_available",
            "cpu_reference_error": "",
            "gpu_reference_error": "",
            "cpu_reference_tolerance": "",
            "gpu_reference_tolerance": "",
            "reference_passed": "n/a",
        }, True

    reference = cpu_reference if cpu_reference is not None else gpu_reference
    assert reference is not None
    references_match = (
        cpu_reference is not None
        and gpu_reference is not None
        and math.isclose(cpu_reference, gpu_reference, rel_tol=0.0, abs_tol=1.0e-12)
    )
    cpu_error = (
        abs(cpu_value - reference) if cpu_reference is not None else math.inf
    )
    gpu_error = (
        abs(gpu_value - reference) if gpu_reference is not None else math.inf
    )
    passed = (
        references_match
        and cpu_error <= cpu_tolerance
        and gpu_error <= gpu_tolerance
    )
    method_key = (
        "reference_method"
        if key == "reference_price"
        else "reference_greeks_method"
    )
    method = cpu.get(method_key, "")
    return {
        "reference": f"{reference:.17g}",
        "reference_method": method,
        "cpu_reference_error": f"{cpu_error:.17g}",
        "gpu_reference_error": f"{gpu_error:.17g}",
        "cpu_reference_tolerance": f"{cpu_tolerance:.17g}",
        "gpu_reference_tolerance": f"{gpu_tolerance:.17g}",
        "reference_passed": str(passed).lower(),
    }, passed


def main() -> None:
    args = parse_args()
    if (
        args.z_score <= 0.0
        or args.price_absolute_tolerance < 0.0
        or args.greek_relative_tolerance < 0.0
        or args.greek_absolute_tolerance < 0.0
    ):
        raise ValueError("comparison tolerances must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cpu = run_backend(args, "cpu")
    gpu = run_backend(args, "cuda")

    price_difference = abs(float(cpu["price"]) - float(gpu["price"]))
    combined_standard_error = math.hypot(
        float(cpu["standard_error"]), float(gpu["standard_error"])
    )
    price_tolerance = (
        args.z_score * combined_standard_error + args.price_absolute_tolerance
    )
    cpu_price = float(cpu["price"])
    gpu_price = float(gpu["price"])
    cpu_price_reference_tolerance = (
        args.z_score * float(cpu["standard_error"])
        + args.price_absolute_tolerance
    )
    gpu_price_reference_tolerance = (
        args.z_score * float(gpu["standard_error"])
        + args.price_absolute_tolerance
    )
    price_reference, price_reference_passed = reference_gate(
        cpu=cpu,
        gpu=gpu,
        key="reference_price",
        cpu_value=cpu_price,
        gpu_value=gpu_price,
        cpu_tolerance=cpu_price_reference_tolerance,
        gpu_tolerance=gpu_price_reference_tolerance,
    )
    price_backend_passed = price_difference <= price_tolerance
    rows: list[dict[str, str]] = [
        {
            "metric": "price",
            "cpu": cpu["price"],
            "gpu": gpu["price"],
            "absolute_difference": f"{price_difference:.17g}",
            "tolerance": f"{price_tolerance:.17g}",
            "backend_passed": str(price_backend_passed).lower(),
            **price_reference,
            "passed": str(price_backend_passed and price_reference_passed).lower(),
            "rule": "backend: joint SE; reference: per-backend SE envelope",
        }
    ]

    if cpu.get("greeks_computed") == "true" and gpu.get("greeks_computed") == "true":
        for metric in ("delta", "gamma", "vega"):
            cpu_value = float(cpu[metric])
            gpu_value = float(gpu[metric])
            difference = abs(cpu_value - gpu_value)
            tolerance = max(
                args.greek_absolute_tolerance,
                args.greek_relative_tolerance * max(abs(cpu_value), abs(gpu_value)),
            )
            reference_value = finite_value(cpu, f"reference_{metric}")
            reference_scale = abs(reference_value) if reference_value is not None else 0.0
            cpu_reference_tolerance = max(
                args.greek_absolute_tolerance,
                args.greek_relative_tolerance * max(abs(cpu_value), reference_scale),
            )
            gpu_reference_tolerance = max(
                args.greek_absolute_tolerance,
                args.greek_relative_tolerance * max(abs(gpu_value), reference_scale),
            )
            reference, reference_passed = reference_gate(
                cpu=cpu,
                gpu=gpu,
                key=f"reference_{metric}",
                cpu_value=cpu_value,
                gpu_value=gpu_value,
                cpu_tolerance=cpu_reference_tolerance,
                gpu_tolerance=gpu_reference_tolerance,
            )
            backend_passed = difference <= tolerance
            rows.append(
                {
                    "metric": metric,
                    "cpu": f"{cpu_value:.17g}",
                    "gpu": f"{gpu_value:.17g}",
                    "absolute_difference": f"{difference:.17g}",
                    "tolerance": f"{tolerance:.17g}",
                    "backend_passed": str(backend_passed).lower(),
                    **reference,
                    "passed": str(backend_passed and reference_passed).lower(),
                    "rule": "backend/reference: max(absolute, relative*scale)",
                }
            )

    comparison_path = args.output_dir / "comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    failed = [row["metric"] for row in rows if row["passed"] != "true"]
    if failed:
        raise SystemExit("CPU/GPU comparison failed: " + ", ".join(failed))
    print("CPU/GPU comparison passed: " + ", ".join(row["metric"] for row in rows))


if __name__ == "__main__":
    main()

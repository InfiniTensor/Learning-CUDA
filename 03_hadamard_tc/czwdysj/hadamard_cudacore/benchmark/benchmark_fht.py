#!/usr/bin/env python3
"""Same-process version benchmark: v0/v1/v2, Dao FHT, and torch.clone."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import DEFAULT_LIBRARY, VERSION_IDS, HadamardLibrary, run_correctness_gate, torch_dtype


REPRESENTATIVE_SHAPES = ((32, 128), (32, 256), (32, 512), (32, 1024), (32, 2048), (32, 4096), (32, 8192), (1, 1024), (128, 1024), (1024, 1024), (4096, 1024), (4096, 8192))
DEFAULT_BATCHES, DEFAULT_DIMS = (1, 32, 128, 1024, 4096), (128, 256, 512, 1024, 2048, 4096, 8192)
IMPLEMENTATIONS = ("v0", "v1", "v2", "dao", "clone")
FIELDS = ("measurement_context", "trial", "order_index", "implementation", "kernel_version", "batch", "head_dim", "dtype", "latency_us", "latency_min_us", "latency_max_us", "variability_pct", "stability", "speedup_vs_v0", "speedup_vs_naive", "speed_vs_dao", "latency_ratio_vs_dao", "percent_latency_gap", "ratio_vs_clone", "comparison_valid", "correct", "max_abs_error", "max_relative_error", "my_vs_reference_max_abs_error", "my_vs_reference_max_relative_error", "dao_vs_reference_max_abs_error", "dao_vs_reference_max_relative_error", "my_vs_dao_max_abs_error", "my_vs_dao_max_relative_error")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", choices=("representative", "full"), default="representative")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32", "all"), default="all")
    parser.add_argument("--implementation", choices=(*IMPLEMENTATIONS, "all"), default="all")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--raw-csv", type=Path, default=Path("results/same_process_raw.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("results/same_process_summary.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("results/metadata.json"))
    parser.add_argument("--dao-commit", default="unknown")
    args = parser.parse_args()
    if args.warmup <= 0 or args.repeat <= 0 or args.trials <= 0:
        parser.error("warmup, repeat and trials must be positive")
    if args.profile:
        if args.batch is None or args.dim is None or args.implementation == "all":
            parser.error("--profile requires --batch, --dim and one --implementation")
        args.repeat, args.trials = 3, 1
    return args


def shapes(args: argparse.Namespace) -> tuple[tuple[int, int], ...]:
    if args.batch is not None or args.dim is not None:
        if args.batch is None or args.dim is None:
            raise SystemExit("--batch and --dim must be supplied together")
        return ((args.batch, args.dim),)
    if args.matrix == "representative":
        return REPRESENTATIVE_SHAPES
    return tuple((batch, dim) for batch in DEFAULT_BATCHES for dim in DEFAULT_DIMS)


def event_latency_us(torch: Any, operation: Callable[[], Any], warmup: int, repeat: int, profile: bool) -> float:
    for _ in range(warmup): operation()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    if profile: torch.cuda.cudart().cudaProfilerStart()
    start.record()
    for _ in range(repeat): operation()
    stop.record()
    stop.synchronize()
    if profile: torch.cuda.cudart().cudaProfilerStop()
    return start.elapsed_time(stop) * 1000.0 / repeat


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def metrics(gate: Any, implementation: str) -> dict[str, object]:
    values: dict[str, object] = {
        "comparison_valid": "", "correct": "", "max_abs_error": "", "max_relative_error": "",
        "my_vs_reference_max_abs_error": "", "my_vs_reference_max_relative_error": "",
        "dao_vs_reference_max_abs_error": "", "dao_vs_reference_max_relative_error": "",
        "my_vs_dao_max_abs_error": "", "my_vs_dao_max_relative_error": "",
    }
    if implementation in VERSION_IDS:
        values.update({
            "comparison_valid": str(gate.passed).lower(), "correct": str(gate.version_vs_reference.passed).lower(),
            "max_abs_error": f"{gate.version_vs_reference.max_abs_error:.10g}", "max_relative_error": f"{gate.version_vs_reference.max_relative_error:.10g}",
            "my_vs_reference_max_abs_error": f"{gate.version_vs_reference.max_abs_error:.10g}", "my_vs_reference_max_relative_error": f"{gate.version_vs_reference.max_relative_error:.10g}",
            "dao_vs_reference_max_abs_error": f"{gate.dao_vs_reference.max_abs_error:.10g}", "dao_vs_reference_max_relative_error": f"{gate.dao_vs_reference.max_relative_error:.10g}",
            "my_vs_dao_max_abs_error": f"{gate.version_vs_dao.max_abs_error:.10g}", "my_vs_dao_max_relative_error": f"{gate.version_vs_dao.max_relative_error:.10g}",
        })
    elif implementation == "dao":
        values.update({"comparison_valid": str(gate.dao_vs_reference.passed).lower(), "correct": str(gate.dao_vs_reference.passed).lower(), "max_abs_error": f"{gate.dao_vs_reference.max_abs_error:.10g}", "max_relative_error": f"{gate.dao_vs_reference.max_relative_error:.10g}"})
    return values


def summarize(raw: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for row in raw:
        groups.setdefault((str(row["implementation"]), str(row["batch"]), str(row["head_dim"]), str(row["dtype"])), []).append(row)
    summary: list[dict[str, object]] = []
    by_case: dict[tuple[str, str, str], dict[str, dict[str, object]]] = {}
    for (implementation, batch, dim, dtype), rows in groups.items():
        samples = [float(row["latency_us"]) for row in rows]
        median, low, high = statistics.median(samples), min(samples), max(samples)
        row = {field: "" for field in FIELDS}
        row.update(rows[0]); row.update({"trial": len(samples), "order_index": "", "latency_us": f"{median:.10g}", "latency_min_us": f"{low:.10g}", "latency_max_us": f"{high:.10g}", "variability_pct": f"{(high-low)/median*100:.10g}", "stability": "UNSTABLE" if (high-low)/median > 0.05 else "stable"})
        summary.append(row); by_case.setdefault((batch, dim, dtype), {})[implementation] = row
    for implementations in by_case.values():
        required = ("v0", "dao", "clone")
        if not all(key in implementations for key in required): continue
        v0_latency, dao_latency, clone_latency = (float(implementations[key]["latency_us"]) for key in required)
        for version in VERSION_IDS:
            row = implementations.get(version)
            if row is None or row["comparison_valid"] != "true": continue
            latency = float(row["latency_us"])
            row.update({"speedup_vs_v0": f"{v0_latency/latency:.10g}", "speedup_vs_naive": f"{v0_latency/latency:.10g}", "speed_vs_dao": f"{dao_latency/latency:.10g}", "latency_ratio_vs_dao": f"{latency/dao_latency:.10g}", "percent_latency_gap": f"{(latency/dao_latency-1)*100:.10g}", "ratio_vs_clone": f"{latency/clone_latency:.10g}"})
    return sorted(summary, key=lambda row: (row["dtype"], int(row["batch"]), int(row["head_dim"]), row["implementation"]))


def main() -> int:
    args = parse_args()
    try:
        import torch
        from fast_hadamard_transform import hadamard_transform
    except ImportError as error:
        raise SystemExit("PyTorch and fast_hadamard_transform are required; run ./scripts/bootstrap_env.sh") from error
    if not torch.cuda.is_available(): raise SystemExit("torch.cuda is unavailable")
    selected = IMPLEMENTATIONS if args.implementation == "all" else (args.implementation,)
    dtypes = ("fp32", "fp16", "bf16") if args.dtype == "all" else (args.dtype,)
    library, raw = HadamardLibrary(torch, args.library), []
    metadata = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "gpu": torch.cuda.get_device_name(0), "compute_capability": list(torch.cuda.get_device_capability(0)), "torch": torch.__version__, "torch_cuda": torch.version.cuda, "dao_commit": args.dao_commit, "matrix": args.matrix, "shapes": [list(item) for item in shapes(args)], "dtypes": list(dtypes), "warmup": args.warmup, "repeat": args.repeat, "trials": args.trials, "implementations": list(selected), "mathematical_contract": "y = x @ H(D) / sqrt(D), Sylvester/Walsh order"}
    args.metadata.parent.mkdir(parents=True, exist_ok=True); args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"GPU: {metadata['gpu']}, SM {tuple(metadata['compute_capability'])}")
    for dtype_name in dtypes:
        for batch, dim in shapes(args):
            if batch <= 0 or dim < 32 or dim > 8192 or dim & (dim - 1):
                print(f"Shape [{batch}, {dim}] unsupported"); continue
            x = torch.randn((batch, dim), device="cuda", dtype=torch_dtype(torch, dtype_name)).contiguous()
            prepared = {version: library.prepare(version, x, dtype_name) for version in VERSION_IDS}
            gates = {version: run_correctness_gate(torch, library, hadamard_transform, version, batch, dim, dtype_name) for version in VERSION_IDS}
            operations: dict[str, Callable[[], Any]] = {**{version: prepared[version].launch for version in VERSION_IDS}, "dao": lambda: hadamard_transform(x, scale=1.0/math.sqrt(dim)), "clone": lambda: x.clone(memory_format=torch.contiguous_format)}
            print(f"Shape [{batch}, {dim}] {dtype_name.upper()}: " + ", ".join(f"{version}={'PASS' if gate.passed else 'FAILED'}" for version, gate in gates.items()))
            for trial in range(args.trials):
                rotation = list(IMPLEMENTATIONS[trial % len(IMPLEMENTATIONS):] + IMPLEMENTATIONS[:trial % len(IMPLEMENTATIONS)])
                for order_index, implementation in enumerate(rotation):
                    if implementation not in selected: continue
                    latency = event_latency_us(torch, operations[implementation], args.warmup, args.repeat, args.profile)
                    gate = gates[implementation] if implementation in VERSION_IDS else gates["v0"]
                    raw.append({"measurement_context": "pytorch_same_process", "trial": trial + 1, "order_index": order_index, "implementation": f"hadamard_{implementation}" if implementation in VERSION_IDS else ("dao_fht" if implementation == "dao" else "torch_clone"), "kernel_version": implementation if implementation in VERSION_IDS else "", "batch": batch, "head_dim": dim, "dtype": dtype_name.upper(), "latency_us": f"{latency:.10g}", **metrics(gate, implementation)})
    write_csv(args.raw_csv, raw)
    summary = summarize([{**row, "implementation": str(row["kernel_version"]) if row["kernel_version"] else ("dao" if row["implementation"] == "dao_fht" else "clone")} for row in raw])
    for row in summary:
        if row["implementation"] in VERSION_IDS: row["implementation"] = f"hadamard_{row['implementation']}"
        elif row["implementation"] == "dao": row["implementation"] = "dao_fht"
        else: row["implementation"] = "torch_clone"
    write_csv(args.summary_csv, summary)
    print(f"same-process raw: {args.raw_csv}\nsame-process summary: {args.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

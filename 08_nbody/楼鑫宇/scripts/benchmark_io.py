#!/usr/bin/env python3
"""Run serial, cold-process trajectory-output cost experiments."""

import argparse
import csv
import json
import statistics
import subprocess
import time
from pathlib import Path


CASES = (
    ("record_off", "configs/cpu_4096.cfg", "off"),
    ("record_k1", "configs/io_k1.cfg", "on"),
    ("record_k10", "configs/cpu_4096.cfg", "on"),
    ("record_k100", "configs/io_k100.cfg", "on"),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("repeat must be positive")

    project = args.project_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    output_root.mkdir(parents=True)

    raw_rows = []
    for case_name, config_name, record_mode in CASES:
        for repetition in range(1, args.repeat + 1):
            run_dir = output_root / f"{case_name}_r{repetition}"
            command = [
                str(project / "build/nbody"),
                "--backend", "cuda-tiled",
                "--block-size", str(args.block_size),
                "--record", record_mode,
                "--diagnostics", "off",
                "--input", str(project / "data/cluster_4096.txt"),
                "--config", str(project / config_name),
                "--output", str(run_dir),
            ]
            begin = time.perf_counter()
            completed = subprocess.run(command, text=True, capture_output=True)
            end_to_end_ms = (time.perf_counter() - begin) * 1000.0
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "driver_stdout.txt").write_text(
                completed.stdout, encoding="utf-8"
            )
            (run_dir / "driver_stderr.txt").write_text(
                completed.stderr, encoding="utf-8"
            )
            row = {
                "case": case_name,
                "record_mode": record_mode,
                "record_interval": 0 if record_mode == "off" else int(config_name.split("k")[-1].split(".")[0]) if "io_k" in config_name else 10,
                "repetition": repetition,
                "exit_code": completed.returncode,
                "end_to_end_ms": end_to_end_ms,
                "output_dir": str(run_dir),
            }
            performance_path = run_dir / "performance.json"
            if completed.returncode == 0 and performance_path.exists():
                performance = json.loads(performance_path.read_text(encoding="utf-8"))
                row.update(
                    simulation_wall_ms=performance["simulation_wall_ms"],
                    force_total_ms=performance["force_total_ms"],
                    trajectory_bytes=(run_dir / "trajectory.bin").stat().st_size
                    if (run_dir / "trajectory.bin").exists() else 0,
                )
            else:
                row.update(
                    simulation_wall_ms="",
                    force_total_ms="",
                    trajectory_bytes="",
                )
            raw_rows.append(row)
            print(case_name, repetition, completed.returncode, f"{end_to_end_ms:.3f} ms")

    raw_fields = list(raw_rows[0])
    with (output_root / "raw.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(raw_rows)

    summaries = []
    for case_name, _, record_mode in CASES:
        successful = [
            row for row in raw_rows
            if row["case"] == case_name and row["exit_code"] == 0
        ]
        if not successful:
            continue
        simulation = [float(row["simulation_wall_ms"]) for row in successful]
        end_to_end = [float(row["end_to_end_ms"]) for row in successful]
        summaries.append(
            {
                "case": case_name,
                "record_mode": record_mode,
                "record_interval": successful[0]["record_interval"],
                "successful_runs": len(successful),
                "simulation_min_ms": min(simulation),
                "simulation_median_ms": statistics.median(simulation),
                "simulation_max_ms": max(simulation),
                "end_to_end_min_ms": min(end_to_end),
                "end_to_end_median_ms": statistics.median(end_to_end),
                "end_to_end_max_ms": max(end_to_end),
                "trajectory_bytes": successful[0]["trajectory_bytes"],
            }
        )

    summary_fields = list(summaries[0])
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)
    (output_root / "summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

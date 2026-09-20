"""Capture and compare tiled and naive CUDA kernels with Nsight Systems."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sqlite3
import sys

from verify_nsys import acceleration_summary, verify_report


def sha256(path: Path) -> str:
    """Hash inputs and potentially large traces without loading them into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_for(nsys: Path, executable: Path, input_file: Path, config: Path,
                output: Path, kernel: str) -> list[str]:
    return [str(nsys), "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
            "--force-overwrite=true", f"--output={output}", str(executable),
            "--input", str(input_file), "--config", str(config), "--backend", "cuda",
            "--kernel", kernel, "--cuda-execution", "direct", "--no-trajectory"]


def run_logged(command: list[str], log: Path, cwd: Path) -> None:
    """Keep tool diagnostics even when the child fails; do not buffer large output."""
    with log.open("xb") as output:
        try:
            subprocess.run(command, check=True, cwd=cwd, stdout=output, stderr=subprocess.STDOUT)
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError(f"Profiler command failed; see {log}: {error}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsys", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory for this capture; existing captures are never replaced.")
    args = parser.parse_args()
    inputs = (args.nsys, args.executable, args.input, args.config)
    absent = [str(path) for path in inputs if not path.is_file()]
    if absent:
        parser.error("Required file missing: " + ", ".join(absent))
    if args.output_dir.exists():
        parser.error("Choose a new output directory; prior captures are preserved")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True)
    args.nsys, args.executable, args.input, args.config = (path.resolve() for path in inputs)
    provenance = {name: {"path": str(path), "sha256": sha256(path)}
                  for name, path in (("nsys", args.nsys), ("executable", args.executable),
                                     ("input", args.input), ("config", args.config),
                                     ("capture_script", Path(__file__).resolve()),
                                     ("analysis_script", Path(__file__).resolve().with_name("verify_nsys.py")))}
    results = {}
    for kernel in ("tiled", "naive"):
        prefix = args.output_dir / kernel
        run_directory = args.output_dir / f"{kernel}-run"
        run_directory.mkdir()
        command = command_for(args.nsys, args.executable, args.input, args.config, prefix, kernel)
        print(f"Capturing {kernel}; log: {prefix.with_suffix('.profile.log')}", flush=True)
        run_logged(command, prefix.with_suffix(".profile.log"), run_directory)
        sqlite_path = prefix.with_suffix(".sqlite")
        if not sqlite_path.is_file():
            run_logged([str(args.nsys), "export", "--type", "sqlite",
                        "--output", str(sqlite_path), str(prefix.with_suffix(".nsys-rep"))],
                       prefix.with_suffix(".export.log"), run_directory)
        results[kernel] = {"sqlite": str(sqlite_path), "sqlite_sha256": sha256(sqlite_path),
                           "command": command, "working_directory": str(run_directory),
                           "rows": verify_report(sqlite_path),
                           "acceleration": acceleration_summary(sqlite_path, expected_kernel=kernel)}
    for item in provenance.values():
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("Profiling input changed during capture: " + item["path"])
    if results["naive"]["acceleration"]["instances"] != results["tiled"]["acceleration"]["instances"]:
        raise ValueError("Acceleration sample counts differ; refusing to compare incomplete captures")
    ratio = results["naive"]["acceleration"]["average_ns"] / results["tiled"]["acceleration"]["average_ns"]
    report = {"format": "nbody-nsys-profile-v1", "tool": str(args.nsys),
              "provenance": provenance,
              "workload": {"input": str(args.input), "config": str(args.config),
                           "backend": "cuda", "cuda_execution": "direct", "trajectory": False},
              "tiled": results["tiled"], "naive": results["naive"],
              "naive_over_tiled_average_ratio": ratio}
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, sqlite3.Error, subprocess.CalledProcessError, ValueError) as error:
        print(f"Nsight Systems profiling failed: {error}", file=sys.stderr)
        raise SystemExit(1)

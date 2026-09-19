#!/usr/bin/env python3
"""Collect reproducible ncu/nsys profiles for exact or IVF search."""

from __future__ import annotations

import argparse
import platform
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", choices=("ncu", "nsys"), required=True)
    parser.add_argument("--binary", type=Path, default=Path("build/vector_search"))
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ncu-launch-count", type=int, default=10)
    return parser.parse_args()


def read_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"')
    return values


def command_output(command: list[str]) -> str:
    executable = shutil.which(command[0])
    if executable is None:
        return f"unavailable: {command[0]}"
    completed = subprocess.run(
        [executable, *command[1:]], capture_output=True, text=True, check=False
    )
    output = (completed.stdout or completed.stderr).strip()
    return (
        output
        if completed.returncode == 0
        else (f"failed ({completed.returncode}): {output}")
    )


def main() -> None:
    args = parse_args()
    if args.ncu_launch_count <= 0:
        raise ValueError("--ncu-launch-count must be positive")
    executable = shutil.which(args.tool)
    if executable is None:
        raise SystemExit(f"{args.tool} is not available on PATH")

    args.binary = args.binary.resolve()
    args.database = args.database.resolve()
    args.queries = args.queries.resolve()
    args.config = args.config.resolve()
    args.index = args.index.resolve() if args.index is not None else None
    args.output_dir = args.output_dir.resolve()
    for path in (args.binary, args.database, args.queries, args.config):
        if not path.is_file():
            raise SystemExit(f"required file does not exist: {path}")
    mode = read_config(args.config).get("search_mode")
    if mode == "ivf_flat" and (args.index is None or not args.index.is_file()):
        raise SystemExit("ivf_flat profiling requires an existing --index")
    if mode == "exact" and args.index is not None:
        raise SystemExit("exact profiling must not provide --index")
    if mode not in ("exact", "ivf_flat"):
        raise SystemExit(f"unsupported search mode: {mode!r}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    common = [
        str(args.binary),
        "search",
        "--database",
        str(args.database),
        "--queries",
        str(args.queries),
        "--config",
        str(args.config),
        "--backend",
        "cuda",
        "--output",
        str(args.output_dir / "profile.results.txt"),
        "--performance",
        str(args.output_dir / "profile.performance.log"),
        "--quality",
        str(args.output_dir / "profile.quality.log"),
    ]
    if args.index is not None:
        common.extend(["--index", str(args.index)])
    if args.tool == "ncu":
        # ncu 聚焦自研 score kernel；CUB 与 CPU/GPU 时间线由 nsys 捕获。
        command = [
            executable,
            "--set",
            "full",
            "--target-processes",
            "all",
            "--kernel-name-base",
            "demangled",
            "--kernel-name",
            "regex:.*score_candidates_kernel.*",
            "--launch-count",
            str(args.ncu_launch_count),
            "--force-overwrite",
            "--export",
            str(args.output_dir / f"{mode}_ncu"),
            *common,
        ]
    else:
        # nsys 保留完整搜索，便于定位 coarse CPU、传输、score、sort 和同步占比。
        command = [
            executable,
            "profile",
            "--stats=true",
            "--force-overwrite=true",
            "--trace=cuda,nvtx,osrt",
            "--output",
            str(args.output_dir / f"{mode}_nsys"),
            *common,
        ]

    (args.output_dir / "command.txt").write_text(
        shlex.join(command) + "\n", encoding="utf-8"
    )
    environment = {
        "utc_start": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        f"{args.tool}_version": command_output([args.tool, "--version"]),
    }
    (args.output_dir / "environment.txt").write_text(
        "".join(f"[{key}]\n{value}\n\n" for key, value in environment.items()),
        encoding="utf-8",
    )
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    console = completed.stdout + completed.stderr
    (args.output_dir / "console.log").write_text(console, encoding="utf-8")
    if console:
        print(console, end="" if console.endswith("\n") else "\n")
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
            stderr=completed.stderr,
        )


if __name__ == "__main__":
    main()

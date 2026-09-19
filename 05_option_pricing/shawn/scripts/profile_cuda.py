#!/usr/bin/env python3
"""Collect reproducible Nsight Compute or Nsight Systems profiles on the GPU server."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", choices=("ncu", "nsys"), required=True)
    parser.add_argument("--binary", type=Path, default=Path("build/derivatives_pricer"))
    parser.add_argument("--option", type=Path, required=True)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/profiles"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    executable = shutil.which(args.tool)
    if executable is None:
        raise SystemExit(f"{args.tool} is not available on PATH")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common = [
        str(args.binary), "--option", str(args.option),
        "--simulation", str(args.simulation), "--backend", "cuda",
        "--reference-paths", "0",
        "--output", str(args.output_dir / "profile.result.txt"),
        "--performance", str(args.output_dir / "profile.performance.log"),
    ]
    if args.tool == "ncu":
        command = [
            executable, "--set", "full", "--target-processes", "all",
            "--export", str(args.output_dir / "profile_ncu"), *common,
        ]
    else:
        command = [
            executable, "profile", "--stats=true", "--force-overwrite=true",
            "--output", str(args.output_dir / "profile_nsys"), *common,
        ]
    # 保存本次采集命令。
    (args.output_dir / "command.txt").write_text(
        " ".join(command) + "\n", encoding="utf-8"
    )
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

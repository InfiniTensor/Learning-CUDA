#!/usr/bin/env python3
"""Run the required CPU/GPU accuracy matrix on a CUDA server."""

from __future__ import annotations

import argparse
import csv
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build/derivatives_pricer"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gpu_validation"))
    parser.add_argument("--include-multi-gpu", action="store_true")
    return parser.parse_args()


def optional_command(command: list[str], cwd: Path | None = None) -> str:
    """执行环境查询命令，记录缺失工具。"""
    executable = shutil.which(command[0])
    if executable is None:
        return f"unavailable: {command[0]}"
    completed = subprocess.run(
        [executable, *command[1:]], capture_output=True, text=True, check=False,
        cwd=cwd,
    )
    output = (completed.stdout or completed.stderr).strip()
    return output if completed.returncode == 0 else (
        f"failed ({completed.returncode}): {output}"
    )


def cmake_cache_summary(binary: Path) -> str:
    """提取与 CUDA 构建可复现性有关的 CMake cache 字段。"""
    cache = binary.parent / "CMakeCache.txt"
    if not cache.is_file():
        return f"unavailable: {cache}"
    prefixes = (
        "CMAKE_BUILD_TYPE:", "CMAKE_CXX_COMPILER:",
        "CMAKE_CUDA_ARCHITECTURES:", "CMAKE_CUDA_COMPILER:",
        "CMAKE_CUDA_FLAGS:", "CUDAToolkit_",
    )
    lines = [
        line for line in cache.read_text(encoding="utf-8").splitlines()
        if line.startswith(prefixes)
    ]
    return "\n".join(lines) if lines else "no matching CMake cache fields"


def write_environment(output_dir: Path, root: Path, binary: Path) -> None:
    """保存服务器硬件与工具链信息。"""
    sections = {
        "git commit": optional_command(["git", "rev-parse", "HEAD"], cwd=root),
        "git status --porcelain": optional_command(
            ["git", "status", "--porcelain"], cwd=root
        ) or "clean",
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "nvidia-smi -L": optional_command(["nvidia-smi", "-L"]),
        "nvidia-smi inventory": optional_command(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "nvcc --version": optional_command(["nvcc", "--version"]),
        "cmake --version": optional_command(["cmake", "--version"]),
        "c++ --version": optional_command(["c++", "--version"]),
        "CMake cache": cmake_cache_summary(binary),
    }
    text = "".join(f"[{name}]\n{value}\n\n" for name, value in sections.items())
    (output_dir / "environment.txt").write_text(text, encoding="utf-8")


def markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def write_report_table(rows: list[dict[str, str]], output_path: Path) -> None:
    """将验证结果转换为 Markdown 表格。"""
    header = (
        "| Case | Option config | Simulation config | Metric | CPU | GPU | "
        "Abs. diff | Backend tol. | Reference | CPU ref. err. | GPU ref. err. | "
        "Ref. pass | Overall pass |\n"
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|\n"
    )
    lines = [header]
    for row in rows:
        fields = [
            row["case"], row["option_config"], row["simulation_config"],
            row["metric"], row["cpu"], row["gpu"],
            row["absolute_difference"], row["tolerance"],
            row.get("reference", ""), row.get("cpu_reference_error", ""),
            row.get("gpu_reference_error", ""),
            row.get("reference_passed", "n/a"), row["passed"],
        ]
        lines.append("| " + " | ".join(markdown_cell(value) for value in fields) + " |\n")
    output_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    binary = args.binary.resolve()
    if not binary.is_file():
        raise SystemExit(f"binary does not exist: {binary}")
    compare = root / "scripts" / "compare_cpu_gpu.py"
    cases = [
        ("gbm_pseudo", "european_call.txt", "simulation_advanced_smoke.txt"),
        ("gbm_qmc", "european_call.txt", "simulation_qmc_smoke.txt"),
        ("heston_pseudo", "heston_call.txt", "simulation_advanced_smoke.txt"),
        ("heston_qmc", "heston_call.txt", "simulation_qmc_smoke.txt"),
        ("local_vol_pseudo", "local_vol_call.txt", "simulation_advanced_smoke.txt"),
        ("local_vol_qmc", "local_vol_call.txt", "simulation_qmc_smoke.txt"),
        (
            "american_pseudo",
            "american_put.txt",
            "simulation_american_validation.txt",
        ),
        (
            "american_qmc",
            "american_put.txt",
            "simulation_american_qmc_validation.txt",
        ),
    ]
    if args.include_multi_gpu:
        cases.append(("multi_gpu", "european_call.txt", "simulation_multi_gpu.txt"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
    )
    write_environment(args.output_dir, root, binary)
    summary: list[dict[str, str]] = []
    failures: list[str] = []
    for name, option, simulation in cases:
        case_dir = args.output_dir / name
        case_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = case_dir / "comparison.csv"
        # 删除同目录的旧比较表，避免本次后端崩溃时误读历史结果。
        comparison_path.unlink(missing_ok=True)
        option_path = root / "configs" / option
        simulation_path = root / "configs" / simulation
        command = [
            sys.executable, str(compare), "--binary", str(binary),
            "--option", str(option_path),
            "--simulation", str(simulation_path),
            "--output-dir", str(case_dir),
        ]
        (case_dir / "command.txt").write_text(
            shlex.join(command) + "\n", encoding="utf-8"
        )
        (case_dir / "option_config.txt").write_text(
            option_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        (case_dir / "simulation_config.txt").write_text(
            simulation_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        completed = subprocess.run(
            command, check=False, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True,
        )
        console = completed.stdout or ""
        (case_dir / "console.log").write_text(console, encoding="utf-8")
        print(console, end="")
        if completed.returncode != 0:
            failures.append(name)
        if comparison_path.is_file():
            with comparison_path.open(encoding="utf-8") as stream:
                comparison_rows = list(csv.DictReader(stream))
        else:
            # 比较表未生成时也记录该案例的失败状态。
            comparison_rows = [
                {
                    "metric": "execution",
                    "cpu": "",
                    "gpu": "",
                    "absolute_difference": "",
                    "tolerance": "",
                    "backend_passed": "false",
                    "reference": "",
                    "reference_method": "not_available",
                    "cpu_reference_error": "",
                    "gpu_reference_error": "",
                    "cpu_reference_tolerance": "",
                    "gpu_reference_tolerance": "",
                    "reference_passed": "n/a",
                    "passed": "false",
                    "rule": f"compare_cpu_gpu exited with {completed.returncode}",
                }
            ]
        for row in comparison_rows:
            summary.append(
                {
                    "case": name,
                    "option_config": option,
                    "simulation_config": simulation,
                    **row,
                }
            )
    if not summary:
        raise RuntimeError("GPU validation matrix produced no comparison rows")
    with (args.output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    write_report_table(summary, args.output_dir / "report_table.md")
    artifacts = (
        f"artifacts: {args.output_dir / 'summary.csv'}, "
        f"{args.output_dir / 'environment.txt'}, "
        f"{args.output_dir / 'report_table.md'}"
    )
    if failures:
        raise SystemExit(
            f"{len(failures)} of {len(cases)} CPU/GPU validation cases failed: "
            f"{', '.join(failures)}; {artifacts}"
        )
    print(
        f"all {len(cases)} CPU/GPU validation cases passed; "
        f"{artifacts}"
    )


if __name__ == "__main__":
    main()

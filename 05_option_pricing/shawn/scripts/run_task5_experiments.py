#!/usr/bin/env python3
"""Run the reproducible Task 05 experiment suite and index every artifact."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Preset:
    paths: tuple[int, ...]
    qmc_paths: tuple[int, ...]
    replications: int
    asian_steps: int
    reference_paths: int
    performance_paths: int
    performance_replications: int
    cpu_benchmark_paths: int


@dataclass(frozen=True)
class Stage:
    name: str
    command: list[str]
    artifact_dir: Path


PRESETS = {
    # quick 用于流程检查，report 用于正式测量。
    "quick": Preset(
        paths=(4_096, 32_768),
        qmc_paths=(4_096, 32_768),
        replications=2,
        asian_steps=32,
        reference_paths=50_000,
        performance_paths=200_000,
        performance_replications=2,
        cpu_benchmark_paths=20_000,
    ),
    # report 覆盖四个数量级、独立 replications 和题目给出的 10M×256 规模。
    "report": Preset(
        paths=(10_000, 100_000, 1_000_000, 10_000_000),
        qmc_paths=(16_384, 65_536, 262_144, 1_048_576),
        replications=10,
        asian_steps=256,
        reference_paths=2_000_000,
        performance_paths=10_000_000,
        performance_replications=7,
        cpu_benchmark_paths=200_000,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary", type=Path, default=Path("build/derivatives_pricer")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/report_experiments")
    )
    parser.add_argument("--preset", choices=tuple(PRESETS), default="report")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--master-seed", type=int, default=20260824)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--include-barrier", action="store_true")
    parser.add_argument(
        "--profile-tools", nargs="*", choices=("ncu", "nsys"), default=[]
    )
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="仅供本地流水线调试；正式报告运行必须使用 clean commit",
    )
    return parser.parse_args()


def command_output(command: list[str], cwd: Path) -> str:
    """读取环境信息；工具缺失时返回 unavailable。"""
    executable = shutil.which(command[0])
    if executable is None:
        return f"unavailable: {command[0]}"
    completed = subprocess.run(
        [executable, *command[1:]], cwd=cwd, capture_output=True,
        text=True, check=False,
    )
    output = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0:
        return f"failed ({completed.returncode}): {output}"
    return output or "clean"


def write_environment(root: Path, binary: Path, output_dir: Path) -> str:
    """记录实验版本、硬件与工具链。"""
    git_status = command_output(["git", "status", "--porcelain"], root)
    sections = {
        "utc_start": datetime.now(timezone.utc).isoformat(),
        "git_commit": command_output(["git", "rev-parse", "HEAD"], root),
        "git_status_porcelain": git_status,
        "binary": str(binary),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "nvidia_smi_L": command_output(["nvidia-smi", "-L"], root),
        "nvidia_smi_inventory": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            root,
        ),
        "nvcc_version": command_output(["nvcc", "--version"], root),
        "cmake_version": command_output(["cmake", "--version"], root),
        "cxx_version": command_output(["c++", "--version"], root),
    }
    text = "".join(f"[{name}]\n{value}\n\n" for name, value in sections.items())
    (output_dir / "environment.txt").write_text(text, encoding="utf-8")
    return git_status


def sweep_command(
    *,
    python: str,
    script: Path,
    binary: Path,
    option: Path,
    output_dir: Path,
    backend: str,
    paths: tuple[int, ...],
    variance: tuple[str, ...],
    steps: int,
    master_seed: int,
    replications: int,
    rng: str,
    reference_paths: int,
    cpu_benchmark_paths: int,
    num_gpus: int,
) -> list[str]:
    """构造参数扫描命令。"""
    return [
        python, str(script), "--binary", str(binary),
        "--option", str(option), "--output-dir", str(output_dir),
        "--backend", backend, "--paths", *(str(value) for value in paths),
        "--variance", *variance, "--steps", str(steps),
        "--master-seed", str(master_seed),
        "--replications", str(replications), "--rng", rng,
        "--qmc-replications", "8", "--num-gpus", str(num_gpus),
        "--reference-paths", str(reference_paths),
        "--cpu-benchmark-paths", str(cpu_benchmark_paths),
    ]


def build_stages(args: argparse.Namespace, root: Path, preset: Preset) -> list[Stage]:
    """按预设生成固定实验计划。"""
    python = sys.executable
    scripts = root / "scripts"
    configs = root / "configs"
    output = args.output_dir
    stages: list[Stage] = []
    if not args.skip_validation:
        stages.append(
            Stage(
                "validation",
                [
                    python, str(scripts / "run_gpu_validation.py"),
                    "--binary", str(args.binary),
                    "--output-dir", str(output / "validation"),
                ],
                output / "validation",
            )
        )

    common = {
        "python": python,
        "script": scripts / "run_sweep.py",
        "binary": args.binary,
        "backend": args.backend,
        "master_seed": args.master_seed,
        "num_gpus": args.num_gpus,
    }
    stages.append(
        Stage(
            "european_mc",
            sweep_command(
                **common, option=configs / "european_call.txt",
                output_dir=output / "european_mc", paths=preset.paths,
                variance=("none", "antithetic", "control_variate"), steps=1,
                replications=preset.replications, rng="curand",
                reference_paths=0, cpu_benchmark_paths=0,
            ),
            output / "european_mc",
        )
    )
    stages.append(
        Stage(
            "asian_mc",
            sweep_command(
                **common, option=configs / "asian_call.txt",
                output_dir=output / "asian_mc", paths=preset.paths,
                variance=("none", "antithetic", "control_variate"),
                steps=preset.asian_steps, replications=preset.replications,
                rng="curand", reference_paths=preset.reference_paths,
                cpu_benchmark_paths=0,
            ),
            output / "asian_mc",
        )
    )
    if args.include_barrier:
        stages.append(
            Stage(
                "barrier_mc",
                sweep_command(
                    **common, option=configs / "barrier_call.txt",
                    output_dir=output / "barrier_mc", paths=preset.paths,
                    variance=("none", "antithetic", "control_variate"),
                    steps=preset.asian_steps, replications=preset.replications,
                    rng="curand", reference_paths=preset.reference_paths,
                    cpu_benchmark_paths=0,
                ),
                output / "barrier_mc",
            )
        )
    stages.append(
        Stage(
            "european_qmc",
            sweep_command(
                **common, option=configs / "european_call.txt",
                output_dir=output / "european_qmc", paths=preset.qmc_paths,
                variance=("none",), steps=1,
                replications=preset.replications, rng="halton",
                reference_paths=0, cpu_benchmark_paths=0,
            ),
            output / "european_qmc",
        )
    )
    stages.append(
        Stage(
            "asian_performance",
            sweep_command(
                **common, option=configs / "asian_call.txt",
                output_dir=output / "asian_performance",
                paths=(preset.performance_paths,),
                variance=("control_variate",), steps=256,
                replications=preset.performance_replications, rng="curand",
                reference_paths=0,
                cpu_benchmark_paths=preset.cpu_benchmark_paths,
            ),
            output / "asian_performance",
        )
    )
    plot_sources = [
        f"european_mc={output / 'european_mc' / 'aggregate.csv'}",
        f"asian_mc={output / 'asian_mc' / 'aggregate.csv'}",
        f"european_qmc={output / 'european_qmc' / 'aggregate.csv'}",
    ]
    if args.include_barrier:
        plot_sources.append(
            f"barrier_mc={output / 'barrier_mc' / 'aggregate.csv'}"
        )
    plot_dir = output / "plots"
    plot_command = [
        python, str(scripts / "generate_report_plots.py"),
        "--output-dir", str(plot_dir),
    ]
    for source in plot_sources:
        plot_command.extend(["--aggregate", source])
    stages.append(Stage("report_plots", plot_command, plot_dir))
    for tool in args.profile_tools:
        profile_dir = output / f"profile_{tool}"
        stages.append(
            Stage(
                f"profile_{tool}",
                [
                    python, str(scripts / "profile_cuda.py"), "--tool", tool,
                    "--binary", str(args.binary),
                    "--option", str(configs / "asian_call.txt"),
                    "--simulation", str(configs / "simulation_profile.txt"),
                    "--output-dir", str(profile_dir),
                ],
                profile_dir,
            )
        )
    return stages


def write_manifest(output_dir: Path, records: list[dict[str, object]]) -> None:
    (output_dir / "manifest.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_artifact_index(output_dir: Path, records: list[dict[str, object]]) -> None:
    """生成阶段索引及合并结果表。"""
    lines = [
        "# Task 05 Experiment Artifacts\n\n",
        "最终结论仅允许引用 `status=passed` 阶段的原始文件。\n\n",
        "| Stage | Status | Seconds | Console | Artifact directory |\n",
        "|---|---|---:|---|---|\n",
    ]
    report_lines = [
        "# Task 05 Report Tables\n\n",
        "本文件由实验驱动器合并生成；数值精度以相邻 CSV 为准。\n",
    ]
    for record in records:
        stage = str(record["stage"])
        artifact = Path(str(record["artifact_dir"]))
        try:
            artifact_rel = artifact.relative_to(output_dir)
        except ValueError:
            artifact_rel = artifact
        log_rel = Path("logs") / f"{stage}.log"
        lines.append(
            f"| {stage} | {record['status']} | {float(record['elapsed_seconds']):.3f} "
            f"| `{log_rel}` | `{artifact_rel}` |\n"
        )
        table = artifact / "report_table.md"
        if table.is_file():
            report_lines.extend(
                [f"\n## {stage}\n\n", table.read_text(encoding="utf-8"), "\n"]
            )
        plot_index = artifact / "plots_index.md"
        if plot_index.is_file():
            plot_text = plot_index.read_text(encoding="utf-8").replace(
                "](", f"]({artifact_rel}/"
            )
            report_lines.extend(["\n", plot_text, "\n"])
    (output_dir / "artifact_index.md").write_text("".join(lines), encoding="utf-8")
    (output_dir / "report_tables.md").write_text(
        "".join(report_lines), encoding="utf-8"
    )


def run_stage(root: Path, output_dir: Path, stage: Stage) -> dict[str, object]:
    """流式显示并同步保存 stdout/stderr；失败阶段保留非零状态和完整日志。"""
    log_path = output_dir / "logs" / f"{stage.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[{stage.name}] {shlex.join(stage.command)}", flush=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            stage.command, cwd=root, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        return_code = process.wait()
    elapsed = time.monotonic() - started
    return {
        "stage": stage.name,
        "status": "passed" if return_code == 0 else "failed",
        "return_code": return_code,
        "elapsed_seconds": elapsed,
        "command": shlex.join(stage.command),
        "console_log": str(log_path),
        "artifact_dir": str(stage.artifact_dir),
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    args.binary = args.binary.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.binary.is_file():
        raise SystemExit(f"binary does not exist: {args.binary}")
    if args.num_gpus <= 0:
        raise SystemExit("--num-gpus must be positive")
    if args.backend != "cuda" and not args.skip_validation:
        raise SystemExit("CPU pipeline checks must add --skip-validation")
    if args.backend != "cuda" and args.profile_tools:
        raise SystemExit("profiling requires --backend cuda")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(
            f"output directory is not empty; choose a new evidence directory: "
            f"{args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    git_status = write_environment(root, args.binary, args.output_dir)
    if git_status != "clean" and not args.allow_dirty:
        raise SystemExit(
            "git worktree is not clean; commit the exact code first or use "
            "--allow-dirty only for a non-report local check"
        )

    preset = PRESETS[args.preset]
    stages = build_stages(args, root, preset)
    plan = {
        "preset": args.preset,
        "preset_values": asdict(preset),
        "backend": args.backend,
        "master_seed": args.master_seed,
        "num_gpus": args.num_gpus,
        "include_barrier": args.include_barrier,
        "profile_tools": args.profile_tools,
        "stages": [
            {"name": stage.name, "command": shlex.join(stage.command)}
            for stage in stages
        ],
    }
    (args.output_dir / "experiment_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    records: list[dict[str, object]] = []
    for stage in stages:
        record = run_stage(root, args.output_dir, stage)
        records.append(record)
        write_manifest(args.output_dir, records)
        write_artifact_index(args.output_dir, records)
        if record["status"] != "passed":
            raise SystemExit(
                f"stage failed: {stage.name}; inspect {record['console_log']}"
            )
    print(
        f"all {len(stages)} stages passed; report assets: "
        f"{args.output_dir / 'report_tables.md'}"
    )


if __name__ == "__main__":
    main()

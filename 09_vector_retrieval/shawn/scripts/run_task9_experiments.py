#!/usr/bin/env python3
"""Run the end-to-end reproducible Task 09 GPU experiment suite."""

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
    num_vectors: int
    num_queries: int
    dim: int
    clusters: int
    nlist: int
    training_samples: int
    quality_queries: int
    exact_repetitions: int
    ivf_repetitions: int
    warmup: int
    nprobe_values: tuple[int, ...]
    batch_values: tuple[int, ...]


@dataclass(frozen=True)
class Stage:
    name: str
    command: list[str]
    artifact_dir: Path
    expected_files: tuple[Path, ...]


PRESETS = {
    # quick 用于流程检查，report 用于正式测量。
    "quick": Preset(
        num_vectors=4_096,
        num_queries=32,
        dim=64,
        clusters=32,
        nlist=32,
        training_samples=2_048,
        quality_queries=10,
        exact_repetitions=2,
        ivf_repetitions=2,
        warmup=1,
        nprobe_values=(1, 4, 16, 32),
        batch_values=(1, 17, 32),
    ),
    # report 按题目规模扫描 K、nprobe 和 batch_size。
    "report": Preset(
        num_vectors=1_000_000,
        num_queries=1_000,
        dim=128,
        clusters=4_096,
        nlist=4_096,
        training_samples=65_536,
        quality_queries=20,
        exact_repetitions=3,
        ivf_repetitions=5,
        warmup=1,
        nprobe_values=(1, 4, 16, 64, 256),
        batch_values=(1, 32, 128),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build/vector_search"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task9_report"))
    parser.add_argument("--preset", choices=tuple(PRESETS), default="report")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--master-seed", type=int, default=20260910)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--profile-tools", nargs="*", choices=("ncu", "nsys"), default=[]
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="仅供流水线调试；正式报告运行必须使用 clean commit",
    )
    return parser.parse_args()


def command_output(command: list[str], cwd: Path) -> str:
    executable = shutil.which(command[0])
    if executable is None:
        return f"unavailable: {command[0]}"
    completed = subprocess.run(
        [executable, *command[1:]],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0:
        return f"failed ({completed.returncode}): {output}"
    return output or "clean"


def write_environment(
    root: Path, binary: Path, output_dir: Path, filename: str = "environment.txt"
) -> str:
    """记录代码版本与运行环境；正式运行要求工作区干净。"""
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
    (output_dir / filename).write_text(
        "".join(f"[{name}]\n{value}\n\n" for name, value in sections.items()),
        encoding="utf-8",
    )
    return git_status


def write_config(
    path: Path,
    *,
    mode: str,
    preset: Preset,
    quality_queries: int,
    top_k: int = 10,
    batch_size: int = 128,
    nprobe: int = 16,
    seed: int,
) -> None:
    """生成建库、搜索与性能分析共用的配置。"""
    values = {
        "top_k": top_k,
        "search_mode": mode,
        "batch_size": batch_size,
        "nlist": preset.nlist,
        "nprobe": min(nprobe, preset.nlist),
        "pq_m": 16,
        "kmeans_iterations": 2,
        "training_samples": preset.training_samples,
        "quality_queries": quality_queries,
        "seed": seed,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{key} = {value}\n" for key, value in values.items()),
        encoding="utf-8",
    )


def sweep_command(
    *,
    python: str,
    script: Path,
    binary: Path,
    database: Path,
    queries: Path,
    config: Path,
    output: Path,
    preset: Preset,
    top_k: tuple[int, ...],
    nprobe: tuple[int, ...],
    batch_size: tuple[int, ...],
    repetitions: int,
    index: Path | None,
    backend: str,
) -> list[str]:
    command = [
        python,
        str(script),
        "--binary",
        str(binary),
        "--database",
        str(database),
        "--queries",
        str(queries),
        "--base-config",
        str(config),
        "--output-dir",
        str(output),
        "--backend",
        backend,
        "--top-k",
        *(str(value) for value in top_k),
        "--nprobe",
        *(str(value) for value in nprobe),
        "--batch-size",
        *(str(value) for value in batch_size),
        "--warmup",
        str(preset.warmup),
        "--repetitions",
        str(repetitions),
        "--allow-existing",
    ]
    if index is not None:
        command.extend(["--index", str(index)])
    return command


def build_stages(
    args: argparse.Namespace, root: Path, preset: Preset, configs: dict[str, Path]
) -> list[Stage]:
    """按预设生成固定顺序的实验阶段。"""
    python = sys.executable
    scripts = root / "scripts"
    output = args.output_dir
    data = output / "data"
    database = data / "database.bin"
    queries = data / "queries.bin"
    index = output / "index" / "database.ivf"
    stages: list[Stage] = []
    if not args.skip_validation:
        validation = output / "validation"
        stages.append(
            Stage(
                "validation",
                [
                    python,
                    str(scripts / "run_gpu_validation.py"),
                    "--binary",
                    str(args.binary),
                    "--output-dir",
                    str(validation),
                ],
                validation,
                (validation / "summary.csv", validation / "report_table.md"),
            )
        )
    stages.append(
        Stage(
            "data_generation",
            [
                python,
                str(scripts / "generate_data.py"),
                "--output-dir",
                str(data),
                "--num-vectors",
                str(preset.num_vectors),
                "--num-queries",
                str(preset.num_queries),
                "--dim",
                str(preset.dim),
                "--dtype",
                "fp32",
                "--metric",
                "l2",
                "--clusters",
                str(preset.clusters),
                "--seed",
                str(args.master_seed),
            ],
            data,
            (database, queries, data / "metadata.json", data / "hashes.sha256"),
        )
    )
    build_dir = output / "index"
    stages.append(
        Stage(
            "ivf_build",
            [
                str(args.binary),
                "build",
                "--database",
                str(database),
                "--config",
                str(configs["ivf"]),
                "--index",
                str(index),
                "--backend",
                args.backend,
                "--performance",
                str(build_dir / "build.performance.log"),
            ],
            build_dir,
            (index, build_dir / "build.performance.log"),
        )
    )
    exact_dir = output / "exact_topk"
    stages.append(
        Stage(
            "exact_topk",
            sweep_command(
                python=python,
                script=scripts / "run_sweep.py",
                binary=args.binary,
                database=database,
                queries=queries,
                config=configs["exact"],
                output=exact_dir,
                preset=preset,
                top_k=(1, 10, 50, 100),
                nprobe=(1,),
                batch_size=(128,),
                repetitions=preset.exact_repetitions,
                index=None,
                backend=args.backend,
            ),
            exact_dir,
            (exact_dir / "aggregate.csv", exact_dir / "report_table.md"),
        )
    )
    nprobe_dir = output / "ivf_nprobe"
    stages.append(
        Stage(
            "ivf_nprobe",
            sweep_command(
                python=python,
                script=scripts / "run_sweep.py",
                binary=args.binary,
                database=database,
                queries=queries,
                config=configs["ivf"],
                output=nprobe_dir,
                preset=preset,
                top_k=(10,),
                nprobe=preset.nprobe_values,
                batch_size=(128,),
                repetitions=preset.ivf_repetitions,
                index=index,
                backend=args.backend,
            ),
            nprobe_dir,
            (nprobe_dir / "aggregate.csv", nprobe_dir / "report_table.md"),
        )
    )
    batch_dir = output / "ivf_batch"
    stages.append(
        Stage(
            "ivf_batch",
            sweep_command(
                python=python,
                script=scripts / "run_sweep.py",
                binary=args.binary,
                database=database,
                queries=queries,
                config=configs["ivf"],
                output=batch_dir,
                preset=preset,
                top_k=(10,),
                nprobe=(16,),
                batch_size=preset.batch_values,
                repetitions=preset.ivf_repetitions,
                index=index,
                backend=args.backend,
            ),
            batch_dir,
            (batch_dir / "aggregate.csv", batch_dir / "report_table.md"),
        )
    )
    if not args.skip_plots:
        plots = output / "plots"
        stages.append(
            Stage(
                "report_plots",
                [
                    python,
                    str(scripts / "generate_report_plots.py"),
                    "--exact",
                    str(exact_dir / "aggregate.csv"),
                    "--ivf-nprobe",
                    str(nprobe_dir / "aggregate.csv"),
                    "--ivf-batch",
                    str(batch_dir / "aggregate.csv"),
                    "--output-dir",
                    str(plots),
                ],
                plots,
                (plots / "plots_index.md", plots / "ivf_nprobe_recall.svg"),
            )
        )

    for tool in args.profile_tools:
        for mode in ("exact", "ivf"):
            profile_dir = output / f"profile_{tool}_{mode}"
            command = [
                python,
                str(scripts / "profile_vector_search.py"),
                "--tool",
                tool,
                "--binary",
                str(args.binary),
                "--database",
                str(database),
                "--queries",
                str(queries),
                "--config",
                str(configs[f"{mode}_profile"]),
                "--output-dir",
                str(profile_dir),
            ]
            # NVIDIA 工具会在 --export/--output 基名后自动补标准扩展名。
            profiler_mode = "ivf_flat" if mode == "ivf" else "exact"
            expected_report = profile_dir / (
                f"{profiler_mode}_ncu.ncu-rep"
                if tool == "ncu"
                else f"{profiler_mode}_nsys.nsys-rep"
            )
            if mode == "ivf":
                command.extend(["--index", str(index)])
            stages.append(
                Stage(
                    f"profile_{tool}_{mode}",
                    command,
                    profile_dir,
                    (profile_dir / "command.txt", expected_report),
                )
            )
    return stages


def plan_dict(
    args: argparse.Namespace, preset: Preset, stages: list[Stage]
) -> dict[str, object]:
    return {
        "preset": args.preset,
        # JSON round-trip 把 dataclass tuple 规范为 list，保证落盘后 --resume 可相等比较。
        "preset_values": json.loads(json.dumps(asdict(preset))),
        "master_seed": args.master_seed,
        "backend": args.backend,
        "skip_validation": args.skip_validation,
        "skip_plots": args.skip_plots,
        "profile_tools": args.profile_tools,
        "stages": [
            {
                "name": stage.name,
                "command": shlex.join(stage.command),
                "expected_files": [str(path) for path in stage.expected_files],
            }
            for stage in stages
        ],
    }


def write_artifact_index(output_dir: Path, records: list[dict[str, object]]) -> None:
    """汇总已通过阶段的索引和结果表。"""
    lines = [
        "# Task 09 Experiment Artifacts\n\n",
        "正式结论只能引用 `status=passed` 且 expected files 完整的阶段。\n\n",
        "| Stage | Status | Seconds | Console | Artifact directory |\n",
        "|---|---|---:|---|---|\n",
    ]
    tables = [
        "# Task 09 Report Tables\n\n",
        "由正式实验驱动器合并生成；精确值与样本量以相邻 CSV/日志为准。\n",
    ]
    for record in records:
        stage = str(record["stage"])
        artifact = Path(str(record["artifact_dir"]))
        try:
            artifact_rel = artifact.relative_to(output_dir)
        except ValueError:
            artifact_rel = artifact
        console = Path(str(record["console_log"]))
        try:
            console_rel = console.relative_to(output_dir)
        except ValueError:
            console_rel = console
        lines.append(
            f"| {stage} | {record['status']} | "
            f"{float(record['elapsed_seconds']):.3f} | `{console_rel}` | "
            f"`{artifact_rel}` |\n"
        )
        table = artifact / "report_table.md"
        if record["status"] == "passed" and table.is_file():
            tables.extend(
                [f"\n## {stage}\n\n", table.read_text(encoding="utf-8"), "\n"]
            )
        plot_index = artifact / "plots_index.md"
        if record["status"] == "passed" and plot_index.is_file():
            plot_text = plot_index.read_text(encoding="utf-8").replace(
                "](", f"]({artifact_rel}/"
            )
            tables.extend(["\n", plot_text, "\n"])
    (output_dir / "artifact_index.md").write_text("".join(lines), encoding="utf-8")
    (output_dir / "report_tables.md").write_text("".join(tables), encoding="utf-8")


def run_stage(root: Path, output_dir: Path, stage: Stage) -> dict[str, object]:
    """流式 tee 单个 stage，并把失败和缺失产物都转成非零状态。"""
    log_path = output_dir / "logs" / f"{stage.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stage.artifact_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[{stage.name}] {shlex.join(stage.command)}", flush=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            stage.command,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        return_code = process.wait()
    missing = [str(path) for path in stage.expected_files if not path.is_file()]
    status = "passed" if return_code == 0 and not missing else "failed"
    return {
        "stage": stage.name,
        "status": status,
        "return_code": return_code,
        "elapsed_seconds": time.monotonic() - started,
        "command": shlex.join(stage.command),
        "console_log": str(log_path),
        "artifact_dir": str(stage.artifact_dir),
        "expected_files": [str(path) for path in stage.expected_files],
        "missing_files": missing,
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    args.binary = args.binary.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.binary.is_file():
        raise SystemExit(f"binary does not exist: {args.binary}")
    if args.backend != "cuda" and not args.skip_validation:
        raise SystemExit("CPU quick checks must add --skip-validation")
    if args.backend != "cuda" and args.profile_tools:
        raise SystemExit("ncu/nsys profiling requires --backend cuda")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise SystemExit(
            f"output directory is not empty: {args.output_dir}; choose a new "
            "directory or use --resume"
        )
    if args.resume and not args.output_dir.is_dir():
        raise SystemExit("--resume requires an existing output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    preset = PRESETS[args.preset]
    configs = {
        "exact": args.output_dir / "configs" / "exact.txt",
        "ivf": args.output_dir / "configs" / "ivf.txt",
        "exact_profile": args.output_dir / "configs" / "exact_profile.txt",
        "ivf_profile": args.output_dir / "configs" / "ivf_profile.txt",
    }
    stages = build_stages(args, root, preset, configs)
    plan = plan_dict(args, preset, stages)
    plan_path = args.output_dir / "experiment_plan.json"
    if args.resume:
        if not plan_path.is_file():
            raise SystemExit("resume directory has no experiment_plan.json")
        previous_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if previous_plan != plan:
            raise SystemExit("resume plan differs from the frozen experiment plan")
    else:
        plan_path.write_text(
            json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    # 先完成 frozen-plan 对照，再写配置，避免错误的 --resume 参数改动旧证据目录。
    for name, mode, quality in (
        ("exact", "exact", preset.quality_queries),
        ("ivf", "ivf_flat", preset.quality_queries),
        ("exact_profile", "exact", 1),
        ("ivf_profile", "ivf_flat", 1),
    ):
        write_config(
            configs[name],
            mode=mode,
            preset=preset,
            quality_queries=quality,
            seed=args.master_seed,
        )

    environment_name = "resume_environment.txt" if args.resume else "environment.txt"
    git_status = write_environment(
        root, args.binary, args.output_dir, filename=environment_name
    )
    if git_status != "clean" and not args.allow_dirty:
        raise SystemExit(
            "git worktree is not clean; commit the exact code first or use "
            "--allow-dirty only for a non-report check"
        )

    manifest_path = args.output_dir / "manifest.json"
    records: list[dict[str, object]] = []
    if args.resume and manifest_path.is_file():
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
    completed = {
        str(record["stage"]): record
        for record in records
        if record.get("status") == "passed"
        and all(Path(path).is_file() for path in record.get("expected_files", []))
    }
    records = []
    for stage in stages:
        if stage.name in completed:
            record = completed[stage.name]
            print(f"[{stage.name}] resume: verified and skipped", flush=True)
        else:
            record = run_stage(root, args.output_dir, stage)
        records.append(record)
        manifest_path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        write_artifact_index(args.output_dir, records)
        if record["status"] != "passed":
            raise SystemExit(
                f"stage failed: {stage.name}; return_code={record['return_code']}; "
                f"missing={record['missing_files']}; inspect {record['console_log']}"
            )
    print(
        f"all {len(stages)} stages passed; report assets: "
        f"{args.output_dir / 'report_tables.md'}"
    )


if __name__ == "__main__":
    main()

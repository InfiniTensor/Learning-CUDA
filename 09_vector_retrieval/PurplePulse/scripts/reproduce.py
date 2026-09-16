#!/usr/bin/env python3
"""Run the complete WarpFusion build, benchmark, report, and validation flow."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("sift1m", "yahoo-minilm-384")
PREPARE_NAMES = {
    "sift1m": "sift1m",
    "yahoo-minilm-384": "yahoo-minilm-384-normalized",
}


def rooted(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build, test, prepare public data, run WarpFusion/FAISS, plot, "
            "build the report, and validate release artifacts."
        )
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--dataset", action="append", choices=DATASETS)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/real/raw"))
    parser.add_argument("--data-root", type=Path, default=Path("data/real"))
    parser.add_argument("--results-root", type=Path, default=Path("results/real"))
    parser.add_argument("--build-dir", type=Path, default=Path("build-reproduce"))
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("output/pdf/WarpFusion_Technical_Report.pdf"),
    )
    parser.add_argument("--cuda-architectures", default="native")
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--nprobes", default="32,64,128,160,192,224")
    parser.add_argument("--training-samples", type=int, default=100000)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-index-build", action="store_true")
    parser.add_argument("--skip-faiss", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def make_step(name: str, command: list[str], log: Path) -> dict[str, object]:
    return {"name": name, "command": command, "log": log}


def build_plan(args: argparse.Namespace) -> tuple[Path, list[dict[str, object]]]:
    root = args.project_root.resolve()
    build = rooted(root, args.build_dir)
    raw = rooted(root, args.raw_dir)
    data = rooted(root, args.data_root)
    results = rooted(root, args.results_root)
    report = rooted(root, args.report_output)
    state_root = results / "reproducibility"
    logs = state_root / "logs"
    python = sys.executable
    datasets = tuple(dict.fromkeys(args.dataset or DATASETS))

    steps = [
        make_step(
            "configure",
            [
                "cmake", "-S", str(root), "-B", str(build), "-G", "Ninja",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_architectures}",
            ],
            logs / "configure.log",
        ),
        make_step(
            "build", ["cmake", "--build", str(build)], logs / "build.log"
        ),
        make_step(
            "test",
            ["ctest", "--test-dir", str(build), "--output-on-failure"],
            logs / "ctest.log",
        ),
    ]

    prepare = [
        python,
        str(root / "scripts" / "prepare_real_datasets.py"),
        "--dataset",
        "all" if set(datasets) == set(DATASETS) else PREPARE_NAMES[datasets[0]],
        "--raw-dir", str(raw),
        "--output-dir", str(data),
    ]
    if args.skip_download:
        prepare.append("--skip-download")
    if set(datasets) == set(DATASETS) or len(datasets) == 1:
        steps.append(make_step("prepare", prepare, logs / "prepare.log"))
    else:
        raise ValueError("unsupported dataset selection")

    for dataset in datasets:
        command = [
            python,
            str(root / "scripts" / "run_real_benchmarks.py"),
            "--dataset", dataset,
            "--project-root", str(root),
            "--data-root", str(data),
            "--results-root", str(results),
            "--build-dir", str(build),
            "--nlist", str(args.nlist),
            "--nprobes", args.nprobes,
            "--training-samples", str(args.training_samples),
            "--iterations", str(args.iterations),
            "--threads", str(args.threads),
            "--batch-size", str(args.batch_size),
            "--warmup", str(args.warmup),
            "--repeat", str(args.repeat),
        ]
        if args.skip_index_build:
            command.append("--skip-index-build")
        if args.skip_faiss:
            command.append("--skip-faiss")
        steps.append(make_step(f"benchmark:{dataset}", command,
                               logs / f"benchmark-{dataset}.log"))
        steps.append(
            make_step(
                f"plot:{dataset}",
                [
                    python,
                    str(root / "scripts" / "plot_results.py"),
                    "--results-dir", str(results / dataset),
                    "--output-dir", str(results / dataset / "figures"),
                ],
                logs / f"plot-{dataset}.log",
            )
        )

    steps.extend(
        [
            make_step(
                "report",
                [
                    python,
                    str(root / "scripts" / "build_final_report_pdf.py"),
                    "--source", str(root / "docs" / "FINAL_REPORT.md"),
                    "--output", str(report),
                ],
                logs / "report.log",
            ),
            make_step(
                "validate",
                [
                    python,
                    str(root / "scripts" / "validate_release.py"),
                    "--project-root", str(root),
                    "--report", str(report),
                ],
                logs / "validate.log",
            ),
        ]
    )
    return state_root, steps


def load_checkpoint(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"completed": [], "steps": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value.get("completed"), list):
        raise ValueError(f"invalid checkpoint: {path}")
    return value


def write_checkpoint(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_step(root: Path, step: dict[str, object]) -> int:
    log = Path(step["log"])
    command = [str(item) for item in step["command"]]
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{step['name']}] {' '.join(command)}", flush=True)
    with log.open("w", encoding="utf-8") as output:
        completed = subprocess.run(
            command,
            cwd=root,
            env=os.environ.copy(),
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return completed.returncode


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    state_root, steps = build_plan(args)
    root = args.project_root.resolve()
    checkpoint_path = state_root / "checkpoint.json"
    if args.dry_run:
        print(json.dumps({"project_root": str(root), "steps": steps},
                         ensure_ascii=False, indent=2, default=str))
        return

    state = load_checkpoint(checkpoint_path) if args.resume else {
        "completed": [], "steps": {}
    }
    completed_names = set(state["completed"])
    state["started_utc"] = state.get(
        "started_utc", datetime.now(timezone.utc).isoformat()
    )
    state["project_root"] = str(root)
    for step in steps:
        name = str(step["name"])
        if args.resume and name in completed_names:
            print(f"[{name}] checkpoint complete; skipping", flush=True)
            continue
        returncode = run_step(root, step)
        state.setdefault("steps", {})[name] = {
            "returncode": returncode,
            "log": str(step["log"]),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        }
        if returncode != 0:
            state["failed_step"] = name
            write_checkpoint(checkpoint_path, state)
            raise SystemExit(f"step failed: {name}; see {step['log']}")
        completed_names.add(name)
        state["completed"] = [
            str(item["name"]) for item in steps
            if str(item["name"]) in completed_names
        ]
        state.pop("failed_step", None)
        write_checkpoint(checkpoint_path, state)
    state["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_checkpoint(checkpoint_path, state)
    print(f"reproduction complete: {checkpoint_path}")


if __name__ == "__main__":
    main()

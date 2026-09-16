#!/usr/bin/env python3
"""Validate that report claims have their required repository artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def validate_csv(path: Path, required_columns: set[str] | None = None) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields = set(reader.fieldnames or ())
        if required_columns and not required_columns <= fields:
            missing = sorted(required_columns - fields)
            raise ValueError(f"{path}: missing columns {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no data rows")
    return len(rows)


def validate_pdf(path: Path) -> int:
    if not path.is_file() or path.stat().st_size < 1024:
        raise ValueError(f"missing or undersized PDF: {path}")
    with path.open("rb") as source:
        if source.read(5) != b"%PDF-":
            raise ValueError(f"invalid PDF header: {path}")
    return path.stat().st_size


def validate(root: Path, report: Path) -> dict[str, object]:
    required_docs = {
        "README.md": "一条命令复现",
        "docs/FINAL_REPORT.md": "7.4",
        "docs/REPRODUCIBILITY.md": "第九阶段工程化入口",
        "docs/STABILITY_MATRIX.md": "Exact 选择规则",
        "docs/API.md": "C++ API",
    }
    for relative, marker in required_docs.items():
        path = root / relative
        if not path.is_file() or marker not in path.read_text(encoding="utf-8"):
            raise ValueError(f"{relative}: missing required marker {marker!r}")

    csv_rows: dict[str, int] = {}
    for dataset in ("sift1m", "yahoo-minilm-384"):
        base = root / "results" / "real" / dataset
        for filename, columns in (
            ("exact_gpu.csv", {"top_k", "qps"}),
            ("faiss_gpu.csv", {"top_k", "qps"}),
            ("ivf_k10.csv", {"nprobe", "recall_at_k", "qps"}),
            ("ivf_k50.csv", {"nprobe", "recall_at_k", "qps"}),
            ("ivf_k100.csv", {"nprobe", "recall_at_k", "qps"}),
        ):
            path = base / filename
            csv_rows[str(path.relative_to(root))] = validate_csv(path, columns)

    for filename, columns in (
        ("exact_gpu.csv", {"top_k", "qps", "run_samples"}),
        ("faiss_cpu_flat.csv", {"top_k", "qps", "run_samples"}),
    ):
        path = root / "results" / "formal_100k15" / filename
        csv_rows[str(path.relative_to(root))] = validate_csv(path, columns)

    nsys_summary = root / "evidence" / "2026-09-04" / "nsys_exact_k10.txt"
    nsys_text = nsys_summary.read_text(encoding="utf-8")
    for marker in ("99.9", "exactFusedWarpTopKKernel", "cuda_gpu_kern_sum"):
        if marker not in nsys_text:
            raise ValueError(f"{nsys_summary.relative_to(root)}: missing marker {marker!r}")

    coverage = root / "results" / "stage8_stability" / "coverage.csv"
    csv_rows[str(coverage.relative_to(root))] = validate_csv(
        coverage, {"suite", "cases", "status"}
    )
    with coverage.open(newline="", encoding="utf-8") as source:
        failed = [row["suite"] for row in csv.DictReader(source)
                  if row["status"] != "passed"]
    if failed:
        raise ValueError(f"stage8 coverage failures: {failed}")

    for path in (
        root / "results" / "figures" / "report-architecture.svg",
        root / "results" / "figures" / "report-exact-fused.svg",
        root / "results" / "figures" / "report-exact-breakdown.svg",
        root / "results" / "figures" / "report-k50-scheduling.svg",
        root / "results" / "figures" / "report-k100-partitioned.svg",
        root / "results" / "figures" / "report-recall-qps.svg",
        root / "results" / "figures" / "report-real-comparison.svg",
    ):
        if not path.is_file() or "<svg" not in path.read_text(encoding="utf-8")[:500]:
            raise ValueError(f"missing or invalid SVG: {path}")

    return {
        "status": "passed",
        "project_root": root.name,
        "validated_csv_rows": csv_rows,
        "report_bytes": validate_pdf(report),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument(
        "--report", type=Path,
        default=Path("output/pdf/WarpFusion_Technical_Report.pdf"),
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    report = args.report.resolve() if args.report.is_absolute() else (root / args.report)
    result = validate(root, report)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.json_output:
        output = (args.json_output if args.json_output.is_absolute()
                  else root / args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()

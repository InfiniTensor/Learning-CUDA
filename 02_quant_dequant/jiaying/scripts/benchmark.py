#!/usr/bin/env python3
"""Run the full distribution/format/scaling/input-dtype matrix and retain real logs."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import tempfile
from generate import generate
from tensor_io import write_tensor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--executable", default="build/low_precision")
    p.add_argument("--results", default="reports/results")
    p.add_argument("--rows", type=int, default=1024)
    p.add_argument("--cols", type=int, default=4096)
    p.add_argument("--iterations", type=int, default=50)
    args = p.parse_args()
    exe = str(Path(args.executable).resolve())
    results = Path(args.results).resolve()
    results.mkdir(parents=True, exist_ok=True)
    records = []
    with tempfile.TemporaryDirectory(prefix="lp_bench_") as temp:
        work = Path(temp)
        for distribution in ["uniform", "normal", "outliers"]:
            a = generate(args.rows, args.cols, distribution, 42)
            for dtype in ["fp32", "fp16"]:
                write_tensor(work / "input.bin", a, dtype)
                for fmt in ["mxfp8", "nvfp4"]:
                    for mode in ["block", "tensor"]:
                        name = f"{distribution}_{dtype}_{fmt}_{mode}"
                        config = work / "config.toml"
                        config.write_text(f'format = "{fmt}"\nscale_mode = "{mode}"\n'
                                          'output_type = "fp32"\nrounding = "nearest"\nseed = 42\n')
                        command = [exe, "run", "--input", str(work / "input.bin"), "--config", str(config),
                                   "--weights", str(work / "weights.bin"), "--output", str(work / "restored.bin"),
                                   "--log", str(results / f"{name}.json"), "--warmup", "5",
                                   "--iterations", str(args.iterations), "--cpu-repeats", "3"]
                        subprocess.run(command, check=True)
                        record = json.loads((results / f"{name}.json").read_text())
                        record["distribution"] = distribution
                        records.append(record)
    with (results / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    columns = ["distribution", "input_type", "format", "scale_mode", "mae", "mse", "max_abs_error",
               "compression_payload", "quant_kernel_ms", "dequant_kernel_ms", "dequant_effective_gbps",
               "gpu_transfer_pipeline_ms", "resident_speedup", "transfer_pipeline_speedup"]
    lines = ["# 实测结果", "", f"矩阵：{args.rows} × {args.cols}；输出 FP32；nearest ties-to-even；seed=42。",
             "CUDA Events：5 次预热，连续多次执行取平均。完整字段见同目录 JSON/CSV。", "",
             "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for record in records:
        values = [f"{record[c]:.6g}" if isinstance(record[c], float) else str(record[c]) for c in columns]
        lines.append("| " + " | ".join(values) + " |")
    (results / "TABLE.md").write_text("\n".join(lines) + "\n")
    print(f"Saved {len(records)} measured cases to {results}")


if __name__ == "__main__":
    main()

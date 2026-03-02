import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def compare_all():
    """对比所有 mx 和 BnB 的结果"""

    base = Path("..")
    mx_dir = base / "mx_results"
    bnb_dir = base / "bnb_results"
    bnb_csv = base / "bnb_benchmark_results.csv"

    results = []

    if not bnb_csv.exists():
        print(f"找不到 {bnb_csv}")
        return

    bnb_data = {}
    with open(bnb_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['Shape']}_bs{row['Blocksize']}"
            bnb_data[key] = {
                "time_ms": float(row["BnB Time (ms)"]),
                "bnb_file": row["BnB Output File"],
            }

    mx_files = list(mx_dir.glob("dequant_*.fp16"))

    print("\n" + "=" * 90)
    print("对比结果汇总")
    print("=" * 90)
    print(
        f"{'Shape':<12} {'Block':<6} {'BnB (ms)':<12} {'MX (ms)':<12} {'Speedup':<8} {'MAE':<12} {'Max Diff':<12}"
    )
    print("-" * 90)

    for mx_file in mx_files:
        filename = mx_file.name
        shape, blocksize = (
            filename.replace("dequant_", "").replace(".fp16", "").split("_bs")
        )

        mx_tensor = torch.from_numpy(np.fromfile(mx_file, dtype=np.float16))

        bnb_file = bnb_dir / f"bnb_{shape}_bs{blocksize}.fp16"
        if not bnb_file.exists():
            print(f"找不到 BnB 文件: {bnb_file}")
            continue

        bnb_tensor = torch.from_numpy(np.fromfile(bnb_file, dtype=np.float16))

        

        mae = torch.mean(torch.abs(bnb_tensor - mx_tensor)).item()
        mse = torch.mean((bnb_tensor - mx_tensor) ** 2).item()
        max_diff = torch.max(torch.abs(bnb_tensor - mx_tensor)).item()

        key = f"{shape}_bs{blocksize}"
        bnb_time = bnb_data.get(key, {}).get("time_ms", 0.0)

        log_file = mx_dir / f"perf_{shape}_bs{blocksize}.log"
        mx_time = 0.0
        if log_file.exists():
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if "kernel_time_ms" in line:
                        mx_time = float(line.strip().split("=")[1])

        speedup = bnb_time / mx_time if mx_time > 0 else 0.0

        print(
            f"{shape:<12} {blocksize:<6} {bnb_time:<12.4f} {mx_time:<12.4f} {speedup:<8.2f} {mae:<12.8f} {max_diff:<12.8f}"
        )

        results.append(
            {
                "shape": shape,
                "blocksize": int(blocksize),
                "bnb_time_ms": bnb_time,
                "mx_time_ms": mx_time,
                "speedup": speedup,
                "mae": mae,
                "mse": mse,
                "max_diff": max_diff,
            }
        )

    print("=" * 90)

    if not results:
        print("没有可用的对比结果。")
        return

    df = pd.DataFrame(results)
    out_csv = "comparison_mx_results.csv"
    df.to_csv(out_csv, index=False, float_format="%.8f")
    print(f"\n对比结果已保存到: {out_csv}")

    md_file = "comparison_mx_results.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# MX vs BnB 对比结果\n\n")
        f.write("| Shape | Block | BnB (ms) | MX (ms) | Speedup | MAE | Max Diff |\n")
        f.write("|-------|-------|----------|---------|---------|-----|----------|\n")
        for _, row in df.iterrows():
            f.write(
                f"| {row['shape']} | {row['blocksize']} | {row['bnb_time_ms']:.4f} | {row['mx_time_ms']:.4f} | {row['speedup']:.2f} | {row['mae']:.8f} | {row['max_diff']:.8f} |\n"
            )

        f.write("\n\n## 详细数据\n\n")
        f.write(
            "| Shape | Block | BnB (ms) | MX (ms) | Speedup | MAE | MSE | Max Diff | BnB Error |\n"
        )
        f.write(
            "|-------|-------|----------|---------|---------|-----|-----|----------|-----------|\n"
        )
        for _, row in df.iterrows():
            f.write(
                f"| {row['shape']} | {row['blocksize']} | {row['bnb_time_ms']:.4f} | {row['mx_time_ms']:.4f} | {row['speedup']:.2f} | {row['mae']:.8f} | {row['mse']:.8f} | {row['max_diff']:.8f}  |\n"
            )

    print(f"Markdown 表格已保存到: {md_file}")


if __name__ == "__main__":
    compare_all()

import torch
import numpy as np
import pandas as pd
import csv
from pathlib import Path


def compare_all():
    """对比所有 musa 和 BnB 的结果"""

    BASE = Path("..")

    MUSA_DIR = BASE / "musa_results"
    BNB_DIR = BASE / "bnb_results"
    BNB_CSV = BASE / "bnb_benchmark_results.csv"

    results = []

    # 读取 BnB benchmark CSV
    if not BNB_CSV.exists():
        print(f" 找不到 {BNB_CSV}")
        return

    bnb_data = {}
    with open(BNB_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['Shape']}_bs{row['Blocksize']}"
            bnb_data[key] = {
                'time_ms': float(row['BnB Time (ms)']),
                'bnb_file': row['BnB Output File']
            }

    # ==================================================
    # 查找所有 MUSA 结果
    # ==================================================
    musa_files = list(MUSA_DIR.glob("dequant_*.fp16"))

    print("\n" + "="*80)
    print("对比结果汇总")
    print("="*80)
    print(f"{'Shape':<12} {'Block':<6} {'BnB (ms)':<12} {'MUSA (ms)':<12} "
          f"{'Speedup':<8} {'MAE':<12} {'Max Diff':<12}")
    print("-"*80)

    for musa_file in musa_files:
        filename = musa_file.name

        parts = filename.replace('dequant_', '').replace('.fp16', '').split('_bs')
        shape = parts[0]
        blocksize = parts[1]

        # 读取 MUSA 输出
        musa_data = np.fromfile(musa_file, dtype=np.float16)
        musa_tensor = torch.from_numpy(musa_data)

        # 读取 BnB 输出
        bnb_file = BNB_DIR / f"bnb_{shape}_bs{blocksize}.fp16"
        if not bnb_file.exists():
            print(f" 找不到 BnB 文件: {bnb_file}")
            continue

        bnb_data_np = np.fromfile(bnb_file, dtype=np.float16)
        bnb_tensor = torch.from_numpy(bnb_data_np)



        # 误差计算
        mae = torch.mean(torch.abs(bnb_tensor - musa_tensor)).item()
        mse = torch.mean((bnb_tensor - musa_tensor) ** 2).item()
        max_diff = torch.max(torch.abs(bnb_tensor - musa_tensor)).item()

        # 时间读取
        key = f"{shape}_bs{blocksize}"
        bnb_time = bnb_data[key]['time_ms']

        log_file = MUSA_DIR / f"perf_{shape}_bs{blocksize}.log"
        musa_time = 0

        if log_file.exists():
            with open(log_file, 'r') as f:
                for line in f:
                    if 'kernel_time_ms' in line:
                        musa_time = float(line.strip().split('=')[1])

        speedup = bnb_time / musa_time if musa_time > 0 else 0

        print(f"{shape:<12} {blocksize:<6} "
              f"{bnb_time:<12.4f} {musa_time:<12.4f} "
              f"{speedup:<8.2f} {mae:<12.8f} {max_diff:<12.8f}")

        results.append({
            'shape': shape,
            'blocksize': int(blocksize),
            'bnb_time_ms': bnb_time,
            'musa_time_ms': musa_time,
            'speedup': speedup,
            'mae': mae,
            'mse': mse,
            'max_diff': max_diff,
        })

    print("="*80)


    # 保存 CSV
    df = pd.DataFrame(results)
    out_csv =  "comparison_musa_results.csv"
    float_format = '%.8f' 
    df.to_csv(out_csv, index=False, float_format=float_format)

    print(f"\n 对比结果已保存到: {out_csv}")
    md_file = "comparison_musa_results.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        # 写入表格标题
        f.write("# MUSA vs BnB 对比结果\n\n")
        
        # 写入表头
        f.write("| Shape | Block | BnB (ms) | MUSA (ms) | Speedup | MAE | Max Diff |\n")
        f.write("|-------|-------|----------|-----------|---------|-----|----------|\n")
        
        # 写入数据
        for _, row in df.iterrows():
            line = (f"| {row['shape']} | {row['blocksize']} | "
                   f"{row['bnb_time_ms']:.4f} | {row['musa_time_ms']:.4f} | "
                   f"{row['speedup']:.2f} | {row['mae']:.8f} | {row['max_diff']:.8f} |")
            f.write(line + '\n')
        
        # 添加详细数据表
        f.write("\n\n## 详细数据\n\n")
        f.write("| Shape | Block | BnB (ms) | MUSA (ms) | Speedup | MAE | MSE | Max Diff | BnB Error |\n")
        f.write("|-------|-------|----------|-----------|---------|-----|-----|----------|-----------|\n")
        
        for _, row in df.iterrows():
            line = (f"| {row['shape']} | {row['blocksize']} | "
                   f"{row['bnb_time_ms']:.4f} | {row['musa_time_ms']:.4f} | "
                   f"{row['speedup']:.2f} | {row['mae']:.8f} | {row['mse']:.8f} | "
                   f"{row['max_diff']:.8f} | ")
            f.write(line + '\n')
    
    print(f" Markdown表格已保存到: {md_file}")

    return df



if __name__ == "__main__":
    compare_all()
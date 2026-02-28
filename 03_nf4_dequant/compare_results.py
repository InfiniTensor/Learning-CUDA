import torch
import numpy as np
import pandas as pd
import glob
import os
import csv

def compare_all():
    """对比所有 CUDA 和 BnB 的结果"""
    results = []
    
    # 读取 BnB 基准结果 CSV
    bnb_csv = "bnb_benchmark_results.csv"
    if not os.path.exists(bnb_csv):
        print(f"❌ 找不到 {bnb_csv}")
        return
    
    bnb_data = {}
    with open(bnb_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['Shape']}_bs{row['Blocksize']}"
            bnb_data[key] = {
                'time_ms': float(row['BnB Time (ms)']),
                'bnb_file': row['BnB Output File']
            }
    
    # 查找所有 CUDA 结果
    cuda_files = glob.glob("cuda_results/dequant_*.fp16")
    
    print("\n" + "="*80)
    print("对比结果汇总")
    print("="*80)
    print(f"{'Shape':<12} {'Block':<6} {'BnB (ms)':<12} {'CUDA (ms)':<12} "
          f"{'Speedup':<8} {'MAE':<12} {'Max Diff':<12}")
    print("-"*80)
    
    for cuda_file in cuda_files:
        # 解析文件名
        # cuda_results/dequant_1024x1024_bs64.fp16
        filename = os.path.basename(cuda_file)
        parts = filename.replace('dequant_', '').replace('.fp16', '').split('_bs')
        shape = parts[0]
        blocksize = parts[1]
        
        # 读取 CUDA 结果
        cuda_data = np.fromfile(cuda_file, dtype=np.float16)
        cuda_tensor = torch.from_numpy(cuda_data)
        
        # 读取对应的 BnB 结果
        bnb_file = f"bnb_results/bnb_{shape}_bs{blocksize}.fp16"
        if not os.path.exists(bnb_file):
            print(f"⚠️ 找不到 BnB 文件: {bnb_file}")
            continue
        
        bnb_data_np = np.fromfile(bnb_file, dtype=np.float16)
        bnb_tensor = torch.from_numpy(bnb_data_np)
        
        # 读取原始权重（可选）
        orig_file = f"bnb_results/original_{shape}_bs{blocksize}.fp16"
        if os.path.exists(orig_file):
            orig_data = np.fromfile(orig_file, dtype=np.float16)
            orig_tensor = torch.from_numpy(orig_data)
            
            # 验证解量化是否正确
            bnb_error = torch.mean(torch.abs(orig_tensor - bnb_tensor)).item()
        else:
            bnb_error = 0
        
        # 计算误差
        mae = torch.mean(torch.abs(bnb_tensor - cuda_tensor)).item()
        mse = torch.mean((bnb_tensor - cuda_tensor) ** 2).item()
        max_diff = torch.max(torch.abs(bnb_tensor - cuda_tensor)).item()
        
        # 获取时间
        key = f"{shape}_bs{blocksize}"
        bnb_time = bnb_data[key]['time_ms']
        
        # 读取 CUDA 日志获取时间
        log_file = f"cuda_results/perf_{shape}_bs{blocksize}.log"
        cuda_time = 0
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                for line in f:
                    if 'kernel_time_ms' in line:
                        cuda_time = float(line.strip().split('=')[1])
        
        speedup = bnb_time / cuda_time if cuda_time > 0 else 0
        
        print(f"{shape:<12} {blocksize:<6} "
              f"{bnb_time:<12.4f} {cuda_time:<12.4f} "
              f"{speedup:<8.2f} {mae:<12.8f} {max_diff:<12.8f}")
        
        results.append({
            'shape': shape,
            'blocksize': int(blocksize),
            'bnb_time_ms': bnb_time,
            'cuda_time_ms': cuda_time,
            'speedup': speedup,
            'mae': mae,
            'mse': mse,
            'max_diff': max_diff,
            'bnb_error': bnb_error if 'bnb_error' in locals() else 0
        })
    
    print("="*80)
    
    # 保存对比结果
    df = pd.DataFrame(results)
    df.to_csv('comparison_results.csv', index=False)
    print(f"\n📊 对比结果已保存到: comparison_results.csv")
    
    return df

if __name__ == "__main__":
    compare_all()

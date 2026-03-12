import torch
import bitsandbytes.functional as F
import numpy as np
import struct
import os
import csv
from pathlib import Path

# 保存权重数据到文件
def save_weight_data(filename, rows, cols, blocksize, 
                     packed, absmax_q, absmax2, code2, offset):
    """
    保存权重数据到二进制文件
    格式：
    [header] 
        num_rows: int64
        num_cols: int64  
        blocksize: int32
    [data]
        packed_weights: uint8[num_rows * num_cols / 2]
        absmax_q: uint8[num_blocks]
        absmax2: float16[num_groups]
        code2: float16[256]
        offset: float32
    """
    with open(filename, 'wb') as f:
        # 写入 header
        f.write(struct.pack('q', rows))           # num_rows (int64)
        f.write(struct.pack('q', cols))           # num_cols (int64)
        f.write(struct.pack('i', blocksize))      # blocksize (int32)
        
        # 写入 packed_weights (uint8)
        packed_np = packed.cpu().numpy().astype(np.uint8)
        f.write(packed_np.tobytes())
        
        # 写入 absmax_q (uint8)
        absmax_q_np = absmax_q.cpu().numpy().astype(np.uint8)
        f.write(absmax_q_np.tobytes())
        
        # 写入 absmax2 (float16)
        absmax2_np = absmax2.cpu().numpy().astype(np.float16)
        f.write(absmax2_np.tobytes())
        
        # 写入 code2 (float16[256]) - 确保长度为256
        code2_np = code2.cpu().numpy().astype(np.float16)
        if len(code2_np) < 256:
            code2_padded = np.zeros(256, dtype=np.float16)
            code2_padded[:len(code2_np)] = code2_np
        else:
            code2_padded = code2_np[:256]
        f.write(code2_padded.tobytes())
        
        # 写入 offset (float32)
        f.write(struct.pack('f', offset))
    
    file_size = os.path.getsize(filename)
    print(f"   权重文件已保存: {filename} ({file_size/1024:.2f} KB)")

# 保存 bitsandbytes 的解量化结果
def save_bnb_output(filename, output_tensor, rows, cols):
    """
    保存 bitsandbytes 的解量化结果
    格式：float16 二进制文件，按行主序存储
    """
    output_np = output_tensor.cpu().numpy().astype(np.float16)
    output_np.tofile(filename)
    file_size = os.path.getsize(filename)
    print(f"   BnB 结果已保存: {filename} ({file_size/1024:.2f} KB)")

# =========================================================
# 运行 bitsandbytes 解量化并计时
# =========================================================
def run_bnb_dequant(packed, state, test_iters=100):
    """
    运行 bitsandbytes 解量化，返回结果和平均执行时间
    """
    print(f"   运行 bitsandbytes 解量化...")
    
    # 预热
    for _ in range(10):
        out_ref = F.dequantize_4bit(packed, state)
    
    torch.cuda.synchronize()
    
    # 计时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(test_iters):
        out_ref = F.dequantize_4bit(packed, state)
    end.record()
    
    torch.cuda.synchronize()
    bnb_time = start.elapsed_time(end) / test_iters
    
    # 最后一次结果用于保存
    final_output = F.dequantize_4bit(packed, state)
    
    print(f"   BnB 完成, 平均时间: {bnb_time:.4f} ms")
    
    return final_output, bnb_time

# =========================================================
# 生成并保存单个测试用例
# =========================================================
def generate_and_test(rows, cols, blocksize, group_size=256, 
                      save_dir="weight_data", bnb_dir="bnb_results"):
    """
    生成一个测试用例，运行 bitsandbytes，保存所有需要的数据
    """
    total = rows * cols
    
    print(f"\n 处理 {rows}x{cols} 矩阵, blocksize={blocksize}...")
    
    # 创建保存目录
    Path(save_dir).mkdir(exist_ok=True)
    Path(bnb_dir).mkdir(exist_ok=True)
    
    # 创建权重数据（在 GPU 上）
    weight = torch.randn(rows, cols, device="cuda", dtype=torch.float16)

    # 量化
    packed, state = F.quantize_4bit(
        weight,
        blocksize=blocksize,
        quant_type="nf4",
        compress_statistics=True
    )

    # 获取量化参数
    absmax_q = state.absmax.contiguous()
    absmax2 = state.state2.absmax.to(torch.float16).contiguous()
    code2 = state.state2.code.to(torch.float16).contiguous()
    offset = float(state.offset)

    # 保存权重文件（供 CUDA 程序读取）
    weight_file = f"{save_dir}/weight_{rows}x{cols}_bs{blocksize}.bin"
    save_weight_data(weight_file, rows, cols, blocksize,
                    packed, absmax_q, absmax2, code2, offset)
    
    # 运行 bitsandbytes 并计时
    bnb_output, bnb_time = run_bnb_dequant(packed, state)
    
    # 保存 bitsandbytes 的解量化结果
    bnb_file = f"{bnb_dir}/bnb_{rows}x{cols}_bs{blocksize}.fp16"
    save_bnb_output(bnb_file, bnb_output, rows, cols)
    

    
    return {
        'shape': f"{rows}x{cols}",
        'rows': rows,
        'cols': cols,
        'blocksize': blocksize,
        'weight_file': weight_file,
        'bnb_file': bnb_file,
        'bnb_time_ms': bnb_time,
        'total_elements': total
    }


# 生成所有测试用例并保存结果
def generate_all():
    """
    生成所有测试用例，运行 bitsandbytes，保存结果
    """
    shapes = [
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 8192),
        (16384, 16384),
        (3421, 3146),
        (6578, 1236),
        (7000, 7000),
    ]

    blocksizes = [64, 128]
    
    save_dir = "weight_data"
    bnb_dir = "bnb_results"
    
    print("=" * 70)
    print("NF4 数据生成和 bitsandbytes 基准测试")
    print("=" * 70)
    print(f"权重文件保存目录: {save_dir}/")
    print(f"BnB 结果保存目录: {bnb_dir}/")
    print()
    
    results = []
    
    for rows, cols in shapes:
        for block in blocksizes:
            info = generate_and_test(rows, cols, block, 
                                    save_dir=save_dir, bnb_dir=bnb_dir)
            results.append(info)
    
    # 保存汇总结果到 CSV
    csv_file = "bnb_benchmark_results.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Shape', 'Blocksize', 'BnB Time (ms)', 
                        'Total Elements', 'Weight File', 'BnB Output File'])
        
        for r in results:
            writer.writerow([
                r['shape'],
                r['blocksize'],
                f"{r['bnb_time_ms']:.4f}",
                r['total_elements'],
                r['weight_file'],
                r['bnb_file']
            ])
    
    print("\n" + "=" * 70)
    print(" 生成完成!")
    print("=" * 70)
    print(f"\n BnB 基准测试结果已保存到: {csv_file}")
    print("\n生成的目录结构:")
    print(f"  {save_dir}/ - 包含以下权重文件（供 CUDA 程序使用）:")
    for r in results:
        print(f"    - weight_{r['shape']}_bs{r['blocksize']}.bin")
    
    print(f"\n  {bnb_dir}/ - 包含以下结果文件（用于对比）:")
    for r in results:
        print(f"    - bnb_{r['shape']}_bs{r['blocksize']}.fp16")
        print(f"    - original_{r['shape']}_bs{r['blocksize']}.fp16")
    
    print("\n" + "=" * 70)
    print("BnB 执行时间汇总:")
    print("-" * 70)
    print(f"{'Shape':<12} {'Block':<6} {'Time (ms)':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['shape']:<12} {r['blocksize']:<6} {r['bnb_time_ms']:<12.4f}")
    print("=" * 70)

# 生成单个测试用例
def generate_single(rows=1024, cols=1024, blocksize=64):
    """
    生成单个测试用例
    """
    save_dir = "weight_data"
    bnb_dir = "bnb_results"
    Path(save_dir).mkdir(exist_ok=True)
    Path(bnb_dir).mkdir(exist_ok=True)
    
    info = generate_and_test(rows, cols, blocksize, 
                            save_dir=save_dir, bnb_dir=bnb_dir)
    
    print("\n" + "=" * 50)
    print("单个测试用例生成完成")
    print("=" * 50)
    print(f"权重文件: {info['weight_file']}")
    print(f"BnB 结果: {info['bnb_file']}")
    print(f"原始权重: {info['original_file']}")
    print(f"BnB 执行时间: {info['bnb_time_ms']:.4f} ms")
    
    return info


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='生成 NF4 测试数据并运行 bitsandbytes')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'single'],
                       help='运行模式: all (所有配置) 或 single (单个配置)')
    parser.add_argument('--rows', type=int, default=1024,
                       help='矩阵行数 (single 模式)')
    parser.add_argument('--cols', type=int, default=1024,
                       help='矩阵列数 (single 模式)')
    parser.add_argument('--blocksize', type=int, default=64,
                       help='块大小 (single 模式)')
    
    args = parser.parse_args()
    
    if args.mode == 'all':
        generate_all()
    else:
        generate_single(args.rows, args.cols, args.blocksize)
import torch
import struct
import math
import numpy as np
import time

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    print("Warning: bitsandbytes not found. Baseline profiling will be skipped.")
    HAS_BNB = False

def profile_bnb_baseline(tensor_shape, blocksize):
    if not HAS_BNB:
        return
    
    print("\n--- Profiling bitsandbytes Baseline ---")
    # 强制在 GPU 上分配测试数据
    x = torch.randn(tensor_shape, dtype=torch.float16, device="cuda")
    
    try:
        # 双重量化
        print("Quantizing tensor...")
        quantized_tensor, quant_state = bnb.functional.quantize_4bit(
            x, 
            quant_type="nf4", 
            compress_statistics=True
        )
        
        # 预热
        print("Warming up dequantize...")
        for _ in range(10):
            _ = bnb.functional.dequantize_4bit(quantized_tensor, quant_state)
        torch.cuda.synchronize()
        
        # 测速
        print("Profiling dequantize (100 runs)...")
        num_runs = 100
        start_time = time.time()
        for _ in range(num_runs):
            _ = bnb.functional.dequantize_4bit(quantized_tensor, quant_state)
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time_ms = ((end_time - start_time) / num_runs) * 1000
        print(f"bitsandbytes Baseline Dequantize Avg Time: {avg_time_ms:.4f} ms")
        
    except Exception as e:
        print(f"Failed to profile bitsandbytes: {e}")

def create_mock_data_and_save(num_rows, num_cols, blocksize):
    print("\n--- Generating Mock Data for C++ Test ---")
    
    total_elements = num_rows * num_cols
    packed_size = total_elements // 2
    num_blocks = math.ceil(total_elements / blocksize)
    num_groups = math.ceil(num_blocks / 256)
    
    # 为了对比验证计算逻辑，我们生成固定的 mock 数据 (方便反推)
    # packed_weights: 随机 0~255
    packed_weights = torch.randint(0, 256, (packed_size,), dtype=torch.uint8, device="cpu")
    # absmax_q (由于是 uint8, mock 范围 0~255)
    absmax_q = torch.randint(0, 256, (num_blocks,), dtype=torch.uint8, device="cpu")
    # absmax2 (float16: mock 一些有效非零数值, e.g. 1.0 ~ 2.0)
    absmax2 = (torch.rand((num_groups,), dtype=torch.float32, device="cpu") + 1.0).to(torch.float16)
    # code2 (float16: mock 256 elements)
    code2 = (torch.rand((256,), dtype=torch.float32, device="cpu") + 1.0).to(torch.float16)
    
    offset_val = 0.0
    
    # 按照公式在 Python 端模拟解量化计算出 Ground Truth (fp16)
    print("Calculating Ground Truth in PyTorch...")
    
    # NF4 规范表
    nf4_table = torch.tensor([
        -1.0, -0.6961928, -0.52507305, -0.3949171, 
        -0.28444138, -0.18477343, -0.091050036, 0.0, 
        0.07958029, 0.1609302, 0.2461123, 0.33791524, 
        0.44070983, 0.562617, 0.72295684, 1.0
    ], dtype=torch.float32, device="cpu")
    
    # 解析出 idx0 和 idx1，展开到 total_elements
    idx0 = (packed_weights >> 4).to(torch.int64)
    idx1 = (packed_weights & 0x0F).to(torch.int64)
    
    # 交叉合并: [idx0_0, idx1_0, idx0_1, idx1_1, ...]
    unpacked_idx = torch.empty((total_elements,), dtype=torch.int64, device="cpu")
    unpacked_idx[0::2] = idx0
    unpacked_idx[1::2] = idx1
    
    # 计算所有元素的全局 block_id 和 group_id
    weight_indices = torch.arange(total_elements, device="cpu")
    block_ids = weight_indices // blocksize
    group_ids = block_ids // 256
    
    # 寻址并计算第一级缩放因子 S1 = (code2[absmax_q] * absmax2) + offset
    absmax_q_val = absmax_q.to(torch.int64)[block_ids]
    code2_val = code2[absmax_q_val].to(torch.float32)
    absmax2_val = absmax2[group_ids].to(torch.float32)
    
    S1 = (code2_val * absmax2_val) + offset_val
    
    # 计算最终值并转为 fp16 存储 (如果您 C++ 端用的是 bf16, 此处为了标准对比用 fp16 保存)
    # 因为 NumPy/C++ 标准流都更容易读写 IEEE fp16
    ground_truth = (nf4_table[unpacked_idx] * S1).to(torch.float16)

    # ---------------------------------------------
    # 写入二进制文件
    # ---------------------------------------------
    import os
    
    # 1. 写入 test_weights.bin
    bin_path = "test_weights.bin"
    print(f"Writing packed binaries to {bin_path}...")
    with open(bin_path, "wb") as f:
        # Header: num_rows(8) + num_cols(8) + blocksize(4) = 20 bytes
        f.write(struct.pack("qqi", num_rows, num_cols, blocksize))
        
        # Data
        f.write(packed_weights.numpy().tobytes())
        f.write(absmax_q.numpy().tobytes())
        f.write(absmax2.numpy().tobytes())
        f.write(code2.numpy().tobytes())
        f.write(struct.pack("f", offset_val))
        
    # 2. 写入 ground_truth.bin
    gt_path = "ground_truth.bin"
    print(f"Writing Ground Truth to {gt_path}...")
    with open(gt_path, "wb") as f:
        f.write(ground_truth.numpy().tobytes())
        
    # 3. 写入 params.txt
    params_path = "params.txt"
    print(f"Writing parameters to {params_path}...")
    with open(params_path, "w") as f:
        f.write(f"blocksize = {blocksize}\n")
        f.write("compute_type = \"bf16\"\n")  # 或者 fp16 根据您的内核实际情况
        f.write("target_gpu = \"A100\"\n")
        
    print("Done! Files generated:")
    print(" - test_weights.bin")
    print(" - ground_truth.bin")
    print(" - params.txt")

if __name__ == "__main__":
    num_rows = 4096
    num_cols = 4096
    blocksize = 64
    tensor_shape = (num_rows, num_cols)
    
    if torch.cuda.is_available():
        print(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")
        profile_bnb_baseline(tensor_shape, blocksize)
    else:
        print("CUDA is NOT available. Skipping BitsAndBytes profiling. Will only generate files.")
        
    create_mock_data_and_save(num_rows, num_cols, blocksize)

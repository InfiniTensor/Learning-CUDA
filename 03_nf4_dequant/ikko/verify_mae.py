import torch
import numpy as np
import os
import struct

def check_mae(output_dir="03_nf4_dequant/ikko/data", cuda_output_file="03_nf4_dequant/ikko/data/output.bin"):
    truth_file = os.path.join(output_dir, "ground_truth.bin")
    input_file = os.path.join(output_dir, "weight_data.bin")
    
    print("=== Starting Verification ===")
    
    # 1. 读取元数据以确定形状 (从输入文件读)
    with open(input_file, "rb") as f:
        # 读取前 20 字节 (rows:8, cols:8, blocksize:4)
        header = f.read(20)
        rows, cols, blocksize = struct.unpack("qqi", header)
    
    print(f"Shape: {rows}x{cols}, Blocksize: {blocksize}")
    
    # 2. 读取官方 Ground Truth
    # 假设它是 BF16 格式 (numpy 不直接支持 bf16，通常视具体情况处理)
    # 这里我们用 pytorch 读取，因为它支持 bf16
    with open(truth_file, "rb") as f:
        truth_bytes = f.read()
    # 将字节流转为 Tensor
    # 注意：Python 的 torch.frombuffer 可能会由于字节对齐问题报错，这里使用 numpy view 变通
    # (由于 numpy 无 bf16，我们假设文件存储的是原生字节，用 int16 读取再转 torch.bfloat16)
    truth_np = np.frombuffer(truth_bytes, dtype=np.int16).copy().reshape(rows, cols)
    truth_tensor = torch.from_numpy(truth_np).view(torch.bfloat16).float() # 转为 float32 用于计算 MAE
    
    # 3. 读取你的 CUDA Kernel 输出
    candidates = [
        cuda_output_file,
        os.path.join(output_dir, cuda_output_file),
        os.path.join(output_dir, "output.bin"),
        "output.bin",
    ]
    unique_candidates = []
    for path in candidates:
        if path not in unique_candidates:
            unique_candidates.append(path)

    cuda_path = None
    for path in unique_candidates:
        if os.path.exists(path):
            cuda_path = path
            break

    if cuda_path is None:
        print(f"Error: CUDA output file not found. Tried: {unique_candidates}")
        return

    with open(cuda_path, "rb") as f:
        cuda_bytes = f.read()
    
    # 检查文件大小是否匹配
    expected_size = rows * cols * 2 # BF16 = 2 bytes
    if len(cuda_bytes) != expected_size:
        print(f"Error: Output size mismatch! Expected {expected_size}, got {len(cuda_bytes)}")
        return

    cuda_np = np.frombuffer(cuda_bytes, dtype=np.int16).copy().reshape(rows, cols)
    cuda_tensor = torch.from_numpy(cuda_np).view(torch.bfloat16).float()
    
    # 4. 计算 MAE (Mean Absolute Error)
    diff = torch.abs(truth_tensor - cuda_tensor)
    mae = torch.mean(diff).item()
    
    # 计算最大误差
    max_diff = torch.max(diff).item()
    
    print("-" * 30)
    print(f"MAE (Mean Absolute Error): {mae:.6f}")
    print(f"Max Error:                 {max_diff:.6f}")
    print("-" * 30)
    
    # 5. 判定标准
    threshold = 1e-2
    if mae < threshold:
        print(f"✅ PASS: MAE ({mae:.6f}) is within threshold ({threshold})")
    else:
        print(f"❌ FAIL: MAE ({mae:.6f}) exceeds threshold ({threshold})")

if __name__ == "__main__":
    check_mae()
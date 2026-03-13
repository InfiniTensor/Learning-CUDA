import torch
from bitsandbytes import functional as F
import struct
import os
import math
from get_param import parse_params

def generate_data(config_file="./data/param.txt"):
    if not os.path.exists("./data"): os.makedirs("./data")
    if not os.path.exists("./data/log"): os.makedirs("./data/log")
    
    # 获取动态配置
    config = parse_params(config_file)
    blocksize, groupsize = config["blocksize"], config["groupsize"]
    rows, cols = config["rows"], config["cols"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if config["compute_type"] == "bf16" else torch.float16
    
    print(f"维度: {rows}x{cols} | Block: {blocksize} | Group: {groupsize}")
    
    # 权重生成与 Padding 对齐
    original_weights = torch.randn(rows, cols, dtype=dtype, device=device) * 0.5
    total_original = rows * cols
    alignment = blocksize * groupsize
    padded_total = math.ceil(total_original / alignment) * alignment
    
    if padded_total > total_original:
        print(f"填充 {padded_total - total_original} 个元素以对齐")
        flat_weights = original_weights.reshape(-1)
        padding = torch.zeros(padded_total - total_original, dtype=dtype, device=device)
        aligned_weights = torch.cat([flat_weights, padding], dim=0).reshape(1, -1)
    else:
        aligned_weights = original_weights.reshape(1, -1)

    # 一级量化 (NF4)
    q_weight, state_1 = F.quantize_4bit(aligned_weights, blocksize=blocksize, quant_type='nf4')
    # 二级量化 (FP8)
    absmax_1 = state_1.absmax
    absmax_q, state_2 = F.quantize_blockwise(absmax_1, blocksize=groupsize, nested=False)
    
    # 写入二进制文件
    code2_np = state_2.code.to(torch.float16).cpu().numpy()
    absmax2_np = state_2.absmax.to(torch.float16).cpu().numpy()
    
    bin_path = "./data/qlora_test.bin"
    with open(bin_path, "wb") as f:
        f.write(struct.pack("<qqi", rows, cols, blocksize)) 
        f.write(q_weight.cpu().numpy().tobytes())
        f.write(absmax_q.cpu().numpy().tobytes()) 
        f.write(absmax2_np.tobytes())
        f.write(code2_np.tobytes())
        f.write(struct.pack("<f", 0.0)) # offset_val
        
    print(f"数据已写入: {bin_path}")

    # 反量化 warm up
    for _ in range(5):
        # 反量化缩放因子
        state_1.absmax = F.dequantize_blockwise(absmax_q, state_2)
        
        # 反量化权重
        dequantized_padded = F.dequantize_4bit(q_weight, state_1)

    # 反量化
    print(f"--- 运行反量化 ---")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(10):
        # 反量化缩放因子
        state_1.absmax = F.dequantize_blockwise(absmax_q, state_2)
        
        # 反量化权重
        dequantized_padded = F.dequantize_4bit(q_weight, state_1)
    
    end_event.record()
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)

    # 性能统计
    bytes_read = (padded_total * 0.5) + absmax_q.numel() + (state_2.absmax.numel() * 2) + 512
    bytes_written = padded_total * 2
    gb_per_sec = ((bytes_read + bytes_written) * 10 / 1e9) / (elapsed_ms / 1000.0)
    
    print(f"耗时: {elapsed_ms:.4f} ms | 吞吐: {gb_per_sec:.2f} GB/s")

    # 写入性能日志
    with open("./data/log/log_py.txt", "w") as f:
        f.write(f"{elapsed_ms:.4f},{gb_per_sec:.4f}\n")

    # 裁剪并保存基准输出
    py_output = dequantized_padded.view(-1)[:total_original].reshape(rows, cols).to(torch.bfloat16).cpu()
    with open("./data/py_output.bin", "wb") as f:
        f.write(py_output.view(torch.int16).numpy().tobytes())
        
    print(f"基准输出已保存: ./data/py_output.bin")

if __name__ == "__main__":
    generate_data()
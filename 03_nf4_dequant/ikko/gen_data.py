import torch
import bitsandbytes as bnb
import struct
import os
import numpy as np

def generate_inputs(rows=16384, cols=16384, blocksize=64, output_dir="03_nf4_dequant/ikko/data", compute_type="bf16"):
    os.makedirs(output_dir, exist_ok=True)
    
    compute_type = compute_type.lower()
    if compute_type not in {"bf16", "fp16"}:
        raise ValueError("compute_type must be 'bf16' or 'fp16'")

    print(f"Generating data: {rows}x{cols}, blocksize={blocksize}, compute_type={compute_type}")
    
    # 1. 准备原始权重 (使用 GPU 和 BF16)
    device = torch.device("cuda")
    # 模拟真实权重分布 (Normal Float 4 针对正态分布优化)
    orig_weight = torch.randn(rows, cols, dtype=torch.bfloat16, device=device)
    
    # 2. 使用 bitsandbytes 进行 NF4 + Double Quantization
    # quant_type='nf4', compress_statistics=True 开启双重量化
    packed_weight, quant_state = bnb.functional.quantize_4bit(
        orig_weight, 
        blocksize=blocksize, 
        quant_type='nf4', 
        compress_statistics=True
    )
    
    # 3. 生成官方参考结果 (Ground Truth)
    # CUDA Kernel 输出必须逼近这个结果
    ref_output = bnb.functional.dequantize_4bit(
        packed_weight, 
        quant_state, 
        quant_type='nf4', 
        blocksize=blocksize
    )
    
    # 4. 提取双重量化参数 (为了写入 input bin 文件)
    # bitsandbytes 的 QuantState 结构解析:
    # - absmax: 一级量化因子 (已被二级量化，uint8)
    # - nested quant state: 兼容 nested_quant_state / state2
    absmax_q = quant_state.absmax.to(torch.uint8)  # uint8
    nested_state = getattr(quant_state, "nested_quant_state", None)
    if nested_state is None:
        nested_state = getattr(quant_state, "state2", None)
    if nested_state is None:
        raise RuntimeError(
            "Double-quantization state not found. "
            "Please ensure bitsandbytes supports compress_statistics=True "
            "and provides nested quantization fields."
        )
    absmax2 = nested_state.absmax  # float32 (需转 float16)
    code2 = nested_state.code      # float32 (需转 float16)
    offset = getattr(quant_state, "offset", 0.0)
    if isinstance(offset, torch.Tensor):
        offset = float(offset.item())
    
    # 5. 写入题目要求的二进制输入文件 (weight.bin)
    input_path = os.path.join(output_dir, "weight_data.bin")
    with open(input_path, "wb") as f:
        # [Header]
        f.write(struct.pack("qqi", rows, cols, blocksize))
        
        # [Data]
        # packed_weights (uint8)
        f.write(packed_weight.cpu().numpy().tobytes())
        # absmax_q (uint8)
        f.write(absmax_q.cpu().numpy().tobytes())
        # absmax2 (float16)
        f.write(absmax2.to(torch.float16).cpu().numpy().tobytes())
        # code2 (float16)
        f.write(code2.to(torch.float16).cpu().numpy().tobytes())
        # offset (float32)
        f.write(struct.pack("f", float(offset)))
        
    print(f"-> Input file saved to: {input_path}")
    
    # 6. 保存 Ground Truth 用于后续验证 (truth.bin)
    truth_path = os.path.join(output_dir, "ground_truth.bin")
    with open(truth_path, "wb") as f:
        # 保存为纯二进制流 (row-major, bf16/fp16)
        if compute_type == "bf16":
            ref_out = ref_output.to(torch.bfloat16)
            ref_bytes = ref_out.view(torch.int16).cpu().numpy().tobytes()
        else:
            ref_out = ref_output.to(torch.float16)
            ref_bytes = ref_out.cpu().numpy().tobytes()
        f.write(ref_bytes)
    
    print(f"-> Ground truth saved to: {truth_path}")

if __name__ == "__main__":
    generate_inputs(compute_type="bf16")
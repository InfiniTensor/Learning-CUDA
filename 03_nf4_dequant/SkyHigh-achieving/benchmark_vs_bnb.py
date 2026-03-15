
import torch
import time
import sys

def benchmark_bnb(rows=8192, cols=8192, repeats=50):
    try:
        import bitsandbytes as bnb
        from bitsandbytes.functional import dequantize_4bit, quantize_4bit
    except ImportError:
        print("bitsandbytes not installed. Run: pip install bitsandbytes")
        return None

    if not torch.cuda.is_available():
        print("CUDA not available")
        return None

    print(f"Benchmarking bitsandbytes on {torch.cuda.get_device_name(0)}...")
    
    # 生成 fp32 权重并量化
    device = torch.device("cuda:0")
    # fp16 input usually for weights in LLMs before quantization, but bnb quantizes from fp16/fp32
    w = torch.randn(rows, cols, device=device, dtype=torch.float16)
    
    # blocksize=64, quant_type='nf4'
    # quantize_4bit returns: (quantized_data, quantization_state)
    # The signature might vary by version, but usually it's input, blocksize, quant_type
    try:
        w_q, quant_state = bnb.functional.quantize_4bit(
            w.reshape(1, -1), blocksize=64, quant_type='nf4'
        )
    except TypeError:
         # Fallback for some versions
         w_q, quant_state = bnb.functional.quantize_4bit(
            w.reshape(1, -1), blocksize=64, quant_type='nf4', compress_statistics=True
        )

    # Warmup
    print("Warmup...")
    for _ in range(5):
        out = bnb.functional.dequantize_4bit(w_q, quant_state, quant_type='nf4')
    torch.cuda.synchronize()
    
    # Benchmark
    print("Benchmarking...")
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = bnb.functional.dequantize_4bit(w_q, quant_state, quant_type='nf4')
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    # Calculate metrics
    ms_per_call = (t1 - t0) / repeats * 1000
    
    # Data transfer: 
    # Read: 4-bit quantized data + quantization metadata (scales, absmax)
    # Write: FP16 output
    # Input size: rows * cols / 2 bytes (4-bit)
    # Output size: rows * cols * 2 bytes (fp16)
    # Metadata is negligible for bandwidth calculation usually, but strict calculation includes it.
    # For comparison with our kernel, we usually count load(compressed) + store(decompressed).
    
    numel = rows * cols
    bytes_in = numel // 2 # 0.5 bytes per element
    bytes_out = numel * 2 # 2 bytes per element
    total_bytes = bytes_in + bytes_out
    
    bw_gbs = (total_bytes) / (ms_per_call / 1000) / 1e9
    
    print(f"bitsandbytes dequantize_4bit ({rows}x{cols}, nf4, blocksize=64):")
    print(f"  Time: {ms_per_call:.3f} ms")
    print(f"  Bandwidth: {bw_gbs:.2f} GB/s")
    
    return ms_per_call, bw_gbs

if __name__ == "__main__":
    benchmark_bnb(8192, 8192)

import ctypes
import torch
import bitsandbytes.functional as F
import time
import numpy as np

# ===============================
# CUDA 库加载
# ===============================
lib = ctypes.cdll.LoadLibrary("./libnf4_cuda.so")

lib.nf4_dequant_cuda_double.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_float, ctypes.c_longlong, ctypes.c_int, ctypes.c_int, ctypes.c_void_p
]


# =========================================================
# 单次测试函数（任意 shape + 任意 blocksize）
# =========================================================
def run_case(rows, cols, blocksize, group_size=256,
             warmup_iters=50, test_iters=500):

    total = rows * cols

    weight = torch.randn(rows, cols, device="cuda", dtype=torch.float16)

    packed, state = F.quantize_4bit(
        weight,
        blocksize=blocksize,
        quant_type="nf4",
        compress_statistics=True
    )

    absmax_q = state.absmax.contiguous()
    absmax2 = state.state2.absmax.to(torch.float16).contiguous()
    code2 = state.state2.code.to(torch.float16).contiguous()

    offset = float(state.offset)

    out_cuda = torch.empty_like(weight)

    # ---------------- Warmup ----------------
    for _ in range(warmup_iters):
        _ = F.dequantize_4bit(packed, state)

        lib.nf4_dequant_cuda_double(
            packed.data_ptr(),
            absmax_q.data_ptr(),
            absmax2.data_ptr(),
            code2.data_ptr(),
            offset,
            total,
            blocksize,
            group_size,
            out_cuda.data_ptr()
        )

    torch.cuda.synchronize()

    # ---------------- bitsandbytes ----------------
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(test_iters):
        out_ref = F.dequantize_4bit(packed, state)
    end.record()

    torch.cuda.synchronize()
    ref_ms = start.elapsed_time(end) / test_iters

    # ---------------- custom ----------------
    start.record()
    for _ in range(test_iters):
        lib.nf4_dequant_cuda_double(
            packed.data_ptr(),
            absmax_q.data_ptr(),
            absmax2.data_ptr(),
            code2.data_ptr(),
            offset,
            total,
            blocksize,
            group_size,
            out_cuda.data_ptr()
        )
    end.record()

    torch.cuda.synchronize()
    cus_ms = start.elapsed_time(end) / test_iters

    mae = torch.mean(torch.abs(out_ref - out_cuda)).item()

    return ref_ms, cus_ms, mae


# =========================================================
# Sweep 多尺寸 + 多 blocksize
# =========================================================
def benchmark():

    shapes = [
        (102, 102),        # 任意 shape（非对齐）
        (512, 768),
        (1024, 1024),
        (2048, 1536),
        (4096, 4096),
    ]

    blocksizes = [64, 128]

    print("\n======== NF4 Dequant Benchmark ========\n")
    print("shape\t\tblock\tbnb(ms)\tcuda(ms)\tspeedup\tMAE")

    for rows, cols in shapes:
        for block in blocksizes:

            ref_ms, cus_ms, mae = run_case(rows, cols, block)

            speedup = ref_ms / cus_ms

            print(
                f"{rows}x{cols}\t{block}\t"
                f"{ref_ms:.4f}\t"
                f"{cus_ms:.4f}\t"
                f"{speedup:.2f}x\t"
                f"{mae:.2e}"
            )


# =========================================================
if __name__ == "__main__":
    benchmark()
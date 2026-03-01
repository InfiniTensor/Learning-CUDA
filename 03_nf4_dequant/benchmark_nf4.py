import ctypes
import torch
import bitsandbytes.functional as F
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# CUDA 动态库加载
# =========================================================
lib = ctypes.cdll.LoadLibrary("./libnf4_cuda.so")

lib.nf4_dequant_cuda_double.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_float, ctypes.c_longlong, ctypes.c_int, ctypes.c_int, ctypes.c_void_p
]
lib.init_nf4_lut.argtypes = []
lib.init_nf4_lut.restype  = None

# =========================================================
# 单次测试
# =========================================================
def run_case(rows, cols, blocksize,
             group_size=256,
             warmup_iters=10,
             test_iters=500):

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
# 画 Speedup 曲线
# =========================================================
def plot_speedup(df):

    plt.figure(figsize=(7, 5))

    for block in df["Block"].unique():

        sub = df[df["Block"] == block]

        # x 轴用参数规模（更合理）
        x = [int(s.split("x")[0]) * int(s.split("x")[1]) for s in sub["Shape"]]
        y = sub["Speedup"]

        plt.plot(x, y, marker='o', label=f"block={block}")

    plt.xscale("log")

    plt.xlabel("Number of elements (log scale)")
    plt.ylabel("Speedup (bnb / cuda)")
    plt.title("NF4 Dequant Speedup vs Matrix Size")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig("speedup.png", dpi=150)
    plt.show()


# =========================================================
# 主 benchmark
# =========================================================
def benchmark():
    lib.init_nf4_lut()
    shapes = [
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 8192),
    ]

    blocksizes = [64, 128]

    results = []

    print("Running benchmark...\n")

    for rows, cols in shapes:
        for block in blocksizes:

            ref_ms, cus_ms, mae = run_case(rows, cols, block)

            speedup = ref_ms / cus_ms

            results.append([
                f"{rows}x{cols}",
                block,
                ref_ms,
                cus_ms,
                speedup,
                mae
            ])

    # ---------------- 表格 ----------------
    df = pd.DataFrame(
        results,
        columns=[
            "Shape",
            "Block",
            "bnb (ms)",
            "cuda (ms)",
            "Speedup",
            "MAE"
        ]
    )

    pd.set_option('display.float_format', lambda x: f'{x:.8f}')

    print("\n======== NF4 Dequant Benchmark ========\n")
    print(df.to_string(index=False))

    # df.to_csv("nf4_benchmark.csv", index=False)
    # print("\n结果已保存到 nf4_benchmark.csv")

    # # ---------------- 曲线图 ----------------
    # plot_speedup(df)


# =========================================================
if __name__ == "__main__":
    benchmark()
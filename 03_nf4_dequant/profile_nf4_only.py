import ctypes
import torch
import bitsandbytes.functional as F

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
# 单次运行函数（供 ncu 分析）
# =========================================================
def run_nf4_dequant():
    lib.init_nf4_lut()
    # 配置参数（固定一个典型规模）
    rows, cols = 4096, 4096  # 16M 参数
    blocksize = 64
    group_size = 256
    
    print(f"初始化数据: {rows}x{cols}, blocksize={blocksize}")
    
    # 准备数据
    weight = torch.randn(rows, cols, device="cuda", dtype=torch.float16)
    packed, state = F.quantize_4bit(
        weight,
        blocksize=blocksize,
        quant_type="nf4",
        compress_statistics=True
    )
    
    # 提取统计量
    absmax_q = state.absmax.contiguous()
    absmax2 = state.state2.absmax.to(torch.float16).contiguous()
    code2 = state.state2.code.to(torch.float16).contiguous()
    offset = float(state.offset)
    out_cuda = torch.empty_like(weight)
    
    print("数据准备完成，等待 ncu 分析...")
    print("\n" + "="*50)
    print("现在运行 ncu 命令:")
    print(f"ncu --target-processes all --kernel-name nf4_dequant_fast_kernel_fp16 python profile_nf4_only.py")
    print("="*50 + "\n")
    
    
    # 只运行一次自定义内核（ncu 会捕获这次执行）
    print("执行 nf4_dequant_cuda_double...")
    lib.nf4_dequant_cuda_double(
        packed.data_ptr(),
        absmax_q.data_ptr(),
        absmax2.data_ptr(),
        code2.data_ptr(),
        offset,
        rows * cols,  # total
        blocksize,
        group_size,
        out_cuda.data_ptr()
    )
    torch.cuda.synchronize()
    
    print("内核执行完成！")
    
    # 可选：验证结果
    out_ref = F.dequantize_4bit(packed, state)
    mae = torch.mean(torch.abs(out_ref - out_cuda)).item()
    print(f"MAE 误差: {mae:.6e}")


# =========================================================
if __name__ == "__main__":
    run_nf4_dequant()
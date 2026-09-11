'''程序入口'''

import argparse
import os
import io_utils as iu
import time
import quant_cpu as qc
import quant_cuda as qg

def compare(packed, scales, dequant, packed_cuda, scales_cuda, dequant_cuda):
    n = min(len(packed), len(packed_cuda))
    print(f"[cpu] packed 长度 {len(packed)} 字节, [cuda] packed 长度 {len(packed_cuda)} 字节")
    if packed == packed_cuda:
        print(f"cuda packed: 字节完全一致 ({n} 字节)")
    else:
        diff_idx = [i for i in range(n) if packed[i] != packed_cuda[i]]
        print(f"cuda packed: {len(diff_idx)}/{n} 字节不同 ({(len(diff_idx) / n * 100):.4f}%)")
        for i in diff_idx[:10]:
            print(f"  位置 {i:>8}   cpu=0x{packed[i]:02X}  cuda=0x{packed_cuda[i]:02X}")
        if len(diff_idx) > 10:
            print(f"  ... 还有 {len(diff_idx) - 10} 处不同，未列出")

    d = max(abs(a - b) for a, b in zip(scales, scales_cuda))
    print(f"cuda scales: {'PASS' if d <= 1e-6 else 'FAIL'}   最大差 {d:.3e}")

    d2 = max(abs(a - b) for a, b in zip(dequant, dequant_cuda))
    print(f"cuda dequant: {'PASS' if d2 <= 1e-6 else 'FAIL'}   最大差 {d2:.3e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nvfp4_block.txt", help="量化参数文件")
    args = parser.parse_args()

    outputs = "outputs"
    os.makedirs(outputs, exist_ok=True)

    # 数据获取
    data, rows, cols, dtype = iu.read_matrix_file()

    # 读取一些配置文件
    cfg = iu.read_config(args.config)
    format = cfg.get("format")
    block_size = int(cfg.get("block_size"))
    scale_mode = cfg.get("scale_mode")
    rounding = cfg.get("rounding")
    output_type = cfg.get("output_type")

    # ========== cpu端，主要是用来进行对比的 ================
    # 量化
    t0 = time.perf_counter()
    packed, scales = qc.quant(data, rows, cols, format, block_size, scale_mode, rounding)
    t1 = time.perf_counter()

    # 反量化
    dequant = qc.dequant(packed, scales, rows, cols, format, block_size)
    t2 = time.perf_counter()

    # 误差计算
    quant_time_cpu = t1 - t0
    dequant_time_cpu = t2 - t1
    input_bytes = rows * cols * (4 if dtype == "fp32" else 2)
    dequant_bytes = len(packed) + len(scales) * 4
    max_abs, mae, mse, compression, bw = iu.compute_error(
        data, dequant, input_bytes, dequant_bytes, 
        quant_time_cpu
    )

    # 保存
    iu.save_file(os.path.join(outputs, f"quantized_{format}.bin"), scales, packed, rows, cols, block_size, format)
    iu.save_matrix_file(dequant, os.path.join(outputs, f"dequant_{format}.bin"), rows, cols, output_type)
    iu.save_error(os.path.join(outputs, f"report_{format}.log"), max_abs, mae, mse, compression, quant_time_cpu, dequant_time_cpu, bw)

    # =========== cuda端 =============================
    packed_cuda, scales_cuda, dequant_cuda, quant_time_cuda, dequant_time_cuda = qg.quant_dequant(data, rows, cols, format, scale_mode, block_size, output_type)
    dequant_bytes = len(packed_cuda) + len(scales_cuda) * 4
    max_abs, mae, mse, compression, bw = iu.compute_error(
        data, dequant_cuda, input_bytes, dequant_bytes, 
        quant_time_cuda
    )
    iu.save_error(os.path.join(outputs, f"report_{format}_gpu.log"), max_abs, mae, mse, compression, quant_time_cuda, dequant_time_cuda, bw)
    # 比较一下两端的结果
    compare(packed, scales, dequant, packed_cuda, scales_cuda, dequant_cuda)
    
    iu.save_file(os.path.join(outputs, f"quantized_cuda_{format}.bin"), scales_cuda, packed_cuda, rows, cols, block_size, format)
    iu.save_matrix_file(dequant_cuda, os.path.join(outputs, f"dequantized_cuda_{format}.bin"), rows, cols, output_type)


if __name__ == "__main__": main()
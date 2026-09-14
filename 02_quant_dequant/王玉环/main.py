'''程序入口'''
import argparse
import os
import io_utils
import time
import quant_cpu
import quant_cuda

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mxfp8_block.txt", help="量化参数文件")
    parser.add_argument("--matrix", choices=["random", "normal", "outlier"], default="outlier")
    args = parser.parse_args()

    # 数据获取
    path = f"inputs/{args.matrix}.bin"
    data, rows, cols, dtype = io_utils.read_matrix_file(path)

    # 读取一些配置文件
    cfg = io_utils.read_config(args.config)
    format = cfg.get("format")
    block_size = int(cfg.get("block_size"))
    scale_mode = cfg.get("scale_mode")
    rounding = cfg.get("rounding")
    output_type = cfg.get("output_type")

    outputs = f"outputs/{args.matrix}/{format}"
    os.makedirs(outputs, exist_ok=True)
    os.makedirs(os.path.join(outputs, "cpu"), exist_ok=True)
    os.makedirs(os.path.join(outputs, "cuda"), exist_ok=True)

    # ========== cpu端，主要是用来进行对比的 ================
    # 量化
    t0 = time.perf_counter()
    packed, scales = quant_cpu.quant(data, rows, cols, format, block_size, scale_mode, rounding)
    t1 = time.perf_counter()

    # 反量化
    dequant = quant_cpu.dequant(packed, scales, rows, cols, format, block_size)
    t2 = time.perf_counter()

    # 误差计算
    quant_time_cpu = t1 - t0
    dequant_time_cpu = t2 - t1
    input_bytes = rows * cols * (4 if dtype == "fp32" else 2)
    dequant_bytes = len(packed) + len(scales) * 4
    max_abs, mae, mse, compression, bw = io_utils.compute_error(
        data, dequant, input_bytes, dequant_bytes, 
        quant_time_cpu
    )

    # 保存
    io_utils.save_file(os.path.join(outputs, "cpu/quantized.bin"), scales, packed, rows, cols, block_size, format)
    io_utils.save_matrix_file(dequant, os.path.join(outputs, "cpu/dequant.bin"), rows, cols, output_type)
    io_utils.save_error(os.path.join(outputs, "cpu/report.log"), max_abs, mae, mse, compression, quant_time_cpu, dequant_time_cpu, bw)

    # =========== cuda端 =============================
    packed_cuda, scales_cuda, dequant_cuda, quant_time_cuda, dequant_time_cuda = quant_cuda.quant_dequant(data, rows, cols, format, scale_mode, block_size, output_type)
    dequant_bytes = len(packed_cuda) + len(scales_cuda) * 4
    max_abs, mae, mse, compression, bw = io_utils.compute_error(
        data, dequant_cuda, input_bytes, dequant_bytes, 
        quant_time_cuda
    )

    io_utils.save_error(os.path.join(outputs, "cuda/report.log"), max_abs, mae, mse, compression, quant_time_cuda, dequant_time_cuda, bw)
    io_utils.save_file(os.path.join(outputs, "cuda/quantized.bin"), scales_cuda, packed_cuda, rows, cols, block_size, format)
    io_utils.save_matrix_file(dequant_cuda, os.path.join(outputs, "cuda/dequantized.bin"), rows, cols, output_type)

if __name__ == "__main__": main()
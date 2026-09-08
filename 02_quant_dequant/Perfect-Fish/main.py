'''程序入口'''

import argparse
import os
import data_utils as du
import io_utils as iu
import time
import quant_cpu as qc

def make_matrix(matrix, rows, cols, dtype):
    if matrix == "random": return du.gen_random(rows, cols, dtype)
    if matrix == "normal": return du.gen_normal(rows, cols, dtype)
    if matrix == "outlier": return du.gen_outlier(rows, cols, dtype)
    raise ValueError(f"未知矩阵类型：{matrix}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", choices=["random", "normal", "outlier"], help="生成三种矩阵", default="random")
    parser.add_argument("--rows", default=256)
    parser.add_argument("--cols", default=256)
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--config", default="configs/nvfp4_block.txt", help="量化参数文件")
    args = parser.parse_args()

    outputs = "outputs"
    inputs = "inputs"
    os.makedirs(outputs, exist_ok=True)
    os.makedirs(inputs, exist_ok=True)

    # 生成数据
    matrix, rows, cols, dtype = args.matrix, args.rows, args.cols, args.dtype
    data = make_matrix(matrix, rows, cols, dtype)

    # 写成二进制文件
    vals = [x for row in data for x in row]
    input_path = os.path.join(inputs, "input.bin")
    iu.write_matrix_file(vals, input_path, rows, cols, dtype)

    # 读取一些配置文件
    cfg = iu.read_config(args.config)
    format = cfg.get("format")
    block_size = int(cfg.get("block_size"))
    scale_mode = cfg.get("scale_mode")
    rounding = cfg.get("rounding")
    output_type = cfg.get("output_type")

    # 量化
    t0 = time.perf_counter()
    packed, scales = qc.quant(data, format, block_size, scale_mode, rounding)
    t1 = time.perf_counter()

    # 反量化
    dequant = qc.dequant(packed, scales, rows, cols, format, block_size)
    t2 = time.perf_counter()

    # 误差计算
    max_abs, mae, mse = iu.error(vals, dequant)
    input_bytes = rows * cols * (4 if dtype == "fp32" else 2)
    dequant_bytes = len(packed) + len(scales) * 4
    compression = input_bytes / dequant_bytes

    # 保存
    iu.save_file(os.path.join(outputs, "quantized.bin"), scales, packed, rows, cols, block_size, format)
    iu.write_matrix_file(dequant, os.path.join(outputs, "dequant,bin"), rows, cols, output_type)
    iu.save_error(os.path.join(outputs, f"report_{format}.log"), max_abs, mae, mse, compression, t0, t1, t2)

if __name__ == "__main__": main()
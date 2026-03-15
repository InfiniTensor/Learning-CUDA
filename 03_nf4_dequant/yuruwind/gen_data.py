#!/usr/bin/python3
import numpy as np
import struct
import argparse

# NF4 标准查找表 
NF4_TABLE = np.array([
    -1.0, -0.69487101, -0.51209301, -0.37391701,
    -0.25611401, -0.14725500, -0.04162400, 0.06282201,
    0.16859101, 0.28551400, 0.40619302, 0.53675699,
    0.68502200, 0.87091398, 1.0, 0.0
], dtype=np.float32)

def gen_test_data(rows, cols, blocksize=64, group_size=256):
    num_elements = rows * cols
    num_blocks = (num_elements + blocksize - 1) // blocksize
    num_groups = (num_blocks + group_size - 1) // group_size

    # 随机生成 4-bit 索引 (0-15)
    indices = np.random.randint(0, 16, size=num_elements, dtype=np.uint8)
    
    # 打包权重: 处理奇数个元素的情况
    num_bytes = (num_elements + 1) // 2
    packed_weights = np.zeros(num_bytes, dtype=np.uint8)
    
    for i in range(num_elements // 2):
        packed_weights[i] = (indices[2*i+1] << 4) | (indices[2*i] & 0x0F)
        
    # 如果总数是奇数，单独处理最后半个 byte (放在低 4 位)
    if num_elements % 2 != 0:
        packed_weights[-1] = indices[-1] & 0x0F

    # 生成二级缩放因子数据
    # absmax_q (每块一个索引，对应 code2)
    absmax_q = np.random.randint(0, 256, size=num_blocks, dtype=np.uint8)
    # code2 (二级码表，256个 fp16)
    code2 = np.random.randn(256).astype(np.float16)
    # absmax2 (每组一个 fp16)
    absmax2 = np.random.randn(num_groups).astype(np.float16)
    offset = 0.0

    # 计算 Ground Truth (用于验证)
    gt_weights = np.zeros(num_elements, dtype=np.float32)
    for i in range(num_elements):
        b_idx = i // blocksize
        g_idx = b_idx // group_size
        scale = f16_to_f32(code2[absmax_q[b_idx]]) * f16_to_f32(absmax2[g_idx])
        gt_weights[i] = NF4_TABLE[indices[i]] * scale

    with open("input.bin", "wb") as f:
        # Header: rows(i64), cols(i64), blocksize(i32)
        f.write(struct.pack("qqi", rows, cols, blocksize))
        f.write(packed_weights.tobytes())
        f.write(absmax_q.tobytes())
        f.write(code2.tobytes())
        f.write(absmax2.tobytes())
        f.write(struct.pack("f", offset))

    gt_weights.astype(np.float16).tofile("gt_output.bin")
    print(f"Successfully generated: {rows}x{cols} ({num_elements} elements)")
    print("Files saved: input.bin, gt_output.bin")

def f16_to_f32(val): # 辅助转换
    return np.array(val, dtype=np.float16).astype(np.float32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NF4 test data.")
    parser.add_argument("rows", type=int, help="Number of rows")
    parser.add_argument("cols", type=int, help="Number of columns")
    parser.add_argument(
        "--blocksize",
        type=int,
        default=64,
        help="Block size (default: 64)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=256,
        help="Group size (default: 256)",
    )

    args = parser.parse_args()

    gen_test_data(args.rows, args.cols, args.blocksize, args.group_size)
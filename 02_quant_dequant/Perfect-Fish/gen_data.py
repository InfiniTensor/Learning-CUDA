'''生成三种类型矩阵'''
import os
import argparse
import random
import struct

# 处理输入的数据类型
def _cast(data, dtype):
    if dtype == "fp16":
        out = []
        for row in data:
            new_row = []
            for x in row:
                try:
                    val = struct.unpack("e", struct.pack("e", float(x)))[0]
                except OverflowError:
                    val = 65504.0 if x > 0 else -65504.0
                new_row.append(val)
            out.append(new_row)
        return out
    
    return data


# 随机矩阵
def _gen_random(rows, cols, dtype = "fp32"):
    rr = random.Random(0)
    data = [[rr.random() * 2 - 1 for _ in range(cols)] for _ in range(rows)]
    return _cast(data, dtype)


# 正态分布矩阵
def _gen_normal(rows, cols, dtype = "fp32"):
    rr = random.Random(1)
    data = [[rr.gauss(0.0, 1.0) for _ in range(cols)] for _ in range(rows)]
    return _cast(data, dtype)

# 含有异常值矩阵
def _gen_outlier(rows, cols, dtype = "fp32"):
    rr = random.Random(2)
    data = [[rr.gauss(0.0, 1.0) for _ in range(cols)] for _ in range(rows)]
    # 随机挑选几个异常值
    for _ in range(8): 
        r, c = rr.randrange(rows), rr.randrange(cols)
        data[r][c] = 1000.0 if rr.choice([True, False]) else -1000.0
    return _cast(data, dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", choices=["random", "normal", "outlier"], help="生成三种矩阵", default="random")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--out", default="inputs/input.bin")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.matrix == "random":
        data = _gen_random(args.rows, args.cols, args.dtype)
    elif args.matrix == "normal":
        data = _gen_normal(args.rows, args.cols, args.dtype)
    else:
        data = _gen_outlier(args.rows, args.cols, args.dtype)
    with open(args.out, "wb") as f:
        f.write(f"num_rows: {args.rows}\nnum_cols: {args.cols}\ndtype: {args.dtype}\n[data]\n".encode())
        fmt = "f" if args.dtype == "fp32" else "e"
        for row in data:
            for x in row:
                f.write(struct.pack(fmt, float(x)))

if __name__ == "__main__":
    main()
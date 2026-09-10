'''生成三种类型矩阵'''

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
def gen_random(rows, cols, dtype = "fp32"):
    rr = random.Random(0)
    data = [[rr.random() * 2 - 1 for _ in range(cols)] for _ in range(rows)]
    return _cast(data, dtype)


# 正态分布矩阵
def gen_normal(rows, cols, dtype = "fp32"):
    rr = random.Random(1)
    data = [[rr.gauss(0.0, 1.0) for _ in range(cols)] for _ in range(rows)]
    return _cast(data, dtype)

# 含有异常值矩阵
def gen_outlier(rows, cols, dtype = "fp32"):
    rr = random.Random(2)
    data = [[rr.gauss(0.0, 1.0) for _ in range(cols)] for _ in range(rows)]
    # 随机挑选几个异常值
    for _ in range(8): 
        r, c = rr.randrange(rows), rr.randrange(cols)
        data[r][c] = 1000.0 if rr.choice([True, False]) else -1000.0
    return _cast(data, dtype)
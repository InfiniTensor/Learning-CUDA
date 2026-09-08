'''定义输入输出文件的格式'''

import struct

def write_matrix_file(vals, path, rows, cols, dtype):
    with open(path, "wb") as f:
        f.write(("num_rows: %d\n" % rows).encode())
        f.write(("num_cols: %d\n" % cols).encode())
        f.write(("dtype: %s\n" % dtype).encode())
        f.write(b"[data]\n")
        fmt = "f" if dtype == "fp32" else "e"
        for v in vals:
            f.write(struct.pack(fmt, float(v)))

def read_config(path):
    cfg = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.split("=", 1)
            cfg[key.strip()] = value.strip()
    return cfg

def save_file(path, scales, packed, rows, cols, block_size, format):
    with open(path, "wb") as f:
        f.write(("format: %s\n" % format).encode())
        f.write(("rows: %d\n" % rows).encode())
        f.write(("cols: %d\n" % cols).encode())
        f.write(("block_size: %d\n" % block_size).encode())
        f.write(("packed_len: %d\n" % len(packed)).encode())
        f.write(("scale_count: %d\n" % len(scales)).encode())
        f.write(b"[packed]\n")
        f.write(bytes(packed))
        f.write(b"[scales]\n")
        for s in scales:
            f.write(struct.pack("<f", float(s)))

def error(original, dequant):
    n = len(original)
    max_abs, sum_abs, sum_sq = 0.0, 0.0, 0.0
    for a, b in zip(original, dequant):
        d = abs(a - b)
        max_abs = max(max_abs, d)
        sum_abs += d
        sum_sq += d * d
    return max_abs, sum_abs / n, sum_sq / n

def save_error(path, max_abs, mae, mse, compression, t0, t1, t2):
    with open(path, "w", encoding="utf-8") as f:
        f.write("max_abs_error = %.9g\n" % max_abs)
        f.write("mae = %.9g\n" % mae)
        f.write("mse = %.9g\n" % mse)
        f.write("compression_ratio = %.4f\n" % compression)
        f.write("quant_time_s = %.9g\n" % (t1 - t0))
        f.write("dequant_time_S = %.9g\n" % (t2 - t1))
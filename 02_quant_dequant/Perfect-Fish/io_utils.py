'''处理一些输入输出文件，以及保存'''

import struct

def read_matrix_file():
    path = "inputs/input.bin"
    with open(path, "rb") as f:
        rows = cols = 0
        dtype = "fp32"
        while True:
            line = f.readline()
            if not line: break
            s = line.decode().strip()
            if s.startswith("num_rows:"):
                rows = int(s.split(":")[1].strip())
            elif s.startswith("num_cols:"):
                cols = int(s.split(":")[1].strip())
            elif s.startswith("dtype:"):
                dtype = s.split(":")[1].strip()
            elif s == "[data]":
                break
        n = rows * cols
        fmt = "f" if dtype == "fp32" else "e"
        size = 4 if dtype == "fp32" else 2
        data = f.read(n * size)
        data = [struct.unpack_from(fmt, data, i * size)[0] for i in range(n)]        

    return data, rows, cols, dtype

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

def compute_error(original, dequant, input_bytes, dequant_bytes, quant_time):
    n = len(original)
    max_abs, sum_abs, sum_sq = 0.0, 0.0, 0.0
    for a, b in zip(original, dequant):
        d = abs(a - b)
        max_abs = max(max_abs, d)
        sum_abs += d
        sum_sq += d * d
    compression = input_bytes / dequant_bytes
    bw = (input_bytes+dequant_bytes) / max(quant_time, 1e-9) / 1e9
    return max_abs, sum_abs / n, sum_sq / n, compression, bw

def save_error(path, max_abs, mae, mse, compression, quant_time, dequant_time, bw):
    with open(path, "w", encoding="utf-8") as f:
        f.write("max_abs_error = %.9g\n" % max_abs)
        f.write("mae = %.9g\n" % mae)
        f.write("mse = %.9g\n" % mse)
        f.write("compression_ratio = %.4f\n" % compression)
        f.write("quant_time_s = %.9g\n" % quant_time)
        f.write("dequant_time_S = %.9g\n" % dequant_time)
        f.write("quant_bandwidth_gbps = %.4f\n" % bw)

def save_matrix_file(dequant, path, rows, cols, dtype):
    with open(path, "wb") as f:
        f.write(("num_rows: %d\n" % rows).encode())
        f.write(("num_cols: %d\n" % cols).encode())
        f.write(("dtype: %s\n" % dtype).encode())
        f.write(b"[data]\n")
        fmt = "f" if dtype == "fp32" else "e"
        for v in dequant:
            f.write(struct.pack(fmt, float(v)))

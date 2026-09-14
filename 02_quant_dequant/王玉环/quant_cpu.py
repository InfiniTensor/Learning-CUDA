import quant_common as qc
import random

def quant(data, rows, cols, format, block_size, scale_mode, rounding, seed = 2026):
    codes = []
    scales = []
    rng = random.Random(seed) if rounding == "stochastic" else None
    global_scale = None
    if scale_mode == "tensor":
        gmax = max(abs(x) for x in data)
        global_scale = gmax / qc.MAX_FENMU[format] if gmax > 0 else 1.0

    for r in range(rows):
        for b in range(0, cols, block_size):
            start = r*cols + b
            blk = data[start:start+block_size]

            if scale_mode == "block":
                bmax = max(abs(x) for x in blk)
                s = bmax / qc.MAX_FENMU[format] if bmax > 0 else 1.0
            else: s = global_scale
            scales.append(s)

            # 对应的数字
            for x in blk:
                if format == "nvfp4":
                    codes.append(qc.float_to_e2m1(x / s, rng))
                else: codes.append(qc.float_to_e4m3(x / s, rng))

        element_bits = 4 if format == "nvfp4" else 8
    return qc.pack(codes, element_bits), scales

def dequant(packed, scales, rows, cols, format, block_size):
    element_bits = 4 if format == "nvfp4" else 8
    codes = qc.unpack(packed, element_bits, rows*cols)

    # 反量化直接乘上缩放因子就行
    out = []
    for i, code in enumerate(codes):
        r = i // cols
        c = i % cols
        s = scales[i // block_size]
        if format == "nvfp4":
            out.append(qc.e2m1_to_float(code) * s)
        else: out.append(qc.e4m3_to_float(code) * s)
    return out

#!/usr/bin/env python3
"""独立交叉验证：用第三方标准实现复现编码，与程序输出的码流逐字节比对。

要点：
  * **不依赖** 本项目自身的编解码实现，属于外部参考验证。
  * E4M3 / E5M2 的舍入使用 ml_dtypes（OCP FP8 标准实现）作为参考。
  * E2M1 (FP4) 使用标准可表示幅值表 {0,.5,1,1.5,2,3,4,6} + 最近偶数舍入的
    独立 numpy 实现作为参考（ml_dtypes 的 float4 为存储型，转换支持有限）。
  * 校验两件事：
      1) 程序写出的 block 缩放因子字节是否需要重算一致；
      2) 使用程序自己写出的缩放因子，独立计算每个元素的编码字节，
         与程序打包的码流比对 —— 这直接检验编码器语义是否符合标准。

用法:
    python3 tools/cross_check.py <tensor_file> <quant_file> [--limit N]
"""
import argparse
import struct
import sys

import numpy as np

try:
    import ml_dtypes
except ImportError:
    print("需要 ml_dtypes：python3 -m pip install ml_dtypes", file=sys.stderr)
    sys.exit(2)

# 与 include/io.h 中 QuantHeader 一致（#pragma pack(1)，共 72 字节）
QHDR_FMT = "<4siiqqiiiiqfqq"
QHDR_SIZE = 72

# E2M1 标准可表示幅值
FP4_VALS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def read_tensor(path):
    with open(path, "rb") as f:
        blob = f.read()
    pos = blob.find(b"[data]")
    if pos < 0:
        raise ValueError("张量文件缺少 [data] 段")
    head = blob[:pos].decode("ascii", "ignore")
    start = blob.find(b"\n", pos) + 1
    rows = cols = None
    dtype = "fp32"
    for line in head.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip().lower(), v.strip()
        if k == "num_rows":
            rows = int(v)
        elif k == "num_cols":
            cols = int(v)
        elif k == "dtype":
            dtype = v
    n = rows * cols
    if dtype == "fp16":
        arr = np.frombuffer(blob[start:start + n * 2], dtype=np.float16).astype(np.float32)
    else:
        arr = np.frombuffer(blob[start:start + n * 4], dtype=np.float32)
    return rows, cols, arr.astype(np.float32).copy()


def read_quant(path):
    with open(path, "rb") as f:
        blob = f.read()
    (magic, fmt, elem, rows, cols, bs, smode, odt, rmode,
     nb, gs, pb, sb) = struct.unpack(QHDR_FMT, blob[:QHDR_SIZE])
    if magic != b"LPQ1":
        raise ValueError(f"magic 不匹配: {magic!r}")
    packed = np.frombuffer(blob[QHDR_SIZE:QHDR_SIZE + pb], dtype=np.uint8).copy()
    scales = np.frombuffer(blob[QHDR_SIZE + pb:QHDR_SIZE + pb + sb], dtype=np.uint8).copy()
    return dict(format=fmt, elem=elem, rows=rows, cols=cols, block_size=bs,
                scale_mode=smode, out_dtype=odt, round_mode=rmode, num_blocks=nb,
                gscale=np.float32(gs), packed=packed, scales=scales)


def e8m0_to_f32(b):
    """独立解码 E8M0（含 c==0 的极小值路径）。"""
    b = np.asarray(b, dtype=np.int64)
    out = np.ldexp(np.float32(1.0), (b - 127).astype(np.int32)).astype(np.float32)
    out = np.where(b == 0, np.float32(np.ldexp(np.float32(1.0), -127)), out)
    return out.astype(np.float32)


def e4m3_byte_to_f32(b):
    """用 ml_dtypes 独立解码 E4M3 字节。"""
    return np.asarray(b, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32)


def fp4_encode_mag(a):
    """E2M1 幅值编码（最近偶数舍入），独立 numpy 实现。a >= 0。"""
    a = np.asarray(a, dtype=np.float32)
    idx = np.searchsorted(FP4_VALS, a, side="left").astype(np.int64)
    idx = np.clip(idx, 1, 7)
    lo = FP4_VALS[idx - 1]
    hi = FP4_VALS[idx]
    dl = a - lo
    dh = hi - a
    take_hi = dh < dl
    tie = (dh == dl)
    take_hi = np.where(tie, (idx % 2) == 0, take_hi)   # 中点取编码为偶数的一侧
    code = np.where(take_hi, idx, idx - 1).astype(np.uint8)
    return np.where(a >= np.float32(6.0), np.uint8(7), code).astype(np.uint8)


def check_mxfp8(x, q):
    n = x.size
    bs = q["block_size"]
    nb = q["num_blocks"]
    pad = np.zeros(nb * bs, dtype=np.float32)
    pad[:n] = x
    blocks = pad.reshape(nb, bs)

    # --- (1) 独立重算 block 缩放因子，与程序写出的字节比对 ---
    amax = np.max(np.abs(blocks), axis=1).astype(np.float32)
    maxv = np.float32(448.0) if q["elem"] == 0 else np.float32(57344.0)
    r = (amax / maxv).astype(np.float32)
    m, e2 = np.frexp(r)
    e = np.where(m == np.float32(0.5), e2 - 1, e2).astype(np.int32)
    e = np.where(amax > 0, np.clip(e, -127, 127), -127).astype(np.int32)
    exp_byte = (e + 127).astype(np.uint8)
    scale_mismatch = int(np.count_nonzero(exp_byte != q["scales"]))

    # --- (2) 用程序写出的缩放因子独立编码，与打包码流比对 ---
    scale = e8m0_to_f32(q["scales"])
    inv = (np.float32(1.0) / scale).astype(np.float32)
    scaled = (blocks * inv[:, None]).astype(np.float32).reshape(-1)[:n]
    f8 = ml_dtypes.float8_e4m3fn if q["elem"] == 0 else ml_dtypes.float8_e5m2
    codes = scaled.astype(f8).view(np.uint8)
    code_mismatch = int(np.count_nonzero(codes != q["packed"]))
    return scale_mismatch, code_mismatch, nb, "E4M3" if q["elem"] == 0 else "E5M2"


def check_nvfp4(x, q):
    n = x.size
    bs = q["block_size"]
    nb = q["num_blocks"]
    gscale = q["gscale"]
    pad = np.zeros(nb * bs, dtype=np.float32)
    pad[:n] = x
    blocks = pad.reshape(nb, bs)

    # --- (1) 独立重算 block 缩放因子字节 ---
    bmax = np.max(np.abs(blocks), axis=1).astype(np.float32)
    target = (bmax / np.float32(6.0)).astype(np.float32)
    ratio = (target / gscale).astype(np.float32)
    ratio = np.where(bmax > 0, ratio, np.float32(0.0))
    exp_byte = ratio.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    scale_mismatch = int(np.count_nonzero(exp_byte != q["scales"]))

    # --- (2) 用程序写出的缩放因子独立编码 ---
    scale = (e4m3_byte_to_f32(q["scales"]) * gscale).astype(np.float32)
    inv = np.where(scale > 0, np.float32(1.0) / scale, np.float32(0.0)).astype(np.float32)
    scaled = (blocks * inv[:, None]).astype(np.float32)
    mag = np.abs(scaled).astype(np.float32)
    code = fp4_encode_mag(mag)
    code = np.where(scaled < 0, code | 0x8, code).astype(np.uint8)
    code = code.reshape(-1)[:n]

    # 解包程序码流：低 nibble = 偶数下标，高 nibble = 奇数下标
    lo = (q["packed"] & 0x0F)
    hi = ((q["packed"] >> 4) & 0x0F)
    unpacked = np.empty(q["packed"].size * 2, dtype=np.uint8)
    unpacked[0::2] = lo
    unpacked[1::2] = hi
    unpacked = unpacked[:n]
    code_mismatch = int(np.count_nonzero(code != unpacked))
    return scale_mismatch, code_mismatch, nb, "E2M1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tensor")
    ap.add_argument("quant")
    ap.add_argument("--limit", type=int, default=0, help="仅检查前 N 个元素（0=全部）")
    args = ap.parse_args()

    rows, cols, x = read_tensor(args.tensor)
    q = read_quant(args.quant)
    if args.limit > 0:
        x = x[:args.limit]

    name = "MXFP8" if q["format"] == 0 else "NVFP4"
    print(f"张量: {rows} x {cols}  格式: {name}  block={q['block_size']}  "
          f"scale_mode={'tensor' if q['scale_mode'] == 0 else 'block'}  "
          f"blocks={q['num_blocks']}  global_scale={q['gscale']:.6g}")

    if q["format"] == 0:
        sm, cm, nb, elem_name = check_mxfp8(x, q)
    else:
        sm, cm, nb, elem_name = check_nvfp4(x, q)

    print(f"  [缩放因子] 独立重算 vs 程序写出: 不符 {sm} / {nb} 个 block "
          f"({100.0 * (nb - sm) / nb:.4f}% 一致)")
    print(f"  [元素编码] {elem_name} 标准舍入 vs 程序码流: 不符 {cm} / {x.size} 个元素 "
          f"({100.0 * (x.size - cm) / x.size:.6f}% 一致)")
    ok = (sm == 0 and cm == 0)
    print(f"  结论: {'完全一致 —— 编码语义符合第三方标准实现' if ok else '存在差异，需排查'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

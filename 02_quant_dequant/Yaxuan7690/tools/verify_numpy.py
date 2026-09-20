#!/usr/bin/env python3
# =============================================================================
#  verify_numpy.py -- NumPy 独立实现, 用于交叉验证 GPU 与 CPU 参考实现
#
#  独立性来源:
#    * 格式转换用「枚举全部可表示值 + 向量化二分 (np.searchsorted)」,
#      不复刻 formats.hpp 的尾数截断位运算, 也不复刻 reference.cu 的循环二分;
#    * 用 NumPy float32 算术显式还原每一步运算顺序, 便于定位差异;
#    * 直接解析低精度权重文件, 对拍 packed / scales / 反量化结果.
#
#  用法:
#     python tools/verify_numpy.py --input data/x.tensor --lowp data/x_lowp.bin
# =============================================================================
import argparse
import os
import struct
import sys

import numpy as np

# magic(8) + 8*u32 + 6*i64 + 2*f32 + u64 + 24B = 128
HEADER_FMT = "<8sIIIIIIIIqqqqqqffQ24s"
HEADER_SIZE = 128
KEYS = ["magic", "version", "format", "elem_bits", "elem_kind", "scale_kind",
        "scale_mode", "rounding", "has_global_scale", "num_rows", "num_cols",
        "block_size", "num_blocks", "packed_bytes", "scale_bytes",
        "global_scale", "global_scale_inv", "seed", "reserved"]


def read_tensor(path):
    with open(path, "rb") as f:
        raw = f.read()
    rows, cols = struct.unpack_from("<qq", raw, 0)
    dlen = raw[16]
    dtype = raw[17:17 + dlen].decode("ascii")
    off = 17 + dlen
    need = rows * cols * (4 if dtype == "fp32" else 2)
    if len(raw) < off + need:
        off = (off + 7) & ~7
    if dtype == "fp32":
        data = np.frombuffer(raw, dtype="<f4", count=rows * cols, offset=off)
    else:
        data = np.frombuffer(raw, dtype="<f2", count=rows * cols, offset=off)
    return rows, cols, np.asarray(data, dtype=np.float32).copy()


def read_lowp(path):
    with open(path, "rb") as f:
        raw = f.read()
    h = dict(zip(KEYS, struct.unpack_from(HEADER_FMT, raw, 0)))
    off = HEADER_SIZE
    packed = np.frombuffer(raw, dtype=np.uint8, count=h["packed_bytes"], offset=off).copy()
    off += h["packed_bytes"]
    scales = np.frombuffer(raw, dtype=np.uint8, count=h["scale_bytes"], offset=off).copy()
    return h, packed, scales


def build_table(ebits, mbits, bias, has_inf, has_nan):
    mag_mask = (1 << (ebits + mbits)) - 1
    max_ef = (1 << ebits) - 1
    emin = 1 - bias
    dec = np.zeros(mag_mask + 1, dtype=np.float64)
    vals, codes = [], []
    for code in range(1, mag_mask + 1):
        ef = code >> mbits
        mf = code & ((1 << mbits) - 1)
        if has_inf and ef == max_ef:
            continue
        if has_nan and not has_inf and ef == max_ef and mf == ((1 << mbits) - 1):
            continue
        if ef == 0:
            v = float(mf) * 2.0 ** (emin - mbits)
        else:
            v = (1.0 + mf * 2.0 ** (-mbits)) * 2.0 ** (ef - bias)
        dec[code] = v
        vals.append(v)
        codes.append(code)
    return dict(dec=dec, vals=np.array(vals), codes=np.array(codes, dtype=np.int64),
                mag_mask=mag_mask, total=ebits + mbits)


E2M1 = build_table(2, 1, 1, False, False)
E4M3 = build_table(4, 3, 7, False, True)
E5M2 = build_table(5, 2, 15, True, True)


def enc_to_f32(tab, code):
    code = np.asarray(code, dtype=np.int64)
    sign = (code >> tab["total"]) & 1
    v = tab["dec"][code & tab["mag_mask"]]
    return np.where(sign != 0, -v, v).astype(np.float32)


def quantize_mag(tab, ax):
    """|x| -> 幅值编码 (RNE, ties-to-even, 饱和). 0 也作为候选幅值参与搜索."""
    vals = np.concatenate([[0.0], tab["vals"]])
    codes = np.concatenate([[0], tab["codes"]])
    n = len(vals)
    out = np.full(np.asarray(ax).shape, codes[-1], dtype=np.int64)
    av = np.asarray(ax, dtype=np.float64)
    m = av < vals[-1]
    a = av[m]
    idx = np.clip(np.searchsorted(vals, a, side="left"), 1, n - 1)
    exact = vals[idx] == a
    res = np.empty(a.shape, dtype=np.int64)
    res[exact] = codes[idx[exact]]
    ne = ~exact
    lo, hi = idx[ne] - 1, idx[ne]
    twice = 2.0 * a[ne]
    s = vals[lo] + vals[hi]
    tie, less = twice == s, twice < s
    lo_even = (codes[lo] & 1) == 0
    res[ne] = codes[np.where(less | (tie & lo_even), lo, hi)]
    out[m] = res
    return out


def quantize_signed(tab, x):
    x = np.asarray(x, dtype=np.float32)
    sign = np.signbit(x).astype(np.int64) << tab["total"]
    return (sign | quantize_mag(tab, np.abs(x.astype(np.float64)))).astype(np.uint8)


def e8m0_to_f32(code):
    return np.exp2((np.asarray(code, dtype=np.int64) - 127).astype(np.float32)).astype(np.float32)


def ilog2_floor(amax):
    e = np.floor(np.log2(amax.astype(np.float64)))
    e = np.where(np.exp2(e) > amax, e - 1, e)
    e = np.where(np.exp2(e) * 2.0 <= amax, e + 1, e)
    return e.astype(np.int64)


def block_amax(x, numel, block_size, num_blocks):
    amax = np.zeros(num_blocks, dtype=np.float32)
    if num_blocks == 1:
        amax[0] = np.max(np.abs(x))
        return amax
    for b in range(num_blocks):
        seg = x[b * block_size:min((b + 1) * block_size, numel)]
        amax[b] = np.max(np.abs(seg)) if seg.size else np.float32(0)
    return amax


def reference_quant(x, cfg):
    numel = x.size
    block_size = numel if cfg["scale_mode"] == "tensor" else cfg["block_size"]
    num_blocks = (numel + block_size - 1) // block_size
    amax = block_amax(x, numel, block_size, num_blocks)

    if cfg["format"] == "mxfp8":
        tab = E5M2 if cfg.get("elem_format", "e4m3") == "e5m2" else E4M3
        emax_elem = 15 if tab is E5M2 else 8
        e = np.full(num_blocks, -127, dtype=np.int64)
        nz = amax > 0
        e[nz] = np.clip(ilog2_floor(amax[nz]) - emax_elem, -127, 127)
        scales = (e + 127).astype(np.uint8)
        inv = (np.float32(1.0) / e8m0_to_f32(scales)).astype(np.float32)
        q = (x * np.repeat(inv, block_size)[:numel]).astype(np.float32)
        return dict(scales=scales, packed=quantize_signed(tab, q), block_size=block_size,
                    num_blocks=num_blocks, global_scale=1.0, global_scale_inv=1.0)

    tamax = np.float32(np.max(np.abs(x)))
    gs = np.float32(np.float32(448.0) * np.float32(6.0) / tamax) if tamax > 0 else np.float32(1.0)
    scales = quantize_signed(E4M3, (amax * gs / np.float32(6.0)).astype(np.float32))
    bs = np.where(enc_to_f32(E4M3, scales) > 0, enc_to_f32(E4M3, scales), np.float32(1.0))
    q = ((x * gs).astype(np.float32) / np.repeat(bs, block_size)[:numel]).astype(np.float32)
    codes = quantize_signed(E2M1, q).astype(np.uint16)
    lo = codes[0::2]
    hi = codes[1::2]
    if hi.size < lo.size:
        hi = np.concatenate([hi, np.zeros(1, dtype=np.uint16)])
    packed = (lo | (hi << 4)).astype(np.uint8)
    return dict(scales=scales, packed=packed, block_size=block_size, num_blocks=num_blocks,
                global_scale=float(gs), global_scale_inv=float(np.float32(1.0) / gs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--lowp", required=True)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    rows, cols, x = read_tensor(args.input)
    h, packed, scales = read_lowp(args.lowp)
    cfg = dict(format="mxfp8" if h["format"] == 0 else "nvfp4",
               block_size=h["block_size"],
               scale_mode="tensor" if h["scale_mode"] == 0 else "block",
               elem_format="e5m2" if h["elem_kind"] == 1 else "e4m3")
    exp = reference_quant(x, cfg)
    bad_s = int(np.sum(exp["scales"] != scales))
    bad_p = int(np.sum(exp["packed"] != packed))
    print("[%s] %s/%s bs=%d blocks=%d | scales_mismatch=%d packed_mismatch=%d"
          % (os.path.basename(args.lowp).replace("_lowp.bin", ""), cfg["format"],
             cfg["scale_mode"], h["block_size"], h["num_blocks"], bad_s, bad_p))
    if bad_s and not args.quiet:
        w = np.flatnonzero(exp["scales"] != scales)[:6]
        print("   scale diffs (idx, numpy, lowp):",
              [(int(i), int(exp["scales"][i]), int(scales[i])) for i in w])
    if bad_p and not args.quiet:
        w = np.flatnonzero(exp["packed"] != packed)[:6]
        print("   packed diffs (idx, numpy, lowp):",
              [(int(i), int(exp["packed"][i]), int(packed[i])) for i in w])
    return 0 if (bad_s == 0 and bad_p == 0) else 1


if __name__ == "__main__":
    sys.exit(main())

"""Compact Triton alternatives; group scaling and RNE, finite input."""

import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib


@tr.jit
def fp8_rne(x):
    # SM89: avoid Triton's FP32->FP16->FP8 double rounding near midpoints.
    bits = tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f32 t, 0.0, $1; cvt.u32.u16 $0, t; }",
        constraints="=r,f",
        args=[x],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    return bits.to(tl.uint8).to(tl.float8e4nv, bitcast=True)


@tr.jit
def mx_compress(X, Q, S, R: tl.constexpr, C: tl.constexpr, CEIL: tl.constexpr = False):
    g = tl.program_id(0) * 8 + tl.arange(0, 8)
    c = (g % tr.cdiv(C, 32))[:, None] * 32 + tl.arange(0, 32)[None, :]
    r = (g // tr.cdiv(C, 32))[:, None]
    x = tl.load(X + r * C + c, (r < R) & (c < C), other=0).to(tl.float32)
    a = tl.max(tl.abs(x), 1)
    e = lib.ilogb(a) - 8
    if CEIL:
        e += lib.ldexp(a, -e) > 448
    e = tl.where(a == 0, -127, tl.minimum(127, tl.maximum(-127, e)))
    q = tl.minimum(448.0, tl.maximum(-448.0, lib.ldexp(x, -e[:, None])))
    tl.store(Q + r * C + c, fp8_rne(q).to(tl.uint8, bitcast=True), (r < R) & (c < C))
    tl.store(S + g, (e + 127).to(tl.uint8), g < R * tr.cdiv(C, 32))


@tr.jit
def mx_decompress(Q, S, Y, R: tl.constexpr, C: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    q = (
        tl.load(Q + i, i < R * C, other=0)
        .to(tl.float8e4nv, bitcast=True)
        .to(tl.float32)
    )
    si = (i // C) * tr.cdiv(C, 32) + (i % C) // 32
    e = tl.load(S + si, i < R * C, other=127).to(tl.int32) - 127
    tl.store(Y + i, lib.ldexp(q, e), i < R * C)


@tr.jit
def fp4_code(v):
    a = tl.abs(v)
    # Round ties to even at each E2M1 spacing, then map exact values to codes.
    z = tl.where(
        a < 2,
        lib.rint(a * 2) * 0.5,
        tl.where(a < 4, lib.rint(a), lib.rint(a * 0.5) * 2),
    )
    z = tl.minimum(z, 6)
    code = tl.where(z <= 2, z * 2, tl.where(z == 3, 5, tl.where(z == 4, 6, 7))).to(
        tl.int32
    )
    return code | ((v.to(tl.int32, bitcast=True) >> 31) & 8), tl.where(v < 0, -z, z)


@tr.jit
def nv_compress(
    X,
    Q,
    S,
    A,
    G,
    R: tl.constexpr,
    C: tl.constexpr,
    FOUR: tl.constexpr,
    BOUND: tl.constexpr,
):
    gi = tl.program_id(0) * 8 + tl.arange(0, 8)
    col = (gi % tr.cdiv(C, 16))[:, None] * 16 + tl.arange(0, 16)[None, :]
    row = (gi // tr.cdiv(C, 16))[:, None]
    valid = (row < R) & (col < C)
    x = tl.load(X + row * C + col, valid, other=0).to(tl.float32)
    a = tl.max(tl.abs(x), 1)
    total = tl.load(A)
    enc = tl.div_rn(6 * BOUND, total)
    g = tl.where(total > 0, tl.div_rn(1.0, enc), 1.0)
    sc = tl.div_rn(a, 6.0) * enc
    sc = tl.where(total > 0, sc, 0.0)
    s6 = fp8_rne(sc)
    d = s6.to(tl.float32) * g
    q6, v6 = fp4_code(tl.where(d[:, None] > 0, x * tl.div_rn(1.0, d[:, None]), 0.0))
    if FOUR:
        s4 = fp8_rne(sc * 1.5)
        d4 = s4.to(tl.float32) * g
        q4, v4 = fp4_code(
            tl.where(d4[:, None] > 0, x * tl.div_rn(1.0, d4[:, None]), 0.0)
        )
        e6 = x - tl.div_rn(v6 * s6.to(tl.float32)[:, None] * total, 6 * BOUND)
        e4 = x - tl.div_rn(v4 * s4.to(tl.float32)[:, None] * total, 6 * BOUND)
        choose = tl.sum(e4 * e4, 1) < tl.sum(e6 * e6, 1)
        q6 = tl.where(choose[:, None], q4, q6)
        s6 = tl.where(
            choose, s4.to(tl.uint8, bitcast=True), s6.to(tl.uint8, bitcast=True)
        ).to(tl.float8e4nv, bitcast=True)
    pairs = tl.reshape(q6, (8, 8, 2))
    packed = tl.sum(pairs << tl.arange(0, 2)[None, None, :] * 4, 2)
    pc = (gi % tr.cdiv(C, 16))[:, None] * 8 + tl.arange(0, 8)[None, :]
    tl.store(
        Q + (gi // tr.cdiv(C, 16))[:, None] * tr.cdiv(C, 2) + pc,
        packed.to(tl.uint8),
        (gi[:, None] < R * tr.cdiv(C, 16)) & (pc < tr.cdiv(C, 2)),
    )
    tl.store(S + gi, s6.to(tl.uint8, bitcast=True), gi < R * tr.cdiv(C, 16))
    if tl.program_id(0) == 0:
        tl.store(G, g)


@tr.jit
def nv_decompress(Q, S, G, Y, R: tl.constexpr, C: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    row = i // C
    col = i % C
    p = tl.load(Q + row * tr.cdiv(C, 2) + col // 2, i < R * C, other=0).to(tl.int32)
    q = (p >> ((col % 2) * 4)) & 15
    m = q & 7
    v = tl.where(m <= 4, m * 0.5, tl.where(m == 5, 3.0, tl.where(m == 6, 4.0, 6.0)))
    v = tl.where((q & 8) != 0, -v, v)
    s = (
        tl.load(S + row * tr.cdiv(C, 16) + col // 16, i < R * C, other=0)
        .to(tl.float8e4nv, bitcast=True)
        .to(tl.float32)
    )
    tl.store(Y + i, (v * s) * tl.load(G), i < R * C)


def quantize_into(x, qt, amax):
    if qt.scale_mode != "block" or qt.rounding != "nearest":
        raise ValueError("Triton teaching variants use block scaling and RNE")
    block = 16 if qt.format == "nvfp4" else 32
    grid = (tr.cdiv(qt.rows * tr.cdiv(qt.cols, block), 8),)
    if block == 32:
        mx_compress[grid](
            x,
            qt.data,
            qt.scales,
            qt.rows,
            qt.cols,
            qt.scale_policy == "rceil",
            num_warps=4,
            enable_fp_fusion=False,
        )
    else:
        nv_compress[grid](
            x,
            qt.data,
            qt.scales,
            amax,
            qt.global_scale,
            qt.rows,
            qt.cols,
            qt.four_over_six,
            qt.fp8_bound,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return qt


def dequantize_into(qt, y):
    grid = (tr.cdiv(qt.rows * qt.cols, 256),)
    if qt.format == "mxfp8":
        mx_decompress[grid](qt.data, qt.scales, y, qt.rows, qt.cols)
    else:
        nv_decompress[grid](
            qt.data,
            qt.scales,
            qt.global_scale,
            y,
            qt.rows,
            qt.cols,
            enable_fp_fusion=False,
        )
    return y

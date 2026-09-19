#!/usr/bin/env python3
"""Quantisation error with and without the Hadamard rotation (report 7.7.1).

The fused kernel is verified bit for bit against "rotate, then quantise"
(`--mode quant`, tests/test_fusion.cu), and the rotation itself against
fast_hadamard_transform.  What those checks do not show is *why* one rotates:
this script measures the quantisation error of the two pipelines

    plain   : x -> quant -> dequant                               -> x_hat
    rotated : x -> fp16(H x / sqrt(d)) -> quant -> dequant -> H^T -> x_hat

on three activation-like distributions, using the kernel's quantisation rules
(scale = amax / 448 for E4M3 with round-to-nearest-even and saturation,
scale = amax / 7 for symmetric INT4 with round-to-nearest-even, per row or per
block of `bs` columns).  It is a CPU simulation with numpy -- the rules match
the kernel, the bits are not claimed to.  The error is measured in the
original domain (H / sqrt(d) is orthonormal, so the inverse is exact).

Usage:
    python tools/quant_error.py [--rows 4096] [--dims 64,128,256,1024]
"""

import argparse

import numpy as np


def hadamard(d):
    h = np.ones((1, 1))
    while h.shape[0] < d:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(d)


def e4m3(v):
    """Round to the nearest E4M3 value (RNE), saturating at +-448."""
    a = np.abs(v)
    e = np.floor(np.log2(np.where(a > 0, a, 1.0)))
    e = np.maximum(e, -6.0)  # subnormals share the step of the smallest normal
    step = np.exp2(e - 3.0)  # 3 mantissa bits
    q = np.round(a / step) * step  # np.round is round-half-even
    return np.sign(v) * np.minimum(q, 448.0)


def quant_dequant(y, fmt, bs):
    rows, d = y.shape
    b = bs if bs > 0 else d
    yb = y.reshape(rows, d // b, b)
    amax = np.abs(yb).max(axis=2, keepdims=True)
    qmax = 448.0 if fmt == "fp8" else 7.0
    sc = np.where(amax > 0, amax / qmax, 1.0)
    v = yb / sc
    if fmt == "fp8":
        q = e4m3(v)
    else:
        q = np.rint(np.clip(v, -8.0, 7.0))
    return (q * sc).reshape(rows, d)


def make_input(kind, rows, d, rng):
    x = rng.standard_normal((rows, d))
    if kind == "outlier":
        # LLM-style: a few fixed channels carry ~20x the typical magnitude.
        cols = rng.choice(d, size=max(1, d // 64), replace=False)
        x[:, cols] *= 20.0
    elif kind == "student_t":
        x = rng.standard_t(3, size=(rows, d))
    return x.astype(np.float16).astype(np.float64)  # fp16 activations


def sqnr_db(x, x_hat):
    return 10.0 * np.log10((x ** 2).sum() / ((x - x_hat) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--dims", default="64,128,256,1024")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("| input | dim | format | scale | SQNR plain (dB) | SQNR rotated (dB) | gain (dB) "
          "| MSE plain / rotated |")
    print("|---|---|---|---|---|---|---|---|")
    for kind in ("gaussian", "outlier", "student_t"):
        for d in [int(s) for s in args.dims.split(",")]:
            x = make_input(kind, args.rows, d, rng)
            h = hadamard(d)
            # numpy 2.x on macOS/Accelerate raises spurious matmul FP warnings; the
            # asserts below check the results themselves instead.
            with np.errstate(all="ignore"):
                y = (x @ h).astype(np.float16).astype(np.float64)  # kernel output is fp16
            assert np.isfinite(y).all()
            for fmt in ("fp8", "int4"):
                for bs in (0, 32):
                    plain = quant_dequant(x, fmt, bs)
                    with np.errstate(all="ignore"):
                        rot = quant_dequant(y, fmt, bs) @ h.T
                    assert np.isfinite(rot).all()
                    sp, sr = sqnr_db(x, plain), sqnr_db(x, rot)
                    mse_ratio = ((x - plain) ** 2).mean() / ((x - rot) ** 2).mean()
                    print(f"| {kind} | {d} | {fmt} | {'row' if bs == 0 else f'block {bs}'} "
                          f"| {sp:.2f} | {sr:.2f} | {sr - sp:+.2f} | {mse_ratio:.2f}x |")


if __name__ == "__main__":
    main()

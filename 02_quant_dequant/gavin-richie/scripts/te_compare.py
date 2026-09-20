#!/usr/bin/env python3
"""Transformer Engine quantize/dequantize benchmark.

Reads the same .bin tensor format as the C++ pipeline
(int64 rows, int64 cols, char[4] dtype, row-major data), quantizes it
through Transformer Engine's cast kernels and reports the same metrics
as `lowp run --log` (max_abs_error / mae / mse / compression_ratio /
kernel ms / bandwidth) so the software-simulated MXFP8/NVFP4 path can be
compared against the production TE path.

Formats
  e4m3 / e5m2 : per-tensor scale (CurrentScaling-style), Float8Tensor path.
                Available on TE 1.x and TE 2.x (quantizer API on 2.x).
  mxfp8       : MXFP8BlockScaling recipe semantics - E4M3 payload with
                E8M0 (power-of-2) scale per 32-element block. Requires
                TE >= 2.0 and Blackwell hardware.
  nvfp4       : NVFP4BlockScaling recipe semantics - E2M1 payload with
                E4M3 scale per 16-element block plus an fp32 global
                scale. Requires TE >= 2.0 and Blackwell hardware.
                --nvfp4-variant vanilla disables RHT / stochastic
                rounding / 2D quantization (direct match for the
                software NVFP4 path); "full" keeps the production
                recipe defaults (random Hadamard transform on).

Bandwidth accounting mirrors src/main.cu: bytes = (fp32 input +
packed payload + scale arrays) / time for both quantize and dequantize.

Usage:
    te_compare.py TENSOR.bin [--format e4m3|e5m2|mxfp8|nvfp4]
                             [--nvfp4-variant vanilla|full]
                             [--dist LABEL] [--out LOGFILE]
                             [--warmup N] [--iters N]
"""
import argparse
import struct
import sys

import numpy as np
import torch

import transformer_engine  # noqa: F401  (must import before the C extension)
import transformer_engine_torch as tex

TE_VERSION = tuple(
    int(p) for p in transformer_engine.__version__.split(".")[:2]
)

FP8_MAX = {"e4m3": 448.0, "e5m2": 57344.0}
FP8_DTYPE = {
    "e4m3": tex.DType.kFloat8E4M3,
    "e5m2": tex.DType.kFloat8E5M2,
}


def read_tensor(path):
    with open(path, "rb") as f:
        rows, cols = struct.unpack("<qq", f.read(16))
        dtype = f.read(4).decode()
        if dtype != "fp32":
            raise ValueError(f"te_compare expects fp32 input, got {dtype}")
        data = np.fromfile(f, dtype="<f4", count=rows * cols)
    if data.size != rows * cols:
        raise ValueError(f"truncated tensor file: {path}")
    return data.reshape(rows, cols), dtype


def time_kernel(fn, warmup, iters):
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def quantize_bytes(n, fmt):
    """Packed payload + scale bytes per element, same as main.cu."""
    if fmt in ("e4m3", "e5m2"):
        return n * 1.0 + 8.0        # fp8 payload + fp32 scale/amax
    if fmt == "mxfp8":
        return n * 1.0 + n / 32.0   # fp8 payload + E8M0 block scales
    if fmt == "nvfp4":
        return n * 0.5 + n / 16.0 + 4.0  # packed nibbles + E4M3 scales + fp32 GS
    raise ValueError(fmt)


def run_per_tensor_legacy(x, rows, cols, fp8_dtype):
    """TE 1.x path: Float8Tensor.quantize_ with an explicit amax scale."""
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor

    vmax = FP8_MAX[ARGS.format]
    amax = x.abs().amax().reshape(1)
    scale = (vmax / amax).reshape(1)
    scale = torch.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)

    data = torch.empty(rows, cols, dtype=torch.uint8, device="cuda")
    scale_inv = torch.empty(1, device="cuda")
    q = Float8Tensor(data=data, fp8_dtype=fp8_dtype,
                     fp8_scale_inv=scale_inv, dtype=torch.float32)
    q.quantize_(x, scale=scale, amax=amax)
    dq = q.dequantize(dtype=torch.float16)

    def quant_full():
        a, s = quantize_amax_scale(x)
        q.quantize_(x, scale=s, amax=a)

    quant_full_ms = time_kernel(quant_full, ARGS.warmup, ARGS.iters)
    quant_cast_only_ms = time_kernel(
        lambda: q.quantize_(x, scale=scale, amax=amax),
        ARGS.warmup, ARGS.iters)
    dequant = time_kernel(
        lambda: q.dequantize(dtype=torch.float16), ARGS.warmup, ARGS.iters)
    return dq, quant_full_ms, quant_cast_only_ms, dequant


def quantize_amax_scale(x):
    """Per-tensor amax + scale, CurrentScaling semantics (guard amax=0)."""
    amax = x.abs().amax().reshape(1)
    scale = FP8_MAX[ARGS.format] / amax
    scale = torch.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)
    return amax, scale


def run_te2(x, rows, cols):
    """TE >= 2.0 path: quantizer API for per-tensor, MXFP8 and NVFP4."""
    from transformer_engine.pytorch.tensor.float8_tensor import (
        Float8CurrentScalingQuantizer,
    )
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    dev = x.device
    if ARGS.format in ("e4m3", "e5m2"):
        quantizer = Float8CurrentScalingQuantizer(
            FP8_DTYPE[ARGS.format], device=dev,
            rowwise=True, columnwise=False)
    elif ARGS.format == "mxfp8":
        quantizer = MXFP8Quantizer(
            tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
    elif ARGS.format == "nvfp4":
        vanilla = ARGS.nvfp4_variant == "vanilla"
        quantizer = NVFP4Quantizer(
            tex.DType.kFloat4E2M1, rowwise=True, columnwise=False,
            with_rht=not vanilla,
            with_post_rht_amax=not vanilla,
            with_2d_quantization=False,
            stochastic_rounding=False)
    else:
        raise ValueError(ARGS.format)

    # The production RHT path quantizes from bf16 activations (fp32 input
    # is rejected by the RHT cast kernel); error metrics below are still
    # computed against the original fp32 tensor.
    xq = x.to(torch.bfloat16) if (
        ARGS.format == "nvfp4" and ARGS.nvfp4_variant == "full") else x

    q = quantizer(xq)
    dq = q.dequantize(dtype=torch.float16)

    quant_update = time_kernel(
        lambda: quantizer.update_quantized(xq, q), ARGS.warmup, ARGS.iters)
    quant_alloc = time_kernel(
        lambda: quantizer(xq), ARGS.warmup, ARGS.iters)
    dequant = time_kernel(
        lambda: q.dequantize(dtype=torch.float16), ARGS.warmup, ARGS.iters)
    return dq, quant_update, quant_alloc, dequant


def cuda_preflight():
    """Fail fast with a root-cause hint when the container GPU access is in
    one of the known broken states (device nodes released, driver libs
    mounted as stubs, compat/kernel version mix). See
    scripts/gpu_env_doctor.sh for the full diagnosis."""
    import glob
    import os
    try:
        torch.cuda.init()
        torch.cuda.get_device_name(0)
        return
    except RuntimeError as e:
        hints = []
        if not glob.glob("/dev/nvidia[0-9]*"):
            hints.append("/dev/nvidia[0-9]* 设备节点缺失：宿主机未注入或已回收 GPU")
        stubs = [p for p in glob.glob("/usr/lib/x86_64-linux-gnu/libcuda.so.*")
                 if os.path.isfile(p) and os.path.getsize(p) <= 4096]
        if stubs:
            hints.append(f"驱动库为空 stub（如 {stubs[0]}），真实 libcuda 未挂载")
        if "803" in str(e):
            hints.append("error 803：用户态 libcuda 与内核驱动版本组合不受支持"
                         "（常见于 LD_LIBRARY_PATH 混入旧 compat 库）")
        hints.append("完整诊断：bash scripts/gpu_env_doctor.sh")
        raise SystemExit(
            "CUDA 不可用: " + str(e) +
            "\n可能的根因:\n" + "\n".join("  - " + h for h in hints))


def main():
    if ARGS.format in ("mxfp8", "nvfp4") and TE_VERSION < (2, 0):
        sys.exit(
            f"format {ARGS.format} needs TransformerEngine >= 2.0 "
            f"(found {transformer_engine.__version__}); "
            "per-tensor e4m3/e5m2 still work with this install")

    tensor, dtype = read_tensor(ARGS.tensor)
    rows, cols = tensor.shape
    n = rows * cols

    cuda_preflight()
    x = torch.from_numpy(tensor).to("cuda", dtype=torch.float32)

    if TE_VERSION >= (2, 0):
        dq, quant_full_ms, quant_extra_ms, dequant = run_te2(x, rows, cols)
        extra_key = "quantize_alloc_ms"
    else:
        dq, quant_full_ms, quant_extra_ms, dequant = run_per_tensor_legacy(
            x, rows, cols, FP8_DTYPE[ARGS.format])
        extra_key = "quantize_cast_only_ms"

    # --- correctness (fp64 metrics on GPU) ---
    diff = dq.double() - x.double()
    max_abs_error = diff.abs().max().item()
    mae = diff.abs().mean().item()
    mse = (diff * diff).mean().item()

    # --- throughput (same accounting as src/main.cu) ---
    in_bytes = n * 4.0                       # fp32 input
    q_bytes = quantize_bytes(n, ARGS.format)
    compression = in_bytes / q_bytes

    if TE_VERSION >= (2, 0):
        recipe = {"e4m3": "Float8CurrentScaling", "e5m2": "Float8CurrentScaling",
                  "mxfp8": "MXFP8BlockScaling",
                  "nvfp4": f"NVFP4BlockScaling({ARGS.nvfp4_variant})"}[ARGS.format]
        scale_mode = {"e4m3": "tensor", "e5m2": "tensor",
                      "mxfp8": "block", "nvfp4": "block"}[ARGS.format]
    else:
        recipe = "per-tensor (TE 1.x Float8Tensor)"
        scale_mode = "tensor"
    block_size = {"e4m3": 0, "e5m2": 0, "mxfp8": 32, "nvfp4": 16}[ARGS.format]

    lines = [
        "# Transformer Engine quantize/dequantize benchmark (cast kernels,",
        f"# same error/bandwidth accounting as lowp run --log; TE"
        f" {transformer_engine.__version__})",
        f'te_version = "{transformer_engine.__version__}"',
        f'gpu = "{torch.cuda.get_device_name(0)}"',
        f'tensor = "{ARGS.tensor}"',
        f'distribution = "{ARGS.dist}"',
        f'format = "te-{ARGS.format}"',
        f'recipe = "{recipe}"',
        f'scale_mode = "{scale_mode}"',
        f'block_size = {block_size}',
        "input_dtype = \"bf16\"" if (ARGS.format == "nvfp4" and
                                     ARGS.nvfp4_variant == "full")
        else "input_dtype = \"fp32\"",
        "output_type = \"fp16\"",
        f"rows = {rows}",
        f"cols = {cols}",
        f"num_elements = {n}",
        f"max_abs_error = {max_abs_error:.6f}",
        f"mae = {mae:.6f}",
        f"mse = {mse:.6f}",
        f"compression_ratio = {compression:.4f}",
        f"quantize_kernel_ms = {quant_full_ms:.4f}",
        f"{extra_key} = {quant_extra_ms:.4f}",
        f"dequantize_kernel_ms = {dequant:.4f}",
        f"quantize_bandwidth_gbps = {(in_bytes + q_bytes) / (quant_full_ms * 1e-3) / 1e9:.4f}",
        f"dequantize_bandwidth_gbps = {(in_bytes + q_bytes) / (dequant * 1e-3) / 1e9:.4f}",
    ]
    report = "\n".join(lines) + "\n"
    if ARGS.out:
        with open(ARGS.out, "w") as f:
            f.write(report)
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tensor", help="input .bin tensor (fp32)")
    parser.add_argument("--format", choices=sorted(FP8_MAX) + ["mxfp8", "nvfp4"],
                        default="e4m3")
    parser.add_argument("--nvfp4-variant", choices=["vanilla", "full"],
                        default="vanilla",
                        help="vanilla: no RHT/stochastic-rounding/2D "
                             "(matches the software path); full: recipe "
                             "defaults with random Hadamard transform")
    parser.add_argument("--dist", default="", help="distribution label for the log")
    parser.add_argument("--out", default="", help="write report to this file")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    ARGS = parser.parse_args()
    sys.exit(main())

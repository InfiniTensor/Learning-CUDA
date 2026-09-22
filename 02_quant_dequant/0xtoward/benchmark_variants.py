"""Matched, preallocated CUDA-event benchmark using the kernel-benchmark skill timer."""

import ctypes
import json
from pathlib import Path
import time

import torch
import codec
import triton_codec
from reference import numpy_reference
from tensor_io import save_tensor, load_tensor, save_quantized, load_quantized

import benchmark_timer as skill
from types import SimpleNamespace

cfg = SimpleNamespace(num_warmup=5, num_trials=31, discard_first=1, device=0)


def timing(fn):
    fn()
    torch.cuda.synchronize()
    return skill._summarize_times(skill._bench_times_cuda_event(fn, cfg))


def main():
    torch.manual_seed(2026)
    root = Path(__file__).parent
    out = root / "results"
    out.mkdir(exist_ok=True)
    native = ctypes.CDLL(str(root / "mxfp8_native_minimal.so"))
    native.solve.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2
    native.solve_decompress.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2
    rows = []
    for shape in [(1024, 3584), (4096, 4096)]:
        x = torch.randn(shape, device="cuda")
        y = torch.empty_like(x)
        for fmt, four in [("mxfp8", False), ("nvfp4", False), ("nvfp4", True)]:
            qt, amax = codec.prepare(x, format=fmt, four_over_six=four)
            codec.quantize_into(x, qt, amax)
            qref = qt.data.clone()
            sref = qt.scales.clone()
            yref = codec.dequantize(qt)
            variants = {
                "manual_cuda": (
                    lambda: codec.quantize_into(x, qt, amax),
                    lambda: codec.dequantize_into(qt, y),
                ),
                "triton": (
                    lambda: triton_codec.quantize_into(x, qt, amax),
                    lambda: triton_codec.dequantize_into(qt, y),
                ),
            }
            if fmt == "mxfp8":
                variants["cuda_math_api"] = (
                    lambda: native.solve(
                        x.data_ptr(), qt.data.data_ptr(), qt.scales.data_ptr(), *shape
                    ),
                    lambda: native.solve_decompress(
                        qt.data.data_ptr(), qt.scales.data_ptr(), y.data_ptr(), *shape
                    ),
                )
            for impl, (qfn, dfn) in variants.items():
                qfn()
                dfn()
                torch.cuda.synchronize()
                assert torch.equal(qref, qt.data), (
                    shape,
                    fmt,
                    impl,
                    "codes",
                    (qref != qt.data).sum().item(),
                )
                assert torch.equal(sref, qt.scales), (shape, fmt, impl, "scales")
                assert torch.equal(yref, y), (shape, fmt, impl, "dequant")
                tq, td = timing(qfn), timing(dfn)
                record = dict(
                    shape=shape,
                    format=fmt,
                    four_over_six=four,
                    implementation=impl,
                    quant_ms=tq,
                    dequant_ms=td,
                    quant_GBs=(x.numel() * 4 + qt.nbytes) / (tq["median"] * 1e6),
                    dequant_GBs=(y.numel() * 4 + qt.nbytes) / (td["median"] * 1e6),
                    payload_bytes=qt.nbytes,
                    compression_vs_fp32=x.numel() * 4 / qt.nbytes,
                    exact_match=True,
                )
                rows.append(record)
                print(json.dumps(record), flush=True)
    # Small independent CPU baseline, including allocation, explicitly not kernel-only.
    x = torch.randn((512, 1024), device="cuda")
    cpu = x.cpu().numpy()
    cpu_ms = {}
    for fmt in ["mxfp8", "nvfp4"]:
        t = time.perf_counter()
        numpy_reference(cpu, format=fmt)
        cpu_ms[fmt] = (time.perf_counter() - t) * 1000
        qt, amax = codec.prepare(x, format=fmt)
        cpu_ms[fmt + "_cuda_quant_ms"] = timing(
            lambda: codec.quantize_into(x, qt, amax)
        )["median"]
    result = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        timing="skill CUDA events; prewarm1, warmup5, 31 samples, discard1, 256MiB cache thrash before each sample",
        quant_scope="amax precomputed; q/scales/global write included",
        rows=rows,
        cpu_comparison=cpu_ms,
    )
    (out / "benchmark.json").write_text(json.dumps(result, indent=2))
    # Odd rows/nibbles, output dtype, and binary roundtrip.
    for fmt in ["mxfp8", "nvfp4"]:
        t = torch.randn((7, 69), device="cuda", dtype=torch.float16)
        qt = codec.quantize(t, format=fmt)
        p = out / (fmt + "_roundtrip.lpq")
        save_quantized(p, qt)
        qr = load_quantized(p)
        assert torch.equal(codec.dequantize(qt), codec.dequantize(qr))
        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            y = codec.dequantize(qr, dtype)
            p = out / "roundtrip.lpt"
            save_tensor(p, y)
            assert torch.equal(y, load_tensor(p, "cuda"))
    save_tensor(out / "input_fp16.lpt", torch.randn((512, 1024), dtype=torch.float16))
    print("PASS benchmark exact matches and binary roundtrips", flush=True)


if __name__ == "__main__":
    main()

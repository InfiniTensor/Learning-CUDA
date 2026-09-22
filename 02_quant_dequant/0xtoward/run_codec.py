"""File-oriented quantize/dequantize CLI matching the project's data contract."""

import argparse
import json
from pathlib import Path
import statistics
import tomllib

import torch
from codec import prepare, quantize_into, dequantize_into
from reference import metrics
from tensor_io import DTYPES, load_tensor, save_tensor, save_quantized, load_quantized


def elapsed(fn, repeats=31):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return statistics.median(values)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()
    cfg = tomllib.loads(Path(args.config).read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    x = load_tensor(args.input, "cuda")
    if x.dtype not in (torch.float16, torch.float32):
        raise ValueError("input file dtype must be fp16 or fp32")
    if not torch.isfinite(x).all():
        raise ValueError("file CLI accepts finite inputs")
    fmt = cfg.get("format", "mxfp8")
    block = 16 if fmt == "nvfp4" else 32
    if cfg.get("block_size", block) != block:
        raise ValueError("MXFP8 block=32, NVFP4 block=16")
    options = {
        k: cfg[k]
        for k in (
            "format",
            "scale_mode",
            "rounding",
            "scale_policy",
            "four_over_six",
            "fp8_bound",
            "seed",
        )
        if k in cfg
    }
    qt, amax = prepare(x, **options)
    qt = quantize_into(x, qt, amax)
    y = torch.empty_like(x, dtype=DTYPES[cfg.get("output_type", "fp32")])
    quant_ms = elapsed(lambda: quantize_into(x, qt, amax))
    dq_ms = elapsed(lambda: dequantize_into(qt, y))
    amax_ms = (
        elapsed(lambda: torch.amax(x.float().abs()))
        if fmt == "nvfp4" or qt.scale_mode == "tensor"
        else 0.0
    )
    weights = args.output_dir / "weights.lpq"
    save_quantized(weights, qt)
    restored = load_quantized(weights)
    y2 = dequantize_into(restored, torch.empty_like(y))
    assert torch.equal(y, y2)
    save_tensor(args.output_dir / "dequantized.lpt", y)
    payload = metrics(x, y)
    payload.update(
        format=fmt,
        shape=list(x.shape),
        output_type=cfg.get("output_type", "fp32"),
        input_bytes=x.numel() * x.element_size(),
        packed_payload_bytes=qt.nbytes,
        file_bytes=weights.stat().st_size,
        compression_ratio=x.numel() * x.element_size() / qt.nbytes,
        quant_kernel_ms=quant_ms,
        dequant_kernel_ms=dq_ms,
        global_amax_ms=amax_ms,
        quant_effective_GBs=(x.numel() * x.element_size() + qt.nbytes)
        / (quant_ms * 1e6),
        dequant_effective_GBs=(y.numel() * y.element_size() + qt.nbytes)
        / (dq_ms * 1e6),
        actual_gpu=torch.cuda.get_device_name(),
        target_gpu=cfg.get("target_gpu", "unspecified"),
        timing="CUDA events, preallocated outputs, warm/no explicit L2 eviction, median31",
        quant_timing_scope="quant kernel; tensor amax reduction reported separately",
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

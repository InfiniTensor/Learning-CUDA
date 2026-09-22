"""Packing, tails, dtypes, scale modes, streams, stochastic and official oracle."""

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from codec import quantize, dequantize
from reference import numpy_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="also compare with the optional fouroversix reference",
    )
    args = parser.parse_args()
    torch.manual_seed(23)
    count = 0
    for shape in [(1, 1), (3, 17), (7, 70), (16, 64), (32, 256)]:
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            x = torch.randn(shape, device="cuda", dtype=dtype)
            for fmt in ("mxfp8", "nvfp4"):
                for mode in ("block", "tensor"):
                    q = quantize(x, format=fmt, scale_mode=mode)
                    ref = numpy_reference(
                        x.float().cpu().numpy(), format=fmt, scale_mode=mode
                    )
                    assert np.array_equal(q.data.cpu().numpy(), ref[0]), (
                        shape,
                        dtype,
                        fmt,
                        mode,
                        "codes",
                    )
                    assert np.array_equal(q.scales.cpu().numpy(), ref[1]), (
                        shape,
                        dtype,
                        fmt,
                        mode,
                        "scales",
                    )
                    for out_dtype in (torch.float32, torch.float16, torch.bfloat16):
                        y = dequantize(q, out_dtype)
                        target = torch.from_numpy(ref[3]).to("cuda", out_dtype)
                        assert torch.equal(y, target), (
                            shape,
                            dtype,
                            fmt,
                            mode,
                            out_dtype,
                            (y.float() - target.float()).abs().max(),
                        )
                        count += 1
    x = torch.randn((32, 64), device="cuda")
    for fmt in ("mxfp8", "nvfp4"):
        q1 = quantize(x, format=fmt, rounding="stochastic", seed=31)
        q2 = quantize(x, format=fmt, rounding="stochastic", seed=31)
        q3 = quantize(x, format=fmt, rounding="stochastic", seed=32)
        assert torch.equal(q1.data, q2.data) and not torch.equal(q1.data, q3.data)
        q0 = quantize(torch.zeros_like(x), format=fmt)
        assert torch.equal(dequantize(q0), torch.zeros_like(x))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        xs = torch.randn((31, 97), device="cuda")
        ys = dequantize(quantize(xs, format="nvfp4", four_over_six=True))
    stream.synchronize()
    assert torch.isfinite(ys).all()

    comparisons = []
    if args.oracle:
        from fouroversix.quantize.pytorch.reference import quantize as official
        from fouroversix.utils import DataType, RoundStyle, ScaleRule

        for dtype in (torch.float32, torch.bfloat16):
            for four6 in (False, True):
                x = torch.randn((128, 256), device="cuda", dtype=dtype)
                rule = ScaleRule.mse if four6 else ScaleRule.static_6
                qr, sr, ar = official(
                    x,
                    fp4_format=DataType.nvfp4,
                    scale_rule=rule,
                    round_style=RoundStyle.nearest,
                    use_blackwell_scale_layout=False,
                )
                ours = quantize(x, format="nvfp4", four_over_six=four6)
                mismatches = (ours.data != qr).sum().item()
                sm = (ours.scales != sr.view(torch.uint8).reshape(-1)).sum().item()
                comparisons.append(
                    dict(
                        dtype=str(dtype),
                        four_over_six=four6,
                        code_mismatches=mismatches,
                        scale_mismatches=sm,
                    )
                )
                print("Official oracle", comparisons[-1], flush=True)
                assert mismatches == 0 and sm == 0, comparisons[-1]
    result = dict(
        dtype_tail_output_cases=count,
        official_oracle=comparisons,
        oracle_commit=(
            subprocess.check_output(
                ["git", "-C", os.environ["FOUROVERSIX_REPO"], "rev-parse", "HEAD"],
                text=True,
            ).strip()
            if args.oracle and os.environ.get("FOUROVERSIX_REPO")
            else None
        ),
        stochastic_same_seed=True,
        nondefault_stream=True,
    )
    Path("results").mkdir(exist_ok=True)
    Path("results/correctness.json").write_text(json.dumps(result, indent=2))
    print("PASS", result)


if __name__ == "__main__":
    main()

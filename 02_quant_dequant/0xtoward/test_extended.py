"""Additional scale policy, bit patterns, official real weights and file tests."""

import json
import os
from pathlib import Path
import tempfile
import numpy as np
import torch
from safetensors import safe_open
from codec import quantize, dequantize, prepare
from reference import numpy_reference
from tensor_io import save_quantized, load_quantized
import triton_codec


def main():
    torch.manual_seed(77)
    comparisons = 0
    # Zeros, signed values, FP8/FP4 rounding ties, ragged rows and both scale policies.
    for shape in [(1, 1), (7, 69), (13, 128)]:
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x = torch.randn(shape, device="cuda", dtype=dtype)
            for fmt, four, policy in [
                ("mxfp8", False, "floor"),
                ("mxfp8", False, "rceil"),
                ("nvfp4", False, "floor"),
                ("nvfp4", True, "floor"),
            ]:
                q = quantize(x, format=fmt, four_over_six=four, scale_policy=policy)
                t, a = prepare(x, format=fmt, four_over_six=four, scale_policy=policy)
                triton_codec.quantize_into(x, t, a)
                assert torch.equal(q.data, t.data) and torch.equal(
                    q.scales, t.scales
                ), (shape, dtype, fmt, four, policy)
                y = dequantize(q, dtype)
                yt = torch.empty_like(y)
                triton_codec.dequantize_into(t, yt)
                assert torch.equal(y, yt)
                if fmt == "mxfp8":
                    ref = numpy_reference(x.float().cpu().numpy(), scale_policy=policy)
                    assert np.array_equal(q.data.cpu().numpy(), ref[0])
                comparisons += 1
    for fmt in ["mxfp8", "nvfp4"]:
        x = torch.zeros((7, 69), device="cuda")
        x[:, 1] = -0.0
        q = quantize(x, format=fmt)
        t, a = prepare(x, format=fmt)
        triton_codec.quantize_into(x, t, a)
        assert torch.equal(q.data, t.data) and torch.equal(q.scales, t.scales)
    # All finite positive E4M3/E2M1 values and their rounding midpoints.
    from reference import E4, E2

    for lut, fmt in [(E4, "mxfp8"), (E2, "nvfp4")]:
        x = torch.tensor(
            np.concatenate([lut, (lut[:-1] + lut[1:]) / 2]), device="cuda"
        ).repeat(4, 1)
        q = quantize(x, format=fmt)
        ref = numpy_reference(x.cpu().numpy(), format=fmt)
        assert np.array_equal(q.data.cpu().numpy(), ref[0])
    from fouroversix.quantize.pytorch.reference import quantize as official
    from fouroversix.utils import DataType, RoundStyle, ScaleRule

    model = os.environ["QWEN_SMALL_CHECKPOINT"]
    with safe_open(model, framework="pt", device="cpu") as f:
        x = (
            f.get_tensor("model.language_model.layers.0.mlp.down_proj.weight")[
                :128, :256
            ]
            .contiguous()
            .cuda()
        )
    for four in [False, True]:
        qo, so, ao = official(
            x,
            fp4_format=DataType.nvfp4,
            scale_rule=ScaleRule.mse if four else ScaleRule.static_6,
            round_style=RoundStyle.nearest,
            use_blackwell_scale_layout=False,
        )
        q = quantize(x, format="nvfp4", four_over_six=four)
        assert torch.equal(q.data, qo) and torch.equal(
            q.scales, so.view(torch.uint8).flatten()
        )
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "q.bin"
        for fmt in ["mxfp8", "nvfp4"]:
            q = quantize(torch.randn((3, 17), device="cuda"), format=fmt)
            save_quantized(path, q)
            qr = load_quantized(path)
            assert torch.equal(q.data, qr.data) and torch.equal(
                dequantize(q), dequantize(qr)
            )
            raw = path.read_bytes()
            path.write_bytes(raw[:-1])
            try:
                load_quantized(path)
            except ValueError:
                pass
            else:
                raise AssertionError("truncated payload accepted")
    data = dict(
        cuda_triton_shape_dtype_policy_comparisons=comparisons,
        real_weight_official_oracle_cases=2,
        zero_signed_zero=True,
        rounding_midpoints=True,
        truncated_file_rejected=True,
        odd_columns_roundtrip=True,
    )
    Path("results/extended_tests.json").write_text(json.dumps(data, indent=2))
    print("PASS", data)


if __name__ == "__main__":
    main()

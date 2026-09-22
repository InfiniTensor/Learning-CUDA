"""Small matrix launch set for compute-sanitizer, including odd-width packing."""

import torch
from codec import quantize, dequantize

for shape in [(1, 1), (3, 17), (7, 69)]:
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        for fmt in ["mxfp8", "nvfp4"]:
            for mode in ["block", "tensor"]:
                x = torch.randn(shape, device="cuda", dtype=dtype)
                q = quantize(x, format=fmt, scale_mode=mode)
                for out in [torch.float32, torch.float16, torch.bfloat16]:
                    assert torch.isfinite(dequantize(q, out)).all()
torch.cuda.synchronize()
print("PASS sanitizer smoke")

"""Distribution and real-weight reconstruction errors, including controlled 4/6 ablation."""

import json
import os
from pathlib import Path
import torch
from safetensors import safe_open
from codec import quantize, dequantize
from reference import metrics

ARMS = {
    "mxfp8_floor": dict(format="mxfp8", scale_policy="floor"),
    "mxfp8_rceil": dict(format="mxfp8", scale_policy="rceil"),
    "nvfp4": dict(format="nvfp4"),
    "nvfp4_static6_bound256": dict(format="nvfp4", fp8_bound=256),
    "nvfp4_4over6": dict(format="nvfp4", four_over_six=True),
}


def compare(name, x):
    result = []
    for arm, options in ARMS.items():
        q = quantize(x, **options)
        y = dequantize(q)
        row = dict(
            tensor=name,
            shape=list(x.shape),
            arm=arm,
            **metrics(x, y),
            packed_bytes=q.nbytes,
            compression_vs_bf16=x.numel() * 2 / q.nbytes,
        )
        result.append(row)
    print(name, {r["arm"]: round(r["rel_l2"], 6) for r in result}, flush=True)
    return result


def main():
    torch.manual_seed(55)
    results = []
    shape = (1024, 1024)
    results += compare("uniform[-1,1]", torch.rand(shape, device="cuda") * 2 - 1)
    results += compare("normal(0,1)", torch.randn(shape, device="cuda"))
    x = torch.randn(shape, device="cuda")
    x.view(-1)[::97] *= 30
    results += compare("normal_with_1pct_30x_outliers", x)
    path = os.environ["QWEN_SMALL_CHECKPOINT"]
    names = []
    with safe_open(path, framework="pt", device="cpu") as f:
        names = [
            n
            for n in f.keys()
            if "language_model.layers." in n
            and n.endswith(".weight")
            and len(f.get_slice(n).get_shape()) == 2
            and min(f.get_slice(n).get_shape()) >= 128
        ]
        names = [
            names[i]
            for i in [
                0,
                1,
                2,
                3,
                len(names) // 2,
                len(names) // 2 + 1,
                len(names) - 2,
                len(names) - 1,
            ]
        ]
        for name in names:
            results += compare(name, f.get_tensor(name).cuda())
    aggregate = []
    for arm in ARMS:
        rs = [r for r in results if r["tensor"] in names and r["arm"] == arm]
        sse = sum(r["squared_error_sum"] for r in rs)
        norm = sum(r["input_squared_sum"] for r in rs)
        n = sum(r["shape"][0] * r["shape"][1] for r in rs)
        aggregate.append(
            dict(
                arm=arm,
                elements=n,
                mse=sse / n,
                rel_l2=(sse / norm) ** 0.5,
                mae=sum(r["mae"] * r["shape"][0] * r["shape"][1] for r in rs) / n,
                max_abs=max(r["max_abs"] for r in rs),
                packed_bytes=sum(r["packed_bytes"] for r in rs),
            )
        )
    Path("results/accuracy.json").write_text(
        json.dumps(dict(rows=results, real_weight_aggregate=aggregate), indent=2)
    )
    print("REAL WEIGHTS", json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()

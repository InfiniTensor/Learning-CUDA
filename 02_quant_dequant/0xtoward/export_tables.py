"""Render measured JSON artifacts into human-readable Markdown tables."""

import json
from pathlib import Path

root = Path(__file__).parent
results = root / "results"
b = json.loads((results / "benchmark.json").read_text())
lines = [
    "# Kernel benchmark",
    "",
    b["gpu"],
    "",
    b["timing"],
    "",
    "FP32 input/output; row-wise groups; MX uses FLOOR. NV tensor amax is precomputed.",
    "All compared packed bytes, scales and FP32 reconstructions match exactly.",
    "",
    "| Shape | Format | Implementation | Quant median (ms) | Dequant median (ms) | Quant GB/s | Dequant GB/s |",
    "|---|---|---|---:|---:|---:|---:|",
]
for r in b["rows"]:
    f = r["format"] + ("+4over6" if r["four_over_six"] else "")
    lines.append(
        f"| {'x'.join(map(str, r['shape']))} | {f} | {r['implementation']} | {r['quant_ms']['median']:.4f} | {r['dequant_ms']['median']:.4f} | {r['quant_GBs']:.1f} | {r['dequant_GBs']:.1f} |"
    )
lines += [
    "",
    "Full min/max/std/p20/p80: [benchmark.json](results/benchmark.json).",
    "Display activity causes occasional outliers; median and p20/p80 are retained rather than reporting only the best sample.",
    "",
    "## CPU reference (512 x 1024)",
    "",
    "NumPy LUT/searchsorted includes allocation. CUDA is the preallocated quantization kernel, not Python wall time.",
    "",
    "| Format | CPU reference (ms) | CUDA kernel (ms) |",
    "|---|---:|---:|",
]
for f in ["mxfp8", "nvfp4"]:
    lines.append(
        f"| {f} | {b['cpu_comparison'][f]:.3f} | {b['cpu_comparison'][f + '_cuda_quant_ms']:.4f} |"
    )
(root / "benchmark.md").write_text("\n".join(lines) + "\n")
a = json.loads((results / "accuracy.json").read_text())
lines = [
    "# Reconstruction errors",
    "",
    "Metrics use FP32 reconstructed values and FP64 accumulation. Relative L2 = sqrt(sum((y-x)^2)/sum(x^2)).",
    "",
    "| Tensor/distribution | Mode | Max abs | MAE | MSE | Relative L2 |",
    "|---|---|---:|---:|---:|---:|",
]
for r in a["rows"]:
    lines.append(
        f"| {r['tensor']} | {r['arm']} | {r['max_abs']:.7g} | {r['mae']:.7g} | {r['mse']:.7g} | {r['rel_l2']:.7g} |"
    )
(root / "accuracy.md").write_text("\n".join(lines) + "\n")

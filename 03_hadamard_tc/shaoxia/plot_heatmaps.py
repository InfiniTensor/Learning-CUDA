#!/usr/bin/env python3
import re
import subprocess

import matplotlib.pyplot as plt
import numpy as np


pattern = r"rows=\s*(\d+), cols=\s*(\d+).*speedup\s+([0-9.]+)x"
cases = {
    "small": ("4", range(1, 5)),    # cols 2..16
    "scalar": ("1", range(5, 14)),  # cols 32..8192
    "vec": ("2", range(8, 14)),     # cols 256..8192
}

for name, (impl, col_powers) in cases.items():
    col_powers = list(col_powers)
    powers = list(range(1, 16))
    rows = ",".join(str(2**p) for p in powers)
    cols = ",".join(str(2**p) for p in col_powers)
    text = subprocess.check_output(
        ["./bench", "--impl", impl, "--rows", rows, "--dims", cols,
         "--dtype", "fp16", "--no-check", "--iters", "20", "--warmup-ms", "1"],
        text=True,
    )
    speedup = {(int(r), int(c)): float(v) for r, c, v in re.findall(pattern, text)}
    data = np.array([[speedup[2**r, 2**c] for c in col_powers] for r in powers])
    print(f"\n{name} speedup vs FHT")
    print("rows\\cols " + " ".join(f"{2**c:>8}" for c in col_powers))
    for r, values in zip(powers, data):
        print(f"{2**r:>9} " + " ".join(f"{value:>8.2f}" for value in values))

    plt.figure(figsize=(6, 8))
    image = plt.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=2.0)
    plt.colorbar(image, label="speedup vs FHT")
    plt.xticks(range(len(col_powers)), [f"2^{p}" for p in col_powers])
    plt.yticks(range(len(powers)), [f"2^{p}" for p in powers])
    plt.xlabel("cols")
    plt.ylabel("rows")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(f"heatmap_{name}.png", dpi=150)
    plt.close()

print("wrote heatmap_small.png, heatmap_scalar.png, heatmap_vec.png")

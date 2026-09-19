#!/usr/bin/env python3
import re
import subprocess

import matplotlib.pyplot as plt
import numpy as np


sizes = [2**p for p in range(1, 15)]
csv = ",".join(map(str, sizes))

text = subprocess.check_output(
    ["./bench", "--impl", "4", "--rows", csv, "--dims", csv,
     "--dtype", "fp16", "--no-check", "--iters", "20", "--warmup-ms", "1"],
    text=True,
)

pattern = r"rows=\s*(\d+), cols=\s*(\d+).*speedup\s+([0-9.]+)x"
speedup = {(int(r), int(c)): float(v) for r, c, v in re.findall(pattern, text)}
data = np.array([[speedup[r, c] for c in sizes] for r in sizes])

print("\nv4 speedup vs FHT")
print("rows\\cols " + " ".join(f"{c:>8}" for c in sizes))
for row, values in zip(sizes, data):
    print(f"{row:>9} " + " ".join(f"{value:>8.2f}" for value in values))

plt.figure(figsize=(12, 10))
image = plt.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=2.0)
plt.colorbar(image, label="speedup vs FHT")
plt.xticks(range(len(sizes)), sizes, rotation=45)
plt.yticks(range(len(sizes)), sizes)
plt.xlabel("cols")
plt.ylabel("rows")
plt.title("v4 speedup vs FHT")
plt.tight_layout()
plt.savefig("heatmap_v4.png", dpi=150)
print("wrote heatmap_v4.png")

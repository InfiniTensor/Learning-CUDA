#!/usr/bin/env bash
set -euo pipefail

bench=$1
source_dir=$2
test_dir=${3:-"$PWD/low_precision_hadamard_test"}

rm -rf "$test_dir"
mkdir -p "$test_dir"

# 生成 64x256 的 FP32 矩阵：中等能量高斯主体 + 稀疏大异常值。列数 256 恰好
# 覆盖 2 个 128-chunk。配合 tensor 级 scale：不旋转时单个异常值拉高整个张量
# 的 scale，把大量中等元素压到粗 grid 上；旋转把异常值能量摊平到每个 chunk，
# 误差随之下降（block=16 的小分组本身已能隔离稀疏异常值，该场景见报告）。
python3 - "$test_dir" <<'PY'
import random
import struct
import sys
from pathlib import Path

out = Path(sys.argv[1])
rng = random.Random(2026)
rows, cols = 64, 256
values = [rng.gauss(0.0, 1.5) for _ in range(rows * cols)]
for index in range(0, rows * cols, 512):
    values[index] = 45.0 if (index // 512) % 2 == 0 else -44.0

header = ("[header]\n"
          f"num_rows: {rows}\n"
          f"num_cols: {cols}\n"
          "dtype: fp32\n\n"
          "[data]\n").encode("ascii")
with (out / "outlier_fp32.bin").open("wb") as tensor_file:
    tensor_file.write(header)
    tensor_file.write(struct.pack(f"<{len(values)}f", *values))

for suffix, mode in (("off", "off"), ("chunk128", "chunk128")):
    (out / f"nvfp4_{suffix}.toml").write_text(
        'format = "nvfp4"\n'
        "block_size = 16\n"
        'scale_mode = "tensor"\n'
        'output_type = "fp16"\n'
        'rounding = "nearest"\n'
        'target_gpu = "T4"\n'
        "seed = 2026\n"
        f'hadamard = "{mode}"\n',
        encoding="utf-8")
PY

"$bench" --input "$test_dir/outlier_fp32.bin" \
  --config "$test_dir/nvfp4_off.toml" \
  --output-dir "$test_dir/off" --warmup 1 --iters 2
"$bench" --input "$test_dir/outlier_fp32.bin" \
  --config "$test_dir/nvfp4_chunk128.toml" \
  --output-dir "$test_dir/chunk128" --warmup 1 --iters 2

# 旋转-量化-反量化-逆旋转的端到端误差必须低于不旋转的直接量化。
python3 - "$test_dir" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
off = json.loads((root / "off" / "metrics.json").read_text())
had = json.loads((root / "chunk128" / "metrics.json").read_text())
assert off["hadamard"] == "off" and had["hadamard"] == "chunk128"
print(f"nvfp4 off:      mae={off['mae']:.6f} "
      f"max_abs={off['max_abs_error']:.6f} mse={off['mse']:.6f}")
print(f"nvfp4 chunk128: mae={had['mae']:.6f} "
      f"max_abs={had['max_abs_error']:.6f} mse={had['mse']:.6f}")
assert had["mae"] < off["mae"], "chunk128 应降低 NVFP4 的 MAE"
assert had["max_abs_error"] < off["max_abs_error"], \
    "chunk128 应降低 NVFP4 的最大绝对误差"
print("PASS hadamard=chunk128 lowers NVFP4 error on the outlier matrix")
PY

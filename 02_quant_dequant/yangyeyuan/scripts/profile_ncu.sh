#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bench=${1:-"$project_dir/build/low_precision_cli"}
fixture_dir="$project_dir/results/profile_fixture"
mkdir -p "$project_dir/results"
# NCU 只采集一次 dequantize launch，减少 profiler 启动和采样开销。
python3 "$project_dir/tests/generate_inputs.py" --output "$fixture_dir" \
  --rows 1024 --cols 1024

ncu --set full --kernel-name regex:dequantize_kernel --launch-count 1 \
  --export "$project_dir/results/dequantize_ncu" --force-overwrite \
  "$bench" --input "$fixture_dir/normal_fp32.bin" \
  --config "$fixture_dir/nvfp4_fp32.toml" \
  --output-dir "$project_dir/results/profile_ncu_run" --warmup 0 --iters 1

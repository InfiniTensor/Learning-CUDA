#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bench=${1:-"$project_dir/build/low_precision_cli"}
fixture_dir="$project_dir/results/profile_fixture"
mkdir -p "$project_dir/results"
# Nsight Systems 记录 CUDA/NVTX/OS runtime 时间线，使用少量迭代避免生成大报告。
python3 "$project_dir/tests/generate_inputs.py" --output "$fixture_dir" \
  --rows 1024 --cols 1024

nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true \
  --output="$project_dir/results/low_precision_nsys" \
  "$bench" --input "$fixture_dir/normal_fp32.bin" \
  --config "$fixture_dir/nvfp4_fp32.toml" \
  --output-dir "$project_dir/results/profile_nsys_run" --warmup 2 --iters 10

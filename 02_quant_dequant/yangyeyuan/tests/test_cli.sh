#!/usr/bin/env bash
set -euo pipefail

bench=$1
source_dir=$2
test_dir=${3:-"$PWD/low_precision_cli_test"}

# 生成小型 fixture，故意使用 35 元素 FP16 输入覆盖尾 nibble 和 16-bit 输出。
python3 "$source_dir/tests/generate_inputs.py" --output "$test_dir/fixtures" \
  --rows 17 --cols 19

"$bench" --self-test
# MXFP8：FP16 输入、BF16 输出。
"$bench" --input "$test_dir/fixtures/small_fp16.bin" \
  --config "$test_dir/fixtures/mxfp8_bf16.toml" \
  --output-dir "$test_dir/mxfp8" --warmup 1 --iters 2
python3 "$source_dir/tests/validate_outputs.py" --run-dir "$test_dir/mxfp8" \
  --elements 35 --format mxfp8 --output-bytes 2

# NVFP4：FP16 输入、FP16 输出，验证另一种 packed layout。
"$bench" --input "$test_dir/fixtures/small_fp16.bin" \
  --config "$test_dir/fixtures/nvfp4_fp16.toml" \
  --output-dir "$test_dir/nvfp4" --warmup 1 --iters 2
python3 "$source_dir/tests/validate_outputs.py" --run-dir "$test_dir/nvfp4" \
  --elements 35 --format nvfp4 --output-bytes 2

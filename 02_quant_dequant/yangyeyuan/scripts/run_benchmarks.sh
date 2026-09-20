#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bench=${1:-"$project_dir/build/low_precision_cli"}
result_dir=${2:-"$project_dir/results"}
fixture_dir="$result_dir/fixtures"

# 先生成固定 seed 的输入，再遍历分布和格式，确保每次基准的输入完全一致。
python3 "$project_dir/tests/generate_inputs.py" --output "$fixture_dir" \
  --rows "${ROWS:-1024}" --cols "${COLS:-1024}"

for distribution in uniform normal outlier; do
  for format in mxfp8 nvfp4; do
    # warmup/iters 可通过环境变量覆盖，结果目录按 distribution_format 分组。
    "$bench" --input "$fixture_dir/${distribution}_fp32.bin" \
      --config "$fixture_dir/${format}_fp32.toml" \
      --output-dir "$result_dir/${distribution}_${format}" \
      --warmup "${WARMUP:-20}" --iters "${ITERS:-200}"
  done
done

echo "Benchmark JSON files are under $result_dir"

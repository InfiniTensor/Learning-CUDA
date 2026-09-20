#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bench=${BENCH_BIN:-"$project_dir/build/hadamard_bench"}
output=${1:-"$project_dir/results/ncu_warp_fht"}

# 只抓取 warp FHT 的一次 launch，避免把预热和其他对照 kernel 混入报告。
ncu --set full --kernel-name regex:warp_fht_kernel \
  --launch-skip 5 --launch-count 1 --export "$output" --force-overwrite \
  "$bench" --batch 4 --seq 128 --heads 32 --dim 128 --dtype fp16 \
  --quant int4 --warmup 0 --iters 1 --no-tensor-core

#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bench=${BENCH_BIN:-"$project_dir/build/hadamard_bench"}
output=${1:-"$project_dir/results/nsys_hadamard"}

# 观察 CUDA kernel、NVTX 标记和 cuBLAS Tensor Core 对照的时间线。
nsys profile --trace=cuda,nvtx,cublas --stats=true --force-overwrite=true \
  --output "$output" "$bench" --batch 4 --seq 128 --heads 32 --dim 128 \
  --dtype fp16 --quant int4 --warmup 5 --iters 20

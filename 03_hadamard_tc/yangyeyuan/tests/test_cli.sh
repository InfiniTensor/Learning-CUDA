#!/usr/bin/env bash
set -euo pipefail

bench=${1:-./build/hadamard_bench}
# 先跑完整自测，再跑两个不同 dtype/维度的命令行冒烟测试。
"$bench" --self-test
"$bench" --batch 1 --seq 8 --heads 2 --dim 64 --dtype fp16 \
  --quant none --warmup 1 --iters 2 --no-tensor-core
"$bench" --batch 1 --seq 8 --heads 2 --dim 256 --dtype bf16 \
  --quant int4 --seed 2026 --warmup 1 --iters 2 --no-tensor-core

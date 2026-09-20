#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bench=${BENCH_BIN:-"$project_dir/build/hadamard_bench"}
log=${1:-"$project_dir/results/performance.csv"}

# 每次运行都重新生成日志，避免不同配置的结果混在同一个 CSV 中。
mkdir -p "$(dirname "$log")"
: > "$log"

# 覆盖两种输入类型、三种 head_dim 和两种主要融合量化格式。
# 无 root 权限无法锁定 GPU 时钟，因此使用较长的 warmup 把频率拉升到稳态，
# 并由程序默认的 5 轮中位数进一步抑制离群值。
for dtype in fp16 bf16; do
  for dim in 64 128 256; do
    for quant in fp8 int4; do
      "$bench" --batch 4 --seq 128 --heads 32 --dim "$dim" \
        --dtype "$dtype" --quant "$quant" --warmup 500 --iters 1000 \
        --csv "$log"
    done
  done
done

echo "Performance log written to $log"

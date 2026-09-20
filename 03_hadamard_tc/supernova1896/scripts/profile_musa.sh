#!/usr/bin/env bash
set -euo pipefail

# MTT S4000 的可重复验证脚本：构建、正确性测试和两种 dtype 性能日志。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MUSA_HOME="${MUSA_HOME:-/usr/local/musa}"
export MUSA_HOME
export PATH="${MUSA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MUSA_HOME}/lib:${LD_LIBRARY_PATH:-}"

"${ROOT_DIR}/scripts/build_musa.sh"
mkdir -p "${ROOT_DIR}/profiling"
rm -f "${ROOT_DIR}/profiling/musa_perf_fp16.csv" \
      "${ROOT_DIR}/profiling/musa_perf_bf16.csv"

"${ROOT_DIR}/build-musa/hadamard_tests_musa" \
  2>&1 | tee "${ROOT_DIR}/profiling/musa_correctness.log"

for dtype in fp16 bf16; do
  "${ROOT_DIR}/build-musa/hadamard_bench_musa" \
    --batch 2 --seq 512 --heads 16 --dim 128 \
    --dtype "${dtype}" --repeats 50 \
    --log "${ROOT_DIR}/profiling/musa_perf_${dtype}.csv"
done

echo "MTT S4000 性能日志已写入 profiling 目录。"

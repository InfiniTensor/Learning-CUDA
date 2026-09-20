#!/usr/bin/env bash
set -euo pipefail

# C500 适配的可重复验证脚本：构建、正确性测试和两种输入类型的性能日志。
# NCU/nsys 是 NVIDIA 工具，不在本脚本中调用；C500 使用 MXCC 和 MACA 运行时。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MACA_PATH="${MACA_PATH:-/opt/maca}"
export MACA_PATH
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${LD_LIBRARY_PATH:-}"

"${ROOT_DIR}/scripts/build_maca.sh"
mkdir -p "${ROOT_DIR}/profiling"
rm -f "${ROOT_DIR}/profiling/maca_perf_fp16.csv" \
      "${ROOT_DIR}/profiling/maca_perf_bf16.csv"

"${ROOT_DIR}/build-maca/hadamard_tests_maca" \
  2>&1 | tee "${ROOT_DIR}/profiling/maca_correctness.log"

for dtype in fp16 bf16; do
  "${ROOT_DIR}/build-maca/hadamard_bench_maca" \
    --batch 2 --seq 512 --heads 16 --dim 128 \
    --dtype "${dtype}" --repeats 50 \
    --log "${ROOT_DIR}/profiling/maca_perf_${dtype}.csv"
done

echo "C500 性能分析日志已写入 profiling 目录。"

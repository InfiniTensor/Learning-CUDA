#!/bin/bash
# Nsight Compute profiling
# 用法: sudo bash kernel/run_test_ncu.sh [run.sh 选项]
#
# 示例:
#   sudo bash kernel/run_test_ncu.sh
#   sudo bash kernel/run_test_ncu.sh --rows 2048 --cols 2048

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"

sudo ncu \
  --target-processes all \
  -k "nf4_dequantize_kernel" \
  -s 10 -c 1 \
  --set full \
  -o profile_result -f \
  "${PROJ_DIR}/run.sh" "$@"

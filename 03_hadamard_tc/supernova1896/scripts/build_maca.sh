#!/usr/bin/env bash
set -euo pipefail

# 沐曦曦云 C500 的独立构建脚本。
# .maca 源文件使用 MXCC 编译，CUDA 兼容运行时需要额外链接 runtime_cu
# 和 symbol_cu；只链接 mcruntime 会留下 wcuda* 未定义符号。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MACA_PATH="${MACA_PATH:-/opt/maca}"
MXCC="${MXCC:-${MACA_PATH}/mxgpu_llvm/bin/mxcc}"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-maca}"

mkdir -p "${BUILD_DIR}"

COMMON_FLAGS=(
  -std=c++17
  -xmaca
  -O3
  -I"${ROOT_DIR}/include"
  -I"${MACA_PATH}/include"
  -I"${MACA_PATH}/tools/cu-bridge/include"
)
LINK_FLAGS=(
  -L"${MACA_PATH}/lib"
  -lruntime_cu
  -lsymbol_cu
  -lmcruntime
  -lmxc-runtime64
  --maca-path="${MACA_PATH}"
  -Wl,-rpath,"${MACA_PATH}/lib"
)

"${MXCC}" "${COMMON_FLAGS[@]}" \
  "${ROOT_DIR}/src/hadamard_maca.maca" \
  "${ROOT_DIR}/src/hadamard_bench_maca.maca" \
  "${LINK_FLAGS[@]}" \
  -o "${BUILD_DIR}/hadamard_bench_maca"

"${MXCC}" "${COMMON_FLAGS[@]}" \
  "${ROOT_DIR}/src/hadamard_maca.maca" \
  "${ROOT_DIR}/tests/test_hadamard_maca.maca" \
  "${LINK_FLAGS[@]}" \
  -o "${BUILD_DIR}/hadamard_tests_maca"

echo "MACA 构建完成：${BUILD_DIR}"

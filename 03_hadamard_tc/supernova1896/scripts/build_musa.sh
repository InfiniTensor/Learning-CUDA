#!/usr/bin/env bash
set -euo pipefail

# 摩尔线程 MTT S4000 的独立 MUSA 构建脚本。
# MUSA 4.3 使用 mcc 编译 .mu 文件，并通过 musart 提供运行时。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MUSA_HOME="${MUSA_HOME:-/usr/local/musa}"
MCC="${MCC:-${MUSA_HOME}/bin/mcc}"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-musa}"
MUSA_ARCH_LIST="${MUSA_ARCH_LIST:-22}"

if [[ ! -x "${MCC}" ]]; then
  echo "未找到 mcc: ${MCC}" >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}"
export MUSA_HOME
export MUSA_ARCH_LIST
export PATH="${MUSA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MUSA_HOME}/lib:${LD_LIBRARY_PATH:-}"

COMMON_FLAGS=(
  -mtgpu
  -O3
  -std=c++17
  -I"${ROOT_DIR}/include"
  -I"${MUSA_HOME}/include"
  -L"${MUSA_HOME}/lib"
)

"${MCC}" "${COMMON_FLAGS[@]}" \
  "${ROOT_DIR}/src/hadamard_musa.mu" \
  "${ROOT_DIR}/src/hadamard_bench_musa.mu" \
  -lmusart \
  -o "${BUILD_DIR}/hadamard_bench_musa"

"${MCC}" "${COMMON_FLAGS[@]}" \
  "${ROOT_DIR}/src/hadamard_musa.mu" \
  "${ROOT_DIR}/tests/test_hadamard_musa.mu" \
  -lmusart \
  -o "${BUILD_DIR}/hadamard_tests_musa"

echo "MUSA 构建完成：${BUILD_DIR}"

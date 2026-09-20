#!/usr/bin/env bash
set -euo pipefail

# MXMACA standalone build. CMake's CUDA language module requires nvcc, while
# MXMACA supplies mxcc, so the two CUDA translation units are compiled directly.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build/metax"
MXCC_BIN="${MXCC_BIN:-$(command -v mxcc || true)}"
if [[ -z "${MXCC_BIN}" ]]; then
  echo "mxcc not found; add it to PATH or set MXCC_BIN" >&2
  exit 1
fi
MXCC_DIR="$(cd "$(dirname "${MXCC_BIN}")" && pwd)"
MACA_ROOT="${MACA_PATH:-$(cd "${MXCC_DIR}/../.." && pwd)}"
MACA_INCLUDE="${MACA_INCLUDE:-${MACA_ROOT}/include}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}/cpu" -G Ninja \
  -DLUXORION_ENABLE_CUDA=OFF
cmake --build "${BUILD_DIR}/cpu" -j "${JOBS:-4}"

INCLUDES=(
  "-I${ROOT_DIR}/gpu_baseline/metax_include"
  "-I${MACA_INCLUDE}"
  "-I${ROOT_DIR}/gpu_baseline/include"
  "-I${ROOT_DIR}/gpu_baseline/src"
  "-I${ROOT_DIR}/cpu_reference/include"
)
mkdir -p "${BUILD_DIR}/objects"

"${MXCC_BIN}" -std=c++17 -c "${ROOT_DIR}/gpu_baseline/src/mxfp8.cu" \
  "${INCLUDES[@]}" -o "${BUILD_DIR}/objects/mxfp8.o"
"${MXCC_BIN}" -std=c++17 -c "${ROOT_DIR}/gpu_baseline/src/nvfp4.cu" \
  "${INCLUDES[@]}" -o "${BUILD_DIR}/objects/nvfp4.o"
"${MXCC_BIN}" -std=c++17 -c "${ROOT_DIR}/gpu_baseline/tools/luxorion_cli.cu" \
  "${INCLUDES[@]}" -o "${BUILD_DIR}/objects/luxorion_cli.o"

"${MXCC_BIN}" "${BUILD_DIR}/objects/luxorion_cli.o" \
  "${BUILD_DIR}/objects/mxfp8.o" "${BUILD_DIR}/objects/nvfp4.o" \
  "${BUILD_DIR}/cpu/libcpu_reference_lib.a" \
  -o "${BUILD_DIR}/luxorion_metax_cli"

# 经过 C500 全输入复测后晋级的方向 7；保留为独立产物，不覆盖 v0 baseline。
"${MXCC_BIN}" -std=c++17 \
  "${ROOT_DIR}/optimization/v7_branch_reduction/luxorion_cli.cu" \
  "${INCLUDES[@]}" "${BUILD_DIR}/cpu/libcpu_reference_lib.a" \
  -o "${BUILD_DIR}/luxorion_metax_v7"

# 正式 baseline benchmark：输出压缩率、量化与反量化有效带宽报告。
"${MXCC_BIN}" -std=c++17 \
  "${ROOT_DIR}/gpu_baseline/tools/benchmark_main.cpp" \
  "${ROOT_DIR}/gpu_baseline/src/mxfp8.cu" \
  "${ROOT_DIR}/gpu_baseline/src/nvfp4.cu" \
  "${INCLUDES[@]}" "${BUILD_DIR}/cpu/libcpu_reference_lib.a" \
  -o "${BUILD_DIR}/gpu_baseline_benchmark_metax"

"${MXCC_BIN}" -std=c++17 "${ROOT_DIR}/gpu_baseline/tests/test_mxfp8_cuda.cpp" \
  "${ROOT_DIR}/gpu_baseline/src/mxfp8.cu" "${INCLUDES[@]}" \
  "${BUILD_DIR}/cpu/libcpu_reference_lib.a" -o "${BUILD_DIR}/test_mxfp8_cuda"
"${MXCC_BIN}" -std=c++17 "${ROOT_DIR}/gpu_baseline/tests/test_nvfp4_cuda.cpp" \
  "${ROOT_DIR}/gpu_baseline/src/nvfp4.cu" "${INCLUDES[@]}" \
  "${BUILD_DIR}/cpu/libcpu_reference_lib.a" -o "${BUILD_DIR}/test_nvfp4_cuda"

ctest --test-dir "${BUILD_DIR}/cpu" --output-on-failure
"${BUILD_DIR}/test_mxfp8_cuda"
"${BUILD_DIR}/test_nvfp4_cuda"

echo "built baseline: ${BUILD_DIR}/luxorion_metax_cli"
echo "built current best: ${BUILD_DIR}/luxorion_metax_v7"
echo "built benchmark: ${BUILD_DIR}/gpu_baseline_benchmark_metax"

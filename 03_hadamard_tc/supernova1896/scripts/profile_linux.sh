#!/usr/bin/env bash
set -u

# Linux 端 profiling 辅助脚本。
# 该脚本不会因为 NCU 权限不足而跳过 Nsight Systems 和 SASS 检查。

set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build-profile"
OUT_DIR="${ROOT_DIR}/profiling"
ARCH="${CUDA_ARCH:-89}"

mkdir -p "${BUILD_DIR}" "${OUT_DIR}"

NVCC="${NVCC:-nvcc}"
NSYS="${NSYS:-$(command -v nsys 2>/dev/null || true)}"
if [[ -z "${NSYS}" && -x /opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys ]]; then
    NSYS=/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys
fi

"${NVCC}" -std=c++17 -O3 -lineinfo -arch="sm_${ARCH}" \
    -I"${ROOT_DIR}/include" \
    "${ROOT_DIR}/src/hadamard_cuda.cu" \
    "${ROOT_DIR}/src/hadamard_bench.cu" \
    -o "${BUILD_DIR}/hadamard_bench"

"${NVCC}" -std=c++17 -O3 -lineinfo -arch="sm_${ARCH}" \
    -I"${ROOT_DIR}/include" \
    "${ROOT_DIR}/src/hadamard_cuda.cu" \
    "${ROOT_DIR}/tests/test_hadamard.cu" \
    -o "${BUILD_DIR}/hadamard_tests"

"${BUILD_DIR}/hadamard_tests" | tee "${OUT_DIR}/correctness_profile_build.log"

if command -v ncu >/dev/null 2>&1; then
    set +e
    ncu --target-processes all \
        --kernel-name regex:hadamard_wmma_fp16_kernel \
        --launch-count 1 \
        --section SpeedOfLight \
        --section InstructionStats \
        --section Occupancy \
        --csv \
        "${BUILD_DIR}/hadamard_bench" \
        --batch 2 --seq 512 --heads 16 --dim 128 --dtype fp16 --repeats 1 \
        --log /tmp/hadamard_ncu_perf.csv \
        > "${OUT_DIR}/ncu_wmma.txt" 2>&1
    echo "NCU exit code: $?" >> "${OUT_DIR}/ncu_wmma.txt"
    set -e
else
    printf '%s\n' "未找到 ncu，跳过 Nsight Compute。" > "${OUT_DIR}/ncu_wmma.txt"
fi

if [[ -n "${NSYS}" && -x "${NSYS}" ]]; then
    "${NSYS}" profile \
        --trace=cuda,nvtx,osrt \
        --force-overwrite=true \
        --output="${OUT_DIR}/nsys_hadamard" \
        "${BUILD_DIR}/hadamard_bench" \
        --batch 2 --seq 512 --heads 16 --dim 128 --dtype fp16 --repeats 5 \
        --log /tmp/hadamard_nsys_perf.csv \
        > "${OUT_DIR}/nsys_run.txt" 2>&1
    "${NSYS}" stats --force-export=true \
        --report cuda_gpu_kern_sum \
        --format csv \
        --output "${OUT_DIR}/nsys_stats" \
        "${OUT_DIR}/nsys_hadamard.nsys-rep" \
        > "${OUT_DIR}/nsys_stats.txt" 2>&1 || true
else
    printf '%s\n' "未找到 nsys，跳过 Nsight Systems。" > "${OUT_DIR}/nsys_run.txt"
fi

if command -v cuobjdump >/dev/null 2>&1; then
    cuobjdump --dump-sass "${BUILD_DIR}/hadamard_bench" \
        | grep -A 8 -B 5 -E 'HMMA|MMA' \
        > "${OUT_DIR}/sass_hmma.txt" || true
fi

echo "profiling 文件已写入：${OUT_DIR}"

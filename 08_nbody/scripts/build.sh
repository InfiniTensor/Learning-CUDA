#!/usr/bin/env bash
# build.sh —— 在 Linux/gcc 上配置并编译。失败立刻退出并打印完整错误。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT}/build"
# sm_89 = RTX 4090D。只编一个架构，编译快一倍。
ARCH="${NBODY_ARCH:-89}"

# 如果 build/ 是从别的机器/路径拷贝过来的（例如本地 Windows 生成的
# build/ 目录被整体上传），CMakeCache.txt 里记着旧的源码/构建路径，
# CMake 会直接报错拒绝复用。与其每次手动 rm -rf，不如自动检测：
# 缓存里的路径和当前 ROOT 不一致就重建，行为始终可预测。
CACHE="${BUILD_DIR}/CMakeCache.txt"
if [ -f "${CACHE}" ] && ! grep -qF "CMAKE_HOME_DIRECTORY:INTERNAL=${ROOT}" "${CACHE}"; then
  echo "stale CMakeCache.txt (from a different path/machine) -> removing ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
fi

echo "=== configure (arch=sm_${ARCH}) ==="
cmake -S "${ROOT}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${ARCH}"

echo
echo "=== build ==="
cmake --build "${BUILD_DIR}" -j "$(nproc)"

echo
echo "=== toolchain ==="
nvcc --version | tail -2
gcc --version | head -1
echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
echo "cores: $(nproc)"
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version \
  --format=csv,noheader
echo
echo "build ok -> ${BUILD_DIR}"

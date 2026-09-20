#!/usr/bin/env bash
# 一键构建 Hadamard 变换加速程序
#   用法: bash build.sh [SM_ARCH]      （默认 90）
set -euo pipefail
cd "$(dirname "$0")"

ARCHS="${1:-90}"
BUILD_DIR="${BUILD_DIR:-build}"

# ---------- 自动探测 CUDA 环境 ----------
CUDA_HOME=""
for d in /usr/local/cuda-12* /usr/local/cuda /usr/local/cuda-11*; do
    if [ -x "$d/bin/nvcc" ]; then CUDA_HOME="$d"; break; fi
done
if [ -z "$CUDA_HOME" ]; then
    echo "ERROR: 未找到 nvcc，请确认已安装 CUDA Toolkit" >&2
    exit 2
fi
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDACXX="$CUDA_HOME/bin/nvcc"

if ! command -v cmake >/dev/null 2>&1; then
    echo "ERROR: 未找到 cmake，请先安装 cmake (>= 3.20)" >&2
    exit 2
fi

echo "[build] CUDA_HOME=$CUDA_HOME  ($("$CUDA_HOME/bin/nvcc" --version | grep -o 'release [0-9.]*' | head -1))"
echo "[build] 目标架构: sm_${ARCHS}"

rm -rf "$BUILD_DIR"
cmake -S . -B "$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${ARCHS}" > /dev/null

cmake --build "$BUILD_DIR" -j"$(nproc)"

echo "构建完成: $(pwd)/$BUILD_DIR/hadamard"

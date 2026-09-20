#!/usr/bin/env bash
# build.sh - 自动构建脚本
set -e
cd "$(dirname "$0")"

export PATH="/usr/local/cuda-12.2/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH}"

mkdir -p build
cd build

cmake -DCMAKE_BUILD_TYPE=Release .. 2>&1 | tail -20
cmake --build . -j"$(nproc)" 2>&1 | tail -40

echo "---- build done ----"
ls -la cuda_pricing 2>/dev/null && echo "OK: ./build/cuda_pricing"

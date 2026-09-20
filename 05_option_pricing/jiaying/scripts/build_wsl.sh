#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${CXX:-g++-13}" \
  -DCMAKE_CUDA_HOST_COMPILER="${CXX:-g++-13}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDAARCHS:-89}"
cmake --build build -j "${BUILD_JOBS:-4}"
ctest --test-dir build --output-on-failure

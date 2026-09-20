#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p reports
cmake -S . -B build-ubsan -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER="${CXX:-g++-13}" -DCMAKE_CUDA_HOST_COMPILER="${CXX:-g++-13}" \
  -DCMAKE_CXX_FLAGS="-fsanitize=undefined -fno-sanitize-recover=all" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=undefined" \
  > reports/ubsan_build.txt 2>&1
cmake --build build-ubsan -j "$(nproc)" >> reports/ubsan_build.txt 2>&1
ctest --test-dir build-ubsan --output-on-failure > reports/ubsan_tests.txt 2>&1

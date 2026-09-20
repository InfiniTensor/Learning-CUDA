#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${CXX:-g++-13}" \
  -DCMAKE_CUDA_HOST_COMPILER="${CXX:-g++-13}" "$@"
cmake --build build -j "$(nproc)"
ctest --test-dir build --output-on-failure
mkdir -p reports
cp build/validation/summary.json reports/validation.json

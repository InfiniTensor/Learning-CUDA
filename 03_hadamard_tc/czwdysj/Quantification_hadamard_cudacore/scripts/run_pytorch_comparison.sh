#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; mode=representative
[[ "${1:-}" == "--full" ]] && mode=full
venv="$root/../hadamard_cudacore/.venv"; test -x "$venv/bin/python"
out="$root/results/pytorch_comparison_$(date -u +%Y%m%dT%H%M%SZ)_$mode"
cmake -S "$root" -B "$root/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$root/build" -j"$(nproc)"; ctest --test-dir "$root/build" --output-on-failure
"$venv/bin/python" "$root/benchmark/benchmark_pytorch_quant.py" --matrix "$mode" --output "$out"
echo "Results: $out"

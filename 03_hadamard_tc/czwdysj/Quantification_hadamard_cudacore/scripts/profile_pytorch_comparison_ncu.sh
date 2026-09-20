#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; venv="$root/../hadamard_cudacore/.venv"; ncu="${NCU_BIN:-/usr/local/cuda/bin/ncu}"
out="${1:-$root/results/pytorch_ncu_$(date -u +%Y%m%dT%H%M%SZ)}"; mkdir -p "$out"
cmake --build "$root/build" -j"$(nproc)"
for impl in pytorch_eager pytorch_compiled; do
  "$ncu" --profile-from-start off --target-processes all --set full --import-source yes --source-folders "$root" \
    --force-overwrite --export "$out/$impl" "$venv/bin/python" "$root/benchmark/benchmark_pytorch_quant.py" \
    --batch 1024 --dim 4096 --dtype fp16 --implementation "$impl" --warmup 100 --repeat 1 --trials 1 --profile --output "$out/${impl}_data"
  "$ncu" --import "$out/$impl.ncu-rep" --page details --print-details header > "$out/${impl}_details.txt"
done
parent="$(dirname "$out")"
[[ -f "$parent/report.md" ]] && "$venv/bin/python" "$root/benchmark/append_pytorch_ncu.py" "$parent/report.md" "$out"
echo "Reports: $out"

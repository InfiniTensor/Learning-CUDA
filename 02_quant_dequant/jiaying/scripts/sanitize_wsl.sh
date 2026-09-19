#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p output reports
python3 scripts/generate.py output/sanitize.bin --rows 19 --cols 1051 --seed 42
for format in mxfp8 nvfp4; do
  for tool in memcheck racecheck synccheck initcheck; do
    compute-sanitizer --tool "$tool" --error-exitcode 99 \
      build/low_precision run --input output/sanitize.bin --config "configs/$format.toml" \
      --weights output/sanitize.weights --output output/sanitize.restored --log output/sanitize.json \
      --iterations 1 --warmup 0 --cpu-repeats 1 >"reports/sanitizer_${format}_${tool}.txt" 2>&1
    cat "reports/sanitizer_${format}_${tool}.txt"
  done
done

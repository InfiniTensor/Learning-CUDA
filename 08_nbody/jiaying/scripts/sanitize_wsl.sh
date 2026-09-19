#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p reports output
python3 scripts/generate.py data/sanitize.txt --particles 257
printf 'dt=0.001\nnum_steps=3\nrecord_interval=2\nsoftening=0.03\n' > output/sanitize.txt
for tool in memcheck racecheck synccheck initcheck; do
  compute-sanitizer --tool "$tool" --error-exitcode 99 build/nbody \
    --input data/sanitize.txt --config output/sanitize.txt \
    --output output/sanitize.bin --log output/sanitize.json \
    > "reports/sanitizer_${tool}.txt" 2>&1
done

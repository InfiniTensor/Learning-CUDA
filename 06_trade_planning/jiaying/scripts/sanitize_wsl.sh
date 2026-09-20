#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p reports output
python3 scripts/generate.py --output data/sanitize --family layered --nodes 1044 --edges 10000
for tool in memcheck racecheck initcheck synccheck; do
  for schedule in thread warp; do
    compute-sanitizer --tool "$tool" --error-exitcode 99 build/trade_flow \
      --graph data/sanitize.bin --queries data/sanitize.queries --schedule "$schedule" --check \
      --output output/sanitize.txt --log output/sanitize.json \
      > "reports/sanitizer_${tool}_${schedule}.txt" 2>&1
  done
done

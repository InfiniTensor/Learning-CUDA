#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ ! -x build/trade_flow ]]; then
  bash scripts/build_wsl.sh
fi
python3 scripts/generate.py --output data/demo --nodes 1044 --edges 10000
build/trade_flow --graph data/demo.bin --queries data/demo.queries --check \
  --output output/demo_results.txt --log output/demo_metrics.json
cat output/demo_results.txt

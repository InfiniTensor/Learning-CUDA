#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python3 scripts/generate.py data/cluster.txt --particles 4096
build/nbody --input data/cluster.txt --config configs/cluster.txt \
  --output output/cluster.bin --log output/cluster.json
python3 visualize.py output/cluster.bin --save output/cluster.gif --snapshot output/cluster.png
python3 scripts/generate.py data/binary.txt --scenario binary
build/nbody --input data/binary.txt --config configs/orbit.txt \
  --output output/binary.bin --log output/binary.json
python3 visualize.py output/binary.bin --stride 4 --trail 80 \
  --title 'TWO-BODY / ORBITAL STABILITY' --save output/binary.gif

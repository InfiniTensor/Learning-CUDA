#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p reports/metrics
python3 scripts/generate.py data/binary.txt --scenario binary
build/nbody --input data/binary.txt --config configs/orbit.txt --output output/binary.bin --log reports/metrics/binary.json > output/binary_run.txt
sed 's/leapfrog/euler/' configs/orbit.txt > output/euler.txt
build/nbody --input data/binary.txt --config output/euler.txt --output output/binary_euler.bin --log reports/metrics/binary_euler.json > output/binary_euler_run.txt
python3 visualize.py output/binary.bin --stride 4 --trail 100 --save reports/binary.gif --snapshot reports/binary.png --title 'TWO-BODY / ORBITAL STABILITY'
python3 visualize.py output/required_cuda_4096_1000.bin --save reports/cluster.gif --snapshot reports/cluster.png --title 'STAR CLUSTER / 4096 BODIES'
python3 visualize.py output/required_cuda_4096_1000.bin --dimensions 3 --stride 4 --save reports/cluster_3d.gif --snapshot reports/cluster_3d.png --title 'STAR CLUSTER / THREE DIMENSIONS'
for scenario in ring collision; do
  python3 scripts/generate.py "data/${scenario}.txt" --scenario "$scenario" --particles 512
  build/nbody --input "data/${scenario}.txt" --config "configs/${scenario}.txt" \
    --output "output/${scenario}.bin" --log "reports/metrics/${scenario}.json" > "output/${scenario}_run.txt"
  python3 visualize.py "output/${scenario}.bin" --risk-radius 0.05 --trail 80 \
    --save "reports/${scenario}.gif" --snapshot "reports/${scenario}.png" --title "${scenario^^} / PROXIMITY AND TRAJECTORIES"
done

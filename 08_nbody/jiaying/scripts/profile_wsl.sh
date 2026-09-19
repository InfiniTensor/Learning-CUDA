#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p reports output
python3 scripts/generate.py data/profile.txt --particles 4096
printf 'dt=0.001\nnum_steps=100\nrecord_interval=10\nsoftening=0.03\n' > output/profile.txt
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --force-overwrite=true --output=output/nbody_trace \
  build/nbody --input data/profile.txt --config output/profile.txt \
  --output output/profile.bin --log output/profile.json \
  > reports/nsys_capture.txt 2>&1
if [[ ! -f output/nbody_trace.nsys-rep && -f output/nbody_trace.qdstrm ]]; then
  "${NSYS_IMPORTER:-/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter}" \
    -i output/nbody_trace.qdstrm -o output/nbody_trace.nsys-rep > reports/nsys_import.txt 2>&1
fi
nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum \
  --format csv output/nbody_trace.nsys-rep > reports/nsys_summary.csv 2> reports/nsys_stats_stderr.txt

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p reports output
if [[ ! -f data/random_100020_1000000.bin ]]; then
  python3 scripts/generate.py --output data/random_100020_1000000 --family random
fi
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --force-overwrite=true --output=output/trade_flow_trace \
  build/trade_flow --graph data/random_100020_1000000.bin \
  --queries data/random_100020_1000000.queries \
  --output output/profile.txt --log output/profile.json \
  > reports/nsys_capture.txt 2>&1
if [[ ! -f output/trade_flow_trace.nsys-rep && -f output/trade_flow_trace.qdstrm ]]; then
  "${NSYS_IMPORTER:-/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter}" \
    -i output/trade_flow_trace.qdstrm -o output/trade_flow_trace.nsys-rep > reports/nsys_import.txt 2>&1
fi
nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_gpu_mem_size_sum,cuda_api_sum \
  --format csv output/trade_flow_trace.nsys-rep > reports/nsys_summary.csv 2> reports/nsys_stats_stderr.txt

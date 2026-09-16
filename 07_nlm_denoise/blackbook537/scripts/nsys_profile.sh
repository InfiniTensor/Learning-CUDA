#!/usr/bin/env bash
# Nsight Systems 时间线采集：确认 NLM kernel 占端到端时间比例、
# 检查 H2D/D2H 气泡，支撑 4K 交互式场景的 stream 分片优化验证。
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p output/work output/results
IMG=output/work/nlm_nsys.png
./build/nlm_denoise gen -o "${IMG}" --size 3840x2160 --channels 3 --sigma 25

nsys profile -o output/work/nsys_4k -f true \
    ./build/nlm_denoise run -i "${IMG}" -o output/work/nlm_nsys_out.png \
        -p params.txt --kernel 2 --log /dev/null

nsys stats output/work/nsys_4k.nsys-rep \
    --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum \
    | tee output/results/nsys_stats_4k.txt
echo "报告文件: output/work/nsys_4k.nsys-rep"

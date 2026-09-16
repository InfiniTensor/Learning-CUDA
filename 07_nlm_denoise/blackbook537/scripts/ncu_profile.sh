#!/usr/bin/env bash
# Nsight Compute 指标采集：对 V0/V1/V2 三个 kernel 版本分别采样，
# 输出 SpeedOfLight / MemoryWorkloadAnalysis / Occupancy 三组关键指标，
# 用于报告中的"瓶颈 -> 优化 -> 指标变化"闭环分析。
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p output/work output/results
IMG=output/work/nlm_ncu.png
./build/nlm_denoise gen -o "${IMG}" --size 1920x1080 --channels 3 --sigma 25

for V in 0 1 2; do
    echo "===== ncu kernel v${V} ====="
    ncu --set speedoflight \
        --section MemoryWorkloadAnalysis \
        --section Occupancy \
        --kernel-name-base demangled \
        -o "output/work/ncu_v${V}" -f \
        ./build/nlm_denoise run -i "${IMG}" -o output/work/nlm_ncu_out.png \
            -p params.txt --kernel "${V}" --log /dev/null
done
echo "报告文件: output/work/ncu_v0.ncu-rep / ncu_v1.ncu-rep / ncu_v2.ncu-rep"

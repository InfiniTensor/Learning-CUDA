#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; out="${1:-$root/results/ncu_$(date -u +%Y%m%dT%H%M%SZ)}"; mkdir -p "$out"
cmake --build "$root/build" -j"$(nproc)"
metrics="gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,launch__shared_mem_per_block,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"
for version in v0 v1 v2; do
  ncu --profile-from-start off --target-processes all --launch-count 1 --metrics "$metrics" --csv --log-file "$out/${version}.csv" \
    "$root/build/benchmark_cuda" --version "$version" --batch 1024 --dim 4096 --dtype fp16 --warmup 100 --repeat 1 --profile >/dev/null
done
report="$(dirname "$out")/report.md"
[[ -f "$report" ]] && python3 "$root/benchmark/report_ncu.py" "$out" "$report"
echo "NCU results: $out"

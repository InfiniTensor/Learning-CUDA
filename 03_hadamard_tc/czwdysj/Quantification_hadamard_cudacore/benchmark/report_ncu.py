#!/usr/bin/env python3
import argparse,csv
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("directory"); p.add_argument("report"); a=p.parse_args()
data={}
for version in ("v0","v1","v2"):
    rows=[r for r in csv.DictReader(line for line in open(Path(a.directory)/f"{version}.csv") if line.startswith('"'))]
    data[version]={r["Metric Name"]:float(r["Metric Value"].replace(",","")) for r in rows}
def g(v,n): return data[v][n]
lines=["", "## Nsight Compute analysis", "", "Profile case: FP16 `[1024,4096]`, one measured launch on RTX 5060 Laptop. NCU instrumentation values are for bottleneck diagnosis and are not substituted for CUDA-event latency.", "",
"| Version | NCU duration (us) | DRAM peak | SM peak | Active warps | Registers | Shared/block | Barrier stall | Long-scoreboard stall |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for v in ("v0","v1","v2"):
    lines.append(f'| fused_{v} | {g(v,"gpu__time_duration.sum")/1000:.3f} | {g(v,"dram__throughput.avg.pct_of_peak_sustained_elapsed"):.2f}% | {g(v,"sm__throughput.avg.pct_of_peak_sustained_elapsed"):.2f}% | {g(v,"sm__warps_active.avg.pct_of_peak_sustained_active"):.2f}% | {g(v,"launch__registers_per_thread"):.0f} | {g(v,"launch__shared_mem_per_block")/1024:.2f} KiB | {g(v,"smsp__warp_issue_stalled_barrier_per_warp_active.pct"):.2f}% | {g(v,"smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"):.2f}% |')
lines += ["", "### Measured bottlenecks", "",
f'- **v0:** DRAM is only {g("v0","dram__throughput.avg.pct_of_peak_sustained_elapsed"):.2f}% of peak while SM utilization is {g("v0","sm__throughput.avg.pct_of_peak_sustained_elapsed"):.2f}%. Its largest sampled stall is long scoreboard ({g("v0","smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"):.2f}%), and the full 256-float reduction raises shared memory to {g("v0","launch__shared_mem_per_block")/1024:.2f} KiB. It is latency/dependency and synchronization constrained rather than DRAM-bandwidth constrained.',
f'- **v1:** vectorized loads and warp reduction cut long-scoreboard stalls from {g("v0","smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"):.2f}% to {g("v1","smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"):.2f}% and reduce NCU duration by {(1-g("v1","gpu__time_duration.sum")/g("v0","gpu__time_duration.sum"))*100:.1f}%. Barrier stalls rise to {g("v1","smsp__warp_issue_stalled_barrier_per_warp_active.pct"):.2f}%, showing that the all-shared FWHT stages are now the clearer limiter.',
f'- **v2:** register/shuffle execution of the first five stages reduces NCU duration another {(1-g("v2","gpu__time_duration.sum")/g("v1","gpu__time_duration.sum"))*100:.1f}% versus v1. SM utilization reaches {g("v2","sm__throughput.avg.pct_of_peak_sustained_elapsed"):.2f}% while DRAM remains only {g("v2","dram__throughput.avg.pct_of_peak_sustained_elapsed"):.2f}%; remaining cost is dominated by arithmetic plus barriers in cross-warp shared-memory stages, not external-memory bandwidth. The next version should target fewer cross-warp synchronizations or multiple rows per block for small D.', ""]
with open(a.report,"a") as f:f.write("\n".join(lines)+"\n")

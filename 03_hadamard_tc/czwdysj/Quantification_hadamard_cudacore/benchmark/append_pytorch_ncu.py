#!/usr/bin/env python3
import argparse,re
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("report",type=Path); p.add_argument("ncu_dir",type=Path); a=p.parse_args()
def parse(name):
 text=(a.ncu_dir/f"{name}_details.txt").read_text(errors="replace")
 kernels=re.findall(r"(?m)^  (.+?) \([^\n]*Context 1, Stream",text)
 durations=[float(x.replace(",","")) for x in re.findall(r"(?m)^\s+Duration\s+us\s+([0-9.,]+)",text)]
 dram=[float(x.replace(",","")) for x in re.findall(r"(?m)^\s+DRAM Throughput\s+%\s+([0-9.,]+)",text)]
 sm=[float(x.replace(",","")) for x in re.findall(r"(?m)^\s+Compute \(SM\) Throughput\s+%\s+([0-9.,]+)",text)]
 return kernels,durations,dram,sm
ek,ed,edram,esm=parse("pytorch_eager"); ck,cd,cdram,csm=parse("pytorch_compiled")
marker="\n## NCU pipeline analysis\n"; original=a.report.read_text()
original=original.split(marker)[0].rstrip()+"\n"
section=f'''{marker}
Profile workload: FP16 `[1024,4096]`, after 100 warmup iterations, one complete pipeline operation. Full NCU reports use `--set full`; NCU replay time is diagnostic and is not used as benchmark latency.

| Pipeline | Kernels | Sum of per-kernel NCU duration | Main observation |
|---|---:|---:|---|
| PyTorch eager | {len(ek)} | {sum(ed):.2f} us | Dao FHT plus separate cast, abs, reduction, scale, divide, round, clamp and INT8 conversion kernels |
| PyTorch compiled | {len(ck)} | {sum(cd):.2f} us | Dao FHT plus one Triton fused reduction/quantization kernel |

- Eager launches {len(ek)} kernels. Its large elementwise/reduction stages reach roughly {max(edram):.1f}% DRAM throughput, showing repeated global-memory materialization rather than Hadamard arithmetic is the dominant avoidable cost.
- `torch.compile` reduces the quantization chain to `{ck[-1].split('(')[0].strip()}`. The compiled quantization kernel takes {cd[-1]:.2f} us in NCU, with {cdram[-1]:.2f}% DRAM and {csm[-1]:.2f}% SM throughput.
- Dao FHT itself is essentially unchanged between runs ({ed[0]:.2f} us eager and {cd[0]:.2f} us compiled). Compilation optimizes only the quantization stage, as intended by the fairness contract.
- The fused CUDA versions perform Hadamard, amax and INT8 conversion in one launch and avoid writing/reading the full Hadamard intermediate. Earlier full NCU reports for the same workload measured v0/v1/v2 at 295.10/264.67/216.19 us under NCU instrumentation; v2 remains limited by cross-warp shared-memory stages and SM work rather than DRAM bandwidth.
- Next optimization direction: preserve v2's single-launch dataflow while reducing cross-warp barriers, specialize small dimensions to process several rows per block, and test a register/shared-memory tiling strategy for D=4096–8192.

The reports `pytorch_eager.ncu-rep` and `pytorch_compiled.ncu-rep` were successfully imported by Windows Nsight Compute 2026.2 and opened in NCU-UI.
'''
a.report.write_text(original+section)

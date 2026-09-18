# L40S profiler run 16967596

- Date: 2026-09-04
- GPU: NVIDIA L40S (SM 8.9)
- Driver / CUDA: 580.82.07 / 13.0.88
- Nsight Systems / Compute: 2025.3.2 / 2025.3.1
- Shape: 4 x 1024 x 32 tokens, head_dim 64/128/256
- Timing: 20 warmups + 100 measured iterations

`timings.csv` is the profiler-free CUDA Event matrix. `nsys_*_summary.csv` is
exported from the representative FP16/d=128 Systems trace. `profile_metrics.csv`
contains timing-derived Roofline inputs; `intensity_byte_source` is deliberately
set to `minimum_io_fallback` because the node's performance counters were busy.
`figures/nsys_timeline.png` visualizes the 56 captured kernel launches and CUDA
Runtime API totals directly from the Systems SQLite export.

All six Nsight Compute attempts returned `driver resource was unavailable`,
which normally indicates DCGM or another profiler owning the counters. The logs
are retained as evidence; no achieved-occupancy/DRAM/stall value was inferred
from them.

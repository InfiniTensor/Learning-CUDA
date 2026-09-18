# Optimized FHT and fused INT4 results

Shape: `4 x 1024 x 32` = `131072` tokens; warmup `20`, measured `100` runs.

## 9.1 optimized Hadamard

| dtype | head_dim | baseline us | optimized us | speedup | bit-exact |
|---|---:|---:|---:|---:|---:|
| BF16 | 64 | 73.55 | 21.51 | 3.42x | PASS |
| BF16 | 128 | 109.52 | 38.91 | 2.82x | PASS |
| BF16 | 256 | 240.99 | 192.82 | 1.25x | PASS |
| FP16 | 64 | 73.43 | 21.50 | 3.42x | PASS |
| FP16 | 128 | 109.37 | 38.62 | 2.83x | PASS |
| FP16 | 256 | 240.74 | 193.64 | 1.24x | PASS |

## 9.2 fused per-token symmetric INT4

| dtype | head_dim | unfused us | fused us | fusion speedup | compression | MAE | exact checks |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 64 | 92.79 | 73.43 | 1.26x | 3.56x | 0.09122 | PASS |
| BF16 | 128 | 110.56 | 73.54 | 1.50x | 3.76x | 0.1002 | PASS |
| BF16 | 256 | 240.69 | 103.40 | 2.33x | 3.88x | 0.1084 | PASS |
| FP16 | 64 | 92.68 | 73.34 | 1.26x | 3.56x | 0.09122 | PASS |
| FP16 | 128 | 110.49 | 73.41 | 1.51x | 3.76x | 0.1002 | PASS |
| FP16 | 256 | 240.77 | 104.47 | 2.30x | 3.88x | 0.1084 | PASS |

`exact checks` requires both fused-vs-unfused and GPU-vs-CPU INT4 packed bytes/scales to be bit-exact.

Extended smoke matrix: **24/24 PASS** across FP16/BF16, normalized/non-normalized, and d=32/64/128/256/512/1024. See `full_shape_checks.log`.

## Nsight Systems representative trace

FP16, d=128, 16384 tokens (5 warmup + 20 measured runs):

| kernel | calls | avg us | grid | block | registers/thread | static shared B | resource-limit occupancy |
|---|---:|---:|---:|---:|---:|---:|---:|
| hadamard_kernel | 26 | 14.86 | 16384x1x1 | 128x1x1 | 18 | 512 | 100% |
| hadamard_optimized_kernel | 52 | 5.83 | 8192x1x1 | 128x1x1 | 20 | 1024 | 100% |
| hadamard_fused_quant_int4_kernel | 26 | 10.27 | 16384x1x1 | 64x1x1 | 18 | 524 | 100% |
| quantize_int4_kernel | 26 | 9.91 | 16384x1x1 | 64x1x1 | 18 | 12 | 100% |

Occupancy is a launch-resource upper bound derived from Systems metadata, not counter-measured achieved occupancy.

Nsight Compute: 2/2 attempts could not acquire performance counters because a driver resource (typically DCGM) was busy; the raw logs are retained under `ncu/`.

![Optimization and fused INT4 plots](figures/advanced_performance.png)

## Environment

```text
2026-09-04T17:47:07-04:00
NVIDIA L40S, GPU-bb1266fd-dd1d-252b-f4a9-955639898f8b, 8.9, 46068 MiB, 580.82.07, 2520 MHz, 9001 MHz
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Aug_20_01:58:59_PM_PDT_2025
Cuda compilation tools, release 13.0, V13.0.88
Build cuda_13.0.r13.0/compiler.36424714_0
```

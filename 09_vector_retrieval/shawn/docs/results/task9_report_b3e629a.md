# Task 09 RTX 5090 Evidence — `b3e629a`

> Evidence status: values below were transcribed from user-provided server terminal output on 2026-09-10. Raw CSV, `.nsys-rep`, SQLite, result files and logs remain under `/root/infra_cuda_2026/task-09-gpu-vector-search/outputs/task9_report_b3e629a/` on the GPU server and must be copied back before final report freeze.

## Run identity

- Git commit: `b3e629a`
- GPU: NVIDIA GeForce RTX 5090, 32 GB
- Formal scale: N=1,000,000, D=128, Q=1,000
- CPU reference queries: 20
- Metric/dtype: L2/FP32

## Exact Top-K excerpt

| K | Reps | Recall@K | Mean QPS | QPS std | P50 ms | P99 ms | CPU ref QPS | Speedup | Device bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 3 | 1 | 1489.005233 | 2.361829 | 0.518709 | 0.530334 | 13.188665 | 143.998785 | 548183295 |
| 100 | 3 | 1 | 1486.491053 | 1.941097 | 0.518643 | 0.530080 | 13.143730 | 144.830018 | 548183295 |

K=50 到 K=100 的 QPS 仅下降 0.169%，与“当前 exact 路径对全部 N 个 score 做 radix sort，K 只影响少量结果回传”的实现一致。K=1/10 行尚待从服务器 `exact_topk/aggregate.csv` 回填本地证据。

## IVF nprobe sweep

| nprobe | Reps | Recall@10 | Mean QPS | QPS std | P50 ms | P99 ms | Candidates/query | CPU ref QPS | Speedup | Device bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 1 | 1327.587843 | 3.136449 | 0.592813 | 0.634965 | 467.405 | 13.201937 | 125.511356 | 512142207 |
| 4 | 5 | 1 | 1086.390822 | 2.829201 | 0.766684 | 0.883830 | 5934.018 | 13.350362 | 95.204000 | 512481471 |
| 16 | 5 | 1 | 633.062800 | 0.679952 | 1.411880 | 1.671482 | 27008.017 | 12.695072 | 53.569053 | 513554495 |
| 64 | 5 | 1 | 246.516816 | 0.098931 | 3.759094 | 4.307572 | 87433.332 | 13.104593 | 19.112216 | 516252863 |
| 256 | 5 | 1 | 97.927395 | 1.488330 | 9.717852 | 11.390438 | 220088.434 | 12.804180 | 7.699922 | 522079359 |

在这组容易的 clustered synthetic 查询上，nprobe=1 已达到 Recall@10=1；因此增大 nprobe 只增加候选数和延迟。nprobe=1 到 256 的候选数增加 470.87 倍，QPS 下降 92.62%，P99 增加 17.94 倍。nprobe=1 的 QPS 仍比 exact K=100 低 10.69%，所以当前证据不能支持“IVF 比 CUDA exact 更快”的结论。

## IVF batch sweep

| Batch | Reps | Recall@10 | Mean QPS | QPS std | P50 ms | P99 ms | Candidates/query | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 1 | 630.617573 | 6.546663 | 1.416869 | 1.673444 | 27008.017 | 53.480933 |
| 32 | 5 | 1 | 618.564893 | 32.477817 | 1.418684 | 1.674368 | 27008.017 | 53.317101 |
| 128 | 5 | 1 | 631.302201 | 2.932556 | 1.411555 | 1.670336 | 27008.017 | 53.654633 |

三种 batch 的最大 QPS 差距为 2.02%，且 Batch=32 的标准差为 32.48 QPS；现有数据只支持“当前逐 query 实现没有观察到稳定 batch 收益”。

## Nsight Systems — IVF, nprobe=16

### CUDA API summary

| API | Time share | Total time | Calls | Median | Maximum |
|---|---:|---:|---:|---:|---:|
| `cudaMemcpy` | 57.2% | 234.999557 ms | 4001 | 11.718 us | 163.095999 ms |
| `cudaMalloc` | 34.2% | 140.480171 ms | 8 | 3.766 us | 140.358871 ms |
| `cudaLaunchKernel` | 4.8% | 19.848133 ms | 7000 | 2.654 us | 98.703 us |
| `cudaMemsetAsync` | 3.5% | 14.167021 ms | 6000 | 2.524 us | 20.983 us |
| `cudaFree` | 0.3% | 1.128204 ms | 8 | 3.751 us | 0.740212 ms |

### GPU kernel summary

| Kernel | Time share | Total time | Instances | Average |
|---|---:|---:|---:|---:|
| CUB radix-sort onesweep | 60.1% | 24.570027 ms | 4000 | 6.143 us |
| `score_candidates_kernel` | 32.9% | 13.444356 ms | 1000 | 13.444 us |
| CUB radix-sort histogram | 4.7% | 1.922387 ms | 1000 | 1.922 us |
| CUB radix-sort exclusive sum | 2.2% | 0.918905 ms | 1000 | 0.919 us |

Measured GPU kernel time is dominated by CUB sorting rather than distance scoring: the three radix-sort kernels account for 67.0% of the listed kernel time, versus 32.9% for `score_candidates_kernel`. This breakdown is observational; hardware-counter throughput and stall attribution are unavailable.

### GPU memory operations

| Operation | Time share | Total time | Count | Median | Maximum | Total bytes |
|---|---:|---:|---:|---:|---:|---:|
| H2D memcpy | 97.2% | 167.639463 ms | 2001 | 2.784 us | 162.737156 ms | 728.576 MB |
| memset | 2.4% | 4.183461 ms | 6000 | 0.704 us | 1.376 us | 25.256 MB |
| D2H memcpy | 0.4% | 0.697311 ms | 2000 | 0.352 us | 0.928 us | 0.120 MB |

The single 512 MB database upload takes 162.737 ms and dominates H2D time. The remaining H2D volume is consistent with per-query query/candidate staging. D2H result traffic is negligible.

## Nsight Compute availability

Both exact and IVF attempts failed before collection with:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.
```

No `.ncu-rep` was generated. Do not report DRAM/L2 throughput, achieved occupancy or stall reasons as measured values unless the provider enables host performance-counter access and ncu is rerun.

### Permission cross-check against Task 5

The same server/container was inspected after the Task 9 run:

```text
Task 5 ncu artifacts: profile_ncu.console.log only; no .ncu-rep
RmProfilingAdminOnly: 1
cap_sys_admin: not present in capsh output
```

This confirms that Task 5 did not produce an importable Nsight Compute report in the current environment either. The earlier Task 5 artifact is only the profiler console log. The failure is an environment-level performance-counter permission restriction, not a Task 9 kernel or build failure. Nsight Systems remains usable because its CUDA trace collection does not require the same protected hardware-counter access.

# NF4 Dequant CUDA Report

## 1. Scope

This project implements a single-kernel NF4 (double-quant) dequant operator with:

- Input format support: `packed_weights + absmax_q + absmax2 + code2 + offset`
- Output type support: BF16 and FP16
- Required vectorization rule: each thread handles 2 x 4-bit indices
- Required packed store rule: one `uint32_t` store for two outputs when in-bounds
- Dynamic shape and boundary support
- Dynamic `blocks_per_group` support

## 2. Environment

- Build: CMake + CUDA
- Current measured GPU from profiler output: `CC 8.9`
- Notes:
- `params.txt` target label is `T4`, but measured runs in this report were collected on a `CC 8.9` device.
- Absolute throughput and timing will differ on real T4.

## 3. Implementation Summary

### 3.1 Correctness-critical kernel behavior

- Nibble order is high-first, then low:
- `idx0 = byte >> 4`, `idx1 = byte & 0x0F`
- Scale path:
- `scale = code2[absmax_q[block_id]] * absmax2[group_id] + offset`
- Dynamic grouping:
- `group_id = block_id / blocks_per_group`
- Read-only cache path:
- `__ldg` on `packed_weights`, `absmax_q`, `absmax2`, `code2`
- Packed write path:
- BF16: two BF16 values packed into one `uint32_t`
- FP16: two FP16 values packed into one `uint32_t`
- Tail handling:
- if `idx + 1 >= num_elements`, scalar write of first value only

### 3.2 Host/runtime engineering

- Device buffer reuse cache (`reuse_device_buffers=true`) for:
- `d_packed_weights`, `d_absmax_q`, `d_absmax2`, `d_code2`, `d_output`
- Input upload dedup by state signature
- Async transfer path in reuse mode:
- H2D and D2H use `cudaMemcpyAsync` on a non-blocking stream
- In-process profiling loop control:
- `profile_loop_iters`
- `profile_loop_iters > 1`: first loop excluded from capture, capture only last `N-1`

## 4. Accuracy Validation

Reference: bitsandbytes dequant path.

- Command:
- `python tests/compare_with_bnb.py --weights-bin tests/data/nf4_r4096_c4096_bs64_bpg256_weights.bin --params tests/data/nf4_r4096_c4096_bs64_bpg256_params_bf16.txt --out-bin tests/data/_report_cmp_large.bin`
- Result:
- `MAE = 0.00115161` (pass, threshold `1e-2`)
- `Max Error = 0.01827669`

## 5. Performance Snapshot (Non-profiler run)

- Command:
- `./build/Release/nf4_dequant.exe tests/data/nf4_r4096_c4096_bs64_bpg256_weights.bin params.txt tests/data/_report_perf_out.bin`
- Result:
- `Kernel time = 1.1571 ms`
- `Effective bandwidth = 36.48 GB/s`
- `Speedup vs bnb = 3.14x` (using `bnb_time_ms=3.634666`)

## 6. Nsight Systems Results

Source files:

- `tests/data/nsys_steady_median.csv`
- `tests/data/nsys_cold_median.csv`

### 6.1 Steady-state median (5 rounds)

Reuse on vs off:

- `kernel_avg_ns`: `+1.54%` (effectively flat)
- `d2h_total_ns`: `+0.29%` (effectively flat)
- `h2d_total_ns`: `-100.00%`
- `cudaMemcpy_api_total_ns`: `-99.76%`
- `cudaMalloc_api_total_ns`: `-100.00%`

Interpretation:

- In steady-state capture window, input re-upload and device malloc overhead are effectively eliminated.
- Kernel compute itself is unchanged; gains come from runtime overhead removal.

### 6.2 Cold-start median (5 rounds)

Reuse on vs off:

- `kernel_avg_ns`: `-0.21%` (flat)
- `d2h_total_ns`: `-0.26%` (flat)
- `h2d_total_ns`: `+11.86%` (small regression/noise range for first-run path)
- `cudaMemcpy_api_total_ns`: `-49.70%`
- `cudaMalloc_api_total_ns`: `+1.50%` (flat)
- `cudaMallocHost_api_total_ns`: `+0.65%` (flat, and dominant cold-start cost)

Interpretation:

- Cold-start is still dominated by one-time pinned host allocation and startup costs.
- Reuse strategy mainly improves steady-state behavior.

## 7. Nsight Compute Snapshot

- Command:
- `ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed ./build/Release/nf4_dequant.exe tests/data/nf4_r4096_c4096_bs64_bpg256_weights.bin params.txt tests/data/_report_ncu_out.bin`
- Metrics:
- `dram__throughput.avg.pct_of_peak_sustained_elapsed = 32.32%`
- `sm__throughput.avg.pct_of_peak_sustained_elapsed = 75.89%`
- Note:
- ncu inflates runtime due to multi-pass instrumentation, so ncu kernel time is not used as runtime KPI.

## 8. Recommended Defaults

For production-like long-running process:

- `block_dim = 256`
- `reuse_device_buffers = true`
- `use_pinned_host_output = true`
- `kernel_warmup_iters = 0`
- `profile_loop_iters = 1`

For stable steady-state profiling:

- `profile_loop_iters = 6`
- use nsys capture range with `cudaProfilerStart/Stop`

For cold-start profiling:

- `profile_loop_iters = 1`
- disable nsys capture range

## 9. Repro Commands

Steady and cold reports together:

```powershell
powershell -ExecutionPolicy Bypass -File tests/run_nsys_steady_cold_report.ps1 -WeightsBin tests/data/nf4_r4096_c4096_bs64_bpg256_weights.bin -ParamsTemplate tests/data/params_nsys_pinned.txt -OutputDir tests/data -Rounds 5 -SteadyProfileLoopIters 6
```

Single mode steady:

```powershell
powershell -ExecutionPolicy Bypass -File tests/run_nsys_reuse_compare.ps1 -WeightsBin tests/data/nf4_r4096_c4096_bs64_bpg256_weights.bin -ParamsTemplate tests/data/params_nsys_pinned.txt -OutputDir tests/data -Rounds 5 -ProfileLoopIters 6 -UseCudaProfilerRange:$true -RunTag steady
```

Single mode cold:

```powershell
powershell -ExecutionPolicy Bypass -File tests/run_nsys_reuse_compare.ps1 -WeightsBin tests/data/nf4_r4096_c4096_bs64_bpg256_weights.bin -ParamsTemplate tests/data/params_nsys_pinned.txt -OutputDir tests/data -Rounds 5 -ProfileLoopIters 1 -UseCudaProfilerRange:$false -RunTag cold
```

## 10. Next Optimizations

- Add persistent service-style API so host pinned output is also reused across multiple external requests.
- Add stream overlap strategy when output copy is not immediately needed by host.
- Evaluate optional `half2` arithmetic and reduced conversion overhead in BF16 path.
- Add CI benchmark gate with fixed dataset and median-of-N policy.

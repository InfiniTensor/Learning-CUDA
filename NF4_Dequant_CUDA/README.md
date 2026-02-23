# NF4_Dequant_CUDA

Single-kernel NF4 (double-quant) dequantization project.

## Build

```bash
cmake -S . -B build
cmake --build build --config Release
```

## Run

```bash
./build/Release/nf4_dequant.exe <weights.bin> <params.txt> <output.bin>
```

Example:

```bash
./build/Release/nf4_dequant.exe tests/data/nf4_r64_c128_bs64_bpg256_weights.bin tests/data/nf4_r64_c128_bs64_bpg256_params_bf16.txt tests/data/out.bin
```

Program output prints:

- Kernel time (ms)
- Effective bandwidth (GB/s)
- Speedup vs bitsandbytes (if `bnb_time_ms` is set in params)
- Selected block dimension (`block_dim`)

It also auto-generates a performance log:

```text
<output.bin>.perf.log
```

## Params

Supported keys in `params.txt`:

- `blocksize` (required)
- `compute_type` (`bf16` or `fp16`)
- `target_gpu` (string label for logs)
- `blocks_per_group` (default `256`)
- `bnb_time_ms` (optional, enables speedup print)
- `block_dim` (default `256`)
- `autotune_block_dim` (`true/false`, default `false`)
- `autotune_repeats` (default `5`)
- `kernel_warmup_iters` (default `0`, extra in-process warmup runs without D2H copy)
- `profile_loop_iters` (default `1`; when `>1`, run N loops in one process and start profiler capture from loop 2, i.e. capture last `N-1`)
- `reuse_device_buffers` (`true/false`, default `true`)
- `use_pinned_host_output` (`true/false`, default `true`)
- `perf_log_path` (optional custom log path)

Note: when `autotune_block_dim=true`, candidate runs measure kernel time only (skip D2H copy) and the final selected run still performs full D2H output copy.

Example A/B auto-tune params:

```text
blocksize = 64
compute_type = "bf16"
target_gpu = "T4"
blocks_per_group = 256
bnb_time_ms = 3.691622
block_dim = 256
autotune_block_dim = true
autotune_repeats = 5
kernel_warmup_iters = 0
profile_loop_iters = 1
reuse_device_buffers = true
use_pinned_host_output = true
```

## Profiling

Use Nsight Compute:

```bash
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed ./build/Release/nf4_dequant.exe <weights.bin> <params.txt> <output.bin>
```

If a metric name is rejected on your Nsight Compute version, list available ones first:

```bash
ncu --query-metrics | rg "throughput"
```

Use Nsight Systems:

```bash
"C:/Program Files/NVIDIA Corporation/Nsight Systems 2024.4.2/target-windows-x64/nsys.exe" profile --trace=cuda --sample=none --cpuctxsw=none --stats=true -o tests/data/nsys_run ./build/Release/nf4_dequant.exe <weights.bin> <params.txt> <output.bin>
```

On Windows, `nsys profile` may require running terminal as Administrator.

One-shot helper script (profile + CSV stats export):

```powershell
powershell -ExecutionPolicy Bypass -File tests/run_nsys_profile.ps1 -WeightsBin tests/data/nf4_r4096_c4096_bs64_bpg256_weights.bin -ParamsFile tests/data/params_bd256.txt -OutputBin tests/data/out_nsys.bin -ReportStem tests/data/nsys_run
```

When your executable uses `cudaProfilerStart/Stop` (for example `profile_loop_iters > 1`), add:

```powershell
-UseCudaProfilerRange:$true
```

Compare two exported nsys report stems:

```bash
python tests/compare_nsys_csv.py --base-stem tests/data/nsys_pageable --new-stem tests/data/nsys_pinned
```

Median compare across multiple rounds:

```bash
python tests/compare_nsys_csv.py --base-stems tests/data/nsys_reuse_off_r1,tests/data/nsys_reuse_off_r2 --new-stems tests/data/nsys_reuse_on_r1,tests/data/nsys_reuse_on_r2
```

One-shot compare for `reuse_device_buffers=false` vs `true`:

```powershell
powershell -ExecutionPolicy Bypass -File tests/run_nsys_reuse_compare.ps1 -WeightsBin tests/data/nf4_r4096_c4096_bs64_bpg256_weights.bin -ParamsTemplate tests/data/params_nsys_pinned.txt -OutputDir tests/data
```

Default behavior of `run_nsys_reuse_compare.ps1`:

- repeats `5` rounds
- injects `profile_loop_iters = 6` and `kernel_warmup_iters = 0`
- each profiled process captures only loops `2..6` (last `N-1`)
- uses `--capture-range=cudaProfilerApi` via `-UseCudaProfilerRange:$true`
- prints median comparison across all rounds
- writes median CSV to `tests/data/nsys_<tag>_median.csv` (default tag: `steady`)

Cold-start median compare (includes first-run allocations/copies):

```powershell
powershell -ExecutionPolicy Bypass -File tests/run_nsys_reuse_compare.ps1 -WeightsBin tests/data/nf4_r4096_c4096_bs64_bpg256_weights.bin -ParamsTemplate tests/data/params_nsys_pinned.txt -OutputDir tests/data -Rounds 5 -ProfileLoopIters 1 -UseCudaProfilerRange:$false -RunTag cold
```

Run both steady-state and cold-start reports in one command:

```powershell
powershell -ExecutionPolicy Bypass -File tests/run_nsys_steady_cold_report.ps1 -WeightsBin tests/data/nf4_r4096_c4096_bs64_bpg256_weights.bin -ParamsTemplate tests/data/params_nsys_pinned.txt -OutputDir tests/data -Rounds 5 -SteadyProfileLoopIters 6
```

## Tests

Generate Phase-5 suite:

```bash
python tests/generate_test_data.py --phase5-suite --skip-ref-fp32
```

Compare against bitsandbytes:

```bash
python tests/compare_with_bnb.py --suite --auto-generate-missing
```

## Measure bnb_time_ms

Benchmark bitsandbytes reference path and print average time (ms):

```bash
python tests/bench_bnb.py --weights-bin tests/data/nf4_r4096_c4096_bs64_bpg256_weights.bin --params tests/data/nf4_r4096_c4096_bs64_bpg256_params_bf16.txt --warmup 20 --iters 200
```

Write measured `bnb_time_ms` back to params file:

```bash
python tests/bench_bnb.py --weights-bin tests/data/nf4_r4096_c4096_bs64_bpg256_weights.bin --params tests/data/nf4_r4096_c4096_bs64_bpg256_params_bf16.txt --warmup 20 --iters 200 --update-params
```

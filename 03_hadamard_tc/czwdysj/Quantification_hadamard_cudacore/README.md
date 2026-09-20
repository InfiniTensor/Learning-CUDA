# Fused normalized Hadamard + dynamic INT8 quantization

This project computes `y = x @ H(D) / sqrt(D)`, then rowwise `amax=max(abs(y))`, `scale=amax/127` (or 1 for an all-zero row), and `q=round(clamp(y/scale,-127,127))`. Every implementation uses FP32 butterflies and emits INT8 data plus FP32 rowwise amax and scale.

## Build and correctness

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
./build/benchmark_cuda --correctness-only --batch 4 --dim 1024 --dtype all --pattern random
```

Supported input dtypes are FP32, FP16, and BF16. `D` must be a power of two from 32 through 8192. Correctness compares the INT8 output, amax, and scale against a CPU FP32 iterative FWHT. INT8 error may be at most one quantization level.

## Benchmark

```bash
./scripts/run_benchmark.sh
./scripts/run_benchmark.sh --full
./build/benchmark_cuda --version v2 --batch 1024 --dim 4096 --dtype fp16 --warmup 100 --repeat 1000
```

CUDA events measure only launches: allocation, initialization, copies, validation, and CSV I/O are outside the timed region. Each result uses three trials and reports the median. `speedup_vs_separate = separate/fused`; values above 1 mean fusion is faster. `percent_latency_reduction` is positive when fusion reduces latency. A range above 5% of the median is marked unstable.

The separate baseline performs three mathematically equivalent kernels and materializes the normalized Hadamard result as FP32 global memory. The fused versions never materialize that intermediate. See `VERSIONING.md` for the optimization history.

## Nsight Compute

```bash
./scripts/profile_ncu.sh
./scripts/profile_ncu_ui.sh
ncu --profile-from-start off --launch-count 1 ./build/benchmark_cuda \
  --version v2 --batch 1024 --dim 4096 --dtype fp16 --warmup 100 --repeat 1 --profile
```

Profile mode keeps normal warmup but executes at most three measured operations. `cudaProfilerStart/Stop` confines NCU collection to that region. The generated report records measured bottlenecks; profile permissions or GPU contention can still prevent collection and must be reported rather than replaced with invented data.

`profile_ncu.sh` produces lightweight CSV metrics. `profile_ncu_ui.sh` produces one
full, source-embedded `.ncu-rep` file for each fused version after 100 synchronized
warmup launches. These reports can be opened directly in NVIDIA Nsight Compute UI.

## PyTorch comparison

The framework compares every fused version and the three-kernel CUDA baseline with
two mathematically equivalent PyTorch pipelines: Dao-AILab FHT plus eager PyTorch
quantization operators, and Dao-AILab FHT plus a `torch.compile`-fused quantization
function. All paths use the same CUDA tensor and stream. Ratios are emitted only
after INT8, amax, and scale pass the cross-implementation correctness gate.

```bash
./scripts/run_pytorch_comparison.sh
./scripts/run_pytorch_comparison.sh --full
./scripts/profile_pytorch_comparison_ncu.sh
```

The PyTorch environment is reused from `../hadamard_cudacore/.venv`, where the
pinned Dao extension and CUDA-enabled PyTorch are installed. `speed_vs_pytorch_*`
is `PyTorch latency / fused latency`, so values above one mean the fused kernel is
faster. PyTorch allocation uses the warmed caching allocator and is included in the
end-to-end library-pipeline timing; input construction and compilation are excluded.

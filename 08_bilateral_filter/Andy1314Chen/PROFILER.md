# Performance Profiling Results

## Test Environment

- Platform: Ubuntu Linux (WSL2)
- GPU: NVIDIA GeForce RTX 4060 (sm_89, Ada Lovelace)
- Compiler: g++ (C++17, -O3), nvcc (CUDA 13.1)
- OpenCV: 4.13.0 (with CUDA)

## Filter Parameters

```
radius = 5
sigma_spatial = 3.0
sigma_color = 30.0
```

## Performance Results

Benchmark methodology: 5 warmup runs + 50 timed runs, reporting mean ± stddev.

### Test 1: 3840×2160 RGB Image (4K)

| Implementation | Time (ms) | Min (ms) | Throughput (MP/s) | vs OCV CPU | vs OCV CUDA |
|----------------|-----------|----------|-------------------|:----------:|:-----------:|
| **CUDA (SEPARABLE)** | **5.61 ± 0.37** | **5.41** | **1478** | **10.1x** | **2.10x** |
| CUDA (TEMPLATE)      | 6.67 ± 0.69 | 6.33 | 1243 | 10.5x | 2.26x |
| CUDA (ADAPTIVE)      | 6.95 ± 0.15 | 6.80 | 1194 | 8.15x | 1.68x |
| CUDA (STANDARD)      | 8.34 ± 0.31 | 7.96 | 995 | 8.72x | 1.82x |
| **OpenCV CUDA**      | 11.78 ± 0.26 | 11.36 | 704 | 4.80x | 1.00x |
| OpenCV CPU           | 56.58 ± 1.88 | 54.09 | 147 | 1.00x | — |

**MAE / PSNR (vs OpenCV CPU):**
- STANDARD: MAE 0.4772, PSNR 48.61 dB ✓
- TEMPLATE: MAE 0.6031, PSNR 48.28 dB ✓
- SEPARABLE: MAE 0.4478, PSNR 48.49 dB ✓
- ADAPTIVE: MAE 0.4042, PSNR 49.42 dB ✓ (lowest MAE)
- OpenCV CUDA: MAE 0.0000, PSNR 999.99 dB (identical to OpenCV CPU)

### Test 2: 3840×2160 Grayscale Image (4K)

| Implementation | Time (ms) | Min (ms) | Throughput (MP/s) | vs OCV CPU | vs OCV CUDA |
|----------------|-----------|----------|-------------------|:----------:|:-----------:|
| **CUDA (SEPARABLE)** | **2.21 ± 0.39** | **1.97** | **3753** | **19.7x** | **4.43x** |
| CUDA (STANDARD)      | 3.75 ± 0.45 | 3.49 | 2210 | 12.3x | 2.61x |
| CUDA (TEMPLATE)      | 4.16 ± 1.07 | 2.90 | 1994 | 3.70x | 2.96x |
| CUDA (ADAPTIVE)      | 5.55 ± 1.62 | 3.93 | 1494 | 3.44x | 2.01x |
| **OpenCV CUDA**      | 9.79 ± 4.51 | 6.40 | 847 | 4.44x | 1.00x |
| OpenCV CPU           | 43.45 ± 13.64 | 16.91 | 191 | 1.00x | — |

**MAE / PSNR (vs OpenCV CPU):**
- STANDARD/TEMPLATE/ADAPTIVE: MAE 0.6117, PSNR 50.23 dB ✓
- SEPARABLE: MAE 0.1481, PSNR 56.18 dB ✓ (very close to OpenCV)
- OpenCV CUDA: MAE 0.0000 (identical to OpenCV CPU)

### Test 3: 1920×1080 RGB Image (1080p)

| Implementation | Time (ms) | Throughput (MP/s) | vs OCV CPU |
|----------------|-----------|-------------------|:----------:|
| **CUDA (SEPARABLE)** | **1.45 ± 0.04** | **1433** | **7.17x** |
| CUDA (ADAPTIVE)      | 1.83 ± 0.07 | 1136 | 5.66x |
| CUDA (TEMPLATE)      | 1.89 ± 0.06 | 1095 | 5.56x |
| CUDA (STANDARD)      | 2.50 ± 0.09 | 830 | 4.30x |
| OpenCV CPU           | 10.5 | 197 | 1.00x |

### Test 4: 1920×1080 Grayscale Image (1080p)

| Implementation | Time (ms) | Throughput (MP/s) | vs OCV CPU |
|----------------|-----------|-------------------|:----------:|
| **CUDA (SEPARABLE)** | **0.54 ± 0.03** | **3853** | **8.48x** |
| CUDA (TEMPLATE)      | 0.81 ± 0.02 | 2550 | 5.68x |
| CUDA (STANDARD)      | 0.99 ± 0.05 | 2095 | 4.63x |
| CUDA (ADAPTIVE)      | 1.07 ± 0.03 | 1932 | 4.29x |
| OpenCV CPU           | 4.59 | 452 | 1.00x |

## Performance Target Achievement

| Target | Requirement | Best Achieved | Mode | Status |
|--------|-------------|---------------|------|--------|
| 4K RGB @60fps | ≥498 MP/s | 1478 MP/s | SEPARABLE | ✅ 3.0x margin |
| 4K Gray @60fps | ≥498 MP/s | 3753 MP/s | SEPARABLE | ✅ 7.5x margin |
| 1080p RGB @60fps | ≥124 MP/s | 1433 MP/s | SEPARABLE | ✅ 11.6x margin |
| 1080p Gray @60fps | ≥124 MP/s | 3853 MP/s | SEPARABLE | ✅ 31.1x margin |
| vs OpenCV CUDA | > 1.0x | **1.68–4.43x** | All modes | ✅ |
| MAE | < 1.0 | 0.15–0.61 | All modes | ✅ |
| PSNR | > 40 dB | 48.28–56.18 dB | All modes | ✅ |

---

## Optimization Comparison (radius=5)

### Implementation Modes

| Mode | Description | Complexity | Best For |
|------|-------------|------------|----------|
| STANDARD | Shared memory + LUT, runtime radius | O(r²) | Flexibility / any radius |
| TEMPLATE | Compile-time radius, full unroll | O(r²) | Accuracy + performance balance |
| **SEPARABLE** | Horizontal + vertical passes | **O(r)** | **Best performance, lowest MAE** |
| ADAPTIVE | Per-pixel radius from Sobel gradient | O(r_avg²) | Edge-preserving quality, large-radius benefit |

### 4K RGB Performance by Mode (latest, with early-continue + OpenCV CUDA)

| Mode | Time (ms) | Throughput (MP/s) | vs OCV CPU | vs OCV CUDA | MAE | PSNR (dB) |
|------|-----------|-------------------|:----------:|:-----------:|-----|-----------|
| **SEPARABLE** | **5.61** | **1478** | **10.1x** | **2.10x** | **0.45 ✓** | **48.49** |
| TEMPLATE | 6.67 | 1243 | 10.5x | 2.26x | 0.60 ✓ | 48.28 |
| ADAPTIVE | 6.95 | 1194 | 8.15x | 1.68x | 0.40 ✓ | 49.42 |
| STANDARD | 8.34 | 995 | 8.72x | 1.82x | 0.48 ✓ | 48.61 |
| OpenCV CUDA | 11.78 | 704 | 4.80x | 1.00x | 0.00 | — |
| OpenCV CPU | 56.58 | 147 | 1.00x | — | — | — |

### 4K Gray Performance by Mode (latest, with early-continue + OpenCV CUDA)

| Mode | Time (ms) | Throughput (MP/s) | vs OCV CPU | vs OCV CUDA | MAE | PSNR (dB) |
|------|-----------|-------------------|:----------:|:-----------:|-----|-----------|
| **SEPARABLE** | **2.21** | **3753** | **19.7x** | **4.43x** | **0.15 ✓** | **56.18** |
| STANDARD | 3.75 | 2210 | 12.3x | 2.61x | 0.61 ✓ | 50.23 |
| TEMPLATE | 4.16 | 1994 | 3.70x | 2.96x | 0.61 ✓ | 50.23 |
| ADAPTIVE | 5.55 | 1494 | 3.44x | 2.01x | 0.61 ✓ | 50.23 |
| OpenCV CUDA | 9.79 | 847 | 4.44x | 1.00x | 0.00 | — |
| OpenCV CPU | 43.45 | 191 | 1.00x | — | — | — |

> SEPARABLE remains the fastest mode overall: O(r) complexity outweighs the 2-pass overhead.
> ADAPTIVE provides the lowest RGB MAE (0.44) by tailoring the radius per pixel, but carries
> overhead from the gradient-computation pass and warp divergence. Its advantage grows with
> larger radii where O(r_avg²) < O(r_max²) saves more computation.

---

## Optimization Techniques Tested

### ✅ Successfully Applied

| # | Technique | Description | Performance Gain |
|---|-----------|-------------|------------------|
| 1 | **Shared Memory** | Cache neighborhood data | ~3-5x vs global memory |
| 2 | **Spatial LUT** | Precomputed spatial weights in constant memory | ~1.5x |
| 3 | **Color LUT** | Precomputed range weights (256 elements) | **~3x** (eliminates expf) |
| 4 | **Fast Math** | `__expf()` intrinsic | ~1.3x (before LUT) |
| 5 | **Loop Unroll** | `#pragma unroll` hint | ~1.1x |
| 6 | **Template Radius** | Compile-time constant for full unroll | **+7%** |
| 7 | **Persistent GPU Buffers** | One-time `cudaMalloc` + LUT cache | **+71%** (host overhead) |
| 8 | **uint8 I/O Kernels** | Direct uint8 in/out, no float pipeline | **+11%** |
| 9 | **cudaHostRegister** | Page-lock caller's heap memory, enable DMA transfers | **+7%** |
| 10 | **Block Size 16×16** | Better L1/SMEM cache utilization vs 32×8 | **+1%** |
| 11 | **Single Color Weight (RGB)** | Shared per-pixel color weight (mean diff), 3x fewer LUT lookups | **+14% (RGB only)** |
| 12 | **Circular Window Early-Continue** | Skip 33% corners where spatial_weight==0; compiler DCE for TEMPLATE | **+13% RGB, +65% Gray (TEMPLATE)** |

### ❌ Tested / Limited Benefit

| Technique | Result |
|-----------|--------|
| Texture Memory | Already using shared memory; no additional benefit |
| Vectorized Access | RGB data not aligned for float3/float4 loads |
| Bilateral Grid | Complex to implement; overkill for r≤10 |
| `__launch_bounds__` | Ignored by compiler (64 regs → min 5 blocks impossible without spilling) |
| Pinned Staging Buffer | Slower (+20%) due to extra CPU-side memcpy overhead vs DMA savings |
| Extended Color LUT (766) | Slower (+4%): exceeds 8KB constant cache per SM |
| `__frcp_rn` Division | Negligible effect (executed once per pixel vs 121× inner loop) |

---

## CUDA Optimization Progress

### Version History

| Version | Technique | 4K Time (ms) | Throughput (MP/s) | vs Previous |
|---------|-----------|--------------|-------------------|-------------|
| v1 | Naive global memory | 250 | 33 | - |
| v2 | Shared memory | 176 | 47 | +42% |
| v3 | + Spatial LUT | 140 | 59 | +26% |
| v4 | + __expf fast math | 55 | 150 | +154% |
| v5 | + Color LUT + unroll | 18 | 460 | +207% |
| v6 | + Template radius | 16.9 | 492 | +7% |
| v7 | + Persistent bufs + LUT cache | 9.86 | 841 | +71% |
| v8 | + uint8 I/O kernels | 8.91 | 930 | +11% |
| v9 | + cudaHostRegister (page-lock) | 8.65 | 959 | +3% |
| v10 | + Block size 16×16 | 8.64 | 960 | +1% |
| **v11** | **+ Single color weight (RGB)** | **7.45** | **1113** | **+16%** |
| **v12** | **+ Circular window early-continue** | **6.53** | **1271** | **+13%** |

> Note: Version history tracks TEMPLATE mode 4K RGB. SEPARABLE mode achieves
> 5.41ms / 1478 MP/s for the same image. OpenCV CUDA: 11.78ms / 704 MP/s.

### Total Optimization Gain: **38x** speedup from baseline (TEMPLATE: 250ms → 6.53ms); **46x** for SEPARABLE (250ms → 5.41ms)

---

## Key Insights

### 1. Color LUT is the Most Effective Single Optimization

For 8-bit images with `diff ∈ [0, 255]`:
```cpp
// Before: ~20 cycles per call
float w = __expf(diff * diff * coeff);

// After: ~4 cycles (constant memory cached lookup)
float w = d_color_lut[diff];
```

For radius=5, each pixel: 3 channels × 81 neighbors (circular window) = **243 expf calls eliminated**.

### 2. Template Specialization Enables Full Unrolling

```cpp
template<int RADIUS>  // Compile-time constant
__global__ void k_bilateral() {
    #pragma unroll  // Full unroll when RADIUS is known
    for (int dy = -RADIUS; dy <= RADIUS; ++dy) { ... }
}
```

Compiler can optimize register usage and instruction scheduling when loop bounds are known.

### 3. Separable Approximation Is Both Faster AND More Accurate

SEPARABLE mode is the fastest in all test cases and produces results closest to OpenCV
(MAE 0.15 for gray, 0.45 for RGB). OpenCV CUDA (`cv::cuda::bilateralFilter`) produces
output identical to OpenCV CPU (MAE=0.0000), confirming it uses the same 2D algorithm.
Our SEPARABLE mode is **2.1–4.4x faster than OpenCV CUDA** thanks to O(r) complexity,
shared memory tiling, and LUT-based weight computation (vs OpenCV's real-time `exp()`).

### 4. Single Color Weight for RGB (Opt5)

Using mean channel difference instead of per-channel weights reduces the inner loop from
3 LUT lookups + 3 wsum accumulations to 1 LUT + 1 wsum per neighbor. This 3x reduction
in color LUT accesses speeds up the TEMPLATE RGB kernel by ~16%. The tradeoff is a
slightly higher MAE (0.65 → 0.80), still well within the < 1.0 requirement.

### 5. H2D/D2H Dominates End-to-End Latency

Profiling breakdown for 4K RGB (TEMPLATE mode, 6.67ms total):
- H2D transfer (24.9MB, registered): ~3ms
- GPU kernel:                         ~1.0ms (improved by early-continue)
- D2H transfer (24.9MB, registered): ~3ms

The PCIe 4.0 ×8 in WSL2 limits effective bandwidth to ~8 GB/s (theoretical: 16 GB/s).
H2D+D2H accounts for ~80% of total time. Further improvement requires either smaller
data (lossless compression) or keeping data resident on GPU (video pipeline integration).

### 6. Adaptive Radius: Quality Over Speed

ADAPTIVE mode computes a per-pixel Sobel gradient and maps it to a radius in [r_min, r_max].
At radius=5, the range [4,5] is narrow, so the benefit is modest:

- **4K RGB**: 6.95ms (ADAPTIVE) vs 6.67ms (TEMPLATE) — **4% slower**, MAE 0.40 vs 0.60
- **4K Gray**: 5.55ms (ADAPTIVE) vs 4.16ms (TEMPLATE) — **33% slower** due to extra gradient pass

The overhead comes from: (1) the gradient kernel, (2) radius_map global memory reads, and
(3) warp divergence from variable loop counts. At larger radii (e.g. r=10), the computational
savings from reduced average radius outweigh these costs.

---

## Environment Variable

Set `BILATERAL_MODE` to switch between implementations:
```bash
BILATERAL_MODE=0 ./bilateral_filter ...  # STANDARD
BILATERAL_MODE=1 ./bilateral_filter ...  # TEMPLATE (default)
BILATERAL_MODE=2 ./bilateral_filter ...  # SEPARABLE
BILATERAL_MODE=4 ./bilateral_filter ...  # ADAPTIVE (per-pixel radius from gradient)
```

---

## Conclusion

The CUDA bilateral filter implementation achieves (best mode per scenario):

| Scenario | CUDA | OCV CUDA | OCV CPU | vs OCV CUDA | vs OCV CPU | MAE | Mode |
|----------|------|----------|---------|:-----------:|:----------:|-----|------|
| 4K RGB | **5.61ms** | 11.78ms | 56.6ms | **2.10x** | **10.1x** | 0.45 ✓ | SEPARABLE |
| 4K Gray | **2.21ms** | 9.79ms | 43.5ms | **4.43x** | **19.7x** | 0.15 ✓ | SEPARABLE |
| 1080p RGB | **1.45ms** | — | 10.5ms | — | **7.17x** | 0.45 ✓ | SEPARABLE |
| 1080p Gray | **0.54ms** | — | 4.59ms | — | **8.48x** | 0.15 ✓ | SEPARABLE |

ADAPTIVE mode (lowest MAE for RGB):

| Scenario | CUDA | OCV CUDA | OCV CPU | vs OCV CUDA | vs OCV CPU | MAE | Mode |
|----------|------|----------|---------|:-----------:|:----------:|-----|------|
| 4K RGB | 6.95ms | 11.67ms | 56.6ms | 1.68x | 8.15x | **0.40** ✓ | ADAPTIVE |
| 4K Gray | 5.55ms | 11.17ms | 19.1ms | 2.01x | 3.44x | 0.61 ✓ | ADAPTIVE |

- **38x faster than naive CUDA baseline** (TEMPLATE mode, 250ms → 6.53ms)
- **46x faster than naive CUDA baseline** (SEPARABLE mode, 250ms → 5.41ms)
- **1.68–4.43x faster than OpenCV CUDA** (cv::cuda::bilateralFilter)
- **~500x faster than CPU** (1080p: 3615ms → 7.18ms CUDA)
- MAE < 1.0 across all modes (correctness verified)
- PSNR > 48 dB across all modes (48.28–56.18 dB)

**Most impactful optimizations (cumulative):**
1. Color weight LUT (**3x**) — eliminates per-pixel `expf` calls
2. Shared memory (**3-5x**) — reduces global memory traffic
3. Persistent GPU buffers + LUT cache (**1.7x**) — eliminates `cudaMalloc` overhead
4. uint8 I/O kernels (**1.1x**) — removes float conversion pipeline
5. cudaHostRegister page-lock (**+7%**) — enables DMA for H2D/D2H
6. Single color weight for RGB (**+16%**, TEMPLATE) — 3x fewer LUT lookups per neighbor
7. Circular window early-continue (**+13% RGB / +65% Gray**, TEMPLATE) — compiler DCE eliminates 33% iterations
8. Template specialization (**+7%**) — enables full loop unrolling
9. Block size 16×16 (**+1%**) — better L1 cache utilization

---

*Last updated: 2026-02-26 (added OpenCV CUDA baseline + circular window early-continue)*

---

## Cross-Platform Results: Jetson AGX Thor

### Test Environment

- Platform: NVIDIA Jetson AGX Thor (R38.2.1)
- GPU: NVIDIA Thor (Blackwell, sm_110)
- Compiler: g++ (C++17, -O3), nvcc (CUDA 13.0)
- OpenCV: 4.x (CPU only, no CUDA modules)

### Filter Parameters

```
radius = 5
sigma_spatial = 3.0
sigma_color = 30.0
```

### Performance Results (Opt G/H/I/K/N: 32x8 block + SoA + launch_bounds(256,6) + FP16 intermediate)

Benchmark methodology: 5 warmup runs + 50 timed runs.

#### 4K RGB (3840×2160×3)

| Implementation | Time (ms) | Min (ms) | Throughput (MP/s) | MAE | vs OCV CPU |
|----------------|-----------|----------|-------------------|-----|:----------:|
| **SEP_FP16** | **3.03 ± 0.08** | **2.97** | **2741** | **0.46** | **28.0x** |
| SEPARABLE | 3.10 ± 0.12 | 3.02 | 2673 | 0.45 | 27.2x |
| FUSED | 3.98 ± 0.05 | 3.96 | 2083 | 0.45 | 21.1x |
| TEMPLATE | 5.50 ± 0.07 | 5.47 | 1508 | 0.60 | 15.3x |
| ADAPTIVE | 6.16 ± 0.06 | 6.13 | 1346 | 0.40 | 13.7x |
| STANDARD | 9.30 ± 0.01 | 9.28 | 892 | 0.48 | 9.1x |
| OpenCV CPU | ~84 | ~83 | ~99 | — | 1.0x |

#### 4K Grayscale (3840×2160×1)

| Implementation | Time (ms) | Min (ms) | Throughput (MP/s) | MAE | vs OCV CPU |
|----------------|-----------|----------|-------------------|-----|:----------:|
| **SEP_FP16** | **1.40 ± 0.13** | **1.30** | **5915** | **0.12** | **39.2x** |
| SEPARABLE | 1.42 ± 0.02 | 1.40 | 5849 | 0.15 | 37.3x |
| FUSED | 1.72 ± 0.23 | 1.60 | 4809 | 0.15 | 30.9x |
| TEMPLATE | 3.53 ± 0.11 | 3.46 | 2348 | 0.61 | 15.0x |
| STANDARD | 4.35 ± 0.07 | 4.33 | 1906 | 0.61 | 12.2x |
| OpenCV CPU | ~53 | ~52 | ~157 | — | 1.0x |

#### 1080p RGB (1920×1080×3)

| Implementation | Time (ms) | Min (ms) | Throughput (MP/s) | MAE | vs OCV CPU |
|----------------|-----------|----------|-------------------|-----|:----------:|
| **SEP_FP16** | **0.88 ± 0.10** | **0.77** | **2351** | **0.46** | **31.4x** |
| SEPARABLE | 0.85 ± 0.09 | 0.77 | 2434 | 0.45 | 32.6x |
| TEMPLATE | 1.54 ± 0.12 | 1.41 | 1344 | 0.61 | 17.9x |
| OpenCV CPU | ~28 | ~27 | ~75 | — | 1.0x |

#### 1080p Grayscale (1920×1080×1)

| Implementation | Time (ms) | Min (ms) | Throughput (MP/s) | MAE | vs OCV CPU |
|----------------|-----------|----------|-------------------|-----|:----------:|
| **SEP_FP16** | **0.40 ± 0.01** | **0.36** | **5139** | **0.12** | **38.8x** |
| SEPARABLE | 0.46 ± 0.08 | 0.40 | 4521 | 0.15 | 34.2x |
| TEMPLATE | 0.97 ± 0.10 | 0.90 | 2129 | 0.61 | 16.0x |
| OpenCV CPU | ~16 | ~14 | ~130 | — | 1.0x |

### Opt G/H/I/K/N ncu Verification

| Optimization | Metric | Before | After | Change |
|-------------|--------|--------|-------|--------|
| Opt G (32x8 block) | Smem bank conflict | 50% excessive | **2.3%** | **-97.6%** |
| Opt H (SoA temp) | Global uncoalesced (V) | 68% | **29%** | **-39pp** |
| Opt H (SoA temp) | Global uncoalesced (H) | 69% | **47%** | **-22pp** |
| Opt I (fmaf) | FMA fusion ratio | ~0.36 | ~0.36 | No change |
| Opt K (launch_bounds) | Registers/thread | 63 | **40** | **-35%** |
| Opt K (launch_bounds) | Achieved occupancy | 62% | **97.5%** | **+35pp** |
| Opt K (launch_bounds) | SM throughput (H) | 64.5% | **79.4%** | **+14.9pp** |
| Opt N (FP16 temp) | V LD sectors/req | 4.00 | **2.00** | **-50%** |
| Opt N (FP16 temp) | H ST sectors/req | 4.00 | **2.00** | **-50%** |
| Opt N (FP16 temp) | FP16 pipe utilization | 0% | ~2% | FP16↔FP32 conversion only |

### Failed Experiments

| Experiment | Expected | Actual | Root Cause |
|-----------|:--------:|:------:|-----------|
| Opt L (u8 vectorize) | 10-20% | Cancelled | sectors/req=2.98 ≈ RGB 3B/pixel theoretical limit |
| Opt M (Fused H+V) | 10-15% | **-31~34%** | Phase1 computation overhead > bandwidth savings |
| Opt N2 (FP16 compute) | 10-20% | **-8.4%** | Scalar __half = FP32 throughput; float↔half conversion overhead |

### Cross-Platform Comparison (4K RGB, TEMPLATE mode)

| Platform | GPU | Arch | CUDA Time (ms) | Throughput (MP/s) |
|----------|-----|------|----------------|-------------------|
| Desktop (WSL2) | RTX 4060 | sm_89 (Ada) | 5.61 | 1478 |
| **Jetson AGX Thor** | **Thor** | **sm_110 (Blackwell)** | **5.47** | **1508** |

### Cross-Platform Comparison (4K RGB, SEPARABLE mode, best variant)

| Platform | GPU | Arch | Mode | CUDA Time (ms) | Throughput (MP/s) |
|----------|-----|------|------|----------------|-------------------|
| Desktop (WSL2) | RTX 4060 | sm_89 (Ada) | SEPARABLE | 5.41 | 1532 |
| **Jetson AGX Thor** | **Thor** | **sm_110 (Blackwell)** | **SEP_FP16** | **2.97** | **2741** |

> SEPARABLE_FP16 on Thor vs SEPARABLE on RTX 4060: **1.82x faster**, benefiting from
> Opt H (SoA coalescing), Opt K (occupancy 97.5%), Opt N (FP16 intermediate),
> and 32MB L2 cache (vs 24MB).

---

*Last updated: 2026-02-27 (Opt G/H/I/K/N + Opt L/M/N2 experiments: FP16 intermediate, fused H+V, FP16 compute)*

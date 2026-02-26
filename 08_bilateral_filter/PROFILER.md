# Performance Profiling Results

## Test Environment

- Platform: Ubuntu Linux (WSL2)
- GPU: NVIDIA GeForce RTX 4060
- Compiler: g++ (C++17, -O3), nvcc (CUDA 13.1)
- OpenCV: 4.x

## Filter Parameters

```
radius = 5
sigma_spatial = 3.0
sigma_color = 30.0
```

## Performance Results

Benchmark methodology: 5 warmup runs + 50 timed runs, reporting mean ± stddev.

### Test 1: 1920×1080 RGB Image (1080p)

| Implementation | Time (ms)   | Throughput (MP/s) | vs OpenCV |
|----------------|-------------|-------------------|-----------|
| **CUDA (SEPARABLE)** | **1.40 ± 0.04** | **1477.92**   | **7.50x** |
| CUDA (TEMPLATE)      | 1.90 ± 0.06 | 1090.62       | 5.71x     |
| CUDA (STANDARD)      | 2.54 ± 0.17 | 815.35        | 4.26x     |
| OpenCV               | 10.8        | 191.47        | 1.00x     |

**MAE (vs OpenCV):**
- STANDARD: 0.6495 ✓
- TEMPLATE: 0.8027 ✓ (single color weight since Opt5)
- SEPARABLE: 0.4496 ✓ (closest to OpenCV)

### Test 2: 3840×2160 RGB Image (4K)

| Implementation | Time (ms)    | Throughput (MP/s) | vs OpenCV |
|----------------|--------------|-------------------|-----------|
| **CUDA (SEPARABLE)** | **5.38 ± 0.18** | **1542.24**   | **5.79x** |
| CUDA (TEMPLATE)      | 7.45 ± 0.22 | 1113.16       | 4.09x     |
| CUDA (STANDARD)      | 9.25 ± 0.21 | 897.17        | 3.29x     |
| OpenCV               | 30.5        | 271.90        | 1.00x     |

**MAE (vs OpenCV):**
- STANDARD: 0.6468 ✓
- TEMPLATE: 0.8003 ✓
- SEPARABLE: 0.4478 ✓

### Test 3: 1920×1080 Grayscale Image (1080p)

| Implementation | Time (ms)   | Throughput (MP/s) | vs OpenCV |
|----------------|-------------|-------------------|-----------|
| **CUDA (SEPARABLE)** | **0.56 ± 0.05** | **3673.84**   | **7.70x** |
| CUDA (TEMPLATE)      | 0.84 ± 0.04 | 2459.84       | 4.95x     |
| CUDA (STANDARD)      | 1.09 ± 0.15 | 1896.94       | 4.12x     |
| OpenCV               | 4.35        | 477.05        | 1.00x     |

**MAE (vs OpenCV):**
- STANDARD/TEMPLATE: 0.6223 ✓
- SEPARABLE: 0.1515 ✓ (very close to OpenCV)

### Test 4: 3840×2160 Grayscale Image (4K)

| Implementation | Time (ms)    | Throughput (MP/s) | vs OpenCV |
|----------------|--------------|-------------------|-----------|
| **CUDA (SEPARABLE)** | **2.06 ± 0.11** | **4022.25**   | **7.99x** |
| CUDA (TEMPLATE)      | 3.00 ± 0.13 | 2763.20       | 5.35x     |
| CUDA (STANDARD)      | 3.89 ± 0.21 | 2133.58       | 4.30x     |
| OpenCV               | 16.5        | 503.29        | 1.00x     |

**MAE (vs OpenCV):**
- STANDARD/TEMPLATE: 0.6217 ✓
- SEPARABLE: 0.1481 ✓ (very close to OpenCV)

## Performance Target Achievement

| Target | Type | Requirement | Best Achieved | Mode | Status |
|--------|------|-------------|---------------|------|--------|
| 1080p RGB @60fps | RGB | ≥124 MP/s | 1478 MP/s | SEPARABLE | ✅ 11.9x margin |
| 4K RGB @60fps | RGB | ≥498 MP/s | 1542 MP/s | SEPARABLE | ✅ 3.1x margin |
| 1080p Gray @60fps | Gray | ≥124 MP/s | 3674 MP/s | SEPARABLE | ✅ 29.6x margin |
| 4K Gray @60fps | Gray | ≥498 MP/s | 4022 MP/s | SEPARABLE | ✅ 8.1x margin |
| MAE | All | < 1.0 | 0.15-0.80 | All modes | ✅ |

---

## Optimization Comparison (radius=5)

### Implementation Modes

| Mode | Description | Complexity | Best For |
|------|-------------|------------|----------|
| STANDARD | Shared memory + LUT, runtime radius | O(r²) | Flexibility / any radius |
| TEMPLATE | Compile-time radius, full unroll | O(r²) | Accuracy + performance balance |
| **SEPARABLE** | Horizontal + vertical passes | **O(r)** | **Best performance, lowest MAE** |

### 4K RGB Performance by Mode (latest)

| Mode | Time (ms) | Throughput (MP/s) | vs STANDARD | MAE |
|------|-----------|-------------------|-------------|-----|
| STANDARD | 9.25 | 897 | 1.00x | 0.65 ✓ |
| TEMPLATE | 7.45 | 1113 | 1.24x | 0.80 ✓ |
| **SEPARABLE** | **5.38** | **1542** | **1.72x** | **0.45 ✓** |

### 4K Gray Performance by Mode (latest)

| Mode | Time (ms) | Throughput (MP/s) | vs STANDARD | MAE |
|------|-----------|-------------------|-------------|-----|
| STANDARD | 3.89 | 2134 | 1.00x | 0.62 ✓ |
| TEMPLATE | 3.00 | 2763 | 1.30x | 0.62 ✓ |
| **SEPARABLE** | **2.06** | **4022** | **1.89x** | **0.15 ✓** |

> SEPARABLE is now the fastest mode overall: O(r) complexity means far fewer operations
> per pixel at r=5. The lower MAE also indicates it approximates OpenCV's algorithm more
> closely, suggesting OpenCV may use a similar separable approach internally.

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

> Note: Version history tracks TEMPLATE mode 4K RGB. SEPARABLE mode (not shown here)
> achieves 5.38ms / 1542 MP/s for the same image after all optimizations.

### Total Optimization Gain: **34x** speedup from baseline (TEMPLATE); **46x** for SEPARABLE

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

For radius=5, each pixel: 3 channels × 121 neighbors = **363 expf calls eliminated**.

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

SEPARABLE mode is now the fastest in all test cases (1.4-8.0x vs OpenCV) AND produces
results closer to OpenCV (MAE 0.15 for gray, 0.45 for RGB), suggesting OpenCV uses a
similar approach internally. The O(r) complexity advantage outweighs the 2-pass overhead
when memory bandwidth is the bottleneck.

### 4. Single Color Weight for RGB (Opt5)

Using mean channel difference instead of per-channel weights reduces the inner loop from
3 LUT lookups + 3 wsum accumulations to 1 LUT + 1 wsum per neighbor. This 3x reduction
in color LUT accesses speeds up the TEMPLATE RGB kernel by ~16%. The tradeoff is a
slightly higher MAE (0.65 → 0.80), still well within the < 1.0 requirement.

### 5. H2D/D2H Dominates End-to-End Latency

Profiling breakdown for 4K RGB (TEMPLATE mode, 7.45ms total):
- H2D transfer (24.9MB, registered): ~3ms
- GPU kernel:                         ~1.5ms
- D2H transfer (24.9MB, registered): ~3ms

The PCIe 4.0 ×8 in WSL2 limits effective bandwidth to ~8 GB/s (theoretical: 16 GB/s).
H2D+D2H accounts for ~80% of total time. Further improvement requires either smaller
data (lossless compression) or keeping data resident on GPU (video pipeline integration).

---

## Environment Variable

Set `BILATERAL_MODE` to switch between implementations:
```bash
BILATERAL_MODE=0 ./bilateral_filter ...  # STANDARD
BILATERAL_MODE=1 ./bilateral_filter ...  # TEMPLATE (default)
BILATERAL_MODE=2 ./bilateral_filter ...  # SEPARABLE
```

---

## Conclusion

The CUDA bilateral filter implementation achieves (best mode per scenario):

| Scenario | CUDA Time | OpenCV | Speedup | MAE | Mode |
|----------|-----------|--------|---------|-----|------|
| 4K RGB | **5.38ms** | 30.5ms | **5.79x** | 0.45 ✓ | SEPARABLE |
| 1080p RGB | **1.40ms** | 10.8ms | **7.50x** | 0.45 ✓ | SEPARABLE |
| 4K Gray | **2.06ms** | 16.5ms | **7.99x** | 0.15 ✓ | SEPARABLE |
| 1080p Gray | **0.56ms** | 4.35ms | **7.70x** | 0.15 ✓ | SEPARABLE |

- **34x faster than naive CUDA baseline** (TEMPLATE mode, 250ms → 7.45ms)
- **46x faster than naive CUDA baseline** (SEPARABLE mode, 250ms → 5.38ms)
- **~800x faster than naive CPU**
- MAE < 1.0 across all modes (correctness verified)

**Most impactful optimizations (cumulative):**
1. Color weight LUT (**3x**) — eliminates per-pixel `expf` calls
2. Shared memory (**3-5x**) — reduces global memory traffic
3. Persistent GPU buffers + LUT cache (**1.7x**) — eliminates `cudaMalloc` overhead
4. uint8 I/O kernels (**1.1x**) — removes float conversion pipeline
5. cudaHostRegister page-lock (**+7%**) — enables DMA for H2D/D2H
6. Single color weight for RGB (**+16%**, TEMPLATE) — 3x fewer LUT lookups per neighbor
7. Template specialization (**+7%**) — enables full loop unrolling
8. Block size 16×16 (**+1%**) — better L1 cache utilization

---

*Last updated: 2026-02-26*

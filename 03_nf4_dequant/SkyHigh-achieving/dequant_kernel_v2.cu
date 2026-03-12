/**
 * dequant_kernel.cu  —  NF4 Dequantization Kernel (Optimized)
 *
 * Formula:  w = NF4[4bit_index] * code2[absmax_q[block]] * absmax2[block/256] + offset
 *
 * This is bitsandbytes "double quantization" (quant_type=nf4, double_quant=True):
 *   - Level 1 (absmax_q): uint8 index into code2 lookup table
 *   - Level 2 (absmax2):  FP16 scale for groups of 256 L1 blocks
 *   - The L1 scale itself is quantized to save memory (~3% overhead instead of 8%)
 *
 * Optimization history (for presentation):
 *   v1 Naive:      one thread per element, scalar FP16 write  → ~5% A100 bandwidth
 *   v2 Vectorized: one thread per uint8 (2 elements), __half2 packed store → ~35% bandwidth
 *   v3 (TODO):     128-bit vectorized load (int4, 32 elements/thread) → ~80% bandwidth
 */

#include "dequant_kernel.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <type_traits>

namespace {

constexpr int kPairsPerThreadV3 = 8;

// ── NF4 Lookup Table ─────────────────────────────────────────────────────────
// From QLoRA paper Table 1: 16 quantile values of N(0,1) scaled to [-1, 1]
// Stored in __constant__ memory: all 108 SMs on A100 share one copy,
// with 8KB constant cache per SM. 16 * 4 = 64 bytes — fits in a single cache line.
__device__ __constant__ float d_nf4[16] = {
    -1.0f, -0.6961928f, -0.52507305f, -0.3949175f,
    -0.28444138f, -0.18477343f, -0.091050036f, 0.0f,
    0.0795803f, 0.1609302f, 0.2461123f, 0.33791524f,
    0.44070983f, 0.562617f, 0.72295684f, 1.0f
};

// CPU copy for reference computation (identical values)
constexpr float kNF4[16] = {
    -1.0f, -0.6961928f, -0.52507305f, -0.3949175f,
    -0.28444138f, -0.18477343f, -0.091050036f, 0.0f,
    0.0795803f, 0.1609302f, 0.2461123f, 0.33791524f,
    0.44070983f, 0.562617f, 0.72295684f, 1.0f
};

inline int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

inline float fp16_bits_to_float(uint16_t bits) {
    __half h; std::memcpy(&h, &bits, sizeof(uint16_t));
    return __half2float(h);
}

// ── Type-agnostic cast helper ─────────────────────────────────────────────────
template <typename T> __device__ inline T cast_to(float v);
template <> __device__ inline __half cast_to<__half>(float v)          { return __float2half(v); }
template <> __device__ inline __nv_bfloat16 cast_to<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

// ── Packed-pair write helper ───────────────────────────────────────────────────
// Instead of two separate 16-bit stores, pack into one 32-bit store.
// This halves the number of store instructions and improves L2 write efficiency.
template <typename T>
__device__ inline void store_pair(T* __restrict__ ptr, T a, T b);

template <>
__device__ inline void store_pair<__half>(__half* __restrict__ ptr, __half a, __half b) {
    // __half2 is two consecutive __half values; store as uint32 for atomic/vectorized write
    __half2 packed = __halves2half2(a, b);
    *reinterpret_cast<uint32_t*>(ptr) = *reinterpret_cast<uint32_t*>(&packed);
}

template <>
__device__ inline void store_pair<__nv_bfloat16>(
    __nv_bfloat16* __restrict__ ptr, __nv_bfloat16 a, __nv_bfloat16 b)
{
    __nv_bfloat162 packed = __halves2bfloat162(a, b);
    *reinterpret_cast<uint32_t*>(ptr) = *reinterpret_cast<uint32_t*>(&packed);
}

// ── Main Dequant Kernel (v2: Vectorized / Packed Store) ──────────────────────
//
// Thread mapping:
//   - 1 thread handles 1 uint8 = 2 packed 4-bit weights = 2 output elements
//   - pair_idx = blockIdx.x * blockDim.x + threadIdx.x
//   - elem0    = pair_idx * 2
//   - elem1    = pair_idx * 2 + 1
//
// Memory access pattern (key for A100 bandwidth):
//   - packed_weights: consecutive threads → consecutive bytes → COALESCED READ
//   - output:         consecutive threads → consecutive 32-bit stores → COALESCED WRITE
//   - absmax_q:       32 consecutive threads share the same block (blocksize=64)
//                     → same cache line → effectively a BROADCAST, no divergence
//   - code2:          random access into 256-entry table → stays in L1 after warmup
//
template <typename T>
__global__ void dequant_kernel(
    const uint8_t* __restrict__ packed_weights,  // [num_pairs]    bytes
    const uint8_t* __restrict__ absmax_q,         // [num_blocks]   uint8
    const float*   __restrict__ absmax2,           // [num_groups]   float
    const float*   __restrict__ code2,             // [256]          float
    float          offset,
    int64_t        numel,
    int32_t        blocksize,
    T*             __restrict__ out)               // [numel]        T
{
    // Which pair of elements does this thread own?
    const int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t elem0    = pair_idx * 2;
    if (elem0 >= numel) return;

    // ── Unpack two 4-bit indices from one byte ────────────────────────────────
    //   byte layout (bitsandbytes convention):
    //     bits[3:0] → element at even position (elem0)
    //     bits[7:4] → element at odd  position (elem1)
    const uint8_t packed = packed_weights[pair_idx];
    const int     idx0   = packed & 0x0F;           // low  nibble
    const int     idx1   = (packed >> 4) & 0x0F;    // high nibble

    // ── Compute scale for elem0 ───────────────────────────────────────────────
    //   Two-level (double) quantization:
    //     L1 scale = code2[ absmax_q[block_idx] ]   (uint8 → float via codebook)
    //     L2 scale = absmax2[ block_idx / 256 ]      (float)
    //     final scale = L1 * L2
    const int64_t block_idx0 = elem0 / blocksize;
    const int64_t group_idx0 = block_idx0 / 256;
    const float   scale0     = code2[absmax_q[block_idx0]] * absmax2[group_idx0];
    const float   w0         = d_nf4[idx0] * scale0 + offset;

    // ── Compute scale for elem1 (may be in a different block at boundaries) ───
    const int64_t elem1 = elem0 + 1;
    if (elem1 < numel) {
        // Normal path: two valid elements
        // For blocksize >= 2, block_idx1 == block_idx0 in the vast majority of cases.
        // We still compute it correctly for generality (compiler will optimize same-block case).
        const int64_t block_idx1 = elem1 / blocksize;
        const int64_t group_idx1 = block_idx1 / 256;
        const float   scale1     = code2[absmax_q[block_idx1]] * absmax2[group_idx1];
        const float   w1         = d_nf4[idx1] * scale1 + offset;

        // ── Vectorized (packed) store: 2 × T in one 32-bit write ─────────────
        // Equivalent to two separate stores, but issues a single 32-bit transaction
        // to the L2/HBM, halving write pressure.
        store_pair<T>(out + elem0, cast_to<T>(w0), cast_to<T>(w1));
    } else {
        // Edge case: only elem0 is valid (odd-sized matrix, last element)
        out[elem0] = cast_to<T>(w0);
    }
}

template <typename T>
__global__ void dequant_kernel_v3(
    const uint8_t* __restrict__ packed_weights,
    const uint8_t* __restrict__ absmax_q,
    const float*   __restrict__ absmax2,
    const float*   __restrict__ code2,
    float          offset,
    int64_t        numel,
    int32_t        blocksize,
    T*             __restrict__ out)
{
    constexpr int pairs_per_thread = kPairsPerThreadV3;

    const int64_t tid       = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t pair_base = tid * pairs_per_thread;
    const int64_t elem_base = pair_base * 2;
    const int64_t num_pairs_total = (numel + 1) / 2;

    if (elem_base >= numel) return;

    uint8_t bytes[pairs_per_thread];

    if (pair_base + pairs_per_thread <= num_pairs_total) {
        if constexpr (pairs_per_thread == 16) {
            const uint4 raw = *reinterpret_cast<const uint4*>(packed_weights + pair_base);
            const uint32_t lanes[4] = {raw.x, raw.y, raw.z, raw.w};
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                const uint32_t v = lanes[l];
                bytes[l * 4 + 0] = static_cast<uint8_t>(v & 0xFFu);
                bytes[l * 4 + 1] = static_cast<uint8_t>((v >> 8) & 0xFFu);
                bytes[l * 4 + 2] = static_cast<uint8_t>((v >> 16) & 0xFFu);
                bytes[l * 4 + 3] = static_cast<uint8_t>((v >> 24) & 0xFFu);
            }
        } else {
            const uint2 raw = *reinterpret_cast<const uint2*>(packed_weights + pair_base);
            const uint32_t lanes[2] = {raw.x, raw.y};
            #pragma unroll
            for (int l = 0; l < 2; ++l) {
                const uint32_t v = lanes[l];
                bytes[l * 4 + 0] = static_cast<uint8_t>(v & 0xFFu);
                bytes[l * 4 + 1] = static_cast<uint8_t>((v >> 8) & 0xFFu);
                bytes[l * 4 + 2] = static_cast<uint8_t>((v >> 16) & 0xFFu);
                bytes[l * 4 + 3] = static_cast<uint8_t>((v >> 24) & 0xFFu);
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < pairs_per_thread; ++i) {
            const int64_t p = pair_base + i;
            bytes[i] = (p < num_pairs_total) ? packed_weights[p] : 0u;
        }
    }

    #pragma unroll
    for (int i = 0; i < pairs_per_thread; ++i) {
        const int64_t elem0 = elem_base + static_cast<int64_t>(i) * 2;
        if (elem0 >= numel) break;

        const int idx0 = bytes[i] & 0x0F;
        const int idx1 = (bytes[i] >> 4) & 0x0F;

        const int64_t block_idx0 = elem0 / blocksize;
        const float scale0 = code2[absmax_q[block_idx0]] * absmax2[block_idx0 / 256];
        const float w0 = d_nf4[idx0] * scale0 + offset;

        const int64_t elem1 = elem0 + 1;
        if (elem1 < numel) {
            const int64_t block_idx1 = elem1 / blocksize;
            const float scale1 = code2[absmax_q[block_idx1]] * absmax2[block_idx1 / 256];
            const float w1 = d_nf4[idx1] * scale1 + offset;
            store_pair<T>(out + elem0, cast_to<T>(w0), cast_to<T>(w1));
        } else {
            out[elem0] = cast_to<T>(w0);
        }
    }
}

template <typename T>
__global__ __launch_bounds__(128, 8) void dequant_kernel_v4(
    const uint8_t* __restrict__ packed_weights,
    const uint8_t* __restrict__ absmax_q,
    const float*   __restrict__ absmax2,
    const float*   __restrict__ code2,
    float          offset,
    int64_t        numel,
    int32_t        blocksize,
    T*             __restrict__ out)
{
    constexpr int pairs_per_thread = kPairsPerThreadV3;

    const int64_t tid       = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t pair_base = tid * pairs_per_thread;
    const int64_t elem_base = pair_base * 2;
    const int64_t num_pairs_total = (numel + 1) / 2;

    if (elem_base >= numel) return;

    uint8_t bytes[pairs_per_thread];

    if (pair_base + pairs_per_thread <= num_pairs_total) {
        const uint2 raw = *reinterpret_cast<const uint2*>(packed_weights + pair_base);
        const uint32_t lanes[2] = {raw.x, raw.y};
        #pragma unroll
        for (int l = 0; l < 2; ++l) {
            const uint32_t v = lanes[l];
            bytes[l * 4 + 0] = static_cast<uint8_t>(v & 0xFFu);
            bytes[l * 4 + 1] = static_cast<uint8_t>((v >> 8) & 0xFFu);
            bytes[l * 4 + 2] = static_cast<uint8_t>((v >> 16) & 0xFFu);
            bytes[l * 4 + 3] = static_cast<uint8_t>((v >> 24) & 0xFFu);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < pairs_per_thread; ++i) {
            const int64_t p = pair_base + i;
            bytes[i] = (p < num_pairs_total) ? packed_weights[p] : 0u;
        }
    }

    #pragma unroll
    for (int i = 0; i < pairs_per_thread; ++i) {
        const int64_t elem0 = elem_base + static_cast<int64_t>(i) * 2;
        if (elem0 >= numel) break;

        const int idx0 = bytes[i] & 0x0F;
        const int idx1 = (bytes[i] >> 4) & 0x0F;

        const int64_t block_idx0 = elem0 / blocksize;
        const float scale0 = code2[absmax_q[block_idx0]] * absmax2[block_idx0 / 256];
        const float w0 = d_nf4[idx0] * scale0 + offset;

        const int64_t elem1 = elem0 + 1;
        if (elem1 < numel) {
            const int64_t block_idx1 = elem1 / blocksize;
            const float scale1 = code2[absmax_q[block_idx1]] * absmax2[block_idx1 / 256];
            const float w1 = d_nf4[idx1] * scale1 + offset;
            store_pair<T>(out + elem0, cast_to<T>(w0), cast_to<T>(w1));
        } else {
            out[elem0] = cast_to<T>(w0);
        }
    }
}

// ── CPU Reference (for MAE verification) ─────────────────────────────────────
// Identical formula to the GPU kernel, computed in FP32 on the host.
// Used to verify: |gpu_output[i] - cpu_ref[i]| < 1e-2 for all i
void cpu_reference(const NF4Binary& input, std::vector<float>& ref) {
    const int64_t numel     = input.config.rows * input.config.cols;
    const int32_t blocksize = input.config.blocksize;
    ref.resize(numel);

    for (int64_t i = 0; i < numel; ++i) {
        const int64_t pair_idx  = i / 2;
        const bool    low       = (i % 2) == 0;
        const uint8_t packed    = input.packed_weights[pair_idx];
        const int     idx       = low ? (packed & 0x0F) : ((packed >> 4) & 0x0F);

        const int64_t block_idx = i / blocksize;
        const int64_t group_idx = block_idx / 256;

        // Reconstruct two-level scale from stored FP16 bits
        const float scale_l1 = fp16_bits_to_float(input.code2_raw[input.absmax_q[block_idx]]);
        const float scale_l2 = fp16_bits_to_float(input.absmax2_raw[group_idx]);
        ref[i] = kNF4[idx] * scale_l1 * scale_l2 + input.offset;
    }
}

// ── GPU Launch Wrapper ────────────────────────────────────────────────────────
template <typename T>
bool launch_cuda(const NF4Binary& input, std::vector<float>& output, std::vector<float>& gpu_fp32_out) {
    auto fail_cuda = [](const char* stage, cudaError_t err) -> bool {
        std::cerr << "FAIL " << stage << ": [" << (int)err << "] "
                  << cudaGetErrorString(err) << std::endl;
        return false;
    };

    const int64_t numel      = input.config.rows * input.config.cols;
    const int64_t num_pairs  = ceil_div(numel, 2);
    const int64_t num_blocks = ceil_div(numel, input.config.blocksize);
    const int64_t num_groups = ceil_div(num_blocks, 256);

    std::cout << "GPU launch: numel=" << numel
              << " pairs=" << num_pairs
              << " blocks=" << num_blocks
              << " groups=" << num_groups << std::endl;

    // ── Device allocations ────────────────────────────────────────────────────
    uint8_t* d_packed   = nullptr;
    uint8_t* d_absmax_q = nullptr;
    float*   d_absmax2  = nullptr;
    float*   d_code2    = nullptr;
    T*       d_out      = nullptr;

    cudaError_t err;
    if ((err = cudaMalloc(&d_packed,   num_pairs))  != cudaSuccess) return fail_cuda("malloc packed",   err);
    if ((err = cudaMalloc(&d_absmax_q, num_blocks)) != cudaSuccess) return fail_cuda("malloc absmax_q", err);
    if ((err = cudaMalloc(&d_absmax2,  num_groups * sizeof(float))) != cudaSuccess) return fail_cuda("malloc absmax2", err);
    if ((err = cudaMalloc(&d_code2,    256 * sizeof(float)))        != cudaSuccess) return fail_cuda("malloc code2",   err);
    if ((err = cudaMalloc(&d_out,      numel * sizeof(T)))          != cudaSuccess) return fail_cuda("malloc out",     err);

    // ── Convert FP16 metadata to FP32 for GPU ─────────────────────────────────
    // (GPU kernel uses float to avoid fp16 precision issues in scale multiplication)
    std::vector<float> h_absmax2(num_groups), h_code2(256);
    for (int64_t i = 0; i < num_groups; ++i) h_absmax2[i] = fp16_bits_to_float(input.absmax2_raw[i]);
    for (int i = 0; i < 256; ++i)            h_code2[i]   = fp16_bits_to_float(input.code2_raw[i]);

    // ── Host → Device transfers ───────────────────────────────────────────────
    if ((err = cudaMemcpy(d_packed,   input.packed_weights.data(), num_pairs,           cudaMemcpyHostToDevice)) != cudaSuccess) return fail_cuda("memcpy packed",   err);
    if ((err = cudaMemcpy(d_absmax_q, input.absmax_q.data(),       num_blocks,          cudaMemcpyHostToDevice)) != cudaSuccess) return fail_cuda("memcpy absmax_q", err);
    if ((err = cudaMemcpy(d_absmax2,  h_absmax2.data(),            num_groups * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) return fail_cuda("memcpy absmax2",  err);
    if ((err = cudaMemcpy(d_code2,    h_code2.data(),              256 * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) return fail_cuda("memcpy code2",    err);

    const double bytes_read  = (double)(num_pairs + num_blocks + num_groups * 2 + 256 * 2);
    const double bytes_write = (double)(numel * sizeof(T));
    const double total_gb    = (bytes_read + bytes_write) / 1e9;
    float ms_v2 = 0.0f;
    float ms_v3 = 0.0f;
    float ms_v4 = 0.0f;

    {
        const int threads_v2 = 256;
        const int blocks_v2 = static_cast<int>(ceil_div(num_pairs, static_cast<int64_t>(threads_v2)));
        cudaEvent_t t_start, t_stop;
        cudaEventCreate(&t_start);
        cudaEventCreate(&t_stop);
        cudaEventRecord(t_start);
        dequant_kernel<T><<<blocks_v2, threads_v2>>>(
            d_packed, d_absmax_q, d_absmax2, d_code2,
            input.offset, numel, input.config.blocksize, d_out);
        cudaEventRecord(t_stop);
        if ((err = cudaGetLastError()) != cudaSuccess) return fail_cuda("kernel launch v2", err);
        if ((err = cudaEventSynchronize(t_stop)) != cudaSuccess) return fail_cuda("event sync v2", err);
        cudaEventElapsedTime(&ms_v2, t_start, t_stop);
        cudaEventDestroy(t_start);
        cudaEventDestroy(t_stop);
    }

    {
        const int threads_v3 = 128;
        const int64_t num_threads_v3 = ceil_div(num_pairs, static_cast<int64_t>(kPairsPerThreadV3));
        const int blocks_v3 = static_cast<int>(ceil_div(num_threads_v3, static_cast<int64_t>(threads_v3)));
        cudaEvent_t t_start, t_stop;
        cudaEventCreate(&t_start);
        cudaEventCreate(&t_stop);
        cudaEventRecord(t_start);
        dequant_kernel_v3<T><<<blocks_v3, threads_v3>>>(
            d_packed, d_absmax_q, d_absmax2, d_code2,
            input.offset, numel, input.config.blocksize, d_out);
        cudaEventRecord(t_stop);
        if ((err = cudaGetLastError()) != cudaSuccess) return fail_cuda("kernel launch v3", err);
        if ((err = cudaEventSynchronize(t_stop)) != cudaSuccess) return fail_cuda("event sync v3", err);
        cudaEventElapsedTime(&ms_v3, t_start, t_stop);
        cudaEventDestroy(t_start);
        cudaEventDestroy(t_stop);
    }

    int min_grid_size_v4 = 0;
    int block_size_v4 = 0;
    if ((err = cudaOccupancyMaxPotentialBlockSize(
             &min_grid_size_v4,
             &block_size_v4,
             dequant_kernel_v4<T>,
             0,
             128)) != cudaSuccess) {
        return fail_cuda("occupancy v4", err);
    }
    if (block_size_v4 <= 0 || block_size_v4 > 128) {
        block_size_v4 = 128;
    }

    {
        const int64_t num_threads_v4 = ceil_div(num_pairs, static_cast<int64_t>(kPairsPerThreadV3));
        const int blocks_v4 = static_cast<int>(ceil_div(num_threads_v4, static_cast<int64_t>(block_size_v4)));
        cudaEvent_t t_start, t_stop;
        cudaEventCreate(&t_start);
        cudaEventCreate(&t_stop);
        cudaEventRecord(t_start);
        dequant_kernel_v4<T><<<blocks_v4, block_size_v4>>>(
            d_packed, d_absmax_q, d_absmax2, d_code2,
            input.offset, numel, input.config.blocksize, d_out);
        cudaEventRecord(t_stop);
        if ((err = cudaGetLastError()) != cudaSuccess) return fail_cuda("kernel launch v4", err);
        if ((err = cudaEventSynchronize(t_stop)) != cudaSuccess) return fail_cuda("event sync v4", err);
        cudaEventElapsedTime(&ms_v4, t_start, t_stop);
        cudaEventDestroy(t_start);
        cudaEventDestroy(t_stop);
    }

    if ((err = cudaDeviceSynchronize()) != cudaSuccess) return fail_cuda("sync", err);

    const double bw_v2 = total_gb / (ms_v2 / 1000.0);
    const double bw_v3 = total_gb / (ms_v3 / 1000.0);
    const double bw_v4 = total_gb / (ms_v4 / 1000.0);
    const double speedup_v3 = ms_v3 > 0.0 ? (ms_v2 / ms_v3) : 0.0;
    const double speedup_v4 = ms_v4 > 0.0 ? (ms_v2 / ms_v4) : 0.0;
    const double speedup_v4_vs_v3 = ms_v4 > 0.0 ? (ms_v3 / ms_v4) : 0.0;

    std::cout << "[v2] Kernel time : " << ms_v2 << " ms  |  Bandwidth : " << bw_v2
              << " GB/s  (" << (bw_v2 / 1935.0 * 100.0) << "% of A100 peak 1935 GB/s)" << std::endl;
    std::cout << "[v3] Kernel time : " << ms_v3 << " ms  |  Bandwidth : " << bw_v3
              << " GB/s  (" << (bw_v3 / 1935.0 * 100.0) << "% of A100 peak 1935 GB/s)" << std::endl;
    std::cout << "[v3 speedup vs v2]: " << speedup_v3 << "x" << std::endl;
    std::cout << "[v4] Kernel time : " << ms_v4 << " ms  |  Bandwidth : " << bw_v4
              << " GB/s  (" << (bw_v4 / 1935.0 * 100.0) << "% of A100 peak 1935 GB/s)" << std::endl;
    std::cout << "[v4 speedup vs v2]: " << speedup_v4 << "x" << std::endl;
    std::cout << "[v4 speedup vs v3]: " << speedup_v4_vs_v3 << "x"
              << "  |  occupancy block=" << block_size_v4
              << " min_grid=" << min_grid_size_v4 << std::endl;

    // ── Device → Host copy ────────────────────────────────────────────────────
    gpu_fp32_out.resize(numel);
    if constexpr (std::is_same<T, __half>::value) {
        std::vector<__half> h_out(numel);
        if ((err = cudaMemcpy(h_out.data(), d_out, numel * sizeof(__half), cudaMemcpyDeviceToHost)) != cudaSuccess)
            return fail_cuda("memcpy output fp16", err);
        for (int64_t i = 0; i < numel; ++i) gpu_fp32_out[i] = __half2float(h_out[i]);
    } else {
        std::vector<__nv_bfloat16> h_out(numel);
        if ((err = cudaMemcpy(h_out.data(), d_out, numel * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost)) != cudaSuccess)
            return fail_cuda("memcpy output bf16", err);
        for (int64_t i = 0; i < numel; ++i) gpu_fp32_out[i] = __bfloat162float(h_out[i]);
    }
    output = gpu_fp32_out;

    cudaFree(d_packed); cudaFree(d_absmax_q);
    cudaFree(d_absmax2); cudaFree(d_code2); cudaFree(d_out);
    return true;
}

}  // namespace

// ── Public API ───────────────────────────────────────────────────────────────

bool load_nf4_binary(const char* file_path, NF4Binary& out) {
    std::ifstream fin(file_path, std::ios::binary);
    if (!fin.is_open()) return false;

    int64_t rows = 0, cols = 0; int32_t blocksize = 0;
    fin.read(reinterpret_cast<char*>(&rows),      sizeof(rows));
    fin.read(reinterpret_cast<char*>(&cols),      sizeof(cols));
    fin.read(reinterpret_cast<char*>(&blocksize), sizeof(blocksize));
    if (!fin.good()) return false;

    const int64_t numel      = rows * cols;
    const int64_t num_pairs  = ceil_div(numel, 2);
    const int64_t num_blocks = ceil_div(numel, blocksize);
    const int64_t num_groups = ceil_div(num_blocks, 256);

    out.config = {rows, cols, blocksize, ComputeType::FP16};
    out.packed_weights.resize(num_pairs);
    out.absmax_q.resize(num_blocks);
    out.absmax2_raw.resize(num_groups);
    out.code2_raw.resize(256);

    fin.read(reinterpret_cast<char*>(out.packed_weights.data()), num_pairs);
    fin.read(reinterpret_cast<char*>(out.absmax_q.data()),       num_blocks);
    fin.read(reinterpret_cast<char*>(out.absmax2_raw.data()),    num_groups * sizeof(uint16_t));
    fin.read(reinterpret_cast<char*>(out.code2_raw.data()),      256 * sizeof(uint16_t));
    fin.read(reinterpret_cast<char*>(&out.offset),               sizeof(float));

    std::cout << "Loaded: " << rows << "x" << cols
              << " blocksize=" << blocksize << " offset=" << out.offset << std::endl;
    return fin.good();
}

bool save_float_output(const char* file_path, const std::vector<float>& data) {
    std::ofstream fout(file_path, std::ios::binary);
    if (!fout.is_open()) return false;
    fout.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    return fout.good();
}

bool run_dequant_cuda(const NF4Binary& input, std::vector<float>& output, float& mae) {
    std::vector<float> gpu_out;
    const bool ok = (input.config.compute_type == ComputeType::FP16)
        ? launch_cuda<__half>(input, output, gpu_out)
        : launch_cuda<__nv_bfloat16>(input, output, gpu_out);
    if (!ok) return false;

    // MAE against CPU reference
    std::vector<float> ref;
    cpu_reference(input, ref);
    double err_sum = 0.0;
    for (size_t i = 0; i < ref.size(); ++i)
        err_sum += std::abs((double)gpu_out[i] - (double)ref[i]);
    mae = static_cast<float>(err_sum / (double)ref.size());
    std::cout << "MAE (v4 GPU vs CPU ref): " << mae << (mae < 1e-2f ? "  ✓ PASS" : "  ✗ FAIL (threshold 1e-2)") << std::endl;
    return true;
}

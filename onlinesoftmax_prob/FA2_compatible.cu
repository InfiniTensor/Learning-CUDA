/*
 * A correctness-first adaptation of src/FA2.cu for the llaisys attention
 * interface. The original work and this adaptation are covered by the MIT
 * license in the repository root.
 *
 * Supported interface:
 *   Q/O: [batch_size, q_len, q_heads, head_dim]
 *   K/V: [batch_size, kv_len, kv_heads, head_dim]
 *
 * P0 change map (search "[P0]" in this file to locate each adjustment):
 * The tiled kernel below keeps the same FA2 online-softmax update order as
 * src/FA2.cu and adds:
 *   - token-major addressing used by llaisys;
 *   - GQA/MHA head mapping;
 *   - PyTorch-compatible upper-left causal masking;
 *   - q_len / kv_len tail handling;
 *   - arbitrary positive head_dim values;
 *   - float/half I/O, plus bfloat16 when the backend exposes cuda_bf16.h;
 *   - FP32 score and output accumulation.
 */

#if __has_include(<cuda_bf16.h>)
#include <cuda_bf16.h>
#define FA2_COMPATIBLE_HAS_BFLOAT16 1
#endif
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../include/FA2_compatible.cuh"
#include "../include/platform_warp.h"

#include <cmath>
#if defined(FA2_TRACE_DEVICE)
#include <cstdio>
#endif
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace fa2_compatible
{
    namespace
    {

        // HeadDim is a compile-time capacity bucket. params.head_dim remains
        // the actual runtime width, so non-power-of-two tails are still valid.
        template <int HeadDim>
        struct Fa2Traits
        {
            static constexpr int kWarpSize = platform::WARP_SIZE;
            static constexpr int kWarpsPerCta = 4;
            static constexpr int kRowsPerWarp = 4;
            static constexpr int kRowsPerCta =
                kWarpsPerCta * kRowsPerWarp;
            static constexpr int kThreads = kWarpSize * kWarpsPerCta;
            static constexpr int kBc = 16;
            static constexpr int kDPerLane =
                (HeadDim + kWarpSize - 1) / kWarpSize;
            static constexpr int kKvTileElements = kBc * HeadDim;
        };

        using DefaultTraits = Fa2Traits<64>;
        static_assert(DefaultTraits::kThreads % DefaultTraits::kWarpSize == 0,
                      "Full-mask reductions require complete warps");
        static_assert(DefaultTraits::kThreads <= 1024,
                      "Attention CTA exceeds the supported thread count");
        static_assert(DefaultTraits::kBc <= DefaultTraits::kWarpSize,
                      "Each tile key is owned by one physical warp lane");

        template <typename T>
        __device__ __forceinline__ float toFloat(T value);

        template <>
        __device__ __forceinline__ float toFloat<float>(float value)
        {
            return value;
        }

        template <>
        __device__ __forceinline__ float toFloat<__half>(__half value)
        {
            return __half2float(value);
        }

#if defined(FA2_COMPATIBLE_HAS_BFLOAT16)
        template <>
        __device__ __forceinline__ float toFloat<__nv_bfloat16>(
            __nv_bfloat16 value)
        {
            return __bfloat162float(value);
        }
#endif

        template <typename T>
        __device__ __forceinline__ T fromFloat(float value);

        template <>
        __device__ __forceinline__ float fromFloat<float>(float value)
        {
            return value;
        }

        template <>
        __device__ __forceinline__ __half fromFloat<__half>(float value)
        {
            return __float2half(value);
        }

#if defined(FA2_COMPATIBLE_HAS_BFLOAT16)
        template <>
        __device__ __forceinline__ __nv_bfloat16
        fromFloat<__nv_bfloat16>(float value)
        {
            return __float2bfloat16(value);
        }
#endif

        template <typename T>
        struct Fa2Params
        {
            T *output;
            const T *query;
            const T *key;
            const T *value;

            // [P0] 原基线缺少 batch/causal；这里集中保存完整接口元数据，避免
            // kernel 内再假设 batch=1 或 causal 恒开启。
            int batch_size;
            int q_len;
            int kv_len;
            int q_heads;
            int kv_heads;
            int head_dim;
            bool is_causal;
            float scale;
        };

        template <typename T, int HeadDim>
        __global__ void fa2CompatibleKernel(Fa2Params<T> params)
        {
            using Traits = Fa2Traits<HeadDim>;

            const int tid = threadIdx.x;
            const int warp_id = tid / Traits::kWarpSize;
            const int warp_lane = tid % Traits::kWarpSize;
            // [P0 CTA mapping] Four warps share one KV tile. Each warp owns
            // four query rows, so one CTA processes 16 consecutive rows.
            const int batch = blockIdx.z;
            const int q_head = blockIdx.y;
            const int cta_q_base = blockIdx.x * Traits::kRowsPerCta;

            // [P0 GQA] q_head 按组映射到共享的 kv_head。
            const int group_size = params.q_heads / params.kv_heads;
            const int kv_head = q_head / group_size;

            // K/V are the only CTA-wide tiles. Q and the online-softmax
            // accumulators stay in per-thread local arrays, matching FA2.cu.
            __shared__ float smem[2 * Traits::kKvTileElements];
            float *k_tile = smem;
            float *v_tile = smem + Traits::kKvTileElements;

            float m_reg[Traits::kRowsPerWarp];
            float l_reg[Traits::kRowsPerWarp];
            float acc_reg[Traits::kRowsPerWarp][Traits::kDPerLane];
            float q_tile[Traits::kRowsPerWarp][Traits::kDPerLane];
#pragma unroll
            for (int row_slot = 0; row_slot < Traits::kRowsPerWarp;
                 ++row_slot)
            {
                m_reg[row_slot] = -INFINITY;
                l_reg[row_slot] = 0.0f;
#pragma unroll
                for (int d_slot = 0; d_slot < Traits::kDPerLane; ++d_slot)
                {
                    const int d = warp_lane + d_slot * Traits::kWarpSize;
                    const int row_in_cta =
                        row_slot * Traits::kWarpsPerCta + warp_id;
                    const int q_row = cta_q_base + row_in_cta;
                    const bool valid_q = q_row < params.q_len;
                    if (valid_q && d < params.head_dim)
                    {
                        const std::size_t q_index =
                            ((static_cast<std::size_t>(batch) * params.q_len +
                              q_row) *
                                 params.q_heads +
                             q_head) *
                                params.head_dim +
                            d;
                        q_tile[row_slot][d_slot] =
                            toFloat(params.query[q_index]);
                    }
                    else
                    {
                        q_tile[row_slot][d_slot] = 0.0f;
                    }
                    acc_reg[row_slot][d_slot] = 0.0f;
                }
            }

            const int kv_tile_nums =
                (params.kv_len + Traits::kBc - 1) / Traits::kBc;

            // FA2-style online softmax: each KV tile updates the running row max,
            // denominator and output accumulator in one pass. This preserves the
            // compatible layout/tail handling without materializing a score matrix.
            for (int kv_tile_idx = 0; kv_tile_idx < kv_tile_nums;
                 ++kv_tile_idx)
            {
                const int kv_tile_base = kv_tile_idx * Traits::kBc;
                const int cta_last_q =
                    min(cta_q_base + Traits::kRowsPerCta - 1,
                        params.q_len - 1);

                // This predicate is uniform for the whole CTA. Once a causal
                // KV tile starts after its last valid query row, every key in
                // the tile is masked for all four warps, so no load or online
                // softmax update is needed.
                if (params.is_causal && kv_tile_base > cta_last_q)
                {
                    continue;
                }

                const int kv_tile_elements = Traits::kBc * params.head_dim;

                // All threads participate because the block-wide barriers below
                // must also be reached by warps in the final partial Q tile.
                for (int i = tid; i < kv_tile_elements; i += Traits::kThreads)
                {
                    const int key_in_tile = i / params.head_dim;
                    const int d = i % params.head_dim;
                    const int key_row = kv_tile_base + key_in_tile;
                    const int tile_index = key_in_tile * HeadDim + d;

                    if (key_row < params.kv_len)
                    {
                        // [P0 layout] K/V 地址统一为
                        // (((b * kv_len + k) * kv_heads + kv_head) * D + d)。
                        const std::size_t kv_index =
                            ((static_cast<std::size_t>(batch) * params.kv_len +
                              key_row) *
                                 params.kv_heads +
                             kv_head) *
                                params.head_dim +
                            d;
                        k_tile[tile_index] = toFloat(params.key[kv_index]);
                        v_tile[tile_index] = toFloat(params.value[kv_index]);
                    }
                    else
                    {
                        k_tile[tile_index] = 0.0f;
                        v_tile[tile_index] = 0.0f;
                    }
                }
                __syncthreads();

                // Each warp updates its four independent online-softmax rows
                // while the same K/V tile remains resident in shared memory.
#pragma unroll
                for (int row_slot = 0; row_slot < Traits::kRowsPerWarp;
                     ++row_slot)
                {
                    const int row_in_cta =
                        row_slot * Traits::kWarpsPerCta + warp_id;
                    const int q_row = cta_q_base + row_in_cta;
                    const bool valid_q = q_row < params.q_len;
#if defined(FA2_TRACE_DEVICE)
                    const bool trace_row =
                        std::is_same<T, float>::value &&
                        ((params.batch_size == 4 && params.q_len == 64 &&
                          batch == 1 && q_row == 39 && q_head == 61) ||
                         (params.batch_size == 4 && params.q_len == 512 &&
                          batch == 2 && q_row == 199 && q_head == 62));
#endif
                    // Traits::kBc <= physical warp size, so lane k owns p[k].
                    float score_and_p = -INFINITY;
                    for (int key_in_tile = 0;
                         key_in_tile < Traits::kBc; ++key_in_tile)
                    {
                        const int key_row = kv_tile_base + key_in_tile;
                        const bool valid_key =
                            valid_q && key_row < params.kv_len &&
                            (!params.is_causal || key_row <= q_row);
                        const float *k_row = k_tile + key_in_tile * HeadDim;
                        float qk_dot = 0.0f;
#if defined(PLATFORM_NVIDIA)
                        if constexpr (std::is_same<T, float>::value)
                        {
                            if (valid_key && warp_lane == 0)
                            {
                                const std::size_t q_base =
                                    ((static_cast<std::size_t>(batch) *
                                          params.q_len +
                                      q_row) *
                                         params.q_heads +
                                     q_head) *
                                    params.head_dim;
                                for (int d = 0; d < params.head_dim; ++d)
                                {
                                    qk_dot = fmaf(params.query[q_base + d],
                                                  k_row[d], qk_dot);
                                }
                            }
                            qk_dot = platform::warpBroadcast(qk_dot, 0);
                        }
                        else
#endif
                        {
#pragma unroll
                            for (int d_slot = 0;
                                 d_slot < Traits::kDPerLane; ++d_slot)
                            {
                                const int d =
                                    warp_lane + d_slot * Traits::kWarpSize;
                                if (valid_key && d < params.head_dim)
                                {
                                    qk_dot = fmaf(q_tile[row_slot][d_slot],
                                                  k_row[d], qk_dot);
                                }
                            }
                            qk_dot = platform::warpReduceSum(qk_dot);
                        }

                        if (valid_key && warp_lane == key_in_tile)
                        {
                            score_and_p = qk_dot * params.scale;
#if defined(FA2_TRACE_DEVICE)
                            if (trace_row)
                            {
                                printf("FA2_SCORE B=%d Q=%d H=%d TILE=%d K=%d "
                                       "score=%.9e\n",
                                       batch, q_row, q_head, kv_tile_base,
                                       key_row, score_and_p);
                            }
#endif
                        }
                    }

                    const float tile_max =
                        platform::warpReduceMax(score_and_p);
                    const float m_old = m_reg[row_slot];
                    const float m_new = fmaxf(m_old, tile_max);
                    const float alpha = m_old == -INFINITY
                                            ? 0.0f
                                            : __expf(m_old - m_new);
                    m_reg[row_slot] = m_new;

                    score_and_p = score_and_p == -INFINITY
                                      ? 0.0f
                                      : __expf(score_and_p - m_new);
                    const float tile_sum =
                        platform::warpReduceSum(score_and_p);
                    l_reg[row_slot] =
                        fmaf(alpha, l_reg[row_slot], tile_sum);

#if defined(FA2_TRACE_DEVICE)
                    if (trace_row && warp_lane == 0)
                    {
                        printf("FA2_TILE B=%d Q=%d H=%d TILE=%d "
                               "m_old=%.9e m_new=%.9e alpha=%.9e "
                               "tile_sum=%.9e l=%.9e\n",
                               batch, q_row, q_head, kv_tile_base, m_old,
                               m_new, alpha, tile_sum, l_reg[row_slot]);
                    }
#endif

#pragma unroll
                    for (int d_slot = 0;
                         d_slot < Traits::kDPerLane; ++d_slot)
                    {
                        const int d =
                            warp_lane + d_slot * Traits::kWarpSize;
                        float tile_acc = 0.0f;
                        for (int key_in_tile = 0;
                             key_in_tile < Traits::kBc; ++key_in_tile)
                        {
                            const float p = platform::warpBroadcast(
                                score_and_p, key_in_tile);
                            if (d < params.head_dim)
                            {
                                tile_acc = fmaf(
                                    p,
                                    v_tile[key_in_tile * HeadDim + d],
                                    tile_acc);
                            }
                        }
                        if (d < params.head_dim)
                        {
                            acc_reg[row_slot][d_slot] = fmaf(
                                alpha, acc_reg[row_slot][d_slot], tile_acc);
#if defined(FA2_TRACE_DEVICE)
                            const int trace_d =
                                params.q_len == 64 ? 14 : 28;
                            if (trace_row && d == trace_d)
                            {
                                printf("FA2_ACC B=%d Q=%d H=%d D=%d TILE=%d "
                                       "tile_acc=%.9e acc=%.9e norm=%.9e\n",
                                       batch, q_row, q_head, d, kv_tile_base,
                                       tile_acc,
                                       acc_reg[row_slot][d_slot],
                                       acc_reg[row_slot][d_slot] /
                                           l_reg[row_slot]);
                            }
#endif
                        }
                    }
                }
                __syncthreads();
            }

#pragma unroll
            for (int row_slot = 0; row_slot < Traits::kRowsPerWarp;
                 ++row_slot)
            {
                const int row_in_cta =
                    row_slot * Traits::kWarpsPerCta + warp_id;
                const int q_row = cta_q_base + row_in_cta;
                if (q_row < params.q_len)
                {
                    const float inv_l = l_reg[row_slot] > 0.0f
                                            ? 1.0f / l_reg[row_slot]
                                            : 0.0f;
#pragma unroll
                    for (int d_slot = 0;
                         d_slot < Traits::kDPerLane; ++d_slot)
                    {
                        const int d =
                            warp_lane + d_slot * Traits::kWarpSize;
                        if (d >= params.head_dim)
                        {
                            continue;
                        }
                        const std::size_t output_index =
                            ((static_cast<std::size_t>(batch) * params.q_len +
                              q_row) *
                                 params.q_heads +
                             q_head) *
                                params.head_dim +
                            d;
                        params.output[output_index] =
                            fromFloat<T>(acc_reg[row_slot][d_slot] * inv_l);
                    }
                }
            }
        }

        template <typename T, int HeadDim>
        cudaError_t launchKernel(const Fa2Params<T> &params,
                                 cudaStream_t stream)
        {
            using Traits = Fa2Traits<HeadDim>;
            const dim3 block(Traits::kThreads);
            const dim3 grid((params.q_len + Traits::kRowsPerCta - 1) /
                                Traits::kRowsPerCta,
                            params.q_heads, params.batch_size);
            fa2CompatibleKernel<T, HeadDim><<<grid, block, 0, stream>>>(params);
            return cudaGetLastError();
        }

    } // namespace

    template <typename T>
    cudaError_t launch(T *output, const T *query, const T *key,
                       const T *value, int batch_size, int q_len, int kv_len,
                       int q_heads, int kv_heads, int head_dim, bool is_causal,
                       float scale, cudaStream_t stream)
    {
        if (batch_size == 0 || q_len == 0)
        {
            return cudaSuccess;
        }
        if (output == nullptr || query == nullptr || key == nullptr ||
            value == nullptr || batch_size < 0 || q_len < 0 || kv_len <= 0 ||
            q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
            q_heads % kv_heads != 0)
        {
            return cudaErrorInvalidValue;
        }

        Fa2Params<T> params{output, query, key, value, batch_size, q_len, kv_len,
                            q_heads, kv_heads, head_dim, is_causal, scale};

        if (head_dim <= 1)
            return launchKernel<T, 1>(params, stream);
        if (head_dim <= 2)
            return launchKernel<T, 2>(params, stream);
        if (head_dim <= 4)
            return launchKernel<T, 4>(params, stream);
        if (head_dim <= 8)
            return launchKernel<T, 8>(params, stream);
        if (head_dim <= 16)
            return launchKernel<T, 16>(params, stream);
        if (head_dim <= 32)
            return launchKernel<T, 32>(params, stream);
        if (head_dim <= 64)
            return launchKernel<T, 64>(params, stream);
        if (head_dim <= 128)
            return launchKernel<T, 128>(params, stream);
        if (head_dim <= 256)
            return launchKernel<T, 256>(params, stream);
        return cudaErrorInvalidValue;
    }

    template cudaError_t launch<float>(
        float *, const float *, const float *, const float *, int, int, int,
        int, int, int, bool, float, cudaStream_t);

    template cudaError_t launch<__half>(
        __half *, const __half *, const __half *, const __half *, int, int,
        int, int, int, int, bool, float, cudaStream_t);

#if defined(FA2_COMPATIBLE_HAS_BFLOAT16)
    template cudaError_t launch<__nv_bfloat16>(
        __nv_bfloat16 *, const __nv_bfloat16 *, const __nv_bfloat16 *,
        const __nv_bfloat16 *, int, int, int, int, int, int, bool, float,
        cudaStream_t);
#endif

} // namespace fa2_compatible

#if defined(FA2_COMPATIBLE_HAS_BFLOAT16)
#undef FA2_COMPATIBLE_HAS_BFLOAT16
#endif

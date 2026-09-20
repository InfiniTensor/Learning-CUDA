#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace fused_hq::detail {
template <typename T> __device__ __forceinline__ float to_float(T value);
template <> __device__ __forceinline__ float to_float(float value) { return value; }
template <> __device__ __forceinline__ float to_float(__half value) { return __half2float(value); }
template <> __device__ __forceinline__ float to_float(__nv_bfloat16 value) { return __bfloat162float(value); }

__device__ __forceinline__ float warp_max(float value) {
    for (int offset = 16; offset; offset >>= 1)
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    return value;
}

__device__ __forceinline__ std::int8_t quantize(float value, float inv_scale) {
    int q = __float2int_rn(value * inv_scale);
    q = max(-127, min(127, q));
    return static_cast<std::int8_t>(q);
}

template <typename T>
__device__ __forceinline__ void scalar_load(const T* input, float* output, int dim) {
    for (int col = threadIdx.x; col < dim; col += blockDim.x) output[col] = to_float(input[col]);
}

template <typename T>
__device__ __forceinline__ void vector_load(const T* input, float* output, int dim) {
    scalar_load(input, output, dim);
}
template <>
__device__ __forceinline__ void vector_load<float>(const float* input, float* output, int dim) {
    const auto* src = reinterpret_cast<const float4*>(input);
    for (int i = threadIdx.x; i < dim / 4; i += blockDim.x) {
        float4 v = src[i]; int c = i * 4;
        output[c] = v.x; output[c + 1] = v.y; output[c + 2] = v.z; output[c + 3] = v.w;
    }
}
template <>
__device__ __forceinline__ void vector_load<__half>(const __half* input, float* output, int dim) {
    const auto* src = reinterpret_cast<const __half2*>(input);
    for (int i = threadIdx.x; i < dim / 2; i += blockDim.x) {
        __half2 v = src[i]; int c = i * 2;
        output[c] = __half2float(v.x); output[c + 1] = __half2float(v.y);
    }
}
template <>
__device__ __forceinline__ void vector_load<__nv_bfloat16>(const __nv_bfloat16* input, float* output, int dim) {
    const auto* src = reinterpret_cast<const __nv_bfloat162*>(input);
    for (int i = threadIdx.x; i < dim / 2; i += blockDim.x) {
        __nv_bfloat162 v = src[i]; int c = i * 2;
        output[c] = __bfloat162float(v.x); output[c + 1] = __bfloat162float(v.y);
    }
}

__device__ __forceinline__ float block_max(float value, float* warp_values) {
    value = warp_max(value);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();
    float result = threadIdx.x < (blockDim.x >> 5) ? warp_values[lane] : 0.0f;
    if (warp == 0) result = warp_max(result);
    if (threadIdx.x == 0) warp_values[0] = result;
    __syncthreads();
    return warp_values[0];
}
}  // namespace fused_hq::detail

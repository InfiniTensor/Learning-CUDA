#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace hadamard::detail {

template <typename T>
__device__ __forceinline__ float to_float(T value);

template <>
__device__ __forceinline__ float to_float<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ float to_float<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(
    __nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value);

template <>
__device__ __forceinline__ float from_float<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ __half from_float<__half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(
    float value) {
    return __float2bfloat16_rn(value);
}

}  // namespace hadamard::detail

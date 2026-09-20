#include "include/hadamard.cuh"
#include "kernels/hadamard_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace hadamard {
namespace {

template <typename T>
__device__ __forceinline__ void vector_load(const T* input, float* output,
                                            int count);

template <>
__device__ __forceinline__ void vector_load<float>(const float* input,
                                                   float* output, int count) {
    for (int col = static_cast<int>(threadIdx.x) * 4; col < count;
         col += static_cast<int>(blockDim.x) * 4) {
        const float4 value = reinterpret_cast<const float4*>(input)[col / 4];
        output[col] = value.x;
        output[col + 1] = value.y;
        output[col + 2] = value.z;
        output[col + 3] = value.w;
    }
}

template <>
__device__ __forceinline__ void vector_load<__half>(const __half* input,
                                                    float* output, int count) {
    for (int col = static_cast<int>(threadIdx.x) * 2; col < count;
         col += static_cast<int>(blockDim.x) * 2) {
        const float2 value =
            __half22float2(reinterpret_cast<const __half2*>(input)[col / 2]);
        output[col] = value.x;
        output[col + 1] = value.y;
    }
}

template <>
__device__ __forceinline__ void vector_load<__nv_bfloat16>(
    const __nv_bfloat16* input, float* output, int count) {
    for (int col = static_cast<int>(threadIdx.x) * 2; col < count;
         col += static_cast<int>(blockDim.x) * 2) {
        const float2 value = __bfloat1622float2(
            reinterpret_cast<const __nv_bfloat162*>(input)[col / 2]);
        output[col] = value.x;
        output[col + 1] = value.y;
    }
}

template <typename T>
__device__ __forceinline__ void vector_store(const float* input, T* output,
                                             int count);

template <>
__device__ __forceinline__ void vector_store<float>(const float* input,
                                                    float* output, int count) {
    for (int col = static_cast<int>(threadIdx.x) * 4; col < count;
         col += static_cast<int>(blockDim.x) * 4) {
        reinterpret_cast<float4*>(output)[col / 4] =
            make_float4(input[col], input[col + 1], input[col + 2],
                        input[col + 3]);
    }
}

template <>
__device__ __forceinline__ void vector_store<__half>(const float* input,
                                                     __half* output,
                                                     int count) {
    for (int col = static_cast<int>(threadIdx.x) * 2; col < count;
         col += static_cast<int>(blockDim.x) * 2) {
        reinterpret_cast<__half2*>(output)[col / 2] =
            __floats2half2_rn(input[col], input[col + 1]);
    }
}

template <>
__device__ __forceinline__ void vector_store<__nv_bfloat16>(
    const float* input, __nv_bfloat16* output, int count) {
    for (int col = static_cast<int>(threadIdx.x) * 2; col < count;
         col += static_cast<int>(blockDim.x) * 2) {
        reinterpret_cast<__nv_bfloat162*>(output)[col / 2] =
            __floats2bfloat162_rn(input[col], input[col + 1]);
    }
}

template <typename T>
__global__ void hadamard_v2_kernel(const T* input, T* output, int head_dim) {
    extern __shared__ float values[];
    const int tid = static_cast<int>(threadIdx.x);
    const std::int64_t row = blockIdx.x;
    const T* row_input = input + row * head_dim;
    T* row_output = output + row * head_dim;

    vector_load<T>(row_input, values, head_dim);
    __syncthreads();

    // Each loop iteration maps contiguous groups of 32 values to a warp, so
    // the first five butterfly stages remain in registers via shuffle.
    for (int base = tid; base < head_dim; base += blockDim.x) {
        float value = values[base];
        const unsigned int mask = __activemask();
        for (int stride = 1; stride < 32; stride <<= 1) {
            const float partner = __shfl_xor_sync(mask, value, stride);
            value = (base & stride) == 0 ? value + partner : partner - value;
        }
        values[base] = value;
    }
    __syncthreads();

    for (int stride = 32; stride < head_dim; stride <<= 1) {
        for (int butterfly = tid; butterfly < head_dim / 2;
             butterfly += blockDim.x) {
            const int group = butterfly / stride;
            const int offset = butterfly % stride;
            const int low = group * 2 * stride + offset;
            const int high = low + stride;
            const float a = values[low];
            const float b = values[high];
            values[low] = a + b;
            values[high] = a - b;
        }
        __syncthreads();
    }

    const float scale = rsqrtf(static_cast<float>(head_dim));
    for (int col = tid; col < head_dim; col += blockDim.x) {
        values[col] *= scale;
    }
    __syncthreads();
    vector_store<T>(values, row_output, head_dim);
}

template <typename T>
cudaError_t launch_typed(const T* input, T* output, std::int64_t batch,
                         int head_dim, cudaStream_t stream) {
    constexpr int threads = 256;
    hadamard_v2_kernel<<<static_cast<unsigned int>(batch), threads,
                          head_dim * sizeof(float), stream>>>(
        input, output, head_dim);
    return cudaGetLastError();
}

}  // namespace

cudaError_t launch_hadamard_v2(const void* input, void* output,
                               std::int64_t batch, int head_dim,
                               DType dtype, cudaStream_t stream) {
    switch (dtype) {
        case DType::kFloat32:
            return launch_typed(static_cast<const float*>(input),
                                static_cast<float*>(output), batch, head_dim,
                                stream);
        case DType::kFloat16:
            return launch_typed(static_cast<const __half*>(input),
                                static_cast<__half*>(output), batch, head_dim,
                                stream);
        case DType::kBFloat16:
            return launch_typed(static_cast<const __nv_bfloat16*>(input),
                                static_cast<__nv_bfloat16*>(output), batch,
                                head_dim, stream);
    }
    return cudaErrorInvalidValue;
}

}  // namespace hadamard

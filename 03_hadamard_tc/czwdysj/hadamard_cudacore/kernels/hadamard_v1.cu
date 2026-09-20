#include "include/hadamard.cuh"
#include "kernels/hadamard_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>

namespace hadamard {
namespace {

template <typename T>
__global__ void hadamard_v1_kernel(const T* input, T* output, int head_dim) {
    extern __shared__ float values[];
    const int tid = static_cast<int>(threadIdx.x);
    const std::int64_t row = blockIdx.x;
    const T* row_input = input + row * head_dim;
    T* row_output = output + row * head_dim;

    for (int col = tid; col < head_dim; col += blockDim.x) {
        values[col] = detail::to_float(row_input[col]);
    }
    __syncthreads();

    for (int stride = 1; stride < head_dim; stride <<= 1) {
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
        row_output[col] = detail::from_float<T>(values[col] * scale);
    }
}

template <typename T>
cudaError_t launch_typed(const T* input, T* output, std::int64_t batch,
                         int head_dim, cudaStream_t stream) {
    constexpr int threads = 256;
    hadamard_v1_kernel<<<static_cast<unsigned int>(batch), threads,
                          head_dim * sizeof(float), stream>>>(
        input, output, head_dim);
    return cudaGetLastError();
}

}  // namespace

cudaError_t launch_hadamard_v1(const void* input, void* output,
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

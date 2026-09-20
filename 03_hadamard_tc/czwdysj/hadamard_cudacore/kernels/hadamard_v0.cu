#include "include/hadamard.cuh"
#include "kernels/hadamard_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>

namespace hadamard {
namespace {

template <typename T>
__global__ void hadamard_stage(const T* input, T* output,
                               std::int64_t butterflies, int head_dim,
                               int stride, float final_scale) {
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= butterflies) {
        return;
    }
    const std::int64_t row = index / (head_dim / 2);
    const int butterfly = static_cast<int>(index % (head_dim / 2));
    const int group = butterfly / stride;
    const int offset = butterfly % stride;
    const std::int64_t low = row * head_dim + group * 2 * stride + offset;
    const std::int64_t high = low + stride;
    const float a = detail::to_float(input[low]);
    const float b = detail::to_float(input[high]);
    output[low] = detail::from_float<T>((a + b) * final_scale);
    output[high] = detail::from_float<T>((a - b) * final_scale);
}

template <typename T>
cudaError_t launch_typed(const T* input, T* output, T* workspace,
                         std::int64_t batch, int head_dim,
                         cudaStream_t stream) {
    int stages = 0;
    for (int value = head_dim; value > 1; value >>= 1) {
        ++stages;
    }
    const std::int64_t butterflies = batch * (head_dim / 2);
    constexpr int threads = 256;
    const int blocks = static_cast<int>((butterflies + threads - 1) / threads);
    const T* source = input;
    for (int stage = 0, stride = 1; stage < stages; ++stage, stride <<= 1) {
        T* destination =
            (stage % 2 == (stages - 1) % 2) ? output : workspace;
        const float scale = stage == stages - 1
                                ? 1.0f / std::sqrt(static_cast<float>(head_dim))
                                : 1.0f;
        hadamard_stage<<<blocks, threads, 0, stream>>>(
            source, destination, butterflies, head_dim, stride, scale);
        const cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            return error;
        }
        source = destination;
    }
    return cudaSuccess;
}

}  // namespace

cudaError_t launch_hadamard_v0(const void* input, void* output,
                               void* workspace, std::int64_t batch,
                               int head_dim, DType dtype,
                               cudaStream_t stream) {
    if (workspace == nullptr) {
        return cudaErrorInvalidValue;
    }
    switch (dtype) {
        case DType::kFloat32:
            return launch_typed(static_cast<const float*>(input),
                                static_cast<float*>(output),
                                static_cast<float*>(workspace), batch, head_dim,
                                stream);
        case DType::kFloat16:
            return launch_typed(static_cast<const __half*>(input),
                                static_cast<__half*>(output),
                                static_cast<__half*>(workspace), batch, head_dim,
                                stream);
        case DType::kBFloat16:
            return launch_typed(static_cast<const __nv_bfloat16*>(input),
                                static_cast<__nv_bfloat16*>(output),
                                static_cast<__nv_bfloat16*>(workspace), batch,
                                head_dim, stream);
    }
    return cudaErrorInvalidValue;
}

}  // namespace hadamard

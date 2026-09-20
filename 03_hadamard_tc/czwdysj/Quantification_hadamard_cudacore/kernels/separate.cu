#include "include/fused_hadamard_quant.cuh"
#include "kernels/common.cuh"
namespace fused_hq { namespace {
template <typename T> __global__ void hadamard_kernel(const T* input, float* output, int dim) {
    extern __shared__ float values[]; const int row = blockIdx.x, tid = threadIdx.x;
    detail::vector_load(input + static_cast<std::int64_t>(row) * dim, values, dim); __syncthreads();
    for (int stride = 1; stride < dim; stride <<= 1) {
        for (int butterfly = tid; butterfly < dim / 2; butterfly += blockDim.x) {
            int group = butterfly / stride, offset = butterfly % stride;
            int low = group * 2 * stride + offset, high = low + stride;
            float x = values[low], y = values[high]; values[low] = x + y; values[high] = x - y;
        } __syncthreads();
    }
    float norm = rsqrtf(static_cast<float>(dim));
    for (int col = tid; col < dim; col += blockDim.x)
        output[static_cast<std::int64_t>(row) * dim + col] = values[col] * norm;
}
__global__ void amax_kernel(const float* input, float* amax_out, float* scale_out, int dim) {
    __shared__ float warps[8]; const int row = blockIdx.x; float local = 0.0f;
    for (int col = threadIdx.x; col < dim; col += blockDim.x)
        local = fmaxf(local, fabsf(input[static_cast<std::int64_t>(row) * dim + col]));
    float maximum = detail::block_max(local, warps);
    if (threadIdx.x == 0) { amax_out[row] = maximum; scale_out[row] = maximum > 0 ? maximum / 127.0f : 1.0f; }
}
__global__ void quantize_kernel(const float* input, std::int8_t* output, const float* scale, std::int64_t count, int dim) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::int64_t>(blockDim.x) * gridDim.x)
        output[i] = detail::quantize(input[i], 1.0f / scale[i / dim]);
}
template <typename T> cudaError_t typed(const void* input, std::int8_t* output, float* amax, float* scale,
                                         void* workspace, std::int64_t batch, int dim, cudaStream_t stream) {
    auto* intermediate = static_cast<float*>(workspace);
    hadamard_kernel<<<static_cast<unsigned>(batch), 256, dim * sizeof(float), stream>>>(static_cast<const T*>(input), intermediate, dim);
    cudaError_t error = cudaGetLastError(); if (error != cudaSuccess) return error;
    amax_kernel<<<static_cast<unsigned>(batch), 256, 0, stream>>>(intermediate, amax, scale, dim);
    error = cudaGetLastError(); if (error != cudaSuccess) return error;
    std::int64_t count = batch * dim; int blocks = static_cast<int>((count + 255) / 256); blocks = min(blocks, 65535);
    quantize_kernel<<<blocks, 256, 0, stream>>>(intermediate, output, scale, count, dim);
    return cudaGetLastError();
}
} cudaError_t launch_separate(const void* i, std::int8_t* o, float* a, float* s, void* w, std::int64_t b, int d, DType t, cudaStream_t st) {
    if (t == DType::kFloat32) return typed<float>(i,o,a,s,w,b,d,st);
    if (t == DType::kFloat16) return typed<__half>(i,o,a,s,w,b,d,st);
    return typed<__nv_bfloat16>(i,o,a,s,w,b,d,st);
} }  // namespace fused_hq

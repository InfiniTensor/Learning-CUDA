#include "include/fused_hadamard_quant.cuh"
#include "kernels/common.cuh"
namespace fused_hq { namespace {
template <typename T> __global__ void fused_v0_kernel(const T* input, std::int8_t* output, float* amax_out, float* scale_out, int dim) {
    extern __shared__ float shared[]; float* values = shared; float* reduction = shared + dim;
    const int row = blockIdx.x, tid = threadIdx.x;
    detail::scalar_load(input + static_cast<std::int64_t>(row) * dim, values, dim); __syncthreads();
    for (int stride = 1; stride < dim; stride <<= 1) {
        for (int butterfly = tid; butterfly < dim / 2; butterfly += blockDim.x) {
            int group = butterfly / stride, offset = butterfly % stride;
            int low = group * 2 * stride + offset, high = low + stride;
            float x = values[low], y = values[high]; values[low] = x + y; values[high] = x - y;
        } __syncthreads();
    }
    float norm = rsqrtf(static_cast<float>(dim)), local = 0.0f;
    for (int col = tid; col < dim; col += blockDim.x) { values[col] *= norm; local = fmaxf(local, fabsf(values[col])); }
    reduction[tid] = local; __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (tid < stride) reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]); __syncthreads();
    }
    float maximum = reduction[0], scale = maximum > 0 ? maximum / 127.0f : 1.0f;
    if (tid == 0) { amax_out[row] = maximum; scale_out[row] = scale; }
    for (int col = tid; col < dim; col += blockDim.x)
        output[static_cast<std::int64_t>(row) * dim + col] = detail::quantize(values[col], 1.0f / scale);
}
template <typename T> cudaError_t typed(const void* in, std::int8_t* out, float* a, float* s, std::int64_t batch, int dim, cudaStream_t stream) {
    fused_v0_kernel<<<static_cast<unsigned>(batch), 256, (dim + 256) * sizeof(float), stream>>>(static_cast<const T*>(in), out, a, s, dim);
    return cudaGetLastError();
}
} cudaError_t launch_v0(const void* i, std::int8_t* o, float* a, float* s, std::int64_t b, int d, DType t, cudaStream_t st) {
    if (t == DType::kFloat32) return typed<float>(i,o,a,s,b,d,st);
    if (t == DType::kFloat16) return typed<__half>(i,o,a,s,b,d,st);
    return typed<__nv_bfloat16>(i,o,a,s,b,d,st);
} }  // namespace fused_hq

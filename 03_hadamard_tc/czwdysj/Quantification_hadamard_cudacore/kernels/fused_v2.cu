#include "include/fused_hadamard_quant.cuh"
#include "kernels/common.cuh"
namespace fused_hq { namespace {
template <typename T> __global__ void fused_v2_kernel(const T* input, std::int8_t* output, float* amax_out, float* scale_out, int dim) {
    extern __shared__ float shared[]; float* values = shared; float* warps = shared + dim;
    const int row = blockIdx.x, tid = threadIdx.x; const std::int64_t base = static_cast<std::int64_t>(row) * dim;
    detail::vector_load(input + base, values, dim); __syncthreads();
    for (int col = tid; col < dim; col += blockDim.x) {
        float value = values[col];
        #pragma unroll
        for (int stride = 1; stride < 32; stride <<= 1) {
            float other = __shfl_xor_sync(0xffffffffu, value, stride);
            value = (col & stride) ? other - value : value + other;
        }
        values[col] = value;
    }
    __syncthreads();
    for (int stride = 32; stride < dim; stride <<= 1) {
        for (int butterfly = tid; butterfly < dim / 2; butterfly += blockDim.x) {
            int group = butterfly / stride, offset = butterfly % stride;
            int low = group * 2 * stride + offset, high = low + stride;
            float x = values[low], y = values[high]; values[low] = x + y; values[high] = x - y;
        } __syncthreads();
    }
    float norm = rsqrtf(static_cast<float>(dim)), local = 0.0f;
    for (int col = tid; col < dim; col += blockDim.x) { values[col] *= norm; local = fmaxf(local, fabsf(values[col])); }
    float maximum = detail::block_max(local, warps), scale = maximum > 0 ? maximum / 127.0f : 1.0f, inv = 1.0f / scale;
    if (tid == 0) { amax_out[row] = maximum; scale_out[row] = scale; }
    auto* packed = reinterpret_cast<char4*>(output + base);
    for (int i = tid; i < dim / 4; i += blockDim.x) {
        int col = i * 4; char4 q;
        q.x = static_cast<char>(detail::quantize(values[col], inv)); q.y = static_cast<char>(detail::quantize(values[col + 1], inv));
        q.z = static_cast<char>(detail::quantize(values[col + 2], inv)); q.w = static_cast<char>(detail::quantize(values[col + 3], inv)); packed[i] = q;
    }
}
template <typename T> cudaError_t typed(const void* in, std::int8_t* out, float* a, float* s, std::int64_t batch, int dim, cudaStream_t stream) {
    fused_v2_kernel<<<static_cast<unsigned>(batch), 256, (dim + 8) * sizeof(float), stream>>>(static_cast<const T*>(in), out, a, s, dim);
    return cudaGetLastError();
}
} cudaError_t launch_v2(const void* i, std::int8_t* o, float* a, float* s, std::int64_t b, int d, DType t, cudaStream_t st) {
    if (t == DType::kFloat32) return typed<float>(i,o,a,s,b,d,st);
    if (t == DType::kFloat16) return typed<__half>(i,o,a,s,b,d,st);
    return typed<__nv_bfloat16>(i,o,a,s,b,d,st);
} }  // namespace fused_hq

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifdef PLATFORM_METAX
static constexpr int WARP_SIZE = 64;
#elif defined(PLATFORM_ILUVATAR)
static constexpr int WARP_SIZE = 64;
#elif defined(PLATFORM_NVIDIA)
static constexpr int WARP_SIZE = 32;
#endif

#if defined(PLATFORM_METAX) || defined(PLATFORM_ILUVATAR)
using WarpMask = unsigned long long;
#elif defined(PLATFORM_NVIDIA)
using WarpMask = unsigned int;
#endif

static constexpr WarpMask FULL_MASK = ~WarpMask{0};

namespace
{

    constexpr int kBlockSize = 256;
    constexpr int kWarpsPerBlock = (kBlockSize + WARP_SIZE - 1) / WARP_SIZE;

    static_assert(kBlockSize % WARP_SIZE == 0,
                  "Full-mask reductions require complete warps");
    static_assert(kWarpsPerBlock <= WARP_SIZE,
                  "The second reduction stage must fit in one warp");

} // namespace
// 这里支持的是矩阵形式的RMSNorm
template <int Width = WARP_SIZE>
__device__ __forceinline__ float warpReduceSumXor(float value)
{
    static_assert(Width > 0 && (Width & (Width - 1)) == 0,
                  "Width must be a power of two");
    static_assert(Width <= WARP_SIZE,
                  "Width cannot exceed the hardware warp size");

#pragma unroll
    for (int offset = Width / 2; offset > 0; offset >>= 1)
    {
        value += __shfl_xor_sync(FULL_MASK, value, offset, Width);
    }
    return value;
}

__device__ float blockReduceSquareSum(float v)
{
    __shared__ float warp_sums[kWarpsPerBlock];
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    v = warpReduceSumXor(v);
    if (lane_id == 0)
        warp_sums[warp_id] = v;
    __syncthreads();

    if (warp_id == 0)
    {
        // Every lane in warp 0 participates; unused lanes contribute zero.
        v = lane_id < kWarpsPerBlock ? warp_sums[lane_id] : 0.0f;
        v = warpReduceSumXor(v);
    }
    return v;
}

__device__ __forceinline__ float toFloat(float x)
{
    return x;
}
__device__ __forceinline__ float toFloat(half x)
{
    return __half2float(x);
}

template <typename T>
__device__ __forceinline__
    T
    fromFloat(float value);

template <>
__device__ __forceinline__ float fromFloat<float>(float value)
{
    return value;
}

template <>
__device__ __forceinline__ half fromFloat<half>(float value)
{
    return __float2half_rn(value);
}

template <typename T>
__global__ void RMSNorm_kernel(const T *input, const T *weight, T *output, int hidden_dim, float eps = 1e-5)
{
    // 1.这里是处理一个Block(也就是一行的kernel)
    int tid = threadIdx.x;
    int gridStride = blockDim.x;
    float partial_sum = 0.0f;
    int rowoffset = blockIdx.x * hidden_dim;

    // 2.将可能超过线程总数的部分 提前压缩到线程数可以处理的部分
    for (int i = tid; i < hidden_dim; i += gridStride)
    {
        float val = toFloat(input[rowoffset + i]);
        partial_sum += val * val;
    }
    // 只有tid == 0时存有正确的sum
    float tmp = blockReduceSquareSum(partial_sum);
    __shared__ float sum;
    if (tid == 0)
        sum = rsqrtf(tmp / hidden_dim + eps); // 全局知道了RMS作为系数

    __syncthreads();
    for (int i = tid; i < hidden_dim; i += gridStride)
    {
        float value = toFloat(input[rowoffset + i]);
        float scale = toFloat(weight[i]);
        output[rowoffset + i] = fromFloat<T>(value * sum * scale);
    }
}

template <typename T>
void launchRMSNorm(const T *input, const T *weight, T *output, int rows, int hidden_dim, float eps)
{
    dim3 block(kBlockSize);
    dim3 grid(rows);

    RMSNorm_kernel<T><<<grid, block>>>(input, weight, output, hidden_dim, eps);
}

template void launchRMSNorm<float>(const float *, const float *, float *, int, int, float);
template void launchRMSNorm<half>(const half *, const half *, half *, int, int, float);

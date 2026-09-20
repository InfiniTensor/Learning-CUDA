#include "hadamard_cuda.cuh"

#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace hadamard {
namespace {

// BF16 在内存中只保存高 16 位。host 侧用 memcpy 避免严格别名问题，
// device 侧用 CUDA intrinsic 完成 bit cast。
__host__ __device__ inline float bf16_bits_to_float_raw(std::uint16_t bits) {
    std::uint32_t word = static_cast<std::uint32_t>(bits) << 16;
#ifdef __CUDA_ARCH__
    return __uint_as_float(word);
#else
    float value;
    std::memcpy(&value, &word, sizeof(value));
    return value;
#endif
}

__host__ __device__ inline std::uint16_t float_to_bf16_bits_raw(float value) {
#ifdef __CUDA_ARCH__
    std::uint32_t word = __float_as_uint(value);
#else
    std::uint32_t word;
    std::memcpy(&word, &value, sizeof(word));
#endif
    const std::uint32_t lsb = (word >> 16) & 1u;
    const std::uint32_t rounding_bias = 0x7fffu + lsb;
    return static_cast<std::uint16_t>((word + rounding_bias) >> 16);
}

__device__ inline float load_value(const void* ptr, int index, DType dtype) {
    if (dtype == DType::FP16) {
        return __half2float(reinterpret_cast<const half*>(ptr)[index]);
    }
    return bf16_bits_to_float_raw(reinterpret_cast<const std::uint16_t*>(ptr)[index]);
}

__device__ inline void store_value(void* ptr, int index, DType dtype, float value) {
    if (dtype == DType::FP16) {
        reinterpret_cast<half*>(ptr)[index] = __float2half_rn(value);
    } else {
        reinterpret_cast<std::uint16_t*>(ptr)[index] = float_to_bf16_bits_raw(value);
    }
}

__device__ inline float round_to_output_dtype(float value, DType dtype) {
    if (dtype == DType::FP16) {
        return __half2float(__float2half_rn(value));
    }
    return bf16_bits_to_float_raw(float_to_bf16_bits_raw(value));
}

// 对称 signed INT4 量化，返回低 4 bit 补码表示，调用方负责打包两个元素。
__device__ inline int quantize_s4(float value, float scale) {
    const float scaled = scale > 0.0f ? value / scale : 0.0f;
    int q = static_cast<int>(lrintf(scaled));
    q = max(-8, min(7, q));
    return q & 0x0f;
}

__device__ inline std::uint8_t encode_e4m3(float value) {
    if (isnan(value)) {
        return 0x7fu;
    }

    const std::uint8_t sign = signbit(value) ? 0x80u : 0u;
    float x = fabsf(value);
    if (x == 0.0f) {
        return sign;
    }

    // 有限 E4M3 格式：1 个符号位、4 个指数位（bias=7）、3 个尾数位。
    // 最大有限值按 240.0 截断；这里作为软件模拟路径，重点保证 fused
    // 与非 fused 量化逐字节一致。
    x = fminf(x, 240.0f);

    int exp2 = 0;
    const float mant = frexpf(x, &exp2);  // x = mant * 2^exp2，mant 位于 [0.5, 1)
    int unbiased = exp2 - 1;
    int biased = unbiased + 7;

    if (biased <= 0) {
        int sub = static_cast<int>(lrintf(x * 512.0f));  // subnormal 区间步长为 2^-9。
        sub = max(0, min(7, sub));
        return sign | static_cast<std::uint8_t>(sub);
    }

    float normalized = mant * 2.0f - 1.0f;
    int mantissa = static_cast<int>(lrintf(normalized * 8.0f));
    if (mantissa == 8) {
        mantissa = 0;
        ++biased;
    }
    if (biased >= 15) {
        biased = 14;
        mantissa = 7;
    }
    return sign | static_cast<std::uint8_t>((biased << 3) | mantissa);
}

__global__ void hadamard_shared_kernel(
    const void* input,
    void* output,
    int rows,
    int dim,
    DType dtype) {
    extern __shared__ float values[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows || tid >= dim) {
        return;
    }

    // 每个 block 处理一个展平后的 (batch, seq, head) 行向量。
    values[tid] = load_value(input, row * dim + tid, dtype);
    __syncthreads();

    // 标准 FHT 蝶形计算：(a, b) -> (a + b, a - b)。
    for (int stride = 1; stride < dim; stride <<= 1) {
        if ((tid & stride) == 0) {
            const float a = values[tid];
            const float b = values[tid + stride];
            values[tid] = a + b;
            values[tid + stride] = a - b;
        }
        __syncthreads();
    }

    // 乘以 1/sqrt(dim) 得到正交归一化 Hadamard 变换。
    const float scale = rsqrtf(static_cast<float>(dim));
    store_value(output, row * dim + tid, dtype, values[tid] * scale);
}

__global__ void quantize_int4_kernel(
    const void* transformed,
    std::uint8_t* packed,
    float* scales,
    int rows,
    int dim,
    DType dtype) {
    extern __shared__ float scratch[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows || tid >= dim) {
        return;
    }

    // 逐行统计最大绝对值，用作该 head 向量的 INT4 对称量化 scale。
    float v = load_value(transformed, row * dim + tid, dtype);
    scratch[tid] = fabsf(v);
    __syncthreads();

    for (int stride = dim >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        }
        __syncthreads();
    }

    const float scale = scratch[0] > 0.0f ? scratch[0] / 7.0f : 1.0f;
    if (tid == 0) {
        scales[row] = scale;
    }

    if (tid < dim / 2) {
        // packed INT4：低 4 bit 存偶数位置，高 4 bit 存奇数位置。
        const float v0 = load_value(transformed, row * dim + 2 * tid, dtype);
        const float v1 = load_value(transformed, row * dim + 2 * tid + 1, dtype);
        const int q0 = quantize_s4(v0, scale);
        const int q1 = quantize_s4(v1, scale);
        packed[row * (dim / 2) + tid] = static_cast<std::uint8_t>(q0 | (q1 << 4));
    }
}


__global__ void fused_hadamard_int4_kernel2(
    const void* input,
    std::uint8_t* packed,
    float* scales,
    int rows,
    int dim,
    DType dtype) {
    extern __shared__ float shared[];
    // values 保存 Hadamard 输出，reduction 单独保存归约工作区，避免覆盖
    // 后续量化还需要读取的变换结果。
    float* values = shared;
    float* reduction = shared + dim;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows || tid >= dim) {
        return;
    }

    values[tid] = load_value(input, row * dim + tid, dtype);
    __syncthreads();

    for (int stride = 1; stride < dim; stride <<= 1) {
        if ((tid & stride) == 0) {
            const float a = values[tid];
            const float b = values[tid + stride];
            values[tid] = a + b;
            values[tid + stride] = a - b;
        }
        __syncthreads();
    }

    // fused 路径量化前显式模拟输出 dtype 舍入，确保与“先写回再量化”一致。
    values[tid] = round_to_output_dtype(values[tid] * rsqrtf(static_cast<float>(dim)), dtype);
    reduction[tid] = fabsf(values[tid]);
    __syncthreads();
    for (int stride = dim >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
        }
        __syncthreads();
    }
    const float scale = reduction[0] > 0.0f ? reduction[0] / 7.0f : 1.0f;
    if (tid == 0) {
        scales[row] = scale;
    }
    if (tid < dim / 2) {
        const int q0 = quantize_s4(values[2 * tid], scale);
        const int q1 = quantize_s4(values[2 * tid + 1], scale);
        packed[row * (dim / 2) + tid] = static_cast<std::uint8_t>(q0 | (q1 << 4));
    }
}

__global__ void quantize_e4m3_kernel(
    const void* transformed,
    std::uint8_t* encoded,
    int rows,
    int dim,
    DType dtype) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * dim;
    if (idx < total) {
        encoded[idx] = encode_e4m3(load_value(transformed, idx, dtype));
    }
}

__global__ void fused_hadamard_e4m3_kernel(
    const void* input,
    std::uint8_t* encoded,
    int rows,
    int dim,
    DType dtype) {
    extern __shared__ float values[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows || tid >= dim) {
        return;
    }

    values[tid] = load_value(input, row * dim + tid, dtype);
    __syncthreads();

    for (int stride = 1; stride < dim; stride <<= 1) {
        if ((tid & stride) == 0) {
            const float a = values[tid];
            const float b = values[tid + stride];
            values[tid] = a + b;
            values[tid + stride] = a - b;
        }
        __syncthreads();
    }

    encoded[row * dim + tid] = encode_e4m3(round_to_output_dtype(values[tid] * rsqrtf(static_cast<float>(dim)), dtype));
}

// WMMA 对照 kernel：把 Hadamard 变换视为矩阵乘法 X * H_scaled。
// 这条路径用于 Tensor Core 对比，主路径仍是复杂度更低的 FHT。
__global__ void hadamard_wmma_fp16_kernel(
    const half* input,
    half* output,
    const half* hmat,
    int rows,
    int dim) {
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    constexpr int tile = 16;
    __shared__ half a_shared[tile * tile];
    __shared__ half b_shared[tile * tile];
    __shared__ float c_shared[tile * tile];

    const int row_base = blockIdx.y * tile;
    const int col_base = blockIdx.x * tile;
    const int lane = threadIdx.x;

    wmma::fragment<wmma::matrix_a, tile, tile, tile, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, tile, tile, tile, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, tile, tile, tile, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k_base = 0; k_base < dim; k_base += tile) {
        for (int i = lane; i < tile * tile; i += blockDim.x) {
            const int r = i / tile;
            const int c = i % tile;
            const int row = row_base + r;
            a_shared[i] = row < rows ? input[row * dim + k_base + c] : __float2half(0.0f);
            b_shared[i] = hmat[(k_base + r) * dim + col_base + c];
        }
        __syncthreads();
        wmma::load_matrix_sync(a_frag, a_shared, tile);
        wmma::load_matrix_sync(b_frag, b_shared, tile);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    wmma::store_matrix_sync(c_shared, c_frag, tile, wmma::mem_row_major);
    __syncthreads();
    for (int i = lane; i < tile * tile; i += blockDim.x) {
        const int r = i / tile;
        const int c = i % tile;
        const int row = row_base + r;
        const int col = col_base + c;
        if (row < rows && col < dim) {
            output[row * dim + col] = __float2half_rn(c_shared[i]);
        }
    }
#endif
}

void validate_dim(int dim) {
    if (dim < 1 || dim > 256 || (dim & (dim - 1)) != 0) {
        throw std::invalid_argument("head_dim must be a power of two in [1, 256]");
    }
}

KernelTiming time_repeated(cudaEvent_t start, cudaEvent_t stop, int repeats) {
    HADAMARD_CUDA_CHECK(cudaEventRecord(stop));
    HADAMARD_CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    HADAMARD_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    KernelTiming timing;
    timing.milliseconds = ms / static_cast<float>(std::max(1, repeats));
    return timing;
}

}  // namespace

const char* dtype_name(DType dtype) {
    return dtype == DType::FP16 ? "fp16" : "bf16";
}

std::size_t element_size(DType dtype) {
    (void)dtype;
    return sizeof(std::uint16_t);
}

std::size_t tensor_bytes(int rows, int dim, DType dtype) {
    return static_cast<std::size_t>(rows) * dim * element_size(dtype);
}

std::size_t int4_packed_bytes(int rows, int dim) {
    return static_cast<std::size_t>(rows) * ((dim + 1) / 2);
}

void check_cuda_status(cudaError_t status, const char* expr, const char* file, int line) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) + " CUDA call failed: " +
            expr + " -> " + cudaGetErrorString(status));
    }
}

KernelTiming launch_hadamard_shared(
    const void* input,
    void* output,
    int rows,
    int dim,
    DType dtype,
    int repeats) {
    validate_dim(dim);
    cudaEvent_t start;
    cudaEvent_t stop;
    HADAMARD_CUDA_CHECK(cudaEventCreate(&start));
    HADAMARD_CUDA_CHECK(cudaEventCreate(&stop));
    HADAMARD_CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        hadamard_shared_kernel<<<rows, dim, dim * sizeof(float)>>>(input, output, rows, dim, dtype);
    }
    HADAMARD_CUDA_CHECK(cudaGetLastError());
    KernelTiming timing = time_repeated(start, stop, repeats);
    HADAMARD_CUDA_CHECK(cudaEventDestroy(start));
    HADAMARD_CUDA_CHECK(cudaEventDestroy(stop));
    return timing;
}

KernelTiming launch_hadamard_wmma_fp16(
    const void* input,
    void* output,
    const void* hadamard_matrix_fp16,
    int rows,
    int dim,
    int repeats) {
    validate_dim(dim);
    if (dim % 16 != 0) {
        throw std::invalid_argument("WMMA path requires dim to be a multiple of 16");
    }
    cudaEvent_t start;
    cudaEvent_t stop;
    HADAMARD_CUDA_CHECK(cudaEventCreate(&start));
    HADAMARD_CUDA_CHECK(cudaEventCreate(&stop));
    const dim3 grid(dim / 16, (rows + 15) / 16);
    HADAMARD_CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        hadamard_wmma_fp16_kernel<<<grid, 128>>>(
            reinterpret_cast<const half*>(input),
            reinterpret_cast<half*>(output),
            reinterpret_cast<const half*>(hadamard_matrix_fp16),
            rows,
            dim);
    }
    HADAMARD_CUDA_CHECK(cudaGetLastError());
    KernelTiming timing = time_repeated(start, stop, repeats);
    HADAMARD_CUDA_CHECK(cudaEventDestroy(start));
    HADAMARD_CUDA_CHECK(cudaEventDestroy(stop));
    return timing;
}

KernelTiming launch_quantize_int4(
    const void* transformed,
    std::uint8_t* packed,
    float* scales,
    int rows,
    int dim,
    DType dtype,
    int repeats) {
    validate_dim(dim);
    cudaEvent_t start;
    cudaEvent_t stop;
    HADAMARD_CUDA_CHECK(cudaEventCreate(&start));
    HADAMARD_CUDA_CHECK(cudaEventCreate(&stop));
    HADAMARD_CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        quantize_int4_kernel<<<rows, dim, dim * sizeof(float)>>>(transformed, packed, scales, rows, dim, dtype);
    }
    HADAMARD_CUDA_CHECK(cudaGetLastError());
    KernelTiming timing = time_repeated(start, stop, repeats);
    HADAMARD_CUDA_CHECK(cudaEventDestroy(start));
    HADAMARD_CUDA_CHECK(cudaEventDestroy(stop));
    return timing;
}

KernelTiming launch_fused_hadamard_int4(
    const void* input,
    std::uint8_t* packed,
    float* scales,
    int rows,
    int dim,
    DType dtype,
    int repeats) {
    validate_dim(dim);
    cudaEvent_t start;
    cudaEvent_t stop;
    HADAMARD_CUDA_CHECK(cudaEventCreate(&start));
    HADAMARD_CUDA_CHECK(cudaEventCreate(&stop));
    HADAMARD_CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        fused_hadamard_int4_kernel2<<<rows, dim, 2 * dim * sizeof(float)>>>(input, packed, scales, rows, dim, dtype);
    }
    HADAMARD_CUDA_CHECK(cudaGetLastError());
    KernelTiming timing = time_repeated(start, stop, repeats);
    HADAMARD_CUDA_CHECK(cudaEventDestroy(start));
    HADAMARD_CUDA_CHECK(cudaEventDestroy(stop));
    return timing;
}

KernelTiming launch_quantize_e4m3(
    const void* transformed,
    std::uint8_t* encoded,
    int rows,
    int dim,
    DType dtype,
    int repeats) {
    validate_dim(dim);
    const int total = rows * dim;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    cudaEvent_t start;
    cudaEvent_t stop;
    HADAMARD_CUDA_CHECK(cudaEventCreate(&start));
    HADAMARD_CUDA_CHECK(cudaEventCreate(&stop));
    HADAMARD_CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        quantize_e4m3_kernel<<<grid, block>>>(transformed, encoded, rows, dim, dtype);
    }
    HADAMARD_CUDA_CHECK(cudaGetLastError());
    KernelTiming timing = time_repeated(start, stop, repeats);
    HADAMARD_CUDA_CHECK(cudaEventDestroy(start));
    HADAMARD_CUDA_CHECK(cudaEventDestroy(stop));
    return timing;
}

KernelTiming launch_fused_hadamard_e4m3(
    const void* input,
    std::uint8_t* encoded,
    int rows,
    int dim,
    DType dtype,
    int repeats) {
    validate_dim(dim);
    cudaEvent_t start;
    cudaEvent_t stop;
    HADAMARD_CUDA_CHECK(cudaEventCreate(&start));
    HADAMARD_CUDA_CHECK(cudaEventCreate(&stop));
    HADAMARD_CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        fused_hadamard_e4m3_kernel<<<rows, dim, dim * sizeof(float)>>>(input, encoded, rows, dim, dtype);
    }
    HADAMARD_CUDA_CHECK(cudaGetLastError());
    KernelTiming timing = time_repeated(start, stop, repeats);
    HADAMARD_CUDA_CHECK(cudaEventDestroy(start));
    HADAMARD_CUDA_CHECK(cudaEventDestroy(stop));
    return timing;
}

std::vector<std::uint16_t> make_hadamard_matrix_fp16(int dim) {
    validate_dim(dim);
    std::vector<float> matrix(static_cast<std::size_t>(dim) * dim, 1.0f);
    // 递归构造 Sylvester 型 Hadamard 矩阵：
    // H_2n = [[H_n, H_n], [H_n, -H_n]]。
    for (int stride = 1; stride < dim; stride <<= 1) {
        for (int r = 0; r < stride; ++r) {
            for (int c = 0; c < stride; ++c) {
                matrix[(r + stride) * dim + c] = matrix[r * dim + c];
                matrix[r * dim + c + stride] = matrix[r * dim + c];
                matrix[(r + stride) * dim + c + stride] = -matrix[r * dim + c];
            }
        }
    }
    const float norm = 1.0f / std::sqrt(static_cast<float>(dim));
    std::vector<std::uint16_t> fp16(matrix.size());
    for (std::size_t i = 0; i < matrix.size(); ++i) {
        const half h = __float2half_rn(matrix[i] * norm);
        std::uint16_t bits;
        std::memcpy(&bits, &h, sizeof(bits));
        fp16[i] = bits;
    }
    return fp16;
}

std::uint16_t float_to_bf16_bits(float value) {
    return float_to_bf16_bits_raw(value);
}

float bf16_bits_to_float(std::uint16_t bits) {
    return bf16_bits_to_float_raw(bits);
}

}  // namespace hadamard



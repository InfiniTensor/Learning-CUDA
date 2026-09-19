#include "low_precision/cuda_quant.hpp"

#include "low_precision/quant.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace low_precision {
namespace {

constexpr std::size_t kMxBlock = 32;
constexpr std::size_t kNvBlock = 16;
constexpr float kMxFp8Max = 448.0f;
constexpr float kNvFp4Max = 6.0f;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void check_kernel(const char* operation) {
    check_cuda(cudaGetLastError(), operation);
    check_cuda(cudaDeviceSynchronize(), operation);
}

void check_shape(std::size_t rows, std::size_t cols) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("matrix dimensions must be nonzero");
    }
}

std::size_t block_count(std::size_t cols, std::size_t block_size) {
    return (cols + block_size - 1) / block_size;
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        check_cuda(cudaMalloc(&pointer_, count * sizeof(T)), "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() { return pointer_; }

private:
    T* pointer_ = nullptr;
};

__device__ float decode_e2m1_device(unsigned code) {
    const unsigned magnitude_code = code & 0x07u;
    const unsigned exponent = magnitude_code >> 1u;
    const unsigned mantissa = magnitude_code & 0x01u;
    float value;
    if (exponent == 0) {
        value = static_cast<float>(mantissa) * 0.5f;
    } else {
        value = ldexpf(1.0f + static_cast<float>(mantissa) * 0.5f,
                       static_cast<int>(exponent) - 1);
    }
    return (code & 0x08u) != 0 ? -value : value;
}

__device__ float random_uniform_device(std::uint64_t seed) {
    std::uint64_t value = seed + 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    value ^= value >> 31u;
    return static_cast<float>(value >> 40u) * (1.0f / 16777216.0f);
}

__device__ unsigned encode_e2m1_device(float value,
                                       Rounding rounding,
                                       std::uint64_t random_seed) {
    const bool negative = signbit(value);
    const float magnitude = fminf(fabsf(value), kNvFp4Max);
    unsigned lower = 0;
    unsigned upper = 7;
    for (unsigned code = 0; code <= 7; ++code) {
        const float candidate = decode_e2m1_device(code);
        if (candidate <= magnitude) {
            lower = code;
        }
        if (candidate >= magnitude) {
            upper = code;
            break;
        }
    }
    unsigned selected = lower;
    const float lower_value = decode_e2m1_device(lower);
    const float upper_value = decode_e2m1_device(upper);
    if (rounding == Rounding::Stochastic && lower != upper && magnitude < kNvFp4Max) {
        const float probability = (magnitude - lower_value) / (upper_value - lower_value);
        selected = random_uniform_device(random_seed) < probability ? upper : lower;
    } else {
        const float lower_distance = magnitude - lower_value;
        const float upper_distance = upper_value - magnitude;
        if (upper_distance < lower_distance ||
            (upper_distance == lower_distance && (lower & 1u) != 0u)) {
            selected = upper;
        }
    }
    return negative ? selected | 0x08u : selected;
}

__device__ float decode_e4m3_magnitude_device(unsigned code) {
    const unsigned exponent = (code >> 3u) & 0x0fu;
    const unsigned mantissa = code & 0x07u;
    if (exponent == 0u) {
        if (mantissa == 0u) {
            return 0.0f;
        }
        unsigned highest_bit = 2u;
        while ((mantissa & (1u << highest_bit)) == 0u) {
            --highest_bit;
        }
        const unsigned float_exponent = highest_bit + 118u;
        const unsigned float_mantissa =
            (mantissa - (1u << highest_bit)) << (23u - highest_bit);
        return __int_as_float(static_cast<int>((float_exponent << 23u) |
                                                float_mantissa));
    }
    if (exponent == 15u && mantissa == 7u) {
        return __int_as_float(0x7fc00000);
    }
    const unsigned float_exponent = exponent + 120u;
    return __int_as_float(static_cast<int>((float_exponent << 23u) |
                                            (mantissa << 20u)));
}

__device__ float decode_e4m3_device(unsigned code) {
    const float magnitude = decode_e4m3_magnitude_device(code & 0x7fu);
    return (code & 0x80u) != 0 ? -magnitude : magnitude;
}

__device__ unsigned encode_e4m3_device(float value,
                                       Rounding rounding = Rounding::NearestEven,
                                       std::uint64_t random_seed = 0) {
    const bool negative = signbit(value);
    const float magnitude = fminf(fabsf(value), kMxFp8Max);
    unsigned lower = 0;
    unsigned upper = 0x7e;
    for (unsigned code = 0; code <= 0x7e; ++code) {
        const float candidate = decode_e4m3_magnitude_device(code);
        if (candidate <= magnitude) {
            lower = code;
        }
        if (candidate >= magnitude) {
            upper = code;
            break;
        }
    }
    unsigned selected = lower;
    const float lower_value = decode_e4m3_magnitude_device(lower);
    const float upper_value = decode_e4m3_magnitude_device(upper);
    if (rounding == Rounding::Stochastic && lower != upper && magnitude < kMxFp8Max) {
        const float probability = (magnitude - lower_value) / (upper_value - lower_value);
        selected = random_uniform_device(random_seed) < probability ? upper : lower;
    } else {
        const float lower_distance = magnitude - lower_value;
        const float upper_distance = upper_value - magnitude;
        if (upper_distance < lower_distance ||
            (upper_distance == lower_distance && (lower & 1u) != 0u)) {
            selected = upper;
        }
    }
    return negative ? selected | 0x80u : selected;
}

__device__ float decode_e8m0_device(unsigned code) {
    if (code == 0u) {
        return __int_as_float(0x00400000);
    }
    return __int_as_float(static_cast<int>(code << 23u));
}

__device__ std::uint8_t encode_e8m0_device(float scale) {
    if (!(scale > 0.0f) || !isfinite(scale)) {
        return 127;
    }
    int exponent = 0;
    const float mantissa = frexpf(scale, &exponent);
    if (mantissa == 0.5f) {
        --exponent;
    }
    exponent = max(0, min(254, exponent + 127));
    return static_cast<std::uint8_t>(exponent);
}

__device__ float fp16_to_float_device(Fp16 value) {
    const unsigned sign = value.bits >> 15u;
    const unsigned exponent = (value.bits >> 10u) & 0x1fu;
    const unsigned mantissa = value.bits & 0x03ffu;
    const unsigned sign_bits = sign << 31u;
    if (exponent == 0u) {
        if (mantissa == 0u) {
            return __int_as_float(static_cast<int>(sign_bits));
        }
        unsigned highest_bit = 9u;
        while ((mantissa & (1u << highest_bit)) == 0u) {
            --highest_bit;
        }
        const unsigned float_exponent = highest_bit + 103u;
        const unsigned float_mantissa =
            (mantissa - (1u << highest_bit)) << (23u - highest_bit);
        return __int_as_float(static_cast<int>(sign_bits |
                                                (float_exponent << 23u) |
                                                float_mantissa));
    }
    if (exponent == 31u) {
        return __int_as_float(static_cast<int>(sign_bits |
                                                (0xffu << 23u) |
                                                (mantissa == 0u ? 0u : 0x400000u)));
    }
    const unsigned float_exponent = exponent + 112u;
    return __int_as_float(static_cast<int>(sign_bits |
                                            (float_exponent << 23u) |
                                            (mantissa << 13u)));
}

__device__ Fp16 fp16_from_float_device(float value) {
    return {__half_as_ushort(__float2half_rn(value))};
}

__device__ Bf16 bf16_from_float_device(float value) {
    unsigned bits = __float_as_uint(value);
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return {static_cast<std::uint16_t>(bits >> 16u)};
}

__device__ float load_input_device(float value) {
    return value;
}

__device__ float load_input_device(Fp16 value) {
    return fp16_to_float_device(value);
}

__device__ void store_output_device(float value, float* output) {
    *output = value;
}

__device__ void store_output_device(float value, Fp16* output) {
    *output = fp16_from_float_device(value);
}

__device__ void store_output_device(float value, Bf16* output) {
    *output = bf16_from_float_device(value);
}

unsigned element_grid(std::size_t element_count) {
    return static_cast<unsigned>((element_count + 255) / 256);
}

__device__ void reduce_max_32(float* values) {
    for (unsigned stride = 16; stride > 0; stride >>= 1u) {
        __syncthreads();
        if (threadIdx.x < stride) {
            values[threadIdx.x] = fmaxf(values[threadIdx.x], values[threadIdx.x + stride]);
        }
    }
    __syncthreads();
}

template <typename Input>
__global__ void block_amax_kernel(const Input* input,
                                  float* block_amax,
                                  std::size_t rows,
                                  std::size_t cols,
                                  std::size_t block_size,
                                  std::size_t blocks_per_row) {
    const std::size_t block_index = blockIdx.x;
    if (block_index >= rows * blocks_per_row) {
        return;
    }
    const std::size_t row = block_index / blocks_per_row;
    const std::size_t block = block_index % blocks_per_row;
    const std::size_t begin = block * block_size;
    const std::size_t col = begin + threadIdx.x;
    float local = 0.0f;
    if (col < cols && threadIdx.x < block_size) {
        local = fabsf(load_input_device(input[row * cols + col]));
    }
    __shared__ float shared[32];
    shared[threadIdx.x] = local;
    reduce_max_32(shared);
    if (threadIdx.x == 0) {
        block_amax[block_index] = shared[0];
    }
}

__global__ void encode_mxfp8_scales_kernel(const float* block_amax,
                                           std::uint8_t* scales,
                                           std::size_t block_count_total) {
    const std::size_t block_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                                    threadIdx.x;
    if (block_index >= block_count_total) {
        return;
    }
    const float block_max = block_amax[block_index];
    const float requested_scale = block_max == 0.0f ? 1.0f : block_max / kMxFp8Max;
    scales[block_index] = encode_e8m0_device(requested_scale);
}

template <typename Input>
__global__ void nvfp4_block_amax_kernel(const Input* input,
                                        float* block_amax,
                                        float* global_amax,
                                        std::size_t rows,
                                        std::size_t cols,
                                        std::size_t blocks_per_row) {
    constexpr unsigned kWarpsPerBlock = 8;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5u;
    const std::size_t block_index = static_cast<std::size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const std::size_t block_count_total = rows * blocks_per_row;
    float local = 0.0f;
    if (block_index < block_count_total && lane < kNvBlock) {
        const std::size_t row = block_index / blocks_per_row;
        const std::size_t block = block_index % blocks_per_row;
        const std::size_t col = block * kNvBlock + lane;
        if (col < cols) {
            local = fabsf(load_input_device(input[row * cols + col]));
        }
    }
    for (unsigned offset = 16; offset > 0; offset >>= 1u) {
        local = fmaxf(local, __shfl_down_sync(0xffffffffu, local, offset));
    }
    if (lane == 0 && block_index < block_count_total) {
        block_amax[block_index] = local;
        atomicMax(reinterpret_cast<unsigned int*>(global_amax), __float_as_uint(local));
    }
}

__global__ void select_nvfp4_global_scale_kernel(const float* global_amax,
                                                 float* global_scale) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const float amax = *global_amax;
        *global_scale = amax == 0.0f ? 1.0f : amax / (kMxFp8Max * kNvFp4Max);
    }
}

__global__ void encode_nvfp4_scales_kernel(const float* block_amax,
                                           const float* global_scale,
                                           std::uint8_t* block_scales,
                                           std::size_t block_count_total,
                                           Rounding rounding,
                                           std::uint64_t seed) {
    const std::size_t block_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (block_index >= block_count_total) {
        return;
    }
    const float selected_global_scale = *global_scale;
    const float block_max = block_amax[block_index];
    const float requested_scale = block_max == 0.0f
                                      ? 1.0f
                                      : fmaxf((block_max / kNvFp4Max) / selected_global_scale,
                                              decode_e4m3_magnitude_device(0x01));
    block_scales[block_index] = static_cast<std::uint8_t>(
        encode_e4m3_device(requested_scale, rounding, seed ^ (block_index * 0x9e3779b97f4a7c15ull)));
}

template <typename Input>
__global__ void quantize_mxfp8_kernel(const Input* input,
                                      std::uint8_t* values,
                                      const std::uint8_t* scales,
                                      std::size_t rows,
                                      std::size_t cols,
                                      std::size_t blocks_per_row,
                                      Rounding rounding,
                                      std::uint64_t seed) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * cols) {
        return;
    }
    const std::size_t row = index / cols;
    const std::size_t col = index % cols;
    const std::size_t block = col / kMxBlock;
    const float scale = decode_e8m0_device(scales[row * blocks_per_row + block]);
    values[index] = static_cast<std::uint8_t>(
        encode_e4m3_device(load_input_device(input[index]) / scale, rounding,
                           seed ^ (index * 0x9e3779b97f4a7c15ull)));
}

template <typename Output>
__global__ void dequantize_mxfp8_kernel(const std::uint8_t* values,
                                        const std::uint8_t* scales,
                                        Output* output,
                                        std::size_t rows,
                                        std::size_t cols,
                                        std::size_t blocks_per_row) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * cols) {
        return;
    }
    const std::size_t row = index / cols;
    const std::size_t col = index % cols;
    const std::size_t block = col / kMxBlock;
    store_output_device(
        decode_e4m3_device(values[index]) * decode_e8m0_device(scales[row * blocks_per_row + block]),
        &output[index]);
}

template <typename Input>
__global__ void quantize_nvfp4_kernel(const Input* input,
                                      std::uint8_t* values,
                                      const std::uint8_t* block_scales,
                                      const float* global_scale,
                                      std::size_t rows,
                                      std::size_t cols,
                                      std::size_t blocks_per_row,
                                      Rounding rounding,
                                      std::uint64_t seed) {
    const std::size_t pair = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t element_count = rows * cols;
    if (pair >= (element_count + 1) / 2) {
        return;
    }
    const std::size_t first = pair * 2;
    const std::size_t second = first + 1;
    const std::size_t first_row = first / cols;
    const std::size_t first_col = first % cols;
    const std::size_t first_block = first_col / kNvBlock;
    const float selected_global_scale = *global_scale;
    const float first_scale = selected_global_scale *
                              decode_e4m3_device(block_scales[first_row * blocks_per_row + first_block]);
    unsigned packed = encode_e2m1_device(
        load_input_device(input[first]) / first_scale, rounding,
        seed ^ (first * 0x9e3779b97f4a7c15ull));
    if (second < element_count) {
        const std::size_t second_row = second / cols;
        const std::size_t second_col = second % cols;
        const std::size_t second_block = second_col / kNvBlock;
        const float second_scale = selected_global_scale *
                                   decode_e4m3_device(block_scales[second_row * blocks_per_row + second_block]);
        packed |= encode_e2m1_device(
                      load_input_device(input[second]) / second_scale, rounding,
                      seed ^ (second * 0x9e3779b97f4a7c15ull))
                  << 4u;
    }
    values[pair] = static_cast<std::uint8_t>(packed);
}

template <typename Output>
__global__ void dequantize_nvfp4_kernel(const std::uint8_t* values,
                                        const std::uint8_t* block_scales,
                                        float global_scale,
                                        Output* output,
                                        std::size_t rows,
                                        std::size_t cols,
                                        std::size_t blocks_per_row) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * cols) {
        return;
    }
    const std::size_t row = index / cols;
    const std::size_t col = index % cols;
    const std::size_t block = col / kNvBlock;
    const unsigned code = (index & 1u) == 0 ? values[index / 2] & 0x0fu : values[index / 2] >> 4u;
    store_output_device(
        decode_e2m1_device(code) * global_scale *
            decode_e4m3_device(block_scales[row * blocks_per_row + block]),
        &output[index]);
}

template <typename Input>
void cuda_quantize_mxfp8_impl(const Input* d_input,
                              std::uint8_t* d_values,
                              std::uint8_t* d_scales,
                              std::size_t rows,
                              std::size_t cols,
                              Rounding rounding,
                              std::uint64_t seed) {
    check_shape(rows, cols);
    const std::size_t blocks_per_row = block_count(cols, kMxBlock);
    const std::size_t block_count_total = rows * blocks_per_row;
    DeviceBuffer<float> d_amax(block_count_total);
    block_amax_kernel<Input><<<static_cast<unsigned>(block_count_total), 32>>>(
        d_input, d_amax.get(), rows, cols, kMxBlock, blocks_per_row);
    check_kernel("MXFP8 block amax");
    encode_mxfp8_scales_kernel<<<static_cast<unsigned>((block_count_total + 255) / 256), 256>>>(
        d_amax.get(), d_scales, block_count_total);
    check_kernel("MXFP8 scale encode");

    const std::size_t element_count = rows * cols;
    quantize_mxfp8_kernel<Input><<<element_grid(element_count), 256>>>(
        d_input, d_values, d_scales, rows, cols, blocks_per_row, rounding, seed);
    check_kernel("MXFP8 quantize");
}

template <typename Output>
void cuda_dequantize_mxfp8_impl(const std::uint8_t* d_values,
                                const std::uint8_t* d_scales,
                                Output* d_output,
                                std::size_t rows,
                                std::size_t cols) {
    check_shape(rows, cols);
    const std::size_t blocks_per_row = block_count(cols, kMxBlock);
    const std::size_t element_count = rows * cols;
    dequantize_mxfp8_kernel<Output><<<element_grid(element_count), 256>>>(
        d_values, d_scales, d_output, rows, cols, blocks_per_row);
    check_kernel("MXFP8 dequantize");
}

template <typename Input>
void cuda_quantize_nvfp4_impl(const Input* d_input,
                              std::uint8_t* d_values,
                              std::uint8_t* d_block_scales,
                              std::size_t rows,
                              std::size_t cols,
                              float* global_scale,
                              Rounding rounding,
                              std::uint64_t seed) {
    check_shape(rows, cols);
    const std::size_t blocks_per_row = block_count(cols, kNvBlock);
    const std::size_t block_count_total = rows * blocks_per_row;
    DeviceBuffer<float> d_amax(block_count_total);
    DeviceBuffer<float> d_global_amax(1);
    DeviceBuffer<float> d_global_scale(1);
    check_cuda(cudaMemset(d_global_amax.get(), 0, sizeof(float)), "NVFP4 global amax reset");
    const unsigned reduction_blocks = static_cast<unsigned>((block_count_total + 7) / 8);
    nvfp4_block_amax_kernel<Input><<<reduction_blocks, 256>>>(
        d_input, d_amax.get(), d_global_amax.get(), rows, cols, blocks_per_row);
    check_kernel("NVFP4 block amax");
    select_nvfp4_global_scale_kernel<<<1, 1>>>(d_global_amax.get(), d_global_scale.get());
    check_kernel("NVFP4 global scale select");

    encode_nvfp4_scales_kernel<<<static_cast<unsigned>((block_count_total + 255) / 256), 256>>>(
        d_amax.get(), d_global_scale.get(), d_block_scales, block_count_total, rounding, seed);
    check_kernel("NVFP4 block scale encode");

    const std::size_t packed_count = (rows * cols + 1) / 2;
    quantize_nvfp4_kernel<Input><<<element_grid(packed_count), 256>>>(
        d_input, d_values, d_block_scales, d_global_scale.get(), rows, cols, blocks_per_row,
        rounding, seed);
    check_kernel("NVFP4 quantize");
    if (global_scale != nullptr) {
        check_cuda(cudaMemcpy(global_scale, d_global_scale.get(), sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "NVFP4 global scale copy");
    }
}

template <typename Output>
void cuda_dequantize_nvfp4_impl(const std::uint8_t* d_values,
                                const std::uint8_t* d_block_scales,
                                float global_scale,
                                Output* d_output,
                                std::size_t rows,
                                std::size_t cols) {
    check_shape(rows, cols);
    const std::size_t blocks_per_row = block_count(cols, kNvBlock);
    const std::size_t element_count = rows * cols;
    dequantize_nvfp4_kernel<Output><<<element_grid(element_count), 256>>>(
        d_values, d_block_scales, global_scale, d_output, rows, cols, blocks_per_row);
    check_kernel("NVFP4 dequantize");
}

}  // namespace

void cuda_quantize_mxfp8(const float* d_input,
                         std::uint8_t* d_values,
                         std::uint8_t* d_scales,
                         std::size_t rows,
                         std::size_t cols,
                         Rounding rounding,
                         std::uint64_t seed) {
    cuda_quantize_mxfp8_impl(d_input, d_values, d_scales, rows, cols, rounding, seed);
}

void cuda_quantize_mxfp8(const Fp16* d_input,
                         std::uint8_t* d_values,
                         std::uint8_t* d_scales,
                         std::size_t rows,
                         std::size_t cols,
                         Rounding rounding,
                         std::uint64_t seed) {
    cuda_quantize_mxfp8_impl(d_input, d_values, d_scales, rows, cols, rounding, seed);
}

void cuda_dequantize_mxfp8(const std::uint8_t* d_values,
                           const std::uint8_t* d_scales,
                           float* d_output,
                           std::size_t rows,
                           std::size_t cols) {
    cuda_dequantize_mxfp8_impl(d_values, d_scales, d_output, rows, cols);
}

void cuda_dequantize_mxfp8(const std::uint8_t* d_values,
                           const std::uint8_t* d_scales,
                           Fp16* d_output,
                           std::size_t rows,
                           std::size_t cols) {
    cuda_dequantize_mxfp8_impl(d_values, d_scales, d_output, rows, cols);
}

void cuda_dequantize_mxfp8(const std::uint8_t* d_values,
                           const std::uint8_t* d_scales,
                           Bf16* d_output,
                           std::size_t rows,
                           std::size_t cols) {
    cuda_dequantize_mxfp8_impl(d_values, d_scales, d_output, rows, cols);
}

void cuda_quantize_nvfp4(const float* d_input,
                         std::uint8_t* d_values,
                         std::uint8_t* d_block_scales,
                         std::size_t rows,
                         std::size_t cols,
                         float* global_scale,
                         Rounding rounding,
                         std::uint64_t seed) {
    cuda_quantize_nvfp4_impl(d_input, d_values, d_block_scales,
                             rows, cols, global_scale, rounding, seed);
}

void cuda_quantize_nvfp4(const Fp16* d_input,
                         std::uint8_t* d_values,
                         std::uint8_t* d_block_scales,
                         std::size_t rows,
                         std::size_t cols,
                         float* global_scale,
                         Rounding rounding,
                         std::uint64_t seed) {
    cuda_quantize_nvfp4_impl(d_input, d_values, d_block_scales,
                             rows, cols, global_scale, rounding, seed);
}

void cuda_dequantize_nvfp4(const std::uint8_t* d_values,
                           const std::uint8_t* d_block_scales,
                           float global_scale,
                           float* d_output,
                           std::size_t rows,
                           std::size_t cols) {
    cuda_dequantize_nvfp4_impl(d_values, d_block_scales, global_scale,
                               d_output, rows, cols);
}

void cuda_dequantize_nvfp4(const std::uint8_t* d_values,
                           const std::uint8_t* d_block_scales,
                           float global_scale,
                           Fp16* d_output,
                           std::size_t rows,
                           std::size_t cols) {
    cuda_dequantize_nvfp4_impl(d_values, d_block_scales, global_scale,
                               d_output, rows, cols);
}

void cuda_dequantize_nvfp4(const std::uint8_t* d_values,
                           const std::uint8_t* d_block_scales,
                           float global_scale,
                           Bf16* d_output,
                           std::size_t rows,
                           std::size_t cols) {
    cuda_dequantize_nvfp4_impl(d_values, d_block_scales, global_scale,
                               d_output, rows, cols);
}

}  // namespace low_precision

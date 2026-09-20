#include "hadamard_musa.muh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace hadamard_musa {
namespace {

__host__ __device__ inline std::uint32_t float_as_bits(float value) {
#ifdef __MUSA_ARCH__
    return __float_as_uint(value);
#else
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
#endif
}

__host__ __device__ inline float bits_as_float(std::uint32_t bits) {
#ifdef __MUSA_ARCH__
    return __uint_as_float(bits);
#else
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
#endif
}

// 软件 FP16 解码，避免依赖 NVIDIA 的 half 类型和头文件。
__host__ __device__ inline float fp16_bits_to_float_raw(std::uint16_t bits) {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
    const std::uint32_t exponent = (bits >> 10) & 0x1fu;
    const std::uint32_t mantissa = bits & 0x03ffu;
    if (exponent == 0) {
        if (mantissa == 0) {
            return bits_as_float(sign);
        }
        return (sign == 0 ? 1.0f : -1.0f) *
            std::ldexp(static_cast<float>(mantissa), -24);
    }
    if (exponent == 31) {
        if (mantissa == 0) {
            return bits_as_float(sign | 0x7f800000u);
        }
        return bits_as_float(sign | 0x7fc00000u);
    }
    const std::uint32_t word =
        sign | ((exponent - 15u + 127u) << 23) | (mantissa << 13);
    return bits_as_float(word);
}

// 软件 FP16 编码，采用 round-to-nearest-even。
__host__ __device__ inline std::uint16_t float_to_fp16_bits_raw(float value) {
    const std::uint32_t word = float_as_bits(value);
    const std::uint32_t sign = (word >> 16) & 0x8000u;
    const std::uint32_t exponent = (word >> 23) & 0xffu;
    const std::uint32_t mantissa = word & 0x7fffffu;

    if (exponent == 255) {
        return static_cast<std::uint16_t>(
            sign | 0x7c00u | (mantissa != 0 ? 0x0200u : 0u));
    }

    const int unbiased = static_cast<int>(exponent) - 127;
    if (unbiased > 15) {
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    if (unbiased >= -14) {
        std::uint32_t half_exponent = static_cast<std::uint32_t>(unbiased + 15);
        std::uint32_t half_mantissa = mantissa >> 13;
        const std::uint32_t remainder = mantissa & 0x1fffu;
        if (remainder > 0x1000u ||
            (remainder == 0x1000u && (half_mantissa & 1u) != 0)) {
            ++half_mantissa;
            if (half_mantissa == 0x400u) {
                half_mantissa = 0;
                ++half_exponent;
            }
        }
        if (half_exponent >= 31) {
            return static_cast<std::uint16_t>(sign | 0x7c00u);
        }
        return static_cast<std::uint16_t>(
            sign | (half_exponent << 10) | half_mantissa);
    }

    if (unbiased >= -24) {
        const int shift = -unbiased - 14;
        std::uint32_t mantissa_with_hidden = mantissa | 0x800000u;
        std::uint32_t half_mantissa = mantissa_with_hidden >> (shift + 13);
        const std::uint32_t remainder_mask = (1u << (shift + 13)) - 1u;
        const std::uint32_t remainder = mantissa_with_hidden & remainder_mask;
        const std::uint32_t halfway = 1u << (shift + 12);
        if (remainder > halfway ||
            (remainder == halfway && (half_mantissa & 1u) != 0)) {
            ++half_mantissa;
        }
        return static_cast<std::uint16_t>(sign | half_mantissa);
    }
    return static_cast<std::uint16_t>(sign);
}

__host__ __device__ inline float bf16_bits_to_float_raw(std::uint16_t bits) {
    return bits_as_float(static_cast<std::uint32_t>(bits) << 16);
}

__host__ __device__ inline std::uint16_t float_to_bf16_bits_raw(float value) {
    const std::uint32_t word = float_as_bits(value);
    const std::uint32_t lsb = (word >> 16) & 1u;
    return static_cast<std::uint16_t>((word + 0x7fffu + lsb) >> 16);
}

__device__ inline float load_value(const void* ptr, int index, DType dtype) {
    const auto* values = reinterpret_cast<const std::uint16_t*>(ptr);
    return dtype == DType::FP16
        ? fp16_bits_to_float_raw(values[index])
        : bf16_bits_to_float_raw(values[index]);
}

__device__ inline void store_value(
    void* ptr,
    int index,
    DType dtype,
    float value) {
    auto* values = reinterpret_cast<std::uint16_t*>(ptr);
    values[index] = dtype == DType::FP16
        ? float_to_fp16_bits_raw(value)
        : float_to_bf16_bits_raw(value);
}

__device__ inline float round_to_output_dtype(float value, DType dtype) {
    return dtype == DType::FP16
        ? fp16_bits_to_float_raw(float_to_fp16_bits_raw(value))
        : bf16_bits_to_float_raw(float_to_bf16_bits_raw(value));
}

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

    x = fminf(x, 240.0f);
    int exponent = 0;
    const float mantissa = frexpf(x, &exponent);
    int biased_exponent = exponent - 1 + 7;
    if (biased_exponent <= 0) {
        int subnormal = static_cast<int>(lrintf(x * 512.0f));
        subnormal = max(0, min(7, subnormal));
        return sign | static_cast<std::uint8_t>(subnormal);
    }

    const float normalized = mantissa * 2.0f - 1.0f;
    int encoded_mantissa = static_cast<int>(lrintf(normalized * 8.0f));
    if (encoded_mantissa == 8) {
        encoded_mantissa = 0;
        ++biased_exponent;
    }
    if (biased_exponent >= 15) {
        biased_exponent = 14;
        encoded_mantissa = 7;
    }
    return sign | static_cast<std::uint8_t>(
        (biased_exponent << 3) | encoded_mantissa);
}

__device__ inline void fht_in_shared(float* values, int dim, int tid) {
    for (int stride = 1; stride < dim; stride <<= 1) {
        if ((tid & stride) == 0) {
            const float a = values[tid];
            const float b = values[tid + stride];
            values[tid] = a + b;
            values[tid + stride] = a - b;
        }
        __syncthreads();
    }
}

__global__ void hadamard_shared_musa_kernel(
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

    values[tid] = load_value(input, row * dim + tid, dtype);
    __syncthreads();
    fht_in_shared(values, dim, tid);
    store_value(
        output,
        row * dim + tid,
        dtype,
        values[tid] * rsqrtf(static_cast<float>(dim)));
}

__global__ void quantize_int4_musa_kernel(
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

    scratch[tid] = fabsf(load_value(transformed, row * dim + tid, dtype));
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
        const int q0 = quantize_s4(
            load_value(transformed, row * dim + 2 * tid, dtype), scale);
        const int q1 = quantize_s4(
            load_value(transformed, row * dim + 2 * tid + 1, dtype), scale);
        packed[row * (dim / 2) + tid] =
            static_cast<std::uint8_t>(q0 | (q1 << 4));
    }
}

__global__ void fused_hadamard_int4_musa_kernel(
    const void* input,
    std::uint8_t* packed,
    float* scales,
    int rows,
    int dim,
    DType dtype) {
    extern __shared__ float shared[];
    float* values = shared;
    float* reduction = shared + dim;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows || tid >= dim) {
        return;
    }

    values[tid] = load_value(input, row * dim + tid, dtype);
    __syncthreads();
    fht_in_shared(values, dim, tid);

    values[tid] = round_to_output_dtype(
        values[tid] * rsqrtf(static_cast<float>(dim)), dtype);
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
        packed[row * (dim / 2) + tid] =
            static_cast<std::uint8_t>(q0 | (q1 << 4));
    }
}

__global__ void quantize_e4m3_musa_kernel(
    const void* transformed,
    std::uint8_t* encoded,
    int rows,
    int dim,
    DType dtype) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < rows * dim) {
        encoded[index] = encode_e4m3(load_value(transformed, index, dtype));
    }
}

__global__ void fused_hadamard_e4m3_musa_kernel(
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
    fht_in_shared(values, dim, tid);
    encoded[row * dim + tid] = encode_e4m3(round_to_output_dtype(
        values[tid] * rsqrtf(static_cast<float>(dim)), dtype));
}

void validate_dim(int dim) {
    if (dim < 1 || dim > 256 || (dim & (dim - 1)) != 0) {
        throw std::invalid_argument("head_dim 必须是 [1, 256] 范围内的 2 的幂");
    }
}

KernelTiming finish_timing(
    musaEvent_t start,
    musaEvent_t stop,
    int repeats) {
    HADAMARD_MUSA_CHECK(musaEventRecord(stop));
    HADAMARD_MUSA_CHECK(musaEventSynchronize(stop));
    float milliseconds = 0.0f;
    HADAMARD_MUSA_CHECK(musaEventElapsedTime(&milliseconds, start, stop));
    KernelTiming result;
    result.milliseconds = milliseconds / static_cast<float>(std::max(1, repeats));
    return result;
}

template <typename Launch>
KernelTiming run_timed(int repeats, Launch launch) {
    musaEvent_t start;
    musaEvent_t stop;
    HADAMARD_MUSA_CHECK(musaEventCreate(&start));
    HADAMARD_MUSA_CHECK(musaEventCreate(&stop));
    HADAMARD_MUSA_CHECK(musaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        launch();
    }
    HADAMARD_MUSA_CHECK(musaGetLastError());
    KernelTiming result = finish_timing(start, stop, repeats);
    HADAMARD_MUSA_CHECK(musaEventDestroy(start));
    HADAMARD_MUSA_CHECK(musaEventDestroy(stop));
    return result;
}

}  // namespace

const char* dtype_name(DType dtype) {
    return dtype == DType::FP16 ? "fp16" : "bf16";
}

std::size_t tensor_bytes(int rows, int dim) {
    return static_cast<std::size_t>(rows) * dim * sizeof(std::uint16_t);
}

std::size_t int4_packed_bytes(int rows, int dim) {
    return static_cast<std::size_t>(rows) * ((dim + 1) / 2);
}

void check_musa_status(
    musaError_t status,
    const char* expression,
    const char* file,
    int line) {
    if (status != musaSuccess) {
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) +
            " MUSA 运行时调用失败: " + expression + " -> " +
            musaGetErrorString(status));
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
    return run_timed(repeats, [&] {
        hadamard_shared_musa_kernel<<<rows, dim, dim * sizeof(float)>>>(
            input, output, rows, dim, dtype);
    });
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
    return run_timed(repeats, [&] {
        quantize_int4_musa_kernel<<<rows, dim, dim * sizeof(float)>>>(
            transformed, packed, scales, rows, dim, dtype);
    });
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
    return run_timed(repeats, [&] {
        fused_hadamard_int4_musa_kernel<<<rows, dim, 2 * dim * sizeof(float)>>>(
            input, packed, scales, rows, dim, dtype);
    });
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
    return run_timed(repeats, [&] {
        quantize_e4m3_musa_kernel<<<grid, block>>>(
            transformed, encoded, rows, dim, dtype);
    });
}

KernelTiming launch_fused_hadamard_e4m3(
    const void* input,
    std::uint8_t* encoded,
    int rows,
    int dim,
    DType dtype,
    int repeats) {
    validate_dim(dim);
    return run_timed(repeats, [&] {
        fused_hadamard_e4m3_musa_kernel<<<rows, dim, dim * sizeof(float)>>>(
            input, encoded, rows, dim, dtype);
    });
}

std::uint16_t float_to_fp16_bits(float value) {
    return float_to_fp16_bits_raw(value);
}

float fp16_bits_to_float(std::uint16_t bits) {
    return fp16_bits_to_float_raw(bits);
}

std::uint16_t float_to_bf16_bits(float value) {
    return float_to_bf16_bits_raw(value);
}

float bf16_bits_to_float(std::uint16_t bits) {
    return bf16_bits_to_float_raw(bits);
}

}  // namespace hadamard_musa

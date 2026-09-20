#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace hadamard {

// 项目中只处理 16 bit 输入/输出。BF16 以 uint16_t 位模式保存，
// 这样基础实现不依赖硬件原生 BF16 指令。
enum class DType : int {
    FP16 = 0,
    BF16 = 1,
};

// 所有 launch_* 接口返回单次 kernel 的平均耗时，单位为 ms。
struct KernelTiming {
    float milliseconds = 0.0f;
};

// INT4 输出采用真实 packed 布局：每个 uint8_t 保存两个 signed 4-bit 元素。
struct Int4Quantized {
    std::uint8_t* packed = nullptr;
    float* scales = nullptr;
};

const char* dtype_name(DType dtype);
std::size_t element_size(DType dtype);
std::size_t tensor_bytes(int rows, int dim, DType dtype);
std::size_t int4_packed_bytes(int rows, int dim);

void check_cuda_status(cudaError_t status, const char* expr, const char* file, int line);

#define HADAMARD_CUDA_CHECK(expr) \
    ::hadamard::check_cuda_status((expr), #expr, __FILE__, __LINE__)

KernelTiming launch_hadamard_shared(
    const void* input,
    void* output,
    int rows,
    int dim,
    DType dtype,
    int repeats = 1);

// Tensor Core 对照路径：仅支持 FP16，把 Hadamard 变换写成 X * H_scaled。
KernelTiming launch_hadamard_wmma_fp16(
    const void* input,
    void* output,
    const void* hadamard_matrix_fp16,
    int rows,
    int dim,
    int repeats = 1);

KernelTiming launch_quantize_int4(
    const void* transformed,
    std::uint8_t* packed,
    float* scales,
    int rows,
    int dim,
    DType dtype,
    int repeats = 1);

// fused 版本在一个 kernel 内完成 Hadamard 和量化，用于验证融合后的字节结果。
KernelTiming launch_fused_hadamard_int4(
    const void* input,
    std::uint8_t* packed,
    float* scales,
    int rows,
    int dim,
    DType dtype,
    int repeats = 1);

KernelTiming launch_quantize_e4m3(
    const void* transformed,
    std::uint8_t* encoded,
    int rows,
    int dim,
    DType dtype,
    int repeats = 1);

KernelTiming launch_fused_hadamard_e4m3(
    const void* input,
    std::uint8_t* encoded,
    int rows,
    int dim,
    DType dtype,
    int repeats = 1);

std::vector<std::uint16_t> make_hadamard_matrix_fp16(int dim);
std::uint16_t float_to_bf16_bits(float value);
float bf16_bits_to_float(std::uint16_t bits);

}  // namespace hadamard


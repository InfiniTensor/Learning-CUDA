#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string_view>
#include <vector>

#include "quant_dequant/quantize.hpp"

namespace {

using quant_dequant::DequantizationConfig;
using quant_dequant::DType;
using quant_dequant::HostTensor;
using quant_dequant::QuantFormat;
using quant_dequant::QuantizationConfig;
using quant_dequant::RoundingMode;
using quant_dequant::ScaleMode;
using quant_dequant::TensorDesc;
using quant_dequant::kNvfp4BlockSize;

/** CUDA device 查询的三态结果，避免把 runtime 故障误作可跳过的无硬件环境。 */
enum class CudaAvailability {
    /** 当前进程可以使用至少一个 CUDA device。 */
    kAvailable,

    /** 没有 device 或 driver，测试应明确跳过。 */
    kUnavailable,

    /** CUDA runtime 返回了其他应报告的错误。 */
    kError,
};

/**
 * @brief 判断当前测试进程是否能实际启动 CUDA kernel。
 *
 * @return 可执行、可跳过和应报告失败三种状态之一。
 */
[[nodiscard]] CudaAvailability query_cuda_availability() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver) {
        std::cout << "CUDA runtime 不可用，跳过 NVFP4 CUDA 反量化对照测试。\n";
        return CudaAvailability::kUnavailable;
    }
    if (status != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount 失败：" << cudaGetErrorString(status) << '\n';
        return CudaAvailability::kError;
    }
    if (device_count == 0) {
        std::cout << "未发现 CUDA device，跳过 NVFP4 CUDA 反量化对照测试。\n";
        return CudaAvailability::kUnavailable;
    }

    return CudaAvailability::kAvailable;
}

/**
 * @brief 构造跨行 packed byte、16 元素 tail block 和奇数总元素数的测试输入。
 *
 * 3×35 同时保证每行有三个 16 元素 scale block、相邻线性元素可跨行配对，并让
 * 总元素 105 为奇数，以测试最后 payload byte 未使用的高 nibble 不会写越界。
 *
 * @return 有限的 row-major FP32 输入。
 */
[[nodiscard]] HostTensor make_input() {
    constexpr std::uint64_t kRows = 3U;
    constexpr std::uint64_t kCols = 35U;
    HostTensor input{
        .desc = TensorDesc{kRows, kCols, DType::kFloat32},
        .values = std::vector<float>(static_cast<std::size_t>(kRows * kCols), 0.0F),
    };

    for (std::size_t index = 0U; index < input.values.size(); ++index) {
        const int bucket = static_cast<int>((index * 17U + 5U) % 47U) - 23;
        input.values[index] = static_cast<float>(bucket) * 0.375F;
    }
    input.values[0U] = 6.0F;
    input.values[15U] = -6.0F;
    input.values[16U] = 0.5F;
    input.values[34U] = -3.0F;
    input.values[35U] = 1.5F;
    input.values.back() = -0.5F;
    return input;
}

/**
 * @brief 以一种输出 dtype 完整比较 NVFP4 CPU 与 CUDA 反量化路径。
 *
 * 被测 CUDA 路径包括 QDWGT 三类 buffer 的 H2D、一个线程一个 packed byte 的
 * 解包、跨行 low/high nibble 的独立 scale 索引和 FP32 结果 D2H。
 *
 * @param input 用于 CPU reference 产生合法 NVFP4 QDWGT 内容的有限 host 张量。
 * @param output_type 请求返回描述记录的物理输出 dtype。
 * @param output_type_name 用于失败诊断的 dtype 文本。
 * @return CPU reference 和 CUDA 回传的描述及 FP32 数值逐项一致时返回 true。
 */
[[nodiscard]] bool compare_dequantization_with_cpu(
    const HostTensor& input,
    const DType output_type,
    const std::string_view output_type_name) {
    const QuantizationConfig quantization_config{
        .format = QuantFormat::kNvfp4,
        .block_size = kNvfp4BlockSize,
        .scale_mode = ScaleMode::kBlock,
        .rounding = RoundingMode::kNearest,
        .stochastic_seed = 0U,
    };
    const DequantizationConfig dequantization_config{
        .output_type = output_type,
    };

    try {
        const auto quantized = quant_dequant::quantize_reference(
            input, quantization_config);
        const HostTensor expected = quant_dequant::dequantize_reference(
            quantized, dequantization_config);
        const HostTensor actual = quant_dequant::dequantize_cuda(
            quantized, dequantization_config);
        if (expected.desc.num_rows != actual.desc.num_rows ||
            expected.desc.num_cols != actual.desc.num_cols ||
            expected.desc.dtype != actual.desc.dtype ||
            expected.values != actual.values) {
            std::cerr << "NVFP4 CUDA 反量化与 CPU reference 不一致：output_type="
                      << output_type_name << "。\n";
            return false;
        }
    } catch (const std::exception& error) {
        std::cerr << "NVFP4 CUDA 反量化对照意外失败：output_type="
                  << output_type_name << "，原因：" << error.what() << '\n';
        return false;
    }

    return true;
}

}  // namespace

/**
 * @brief 运行 NVFP4 CUDA 反量化与 CPU reference 的正确性对照。
 *
 * 三种输出请求共享一个 FP32 device kernel；测试检查数值结果始终与 reference
 * 位级一致，同时确认 HostTensor 描述保留 FP16、BF16、FP32 的目标物理类型。
 *
 * @return 有 CUDA device 时所有对照通过返回 0；无可用 CUDA runtime 时跳过并
 * 返回 0；其他错误返回 1。
 */
int run_nvfp4_dequantize_cuda_tests() {
    const CudaAvailability cuda_availability = query_cuda_availability();
    if (cuda_availability == CudaAvailability::kUnavailable) {
        return 0;
    }
    if (cuda_availability == CudaAvailability::kError) {
        return 1;
    }

    const HostTensor input = make_input();
    constexpr DType kOutputTypes[]{
        DType::kFloat16,
        DType::kBFloat16,
        DType::kFloat32,
    };
    constexpr std::string_view kOutputTypeNames[]{"fp16", "bf16", "fp32"};
    for (std::size_t index = 0U; index < std::size(kOutputTypes); ++index) {
        if (!compare_dequantization_with_cpu(
                input, kOutputTypes[index], kOutputTypeNames[index])) {
            return 1;
        }
    }

    return 0;
}

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
using quant_dequant::kMxfp8BlockSize;

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
 * 无 GPU 或缺少 driver 是本项目 CI/开发容器的正常情况；这些环境应跳过数值
 * 对照，而不是把“没有硬件”误报为反量化逻辑错误。
 *
 * @return 可执行、可跳过和应报告失败三种状态之一。
 */
[[nodiscard]] CudaAvailability query_cuda_availability() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver) {
        std::cout << "CUDA runtime 不可用，跳过 MXFP8 CUDA 反量化对照测试。\n";
        return CudaAvailability::kUnavailable;
    }

    if (status != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount 失败：" << cudaGetErrorString(status) << '\n';
        return CudaAvailability::kError;
    }

    if (device_count == 0) {
        std::cout << "未发现 CUDA device，跳过 MXFP8 CUDA 反量化对照测试。\n";
        return CudaAvailability::kUnavailable;
    }

    return CudaAvailability::kAvailable;
}

/**
 * @brief 构造带有两个完整 block 和一个 tail block 的有限 FP32 输入。
 *
 * 三行各有 35 列，因而 block-scale 模式每行需要两个 local scale，最后 3 个
 * 元素验证 `column / 32` 对 tail block 的索引。每一行刻意包含不同幅值，避免
 * 所有 block 都退化为同一个 E8M0 scale。
 *
 * @return 用于 CPU reference 生成合法 MXFP8 payload/scale 的 host 输入。
 */
[[nodiscard]] HostTensor make_input() {
    constexpr std::uint64_t kRows = 3U;
    constexpr std::uint64_t kCols = 35U;

    HostTensor input{};
    input.desc = TensorDesc{kRows, kCols, DType::kFloat32};
    input.values.resize(static_cast<std::size_t>(kRows * kCols), 0.0F);

    for (std::size_t index = 0U; index < input.values.size(); ++index) {
        const int bucket =
            static_cast<int>((index * 13U + 7U) % 41U) - 20;
        input.values[index] = static_cast<float>(bucket) * 0.25F;
    }

    input.values[0U] = 448.0F;
    input.values[31U] = -448.0F;
    input.values[32U] = 0x1.0p-9F;
    input.values[34U] = -0x1.0p-8F;
    input.values[35U] = 896.0F;
    input.values[70U] = -896.0F;
    return input;
}

/**
 * @brief 比较 CPU reference 与 CUDA 反量化得到的 host FP32 张量。
 *
 * CPU、CUDA 都调用相同的无状态 codec 规则；对同一 payload/scale 的结果应逐
 * 元素位级相同。因此这里用 `vector<float>` 的精确比较，能够同时发现 scale
 * 下标、payload 下标、D2H 长度和输出描述 dtype 的错误。
 *
 * @param expected CPU reference 反量化结果。
 * @param actual CUDA kernel 经 D2H 回传的结果。
 * @param scale_mode_name 用于失败诊断的 scale 模式名称。
 * @param output_type_name 用于失败诊断的输出类型名称。
 * @return 描述和每个 FP32 数值均一致时返回 true。
 */
[[nodiscard]] bool equal_dequantized_tensors(
    const HostTensor& expected,
    const HostTensor& actual,
    const std::string_view scale_mode_name,
    const std::string_view output_type_name) {
    if (expected.desc.num_rows != actual.desc.num_rows ||
        expected.desc.num_cols != actual.desc.num_cols ||
        expected.desc.dtype != actual.desc.dtype ||
        expected.values != actual.values) {
        std::cerr << "MXFP8 CUDA 反量化与 CPU reference 不一致：scale_mode="
                  << scale_mode_name << "，output_type=" << output_type_name
                  << "。\n";
        return false;
    }

    return true;
}

/**
 * @brief 用一种 scale 模式和一种输出类型执行完整 CPU/CUDA 反量化对照。
 *
 * 低精度输入由 CPU reference 量化产生，以确保 payload、E8M0 scale 和 metadata
 * 均符合 QDWGT 规范；被测对象是 `dequantize_cuda()` 的 H2D、kernel scale 索引、
 * E4M3/E8M0 解码、D2H 以及目标 dtype 描述传递。
 *
 * @param input 原始有限 FP32 数据，仅用于生成合法的 MXFP8 输入。
 * @param scale_mode tensor 或 block。
 * @param output_type 请求返回 HostTensor 描述记录的物理输出类型。
 * @param scale_mode_name 用于失败诊断的 scale 模式名称。
 * @param output_type_name 用于失败诊断的输出类型名称。
 * @return CPU reference 与 CUDA 输出完全一致时返回 true。
 */
[[nodiscard]] bool compare_dequantization_with_cpu(
    const HostTensor& input,
    const ScaleMode scale_mode,
    const DType output_type,
    const std::string_view scale_mode_name,
    const std::string_view output_type_name) {
    QuantizationConfig quantization_config{};
    quantization_config.format = QuantFormat::kMxfp8;
    quantization_config.block_size = kMxfp8BlockSize;
    quantization_config.scale_mode = scale_mode;
    quantization_config.rounding = RoundingMode::kNearest;

    DequantizationConfig dequantization_config{};
    dequantization_config.output_type = output_type;

    try {
        const auto quantized =
            quant_dequant::quantize_reference(input, quantization_config);
        const HostTensor expected =
            quant_dequant::dequantize_reference(quantized, dequantization_config);
        const HostTensor actual =
            quant_dequant::dequantize_cuda(quantized, dequantization_config);
        return equal_dequantized_tensors(
            expected, actual, scale_mode_name, output_type_name);
    } catch (const std::exception& error) {
        std::cerr << "MXFP8 CUDA 反量化对照意外失败：scale_mode="
                  << scale_mode_name << "，output_type=" << output_type_name
                  << "，原因：" << error.what() << '\n';
        return false;
    }
}

}  // namespace

/**
 * @brief 运行 MXFP8 CUDA 反量化与 CPU reference 的正确性对照。
 *
 * 测试分别覆盖 tensor-scale 和 rowwise block-scale，并对 FP16、BF16、FP32 三种
 * 输出类型检查：数值内存始终为 FP32，但 `HostTensor::desc.dtype` 必须正确保留
 * 请求类型，供后续 QDTENSOR 写入器进行物理转换。
 *
 * @return 有 CUDA device 时所有对照通过返回 0；无可用 CUDA runtime 时跳过并
 * 返回 0；其他错误返回 1。
 */
int run_mxfp8_dequantize_cuda_tests() {
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

    constexpr ScaleMode kScaleModes[]{
        ScaleMode::kTensor,
        ScaleMode::kBlock,
    };
    constexpr std::string_view kScaleModeNames[]{"tensor", "block"};

    for (std::size_t scale_mode_index = 0U;
         scale_mode_index < std::size(kScaleModes);
         ++scale_mode_index) {
        for (std::size_t output_type_index = 0U;
             output_type_index < std::size(kOutputTypes);
             ++output_type_index) {
            if (!compare_dequantization_with_cpu(
                    input,
                    kScaleModes[scale_mode_index],
                    kOutputTypes[output_type_index],
                    kScaleModeNames[scale_mode_index],
                    kOutputTypeNames[output_type_index])) {
                return 1;
            }
        }
    }

    return 0;
}

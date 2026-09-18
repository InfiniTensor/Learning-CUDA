#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
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

/**
 * @brief 判断当前环境是否实际可发射 CUDA kernel。
 *
 * @return 检测到至少一个可用 CUDA device 时为 true；无 driver 或无 device 时
 * 为 false。其他 runtime 错误由调用者作为失败处理。
 */
[[nodiscard]] bool has_cuda_device() {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

/**
 * @brief 构造跨越 MXFP8 block 边界的确定性有限输入。
 *
 * 33 列使每行都含一个完整 32 元素 block 与一个 tail block；tensor-scale 配置
 * 会额外覆盖“全局 amax 规约 + scale 写入 + 编码”三段 kernel 被同一组 event
 * 包住的场景。
 *
 * @return 合法的 3×33 FP32 host 张量。
 */
[[nodiscard]] HostTensor make_input() {
    constexpr std::uint64_t kRows = 3U;
    constexpr std::uint64_t kCols = 33U;
    std::vector<float> values;
    values.reserve(static_cast<std::size_t>(kRows * kCols));

    for (std::uint64_t index = 0U; index < kRows * kCols; ++index) {
        const int bucket = static_cast<int>((index * 19U + 7U) % 59U) - 29;
        values.push_back(static_cast<float>(bucket) * 0.25F);
    }
    values.front() = 448.0F;

    return HostTensor{
        .desc = TensorDesc{
            .num_rows = kRows,
            .num_cols = kCols,
            .dtype = DType::kFloat32,
        },
        .values = std::move(values),
    };
}

/**
 * @brief 判断 CUDA Event 计时结果是否适合交给 metrics 层记录。
 *
 * CUDA event 的有限分辨率允许极短 kernel 报告 0 ms，因此这里不强制大于零；
 * 只要求它是有限且非负的纯 device 时间。
 *
 * @param kernel_ms `*_cuda_profiled()` 返回的 kernel 时间。
 * @return 数值可用时为 true。
 */
[[nodiscard]] bool is_valid_kernel_time(const double kernel_ms) {
    return std::isfinite(kernel_ms) && kernel_ms >= 0.0;
}

/**
 * @brief 比较 profile CUDA pipeline 与 CPU reference 的端到端结果和计时。
 *
 * @return 量化后的持久化字段、反量化 FP32 值均与 reference 一致，且两个 Event
 * 时间均有效时返回 true。
 */
[[nodiscard]] bool test_profiled_mxfp8_pipeline() {
    const HostTensor input = make_input();
    const QuantizationConfig quantization_config{
        .format = QuantFormat::kMxfp8,
        .block_size = kMxfp8BlockSize,
        .scale_mode = ScaleMode::kTensor,
        .rounding = RoundingMode::kNearest,
        .stochastic_seed = 0U,
    };
    const DequantizationConfig dequantization_config{
        .output_type = DType::kFloat32,
    };

    try {
        const auto expected_quantized =
            quant_dequant::quantize_reference(input, quantization_config);
        const auto profiled_quantized =
            quant_dequant::quantize_cuda_profiled(input, quantization_config);

        if (!is_valid_kernel_time(profiled_quantized.kernel_ms)) {
            std::cerr << "MXFP8 profile 量化 kernel_ms 非法。\n";
            return false;
        }
        if (profiled_quantized.tensor.payload != expected_quantized.payload ||
            profiled_quantized.tensor.local_scales !=
                expected_quantized.local_scales ||
            profiled_quantized.tensor.global_scale !=
                expected_quantized.global_scale) {
            std::cerr << "MXFP8 profile 量化结果与 CPU reference 不一致。\n";
            return false;
        }

        const auto expected_dequantized = quant_dequant::dequantize_reference(
            expected_quantized, dequantization_config);
        const auto profiled_dequantized = quant_dequant::dequantize_cuda_profiled(
            profiled_quantized.tensor, dequantization_config);

        if (!is_valid_kernel_time(profiled_dequantized.kernel_ms)) {
            std::cerr << "MXFP8 profile 反量化 kernel_ms 非法。\n";
            return false;
        }
        if (profiled_dequantized.tensor.desc.num_rows !=
                expected_dequantized.desc.num_rows ||
            profiled_dequantized.tensor.desc.num_cols !=
                expected_dequantized.desc.num_cols ||
            profiled_dequantized.tensor.desc.dtype !=
                expected_dequantized.desc.dtype ||
            profiled_dequantized.tensor.values != expected_dequantized.values) {
            std::cerr << "MXFP8 profile 反量化结果与 CPU reference 不一致。\n";
            return false;
        }
    } catch (const std::exception& error) {
        std::cerr << "CUDA profile pipeline 测试意外失败：" << error.what() << '\n';
        return false;
    }

    return true;
}

}  // namespace

/**
 * @brief 运行 CUDA profile pipeline 的 MXFP8 端到端测试。
 *
 * 无 CUDA device 或 driver 时按照项目 CUDA 测试约定跳过；有 GPU 时验证 profile
 * 接口确实完成 H2D、同 stream kernel 计时与 D2H，并保留与 CPU reference 相同的
 * 功能结果。
 *
 * @return 通过或跳过时为 0，失败时为 1。
 */
int run_cuda_profile_tests() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && device_count == 0)) {
        std::cout << "CUDA runtime 不可用，跳过 CUDA profile pipeline 测试。\n";
        return 0;
    }
    if (status != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount 失败：" << cudaGetErrorString(status) << '\n';
        return 1;
    }

    return has_cuda_device() && test_profiled_mxfp8_pipeline() ? 0 : 1;
}

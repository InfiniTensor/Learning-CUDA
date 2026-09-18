#include <iostream>
#include <optional>
#include <string_view>
#include <vector>

#include "quant_dequant/quantize.hpp"

namespace {

/**
 * @brief 验证调用会以包含指定文本的 ReferenceError 失败。
 *
 * 此测试只锁定 public API 的格式分发行为；MXFP8 的位级量化与实际文件
 * round-trip 由 test_mxfp8_reference.cpp 单独覆盖。
 *
 * @tparam Callable 不接受参数、可能抛出 ReferenceError 的可调用对象类型。
 * @param callable 待调用的 CPU reference API。
 * @param expected_text 错误文本中必须出现的格式或阶段说明。
 * @return 接收到符合预期的 ReferenceError 时返回 true。
 */
template <typename Callable>
[[nodiscard]] bool throws_expected_reference_error(
    Callable&& callable,
    const std::string_view expected_text) {
    try {
        callable();
    } catch (const quant_dequant::ReferenceError& error) {
        return std::string_view{error.what()}.find(expected_text) !=
               std::string_view::npos;
    }

    return false;
}

/**
 * @brief 构造可通过 CPU reference 入口校验的最小 MXFP8 量化结果。
 *
 * @return 形状为 1 x 1 的合法 MXFP8 QuantizedTensor。
 */
[[nodiscard]] quant_dequant::QuantizedTensor make_mxfp8_tensor() {
    return {
        .desc = {
            .source_desc = {
                .num_rows = 1U,
                .num_cols = 1U,
                .dtype = quant_dequant::DType::kFloat32,
            },
            .format = quant_dequant::QuantFormat::kMxfp8,
            .scale_mode = quant_dequant::ScaleMode::kBlock,
            .rounding = quant_dequant::RoundingMode::kNearest,
            .stochastic_seed = 0U,
            .block_size = quant_dequant::kMxfp8BlockSize,
            .scale_layout = quant_dequant::ScaleLayout::kRowwise,
        },
        .payload = {0x00U},
        .local_scales = {0x00U},
        .global_scale = std::nullopt,
    };
}

}  // namespace

/**
 * @brief 验证 CPU reference 的格式分派和两个已实现量化路径。
 *
 * @return 所有骨架断言通过时返回 0；否则打印错误并返回 1。
 */
int run_reference_dispatch_tests() {
    const quant_dequant::HostTensor input{
        .desc = {
            .num_rows = 1U,
            .num_cols = 1U,
            .dtype = quant_dequant::DType::kFloat32,
        },
        .values = {1.0F},
    };
    const quant_dequant::QuantizationConfig mxfp8_config{
        .format = quant_dequant::QuantFormat::kMxfp8,
        .block_size = quant_dequant::kMxfp8BlockSize,
        .scale_mode = quant_dequant::ScaleMode::kBlock,
        .rounding = quant_dequant::RoundingMode::kNearest,
        .stochastic_seed = 0U,
    };
    const quant_dequant::QuantizationConfig nvfp4_config{
        .format = quant_dequant::QuantFormat::kNvfp4,
        .block_size = quant_dequant::kNvfp4BlockSize,
        .scale_mode = quant_dequant::ScaleMode::kBlock,
        .rounding = quant_dequant::RoundingMode::kNearest,
        .stochastic_seed = 0U,
    };
    const quant_dequant::DequantizationConfig dequantization_config{
        .output_type = quant_dequant::DType::kFloat32,
    };

    try {
        const quant_dequant::QuantizedTensor result =
            quant_dequant::quantize_reference(input, mxfp8_config);
        if (!result.isConsistent() || result.desc.format !=
                                      quant_dequant::QuantFormat::kMxfp8) {
            std::cerr << "MXFP8 CPU reference 量化未返回自洽结果。\n";
            return 1;
        }
    } catch (const quant_dequant::ReferenceError& error) {
        std::cerr << "MXFP8 CPU reference 量化意外失败：" << error.what() << '\n';
        return 1;
    }

    try {
        const quant_dequant::QuantizedTensor result =
            quant_dequant::quantize_reference(input, nvfp4_config);
        if (!result.isConsistent() || result.desc.format !=
                                      quant_dequant::QuantFormat::kNvfp4 ||
            !result.global_scale.has_value()) {
            std::cerr << "NVFP4 CPU reference 量化未返回自洽结果。\n";
            return 1;
        }
    } catch (const quant_dequant::ReferenceError& error) {
        std::cerr << "NVFP4 CPU reference 量化意外失败：" << error.what() << '\n';
        return 1;
    }

    const quant_dequant::QuantizedTensor mxfp8_tensor = make_mxfp8_tensor();
    try {
        const quant_dequant::HostTensor result =
            quant_dequant::dequantize_reference(
                mxfp8_tensor, dequantization_config);
        if (result.desc.num_rows != 1U || result.desc.num_cols != 1U ||
            result.desc.dtype != quant_dequant::DType::kFloat32 ||
            result.values != std::vector<float>{0.0F}) {
            std::cerr << "MXFP8 CPU reference 反量化返回结果不符合预期。\n";
            return 1;
        }
    } catch (const quant_dequant::ReferenceError& error) {
        std::cerr << "MXFP8 CPU reference 反量化意外失败：" << error.what() << '\n';
        return 1;
    }

    return 0;
}

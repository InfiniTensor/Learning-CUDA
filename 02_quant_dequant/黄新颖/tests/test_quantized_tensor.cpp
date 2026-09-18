#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "quant_dequant/quantized_tensor.hpp"

namespace {

/**
 * @brief 构造一个可复用的 MXFP8 block 模式元数据。
 *
 * @return 形状为 1 x 33 的合法 MXFP8 描述；它特意包含一个尾 block。
 */
[[nodiscard]] quant_dequant::QuantizedTensorDesc make_mxfp8_block_desc() {
    return {
        .source_desc = {
            .num_rows = 1U,
            .num_cols = 33U,
            .dtype = quant_dequant::DType::kFloat32,
        },
        .format = quant_dequant::QuantFormat::kMxfp8,
        .scale_mode = quant_dequant::ScaleMode::kBlock,
        .rounding = quant_dequant::RoundingMode::kNearest,
        .stochastic_seed = 0U,
        .block_size = quant_dequant::kMxfp8BlockSize,
        .scale_layout = quant_dequant::ScaleLayout::kRowwise,
    };
}

/**
 * @brief 构造一个可复用的 NVFP4 block 模式元数据。
 *
 * @return 形状为 1 x 17 的合法 NVFP4 描述；它覆盖尾 block 和奇数 nibble。
 */
[[nodiscard]] quant_dequant::QuantizedTensorDesc make_nvfp4_block_desc() {
    return {
        .source_desc = {
            .num_rows = 1U,
            .num_cols = 17U,
            .dtype = quant_dequant::DType::kFloat16,
        },
        .format = quant_dequant::QuantFormat::kNvfp4,
        .scale_mode = quant_dequant::ScaleMode::kBlock,
        .rounding = quant_dequant::RoundingMode::kStochastic,
        .stochastic_seed = 2026U,
        .block_size = quant_dequant::kNvfp4BlockSize,
        .scale_layout = quant_dequant::ScaleLayout::kRowwise,
    };
}

}  // namespace

/**
 * @brief 验证统一量化结果对象的尺寸推导和 MXFP8/NVFP4 格式不变量。
 *
 * @return 所有断言通过时返回 0；否则打印错误并返回 1。
 */
int run_quantized_tensor_tests() {
    const quant_dequant::QuantizedTensorDesc mxfp8_desc = make_mxfp8_block_desc();
    if (!mxfp8_desc.isMetadataValid() ||
        mxfp8_desc.expectedPayloadBytes() != std::optional<std::uint64_t>{33U} ||
        mxfp8_desc.blocksPerRow() != std::optional<std::uint64_t>{2U} ||
        mxfp8_desc.expectedLocalScaleCount() != std::optional<std::uint64_t>{2U} ||
        mxfp8_desc.usesGlobalScale()) {
        std::cerr << "MXFP8 描述的长度推导或元数据校验错误。\n";
        return 1;
    }

    quant_dequant::QuantizedTensor mxfp8_tensor{
        .desc = mxfp8_desc,
        .payload = std::vector<std::uint8_t>(33U, 0U),
        .local_scales = {0U, 0U},
        .global_scale = std::nullopt,
    };
    if (!mxfp8_tensor.isConsistent()) {
        std::cerr << "合法 MXFP8 量化结果被错误拒绝。\n";
        return 1;
    }

    mxfp8_tensor.global_scale = 1.0F;
    if (mxfp8_tensor.isConsistent()) {
        std::cerr << "MXFP8 不应接受内存语义上的 global_scale。\n";
        return 1;
    }

    const quant_dequant::QuantizedTensorDesc nvfp4_desc = make_nvfp4_block_desc();
    if (!nvfp4_desc.isMetadataValid() ||
        nvfp4_desc.expectedPayloadBytes() != std::optional<std::uint64_t>{9U} ||
        nvfp4_desc.blocksPerRow() != std::optional<std::uint64_t>{2U} ||
        nvfp4_desc.expectedLocalScaleCount() != std::optional<std::uint64_t>{2U} ||
        !nvfp4_desc.usesGlobalScale()) {
        std::cerr << "NVFP4 描述的长度推导或元数据校验错误。\n";
        return 1;
    }

    quant_dequant::QuantizedTensor nvfp4_tensor{
        .desc = nvfp4_desc,
        .payload = std::vector<std::uint8_t>(9U, 0x21U),
        .local_scales = {0x38U, 0x40U},
        .global_scale = 0.5F,
    };
    // 第 17 个元素只使用最后字节的低 nibble；高 nibble 必须清零。
    nvfp4_tensor.payload.back() = 0x01U;
    if (!nvfp4_tensor.isConsistent()) {
        std::cerr << "合法 NVFP4 量化结果被错误拒绝。\n";
        return 1;
    }

    nvfp4_tensor.payload.back() = 0xF1U;
    if (nvfp4_tensor.isConsistent()) {
        std::cerr << "NVFP4 奇数尾元素的未使用高 nibble 未被拒绝。\n";
        return 1;
    }

    nvfp4_tensor.payload.back() = 0x01U;
    nvfp4_tensor.global_scale.reset();
    if (nvfp4_tensor.isConsistent()) {
        std::cerr << "NVFP4 缺少 global_scale 未被拒绝。\n";
        return 1;
    }

    nvfp4_tensor.global_scale = std::numeric_limits<float>::infinity();
    if (nvfp4_tensor.isConsistent()) {
        std::cerr << "NVFP4 无穷 global_scale 未被拒绝。\n";
        return 1;
    }

    nvfp4_tensor.global_scale = 0.5F;
    nvfp4_tensor.local_scales.pop_back();
    if (nvfp4_tensor.isConsistent()) {
        std::cerr << "NVFP4 local scale 数量错误未被拒绝。\n";
        return 1;
    }

    quant_dequant::QuantizedTensorDesc nvfp4_tensor_scale_desc = nvfp4_desc;
    nvfp4_tensor_scale_desc.scale_mode = quant_dequant::ScaleMode::kTensor;
    if (nvfp4_tensor_scale_desc.isMetadataValid() ||
        nvfp4_tensor_scale_desc.expectedLocalScaleCount().has_value()) {
        std::cerr << "NVFP4 tensor scale 元数据未被拒绝。\n";
        return 1;
    }

    return 0;
}

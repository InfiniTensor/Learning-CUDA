#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "formats/nvfp4_codec.cuh"
#include "quant_dequant/quantize.hpp"

namespace {

/**
 * @brief 构造包含完整 block、尾 block 和跨行 packed-byte 边界的 NVFP4 输入。
 *
 * 单行 17 个元素使第 17 个元素占一行末尾 byte 的低 nibble，下一行第一个元素
 * 占同一 byte 的高 nibble；这验证 CPU reference 的 payload 配对按线性 row-major
 * 顺序，而不是错误地把每行尾部 padding 成偶数长度。
 *
 * @return 2×17、元素总数为偶数的有限 FP32 输入张量。
 */
[[nodiscard]] quant_dequant::HostTensor make_reference_input() {
    std::vector<float> values(34U, 0.0F);
    values[0U] = 6.0F;
    values[1U] = -6.0F;
    values[15U] = 3.0F;
    values[16U] = 1.0F;
    values[17U] = -1.0F;
    values[18U] = 0.5F;
    values[33U] = -2.0F;

    return {
        .desc = {
            .num_rows = 2U,
            .num_cols = 17U,
            .dtype = quant_dequant::DType::kFloat32,
        },
        .values = std::move(values),
    };
}

/**
 * @brief 构造严格 NVFP4 block-scale 量化配置。
 *
 * @param rounding 目标 E2M1 舍入模式。
 * @param seed stochastic rounding 的确定性种子。
 * @return 可传给 public CPU reference API 的合法配置。
 */
[[nodiscard]] quant_dequant::QuantizationConfig make_nvfp4_config(
    const quant_dequant::RoundingMode rounding,
    const std::uint64_t seed = 0U) {
    return {
        .format = quant_dequant::QuantFormat::kNvfp4,
        .block_size = quant_dequant::kNvfp4BlockSize,
        .scale_mode = quant_dequant::ScaleMode::kBlock,
        .rounding = rounding,
        .stochastic_seed = seed,
    };
}

/**
 * @brief 验证 nearest reference 的层次 scale、byte 打包和跨行配对。
 *
 * @return 所有 NVFP4 持久化字段满足预期时返回 true。
 */
[[nodiscard]] bool test_nearest_reference_layout() {
    using namespace quant_dequant;

    const HostTensor input = make_reference_input();
    QuantizedTensor result{};
    try {
        result = quantize_reference(
            input, make_nvfp4_config(RoundingMode::kNearest));
    } catch (const ReferenceError& error) {
        std::cerr << "NVFP4 nearest CPU reference 意外失败：" << error.what() << '\n';
        return false;
    }

    const bool metadata_is_valid = result.isConsistent() &&
        result.desc.format == QuantFormat::kNvfp4 &&
        result.desc.scale_mode == ScaleMode::kBlock &&
        result.desc.block_size == kNvfp4BlockSize &&
        result.payload.size() == 17U && result.local_scales.size() == 4U &&
        result.global_scale.has_value();
    if (!metadata_is_valid) {
        return false;
    }

    // 全张量 amax 为 6，所以 global_scale 为 1/448；第一行第一个完整 block
    // 的 local scale 是 E4M3(448)=0x7e，{+6,-6} 打包为 0xf7。
    const bool standard_vector =
        result.global_scale == std::optional<float>{1.0F / 448.0F} &&
        result.local_scales[0U] == 0x7eU && result.payload[0U] == 0xf7U;

    // 线性下标 16 与 17 处于不同行、不同 16 元素 local block，却必须共用
    // payload[8] 的低/高 nibble。它们的实际 code 由各自 scale 决定；这里用
    // codec 复算并验证打包位置，避免把跨行 pair 错误分成两个 byte。
    const std::uint8_t row0_tail = formats::encode_nvfp4_element(
        input.values[16U], result.local_scales[1U], *result.global_scale,
        RoundingMode::kNearest);
    const std::uint8_t row1_head = formats::encode_nvfp4_element(
        input.values[17U], result.local_scales[2U], *result.global_scale,
        RoundingMode::kNearest);
    const bool cross_row_pair = result.payload[8U] ==
        formats::pack_e2m1_nibbles(row0_tail, row1_head);

    return standard_vector && cross_row_pair;
}

/**
 * @brief 验证 stochastic reference 可由 seed 完全复现。
 *
 * @return 同 seed 两次量化的 payload、local scale 与 global scale 全部一致时返回 true。
 */
[[nodiscard]] bool test_stochastic_reference_is_deterministic() {
    using namespace quant_dequant;

    const HostTensor input = make_reference_input();
    const QuantizationConfig config = make_nvfp4_config(
        RoundingMode::kStochastic, 0xd1b54a32d192ed03ULL);
    try {
        const QuantizedTensor first = quantize_reference(input, config);
        const QuantizedTensor second = quantize_reference(input, config);
        return first.isConsistent() && second.isConsistent() &&
               first.payload == second.payload &&
               first.local_scales == second.local_scales &&
               first.global_scale == second.global_scale;
    } catch (const ReferenceError& error) {
        std::cerr << "NVFP4 stochastic CPU reference 意外失败："
                  << error.what() << '\n';
        return false;
    }
}

/**
 * @brief 验证 NVFP4 CPU reference 在写出结果前拒绝非有限输入。
 *
 * @return 出现包含 NaN/Inf 诊断的 ReferenceError 时返回 true。
 */
[[nodiscard]] bool test_rejects_nonfinite_input() {
    using namespace quant_dequant;

    HostTensor input = make_reference_input();
    input.values[9U] = std::numeric_limits<float>::infinity();
    try {
        static_cast<void>(quantize_reference(
            input, make_nvfp4_config(RoundingMode::kNearest)));
    } catch (const ReferenceError& error) {
        return std::string_view{error.what()}.find("NaN 或 Inf") !=
               std::string_view::npos;
    }

    return false;
}

/**
 * @brief 验证 NVFP4 CPU reference 能按 packed-byte 顺序完整反量化。
 *
 * 测试复用 2×17 的奇数列输入：payload[8] 的低/high nibble 分属不同行。若实现
 * 错把两个 nibble 套用同一个 local scale，或未跳过奇数总元素尾 byte 的高 nibble，
 * 此处会产生与逐元素 codec 解码不同的结果。
 *
 * @return 输出描述、长度和所有 FP32 解码数值正确时返回 true。
 */
[[nodiscard]] bool test_reference_dequantization() {
    using namespace quant_dequant;

    const HostTensor input = make_reference_input();
    const DequantizationConfig config{
        .output_type = DType::kBFloat16,
    };
    try {
        const QuantizedTensor quantized = quantize_reference(
            input, make_nvfp4_config(RoundingMode::kNearest));
        const HostTensor result = dequantize_reference(quantized, config);
        if (result.desc.num_rows != input.desc.num_rows ||
            result.desc.num_cols != input.desc.num_cols ||
            result.desc.dtype != DType::kBFloat16 ||
            result.values.size() != input.values.size()) {
            return false;
        }

        const auto blocks_per_row_u64 = quantized.desc.blocksPerRow();
        if (!blocks_per_row_u64.has_value()) {
            return false;
        }
        const std::size_t blocks_per_row =
            static_cast<std::size_t>(*blocks_per_row_u64);

        for (std::size_t linear_index = 0U;
             linear_index < result.values.size();
             ++linear_index) {
            const std::size_t row = linear_index /
                static_cast<std::size_t>(quantized.desc.source_desc.num_cols);
            const std::size_t column = linear_index %
                static_cast<std::size_t>(quantized.desc.source_desc.num_cols);
            const std::size_t scale_index =
                row * blocks_per_row +
                column / static_cast<std::size_t>(kNvfp4BlockSize);
            const std::uint8_t packed_byte = quantized.payload[linear_index / 2U];
            const std::uint8_t e2m1_code = linear_index % 2U == 0U
                ? formats::unpack_e2m1_low_nibble(packed_byte)
                : formats::unpack_e2m1_high_nibble(packed_byte);
            const float expected = formats::decode_nvfp4_element(
                e2m1_code,
                quantized.local_scales[scale_index],
                *quantized.global_scale);
            if (result.values[linear_index] != expected) {
                return false;
            }
        }
    } catch (const ReferenceError& error) {
        std::cerr << "NVFP4 CPU reference 反量化意外失败："
                  << error.what() << '\n';
        return false;
    }

    return true;
}

}  // namespace

/**
 * @brief 运行严格 NVFP4 CPU reference 的量化正确性测试。
 *
 * @return 所有测试通过时返回 0；否则打印诊断并返回 1。
 */
int run_nvfp4_reference_tests() {
    if (!test_nearest_reference_layout()) {
        std::cerr << "NVFP4 nearest reference 的层次 scale 或 packed layout 测试失败。\n";
        return 1;
    }

    if (!test_stochastic_reference_is_deterministic()) {
        std::cerr << "NVFP4 stochastic reference 的确定性测试失败。\n";
        return 1;
    }

    if (!test_rejects_nonfinite_input()) {
        std::cerr << "NVFP4 CPU reference 未拒绝 Inf 输入。\n";
        return 1;
    }

    if (!test_reference_dequantization()) {
        std::cerr << "NVFP4 CPU reference packed 反量化测试失败。\n";
        return 1;
    }

    return 0;
}

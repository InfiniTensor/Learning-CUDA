#include "reference_detail.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "formats/nvfp4_codec.cuh"

namespace quant_dequant::reference_detail {
namespace {

/**
 * @brief 将 uint64 元数据长度安全转换为 host vector 使用的 size_t。
 *
 * @param count 待转换的元素或字节数量。
 * @param field_name 用于诊断的字段名称。
 * @return 可安全用于 host 容器长度或下标的 size_t。
 * @throws ReferenceError count 超出当前平台 host 容器可表示范围时抛出。
 */
[[nodiscard]] std::size_t checked_size(const std::uint64_t count,
                                       const std::string_view field_name) {
    constexpr std::uint64_t kMaxSize =
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
    if (count > kMaxSize) {
        throw ReferenceError{
            "NVFP4 CPU reference 的 \"" + std::string{field_name} +
            "\" 超出当前 host 容器可表示范围。"};
    }

    return static_cast<std::size_t>(count);
}

/**
 * @brief 计算连续有限 FP32 值的最大绝对值，并定位非有限输入。
 *
 * @param values 待扫描的连续 row-major 元素区间。
 * @param first_linear_index values[0] 在完整张量中的线性下标。
 * @return 区间的 amax；全零区间返回正零。
 * @throws ReferenceError 发现 NaN、+Inf 或 -Inf 时抛出。
 */
[[nodiscard]] float calculate_finite_amax(const std::span<const float> values,
                                          const std::uint64_t first_linear_index) {
    float amax = 0.0F;
    for (std::size_t offset = 0U; offset < values.size(); ++offset) {
        const float value = values[offset];
        if (!formats::fp32::is_finite(value)) {
            throw ReferenceError{
                "NVFP4 CPU reference 不接受 NaN 或 Inf 输入；"
                "发现非有限值的线性下标为 " +
                std::to_string(first_linear_index +
                               static_cast<std::uint64_t>(offset)) +
                "。"};
        }

        const float magnitude = formats::fp32::absolute_value(value);
        amax = amax > magnitude ? amax : magnitude;
    }

    return amax;
}

/**
 * @brief 从已验证的输入描述和量化配置构造严格 NVFP4 输出元数据。
 *
 * @param input_desc 原始 QDTENSOR 的形状和物理输入 dtype。
 * @param config 已通过公共 dispatcher 校验的 NVFP4 配置。
 * @return 尚未填充 payload、scale 的 QuantizedTensorDesc。
 */
[[nodiscard]] QuantizedTensorDesc make_nvfp4_desc(
    const TensorDesc& input_desc,
    const QuantizationConfig& config) {
    return {
        .source_desc = input_desc,
        .format = QuantFormat::kNvfp4,
        .scale_mode = ScaleMode::kBlock,
        .rounding = config.rounding,
        .stochastic_seed = config.stochastic_seed,
        .block_size = kNvfp4BlockSize,
        .scale_layout = ScaleLayout::kRowwise,
    };
}

/**
 * @brief 按 row-major 元素坐标返回其 rowwise 16 元素 block 的 scale 下标。
 *
 * @param row 当前元素所在行。
 * @param column 当前元素所在列。
 * @param blocks_per_row 每行的 NVFP4 logical block 数量。
 * @return local_scales 内的零起始下标。
 */
[[nodiscard]] constexpr std::size_t local_scale_index(
    const std::size_t row,
    const std::size_t column,
    const std::size_t blocks_per_row) noexcept {
    return row * blocks_per_row +
           column / static_cast<std::size_t>(kNvfp4BlockSize);
}

/**
 * @brief 为一个线性元素下标生成与 CUDA 相同的 stochastic rounding 随机数。
 *
 * 当前项目让 MXFP8 与 NVFP4 共用同一 seed/index 到 `[0,1)` 的无状态映射；
 * format 不参与随机数状态，因此同一元素在两条后端中不会受线程调度影响。
 *
 * @param config 当前 NVFP4 量化配置。
 * @param linear_index 元素的 row-major 线性下标。
 * @return nearest 模式返回 0，stochastic 模式返回确定性 `[0,1)` 随机数。
 */
[[nodiscard]] float element_uniform_random(const QuantizationConfig& config,
                                           const std::uint64_t linear_index) noexcept {
    return config.rounding == RoundingMode::kStochastic
        ? formats::nvfp4_stochastic_uniform_for_element(
              config.stochastic_seed, linear_index)
        : 0.0F;
}

}  // namespace

QuantizedTensor quantize_nvfp4_reference(
    const HostTensor& input,
    const QuantizationConfig& config) {
    // 阶段 0：NVFP4 没有 tensor local-scale 模式。公共配置校验也会拒绝它，
    // 此处额外固定格式前提，防止内部调用绕过 dispatcher。
    if (!config.isValid() || config.format != QuantFormat::kNvfp4 ||
        config.scale_mode != ScaleMode::kBlock ||
        config.block_size != kNvfp4BlockSize) {
        throw ReferenceError{
            "NVFP4 CPU reference 要求 format = nvfp4、block_size = 16 且 "
            "scale_mode = block。"};
    }

    // 阶段 1：先验证并构造由文件、CPU reference 与 CUDA pipeline 共同使用的
    // 描述。NVFP4 的 expectedPayloadBytes() 在这里确保 payload 真正按两元素
    // 每字节分配，而非以一个 uint8_t 保存一个逻辑 FP4 元素。
    const QuantizedTensorDesc desc = make_nvfp4_desc(input.desc, config);
    const auto expected_payload_bytes = desc.expectedPayloadBytes();
    const auto expected_local_scale_count = desc.expectedLocalScaleCount();
    const auto blocks_per_row_u64 = desc.blocksPerRow();
    if (!desc.isMetadataValid() || !expected_payload_bytes.has_value() ||
        !expected_local_scale_count.has_value() || !blocks_per_row_u64.has_value()) {
        throw ReferenceError{
            "NVFP4 CPU reference 无法构造合法的量化输出元数据。"};
    }

    const std::size_t row_count = checked_size(input.desc.num_rows, "行数");
    const std::size_t column_count = checked_size(input.desc.num_cols, "列数");
    const std::size_t blocks_per_row =
        checked_size(*blocks_per_row_u64, "每行 block 数量");
    const std::size_t element_count = input.values.size();

    // 阶段 2：全张量扫描同时完成 amax 和非有限输入拒绝。global_scale 不是
    // scale_mode=tensor：它始终与后续每个 16 元素 local scale 相乘。
    const float tensor_amax = calculate_finite_amax(
        std::span<const float>{input.values.data(), input.values.size()}, 0U);
    const float global_scale = formats::compute_nvfp4_global_scale(tensor_amax);
    if (!formats::fp32::is_finite(global_scale) || global_scale <= 0.0F) {
        throw ReferenceError{
            "NVFP4 CPU reference 无法为有限输入构造有限正的 global_scale。"};
    }

    QuantizedTensor output{
        .desc = desc,
        .payload = std::vector<std::uint8_t>(
            checked_size(*expected_payload_bytes, "payload 字节数"), 0U),
        .local_scales = std::vector<std::uint8_t>(
            checked_size(*expected_local_scale_count, "local scale 数量"), 0U),
        .global_scale = global_scale,
    };

    // 阶段 3：每行独立切为 1×16 block。尾 block 只扫描真实元素；逻辑补零
    // 不会改变 amax，也绝不能出现在 payload 中。
    for (std::size_t row = 0U; row < row_count; ++row) {
        const std::size_t row_offset = row * column_count;
        for (std::size_t block_in_row = 0U;
             block_in_row < blocks_per_row;
             ++block_in_row) {
            const std::size_t first_column =
                block_in_row * static_cast<std::size_t>(kNvfp4BlockSize);
            const std::size_t valid_count = std::min(
                static_cast<std::size_t>(kNvfp4BlockSize),
                column_count - first_column);
            const std::size_t first_linear_index = row_offset + first_column;
            const float block_amax = calculate_finite_amax(
                std::span<const float>{
                    input.values.data() + first_linear_index, valid_count},
                static_cast<std::uint64_t>(first_linear_index));
            const std::uint8_t local_scale =
                formats::compute_nvfp4_local_scale_code(
                    block_amax, global_scale);
            if (formats::is_e4m3_nan(local_scale)) {
                throw ReferenceError{
                    "NVFP4 CPU reference 为有限输入生成了非法 E4M3 local scale。"};
            }

            output.local_scales[row * blocks_per_row + block_in_row] =
                local_scale;
        }
    }

    // 阶段 4：两个相邻线性元素共享一个物理 payload byte。先分别用各自行/列
    // 推导 local scale，再由 pack_e2m1_nibbles() 一次性写出低/高 nibble；这
    // 与后续 CUDA 的“偶数 lane 独占 packed store”拥有同一个字节布局。
    for (std::size_t even_index = 0U;
         even_index < element_count;
         even_index += 2U) {
        const std::size_t even_row = even_index / column_count;
        const std::size_t even_column = even_index % column_count;
        const std::size_t even_scale_index = local_scale_index(
            even_row, even_column, blocks_per_row);
        const std::uint8_t low_nibble = formats::encode_nvfp4_element(
            input.values[even_index],
            output.local_scales[even_scale_index],
            global_scale,
            config.rounding,
            element_uniform_random(config, static_cast<std::uint64_t>(even_index)));

        const std::size_t odd_index = even_index + 1U;
        std::uint8_t high_nibble = 0x00U;
        if (odd_index < element_count) {
            const std::size_t odd_row = odd_index / column_count;
            const std::size_t odd_column = odd_index % column_count;
            const std::size_t odd_scale_index = local_scale_index(
                odd_row, odd_column, blocks_per_row);
            high_nibble = formats::encode_nvfp4_element(
                input.values[odd_index],
                output.local_scales[odd_scale_index],
                global_scale,
                config.rounding,
                element_uniform_random(config, static_cast<std::uint64_t>(odd_index)));
        }

        output.payload[even_index / 2U] = formats::pack_e2m1_nibbles(
            low_nibble, high_nibble);
    }

    if (!output.isConsistent()) {
        throw ReferenceError{
            "NVFP4 CPU reference 构造出了不自洽的 QuantizedTensor。"};
    }

    return output;
}

HostTensor dequantize_nvfp4_reference(
    const QuantizedTensor& input,
    const DequantizationConfig& config) {
    // 阶段 0：public dispatcher 会执行同样的基本校验；格式专用函数仍自行固定
    // NVFP4 的不变量，防止内部调用把其他量化布局误派发到 E2M1 解码规则。
    if (!input.isConsistent() || !config.isValid() ||
        input.desc.format != QuantFormat::kNvfp4 ||
        input.desc.scale_mode != ScaleMode::kBlock ||
        input.desc.block_size != kNvfp4BlockSize ||
        !input.global_scale.has_value()) {
        throw ReferenceError{
            "NVFP4 CPU reference 反量化收到了非法输入或输出配置。"};
    }

    // 阶段 1：QDWGT 的 source_desc 记录量化前形状；反量化只替换请求的物理
    // 输出 dtype。所有 reference 数值仍保存在统一 FP32 HostTensor::values 中。
    TensorDesc output_desc = input.desc.source_desc;
    output_desc.dtype = config.output_type;
    const auto element_count_u64 = output_desc.elementCount();
    const auto blocks_per_row_u64 = input.desc.blocksPerRow();
    if (!output_desc.isValid() || !is_supported_output_dtype(output_desc.dtype) ||
        !element_count_u64.has_value() || !blocks_per_row_u64.has_value()) {
        throw ReferenceError{
            "NVFP4 CPU reference 无法构造合法的反量化输出描述或 scale 布局。"};
    }

    const std::size_t element_count =
        checked_size(*element_count_u64, "反量化元素数量");
    const std::size_t column_count =
        checked_size(input.desc.source_desc.num_cols, "列数");
    const std::size_t blocks_per_row =
        checked_size(*blocks_per_row_u64, "每行 block 数量");
    const float global_scale = *input.global_scale;

    HostTensor output{
        .desc = output_desc,
        .values = std::vector<float>(element_count, 0.0F),
    };

    // 阶段 2：一个物理 byte 刚好描述两个相邻的 row-major 逻辑元素。每次先
    // 解包 low/high nibble，再为偶数和奇数元素分别恢复坐标与 local-scale 下标。
    // 因此列数为奇数时，行尾 low nibble 和下一行开头 high nibble 仍会使用各自
    // 正确的 rowwise 16 元素 block scale；最后单元素 tail 的高 nibble 不读取。
    for (std::size_t payload_index = 0U;
         payload_index < input.payload.size();
         ++payload_index) {
        const std::size_t even_index = payload_index * 2U;
        const std::size_t even_row = even_index / column_count;
        const std::size_t even_column = even_index % column_count;
        const std::size_t even_scale_index = local_scale_index(
            even_row, even_column, blocks_per_row);
        const std::uint8_t packed_byte = input.payload[payload_index];

        output.values[even_index] = formats::decode_nvfp4_element(
            formats::unpack_e2m1_low_nibble(packed_byte),
            input.local_scales[even_scale_index],
            global_scale);

        const std::size_t odd_index = even_index + 1U;
        if (odd_index < element_count) {
            const std::size_t odd_row = odd_index / column_count;
            const std::size_t odd_column = odd_index % column_count;
            const std::size_t odd_scale_index = local_scale_index(
                odd_row, odd_column, blocks_per_row);
            output.values[odd_index] = formats::decode_nvfp4_element(
                formats::unpack_e2m1_high_nibble(packed_byte),
                input.local_scales[odd_scale_index],
                global_scale);
        }
    }

    if (output.values.size() != element_count) {
        throw ReferenceError{
            "NVFP4 CPU reference 构造出了长度不自洽的反量化结果。"};
    }

    return output;
}

}  // namespace quant_dequant::reference_detail

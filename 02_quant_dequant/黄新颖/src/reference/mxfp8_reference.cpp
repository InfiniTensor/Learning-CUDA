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

#include "formats/mxfp8_codec.cuh"
namespace quant_dequant::reference_detail {


namespace {
/**
 * @brief 将无符号计数安全转换为 vector 所使用的 std::size_t。
 *
 * `TensorDesc` 中的维度使用 uint64_t，而 host 容器使用 std::size_t。
 * 正常情况下 dispatcher 已经保证输入 values 可以被 host 容器持有；
 * 此处仍做一次检查，使 reference 后端即使被内部错误调用也不会发生截断。
 *
 * @param count 待转换的非负计数。
 * @param field_name 用于异常信息的字段名称。
 * @return 可安全作为 vector 长度或下标的 std::size_t。
 * @throws ReferenceError count 超出当前平台 std::size_t 范围时抛出。
 */
[[nodiscard]] std::size_t checked_size(
    const std::uint64_t count,
    const std::string_view field_name) {
    constexpr std::uint64_t kMaxSize =
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());

    if (count > kMaxSize) {
        throw ReferenceError{
            std::string{"MXFP8 CPU reference 的 \""} +
            std::string{field_name} +
            "\" 超出当前 host 容器可表示范围。"};
    }

    return static_cast<std::size_t>(count);
}
/**
 * @brief 计算一段连续输入数据的最大绝对值，并拒绝非有限输入。
 *
 * 题目和 format_spec.md 规定第一版只接受有限 FP16/FP32 输入。HostTensor
 * 的 values 虽然统一是 FP32，但仍可能包含由外部调用方构造的 NaN 或 Inf；
 * 必须在写出任何量化结果前拒绝它们。
 *
 * @param values 待扫描的连续 row-major 元素区间，不能为空。
 * @param first_linear_index values[0] 在完整张量中的线性下标，用于错误定位。
 * @return 区间内所有元素绝对值的最大值；全零区间返回 +0。
 * @throws ReferenceError values 中存在 NaN、+Inf 或 -Inf 时抛出。
 */
[[nodiscard]] float calculate_finite_amax(
    const std::span<const float> values,
    const std::uint64_t first_linear_index) {
    float amax = 0.0F;

    for (std::size_t offset = 0U; offset < values.size(); ++offset) {
        const float value = values[offset];

        if (!formats::fp32::is_finite(value)) {
            throw ReferenceError{
                "MXFP8 CPU reference 不接受 NaN 或 Inf 输入；"
                "发现非有限值的线性下标为 " +
                std::to_string(first_linear_index +
                               static_cast<std::uint64_t>(offset)) +
                "。"};
        }

        // absolute_value 通过清除 FP32 符号位求幅值，能正确处理 -0。
        const float magnitude = formats::fp32::absolute_value(value);
        if (magnitude > amax) {
            amax = magnitude;
        }
    }

    return amax;
}
/**
 * @brief 根据一段有限 FP32 数据生成一个 MXFP8 E8M0 scale code。
 *
 * 复用 codec 的 `compute_mxfp8_scale_code()`，其数值语义为：
 *
 * amax -> ceil_to_e8m0(amax / 448)
 *
 * 因而得到的 scale 不会小于该范围实际所需 scale，避免最大元素在 E4M3
 * 编码阶段因 scale 偏小而额外溢出。
 *
 * @param values 一个 tensor 或一个有效 rowwise block 的数据。
 * @param first_linear_index values[0] 的全局线性下标。
 * @return 合法的 E8M0 scale byte；全零区间返回 0x00。
 * @throws ReferenceError 输入出现非有限值或内部得到非法 E8M0 code 时抛出。
 */
[[nodiscard]] std::uint8_t make_mxfp8_scale_code(
    const std::span<const float> values,
    const std::uint64_t first_linear_index) {
    const float amax = calculate_finite_amax(values, first_linear_index);
    const std::uint8_t scale_code =
        formats::compute_mxfp8_scale_code(amax);

    // 有限输入按当前公式绝不应产生 0xff；保留检查以避免内部规则被修改后
    // 静默写出 NaN scale。
    if (formats::is_e8m0_nan(scale_code)) {
        throw ReferenceError{
            "MXFP8 CPU reference 为有限输入生成了非法 E8M0 NaN scale。"};
    }

    return scale_code;
}
/**
 * @brief 根据量化配置和输入描述构造 MXFP8 QuantizedTensor 元数据。
 *
 * @param input_desc 原始 FP16/FP32 输入张量的物理描述。
 * @param config 已经由 dispatcher 校验过的量化方向配置。
 * @return 尚未填充 payload 和 local_scales 的 MXFP8 描述。
 */
[[nodiscard]] QuantizedTensorDesc make_mxfp8_desc(
    const TensorDesc& input_desc,
    const QuantizationConfig& config) {
    return {
        .source_desc = input_desc,
        .format = QuantFormat::kMxfp8,
        .scale_mode = config.scale_mode,
        .rounding = config.rounding,
        .stochastic_seed = config.stochastic_seed,
        .block_size = config.block_size,
        .scale_layout = ScaleLayout::kRowwise,
    };
}
}// namespace

QuantizedTensor quantize_mxfp8_reference(
    const HostTensor& input,
    const QuantizationConfig& config) {
    // 阶段 0：dispatcher 已验证输入张量的 shape、dtype 与 values 长度；
    // 此处再验证格式专用实现的前提，避免内部调用绕过 public API 时产生歧义。
    if (!config.isValid() || config.format != QuantFormat::kMxfp8) {
        throw ReferenceError{"MXFP8 CPU reference 收到了非法量化配置。"};
    }

    // 阶段 1：把方向配置和输入文件描述组合为 QuantizedTensor 的元数据。
    // 这一步尚未计算任何数值；desc 决定 payload 和 local_scales 的精确长度。
    const QuantizedTensorDesc desc = make_mxfp8_desc(input.desc,config);
    const auto expected_payload_bytes = desc.expectedPayloadBytes();
    const auto expected_local_scale_count = desc.expectedLocalScaleCount();

    // expectedPayloadBytes() 与 expectedLocalScaleCount() 同时完成格式、shape、
    // rowwise block 组织和溢出检查。任何一项无法推导都不能继续分配输出数组。
    if (!desc.isMetadataValid() || !expected_payload_bytes.has_value() ||
        !expected_local_scale_count.has_value()) {
        throw ReferenceError{"MXFP8 CPU reference 无法构造合法量化元数据。"};
    }

    // 阶段 2：一次性分配最终结果。MXFP8 每个逻辑元素恰好占一个 payload byte；
    // local_scales 在 tensor 模式有一项，在 block 模式按 rowwise block 排列。
    // MXFP8 没有 NVFP4 式 global_scale，因此必须保持 std::nullopt。
    QuantizedTensor output{
        .desc = desc,
        .payload = std::vector<std::uint8_t>(
            checked_size(*expected_payload_bytes, "payload 字节数"),
            std::uint8_t{0U}),
        .local_scales = std::vector<std::uint8_t>(
            checked_size(*expected_local_scale_count, "local scale 数量"),
            std::uint8_t{0U}),
        .global_scale = std::nullopt,
    };

    // 这些是 host vector 下标使用的 size_t 版本。checked_size() 保证由 uint64
    // 文件元数据转换而来时不会在当前平台截断；row_count * column_count 已由
    // dispatcher 的 values 长度检查间接保证可安全表示。
    const std::size_t row_count =
        checked_size(input.desc.num_rows, "行数");
    const std::size_t column_count =
        checked_size(input.desc.num_cols, "列数");
    const std::size_t block_size =
        static_cast<std::size_t>(config.block_size);

    // 阶段 3：先完成所有 local scale 的计算，再编码任何元素。这样每个 payload
    // 元素在阶段 4 都能按唯一确定的 scale 索引编码；calculate_finite_amax() 也会
    // 在这个阶段拒绝 NaN/Inf，函数不会向调用方返回半成品 QuantizedTensor。
    if (config.scale_mode == ScaleMode::kTensor) {
        // tensor mode 仅保存一个 scale，覆盖整个 row-major values 数组。
        // 线性起点是 0，供非有限值异常报告原始元素位置。
        output.local_scales.front() = make_mxfp8_scale_code(
            std::span<const float>{input.values.data(), input.values.size()},
            0U);
    } else {
        const auto blocks_per_row_u64 = desc.blocksPerRow();
        if (!blocks_per_row_u64.has_value()) {
            throw ReferenceError{
                "MXFP8 block 模式无法推导每行的 block 数量。"};
        }

        const std::size_t blocks_per_row =
            checked_size(*blocks_per_row_u64, "每行 block 数量");

        for (std::size_t row = 0U; row < row_count; ++row) {
            // row_offset 是当前行第 0 列在完整 row-major values 中的线性下标。
            const std::size_t row_offset = row * column_count;

            for (std::size_t block = 0U; block < blocks_per_row; ++block) {
                // block_column_offset 是该 block 在本行中的起始列；它不会跨行。
                const std::size_t block_column_offset = block * block_size;
                // 最后一个 block 可以不足 32 元素。只扫描真实元素即可：逻辑补的
                // 零不会改变 amax，也不应该写入 payload。
                const std::size_t valid_block_elements = std::min(
                    block_size, column_count - block_column_offset);
                // first_element 是当前 block 第一个真实元素的全局线性下标。
                const std::size_t first_element =
                    row_offset + block_column_offset;
                // local_scales 的 rowwise 顺序是先整行第 0 block、第 1 block，
                // 再进入下一行；它必须与后续 payload pass 的索引公式一致。
                const std::size_t scale_index =
                    row * blocks_per_row + block;

                // 尾 block 的 span 只包含真实数据。规范中的零填充只是逻辑概念，
                // 全零不会改变 amax，所以无需真的分配或扫描 padding。
                output.local_scales[scale_index] = make_mxfp8_scale_code(
                    std::span<const float>{
                        input.values.data() + first_element,
                        valid_block_elements,
                    },
                    static_cast<std::uint64_t>(first_element));
            }
        }
    }

    // 阶段 4：逐元素生成 E4M3 payload。block 模式需要再次得到每行 block 数，
    // 以把任意列映射回阶段 3 已写入的 local scale；tensor 模式固定使用索引 0。
    std::size_t blocks_per_row = 0U;
    if (config.scale_mode == ScaleMode::kBlock) {
        blocks_per_row = checked_size(
            *desc.blocksPerRow(), "每行 block 数量");
    }
    for (std::size_t row = 0U; row < row_count; ++row) {
        const std::size_t row_offset = row * column_count;

        for (std::size_t column = 0U; column < column_count; ++column) {
            // linear_index 同时是 input.values 和 MXFP8 output.payload 的下标，
            // 因为两者都采用无 padding 的 row-major 连续布局。
            const std::size_t linear_index = row_offset + column;
            // block 模式的 column / block_size 给出当前元素在该行属于第几个
            // block；row * blocks_per_row 再把它平移到 local_scales 全局下标。
            const std::size_t scale_index =
                config.scale_mode == ScaleMode::kTensor
                    ? 0U
                    : row * blocks_per_row + column / block_size;

            // nearest 模式不读取随机数。stochastic 模式将配置 seed 与全局线性
            // 下标映射为无状态 [0, 1) 随机数，保证未来 CPU/CUDA 可逐元素复现。
            const float uniform_random =
                config.rounding == RoundingMode::kStochastic
                    ? formats::mxfp8_stochastic_uniform_for_element(
                          config.stochastic_seed,
                          static_cast<std::uint64_t>(linear_index))
                    : 0.0F;

            // codec 内部负责 E8M0 decode、value / scale、E4M3 的 RNE 或
            // stochastic rounding、饱和，以及 -0 -> +0 的项目规范化。
            output.payload[linear_index] = formats::encode_mxfp8_element(
                input.values[linear_index],
                output.local_scales[scale_index],
                config.rounding,
                uniform_random);
        }
    }

    // 阶段 5：在返回前再次验证元数据、数组长度和 MXFP8 专属不变量。这个检查
    // 还能防止未来重构索引公式时把不自洽的结果交给 QDWGT 写入器。
    if (!output.isConsistent()) {
        throw ReferenceError{
            "MXFP8 CPU reference 构造出了不自洽的 QuantizedTensor。"};
    }

    return output;
}

HostTensor dequantize_mxfp8_reference(
    const QuantizedTensor& input,
    const DequantizationConfig& config) {
    // 阶段 0：public dispatcher 已完成这两项检查；这里仍保留格式专用的
    // 前置条件验证，使内部调用无法绕过 API 时也不会把其他格式误作 MXFP8。
    if (!input.isConsistent() || !config.isValid() ||
        input.desc.format != QuantFormat::kMxfp8) {
        throw ReferenceError{
            "MXFP8 CPU reference 反量化收到了非法输入或输出配置。"};
    }

    // 阶段 1：反量化不改变矩阵形状，只把 QDWGT 中记录的原始输入 dtype 替换为
    // 调用者所要求的输出物理 dtype。HostTensor 的 values 仍统一持有 FP32，
    // 真正转写为 FP16/BF16/FP32 payload 的工作由 write_dequantized_tensor()
    // 在 QDTENSOR I/O 边界完成。
    TensorDesc output_desc = input.desc.source_desc;
    output_desc.dtype = config.output_type;
    const auto element_count = output_desc.elementCount();
    if (!output_desc.isValid() || !element_count.has_value()) {
        throw ReferenceError{"MXFP8 CPU reference 无法构造合法输出张量描述。"};
    }

    HostTensor output{
        .desc = output_desc,
        .values = std::vector<float>(
            checked_size(*element_count, "反量化元素数量"), 0.0F),
    };

    // 这些 size_t 版本只用于 host vector 下标。input.isConsistent() 已保证
    // payload 的逻辑长度正确，checked_size() 额外防止 uint64_t 元数据在当前
    // 平台截断。
    const std::size_t row_count =
        checked_size(input.desc.source_desc.num_rows, "行数");
    const std::size_t column_count =
        checked_size(input.desc.source_desc.num_cols, "列数");
    const std::size_t block_size =
        static_cast<std::size_t>(input.desc.block_size);

    // 阶段 2：建立与量化阶段完全相同的 local scale 索引规则。tensor 模式中
    // 所有元素共享 local_scales[0]；block 模式中第 (row, column) 个元素使用
    // row * blocks_per_row + column / block_size。尾 block 不含 padding，仍共享
    // 自己对应的一项 scale。
    std::size_t blocks_per_row = 0U;
    if (input.desc.scale_mode == ScaleMode::kBlock) {
        const auto blocks_per_row_u64 = input.desc.blocksPerRow();
        if (!blocks_per_row_u64.has_value()) {
            throw ReferenceError{
                "MXFP8 block 模式无法推导每行的 block 数量。"};
        }

        blocks_per_row = checked_size(
            *blocks_per_row_u64, "每行 block 数量");
    }

    // 阶段 3：payload 与 output.values 都是无 padding 的连续 row-major 数组，
    // 所以 linear_index 同时是两者下标。codec 解码 E4M3 后乘以 E8M0 scale；
    // 若外部 QDWGT 带有合法长度但含 NaN code，IEEE FP32 运算会自然生成 NaN，
    // 这符合 format_spec.md 对特殊编码的解码语义，不应在此静默改写。
    for (std::size_t row = 0U; row < row_count; ++row) {
        const std::size_t row_offset = row * column_count;

        for (std::size_t column = 0U; column < column_count; ++column) {
            const std::size_t linear_index = row_offset + column;
            const std::size_t scale_index =
                input.desc.scale_mode == ScaleMode::kTensor
                    ? 0U
                    : row * blocks_per_row + column / block_size;

            output.values[linear_index] = formats::decode_mxfp8_element(
                input.payload[linear_index], input.local_scales[scale_index]);
        }
    }

    // 阶段 4：在返回前核对 HostTensor 自己的形状、物理 dtype 与 FP32 数组长度。
    // 这与 dispatcher 的输入检查相呼应，能及早发现未来修改索引或元数据逻辑时的
    // 意外错误。
    if (output.values.size() != checked_size(*element_count, "反量化元素数量")) {
        throw ReferenceError{
            "MXFP8 CPU reference 构造出了长度不自洽的反量化结果。"};
    }

    return output;
}

}  // namespace quant_dequant::reference_detail

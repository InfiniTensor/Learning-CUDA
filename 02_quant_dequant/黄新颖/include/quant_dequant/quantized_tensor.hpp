#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "quant_dequant/types.hpp"

namespace quant_dequant {

/**
 * @brief 描述一个已经完成量化、可保存为 QDWGT 文件的张量。
 *
 * 这是 CPU reference 和 CUDA pipeline 在完成量化后共同交付的元数据。
 * 它刻意不持有 CUDA device 指针或 `thrust::device_vector`：那些对象只属于
 * pipeline 的短暂执行过程；本描述和 `QuantizedTensor` 则可安全地传给 I/O、
 * 测试和误差统计模块。
 *
 * `source_desc.dtype` 记录量化前文件中的物理类型，且第一版仅允许 FP16 或
 * FP32。反量化目标类型不属于量化结果本身，应从
 * `DequantizationConfig::output_type` 或反量化 API 的参数取得。
 */
struct QuantizedTensorDesc {
    /** 原始 row-major 矩阵的形状和物理输入类型。 */
    TensorDesc source_desc{};

    /** MXFP8 或 NVFP4，决定 payload 和 local scale 的解释规则。 */
    QuantFormat format{QuantFormat::kUnknown};

    /** MXFP8 可共享整张张量或每个 rowwise block 的 scale；NVFP4 仅允许后者。 */
    ScaleMode scale_mode{ScaleMode::kUnknown};

    /** 产生 payload 时使用的舍入规则，写入文件以支持可复现实验。 */
    RoundingMode rounding{RoundingMode::kUnknown};

    /**
     * @brief stochastic rounding 的确定性种子。
     *
     * nearest 模式没有随机过程，必须为 0；stochastic 模式允许任意
     * `uint64_t` 值（包括 0）。
     */
    std::uint64_t stochastic_seed{0U};

    /**
     * @brief 格式的原生 block size。
     *
     * MXFP8 固定为 32，NVFP4 固定为 16。MXFP8 即使 `scale_mode` 为 tensor，
     * 也保留该值，避免同一格式出现多种不兼容的 header 解释；NVFP4 不支持
     * tensor mode。
     */
    std::uint32_t block_size{0U};

    /** 第一版仅支持按行切分、按行优先排列的局部 scale。 */
    ScaleLayout scale_layout{ScaleLayout::kRowwise};

    /**
     * @brief 返回逻辑元素个数。
     *
     * @return `num_rows * num_cols`；零维或乘法溢出时返回空值。
     */
    [[nodiscard]] constexpr std::optional<std::uint64_t> elementCount() const noexcept {
        return source_desc.elementCount();
    }

    /**
     * @brief 返回最终 packed payload 的精确字节数。
     *
     * MXFP8 的每个 E4M3 code 占一个字节；NVFP4 的两个 E2M1 nibble
     * 共用一个字节，因此奇数个元素时最后一个字节的高 nibble 保留为零。
     *
     * @return 合法形状和已知格式时返回 payload 字节数；否则返回空值。
     */
    [[nodiscard]] constexpr std::optional<std::uint64_t> expectedPayloadBytes() const noexcept {
        const auto element_count = elementCount();
        if (!element_count.has_value()) {
            return std::nullopt;
        }

        switch (format) {
            case QuantFormat::kMxfp8:
                return element_count;

            case QuantFormat::kNvfp4:
                // 用除法和余数计算 ceil(n / 2)，避免先做 n + 1 的上溢风险。
                return (*element_count / 2U) + (*element_count % 2U);

            case QuantFormat::kUnknown:
                return std::nullopt;
        }

        return std::nullopt;
    }

    /**
     * @brief 返回 rowwise block 模式下一行所需的 local scale 数量。
     *
     * @return block 模式下返回 `ceil(num_cols / block_size)`；格式、形状或
     * block size 非法时返回空值。
     */
    [[nodiscard]] constexpr std::optional<std::uint64_t> blocksPerRow() const noexcept {
        if (scale_mode != ScaleMode::kBlock ||
            !is_valid_scale_mode(format, scale_mode) ||
            !is_valid_block_size(format, block_size) ||
            !source_desc.elementCount().has_value()) {
            return std::nullopt;
        }

        const std::uint64_t block_size_u64 = static_cast<std::uint64_t>(block_size);
        return (source_desc.num_cols / block_size_u64) +
            (source_desc.num_cols % block_size_u64 == 0U ? 0U : 1U);
    }

    /**
     * @brief 返回 local scale 字节数组中应有的元素数量。
     *
     * 两种已支持格式的 local scale 都恰好编码为一个字节：MXFP8 是 E8M0，
     * NVFP4 是 E4M3。因此 scale 数量也就是 scale 字节数。具体编码类型由
     * `local_scale_type(format)` 推导，不在此结构中重复保存。
     *
     * @return tensor 模式返回 1；block 模式返回
     * `num_rows * ceil(num_cols / block_size)`；非法元数据或溢出时返回空值。
     */
    [[nodiscard]] constexpr std::optional<std::uint64_t> expectedLocalScaleCount() const noexcept {
        if (!is_valid_block_size(format, block_size) ||
            !is_valid_scale_mode(format, scale_mode) ||
            !source_desc.elementCount().has_value()) {
            return std::nullopt;
        }

        if (scale_mode == ScaleMode::kTensor) {
            return 1U;
        }

        const auto blocks_per_row = blocksPerRow();
        if (!blocks_per_row.has_value()) {
            return std::nullopt;
        }

        constexpr std::uint64_t kMaxValue = std::numeric_limits<std::uint64_t>::max();
        if (source_desc.num_rows > kMaxValue / *blocks_per_row) {
            return std::nullopt;
        }

        return source_desc.num_rows * *blocks_per_row;
    }

    /**
     * @brief 判断该格式在内存语义上是否需要 `global_scale`。
     *
     * NVFP4 的反量化公式包含一个解码方向的 FP32 global scale；MXFP8 不使用
     * 它。QDWGT 文件中 MXFP8 的相应 header 槽仍写固定的 1.0F，属于文件
     * 布局哨兵，而不是本对象的一项语义数据。
     *
     * @return 仅 NVFP4 返回 true。
     */
    [[nodiscard]] constexpr bool usesGlobalScale() const noexcept {
        return format == QuantFormat::kNvfp4;
    }

    /**
     * @brief 验证不依赖 payload 内容的量化元数据。
     *
     * @return 格式、输入描述、scale 组织和舍入元数据均符合第一版规范时返回
     * true。
     */
    [[nodiscard]] constexpr bool isMetadataValid() const noexcept {
        const bool valid_format = format == QuantFormat::kMxfp8 ||
                                  format == QuantFormat::kNvfp4;
        const bool valid_rounding = rounding == RoundingMode::kNearest ||
                                    rounding == RoundingMode::kStochastic;

        return valid_format && is_valid_scale_mode(format, scale_mode) && valid_rounding &&
               source_desc.isValid() && is_supported_input_dtype(source_desc.dtype) &&
               is_valid_block_size(format, block_size) &&
               scale_layout == ScaleLayout::kRowwise &&
               (rounding != RoundingMode::kNearest || stochastic_seed == 0U) &&
               expectedPayloadBytes().has_value() &&
               expectedLocalScaleCount().has_value();
    }
};

/**
 * @brief CPU reference 与 CUDA quantize pipeline 的统一 host 侧量化结果。
 *
 * 所有 vector 都拥有 host 内存。CUDA 路径在 device 上完成计算并 D2H 拷回后，
 * 构造同一个对象；CPU reference 则直接构造它。因此后续 QDWGT I/O、逐字节
 * CPU/CUDA 对照和日志模块都只需处理一种结果类型。
 *
 * `payload` 的存储约定由 `desc.format` 决定：
 * - MXFP8：一个 E4M3 code 一个字节；
 * - NVFP4：偶数线性下标在低 nibble，奇数线性下标在高 nibble。
 *
 * `local_scales` 每项均为一个编码字节，但 MXFP8 解释为 E8M0，NVFP4 解释为
 * E4M3。`global_scale` 只在 NVFP4 中存在，且应为有限正的 FP32 值。
 */
struct QuantizedTensor {
    /** 与所有后续数组共同解释的不可缺少元数据。 */
    QuantizedTensorDesc desc{};

    /** 按 row-major 线性顺序存储的真实位宽 packed 元素 code。 */
    std::vector<std::uint8_t> payload{};

    /** 按 `desc.scale_layout` 排列的格式特有局部 scale code。 */
    std::vector<std::uint8_t> local_scales{};

    /**
     * @brief NVFP4 的解码方向 FP32 global scale；MXFP8 必须为 `std::nullopt`。
     *
     * 这样在内存模型中区分“不存在 global scale”的 MXFP8 与“数值恰好等于
     * 1.0F”的 NVFP4。序列化时由 `quantized_io` 将 MXFP8 的物理 header 槽写为
     * 固定 1.0F。
     */
    std::optional<float> global_scale{};

    /**
     * @brief 验证元数据、数组长度和格式特有的内存不变量。
     *
     * 本函数验证容器可以安全地被后续量化文件写入器和反量化器解释，但不会
     * 判断每一个 E4M3/E2M1 code 是否数值“正常”：NaN code 仍是可表示的低
     * 精度 bit pattern，应由格式 codec 决定它的数值语义。
     *
     * @return 完整对象自洽时返回 true。
     */
    [[nodiscard]] bool isConsistent() const noexcept {
        if (!desc.isMetadataValid()) {
            return false;
        }

        const auto expected_payload_bytes = desc.expectedPayloadBytes();
        const auto expected_local_scale_count = desc.expectedLocalScaleCount();
        constexpr std::uint64_t kMaxSize =
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());

        if (!expected_payload_bytes.has_value() ||
            !expected_local_scale_count.has_value() ||
            *expected_payload_bytes > kMaxSize ||
            *expected_local_scale_count > kMaxSize ||
            payload.size() != static_cast<std::size_t>(*expected_payload_bytes) ||
            local_scales.size() != static_cast<std::size_t>(*expected_local_scale_count)) {
            return false;
        }

        if (desc.format == QuantFormat::kMxfp8) {
            return !global_scale.has_value();
        }

        if (!global_scale.has_value() || !std::isfinite(*global_scale) ||
            *global_scale <= 0.0F) {
            return false;
        }

        const auto element_count = desc.elementCount();
        const bool has_odd_element_count = element_count.has_value() &&
                                           (*element_count % 2U != 0U);
        // 奇数个 NVFP4 元素时，末尾字节的高 nibble 不属于任何逻辑元素。
        return !has_odd_element_count ||
               ((payload.back() & static_cast<std::uint8_t>(0xF0U)) == 0U);
    }
};

}  // namespace quant_dequant

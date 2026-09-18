#pragma once
#include<cstddef>
#include<cstdint>
#include<limits>
#include<optional>
#include<string>
#include<string_view>

namespace quant_dequant
{

/**
 * @brief MXFP8 rowwise block 的固定元素数量。
 */
inline constexpr std::uint32_t kMxfp8BlockSize = 32U;

/**
 * @brief NVFP4 rowwise block 的固定元素数量。
 */
inline constexpr std::uint32_t kNvfp4BlockSize = 16U;

/**
 * @brief 文件格式版本 1 使用的小端字节序标记。
 */
inline constexpr std::uint8_t kLittleEndianByteOrder = 1U;   
    

/**
 * @brief 普通张量元素的数据类型。
 *
 * 枚举值与 `docs/file_format.md` 中 `QDTENSOR` header 的 `dtype`
 * 字段一一对应，不能在不提升文件格式版本的情况下修改。
 */

enum class DType:std::uint8_t{
    /** 未初始化或非法值。 */
    kUnknown = 0U,

    /** IEEE 754 binary16。 */
    kFloat16 = 1U,

    /** bfloat16。 */
    kBFloat16 = 2U,

    /** IEEE 754 binary32。 */
    kFloat32 = 3U,
};
/**
 * @brief 低精度量化格式。
 *
 * 枚举值与 `QDWGT` header 的 `format` 字段一一对应。
 */
enum class QuantFormat : std::uint8_t {
    /** 未初始化或非法值。 */
    kUnknown = 0U,

    /** MXFP8：E4M3 元素和 E8M0 局部 scale。 */
    kMxfp8 = 1U,

    /** NVFP4：E2M1 元素、E4M3 局部 scale 和 FP32 全局 scale。 */
    kNvfp4 = 2U,
};
/**
 * @brief scale 的共享粒度。
 */
enum class ScaleMode : std::uint8_t {
    /** 未初始化或非法值。 */
    kUnknown = 0U,

    /** 整张矩阵共享 scale。 */
    kTensor = 1U,

    /** 每个 rowwise block 独立保存 scale。 */
    kBlock = 2U,
};
/**
 * @brief 元素编码时使用的舍入策略。
 */
enum class RoundingMode : std::uint8_t {
    /** 未初始化或非法值。 */
    kUnknown = 0U,

    /** Round-to-nearest, ties-to-even。 */
    kNearest = 1U,

    /** 按相邻可表示值距离进行概率选择。 */
    kStochastic = 2U,
};
/**
 * @brief `QDTENSOR` 文件的张量用途。
 */
enum class TensorRole : std::uint8_t {
    /** 输入张量。 */
    kInput = 0U,

    /** CUDA 反量化后的输出张量。 */
    kDequantizedOutput = 1U,
};
/**
 * @brief 量化权重文件中局部 scale 的编码格式。
 */
enum class LocalScaleType : std::uint8_t {
    /** 未初始化或非法值。 */
    kUnknown = 0U,

    /** MXFP8 使用的 E8M0 scale。 */
    kE8M0 = 1U,

    /** NVFP4 使用的 E4M3 scale。 */
    kE4M3 = 2U,
};
/**
 * @brief 局部 scale 的线性排列方式。
 */
enum class ScaleLayout : std::uint8_t {
    /** 未初始化或非法值。 */
    kUnknown = 0U,

    /**
     * @brief 每行独立分 block，随后按行优先顺序排列 scale。
     */
    kRowwise = 1U,
};
/**
 * @brief 返回 dtype 的单个元素字节数。
 *
 * @param dtype 要查询的元素类型。
 * @return 合法类型返回对应字节数；未知类型返回 0。
 */
[[nodiscard]] constexpr std::size_t dtype_size_bytes(const DType dtype) noexcept {
    switch (dtype) {
        case DType::kFloat16:
        case DType::kBFloat16:
            return 2U;

        case DType::kFloat32:
            return 4U;

        case DType::kUnknown:
            return 0U;
    }

    return 0U;
}
/**
 * @brief 判断 dtype 是否可作为输入张量类型。
 *
 * @param dtype 要检查的类型。
 * @return 仅 FP16 和 FP32 返回 true。
 */
[[nodiscard]] constexpr bool is_supported_input_dtype(const DType dtype) noexcept {
    return dtype == DType::kFloat16 || dtype == DType::kFloat32;
}
/**
 * @brief 判断 dtype 是否可作为反量化输出类型。
 *
 * @param dtype 要检查的类型。
 * @return FP16、BF16、FP32 返回 true。
 */
[[nodiscard]] constexpr bool is_supported_output_dtype(const DType dtype) noexcept {
    return dtype == DType::kFloat16 || dtype == DType::kBFloat16 ||
           dtype == DType::kFloat32;
}
/**
 * @brief 返回量化格式的默认且唯一合法 block size。
 *
 * @param format 量化格式。
 * @return MXFP8 返回 32，NVFP4 返回 16；未知格式返回 0。
 */
[[nodiscard]] constexpr std::uint32_t default_block_size(
    const QuantFormat format) noexcept {
    switch (format) {
        case QuantFormat::kMxfp8:
            return kMxfp8BlockSize;

        case QuantFormat::kNvfp4:
            return kNvfp4BlockSize;

        case QuantFormat::kUnknown:
            return 0U;
    }

    return 0U;
}
/**
 * @brief 判断 block size 是否符合当前量化格式。
 *
 * MXFP8 的 tensor mode 也保留原生 block size，以避免同一种格式出现多种
 * 不兼容解释。严格 NVFP4 只允许 block mode。
 *
 * @param format 量化格式。
 * @param block_size 待校验的 block size。
 * @return block size 与格式匹配时返回 true。
 */
[[nodiscard]] constexpr bool is_valid_block_size(const QuantFormat format,
                                                 const std::uint32_t block_size) noexcept {
    return block_size != 0U && block_size == default_block_size(format);
}

/**
 * @brief 判断量化格式与 scale 共享粒度是否是受支持的组合。
 *
 * MXFP8 支持 tensor 与 rowwise block 两种 scale 组织。NVFP4 遵循标准的
 * 分层 block scaling：一个 FP32 global scale 加上每个 16 元素 block 的
 * E4M3 local scale，因此不接受 tensor mode。
 *
 * @param format 待检查的低精度格式。
 * @param scale_mode 待检查的 scale 共享粒度。
 * @return 该格式可使用该 scale 模式时返回 true。
 */
[[nodiscard]] constexpr bool is_valid_scale_mode(const QuantFormat format,
                                                 const ScaleMode scale_mode) noexcept {
    switch (format) {
        case QuantFormat::kMxfp8:
            return scale_mode == ScaleMode::kTensor ||
                   scale_mode == ScaleMode::kBlock;

        case QuantFormat::kNvfp4:
            return scale_mode == ScaleMode::kBlock;

        case QuantFormat::kUnknown:
            return false;
    }

    return false;
}
/**
 * @brief 返回量化 payload 中一个逻辑元素占用的 bit 数。
 *
 * @param format 量化格式。
 * @return MXFP8 返回 8，NVFP4 返回 4；未知格式返回 0。
 */
[[nodiscard]] constexpr std::uint8_t payload_element_bits(
    const QuantFormat format) noexcept {
    switch (format) {
        case QuantFormat::kMxfp8:
            return 8U;

        case QuantFormat::kNvfp4:
            return 4U;

        case QuantFormat::kUnknown:
            return 0U;
    }

    return 0U;
}
/**
 * @brief 返回量化格式所使用的局部 scale 类型。
 *
 * @param format 量化格式。
 * @return MXFP8 返回 E8M0，NVFP4 返回 E4M3；未知格式返回 Unknown。
 */
[[nodiscard]] constexpr LocalScaleType local_scale_type(
    const QuantFormat format) noexcept {
    switch (format) {
        case QuantFormat::kMxfp8:
            return LocalScaleType::kE8M0;

        case QuantFormat::kNvfp4:
            return LocalScaleType::kE4M3;

        case QuantFormat::kUnknown:
            return LocalScaleType::kUnknown;
    }

    return LocalScaleType::kUnknown;
}
/**
 * @brief 将 dtype 转换为配置、日志中使用的稳定文本。
 *
 * @param dtype 待转换的类型。
 * @return 不拥有内存的静态字符串视图。
 */
[[nodiscard]] constexpr std::string_view to_string(const DType dtype) noexcept {
    switch (dtype) {
        case DType::kFloat16:
            return "fp16";

        case DType::kBFloat16:
            return "bf16";

        case DType::kFloat32:
            return "fp32";

        case DType::kUnknown:
            return "unknown";
    }

    return "unknown";
}

/**
 * @brief 将量化格式转换为配置、日志中使用的稳定文本。
 *
 * @param format 待转换的量化格式。
 * @return 不拥有内存的静态字符串视图。
 */
[[nodiscard]] constexpr std::string_view to_string(const QuantFormat format) noexcept {
    switch (format) {
        case QuantFormat::kMxfp8:
            return "mxfp8";

        case QuantFormat::kNvfp4:
            return "nvfp4";

        case QuantFormat::kUnknown:
            return "unknown";
    }

    return "unknown";
}

/**
 * @brief 将 scale mode 转换为配置、日志中使用的稳定文本。
 *
 * @param scale_mode 待转换的 scale mode。
 * @return 不拥有内存的静态字符串视图。
 */
[[nodiscard]] constexpr std::string_view to_string(const ScaleMode scale_mode) noexcept {
    switch (scale_mode) {
        case ScaleMode::kTensor:
            return "tensor";

        case ScaleMode::kBlock:
            return "block";

        case ScaleMode::kUnknown:
            return "unknown";
    }

    return "unknown";
}

/**
 * @brief 将舍入模式转换为配置、日志中使用的稳定文本。
 *
 * @param rounding_mode 待转换的舍入模式。
 * @return 不拥有内存的静态字符串视图。
 */
[[nodiscard]] constexpr std::string_view to_string(
    const RoundingMode rounding_mode) noexcept {
    switch (rounding_mode) {
        case RoundingMode::kNearest:
            return "nearest";

        case RoundingMode::kStochastic:
            return "stochastic";

        case RoundingMode::kUnknown:
            return "unknown";
    }

    return "unknown";
}
/**
 * @brief 描述一个不拥有数据的 row-major 二维张量。
 *
 * 该结构只保存元数据；host 端元素由 `std::vector<float>` 持有，
 * device 端元素由 `thrust::device_vector<float>` 持有。
 */
struct TensorDesc {
    /** 矩阵行数，必须大于 0。 */
    std::uint64_t num_rows{0U};

    /** 矩阵列数，必须大于 0。 */
    std::uint64_t num_cols{0U};

    /** 文件中的物理元素类型。 */
    DType dtype{DType::kUnknown};

    /**
     * @brief 返回元素总数，并检测乘法溢出。
     *
     * @return 合法形状时返回 `num_rows * num_cols`；零维或溢出时返回空值。
     */
    [[nodiscard]] constexpr std::optional<std::uint64_t> elementCount() const noexcept {
        if (num_rows == 0U || num_cols == 0U) {
            return std::nullopt;
        }

        constexpr std::uint64_t kMaxValue =
            std::numeric_limits<std::uint64_t>::max();

        if (num_rows > kMaxValue / num_cols) {
            return std::nullopt;
        }

        return num_rows * num_cols;
    }

    /**
     * @brief 返回连续 row-major payload 所需字节数，并检测溢出。
     *
     * @return 合法 dtype 和形状时返回字节数；否则返回空值。
     */
    [[nodiscard]] constexpr std::optional<std::uint64_t> dataBytes() const noexcept {
        const auto element_count = elementCount();
        const std::size_t element_size = dtype_size_bytes(dtype);

        if (!element_count.has_value() || element_size == 0U) {
            return std::nullopt;
        }

        constexpr std::uint64_t kMaxValue =
            std::numeric_limits<std::uint64_t>::max();
        const auto element_size_u64 = static_cast<std::uint64_t>(element_size);

        if (*element_count > kMaxValue / element_size_u64) {
            return std::nullopt;
        }

        return *element_count * element_size_u64;
    }

    /**
     * @brief 判断张量描述是否可用于普通张量文件。
     *
     * @return 形状、dtype 与 payload 字节数都合法时返回 true。
     */
    [[nodiscard]] constexpr bool isValid() const noexcept {
        return elementCount().has_value() && dataBytes().has_value();
    }
};


/**
 * @brief 仅描述“原始浮点张量如何量化”为低精度 code 的数值配置。
 *
 * 此结构不包含输入 dtype：输入 dtype 从 QDTENSOR header 取得；也不包含
 * 反量化输出类型和日志信息。因此 CPU reference、CUDA quantize kernel 与
 * 单独的量化命令都可以只依赖它。
 */
struct QuantizationConfig {
    /** 目标低精度格式。 */
    QuantFormat format{QuantFormat::kUnknown};

    /** 格式指定的原生 block size。 */
    std::uint32_t block_size{0U};

    /** MXFP8 支持整张张量或每个 rowwise block 共享 scale；NVFP4 仅支持 block。 */
    ScaleMode scale_mode{ScaleMode::kUnknown};

    /** 元素编码时使用的舍入模式。 */
    RoundingMode rounding{RoundingMode::kUnknown};

    /** stochastic rounding 的确定性随机种子；nearest 模式下必须为 0。 */
    std::uint64_t stochastic_seed{0U};

    /**
     * @brief 判断量化阶段所需的数值参数是否自洽。
     *
     * @return 格式、block size、scale mode、舍入模式及 seed 均合法时返回 true。
     */
    [[nodiscard]] constexpr bool isValid() const noexcept {
        const bool valid_format = format == QuantFormat::kMxfp8 ||
                                  format == QuantFormat::kNvfp4;
        const bool valid_rounding = rounding == RoundingMode::kNearest ||
                                    rounding == RoundingMode::kStochastic;

        return valid_format && is_valid_scale_mode(format, scale_mode) && valid_rounding &&
               is_valid_block_size(format, block_size) &&
               (rounding != RoundingMode::kNearest || stochastic_seed == 0U);
    }
};

/**
 * @brief 仅描述“低精度 code 如何反量化并输出”的数值配置。
 *
 * 量化权重文件本身并不固化输出类型；同一份 QDWGT 可按此配置分别反量化为
 * FP16、BF16 或 FP32。
 */
struct DequantizationConfig {
    /** 反量化输出 QDTENSOR payload 的物理类型。 */
    DType output_type{DType::kUnknown};

    /**
     * @brief 判断反量化阶段的输出类型是否受支持。
     *
     * @return FP16、BF16 或 FP32 时返回 true。
     */
    [[nodiscard]] constexpr bool isValid() const noexcept {
        return is_supported_output_dtype(output_type);
    }
};

/**
 * @brief 描述一次 app 运行的报告元数据，不参与任何数值计算。
 */
struct ReportConfig {
    /** 用于日志和实验报告的目标 GPU 名称。 */
    std::string target_gpu{};

    /**
     * @brief 判断报告元数据是否可写入一次完整 app 运行的日志。
     *
     * @return GPU 名称非空时返回 true。
     */
    [[nodiscard]] bool isValid() const noexcept {
        return !target_gpu.empty();
    }
};

/**
 * @brief `apps/main.cpp` 专用的完整流程配置。
 *
 * 它只是配置文件与命令行 app 的编排对象：app 可以连续执行量化、写 QDWGT、
 * 反量化和日志记录。数值层绝不能把整个对象传给单方向接口；量化接口只接收
 * `QuantizationConfig`，反量化接口只接收 `DequantizationConfig`。
 */
struct AppConfig {
    /** 完整流程中量化阶段所需的参数。 */
    QuantizationConfig quantization{};

    /** 完整流程中反量化阶段所需的参数。 */
    DequantizationConfig dequantization{};

    /** 完整流程中日志和报告所需的非数值元数据。 */
    ReportConfig report{};

    /**
     * @brief 判断完整 app 流程所需的三类配置是否都合法。
     *
     * @return 三个子配置均合法时返回 true。
     */
    [[nodiscard]] bool isValid() const noexcept {
        return quantization.isValid() && dequantization.isValid() &&
               report.isValid();
    }
};

static_assert(static_cast<std::uint8_t>(DType::kFloat16) == 1U);
static_assert(static_cast<std::uint8_t>(DType::kBFloat16) == 2U);
static_assert(static_cast<std::uint8_t>(DType::kFloat32) == 3U);
static_assert(static_cast<std::uint8_t>(QuantFormat::kMxfp8) == 1U);
static_assert(static_cast<std::uint8_t>(QuantFormat::kNvfp4) == 2U);
} // namespace quant_dequant

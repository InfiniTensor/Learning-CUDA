#include "quant_dequant/quantized_io.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace quant_dequant {
namespace {

static_assert(sizeof(float) == sizeof(std::uint32_t));
static_assert(std::numeric_limits<float>::is_iec559);

/** QDWGT v1 固定 header 的字节数。 */
constexpr std::size_t kQuantizedHeaderBytes = 128U;

/** QDWGT v1 文件格式版本。 */
constexpr std::uint16_t kQuantizedFileVersion = 1U;

/** QDWGT v1 写入器使用的固定 payload offset。 */
constexpr std::uint64_t kCanonicalPayloadOffset = 128U;

/** QDWGT v1 的 section 对齐粒度。 */
constexpr std::uint64_t kSectionAlignment = 8U;

/** NVFP4 的 flags bit 0：最后一个 payload 字节高 nibble 已清零。 */
constexpr std::uint32_t kNvfp4TailNibbleZeroFlag = 1U;

/** QDWGT 文件 magic，ASCII `QDWGT` 后补三个零字节。 */
constexpr std::array<std::uint8_t, 8U> kQuantizedMagic{
    'Q', 'D', 'W', 'G', 'T', 0U, 0U, 0U,
};

/** QDWGT header 的固定字段 offset。 */
constexpr std::size_t kMagicOffset = 0U;
constexpr std::size_t kVersionOffset = 8U;
constexpr std::size_t kHeaderBytesOffset = 10U;
constexpr std::size_t kByteOrderOffset = 12U;
constexpr std::size_t kFormatOffset = 13U;
constexpr std::size_t kScaleModeOffset = 14U;
constexpr std::size_t kRoundingOffset = 15U;
constexpr std::size_t kSourceDTypeOffset = 16U;
constexpr std::size_t kPayloadElementBitsOffset = 17U;
constexpr std::size_t kLocalScaleTypeOffset = 18U;
constexpr std::size_t kScaleLayoutOffset = 19U;
constexpr std::size_t kBlockSizeOffset = 20U;
constexpr std::size_t kRowsOffset = 24U;
constexpr std::size_t kColsOffset = 32U;
constexpr std::size_t kElementCountOffset = 40U;
constexpr std::size_t kPayloadOffsetOffset = 48U;
constexpr std::size_t kPayloadBytesOffset = 56U;
constexpr std::size_t kLocalScaleOffsetOffset = 64U;
constexpr std::size_t kLocalScaleCountOffset = 72U;
constexpr std::size_t kLocalScaleBytesOffset = 80U;
constexpr std::size_t kGlobalScaleOffset = 88U;
constexpr std::size_t kFlagsOffset = 92U;
constexpr std::size_t kStochasticSeedOffset = 96U;

/**
 * @brief 保存已完成 header 校验、但尚未读取 section 内容的 QDWGT 元数据。
 */
struct ParsedQuantizedHeader {
    QuantizedTensorDesc desc{};
    std::uint64_t payload_offset{0U};
    std::uint64_t payload_bytes{0U};
    std::uint64_t local_scale_offset{0U};
    std::uint64_t local_scale_bytes{0U};
    std::optional<float> global_scale{};
};

/**
 * @brief 生成统一的 QuantizedIoError 文本。
 *
 * @param file_path 发生错误的文件路径。
 * @param detail 具体错误说明。
 * @return 可传递给 `std::runtime_error` 的完整错误文本。
 */
[[nodiscard]] std::string make_error_message(
    const std::filesystem::path& file_path,
    const std::string_view detail) {
    return "量化权重文件 \"" + file_path.string() + "\"：" +
           std::string{detail};
}

/**
 * @brief 抛出带文件路径的 QuantizedIoError。
 *
 * @param file_path 发生错误的文件路径。
 * @param detail 具体错误说明。
 */
[[noreturn]] void throw_quantized_io_error(
    const std::filesystem::path& file_path,
    std::string detail) {
    throw QuantizedIoError(file_path, std::move(detail));
}

/**
 * @brief 从 little-endian 字节区读取无符号整数。
 *
 * @tparam UInt 目标无符号整数类型。
 * @param bytes 输入字节区。
 * @param offset 整数起始 offset。
 * @return 解码后的无符号整数。
 */
template <typename UInt>
[[nodiscard]] UInt load_unsigned_le(
    const std::span<const std::uint8_t> bytes,
    const std::size_t offset) noexcept {
    static_assert(std::is_unsigned_v<UInt>);

    UInt result{0U};
    for (std::size_t index = 0U; index < sizeof(UInt); ++index) {
        result |= static_cast<UInt>(bytes[offset + index])
                  << static_cast<unsigned int>(index * 8U);
    }

    return result;
}

/**
 * @brief 向 little-endian 字节区写入无符号整数。
 *
 * @tparam UInt 源无符号整数类型。
 * @param bytes 输出字节区。
 * @param offset 整数起始 offset。
 * @param value 要编码的整数。
 */
template <typename UInt>
void store_unsigned_le(std::span<std::uint8_t> bytes,
                       const std::size_t offset,
                       const UInt value) noexcept {
    static_assert(std::is_unsigned_v<UInt>);

    for (std::size_t index = 0U; index < sizeof(UInt); ++index) {
        bytes[offset + index] = static_cast<std::uint8_t>(
            value >> static_cast<unsigned int>(index * 8U));
    }
}

/**
 * @brief 返回向上对齐到 8 字节后的值，并检测加法溢出。
 *
 * @param value 待对齐的非负字节 offset。
 * @param file_path 当前文件路径，用于错误信息。
 * @return 满足 `result % 8 == 0` 且 `result >= value` 的 offset。
 * @throws QuantizedIoError 对齐所需加法溢出时抛出。
 */
[[nodiscard]] std::uint64_t align_up_to_section_boundary(
    const std::uint64_t value,
    const std::filesystem::path& file_path) {
    constexpr std::uint64_t kPaddingLimit = kSectionAlignment - 1U;
    if (value > std::numeric_limits<std::uint64_t>::max() - kPaddingLimit) {
        throw_quantized_io_error(file_path, "section offset 对齐时发生 uint64 溢出。");
    }

    return (value + kPaddingLimit) & ~kPaddingLimit;
}

/**
 * @brief 安全计算两个 uint64 的和。
 *
 * @param left 加数。
 * @param right 加数。
 * @param file_path 当前文件路径，用于错误信息。
 * @param context 错误信息中说明该和的用途。
 * @return `left + right`。
 * @throws QuantizedIoError 相加溢出时抛出。
 */
[[nodiscard]] std::uint64_t checked_add(
    const std::uint64_t left,
    const std::uint64_t right,
    const std::filesystem::path& file_path,
    const std::string_view context) {
    if (left > std::numeric_limits<std::uint64_t>::max() - right) {
        throw_quantized_io_error(file_path,
                                 std::string{context} + " 发生 uint64 溢出。");
    }

    return left + right;
}

/**
 * @brief 检查 uint64 数值能否安全转换为 host 容器使用的 size_t。
 *
 * @param value 待转换数值。
 * @param file_path 当前文件路径，用于错误信息。
 * @param field_name 对应 header 字段名。
 * @return 可安全用于 vector 大小和 stream I/O 的 size_t。
 */
[[nodiscard]] std::size_t checked_size_cast(
    const std::uint64_t value,
    const std::filesystem::path& file_path,
    const std::string_view field_name) {
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw_quantized_io_error(
            file_path,
            "字段 \"" + std::string{field_name} +
                "\" 超出当前 host 的 size_t 表示范围。");
    }

    return static_cast<std::size_t>(value);
}

/**
 * @brief 检查 uint64 offset 能否用于 `std::istream::seekg`。
 *
 * @param offset 文件内目标 offset。
 * @param file_path 当前文件路径，用于错误信息。
 * @param field_name 对应 header 字段名。
 * @return 可安全转换的 streamoff。
 */
[[nodiscard]] std::streamoff checked_streamoff_cast(
    const std::uint64_t offset,
    const std::filesystem::path& file_path,
    const std::string_view field_name) {
    if (offset > static_cast<std::uint64_t>(
                     std::numeric_limits<std::streamoff>::max())) {
        throw_quantized_io_error(
            file_path,
            "字段 \"" + std::string{field_name} +
                "\" 超出 std::streamoff 表示范围。");
    }

    return static_cast<std::streamoff>(offset);
}

/**
 * @brief 从二进制流读取指定数量字节。
 *
 * @param input_stream 已打开的二进制输入流。
 * @param bytes 输出字节区。
 * @param file_path 当前文件路径，用于错误信息。
 * @throws QuantizedIoError 文件截断或 I/O 失败时抛出。
 */
void read_exact(std::ifstream& input_stream,
                const std::span<std::uint8_t> bytes,
                const std::filesystem::path& file_path) {
    if (bytes.size() >
        static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw_quantized_io_error(file_path, "读取长度超出 std::streamsize 范围。");
    }

    input_stream.read(reinterpret_cast<char*>(bytes.data()),
                      static_cast<std::streamsize>(bytes.size()));
    if (!input_stream) {
        throw_quantized_io_error(file_path, "文件截断或读取失败。");
    }
}

/**
 * @brief 向二进制流写入指定数量字节。
 *
 * @param output_stream 已打开的二进制输出流。
 * @param bytes 输入字节区。
 * @param file_path 当前文件路径，用于错误信息。
 * @throws QuantizedIoError 写入失败时抛出。
 */
void write_exact(std::ofstream& output_stream,
                 const std::span<const std::uint8_t> bytes,
                 const std::filesystem::path& file_path) {
    if (bytes.size() >
        static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw_quantized_io_error(file_path, "写入长度超出 std::streamsize 范围。");
    }

    output_stream.write(reinterpret_cast<const char*>(bytes.data()),
                        static_cast<std::streamsize>(bytes.size()));
    if (!output_stream) {
        throw_quantized_io_error(file_path, "写入文件失败。");
    }
}

/**
 * @brief 移动输入流到已通过范围验证的 section 起点。
 *
 * @param input_stream 已打开的二进制输入流。
 * @param offset section 的文件内起点。
 * @param file_path 当前文件路径，用于错误信息。
 * @param field_name 对应 header 字段名。
 * @throws QuantizedIoError 定位失败时抛出。
 */
void seek_to(std::ifstream& input_stream,
             const std::uint64_t offset,
             const std::filesystem::path& file_path,
             const std::string_view field_name) {
    input_stream.clear();
    input_stream.seekg(checked_streamoff_cast(offset, file_path, field_name),
                       std::ios::beg);
    if (!input_stream) {
        throw_quantized_io_error(
            file_path,
            "无法定位到字段 \"" + std::string{field_name} + "\" 指定的 section。");
    }
}

/**
 * @brief 验证一个 section 是否完全落在文件范围内。
 *
 * @param offset section 起始 offset。
 * @param bytes section 长度。
 * @param file_size 文件总长度。
 * @param file_path 当前文件路径，用于错误信息。
 * @param section_name section 的说明名称。
 */
void validate_section_range(
    const std::uint64_t offset,
    const std::uint64_t bytes,
    const std::uint64_t file_size,
    const std::filesystem::path& file_path,
    const std::string_view section_name) {
    if (offset > file_size || bytes > file_size - offset) {
        throw_quantized_io_error(
            file_path,
            std::string{section_name} + " section 超出文件范围。");
    }
}

/**
 * @brief 从 header 解析 QDWGT v1 的全局 scale 字段。
 *
 * @param header 完整固定 header。
 * @return 与文件格式无关的原始 FP32 值。
 */
[[nodiscard]] float load_float32_le(
    const std::span<const std::uint8_t> header) noexcept {
    return std::bit_cast<float>(
        load_unsigned_le<std::uint32_t>(header, kGlobalScaleOffset));
}

/**
 * @brief 向 header 写入一个 little-endian FP32 bit pattern。
 *
 * @param header 完整固定 header。
 * @param value 要写入的 FP32 值。
 */
void store_float32_le(std::span<std::uint8_t> header,
                      const float value) noexcept {
    store_unsigned_le<std::uint32_t>(
        header, kGlobalScaleOffset, std::bit_cast<std::uint32_t>(value));
}

/**
 * @brief 根据内存量化结果生成 QDWGT v1 的 canonical header。
 *
 * @param tensor 已通过 `isConsistent()` 校验的量化结果。
 * @param payload_offset 将写入 header 的 payload 起点。
 * @param local_scale_offset 将写入 header 的 local scale 起点。
 * @return 完整 128 字节 little-endian header。
 */
[[nodiscard]] std::array<std::uint8_t, kQuantizedHeaderBytes>
make_quantized_header(const QuantizedTensor& tensor,
                      const std::uint64_t payload_offset,
                      const std::uint64_t local_scale_offset) {
    std::array<std::uint8_t, kQuantizedHeaderBytes> header{};
    const QuantizedTensorDesc& desc = tensor.desc;
    const std::uint64_t element_count = *desc.elementCount();
    const std::uint64_t payload_bytes = *desc.expectedPayloadBytes();
    const std::uint64_t local_scale_count = *desc.expectedLocalScaleCount();

    std::copy(kQuantizedMagic.begin(), kQuantizedMagic.end(), header.begin());
    store_unsigned_le<std::uint16_t>(header, kVersionOffset, kQuantizedFileVersion);
    store_unsigned_le<std::uint16_t>(
        header, kHeaderBytesOffset,
        static_cast<std::uint16_t>(kQuantizedHeaderBytes));

    header[kByteOrderOffset] = kLittleEndianByteOrder;
    header[kFormatOffset] = static_cast<std::uint8_t>(desc.format);
    header[kScaleModeOffset] = static_cast<std::uint8_t>(desc.scale_mode);
    header[kRoundingOffset] = static_cast<std::uint8_t>(desc.rounding);
    header[kSourceDTypeOffset] = static_cast<std::uint8_t>(desc.source_desc.dtype);
    header[kPayloadElementBitsOffset] = payload_element_bits(desc.format);
    header[kLocalScaleTypeOffset] =
        static_cast<std::uint8_t>(local_scale_type(desc.format));
    header[kScaleLayoutOffset] = static_cast<std::uint8_t>(desc.scale_layout);

    store_unsigned_le<std::uint32_t>(header, kBlockSizeOffset, desc.block_size);
    store_unsigned_le<std::uint64_t>(header, kRowsOffset, desc.source_desc.num_rows);
    store_unsigned_le<std::uint64_t>(header, kColsOffset, desc.source_desc.num_cols);
    store_unsigned_le<std::uint64_t>(header, kElementCountOffset, element_count);
    store_unsigned_le<std::uint64_t>(header, kPayloadOffsetOffset, payload_offset);
    store_unsigned_le<std::uint64_t>(header, kPayloadBytesOffset, payload_bytes);
    store_unsigned_le<std::uint64_t>(
        header, kLocalScaleOffsetOffset, local_scale_offset);
    store_unsigned_le<std::uint64_t>(
        header, kLocalScaleCountOffset, local_scale_count);
    store_unsigned_le<std::uint64_t>(
        header, kLocalScaleBytesOffset, local_scale_count);

    const bool is_nvfp4 = desc.format == QuantFormat::kNvfp4;
    store_float32_le(header, is_nvfp4 ? *tensor.global_scale : 1.0F);

    const bool has_odd_element_count = element_count % 2U != 0U;
    const std::uint32_t flags = is_nvfp4 && has_odd_element_count
                                    ? kNvfp4TailNibbleZeroFlag
                                    : 0U;
    store_unsigned_le<std::uint32_t>(header, kFlagsOffset, flags);
    store_unsigned_le<std::uint64_t>(
        header, kStochasticSeedOffset, desc.stochastic_seed);

    return header;
}

/**
 * @brief 解析并验证 QDWGT v1 固定 header 的数值和格式组合。
 *
 * section 的文件范围和相互关系依赖总文件长度，在本函数之后由
 * `validate_file_layout()` 完成校验。
 *
 * @param header 固定 128 字节 header。
 * @param file_path 当前文件路径，用于错误信息。
 * @return 已通过字段语义校验的 header 元数据。
 */
[[nodiscard]] ParsedQuantizedHeader parse_quantized_header(
    const std::array<std::uint8_t, kQuantizedHeaderBytes>& header,
    const std::filesystem::path& file_path) {
    const std::span<const std::uint8_t> bytes{header};

    if (!std::equal(kQuantizedMagic.begin(), kQuantizedMagic.end(), bytes.begin())) {
        throw_quantized_io_error(file_path, "magic 不匹配，不是 QDWGT 文件。");
    }

    if (load_unsigned_le<std::uint16_t>(bytes, kVersionOffset) !=
        kQuantizedFileVersion) {
        throw_quantized_io_error(file_path, "不支持的 QDWGT 文件版本。");
    }

    if (load_unsigned_le<std::uint16_t>(bytes, kHeaderBytesOffset) !=
        kQuantizedHeaderBytes) {
        throw_quantized_io_error(file_path, "header_bytes 必须为 128。");
    }

    if (bytes[kByteOrderOffset] != kLittleEndianByteOrder) {
        throw_quantized_io_error(file_path, "文件不是 little-endian 编码。");
    }

    QuantizedTensorDesc desc{
        .source_desc = {
            .num_rows = load_unsigned_le<std::uint64_t>(bytes, kRowsOffset),
            .num_cols = load_unsigned_le<std::uint64_t>(bytes, kColsOffset),
            .dtype = static_cast<DType>(bytes[kSourceDTypeOffset]),
        },
        .format = static_cast<QuantFormat>(bytes[kFormatOffset]),
        .scale_mode = static_cast<ScaleMode>(bytes[kScaleModeOffset]),
        .rounding = static_cast<RoundingMode>(bytes[kRoundingOffset]),
        .stochastic_seed = load_unsigned_le<std::uint64_t>(bytes, kStochasticSeedOffset),
        .block_size = load_unsigned_le<std::uint32_t>(bytes, kBlockSizeOffset),
        .scale_layout = static_cast<ScaleLayout>(bytes[kScaleLayoutOffset]),
    };

    if (!is_valid_scale_mode(desc.format, desc.scale_mode)) {
        throw_quantized_io_error(
            file_path,
            "NVFP4 QDWGT 仅允许 scale_mode = block；tensor mode 不是本项目支持的 "
            "NVFP4 文件语义。");
    }

    if (!desc.isMetadataValid()) {
        throw_quantized_io_error(file_path,
                                 "header 的格式、shape 或量化元数据组合非法。");
    }

    if (bytes[kPayloadElementBitsOffset] != payload_element_bits(desc.format)) {
        throw_quantized_io_error(file_path,
                                 "payload_element_bits 与量化格式不一致。");
    }

    if (bytes[kLocalScaleTypeOffset] !=
        static_cast<std::uint8_t>(local_scale_type(desc.format))) {
        throw_quantized_io_error(file_path,
                                 "local_scale_type 与量化格式不一致。");
    }

    const std::uint64_t stored_element_count =
        load_unsigned_le<std::uint64_t>(bytes, kElementCountOffset);
    const std::uint64_t stored_payload_bytes =
        load_unsigned_le<std::uint64_t>(bytes, kPayloadBytesOffset);
    const std::uint64_t stored_local_scale_count =
        load_unsigned_le<std::uint64_t>(bytes, kLocalScaleCountOffset);
    const std::uint64_t stored_local_scale_bytes =
        load_unsigned_le<std::uint64_t>(bytes, kLocalScaleBytesOffset);

    if (stored_element_count != *desc.elementCount()) {
        throw_quantized_io_error(file_path,
                                 "element_count 与 num_rows * num_cols 不一致。");
    }

    if (stored_payload_bytes != *desc.expectedPayloadBytes()) {
        throw_quantized_io_error(file_path,
                                 "payload_bytes 与 shape/format 推导结果不一致。");
    }

    if (stored_local_scale_count != *desc.expectedLocalScaleCount() ||
        stored_local_scale_bytes != stored_local_scale_count) {
        throw_quantized_io_error(file_path,
                                 "local scale 的数量或字节数与 scale_mode 不一致。");
    }

    const std::uint32_t flags = load_unsigned_le<std::uint32_t>(bytes, kFlagsOffset);
    const bool is_nvfp4 = desc.format == QuantFormat::kNvfp4;
    const bool has_odd_element_count = *desc.elementCount() % 2U != 0U;
    const std::uint32_t expected_flags = is_nvfp4 && has_odd_element_count
                                             ? kNvfp4TailNibbleZeroFlag
                                             : 0U;
    if (flags != expected_flags) {
        throw_quantized_io_error(file_path,
                                 "flags 与量化格式或尾 nibble 状态不一致。");
    }

    const float file_global_scale = load_float32_le(bytes);
    std::optional<float> global_scale{};
    if (is_nvfp4) {
        if (!std::isfinite(file_global_scale) || file_global_scale <= 0.0F) {
            throw_quantized_io_error(file_path,
                                     "NVFP4 的 global_scale 必须为有限正 FP32 数。");
        }
        global_scale = file_global_scale;
    } else if (std::bit_cast<std::uint32_t>(file_global_scale) != 0x3F800000U) {
        throw_quantized_io_error(file_path,
                                 "MXFP8 的 global_scale 必须为 1.0F。");
    }

    return ParsedQuantizedHeader{
        .desc = desc,
        .payload_offset = load_unsigned_le<std::uint64_t>(bytes, kPayloadOffsetOffset),
        .payload_bytes = stored_payload_bytes,
        .local_scale_offset =
            load_unsigned_le<std::uint64_t>(bytes, kLocalScaleOffsetOffset),
        .local_scale_bytes = stored_local_scale_bytes,
        .global_scale = global_scale,
    };
}

/**
 * @brief 验证 QDWGT v1 section 位置、对齐、顺序和完整文件长度。
 *
 * @param parsed_header 已通过字段语义校验的 header。
 * @param file_size 实际文件字节数。
 * @param file_path 当前文件路径，用于错误信息。
 */
void validate_file_layout(const ParsedQuantizedHeader& parsed_header,
                          const std::uint64_t file_size,
                          const std::filesystem::path& file_path) {
    if (parsed_header.payload_offset < kQuantizedHeaderBytes) {
        throw_quantized_io_error(file_path,
                                 "payload_offset 不能位于固定 header 内部。");
    }

    if (parsed_header.local_scale_offset % kSectionAlignment != 0U) {
        throw_quantized_io_error(file_path,
                                 "local_scale_offset 必须满足 8 字节对齐。");
    }

    validate_section_range(parsed_header.payload_offset,
                           parsed_header.payload_bytes,
                           file_size,
                           file_path,
                           "payload");
    validate_section_range(parsed_header.local_scale_offset,
                           parsed_header.local_scale_bytes,
                           file_size,
                           file_path,
                           "local scale");

    const std::uint64_t payload_end = checked_add(parsed_header.payload_offset,
                                                  parsed_header.payload_bytes,
                                                  file_path,
                                                  "payload section 终点");
    if (parsed_header.local_scale_offset < payload_end) {
        throw_quantized_io_error(file_path,
                                 "local scale section 与 payload section 重叠。");
    }

    const std::uint64_t local_scale_end =
        checked_add(parsed_header.local_scale_offset,
                    parsed_header.local_scale_bytes,
                    file_path,
                    "local scale section 终点");
    if (local_scale_end != file_size) {
        throw_quantized_io_error(
            file_path,
            "文件总长度必须恰好等于 local scale section 的终点。");
    }
}

/**
 * @brief 获取文件总长度，并检查它能用 QDWGT 的 uint64 offset 表示。
 *
 * @param file_path 待检查的文件路径。
 * @return 文件总字节数。
 * @throws QuantizedIoError 无法查询长度或长度过大时抛出。
 */
[[nodiscard]] std::uint64_t get_file_size(const std::filesystem::path& file_path) {
    std::error_code error_code{};
    const std::uintmax_t file_size = std::filesystem::file_size(file_path, error_code);
    if (error_code) {
        throw_quantized_io_error(file_path, "无法获取文件长度。");
    }

    if (file_size > std::numeric_limits<std::uint64_t>::max()) {
        throw_quantized_io_error(file_path, "文件长度超出 QDWGT v1 的 uint64 范围。");
    }

    return static_cast<std::uint64_t>(file_size);
}

/**
 * @brief 分配读取 section 所需的 byte vector，并检查 host 容量。
 *
 * @param byte_count 需要分配的字节数。
 * @param file_path 当前文件路径，用于错误信息。
 * @param field_name 对应 header 字段名。
 * @return 长度恰为 `byte_count` 的零初始化 byte vector。
 */
[[nodiscard]] std::vector<std::uint8_t> make_byte_vector(
    const std::uint64_t byte_count,
    const std::filesystem::path& file_path,
    const std::string_view field_name) {
    const std::size_t byte_count_size =
        checked_size_cast(byte_count, file_path, field_name);
    if (byte_count_size > std::vector<std::uint8_t>{}.max_size()) {
        throw_quantized_io_error(
            file_path,
            "字段 \"" + std::string{field_name} +
                "\" 超出 payload 容器容量。");
    }

    return std::vector<std::uint8_t>(byte_count_size);
}

}  // namespace

QuantizedIoError::QuantizedIoError(std::filesystem::path file_path,
                                   std::string detail)
    : std::runtime_error(make_error_message(file_path, detail)),
      mFilePath(std::move(file_path)) {}

const std::filesystem::path& QuantizedIoError::filePath() const noexcept {
    return mFilePath;
}

void write_quantized_tensor(const std::filesystem::path& output_path,
                            const QuantizedTensor& tensor) {
    if (!tensor.isConsistent()) {
        throw_quantized_io_error(output_path,
                                 "待写入的 QuantizedTensor 不满足格式不变量。");
    }

    const std::uint64_t payload_bytes = *tensor.desc.expectedPayloadBytes();
    const std::uint64_t payload_end = checked_add(kCanonicalPayloadOffset,
                                                  payload_bytes,
                                                  output_path,
                                                  "payload section 终点");
    const std::uint64_t local_scale_offset =
        align_up_to_section_boundary(payload_end, output_path);
    static_cast<void>(checked_add(local_scale_offset,
                                  *tensor.desc.expectedLocalScaleCount(),
                                  output_path,
                                  "local scale section 终点"));

    const auto header = make_quantized_header(tensor,
                                              kCanonicalPayloadOffset,
                                              local_scale_offset);
    const std::uint64_t padding_bytes = local_scale_offset - payload_end;
    constexpr std::array<std::uint8_t, kSectionAlignment> kZeroPadding{};

    std::ofstream output_stream{
        output_path,
        std::ios::binary | std::ios::trunc,
    };
    if (!output_stream.is_open()) {
        throw_quantized_io_error(output_path, "无法创建输出文件。");
    }

    write_exact(output_stream, header, output_path);
    write_exact(output_stream, tensor.payload, output_path);
    write_exact(output_stream,
                std::span<const std::uint8_t>{kZeroPadding}.first(
                    static_cast<std::size_t>(padding_bytes)),
                output_path);
    write_exact(output_stream, tensor.local_scales, output_path);

    output_stream.flush();
    if (!output_stream) {
        throw_quantized_io_error(output_path, "刷新输出文件失败。");
    }
}

QuantizedTensor read_quantized_tensor(const std::filesystem::path& input_path) {
    std::ifstream input_stream{input_path, std::ios::binary};
    if (!input_stream.is_open()) {
        throw_quantized_io_error(input_path, "无法打开输入文件。");
    }

    std::array<std::uint8_t, kQuantizedHeaderBytes> header{};
    read_exact(input_stream, header, input_path);

    const ParsedQuantizedHeader parsed_header =
        parse_quantized_header(header, input_path);
    validate_file_layout(parsed_header, get_file_size(input_path), input_path);

    std::vector<std::uint8_t> payload = make_byte_vector(
        parsed_header.payload_bytes, input_path, "payload_bytes");
    std::vector<std::uint8_t> local_scales = make_byte_vector(
        parsed_header.local_scale_bytes, input_path, "local_scale_bytes");

    seek_to(input_stream, parsed_header.payload_offset, input_path, "payload_offset");
    read_exact(input_stream, payload, input_path);
    seek_to(input_stream,
            parsed_header.local_scale_offset,
            input_path,
            "local_scale_offset");
    read_exact(input_stream, local_scales, input_path);

    QuantizedTensor tensor{
        .desc = parsed_header.desc,
        .payload = std::move(payload),
        .local_scales = std::move(local_scales),
        .global_scale = parsed_header.global_scale,
    };
    if (!tensor.isConsistent()) {
        throw_quantized_io_error(
            input_path,
            "payload、local scale 或 NVFP4 尾 nibble 不满足格式不变量。");
    }

    return tensor;
}

}  // namespace quant_dequant

#include "quant_dequant/tensor_io.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace quant_dequant {
namespace {
static_assert(sizeof(float) == sizeof(std::uint32_t));
static_assert(std::numeric_limits<float>::is_iec559);


/**
 * @brief QDTENSOR v1 的固定 header 字节数。
 */
constexpr std::size_t kTensorHeaderBytes = 64U;

/**
 * @brief QDTENSOR v1 的固定 data offset。
 */
constexpr std::uint64_t kTensorDataOffset = 64U;

/**
 * @brief QDTENSOR 文件 magic。
 */
constexpr std::array<std::uint8_t, 8U> kTensorMagic{
    'Q', 'D', 'T', 'E', 'N', 'S', 'O', 'R',
};

/**
 * @brief QDTENSOR 文件格式版本。
 */
constexpr std::uint16_t kTensorFileVersion = 1U;


/**
 * @brief QDTENSOR header 中各字段的固定 offset。
 */
constexpr std::size_t kMagicOffset = 0U;
constexpr std::size_t kVersionOffset = 8U;
constexpr std::size_t kHeaderBytesOffset = 10U;
constexpr std::size_t kByteOrderOffset = 12U;
constexpr std::size_t kDTypeOffset = 13U;
constexpr std::size_t kTensorRoleOffset = 14U;
constexpr std::size_t kRowsOffset = 16U;
constexpr std::size_t kColsOffset = 24U;
constexpr std::size_t kElementCountOffset = 32U;
constexpr std::size_t kDataOffsetOffset = 40U;
constexpr std::size_t kDataBytesOffset = 48U;

/**
 * @brief 保存已通过基本 header 校验的 QDTENSOR 元数据。
 */
struct ParsedTensorHeader {
    TensorDesc desc{};
    TensorRole role{TensorRole::kInput};
    std::uint64_t data_offset{0U};
    std::uint64_t data_bytes{0U};
};
/**
 * @brief 生成统一的 TensorIoError 文本。
 *
 * @param file_path 发生错误的文件路径。
 * @param detail 具体错误说明。
 * @return 可传递给 std::runtime_error 的完整错误文本。
 */
[[nodiscard]] std::string make_error_message(
    const std::filesystem::path& file_path,
    const std::string_view detail) {
    return "张量文件 \"" + file_path.string() + "\"：" + std::string{detail};
}

/**
 * @brief 抛出带文件路径的 TensorIoError。
 *
 * @param file_path 发生错误的文件路径。
 * @param detail 具体错误说明。
 */
[[noreturn]] void throw_tensor_io_error(
    const std::filesystem::path& file_path,
    std::string detail) {
    throw TensorIoError(file_path, std::move(detail));
}
/**
 * @brief 确认当前 host 使用 little-endian。
 *
 * 第一版文件格式只支持 little-endian host，避免在浮点 payload 上隐式
 * 执行 byte swap。
 *
 * @param file_path 当前操作的文件路径，用于错误信息。
 */
void ensure_little_endian_host(const std::filesystem::path& file_path) {
    if constexpr (std::endian::native != std::endian::little) {
        throw_tensor_io_error(file_path,
                              "当前 host 不是 little-endian，QDTENSOR v1 暂不支持。");
    }
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
 * @brief 使用 CUDA 官方接口将 FP16 bit pattern 转换为 FP32。
 *
 * @param half_bits little-endian 解码后的 FP16 bit pattern。
 * @return `__half2float` 产生的 FP32 数值。
 */
[[nodiscard]] float half_bits_to_float(const std::uint16_t half_bits) noexcept {
    const __half_raw raw_half{.x = half_bits};
    return __half2float(__half{raw_half});
}

/**
 * @brief 使用 CUDA 官方 RNE 接口将 FP32 转换为 FP16 bit pattern。
 *
 * @param value 要转换的 FP32 数值。
 * @return `__float2half_rn` 产生的 FP16 bit pattern。
 */
[[nodiscard]] std::uint16_t float_to_half_bits(const float value) noexcept {
    const __half half_value = __float2half_rn(value);
    return static_cast<__half_raw>(half_value).x;
}

/**
 * @brief 使用 CUDA 官方接口将 BF16 bit pattern 转换为 FP32。
 *
 * @param bfloat16_bits little-endian 解码后的 BF16 bit pattern。
 * @return `__bfloat162float` 产生的 FP32 数值。
 */
[[nodiscard]] float bfloat16_bits_to_float(
    const std::uint16_t bfloat16_bits) noexcept {
    const __nv_bfloat16_raw raw_bfloat16{.x = bfloat16_bits};
    return __bfloat162float(__nv_bfloat16{raw_bfloat16});
}

/**
 * @brief 使用 CUDA 官方 RNE 接口将 FP32 转换为 BF16 bit pattern。
 *
 * @param value 要转换的 FP32 数值。
 * @return `__float2bfloat16_rn` 产生的 BF16 bit pattern。
 */
[[nodiscard]] std::uint16_t float_to_bfloat16_bits(
    const float value) noexcept {
    const __nv_bfloat16 bfloat16_value = __float2bfloat16_rn(value);
    return static_cast<__nv_bfloat16_raw>(bfloat16_value).x;
}

/**
 * @brief 检查 uint64 数值能否安全转换为 host 容器使用的 size_t。
 *
 * @param value 待转换数值。
 * @param file_path 当前文件路径，用于错误信息。
 * @param field_name 对应 header 字段名。
 * @return 可安全用于 vector 大小和 stream 操作的 size_t。
 */
[[nodiscard]] std::size_t checked_size_cast(
    const std::uint64_t value,
    const std::filesystem::path& file_path,
    const std::string_view field_name) {
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw_tensor_io_error(
            file_path,
            "字段 \"" + std::string{field_name} +
                "\" 超出当前 host 的 size_t 表示范围。");
    }

    return static_cast<std::size_t>(value);
}

/**
 * @brief 从二进制流读取指定数量字节。
 *
 * @param input_stream 已打开的二进制输入流。
 * @param bytes 输出字节区。
 * @param file_path 当前文件路径，用于错误信息。
 * @throws TensorIoError 文件截断或 I/O 失败时抛出。
 */
void read_exact(std::ifstream& input_stream,
                const std::span<std::uint8_t> bytes,
                const std::filesystem::path& file_path) {
    if (bytes.size() >
        static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw_tensor_io_error(file_path, "读取长度超出 std::streamsize 范围。");
    }

    input_stream.read(reinterpret_cast<char*>(bytes.data()),
                      static_cast<std::streamsize>(bytes.size()));

    if (!input_stream) {
        throw_tensor_io_error(file_path, "文件截断或读取失败。");
    }
}

/**
 * @brief 向二进制流写入指定数量字节。
 *
 * @param output_stream 已打开的二进制输出流。
 * @param bytes 输入字节区。
 * @param file_path 当前文件路径，用于错误信息。
 * @throws TensorIoError 写入失败时抛出。
 */
void write_exact(std::ofstream& output_stream,
                 const std::span<const std::uint8_t> bytes,
                 const std::filesystem::path& file_path) {
    if (bytes.size() >
        static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw_tensor_io_error(file_path, "写入长度超出 std::streamsize 范围。");
    }

    output_stream.write(reinterpret_cast<const char*>(bytes.data()),
                        static_cast<std::streamsize>(bytes.size()));

    if (!output_stream) {
        throw_tensor_io_error(file_path, "写入文件失败。");
    }
}

/**
 * @brief 解析并验证 QDTENSOR v1 固定 header。
 *
 * @param header 固定 64 字节 header。
 * @param file_path 当前文件路径，用于错误信息。
 * @return 已通过 header 语义校验的元数据。
 */
[[nodiscard]] ParsedTensorHeader parse_tensor_header(
    const std::array<std::uint8_t, kTensorHeaderBytes>& header,
    const std::filesystem::path& file_path) {
    const std::span<const std::uint8_t> bytes{header};

    if (!std::equal(kTensorMagic.begin(), kTensorMagic.end(),
                    bytes.begin() + static_cast<std::ptrdiff_t>(kMagicOffset))) {
        throw_tensor_io_error(file_path, "magic 不匹配，不是 QDTENSOR 文件。");
    }

    if (load_unsigned_le<std::uint16_t>(bytes, kVersionOffset) !=
        kTensorFileVersion) {
        throw_tensor_io_error(file_path, "不支持的 QDTENSOR 文件版本。");
    }

    if (load_unsigned_le<std::uint16_t>(bytes, kHeaderBytesOffset) !=
        kTensorHeaderBytes) {
        throw_tensor_io_error(file_path, "header_bytes 必须为 64。");
    }

    if (bytes[kByteOrderOffset] != kLittleEndianByteOrder) {
        throw_tensor_io_error(file_path, "文件不是 little-endian 编码。");
    }

    const DType dtype = static_cast<DType>(bytes[kDTypeOffset]);
    if (dtype != DType::kFloat16 && dtype != DType::kBFloat16 &&
        dtype != DType::kFloat32) {
        throw_tensor_io_error(file_path, "header 包含不支持的 dtype。");
    }

    const TensorRole role = static_cast<TensorRole>(bytes[kTensorRoleOffset]);
    if (role != TensorRole::kInput &&
        role != TensorRole::kDequantizedOutput) {
        throw_tensor_io_error(file_path, "header 包含不支持的 tensor_role。");
    }

    TensorDesc desc{
        .num_rows = load_unsigned_le<std::uint64_t>(bytes, kRowsOffset),
        .num_cols = load_unsigned_le<std::uint64_t>(bytes, kColsOffset),
        .dtype = dtype,
    };

    const auto expected_element_count = desc.elementCount();
    const auto expected_data_bytes = desc.dataBytes();

    if (!expected_element_count.has_value() ||
        !expected_data_bytes.has_value()) {
        throw_tensor_io_error(file_path, "矩阵形状、dtype 或 payload 大小非法。");
    }

    const std::uint64_t stored_element_count =
        load_unsigned_le<std::uint64_t>(bytes, kElementCountOffset);
    const std::uint64_t stored_data_offset =
        load_unsigned_le<std::uint64_t>(bytes, kDataOffsetOffset);
    const std::uint64_t stored_data_bytes =
        load_unsigned_le<std::uint64_t>(bytes, kDataBytesOffset);

    if (stored_element_count != *expected_element_count) {
        throw_tensor_io_error(file_path,
                              "element_count 与 num_rows * num_cols 不一致。");
    }

    if (stored_data_offset != kTensorDataOffset) {
        throw_tensor_io_error(file_path,
                              "QDTENSOR v1 的 data_offset 必须为 64。");
    }

    if (stored_data_bytes != *expected_data_bytes) {
        throw_tensor_io_error(file_path,
                              "data_bytes 与 shape/dtype 推导结果不一致。");
    }

    return ParsedTensorHeader{
        .desc = desc,
        .role = role,
        .data_offset = stored_data_offset,
        .data_bytes = stored_data_bytes,
    };
}

/**
 * @brief 验证文件总长度恰好等于 QDTENSOR v1 header 与 payload 之和。
 *
 * @param file_path 待检查的文件路径。
 * @param data_bytes header 中已验证的 payload 字节数。
 */
void validate_file_size(const std::filesystem::path& file_path,
                        const std::uint64_t data_bytes) {
    if (data_bytes >
        std::numeric_limits<std::uint64_t>::max() - kTensorDataOffset) {
        throw_tensor_io_error(file_path, "data_bytes 导致文件长度溢出。");
    }

    std::error_code error_code{};
    const std::uintmax_t file_size =
        std::filesystem::file_size(file_path, error_code);

    if (error_code) {
        throw_tensor_io_error(file_path, "无法获取文件长度。");
    }

    const std::uint64_t expected_file_size =
        kTensorDataOffset + data_bytes;

    if (file_size != static_cast<std::uintmax_t>(expected_file_size)) {
        throw_tensor_io_error(
            file_path,
            "文件总长度与 header 声明的 data_bytes 不一致。");
    }
}

/**
 * @brief 将普通张量 payload 解码为连续 FP32 数组。
 *
 * @param desc 已验证的张量描述。
 * @param payload little-endian row-major payload。
 * @param file_path 当前文件路径，用于错误信息。
 * @return 解码后的 FP32 元素。
 */
[[nodiscard]] std::vector<float> decode_tensor_payload(
    const TensorDesc& desc,
    const std::span<const std::uint8_t> payload,
    const std::filesystem::path& file_path) {
    const auto element_count = desc.elementCount();
    if (!element_count.has_value()) {
        throw_tensor_io_error(file_path, "非法张量形状。");
    }

    const std::size_t element_count_size =
        checked_size_cast(*element_count, file_path, "element_count");

    if (element_count_size > std::vector<float>{}.max_size()) {
        throw_tensor_io_error(file_path, "element_count 超出 FP32 容器容量。");
    }

    std::vector<float> values(element_count_size);

    for (std::size_t index = 0U; index < element_count_size; ++index) {
        switch (desc.dtype) {
            case DType::kFloat16: {
                const std::uint16_t bits =
                    load_unsigned_le<std::uint16_t>(payload, index * 2U);
                values[index] = half_bits_to_float(bits);
                break;
            }

            case DType::kBFloat16: {
                const std::uint16_t bits =
                    load_unsigned_le<std::uint16_t>(payload, index * 2U);
                values[index] = bfloat16_bits_to_float(bits);
                break;
            }

            case DType::kFloat32: {
                const std::uint32_t bits =
                    load_unsigned_le<std::uint32_t>(payload, index * 4U);
                values[index] = std::bit_cast<float>(bits);
                break;
            }

            case DType::kUnknown:
                throw_tensor_io_error(file_path, "不支持的 dtype。");
        }
    }

    return values;
}

/**
 * @brief 将 host FP32 数组编码成目标普通张量 payload。
 *
 * @param output_desc 输出张量的形状和物理 dtype。
 * @param values 输入 FP32 数组，按 row-major 排列。
 * @param file_path 当前文件路径，用于错误信息。
 * @return little-endian 编码后的连续 payload。
 */
[[nodiscard]] std::vector<std::uint8_t> encode_tensor_payload(
    const TensorDesc& output_desc,
    const std::span<const float> values,
    const std::filesystem::path& file_path) {
    const auto element_count = output_desc.elementCount();
    const auto data_bytes = output_desc.dataBytes();

    if (!element_count.has_value() || !data_bytes.has_value()) {
        throw_tensor_io_error(file_path, "输出张量描述非法。");
    }

    const std::size_t element_count_size =
        checked_size_cast(*element_count, file_path, "element_count");
    const std::size_t data_bytes_size =
        checked_size_cast(*data_bytes, file_path, "data_bytes");

    if (values.size() != element_count_size) {
        throw_tensor_io_error(file_path,
                              "values 长度与输出张量元素数不一致。");
    }

    if (data_bytes_size > std::vector<std::uint8_t>{}.max_size()) {
        throw_tensor_io_error(file_path, "data_bytes 超出 payload 容器容量。");
    }

    std::vector<std::uint8_t> payload(data_bytes_size);

    for (std::size_t index = 0U; index < element_count_size; ++index) {
        switch (output_desc.dtype) {
            case DType::kFloat16:
                store_unsigned_le<std::uint16_t>(
                    payload, index * 2U, float_to_half_bits(values[index]));
                break;

            case DType::kBFloat16:
                store_unsigned_le<std::uint16_t>(
                    payload, index * 2U, float_to_bfloat16_bits(values[index]));
                break;

            case DType::kFloat32:
                store_unsigned_le<std::uint32_t>(
                    payload, index * 4U,
                    std::bit_cast<std::uint32_t>(values[index]));
                break;

            case DType::kUnknown:
                throw_tensor_io_error(file_path, "不支持的输出 dtype。");
        }
    }

    return payload;
}

/**
 * @brief 生成 QDTENSOR v1 的反量化输出 header。
 *
 * @param output_desc 输出形状与物理 dtype。
 * @param data_bytes 已验证的 payload 字节数。
 * @return 完整 64 字节 little-endian header。
 */
[[nodiscard]] std::array<std::uint8_t, kTensorHeaderBytes>
make_output_header(const TensorDesc& output_desc,
                   const std::uint64_t data_bytes) {
    std::array<std::uint8_t, kTensorHeaderBytes> header{};

    std::copy(kTensorMagic.begin(), kTensorMagic.end(),
              header.begin() + static_cast<std::ptrdiff_t>(kMagicOffset));

    store_unsigned_le<std::uint16_t>(
        header, kVersionOffset, kTensorFileVersion);
    store_unsigned_le<std::uint16_t>(
        header, kHeaderBytesOffset,
        static_cast<std::uint16_t>(kTensorHeaderBytes));

    header[kByteOrderOffset] = kLittleEndianByteOrder;
    header[kDTypeOffset] = static_cast<std::uint8_t>(output_desc.dtype);
    header[kTensorRoleOffset] =
        static_cast<std::uint8_t>(TensorRole::kDequantizedOutput);

    const std::uint64_t element_count = *output_desc.elementCount();

    store_unsigned_le<std::uint64_t>(
        header, kRowsOffset, output_desc.num_rows);
    store_unsigned_le<std::uint64_t>(
        header, kColsOffset, output_desc.num_cols);
    store_unsigned_le<std::uint64_t>(
        header, kElementCountOffset, element_count);
    store_unsigned_le<std::uint64_t>(
        header, kDataOffsetOffset, kTensorDataOffset);
    store_unsigned_le<std::uint64_t>(
        header, kDataBytesOffset, data_bytes);

    return header;
}

}  // namespace

TensorIoError::TensorIoError(std::filesystem::path file_path,
                             std::string detail)
    : std::runtime_error(make_error_message(file_path, detail)),
      mFilePath(std::move(file_path)) {}

const std::filesystem::path& TensorIoError::filePath() const noexcept {
    return mFilePath;
}

namespace {

/**
 * @brief 读取指定用途、指定 dtype 范围的 QDTENSOR，并将 payload 解码为 FP32。
 *
 * 输入文件与反量化输出文件的二进制 header/payload 规则完全相同，区别只在
 * `tensor_role` 与允许的 dtype。本内部函数集中完成一次严格读取，两个 public
 * 入口只各自声明它们允许的角色和 dtype，避免格式校验逻辑分叉。
 *
 * @param tensor_path 待读取的 QDTENSOR 文件。
 * @param expected_role 调用方所要求的 `TensorRole`。
 * @param require_input_dtype true 时仅接受 FP16/FP32；false 时接受 FP16/BF16/FP32。
 * @param function_name 用于说明哪个 public 入口拒绝了 role。
 * @return 保留物理 dtype、values 统一为 FP32 的 host 张量。
 * @throws TensorIoError 文件、角色、dtype、header 或 payload 不合法时抛出。
 */
[[nodiscard]] HostTensor read_tensor_with_role(
    const std::filesystem::path& tensor_path,
    const TensorRole expected_role,
    const bool require_input_dtype,
    const std::string_view function_name) {
    ensure_little_endian_host(tensor_path);

    std::ifstream input_stream{tensor_path, std::ios::binary};
    if (!input_stream.is_open()) {
        throw_tensor_io_error(tensor_path, "无法打开输入文件。");
    }

    std::array<std::uint8_t, kTensorHeaderBytes> header{};
    read_exact(input_stream, header, tensor_path);

    const ParsedTensorHeader parsed_header =
        parse_tensor_header(header, tensor_path);

    if (parsed_header.role != expected_role) {
        throw_tensor_io_error(
            tensor_path,
            std::string{function_name} + " 接收的 tensor_role 与文件不匹配。");
    }

    const bool valid_dtype = require_input_dtype
        ? is_supported_input_dtype(parsed_header.desc.dtype)
        : is_supported_output_dtype(parsed_header.desc.dtype);
    if (!valid_dtype) {
        throw_tensor_io_error(
            tensor_path,
            require_input_dtype
                ? "输入张量 dtype 只能是 FP16 或 FP32。"
                : "反量化输出 dtype 只能是 FP16、BF16 或 FP32。");
    }

    validate_file_size(tensor_path, parsed_header.data_bytes);

    const std::size_t payload_size =
        checked_size_cast(parsed_header.data_bytes, tensor_path, "data_bytes");

    if (payload_size > std::vector<std::uint8_t>{}.max_size()) {
        throw_tensor_io_error(tensor_path, "data_bytes 超出 payload 容器容量。");
    }

    std::vector<std::uint8_t> payload(payload_size);
    read_exact(input_stream, payload, tensor_path);

    return HostTensor{
        .desc = parsed_header.desc,
        .values = decode_tensor_payload(parsed_header.desc, payload, tensor_path),
    };
}

}  // namespace

HostTensor read_input_tensor(const std::filesystem::path& input_path) {
    return read_tensor_with_role(
        input_path, TensorRole::kInput, true, "read_input_tensor()");
}

HostTensor read_dequantized_tensor(const std::filesystem::path& output_path) {
    return read_tensor_with_role(
        output_path, TensorRole::kDequantizedOutput, false,
        "read_dequantized_tensor()");
}

void write_dequantized_tensor(
    const std::filesystem::path& output_path,
    const HostTensor& tensor) {
    ensure_little_endian_host(output_path);

    const TensorDesc& output_desc = tensor.desc;

    if (!output_desc.isValid()) {
        throw_tensor_io_error(output_path, "输出张量描述非法。");
    }

    if (!is_supported_output_dtype(output_desc.dtype)) {
        throw_tensor_io_error(
            output_path,
            "反量化输出 dtype 只能是 FP16、BF16 或 FP32。");
    }

    const auto data_bytes = output_desc.dataBytes();
    if (!data_bytes.has_value()) {
        throw_tensor_io_error(output_path, "无法计算输出 payload 字节数。");
    }

    const std::vector<std::uint8_t> payload =
        encode_tensor_payload(output_desc, tensor.values, output_path);
    const auto header = make_output_header(output_desc, *data_bytes);

    std::ofstream output_stream{
        output_path,
        std::ios::binary | std::ios::trunc,
    };

    if (!output_stream.is_open()) {
        throw_tensor_io_error(output_path, "无法创建输出文件。");
    }

    write_exact(output_stream, header, output_path);
    write_exact(output_stream, payload, output_path);

    output_stream.flush();
    if (!output_stream) {
        throw_tensor_io_error(output_path, "刷新输出文件失败。");
    }
}

}

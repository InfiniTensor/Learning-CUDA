#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include "quant_dequant/tensor_io.hpp"

namespace {

constexpr std::size_t kHeaderBytes = 64U;
constexpr std::uint64_t kDataOffset = 64U;

constexpr std::array<std::uint8_t, 8U> kTensorMagic{
    'Q', 'D', 'T', 'E', 'N', 'S', 'O', 'R',
};

constexpr std::size_t kVersionOffset = 8U;
constexpr std::size_t kHeaderBytesOffset = 10U;
constexpr std::size_t kDTypeOffset = 13U;
constexpr std::size_t kTensorRoleOffset = 14U;
constexpr std::size_t kRowsOffset = 16U;
constexpr std::size_t kColsOffset = 24U;
constexpr std::size_t kElementCountOffset = 32U;
constexpr std::size_t kDataOffsetOffset = 40U;
constexpr std::size_t kDataBytesOffset = 48U;

/**
 * @brief 在析构时删除测试专用临时目录。
 */
class TemporaryDirectoryCleanup final {
public:
    /**
     * @brief 记录需要清理的临时目录。
     *
     * @param directory_path 测试创建的目录路径。
     */
    explicit TemporaryDirectoryCleanup(std::filesystem::path directory_path)
        : mDirectoryPath(std::move(directory_path)) {}

    TemporaryDirectoryCleanup(const TemporaryDirectoryCleanup&) = delete;
    TemporaryDirectoryCleanup& operator=(const TemporaryDirectoryCleanup&) = delete;
    TemporaryDirectoryCleanup(TemporaryDirectoryCleanup&&) = delete;
    TemporaryDirectoryCleanup& operator=(TemporaryDirectoryCleanup&&) = delete;

    /**
     * @brief 尽力清理临时目录，不覆盖原始测试结果。
     */
    ~TemporaryDirectoryCleanup() {
        std::error_code error_code{};
        std::filesystem::remove_all(mDirectoryPath, error_code);
    }

private:
    std::filesystem::path mDirectoryPath;
};

/**
 * @brief 向 little-endian 字节区写入无符号整数。
 *
 * @tparam UInt 要写入的无符号整数类型。
 * @param bytes 目标字节区。
 * @param offset 起始字节 offset。
 * @param value 要写入的数值。
 */
template <typename UInt>
void store_unsigned_le(std::span<std::uint8_t> bytes, const std::size_t offset,
                       const UInt value) noexcept {
    static_assert(std::is_unsigned_v<UInt>);

    for (std::size_t index = 0U; index < sizeof(UInt); ++index) {
        bytes[offset + index] = static_cast<std::uint8_t>(
            value >> static_cast<unsigned int>(index * 8U));
    }
}

/**
 * @brief 从 little-endian 字节区读取无符号整数。
 *
 * @tparam UInt 目标无符号整数类型。
 * @param bytes 源字节区。
 * @param offset 起始字节 offset。
 * @return 解码后的整数。
 */
template <typename UInt>
[[nodiscard]] UInt load_unsigned_le(std::span<const std::uint8_t> bytes,
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
 * @brief 向 vector 末尾附加 little-endian 无符号整数。
 *
 * @tparam UInt 要附加的无符号整数类型。
 * @param bytes 要扩展的字节数组。
 * @param value 要编码的数值。
 */
template <typename UInt>
void append_unsigned_le(std::vector<std::uint8_t>& bytes, const UInt value) {
    static_assert(std::is_unsigned_v<UInt>);

    const std::size_t previous_size = bytes.size();
    bytes.resize(previous_size + sizeof(UInt));
    store_unsigned_le<UInt>(bytes, previous_size, value);
}

/**
 * @brief 构造独立于被测实现的 QDTENSOR v1 文件字节。
 *
 * 该帮助函数直接按 `docs/file_format.md` 编码 header，因此能够验证读取器，
 * 而不是用被测写入器生成再由被测读取器读取。
 *
 * @param dtype 文件中元素的物理 dtype。
 * @param role 文件用途。
 * @param rows 矩阵行数。
 * @param cols 矩阵列数。
 * @param payload 连续 row-major payload。
 * @return 完整 QDTENSOR 文件的字节数组。
 */
[[nodiscard]] std::vector<std::uint8_t> make_qdtensor_file(
    const quant_dequant::DType dtype, const quant_dequant::TensorRole role,
    const std::uint64_t rows, const std::uint64_t cols,
    const std::span<const std::uint8_t> payload) {
    std::vector<std::uint8_t> file_bytes(kHeaderBytes + payload.size(), 0U);
    std::copy(kTensorMagic.begin(), kTensorMagic.end(), file_bytes.begin());

    store_unsigned_le<std::uint16_t>(file_bytes, kVersionOffset, 1U);
    store_unsigned_le<std::uint16_t>(file_bytes, kHeaderBytesOffset,
                                     static_cast<std::uint16_t>(kHeaderBytes));
    file_bytes[12U] = quant_dequant::kLittleEndianByteOrder;
    file_bytes[kDTypeOffset] = static_cast<std::uint8_t>(dtype);
    file_bytes[kTensorRoleOffset] = static_cast<std::uint8_t>(role);

    const std::uint64_t element_count = rows * cols;
    store_unsigned_le<std::uint64_t>(file_bytes, kRowsOffset, rows);
    store_unsigned_le<std::uint64_t>(file_bytes, kColsOffset, cols);
    store_unsigned_le<std::uint64_t>(file_bytes, kElementCountOffset,
                                     element_count);
    store_unsigned_le<std::uint64_t>(file_bytes, kDataOffsetOffset, kDataOffset);
    store_unsigned_le<std::uint64_t>(file_bytes, kDataBytesOffset,
                                     static_cast<std::uint64_t>(payload.size()));

    std::copy(payload.begin(), payload.end(),
              file_bytes.begin() + static_cast<std::ptrdiff_t>(kHeaderBytes));
    return file_bytes;
}

/**
 * @brief 将完整二进制文件字节写入测试临时目录。
 *
 * @param file_path 目标文件路径。
 * @param bytes 待写入的完整文件内容。
 * @return 写入成功时返回 true。
 */
[[nodiscard]] bool write_binary_file(const std::filesystem::path& file_path,
                                     const std::span<const std::uint8_t> bytes) {
    std::ofstream output_stream{file_path, std::ios::binary | std::ios::trunc};
    if (!output_stream.is_open()) {
        return false;
    }

    output_stream.write(reinterpret_cast<const char*>(bytes.data()),
                        static_cast<std::streamsize>(bytes.size()));
    output_stream.flush();
    return output_stream.good();
}

/**
 * @brief 读取完整测试输出文件的原始字节。
 *
 * @param file_path 待读取的文件路径。
 * @param bytes 输出字节数组。
 * @return 完整读取成功时返回 true。
 */
[[nodiscard]] bool read_binary_file(const std::filesystem::path& file_path,
                                    std::vector<std::uint8_t>* bytes) {
    std::error_code error_code{};
    const std::uintmax_t file_size = std::filesystem::file_size(file_path, error_code);
    if (error_code || file_size > std::numeric_limits<std::size_t>::max()) {
        return false;
    }

    bytes->assign(static_cast<std::size_t>(file_size), 0U);
    std::ifstream input_stream{file_path, std::ios::binary};
    if (!input_stream.is_open()) {
        return false;
    }

    input_stream.read(reinterpret_cast<char*>(bytes->data()),
                      static_cast<std::streamsize>(bytes->size()));
    return input_stream.good() || input_stream.eof();
}

/**
 * @brief 验证输出 QDTENSOR header 的公共字段。
 *
 * @param file_bytes 读取到的完整文件字节。
 * @param dtype 期望的输出 dtype。
 * @param expected_data_bytes 期望 payload 字节数。
 * @return header 与 QDTENSOR v1 约定一致时返回 true。
 */
[[nodiscard]] bool has_valid_output_header(
    const std::span<const std::uint8_t> file_bytes,
    const quant_dequant::DType dtype,
    const std::uint64_t expected_data_bytes) {
    if (file_bytes.size() != kHeaderBytes + expected_data_bytes) {
        return false;
    }

    if (!std::equal(kTensorMagic.begin(), kTensorMagic.end(), file_bytes.begin())) {
        return false;
    }

    return load_unsigned_le<std::uint16_t>(file_bytes, kVersionOffset) == 1U &&
           load_unsigned_le<std::uint16_t>(file_bytes, kHeaderBytesOffset) ==
               kHeaderBytes &&
           file_bytes[12U] == quant_dequant::kLittleEndianByteOrder &&
           file_bytes[kDTypeOffset] == static_cast<std::uint8_t>(dtype) &&
           file_bytes[kTensorRoleOffset] ==
               static_cast<std::uint8_t>(quant_dequant::TensorRole::kDequantizedOutput) &&
           load_unsigned_le<std::uint64_t>(file_bytes, kRowsOffset) == 2U &&
           load_unsigned_le<std::uint64_t>(file_bytes, kColsOffset) == 3U &&
           load_unsigned_le<std::uint64_t>(file_bytes, kElementCountOffset) == 6U &&
           load_unsigned_le<std::uint64_t>(file_bytes, kDataOffsetOffset) ==
               kDataOffset &&
           load_unsigned_le<std::uint64_t>(file_bytes, kDataBytesOffset) ==
               expected_data_bytes;
}

/**
 * @brief 验证手工构造的 FP32 输入能够被正确读取。
 *
 * @param directory 测试临时目录。
 * @return 读取结果完全匹配时返回 true。
 */
[[nodiscard]] bool test_fp32_input_read(const std::filesystem::path& directory) {
    constexpr std::array<float, 6U> kExpectedValues{
        0.0F, 1.0F, -1.0F, 0.5F, -2.0F, 448.0F,
    };

    std::vector<std::uint8_t> payload{};
    payload.reserve(kExpectedValues.size() * sizeof(float));
    for (const float value : kExpectedValues) {
        append_unsigned_le<std::uint32_t>(payload,
                                          std::bit_cast<std::uint32_t>(value));
    }

    const std::filesystem::path input_path = directory / "input_fp32.bin";
    const auto file_bytes = make_qdtensor_file(
        quant_dequant::DType::kFloat32, quant_dequant::TensorRole::kInput,
        2U, 3U, payload);

    if (!write_binary_file(input_path, file_bytes)) {
        std::cerr << "无法写入 FP32 输入测试文件。\n";
        return false;
    }

    const quant_dequant::HostTensor tensor =
        quant_dequant::read_input_tensor(input_path);

    return tensor.desc.num_rows == 2U && tensor.desc.num_cols == 3U &&
           tensor.desc.dtype == quant_dequant::DType::kFloat32 &&
           tensor.values == std::vector<float>{kExpectedValues.begin(),
                                                kExpectedValues.end()};
}

/**
 * @brief 验证 FP16 normal、subnormal 与最大有限值能被正确扩展为 FP32。
 *
 * @param directory 测试临时目录。
 * @return 读取结果完全匹配时返回 true。
 */
[[nodiscard]] bool test_fp16_input_read(const std::filesystem::path& directory) {
    constexpr std::array<std::uint16_t, 6U> kHalfBits{
        0x0000U, 0x3C00U, 0xBC00U, 0x3800U, 0x0001U, 0x7BFFU,
    };
    const std::array<float, 6U> expected_values{
        0.0F, 1.0F, -1.0F, 0.5F, std::ldexp(1.0F, -24), 65504.0F,
    };

    std::vector<std::uint8_t> payload{};
    payload.reserve(kHalfBits.size() * sizeof(std::uint16_t));
    for (const std::uint16_t bits : kHalfBits) {
        append_unsigned_le<std::uint16_t>(payload, bits);
    }

    const std::filesystem::path input_path = directory / "input_fp16.bin";
    const auto file_bytes = make_qdtensor_file(
        quant_dequant::DType::kFloat16, quant_dequant::TensorRole::kInput,
        2U, 3U, payload);

    if (!write_binary_file(input_path, file_bytes)) {
        std::cerr << "无法写入 FP16 输入测试文件。\n";
        return false;
    }

    const quant_dequant::HostTensor tensor =
        quant_dequant::read_input_tensor(input_path);

    return tensor.desc.dtype == quant_dequant::DType::kFloat16 &&
           tensor.values == std::vector<float>{expected_values.begin(),
                                                expected_values.end()};
}

/**
 * @brief 验证反量化输出的 header、payload 编码及输出用途文件读取。
 *
 * @param directory 测试临时目录。
 * @param dtype 目标输出 dtype。
 * @param expected_payload 期望的 little-endian payload。
 * @return 写入字节、输出 role/dtype 和重新扩展为 FP32 的数值均匹配时返回 true。
 */
[[nodiscard]] bool test_dequantized_output_write(
    const std::filesystem::path& directory, const quant_dequant::DType dtype,
    const std::span<const std::uint8_t> expected_payload) {
    constexpr std::array<float, 6U> kValues{
        0.0F, 1.5F, -2.25F, 0.5F, 3.5F, 0.0078125F,
    };

    const quant_dequant::TensorDesc output_desc{
        .num_rows = 2U,
        .num_cols = 3U,
        .dtype = dtype,
    };
    const std::filesystem::path output_path =
        directory / ("output_" + std::string{quant_dequant::to_string(dtype)} + ".bin");

    const quant_dequant::HostTensor output_tensor{
        .desc = output_desc,
        .values = {kValues.begin(), kValues.end()},
    };
    quant_dequant::write_dequantized_tensor(output_path, output_tensor);

    std::vector<std::uint8_t> file_bytes{};
    if (!read_binary_file(output_path, &file_bytes)) {
        std::cerr << "无法读取反量化输出测试文件。\n";
        return false;
    }

    if (!has_valid_output_header(file_bytes, dtype,
                                 static_cast<std::uint64_t>(expected_payload.size()))) {
        return false;
    }

    if (!std::equal(expected_payload.begin(), expected_payload.end(),
                    file_bytes.begin() + static_cast<std::ptrdiff_t>(kHeaderBytes))) {
        return false;
    }

    // 完整 app 以此接口读回实际写出的物理 payload，确保 FP16/BF16 的窄化误差
    // 也计入最终指标。当前样例值在三种输出 dtype 中都可精确表示。
    const quant_dequant::HostTensor reread_tensor =
        quant_dequant::read_dequantized_tensor(output_path);
    return reread_tensor.desc.num_rows == output_desc.num_rows &&
           reread_tensor.desc.num_cols == output_desc.num_cols &&
           reread_tensor.desc.dtype == output_desc.dtype &&
           reread_tensor.values == std::vector<float>{kValues.begin(), kValues.end()};
}

/**
 * @brief 验证读取器能拒绝损坏 magic 与截断 payload。
 *
 * @param directory 测试临时目录。
 * @return 两种损坏文件均被 TensorIoError 拒绝时返回 true。
 */
[[nodiscard]] bool test_invalid_input_rejection(
    const std::filesystem::path& directory) {
    constexpr std::array<float, 6U> kValues{
        0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F,
    };
    std::vector<std::uint8_t> payload{};
    payload.reserve(kValues.size() * sizeof(float));
    for (const float value : kValues) {
        append_unsigned_le<std::uint32_t>(payload,
                                          std::bit_cast<std::uint32_t>(value));
    }

    auto corrupt_magic = make_qdtensor_file(
        quant_dequant::DType::kFloat32, quant_dequant::TensorRole::kInput,
        2U, 3U, payload);
    corrupt_magic[0U] = 0U;

    const std::filesystem::path magic_path = directory / "invalid_magic.bin";
    if (!write_binary_file(magic_path, corrupt_magic)) {
        return false;
    }

    try {
        static_cast<void>(quant_dequant::read_input_tensor(magic_path));
        return false;
    } catch (const quant_dequant::TensorIoError&) {
    }

    const std::filesystem::path truncated_path = directory / "truncated.bin";
    const auto valid_file = make_qdtensor_file(
        quant_dequant::DType::kFloat32, quant_dequant::TensorRole::kInput,
        2U, 3U, payload);
    if (!write_binary_file(truncated_path, valid_file)) {
        return false;
    }

    std::error_code error_code{};
    std::filesystem::resize_file(truncated_path, kHeaderBytes + 1U, error_code);
    if (error_code) {
        return false;
    }

    try {
        static_cast<void>(quant_dequant::read_input_tensor(truncated_path));
        return false;
    } catch (const quant_dequant::TensorIoError&) {
        return true;
    }
}

/**
 * @brief 验证反量化输出文件不会被误当作输入张量读取。
 *
 * @param directory 测试临时目录。
 * @return role 不匹配被正确拒绝时返回 true。
 */
[[nodiscard]] bool test_output_role_rejection(
    const std::filesystem::path& directory) {
    constexpr std::array<float, 6U> kValues{
        0.0F, 1.0F, -1.0F, 0.5F, -0.5F, 2.0F,
    };
    const quant_dequant::TensorDesc output_desc{
        .num_rows = 2U,
        .num_cols = 3U,
        .dtype = quant_dequant::DType::kFloat32,
    };
    const std::filesystem::path output_path = directory / "dequantized.bin";

    const quant_dequant::HostTensor output_tensor{
        .desc = output_desc,
        .values = {kValues.begin(), kValues.end()},
    };
    quant_dequant::write_dequantized_tensor(output_path, output_tensor);

    try {
        static_cast<void>(quant_dequant::read_input_tensor(output_path));
        return false;
    } catch (const quant_dequant::TensorIoError&) {
        return true;
    }
}

}  // namespace

/**
 * @brief 运行 QDTENSOR 输入读取、输出写入与损坏文件拒绝测试。
 *
 * @return 所有 tensor I/O 测试通过时返回 0；失败时打印原因并返回 1。
 */
int run_tensor_io_tests() {
    const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() /
        ("quant_dequant_tensor_io_test_" + std::to_string(timestamp));

    std::error_code error_code{};
    if (!std::filesystem::create_directory(directory, error_code) || error_code) {
        std::cerr << "无法创建 tensor I/O 测试临时目录。\n";
        return 1;
    }
    const TemporaryDirectoryCleanup cleanup{directory};

    try {
        if (!test_fp32_input_read(directory)) {
            std::cerr << "FP32 输入读取测试失败。\n";
            return 1;
        }

        if (!test_fp16_input_read(directory)) {
            std::cerr << "FP16 输入读取测试失败。\n";
            return 1;
        }

        constexpr std::array<std::uint8_t, 12U> kFp16Payload{
            0x00U, 0x00U, 0x00U, 0x3EU, 0x80U, 0xC0U,
            0x00U, 0x38U, 0x00U, 0x43U, 0x00U, 0x20U,
        };
        if (!test_dequantized_output_write(directory,
                                           quant_dequant::DType::kFloat16,
                                           kFp16Payload)) {
            std::cerr << "FP16 输出写入测试失败。\n";
            return 1;
        }

        constexpr std::array<std::uint8_t, 12U> kBf16Payload{
            0x00U, 0x00U, 0xC0U, 0x3FU, 0x10U, 0xC0U,
            0x00U, 0x3FU, 0x60U, 0x40U, 0x00U, 0x3CU,
        };
        if (!test_dequantized_output_write(directory,
                                           quant_dequant::DType::kBFloat16,
                                           kBf16Payload)) {
            std::cerr << "BF16 输出写入测试失败。\n";
            return 1;
        }

        constexpr std::array<float, 6U> kFp32Values{
            0.0F, 1.5F, -2.25F, 0.5F, 3.5F, 0.0078125F,
        };
        std::vector<std::uint8_t> fp32_payload{};
        fp32_payload.reserve(kFp32Values.size() * sizeof(float));
        for (const float value : kFp32Values) {
            append_unsigned_le<std::uint32_t>(fp32_payload,
                                              std::bit_cast<std::uint32_t>(value));
        }
        if (!test_dequantized_output_write(directory,
                                           quant_dequant::DType::kFloat32,
                                           fp32_payload)) {
            std::cerr << "FP32 输出写入测试失败。\n";
            return 1;
        }

        if (!test_invalid_input_rejection(directory)) {
            std::cerr << "损坏输入文件拒绝测试失败。\n";
            return 1;
        }

        if (!test_output_role_rejection(directory)) {
            std::cerr << "输出 role 拒绝测试失败。\n";
            return 1;
        }
    } catch (const quant_dequant::TensorIoError& error) {
        std::cerr << "Tensor I/O 测试意外抛出异常：" << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "Tensor I/O 测试发生未预期异常：" << error.what() << '\n';
        return 1;
    }

    return 0;
}

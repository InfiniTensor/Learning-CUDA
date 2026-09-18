#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include "quant_dequant/quantize.hpp"
#include "quant_dequant/quantized_io.hpp"
#include "quant_dequant/tensor_io.hpp"

namespace {

constexpr std::size_t kQdTensorHeaderBytes = 64U;
constexpr std::size_t kQdTensorVersionOffset = 8U;
constexpr std::size_t kQdTensorHeaderBytesOffset = 10U;
constexpr std::size_t kQdTensorByteOrderOffset = 12U;
constexpr std::size_t kQdTensorDTypeOffset = 13U;
constexpr std::size_t kQdTensorRoleOffset = 14U;
constexpr std::size_t kQdTensorRowsOffset = 16U;
constexpr std::size_t kQdTensorColsOffset = 24U;
constexpr std::size_t kQdTensorElementCountOffset = 32U;
constexpr std::size_t kQdTensorDataOffsetOffset = 40U;
constexpr std::size_t kQdTensorDataBytesOffset = 48U;

constexpr std::array<std::uint8_t, 8U> kQdTensorMagic{
    'Q', 'D', 'T', 'E', 'N', 'S', 'O', 'R',
};

/**
 * @brief 在析构时删除 CPU reference 端到端测试创建的临时目录。
 */
class TemporaryDirectoryCleanup final {
public:
    /**
     * @brief 保存要在析构时清理的目录。
     *
     * @param directory_path 本测试独占创建的临时目录。
     */
    explicit TemporaryDirectoryCleanup(std::filesystem::path directory_path)
        : mDirectoryPath(std::move(directory_path)) {}

    TemporaryDirectoryCleanup(const TemporaryDirectoryCleanup&) = delete;
    TemporaryDirectoryCleanup& operator=(const TemporaryDirectoryCleanup&) = delete;
    TemporaryDirectoryCleanup(TemporaryDirectoryCleanup&&) = delete;
    TemporaryDirectoryCleanup& operator=(TemporaryDirectoryCleanup&&) = delete;

    /**
     * @brief 尽力删除临时目录，不让清理错误掩盖测试结果。
     */
    ~TemporaryDirectoryCleanup() {
        std::error_code error_code{};
        std::filesystem::remove_all(mDirectoryPath, error_code);
    }

private:
    std::filesystem::path mDirectoryPath;
};

/**
 * @brief 向字节数组指定位置写入一个 little-endian 无符号整数。
 *
 * @tparam UInt 待写入的无符号整数类型。
 * @param bytes 目标文件字节数组。
 * @param offset 整数在 bytes 中的起始下标。
 * @param value 待写入的数值。
 */
template <typename UInt>
void store_unsigned_le(std::span<std::uint8_t> bytes,
                       const std::size_t offset,
                       const UInt value) noexcept {
    static_assert(std::is_unsigned_v<UInt>);

    for (std::size_t byte_index = 0U; byte_index < sizeof(UInt); ++byte_index) {
        bytes[offset + byte_index] = static_cast<std::uint8_t>(
            value >> static_cast<unsigned int>(byte_index * 8U));
    }
}

/**
 * @brief 从字节数组的指定位置读取一个 little-endian 无符号整数。
 *
 * @tparam UInt 要读取的无符号整数类型。
 * @param bytes 包含完整 QDTENSOR 文件的字节数组。
 * @param offset 整数在 bytes 中的起始下标。
 * @return 由 little-endian 字节序还原的整数。
 */
template <typename UInt>
[[nodiscard]] UInt load_unsigned_le(const std::span<const std::uint8_t> bytes,
                                    const std::size_t offset) noexcept {
    static_assert(std::is_unsigned_v<UInt>);

    UInt result{0U};
    for (std::size_t byte_index = 0U; byte_index < sizeof(UInt); ++byte_index) {
        result |= static_cast<UInt>(bytes[offset + byte_index])
                  << static_cast<unsigned int>(byte_index * 8U);
    }

    return result;
}

/**
 * @brief 构造独立于 tensor_io 写入器的 FP32 输入 QDTENSOR v1 文件。
 *
 * 项目目前只公开了“反量化输出 QDTENSOR”的写接口，而输入读取器必须校验
 * tensor_role=input。因此端到端测试在这里按 file_format.md 手工生成输入
 * 文件，确保 `read_input_tensor()` 的实际解析路径也被覆盖。
 *
 * @param rows 输入矩阵行数。
 * @param cols 输入矩阵列数。
 * @param values 连续 row-major FP32 输入数据。
 * @return 完整的 little-endian QDTENSOR 文件字节。
 */
[[nodiscard]] std::vector<std::uint8_t> make_fp32_input_file(
    const std::uint64_t rows,
    const std::uint64_t cols,
    const std::span<const float> values) {
    const std::size_t payload_bytes = values.size() * sizeof(float);
    std::vector<std::uint8_t> file_bytes(kQdTensorHeaderBytes + payload_bytes,
                                         0U);

    std::copy(kQdTensorMagic.begin(), kQdTensorMagic.end(), file_bytes.begin());
    store_unsigned_le<std::uint16_t>(file_bytes, 8U, 1U);
    store_unsigned_le<std::uint16_t>(
        file_bytes, 10U, static_cast<std::uint16_t>(kQdTensorHeaderBytes));
    file_bytes[12U] = quant_dequant::kLittleEndianByteOrder;
    file_bytes[13U] = static_cast<std::uint8_t>(quant_dequant::DType::kFloat32);
    file_bytes[14U] = static_cast<std::uint8_t>(quant_dequant::TensorRole::kInput);
    store_unsigned_le<std::uint64_t>(file_bytes, 16U, rows);
    store_unsigned_le<std::uint64_t>(file_bytes, 24U, cols);
    store_unsigned_le<std::uint64_t>(file_bytes, 32U, rows * cols);
    store_unsigned_le<std::uint64_t>(
        file_bytes, 40U, static_cast<std::uint64_t>(kQdTensorHeaderBytes));
    store_unsigned_le<std::uint64_t>(
        file_bytes, 48U, static_cast<std::uint64_t>(payload_bytes));

    for (std::size_t index = 0U; index < values.size(); ++index) {
        store_unsigned_le<std::uint32_t>(
            file_bytes,
            kQdTensorHeaderBytes + index * sizeof(float),
            std::bit_cast<std::uint32_t>(values[index]));
    }

    return file_bytes;
}

/**
 * @brief 将完整二进制文件写到测试目录。
 *
 * @param file_path 目标文件路径。
 * @param bytes 待写入的完整文件内容。
 * @return 所有字节都成功写入时返回 true。
 */
[[nodiscard]] bool write_binary_file(
    const std::filesystem::path& file_path,
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
 * @brief 将完整二进制文件读入测试用字节数组。
 *
 * 此辅助函数不调用 QDTENSOR 读取器，因为输出文件带有
 * `tensor_role = dequantized_output`，公共输入读取接口必须拒绝该 role。
 * 测试需要直接检查其 header 和物理 payload。
 *
 * @param file_path 待读取的文件路径。
 * @param bytes 输出参数，成功后保存文件全部字节。
 * @return 文件大小可表示且所有字节读取成功时返回 true。
 */
[[nodiscard]] bool read_binary_file(
    const std::filesystem::path& file_path,
    std::vector<std::uint8_t>* const bytes) {
    std::error_code error_code{};
    const std::uintmax_t file_size =
        std::filesystem::file_size(file_path, error_code);
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
 * @brief 构造本 MXFP8 测试向量反量化后应写出的物理 payload。
 *
 * 测试矩阵除 `+1`、`-1`、`+448`、`-896` 外均为零，且这四个非零值恰好都能由
 * FP16、BF16 和 FP32 精确表示。因此这里以手工固定的 IEEE bit pattern 校验
 * 写入器，避免测试再次调用与被测代码相同的转换接口。
 *
 * @param dtype 要验证的反量化输出物理类型。
 * @param values 已验证等于本测试源数据的 FP32 反量化结果。
 * @return 对应 dtype 的 little-endian row-major payload；未知类型返回空数组。
 */
[[nodiscard]] std::vector<std::uint8_t> make_expected_output_payload(
    const quant_dequant::DType dtype,
    const std::span<const float> values) {
    switch (dtype) {
        case quant_dequant::DType::kFloat16: {
            std::vector<std::uint8_t> payload(values.size() * 2U, 0U);
            store_unsigned_le<std::uint16_t>(payload, 1U * 2U, 0x3c00U);
            store_unsigned_le<std::uint16_t>(payload, 2U * 2U, 0xbc00U);
            store_unsigned_le<std::uint16_t>(payload, 3U * 2U, 0x5f00U);
            store_unsigned_le<std::uint16_t>(payload, 32U * 2U, 0xe300U);
            return payload;
        }

        case quant_dequant::DType::kBFloat16: {
            std::vector<std::uint8_t> payload(values.size() * 2U, 0U);
            store_unsigned_le<std::uint16_t>(payload, 1U * 2U, 0x3f80U);
            store_unsigned_le<std::uint16_t>(payload, 2U * 2U, 0xbf80U);
            store_unsigned_le<std::uint16_t>(payload, 3U * 2U, 0x43e0U);
            store_unsigned_le<std::uint16_t>(payload, 32U * 2U, 0xc460U);
            return payload;
        }

        case quant_dequant::DType::kFloat32: {
            std::vector<std::uint8_t> payload(values.size() * 4U, 0U);
            for (std::size_t index = 0U; index < values.size(); ++index) {
                store_unsigned_le<std::uint32_t>(
                    payload,
                    index * 4U,
                    std::bit_cast<std::uint32_t>(values[index]));
            }
            return payload;
        }

        case quant_dequant::DType::kUnknown:
            return {};
    }

    return {};
}

/**
 * @brief 检查 CPU 反量化结果写出的完整 QDTENSOR 文件。
 *
 * @param file_bytes 实际读回的完整文件字节。
 * @param dtype 期望的输出 dtype。
 * @param rows 期望矩阵行数。
 * @param cols 期望矩阵列数。
 * @param expected_payload 期望的 little-endian row-major payload。
 * @return QDTENSOR header、文件长度和 payload 全部匹配时返回 true。
 */
[[nodiscard]] bool is_expected_dequantized_output_file(
    const std::span<const std::uint8_t> file_bytes,
    const quant_dequant::DType dtype,
    const std::uint64_t rows,
    const std::uint64_t cols,
    const std::span<const std::uint8_t> expected_payload) {
    if (file_bytes.size() != kQdTensorHeaderBytes + expected_payload.size()) {
        return false;
    }

    const bool valid_header =
        std::equal(kQdTensorMagic.begin(), kQdTensorMagic.end(),
                   file_bytes.begin()) &&
        load_unsigned_le<std::uint16_t>(file_bytes, kQdTensorVersionOffset) ==
            1U &&
        load_unsigned_le<std::uint16_t>(file_bytes, kQdTensorHeaderBytesOffset) ==
            kQdTensorHeaderBytes &&
        file_bytes[kQdTensorByteOrderOffset] ==
            quant_dequant::kLittleEndianByteOrder &&
        file_bytes[kQdTensorDTypeOffset] == static_cast<std::uint8_t>(dtype) &&
        file_bytes[kQdTensorRoleOffset] ==
            static_cast<std::uint8_t>(
                quant_dequant::TensorRole::kDequantizedOutput) &&
        load_unsigned_le<std::uint64_t>(file_bytes, kQdTensorRowsOffset) == rows &&
        load_unsigned_le<std::uint64_t>(file_bytes, kQdTensorColsOffset) == cols &&
        load_unsigned_le<std::uint64_t>(file_bytes, kQdTensorElementCountOffset) ==
            rows * cols &&
        load_unsigned_le<std::uint64_t>(file_bytes, kQdTensorDataOffsetOffset) ==
            kQdTensorHeaderBytes &&
        load_unsigned_le<std::uint64_t>(file_bytes, kQdTensorDataBytesOffset) ==
            expected_payload.size();

    return valid_header && std::equal(
                               expected_payload.begin(), expected_payload.end(),
                               file_bytes.begin() + static_cast<std::ptrdiff_t>(
                                                        kQdTensorHeaderBytes));
}

/**
 * @brief 比较端到端测试关心的 MXFP8 量化结果字段。
 *
 * @param left 量化器刚生成的结果。
 * @param right 经 QDWGT 写入并读取后的结果。
 * @return 元数据、payload、local scale 与 global scale 语义均一致时返回 true。
 */
[[nodiscard]] bool same_quantized_tensor(
    const quant_dequant::QuantizedTensor& left,
    const quant_dequant::QuantizedTensor& right) {
    const auto& left_desc = left.desc;
    const auto& right_desc = right.desc;

    return left_desc.source_desc.num_rows == right_desc.source_desc.num_rows &&
           left_desc.source_desc.num_cols == right_desc.source_desc.num_cols &&
           left_desc.source_desc.dtype == right_desc.source_desc.dtype &&
           left_desc.format == right_desc.format &&
           left_desc.scale_mode == right_desc.scale_mode &&
           left_desc.rounding == right_desc.rounding &&
           left_desc.stochastic_seed == right_desc.stochastic_seed &&
           left_desc.block_size == right_desc.block_size &&
           left_desc.scale_layout == right_desc.scale_layout &&
           left.payload == right.payload &&
           left.local_scales == right.local_scales &&
           !left.global_scale.has_value() && !right.global_scale.has_value();
}

/**
 * @brief 验证 FP32 输入到 MXFP8 QDWGT 再到三种 QDTENSOR 输出的 CPU 链路。
 *
 * 输入形状刻意取 1 x 33：前 32 个元素形成一个完整 block，最后一个元素形成
 * 尾 block。前一 block 的 amax 是 448，故 E8M0 scale 为 1（0x7f）；尾
 * block 的唯一元素为 -896，故 scale 为 2（0x80）。这同时锁定 rowwise
 * block 分组、scale 计算、E4M3 编码、尾 block 和 QDWGT I/O。读回 QDWGT 后
 * 还会经 public 反量化入口恢复为 FP32，并立即写出 FP16、BF16、FP32 三种
 * QDTENSOR 文件。这里的测试值全都恰好可表示，因此三种文件的 payload 都可
 * 以固定 bit pattern 验证。
 *
 * @param directory 测试专用临时目录。
 * @return 完整文件链路及全部位级和数值期望一致时返回 true。
 */
[[nodiscard]] bool test_mxfp8_cpu_round_trip_with_io(
    const std::filesystem::path& directory) {
    constexpr std::uint64_t kRows = 1U;
    constexpr std::uint64_t kCols = 33U;

    std::vector<float> source_values(static_cast<std::size_t>(kRows * kCols),
                                     0.0F);
    source_values[0U] = 0.0F;
    source_values[1U] = 1.0F;
    source_values[2U] = -1.0F;
    source_values[3U] = 448.0F;
    source_values[32U] = -896.0F;

    const std::filesystem::path input_path = directory / "input.qdtensor";
    const std::vector<std::uint8_t> input_file = make_fp32_input_file(
        kRows, kCols, source_values);
    if (!write_binary_file(input_path, input_file)) {
        std::cerr << "无法写入 CPU MXFP8 端到端测试输入文件。\n";
        return false;
    }

    const quant_dequant::HostTensor input =
        quant_dequant::read_input_tensor(input_path);
    if (input.desc.num_rows != kRows || input.desc.num_cols != kCols ||
        input.desc.dtype != quant_dequant::DType::kFloat32 ||
        input.values != source_values) {
        std::cerr << "QDTENSOR 输入读取结果与原始 FP32 数据不一致。\n";
        return false;
    }

    const quant_dequant::QuantizationConfig config{
        .format = quant_dequant::QuantFormat::kMxfp8,
        .block_size = quant_dequant::kMxfp8BlockSize,
        .scale_mode = quant_dequant::ScaleMode::kBlock,
        .rounding = quant_dequant::RoundingMode::kNearest,
        .stochastic_seed = 0U,
    };
    const quant_dequant::QuantizedTensor quantized =
        quant_dequant::quantize_reference(input, config);

    std::vector<std::uint8_t> expected_payload(33U, 0x00U);
    expected_payload[1U] = 0x38U;
    expected_payload[2U] = 0xb8U;
    expected_payload[3U] = 0x7eU;
    expected_payload[32U] = 0xfeU;

    if (!quantized.isConsistent() ||
        quantized.desc.source_desc.dtype != quant_dequant::DType::kFloat32 ||
        quantized.desc.format != quant_dequant::QuantFormat::kMxfp8 ||
        quantized.desc.scale_mode != quant_dequant::ScaleMode::kBlock ||
        quantized.local_scales != std::vector<std::uint8_t>{0x7fU, 0x80U} ||
        quantized.payload != expected_payload || quantized.global_scale.has_value()) {
        std::cerr << "MXFP8 CPU 量化的 scale、payload 或元数据与预期不一致。\n";
        return false;
    }

    const std::filesystem::path quantized_path = directory / "output.qdwgt";
    quant_dequant::write_quantized_tensor(quantized_path, quantized);

    std::error_code error_code{};
    const std::uintmax_t output_file_size =
        std::filesystem::file_size(quantized_path, error_code);
    if (error_code || output_file_size != 170U) {
        std::cerr << "MXFP8 QDWGT 文件大小或 section 对齐不符合预期。\n";
        return false;
    }

    const quant_dequant::QuantizedTensor loaded =
        quant_dequant::read_quantized_tensor(quantized_path);
    if (!same_quantized_tensor(quantized, loaded)) {
        std::cerr << "MXFP8 QDWGT 写入并读取后量化结果不一致。\n";
        return false;
    }

    // CPU reference 始终以 FP32 保存计算结果；三个 output_type 的差异在随后
    // write_dequantized_tensor() 写 QDTENSOR 时才落实为不同物理 payload。这里
    // 不仅检查 desc.dtype，还会读回文件并比对固定 header 和实际 payload。
    constexpr std::array<quant_dequant::DType, 3U> kOutputTypes{
        quant_dequant::DType::kFloat16,
        quant_dequant::DType::kBFloat16,
        quant_dequant::DType::kFloat32,
    };
    for (const quant_dequant::DType output_type : kOutputTypes) {
        const quant_dequant::DequantizationConfig dequantization_config{
            .output_type = output_type,
        };
        const quant_dequant::HostTensor dequantized =
            quant_dequant::dequantize_reference(loaded, dequantization_config);
        if (dequantized.desc.num_rows != kRows ||
            dequantized.desc.num_cols != kCols ||
            dequantized.desc.dtype != output_type ||
            dequantized.values != source_values) {
            std::cerr << "MXFP8 CPU 反量化结果或输出类型与预期不一致。\n";
            return false;
        }

        const std::filesystem::path dequantized_path =
            directory /
            ("dequantized_" + std::string{quant_dequant::to_string(output_type)} +
             ".qdtensor");
        quant_dequant::write_dequantized_tensor(dequantized_path, dequantized);

        const std::vector<std::uint8_t> expected_output_payload =
            make_expected_output_payload(output_type, dequantized.values);
        std::vector<std::uint8_t> output_file_bytes{};
        if (!read_binary_file(dequantized_path, &output_file_bytes) ||
            !is_expected_dequantized_output_file(
                output_file_bytes,
                output_type,
                kRows,
                kCols,
                expected_output_payload)) {
            std::cerr << "MXFP8 CPU 反量化输出 QDTENSOR 的 header 或 payload 不正确。\n";
            return false;
        }
    }

    return true;
}

}  // namespace

/**
 * @brief 运行 CPU MXFP8 量化、QDWGT I/O 与反量化端到端测试。
 *
 * @return 所有 QDTENSOR -> CPU quantize -> QDWGT -> CPU dequantize 步骤通过时返回 0。
 */
int run_mxfp8_reference_tests() {
    const auto timestamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() /
        ("quant_dequant_mxfp8_reference_test_" + std::to_string(timestamp));

    std::error_code error_code{};
    if (!std::filesystem::create_directory(directory, error_code) || error_code) {
        std::cerr << "无法创建 MXFP8 CPU reference 测试临时目录。\n";
        return 1;
    }
    const TemporaryDirectoryCleanup cleanup{directory};

    try {
        if (!test_mxfp8_cpu_round_trip_with_io(directory)) {
            return 1;
        }
    } catch (const quant_dequant::TensorIoError& error) {
        std::cerr << "MXFP8 CPU reference 测试发生 Tensor I/O 错误："
                  << error.what() << '\n';
        return 1;
    } catch (const quant_dequant::QuantizedIoError& error) {
        std::cerr << "MXFP8 CPU reference 测试发生 Quantized I/O 错误："
                  << error.what() << '\n';
        return 1;
    } catch (const quant_dequant::ReferenceError& error) {
        std::cerr << "MXFP8 CPU reference 测试发生量化错误："
                  << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "MXFP8 CPU reference 测试发生未预期异常："
                  << error.what() << '\n';
        return 1;
    }

    return 0;
}

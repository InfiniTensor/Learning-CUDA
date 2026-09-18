#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "quant_dequant/quantized_io.hpp"

namespace {

constexpr std::size_t kHeaderBytes = 128U;
constexpr std::size_t kFormatOffset = 13U;
constexpr std::size_t kScaleModeOffset = 14U;
constexpr std::size_t kPayloadElementBitsOffset = 17U;
constexpr std::size_t kPayloadOffsetOffset = 48U;
constexpr std::size_t kLocalScaleOffsetOffset = 64U;
constexpr std::size_t kGlobalScaleOffset = 88U;
constexpr std::size_t kFlagsOffset = 92U;

constexpr std::array<std::uint8_t, 8U> kQuantizedMagic{
    'Q', 'D', 'W', 'G', 'T', 0U, 0U, 0U,
};

/**
 * @brief 在析构时删除量化 I/O 测试专用的临时目录。
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
 * @brief 从 little-endian 字节区读取无符号整数。
 *
 * @tparam UInt 目标无符号整数类型。
 * @param bytes 输入字节区。
 * @param offset 整数起始 offset。
 * @return 解码后的无符号整数。
 */
template <typename UInt>
[[nodiscard]] UInt load_unsigned_le(const std::span<const std::uint8_t> bytes,
                                    const std::size_t offset) noexcept {
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
void store_unsigned_le(const std::span<std::uint8_t> bytes,
                       const std::size_t offset,
                       const UInt value) noexcept {
    for (std::size_t index = 0U; index < sizeof(UInt); ++index) {
        bytes[offset + index] = static_cast<std::uint8_t>(
            value >> static_cast<unsigned int>(index * 8U));
    }
}

/**
 * @brief 读取测试生成的完整二进制文件。
 *
 * @param file_path 待读取的文件路径。
 * @param bytes 输出字节数组。
 * @return 文件完整读取成功时返回 true。
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
 * @brief 将完整二进制字节写入测试临时目录。
 *
 * @param file_path 目标文件路径。
 * @param bytes 完整文件内容。
 * @return 文件完整写入成功时返回 true。
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
 * @brief 判断给定文件是否被 QDWGT 读取器正确拒绝。
 *
 * @param file_path 待读取的损坏文件。
 * @return 抛出 `QuantizedIoError` 时返回 true。
 */
[[nodiscard]] bool is_rejected_by_reader(const std::filesystem::path& file_path) {
    try {
        static_cast<void>(quant_dequant::read_quantized_tensor(file_path));
    } catch (const quant_dequant::QuantizedIoError&) {
        return true;
    }

    return false;
}

/**
 * @brief 构造覆盖 rowwise 尾 block 的 MXFP8 量化结果。
 *
 * @return 形状为 2 x 33、具有四个 E8M0 local scale 的合法结果。
 */
[[nodiscard]] quant_dequant::QuantizedTensor make_mxfp8_block_tensor() {
    quant_dequant::QuantizedTensor tensor{
        .desc = {
            .source_desc = {
                .num_rows = 2U,
                .num_cols = 33U,
                .dtype = quant_dequant::DType::kFloat32,
            },
            .format = quant_dequant::QuantFormat::kMxfp8,
            .scale_mode = quant_dequant::ScaleMode::kBlock,
            .rounding = quant_dequant::RoundingMode::kNearest,
            .stochastic_seed = 0U,
            .block_size = quant_dequant::kMxfp8BlockSize,
            .scale_layout = quant_dequant::ScaleLayout::kRowwise,
        },
        .payload = std::vector<std::uint8_t>(66U),
        .local_scales = {0x7EU, 0x7FU, 0x80U, 0x81U},
        .global_scale = std::nullopt,
    };

    for (std::size_t index = 0U; index < tensor.payload.size(); ++index) {
        tensor.payload[index] = static_cast<std::uint8_t>(index * 17U + 3U);
    }

    return tensor;
}

/**
 * @brief 构造覆盖奇数元素尾 nibble 的 NVFP4 量化结果。
 *
 * @return 形状为 1 x 17、具有两个 E4M3 local scale 的合法结果。
 */
[[nodiscard]] quant_dequant::QuantizedTensor make_nvfp4_block_tensor() {
    return {
        .desc = {
            .source_desc = {
                .num_rows = 1U,
                .num_cols = 17U,
                .dtype = quant_dequant::DType::kFloat16,
            },
            .format = quant_dequant::QuantFormat::kNvfp4,
            .scale_mode = quant_dequant::ScaleMode::kBlock,
            .rounding = quant_dequant::RoundingMode::kStochastic,
            .stochastic_seed = 123456U,
            .block_size = quant_dequant::kNvfp4BlockSize,
            .scale_layout = quant_dequant::ScaleLayout::kRowwise,
        },
        .payload = {
            0x21U, 0x43U, 0x65U, 0x07U, 0x89U, 0xABU, 0xCDU, 0xEFU, 0x05U,
        },
        .local_scales = {0x38U, 0x40U},
        .global_scale = 0.25F,
    };
}

/**
 * @brief 构造 tensor scale 模式下的 MXFP8 量化结果。
 *
 * @return 一个只有一个 local scale 的合法结果。
 */
[[nodiscard]] quant_dequant::QuantizedTensor make_mxfp8_tensor_scale_tensor() {
    return {
        .desc = {
            .source_desc = {
                .num_rows = 2U,
                .num_cols = 3U,
                .dtype = quant_dequant::DType::kFloat32,
            },
            .format = quant_dequant::QuantFormat::kMxfp8,
            .scale_mode = quant_dequant::ScaleMode::kTensor,
            .rounding = quant_dequant::RoundingMode::kNearest,
            .stochastic_seed = 0U,
            .block_size = quant_dequant::kMxfp8BlockSize,
            .scale_layout = quant_dequant::ScaleLayout::kRowwise,
        },
        .payload = {0x00U, 0x01U, 0x02U, 0x03U, 0x04U, 0x05U},
        .local_scales = {0x7FU},
        .global_scale = std::nullopt,
    };
}

/**
 * @brief 比较两个量化结果中的全部公共元数据和二进制数组。
 *
 * 测试样本中的 global scale 均是可精确表示的有限数，因此可直接比较 FP32
 * bit pattern，而不是使用误差阈值。
 *
 * @param left 左侧量化结果。
 * @param right 右侧量化结果。
 * @return 全部字段一致时返回 true。
 */
[[nodiscard]] bool quantized_tensors_equal(
    const quant_dequant::QuantizedTensor& left,
    const quant_dequant::QuantizedTensor& right) {
    const auto& left_desc = left.desc;
    const auto& right_desc = right.desc;

    const bool same_metadata =
        left_desc.source_desc.num_rows == right_desc.source_desc.num_rows &&
        left_desc.source_desc.num_cols == right_desc.source_desc.num_cols &&
        left_desc.source_desc.dtype == right_desc.source_desc.dtype &&
        left_desc.format == right_desc.format &&
        left_desc.scale_mode == right_desc.scale_mode &&
        left_desc.rounding == right_desc.rounding &&
        left_desc.stochastic_seed == right_desc.stochastic_seed &&
        left_desc.block_size == right_desc.block_size &&
        left_desc.scale_layout == right_desc.scale_layout;

    if (!same_metadata || left.payload != right.payload ||
        left.local_scales != right.local_scales ||
        left.global_scale.has_value() != right.global_scale.has_value()) {
        return false;
    }

    return !left.global_scale.has_value() ||
           std::bit_cast<std::uint32_t>(*left.global_scale) ==
               std::bit_cast<std::uint32_t>(*right.global_scale);
}

}  // namespace

/**
 * @brief 验证 QDWGT 的真实文件 round-trip、物理 header 和损坏文件拒绝逻辑。
 *
 * @return 所有断言通过时返回 0；否则打印错误并返回 1。
 */
int run_quantized_io_tests() {
    const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path test_directory =
        std::filesystem::temp_directory_path() /
        ("quant_dequant_quantized_io_test_" + std::to_string(timestamp));
    const TemporaryDirectoryCleanup cleanup{test_directory};

    std::error_code error_code{};
    if (!std::filesystem::create_directories(test_directory, error_code) || error_code) {
        std::cerr << "无法创建量化 I/O 测试临时目录。\n";
        return 1;
    }

    const std::filesystem::path mxfp8_path = test_directory / "mxfp8_block.qdwgt";
    const quant_dequant::QuantizedTensor mxfp8_tensor = make_mxfp8_block_tensor();
    if (!mxfp8_tensor.isConsistent()) {
        std::cerr << "MXFP8 测试输入不自洽。\n";
        return 1;
    }

    try {
        quant_dequant::write_quantized_tensor(mxfp8_path, mxfp8_tensor);
    } catch (const quant_dequant::QuantizedIoError& error) {
        std::cerr << "MXFP8 QDWGT 写入意外失败：" << error.what() << '\n';
        return 1;
    }

    std::vector<std::uint8_t> mxfp8_bytes{};
    if (!read_binary_file(mxfp8_path, &mxfp8_bytes) || mxfp8_bytes.size() != 204U ||
        !std::equal(kQuantizedMagic.begin(), kQuantizedMagic.end(),
                    mxfp8_bytes.begin()) ||
        mxfp8_bytes[kFormatOffset] !=
            static_cast<std::uint8_t>(quant_dequant::QuantFormat::kMxfp8) ||
        mxfp8_bytes[kPayloadElementBitsOffset] != 8U ||
        load_unsigned_le<std::uint64_t>(mxfp8_bytes, kPayloadOffsetOffset) != 128U ||
        load_unsigned_le<std::uint64_t>(mxfp8_bytes, kLocalScaleOffsetOffset) != 200U ||
        load_unsigned_le<std::uint32_t>(mxfp8_bytes, kGlobalScaleOffset) != 0x3F800000U ||
        load_unsigned_le<std::uint32_t>(mxfp8_bytes, kFlagsOffset) != 0U ||
        !std::equal(mxfp8_tensor.payload.begin(), mxfp8_tensor.payload.end(),
                    mxfp8_bytes.begin() + static_cast<std::ptrdiff_t>(kHeaderBytes)) ||
        !std::all_of(mxfp8_bytes.begin() + 194, mxfp8_bytes.begin() + 200,
                     [](const std::uint8_t value) { return value == 0U; }) ||
        !std::equal(mxfp8_tensor.local_scales.begin(), mxfp8_tensor.local_scales.end(),
                    mxfp8_bytes.begin() + 200)) {
        std::cerr << "MXFP8 QDWGT 的实际 header、section 或对齐布局错误。\n";
        return 1;
    }

    try {
        if (!quantized_tensors_equal(mxfp8_tensor,
                                     quant_dequant::read_quantized_tensor(mxfp8_path))) {
            std::cerr << "MXFP8 QDWGT round-trip 后内容不一致。\n";
            return 1;
        }
    } catch (const quant_dequant::QuantizedIoError& error) {
        std::cerr << "MXFP8 QDWGT 读取意外失败：" << error.what() << '\n';
        return 1;
    }

    const std::filesystem::path nvfp4_path = test_directory / "nvfp4_block.qdwgt";
    const quant_dequant::QuantizedTensor nvfp4_tensor = make_nvfp4_block_tensor();
    if (!nvfp4_tensor.isConsistent()) {
        std::cerr << "NVFP4 测试输入不自洽。\n";
        return 1;
    }

    try {
        quant_dequant::write_quantized_tensor(nvfp4_path, nvfp4_tensor);
    } catch (const quant_dequant::QuantizedIoError& error) {
        std::cerr << "NVFP4 QDWGT 写入意外失败：" << error.what() << '\n';
        return 1;
    }

    std::vector<std::uint8_t> nvfp4_bytes{};
    if (!read_binary_file(nvfp4_path, &nvfp4_bytes) || nvfp4_bytes.size() != 146U ||
        nvfp4_bytes[kFormatOffset] !=
            static_cast<std::uint8_t>(quant_dequant::QuantFormat::kNvfp4) ||
        nvfp4_bytes[kPayloadElementBitsOffset] != 4U ||
        load_unsigned_le<std::uint64_t>(nvfp4_bytes, kPayloadOffsetOffset) != 128U ||
        load_unsigned_le<std::uint64_t>(nvfp4_bytes, kLocalScaleOffsetOffset) != 144U ||
        load_unsigned_le<std::uint32_t>(nvfp4_bytes, kGlobalScaleOffset) != 0x3E800000U ||
        load_unsigned_le<std::uint32_t>(nvfp4_bytes, kFlagsOffset) != 1U ||
        !std::equal(nvfp4_tensor.payload.begin(), nvfp4_tensor.payload.end(),
                    nvfp4_bytes.begin() + static_cast<std::ptrdiff_t>(kHeaderBytes)) ||
        !std::all_of(nvfp4_bytes.begin() + 137, nvfp4_bytes.begin() + 144,
                     [](const std::uint8_t value) { return value == 0U; }) ||
        !std::equal(nvfp4_tensor.local_scales.begin(), nvfp4_tensor.local_scales.end(),
                    nvfp4_bytes.begin() + 144)) {
        std::cerr << "NVFP4 QDWGT 的实际 header、section 或对齐布局错误。\n";
        return 1;
    }

    try {
        if (!quantized_tensors_equal(nvfp4_tensor,
                                     quant_dequant::read_quantized_tensor(nvfp4_path))) {
            std::cerr << "NVFP4 QDWGT round-trip 后内容不一致。\n";
            return 1;
        }
    } catch (const quant_dequant::QuantizedIoError& error) {
        std::cerr << "NVFP4 QDWGT 读取意外失败：" << error.what() << '\n';
        return 1;
    }

    const std::filesystem::path tensor_scale_path = test_directory / "mxfp8_tensor.qdwgt";
    const quant_dequant::QuantizedTensor tensor_scale_tensor =
        make_mxfp8_tensor_scale_tensor();
    try {
        quant_dequant::write_quantized_tensor(tensor_scale_path, tensor_scale_tensor);
        if (!quantized_tensors_equal(
                tensor_scale_tensor,
                quant_dequant::read_quantized_tensor(tensor_scale_path))) {
            std::cerr << "MXFP8 tensor scale 模式 round-trip 后内容不一致。\n";
            return 1;
        }
    } catch (const quant_dequant::QuantizedIoError& error) {
        std::cerr << "MXFP8 tensor scale 模式 I/O 意外失败：" << error.what() << '\n';
        return 1;
    }

    const std::filesystem::path malformed_path = test_directory / "malformed.qdwgt";
    std::vector<std::uint8_t> malformed_bytes = nvfp4_bytes;

    malformed_bytes[0U] = 'X';
    if (!write_binary_file(malformed_path, malformed_bytes) ||
        !is_rejected_by_reader(malformed_path)) {
        std::cerr << "损坏 magic 未被 QDWGT 读取器拒绝。\n";
        return 1;
    }

    malformed_bytes = nvfp4_bytes;
    malformed_bytes[kScaleModeOffset] =
        static_cast<std::uint8_t>(quant_dequant::ScaleMode::kTensor);
    if (!write_binary_file(malformed_path, malformed_bytes) ||
        !is_rejected_by_reader(malformed_path)) {
        std::cerr << "NVFP4 tensor scale QDWGT 未被读取器拒绝。\n";
        return 1;
    }

    malformed_bytes = nvfp4_bytes;
    malformed_bytes[kPayloadElementBitsOffset] = 8U;
    if (!write_binary_file(malformed_path, malformed_bytes) ||
        !is_rejected_by_reader(malformed_path)) {
        std::cerr << "错误 payload_element_bits 未被 QDWGT 读取器拒绝。\n";
        return 1;
    }

    malformed_bytes = nvfp4_bytes;
    malformed_bytes[kHeaderBytes + 8U] = 0xF5U;
    if (!write_binary_file(malformed_path, malformed_bytes) ||
        !is_rejected_by_reader(malformed_path)) {
        std::cerr << "NVFP4 尾部高 nibble 非零未被 QDWGT 读取器拒绝。\n";
        return 1;
    }

    malformed_bytes = nvfp4_bytes;
    store_unsigned_le<std::uint32_t>(malformed_bytes, kFlagsOffset, 0U);
    if (!write_binary_file(malformed_path, malformed_bytes) ||
        !is_rejected_by_reader(malformed_path)) {
        std::cerr << "NVFP4 尾 nibble flags 错误未被 QDWGT 读取器拒绝。\n";
        return 1;
    }

    malformed_bytes = nvfp4_bytes;
    store_unsigned_le<std::uint64_t>(malformed_bytes, kLocalScaleOffsetOffset, 145U);
    if (!write_binary_file(malformed_path, malformed_bytes) ||
        !is_rejected_by_reader(malformed_path)) {
        std::cerr << "未对齐 local_scale_offset 未被 QDWGT 读取器拒绝。\n";
        return 1;
    }

    malformed_bytes = mxfp8_bytes;
    store_unsigned_le<std::uint32_t>(malformed_bytes, kGlobalScaleOffset, 0U);
    if (!write_binary_file(malformed_path, malformed_bytes) ||
        !is_rejected_by_reader(malformed_path)) {
        std::cerr << "MXFP8 非 1.0F global_scale 未被 QDWGT 读取器拒绝。\n";
        return 1;
    }

    malformed_bytes = nvfp4_bytes;
    malformed_bytes.pop_back();
    if (!write_binary_file(malformed_path, malformed_bytes) ||
        !is_rejected_by_reader(malformed_path)) {
        std::cerr << "截断 QDWGT 文件未被读取器拒绝。\n";
        return 1;
    }

    return 0;
}

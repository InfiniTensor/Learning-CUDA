#include "low_precision/io.hpp"

#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace low_precision {
namespace {

constexpr std::array<char, 8> kMagic{'L', 'P', 'C', 'Q', 'N', 'T', '1', '\0'};
constexpr std::uint32_t kVersion = 2;
constexpr std::size_t kMxBlock = 32;
constexpr std::size_t kNvBlock = 16;

template <typename T>
void write_scalar(std::ofstream& output, T value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!output) {
        throw std::runtime_error("failed to write quantized file");
    }
}

template <typename T>
T read_scalar(std::ifstream& input, const char* field) {
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) {
        throw std::runtime_error(std::string("truncated quantized file at ") + field);
    }
    return value;
}

std::size_t checked_elements(std::size_t rows, std::size_t cols) {
    if (rows == 0 || cols == 0 || rows > std::numeric_limits<std::size_t>::max() / cols) {
        throw std::runtime_error("invalid matrix dimensions in quantized file");
    }
    return rows * cols;
}

std::size_t block_count(std::size_t cols, std::size_t block_size) {
    return (cols + block_size - 1) / block_size;
}

void validate(const QuantizedFile& file) {
    const std::size_t elements = checked_elements(file.rows, file.cols);
    if (file.format != Format::MxFp8 && file.format != Format::NvFp4) {
        throw std::runtime_error("unsupported quantization format");
    }
    if (file.input_type != InputType::Fp32 && file.input_type != InputType::Fp16) {
        throw std::runtime_error("unsupported input type");
    }
    if (file.rounding != Rounding::NearestEven && file.rounding != Rounding::Stochastic) {
        throw std::runtime_error("unsupported rounding mode");
    }

    const std::uint32_t expected_block_size =
        file.format == Format::MxFp8 ? static_cast<std::uint32_t>(kMxBlock)
                                     : static_cast<std::uint32_t>(kNvBlock);
    const ScaleMode expected_scale_mode =
        file.format == Format::MxFp8 ? ScaleMode::Block : ScaleMode::TensorAndBlock;
    if (file.block_size != expected_block_size || file.scale_mode != expected_scale_mode) {
        throw std::runtime_error("scale metadata does not match quantization format");
    }

    std::size_t expected_values = elements;
    std::size_t expected_scales = file.rows * block_count(file.cols, kMxBlock);
    if (file.format == Format::NvFp4) {
        expected_values = (elements + 1) / 2;
        expected_scales = file.rows * block_count(file.cols, kNvBlock);
        if (!(file.global_scale > 0.0f) || !std::isfinite(file.global_scale)) {
            throw std::runtime_error("invalid NVFP4 global scale");
        }
    }
    if (file.values.size() != expected_values || file.scales.size() != expected_scales) {
        throw std::runtime_error("quantized payload size does not match matrix dimensions");
    }
}

}  // namespace

void write_quantized_file(const std::string& path, const QuantizedFile& file) {
    validate(file);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot open quantized output file: " + path);
    }

    output.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
    write_scalar(output, kVersion);
    write_scalar(output, static_cast<std::uint32_t>(file.format));
    write_scalar(output, static_cast<std::uint32_t>(file.input_type));
    write_scalar(output, static_cast<std::uint32_t>(file.rounding));
    write_scalar(output, static_cast<std::uint64_t>(file.rows));
    write_scalar(output, static_cast<std::uint64_t>(file.cols));
    write_scalar(output, static_cast<std::uint64_t>(file.values.size()));
    write_scalar(output, static_cast<std::uint64_t>(file.scales.size()));
    write_scalar(output, file.global_scale);
    const std::uint64_t scale_metadata = static_cast<std::uint64_t>(file.block_size) |
                                         (static_cast<std::uint64_t>(file.scale_mode) << 32u);
    write_scalar(output, scale_metadata);
    output.write(reinterpret_cast<const char*>(file.values.data()),
                 static_cast<std::streamsize>(file.values.size()));
    output.write(reinterpret_cast<const char*>(file.scales.data()),
                 static_cast<std::streamsize>(file.scales.size()));
    if (!output) {
        throw std::runtime_error("failed to write quantized payload");
    }
}

QuantizedFile read_quantized_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open quantized input file: " + path);
    }

    std::array<char, 8> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!input || magic != kMagic) {
        throw std::runtime_error("invalid quantized file magic");
    }
    const std::uint32_t version = read_scalar<std::uint32_t>(input, "version");
    if (version != 1 && version != kVersion) {
        throw std::runtime_error("unsupported quantized file version");
    }

    QuantizedFile file;
    const std::uint32_t format = read_scalar<std::uint32_t>(input, "format");
    const std::uint32_t input_type = read_scalar<std::uint32_t>(input, "input type");
    const std::uint32_t rounding = read_scalar<std::uint32_t>(input, "rounding");
    const std::uint64_t rows = read_scalar<std::uint64_t>(input, "rows");
    const std::uint64_t cols = read_scalar<std::uint64_t>(input, "cols");
    const std::uint64_t values_size = read_scalar<std::uint64_t>(input, "values size");
    const std::uint64_t scales_size = read_scalar<std::uint64_t>(input, "scales size");
    file.global_scale = read_scalar<float>(input, "global scale");
    const std::uint64_t scale_metadata = read_scalar<std::uint64_t>(input, "scale metadata");

    if (rows > std::numeric_limits<std::size_t>::max() ||
        cols > std::numeric_limits<std::size_t>::max() ||
        values_size > std::numeric_limits<std::size_t>::max() ||
        scales_size > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("quantized file dimensions exceed host limits");
    }
    file.format = static_cast<Format>(format);
    file.input_type = static_cast<InputType>(input_type);
    file.rounding = static_cast<Rounding>(rounding);
    file.rows = static_cast<std::size_t>(rows);
    file.cols = static_cast<std::size_t>(cols);
    if (version == 1) {
        file.block_size = file.format == Format::MxFp8 ? static_cast<std::uint32_t>(kMxBlock)
                                                       : static_cast<std::uint32_t>(kNvBlock);
        file.scale_mode = file.format == Format::MxFp8 ? ScaleMode::Block
                                                        : ScaleMode::TensorAndBlock;
    } else {
        file.block_size = static_cast<std::uint32_t>(scale_metadata & 0xffffffffu);
        file.scale_mode = static_cast<ScaleMode>(scale_metadata >> 32u);
    }
    file.values.resize(static_cast<std::size_t>(values_size));
    file.scales.resize(static_cast<std::size_t>(scales_size));
    input.read(reinterpret_cast<char*>(file.values.data()),
               static_cast<std::streamsize>(file.values.size()));
    input.read(reinterpret_cast<char*>(file.scales.data()),
               static_cast<std::streamsize>(file.scales.size()));
    if (!input) {
        throw std::runtime_error("truncated quantized payload");
    }
    char extra = 0;
    if (input.get(extra)) {
        throw std::runtime_error("unexpected trailing data in quantized file");
    }
    validate(file);
    return file;
}

}  // namespace low_precision

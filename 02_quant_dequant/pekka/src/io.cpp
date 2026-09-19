#include "qd/io.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace qd {
namespace {

#pragma pack(push, 1)
struct QdatHeader {
  char magic[4];
  std::uint16_t version;
  std::uint8_t dtype;
  std::uint8_t reserved;
  std::uint64_t rows;
  std::uint64_t cols;
};

struct QlowHeader {
  char magic[4];
  std::uint16_t version;
  std::uint16_t header_bytes;
  std::uint8_t format;
  std::uint8_t original_dtype;
  std::uint8_t scale_mode;
  std::uint8_t rounding;
  std::uint64_t rows;
  std::uint64_t cols;
  std::uint32_t block_size;
  std::uint32_t reserved0;
  std::uint64_t packed_bytes;
  std::uint64_t scale_count;
  std::uint64_t data_offset;
  std::uint64_t scale_offset;
  std::uint64_t global_scale_offset;
  std::uint32_t checksum;
  std::uint32_t reserved1;
  std::uint64_t total_bytes;
};
#pragma pack(pop)

static_assert(sizeof(QdatHeader) == 24);
static_assert(sizeof(QlowHeader) == 92);

std::uint64_t align16(std::uint64_t value) { return (value + 15U) & ~15ULL; }

void require_little_endian() {
  const std::uint16_t marker = 1;
  if (*reinterpret_cast<const std::uint8_t *>(&marker) != 1) {
    throw std::runtime_error("QDAT/QLOW v1 requires a little-endian host");
  }
}

template <typename T> T read_scalar(const std::uint8_t *source) {
  T value{};
  std::memcpy(&value, source, sizeof(T));
  return value;
}

// IEEE conversion is implemented explicitly so the I/O layer does not depend
// on CUDA headers and can be unit-tested on a CPU-only build machine.
float half_to_float(std::uint16_t bits) {
  const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16U;
  std::uint32_t exponent = (bits >> 10U) & 0x1FU;
  std::uint32_t mantissa = bits & 0x03FFU;
  std::uint32_t output = 0;
  if (exponent == 0) {
    if (mantissa == 0) output = sign;
    else {
      int shift = 0;
      while ((mantissa & 0x0400U) == 0) { mantissa <<= 1U; ++shift; }
      mantissa &= 0x03FFU;
      output = sign | static_cast<std::uint32_t>(127 - 14 - shift) << 23U |
               mantissa << 13U;
    }
  } else if (exponent == 31) {
    output = sign | 0x7F800000U | mantissa << 13U;
  } else {
    output = sign | (exponent + (127 - 15)) << 23U | mantissa << 13U;
  }
  float value = 0.0F;
  std::memcpy(&value, &output, sizeof(value));
  return value;
}

std::uint32_t fnv1a(const std::vector<std::uint8_t> &values,
                    const std::vector<std::uint8_t> &scales, float global) {
  std::uint32_t hash = 2166136261U;
  const auto add = [&hash](std::uint8_t byte) { hash = (hash ^ byte) * 16777619U; };
  for (std::uint8_t byte : values) add(byte);
  for (std::uint8_t byte : scales) add(byte);
  std::uint8_t bytes[sizeof(float)];
  std::memcpy(bytes, &global, sizeof(float));
  for (std::uint8_t byte : bytes) add(byte);
  return hash;
}

void write_padding(std::ofstream &output, std::uint64_t current,
                   std::uint64_t target) {
  const std::vector<char> zeros(static_cast<std::size_t>(target - current), 0);
  output.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
}

} // namespace

Matrix read_qdat(const std::filesystem::path &path) {
  require_little_endian();
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open QDAT: " + path.string());
  QdatHeader header{};
  input.read(reinterpret_cast<char *>(&header), sizeof(header));
  if (!input || std::memcmp(header.magic, "QDAT", 4) != 0 || header.version != 1 ||
      header.reserved != 0) {
    throw std::runtime_error("invalid or truncated QDAT header: " + path.string());
  }
  if (header.dtype != static_cast<std::uint8_t>(DataType::kFp16) &&
      header.dtype != static_cast<std::uint8_t>(DataType::kFp32) &&
      header.dtype != static_cast<std::uint8_t>(DataType::kBf16)) {
    throw std::runtime_error("unsupported QDAT dtype");
  }
  if (header.rows == 0 || header.cols == 0 ||
      header.rows > std::numeric_limits<std::uint64_t>::max() / header.cols) {
    throw std::runtime_error("invalid QDAT shape");
  }
  Matrix matrix{header.rows, header.cols, static_cast<DataType>(header.dtype), {}};
  const std::size_t count = element_count(header.rows, header.cols);
  const std::size_t item_size = data_type_size(matrix.original_type);
  std::vector<std::uint8_t> payload(count * item_size);
  input.read(reinterpret_cast<char *>(payload.data()),
             static_cast<std::streamsize>(payload.size()));
  if (input.gcount() != static_cast<std::streamsize>(payload.size()) || input.peek() != EOF) {
    throw std::runtime_error("QDAT payload size does not match shape");
  }
  matrix.values.resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    if (matrix.original_type == DataType::kFp32) {
      matrix.values[i] = read_scalar<float>(payload.data() + i * 4U);
    } else if (matrix.original_type == DataType::kFp16) {
      matrix.values[i] = half_to_float(read_scalar<std::uint16_t>(payload.data() + i * 2U));
    } else {
      const std::uint32_t bits = static_cast<std::uint32_t>(
          read_scalar<std::uint16_t>(payload.data() + i * 2U)) << 16U;
      std::memcpy(&matrix.values[i], &bits, sizeof(float));
    }
  }
  return matrix;
}

void write_qdat(const std::filesystem::path &path, std::uint64_t rows,
                std::uint64_t cols, DataType type,
                const std::vector<std::uint8_t> &payload) {
  require_little_endian();
  if (payload.size() != element_count(rows, cols) * data_type_size(type)) {
    throw std::runtime_error("output QDAT payload has incorrect size");
  }
  std::ofstream output(path, std::ios::binary);
  if (!output) throw std::runtime_error("cannot create QDAT: " + path.string());
  QdatHeader header{{'Q', 'D', 'A', 'T'}, 1, static_cast<std::uint8_t>(type), 0,
                    rows, cols};
  output.write(reinterpret_cast<const char *>(&header), sizeof(header));
  output.write(reinterpret_cast<const char *>(payload.data()),
               static_cast<std::streamsize>(payload.size()));
  if (!output) throw std::runtime_error("failed while writing QDAT");
}

std::uint64_t qlow_file_size(const QuantizedData &data) {
  const std::uint64_t data_offset = align16(sizeof(QlowHeader));
  const std::uint64_t scale_offset = align16(data_offset + data.packed_values.size());
  const std::uint64_t global_offset = align16(scale_offset + data.local_scales.size());
  return global_offset + sizeof(float);
}

void write_qlow(const std::filesystem::path &path, const QuantizedData &data) {
  require_little_endian();
  const std::uint64_t data_offset = align16(sizeof(QlowHeader));
  const std::uint64_t scale_offset = align16(data_offset + data.packed_values.size());
  const std::uint64_t global_offset = align16(scale_offset + data.local_scales.size());
  QlowHeader header{{'Q', 'L', 'O', 'W'}, 1, sizeof(QlowHeader),
                    static_cast<std::uint8_t>(data.format),
                    static_cast<std::uint8_t>(data.original_type),
                    static_cast<std::uint8_t>(data.scale_mode),
                    static_cast<std::uint8_t>(data.rounding), data.rows, data.cols,
                    data.block_size, 0, data.packed_values.size(),
                    data.local_scales.size(), data_offset, scale_offset,
                    global_offset,
                    fnv1a(data.packed_values, data.local_scales, data.global_scale),
                    0, global_offset + sizeof(float)};
  std::ofstream output(path, std::ios::binary);
  if (!output) throw std::runtime_error("cannot create QLOW: " + path.string());
  output.write(reinterpret_cast<const char *>(&header), sizeof(header));
  write_padding(output, sizeof(header), data_offset);
  output.write(reinterpret_cast<const char *>(data.packed_values.data()),
               static_cast<std::streamsize>(data.packed_values.size()));
  write_padding(output, data_offset + data.packed_values.size(), scale_offset);
  output.write(reinterpret_cast<const char *>(data.local_scales.data()),
               static_cast<std::streamsize>(data.local_scales.size()));
  write_padding(output, scale_offset + data.local_scales.size(), global_offset);
  output.write(reinterpret_cast<const char *>(&data.global_scale), sizeof(float));
  if (!output) throw std::runtime_error("failed while writing QLOW");
}

QuantizedData read_qlow(const std::filesystem::path &path) {
  require_little_endian();
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("cannot open QLOW: " + path.string());
  const auto file_size = static_cast<std::uint64_t>(input.tellg());
  input.seekg(0);
  QlowHeader header{};
  input.read(reinterpret_cast<char *>(&header), sizeof(header));
  if (!input || std::memcmp(header.magic, "QLOW", 4) != 0 || header.version != 1 ||
      header.header_bytes != sizeof(header) || header.total_bytes != file_size) {
    throw std::runtime_error("invalid or truncated QLOW header");
  }
  if ((header.format != static_cast<std::uint8_t>(Format::kMxfp8) &&
       header.format != static_cast<std::uint8_t>(Format::kNvfp4)) ||
      (header.original_dtype != static_cast<std::uint8_t>(DataType::kFp16) &&
       header.original_dtype != static_cast<std::uint8_t>(DataType::kFp32) &&
       header.original_dtype != static_cast<std::uint8_t>(DataType::kBf16)) ||
      (header.scale_mode != static_cast<std::uint8_t>(ScaleMode::kTensor) &&
       header.scale_mode != static_cast<std::uint8_t>(ScaleMode::kBlock)) ||
      (header.rounding != static_cast<std::uint8_t>(Rounding::kNearest) &&
       header.rounding != static_cast<std::uint8_t>(Rounding::kStochastic)) ||
      header.rows == 0 || header.cols == 0 || header.block_size == 0 ||
      header.rows > std::numeric_limits<std::uint64_t>::max() / header.cols) {
    throw std::runtime_error("QLOW metadata contains an unsupported value");
  }
  const std::uint64_t count = header.rows * header.cols;
  const std::uint64_t expected_packed =
      header.format == static_cast<std::uint8_t>(Format::kMxfp8)
          ? count
          : (count + 1U) / 2U;
  const std::uint64_t expected_scales =
      header.scale_mode == static_cast<std::uint8_t>(ScaleMode::kTensor)
          ? 1U
          : (count + header.block_size - 1U) / header.block_size;
  if (header.packed_bytes != expected_packed ||
      header.scale_count != expected_scales ||
      header.data_offset % 16U != 0 || header.scale_offset % 16U != 0 ||
      header.global_scale_offset % 16U != 0 ||
      header.data_offset < align16(sizeof(header)) ||
      header.scale_offset < header.data_offset + header.packed_bytes ||
      header.global_scale_offset < header.scale_offset + header.scale_count) {
    throw std::runtime_error("QLOW section sizes or offsets are inconsistent");
  }
  if (header.data_offset + header.packed_bytes > file_size ||
      header.scale_offset + header.scale_count > file_size ||
      header.global_scale_offset + sizeof(float) > file_size) {
    throw std::runtime_error("QLOW section lies outside file");
  }
  QuantizedData data;
  data.format = static_cast<Format>(header.format);
  data.original_type = static_cast<DataType>(header.original_dtype);
  data.scale_mode = static_cast<ScaleMode>(header.scale_mode);
  data.rounding = static_cast<Rounding>(header.rounding);
  data.rows = header.rows;
  data.cols = header.cols;
  data.block_size = header.block_size;
  data.packed_values.resize(static_cast<std::size_t>(header.packed_bytes));
  data.local_scales.resize(static_cast<std::size_t>(header.scale_count));
  input.seekg(static_cast<std::streamoff>(header.data_offset));
  input.read(reinterpret_cast<char *>(data.packed_values.data()),
             static_cast<std::streamsize>(data.packed_values.size()));
  input.seekg(static_cast<std::streamoff>(header.scale_offset));
  input.read(reinterpret_cast<char *>(data.local_scales.data()),
             static_cast<std::streamsize>(data.local_scales.size()));
  input.seekg(static_cast<std::streamoff>(header.global_scale_offset));
  input.read(reinterpret_cast<char *>(&data.global_scale), sizeof(float));
  if (!input || fnv1a(data.packed_values, data.local_scales, data.global_scale) !=
                    header.checksum) {
    throw std::runtime_error("QLOW checksum mismatch or truncated section");
  }
  if (!std::isfinite(data.global_scale) || data.global_scale <= 0.0F) {
    throw std::runtime_error("QLOW global scale must be finite and positive");
  }
  return data;
}

} // namespace qd

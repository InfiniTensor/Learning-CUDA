/*
 * tensor.bin文件IO。header固定20字节，不把C++ struct直接写盘，避免结构体填充
 * 和宿主字节序差异。data按行主序保存；FP16读取后转换为FP32供计算模块使用。
 */
#include "io.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace luxorion {
namespace {

constexpr std::size_t kHeaderBytes = 20;

std::uint64_t decode_u64_le(const std::uint8_t* bytes) {
  std::uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i) {
    value |= static_cast<std::uint64_t>(bytes[i]) << (8 * i);
  }
  return value;
}

void encode_u64_le(std::uint64_t value, std::uint8_t* bytes) {
  for (unsigned i = 0; i < 8; ++i) {
    bytes[i] = static_cast<std::uint8_t>((value >> (8 * i)) & 0xFFU);
  }
}

std::uint32_t decode_u32_le(const std::uint8_t* bytes) {
  std::uint32_t value = 0;
  for (unsigned i = 0; i < 4; ++i) {
    value |= static_cast<std::uint32_t>(bytes[i]) << (8 * i);
  }
  return value;
}

void encode_u32_le(std::uint32_t value, std::uint8_t* bytes) {
  for (unsigned i = 0; i < 4; ++i) {
    bytes[i] = static_cast<std::uint8_t>((value >> (8 * i)) & 0xFFU);
  }
}

std::uint16_t decode_u16_le(const std::uint8_t* bytes) {
  return static_cast<std::uint16_t>(bytes[0])
       | static_cast<std::uint16_t>(static_cast<unsigned>(bytes[1]) << 8);
}

void encode_u16_le(std::uint16_t value, std::uint8_t* bytes) {
  bytes[0] = static_cast<std::uint8_t>(value & 0xFFU);
  bytes[1] = static_cast<std::uint8_t>((value >> 8) & 0xFFU);
}

std::size_t checked_element_count(std::int64_t rows, std::int64_t cols) {
  if (rows <= 0 || cols <= 0) {
    throw std::invalid_argument("tensor dimensions must be positive");
  }
  const auto unsigned_rows = static_cast<std::uint64_t>(rows);
  const auto unsigned_cols = static_cast<std::uint64_t>(cols);
  if (unsigned_rows > std::numeric_limits<std::size_t>::max() / unsigned_cols) {
    throw std::length_error("tensor element count overflows size_t");
  }
  return static_cast<std::size_t>(unsigned_rows * unsigned_cols);
}

std::size_t element_bytes(TensorDType dtype) {
  switch (dtype) {
    case TensorDType::FP32: return 4;
    case TensorDType::FP16: return 2;
  }
  throw std::invalid_argument("unsupported tensor dtype");
}

TensorDType decode_dtype(const std::uint8_t* bytes) {
  if (std::memcmp(bytes, "fp32", 4) == 0) return TensorDType::FP32;
  if (std::memcmp(bytes, "fp16", 4) == 0) return TensorDType::FP16;
  throw std::invalid_argument("tensor dtype must be fp32 or fp16");
}

void encode_dtype(TensorDType dtype, std::uint8_t* bytes) {
  switch (dtype) {
    case TensorDType::FP32:
      std::memcpy(bytes, "fp32", 4);
      return;
    case TensorDType::FP16:
      std::memcpy(bytes, "fp16", 4);
      return;
  }
  throw std::invalid_argument("unsupported tensor dtype");
}

float decode_fp16(std::uint16_t code) {
  const unsigned sign = code >> 15;
  const unsigned exponent = (code >> 10) & 0x1FU;
  const unsigned mantissa = code & 0x3FFU;
  if (exponent == 0x1FU) {
    throw std::invalid_argument("tensor data must not contain FP16 NaN or Inf");
  }
  const double magnitude = exponent == 0
      ? std::ldexp(static_cast<double>(mantissa), -24)
      : std::ldexp(1.0 + static_cast<double>(mantissa) / 1024.0,
                   static_cast<int>(exponent) - 15);
  return std::copysign(static_cast<float>(magnitude), sign ? -1.0F : 1.0F);
}

int round_nearest_even(double value) {
  const int lower = static_cast<int>(std::floor(value));
  const double fraction = value - lower;
  return fraction > 0.5 || (fraction == 0.5 && (lower & 1))
      ? lower + 1 : lower;
}

std::uint16_t encode_fp16(float value) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument("tensor data must contain finite values");
  }
  const unsigned sign = std::signbit(value) ? 0x8000U : 0U;
  const double magnitude = std::fabs(static_cast<double>(value));
  if (magnitude > 65504.0) {
    throw std::overflow_error("FP32 value exceeds the finite FP16 range");
  }
  if (magnitude < std::ldexp(1.0, -14)) {
    const int mantissa = round_nearest_even(std::ldexp(magnitude, 24));
    return static_cast<std::uint16_t>(sign | static_cast<unsigned>(mantissa));
  }

  int power = 0;
  std::frexp(magnitude, &power);
  int exponent = power - 1;
  int significand = round_nearest_even(std::ldexp(magnitude, 10 - exponent));
  if (significand == 2048) {
    ++exponent;
    significand = 1024;
  }
  return static_cast<std::uint16_t>(
      sign | (static_cast<unsigned>(exponent + 15) << 10)
      | static_cast<unsigned>(significand - 1024));
}

std::size_t checked_file_size(std::size_t count, TensorDType dtype) {
  const std::size_t width = element_bytes(dtype);
  if (count > (std::numeric_limits<std::size_t>::max() - kHeaderBytes) / width) {
    throw std::length_error("tensor file size overflows size_t");
  }
  const std::size_t size = kHeaderBytes + count * width;
  if (size > static_cast<std::size_t>(
                 std::numeric_limits<std::streamoff>::max())) {
    throw std::length_error("tensor file is too large for stream offsets");
  }
  return size;
}

}  // namespace

TensorData read_tensor_file(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("cannot open tensor file: " + path);
  const std::streamoff end = file.tellg();
  if (end < static_cast<std::streamoff>(kHeaderBytes)) {
    throw std::invalid_argument("tensor file is shorter than the 20-byte header");
  }
  file.seekg(0);

  std::array<std::uint8_t, kHeaderBytes> header{};
  file.read(reinterpret_cast<char*>(header.data()), header.size());
  if (!file) throw std::runtime_error("failed to read tensor header");

  TensorData tensor;
  const std::uint64_t raw_rows = decode_u64_le(header.data());
  const std::uint64_t raw_cols = decode_u64_le(header.data() + 8);
  if (raw_rows > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
      || raw_cols > static_cast<std::uint64_t>(
          std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument("tensor dimensions must be positive int64 values");
  }
  tensor.num_rows = static_cast<std::int64_t>(raw_rows);
  tensor.num_cols = static_cast<std::int64_t>(raw_cols);
  tensor.dtype = decode_dtype(header.data() + 16);
  const std::size_t count = checked_element_count(tensor.num_rows, tensor.num_cols);
  const std::size_t expected_size = checked_file_size(count, tensor.dtype);
  if (end != static_cast<std::streamoff>(expected_size)) {
    throw std::invalid_argument("tensor file size does not match its header");
  }

  const std::size_t payload_size = expected_size - kHeaderBytes;
  std::vector<std::uint8_t> payload(payload_size);
  file.read(reinterpret_cast<char*>(payload.data()), payload.size());
  if (!file) throw std::runtime_error("failed to read tensor data");

  tensor.values.resize(count);
  if (tensor.dtype == TensorDType::FP16) tensor.fp16_bits.resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    float value = 0.0F;
    if (tensor.dtype == TensorDType::FP32) {
      const std::uint32_t bits = decode_u32_le(payload.data() + i * 4);
      std::memcpy(&value, &bits, sizeof(value));
      if (!std::isfinite(value)) {
        throw std::invalid_argument("tensor data must not contain FP32 NaN or Inf");
      }
    } else {
      const std::uint16_t bits = decode_u16_le(payload.data() + i * 2);
      tensor.fp16_bits[i] = bits;
      value = decode_fp16(bits);
    }
    tensor.values[i] = value;
  }
  return tensor;
}

void write_tensor_file(const std::string& path, const TensorData& tensor) {
  const std::size_t count = checked_element_count(tensor.num_rows, tensor.num_cols);
  if (tensor.values.size() != count) {
    throw std::invalid_argument("tensor value count does not match rows * cols");
  }
  const std::size_t expected_size = checked_file_size(count, tensor.dtype);
  std::array<std::uint8_t, kHeaderBytes> header{};
  encode_u64_le(static_cast<std::uint64_t>(tensor.num_rows), header.data());
  encode_u64_le(static_cast<std::uint64_t>(tensor.num_cols), header.data() + 8);
  encode_dtype(tensor.dtype, header.data() + 16);

  std::vector<std::uint8_t> payload(expected_size - kHeaderBytes);
  for (std::size_t i = 0; i < count; ++i) {
    const float value = tensor.values[i];
    if (!std::isfinite(value)) {
      throw std::invalid_argument("tensor data must contain finite values");
    }
    if (tensor.dtype == TensorDType::FP32) {
      std::uint32_t bits = 0;
      std::memcpy(&bits, &value, sizeof(bits));
      encode_u32_le(bits, payload.data() + i * 4);
    } else {
      encode_u16_le(encode_fp16(value), payload.data() + i * 2);
    }
  }

  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file) throw std::runtime_error("cannot create tensor file: " + path);
  file.write(reinterpret_cast<const char*>(header.data()), header.size());
  file.write(reinterpret_cast<const char*>(payload.data()), payload.size());
  if (!file) throw std::runtime_error("failed to write tensor file: " + path);
}

}  // namespace luxorion

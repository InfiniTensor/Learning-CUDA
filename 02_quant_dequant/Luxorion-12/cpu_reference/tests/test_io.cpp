// 验证固定20字节header、FP32/FP16行主序数据以及严格错误检查。
#include "io.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
unsigned checks = 0;

void check(bool condition, const std::string& message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}

template <typename Exception, typename Function>
void expect_error(Function function, const std::string& message) {
  bool rejected = false;
  try { function(); } catch (const Exception&) { rejected = true; }
  check(rejected, message);
}

struct TemporaryFile {
  explicit TemporaryFile(std::string name) : path(std::move(name)) {
    std::remove(path.c_str());
  }
  ~TemporaryFile() { std::remove(path.c_str()); }
  std::string path;
};

std::vector<std::uint8_t> read_bytes(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("cannot inspect test file");
  const auto size = static_cast<std::size_t>(file.tellg());
  file.seekg(0);
  std::vector<std::uint8_t> bytes(size);
  file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  return bytes;
}

void write_bytes(const std::string& path, const std::vector<std::uint8_t>& bytes) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

std::array<std::uint8_t, 20> header(std::int64_t rows, std::int64_t cols,
                                    const char* dtype) {
  std::array<std::uint8_t, 20> bytes{};
  const std::uint64_t r = static_cast<std::uint64_t>(rows);
  const std::uint64_t c = static_cast<std::uint64_t>(cols);
  for (unsigned i = 0; i < 8; ++i) {
    bytes[i] = static_cast<std::uint8_t>(r >> (8 * i));
    bytes[8 + i] = static_cast<std::uint8_t>(c >> (8 * i));
  }
  std::memcpy(bytes.data() + 16, dtype, 4);
  return bytes;
}

void test_fp32_round_trip_and_layout() {
  TemporaryFile file("test_tensor_fp32.bin");
  const luxorion::TensorData source{
      2, 3, luxorion::TensorDType::FP32,
      {1.0F, -0.0F, -2.5F, 3.25F, 6000.0F, 1.0e-30F}};
  luxorion::write_tensor_file(file.path, source);
  const auto bytes = read_bytes(file.path);
  check(bytes.size() == 20 + 6 * 4, "FP32 file size");
  check(bytes[0] == 2 && bytes[8] == 3, "little-endian dimensions");
  check(std::memcmp(bytes.data() + 16, "fp32", 4) == 0, "FP32 dtype bytes");

  const auto restored = luxorion::read_tensor_file(file.path);
  check(restored.num_rows == 2 && restored.num_cols == 3, "FP32 shape");
  check(restored.dtype == luxorion::TensorDType::FP32, "FP32 dtype");
  check(restored.values == source.values, "FP32 values are bit-preserving numerically");
  check(std::signbit(restored.values[1]), "FP32 negative zero sign");
}

void test_fp16_round_trip_and_rounding() {
  TemporaryFile file("test_tensor_fp16.bin");
  const float min_subnormal = std::ldexp(1.0F, -24);
  const float midpoint = 1.0F + std::ldexp(1.0F, -11);
  const luxorion::TensorData source{
      2, 4, luxorion::TensorDType::FP16,
      {0.0F, -0.0F, 1.0F, -2.0F, 65504.0F, min_subnormal, 0.5F, midpoint}};
  luxorion::write_tensor_file(file.path, source);
  const auto bytes = read_bytes(file.path);
  check(bytes.size() == 20 + 8 * 2, "FP16 file size");
  check(std::memcmp(bytes.data() + 16, "fp16", 4) == 0, "FP16 dtype bytes");

  const auto restored = luxorion::read_tensor_file(file.path);
  check(restored.dtype == luxorion::TensorDType::FP16, "FP16 dtype");
  check(restored.values.size() == 8, "FP16 element count");
  for (std::size_t i = 0; i < 7; ++i) {
    check(restored.values[i] == source.values[i],
          "exact FP16 value " + std::to_string(i));
  }
  check(std::signbit(restored.values[1]), "FP16 negative zero sign");
  check(restored.values[7] == 1.0F, "FP16 midpoint rounds to even");
}

void test_invalid_files_and_writes() {
  TemporaryFile file("test_tensor_invalid.bin");
  write_bytes(file.path, std::vector<std::uint8_t>(19, 0));
  expect_error<std::invalid_argument>(
      [&] { luxorion::read_tensor_file(file.path); }, "short header");

  auto invalid_dtype = header(1, 1, "bad!");
  write_bytes(file.path, std::vector<std::uint8_t>(
      invalid_dtype.begin(), invalid_dtype.end()));
  expect_error<std::invalid_argument>(
      [&] { luxorion::read_tensor_file(file.path); }, "invalid dtype");

  auto missing_data = header(1, 1, "fp32");
  write_bytes(file.path, std::vector<std::uint8_t>(
      missing_data.begin(), missing_data.end()));
  expect_error<std::invalid_argument>(
      [&] { luxorion::read_tensor_file(file.path); }, "payload size mismatch");

  std::vector<std::uint8_t> infinity(missing_data.begin(), missing_data.end());
  infinity.insert(infinity.end(), {0x00, 0x00, 0x80, 0x7F});
  write_bytes(file.path, infinity);
  expect_error<std::invalid_argument>(
      [&] { luxorion::read_tensor_file(file.path); }, "FP32 infinity");

  auto negative_shape = header(-1, 1, "fp32");
  write_bytes(file.path, std::vector<std::uint8_t>(
      negative_shape.begin(), negative_shape.end()));
  expect_error<std::invalid_argument>(
      [&] { luxorion::read_tensor_file(file.path); }, "negative dimension");

  expect_error<std::invalid_argument>([&] {
    luxorion::write_tensor_file(file.path,
        {2, 2, luxorion::TensorDType::FP32, {1.0F}});
  }, "write value count mismatch");
  expect_error<std::overflow_error>([&] {
    luxorion::write_tensor_file(file.path,
        {1, 1, luxorion::TensorDType::FP16, {70000.0F}});
  }, "write FP16 overflow");
}
}  // namespace

int main() {
  try {
    test_fp32_round_trip_and_layout();
    test_fp16_round_trip_and_rounding();
    test_invalid_files_and_writes();
    std::cout << "Tensor IO: " << checks << " checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Tensor IO test failed: " << error.what() << '\n';
    return 1;
  }
}

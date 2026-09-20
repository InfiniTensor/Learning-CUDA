/*
 * 生成GPU性能baseline使用的固定输入。生成时间不计入性能测量；三个文件均
 * 使用20字节tensor header和FP32行主序data，并在写出后重新读取逐元素校验。
 */
#include "io.h"

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::int64_t kRows = 1024;
constexpr std::int64_t kCols = 1024;
constexpr std::int64_t kTailRows = 1023;
constexpr std::int64_t kTailCols = 1025;
constexpr std::uint32_t kSeed = 1234;
constexpr float kUniformLimit = 6000.0F;
constexpr float kNormalMean = 0.0F;
constexpr float kNormalStddev = 20.0F;
constexpr std::size_t kOutlierInterval = 1000;
constexpr float kOutlierMagnitude = 6000.0F;
constexpr float kNonzeroProbability = 0.1F;

std::string filename(const std::string& distribution) {
  return distribution + "_fp32_1024x1024_seed1234.tensor.bin";
}

void write_and_verify(const std::filesystem::path& path,
                      std::int64_t rows, std::int64_t cols,
                      luxorion::TensorDType dtype,
                      const std::vector<float>& values) {
  const luxorion::TensorData tensor{
      rows, cols, dtype, values};
  luxorion::write_tensor_file(path.string(), tensor);
  const auto restored = luxorion::read_tensor_file(path.string());
  if (restored.num_rows != rows || restored.num_cols != cols
      || restored.dtype != dtype || restored.values.size() != values.size()) {
    throw std::runtime_error("generated tensor verification failed: "
                             + path.string());
  }
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (!std::isfinite(restored.values[i])) {
      throw std::runtime_error("generated tensor contains a non-finite value: "
                               + path.string());
    }
    if (dtype == luxorion::TensorDType::FP32 && restored.values[i] != values[i]) {
      throw std::runtime_error("generated FP32 tensor changed during roundtrip: "
                               + path.string());
    }
  }
}

void write_manifest(const std::filesystem::path& directory) {
  std::ofstream file(directory / "manifest.json", std::ios::trunc);
  if (!file) throw std::runtime_error("cannot create benchmark manifest");
  file << "{\n"
       << "  \"schema_version\": 1,\n"
       << "  \"tensor_header_bytes\": 20,\n"
       << "  \"endianness\": \"little\",\n"
       << "  \"row_major\": true,\n"
       << "  \"block_layout\": \"continuous_flattened\",\n"
       << "  \"seed\": " << kSeed << ",\n"
       << "  \"notes\": \"Synthetic reproducible coverage set; not a substitute for real model weights or activations\",\n"
       << "  \"datasets\": [\n"
       << "    {\"file\": \"" << filename("uniform")
       << "\", \"distribution\": \"uniform\", \"dtype\": \"fp32\", "
          "\"num_rows\": 1024, \"num_cols\": 1024, \"element_count\": 1048576, "
          "\"min\": -6000.0, \"max\": 6000.0},\n"
       << "    {\"file\": \"" << filename("normal")
       << "\", \"distribution\": \"normal\", \"dtype\": \"fp32\", "
          "\"num_rows\": 1024, \"num_cols\": 1024, \"element_count\": 1048576, "
          "\"mean\": 0.0, \"stddev\": 20.0},\n"
       << "    {\"file\": \"" << filename("outliers")
       << "\", \"distribution\": \"normal_with_outliers\", \"dtype\": \"fp32\", "
          "\"num_rows\": 1024, \"num_cols\": 1024, \"element_count\": 1048576, \"mean\": 0.0, "
          "\"stddev\": 20.0, \"outlier_interval\": 1000, "
          "\"outlier_magnitude\": 6000.0},\n"
       << "    {\"file\": \"zero_heavy_fp32_1024x1024_seed1234.tensor.bin\", "
          "\"distribution\": \"zero_heavy_normal\", \"dtype\": \"fp32\", "
          "\"num_rows\": 1024, \"num_cols\": 1024, \"element_count\": 1048576, "
          "\"nonzero_probability\": 0.1, \"nonzero_stddev\": 20.0},\n"
       << "    {\"file\": \"multiscale_fp32_1024x1024_seed1234.tensor.bin\", "
          "\"distribution\": \"block_scaled_normal\", \"dtype\": \"fp32\", "
          "\"num_rows\": 1024, \"num_cols\": 1024, \"element_count\": 1048576, "
          "\"block_size\": 32, \"power_cycle_min\": -8, \"power_cycle_max\": 8},\n"
       << "    {\"file\": \"normal_fp16_1024x1024_seed1234.tensor.bin\", "
          "\"distribution\": \"normal\", \"dtype\": \"fp16\", "
          "\"num_rows\": 1024, \"num_cols\": 1024, \"element_count\": 1048576, "
          "\"mean\": 0.0, \"stddev\": 20.0},\n"
       << "    {\"file\": \"uniform_fp16_1024x1024_seed1234.tensor.bin\", "
          "\"distribution\": \"uniform\", \"dtype\": \"fp16\", "
          "\"num_rows\": 1024, \"num_cols\": 1024, \"element_count\": 1048576, "
          "\"min\": -6000.0, \"max\": 6000.0},\n"
       << "    {\"file\": \"outliers_fp16_1024x1024_seed1234.tensor.bin\", "
          "\"distribution\": \"normal_with_outliers\", \"dtype\": \"fp16\", "
          "\"num_rows\": 1024, \"num_cols\": 1024, \"element_count\": 1048576, "
          "\"mean\": 0.0, \"stddev\": 20.0, \"outlier_interval\": 1000, "
          "\"outlier_magnitude\": 6000.0},\n"
       << "    {\"file\": \"zero_heavy_fp16_1024x1024_seed1234.tensor.bin\", "
          "\"distribution\": \"zero_heavy_normal\", \"dtype\": \"fp16\", "
          "\"num_rows\": 1024, \"num_cols\": 1024, \"element_count\": 1048576, "
          "\"nonzero_probability\": 0.1, \"nonzero_stddev\": 20.0},\n"
       << "    {\"file\": \"multiscale_fp16_1024x1024_seed1234.tensor.bin\", "
          "\"distribution\": \"block_scaled_normal\", \"dtype\": \"fp16\", "
          "\"num_rows\": 1024, \"num_cols\": 1024, \"element_count\": 1048576, "
          "\"block_size\": 32, \"power_cycle_min\": -8, \"power_cycle_max\": 8},\n"
       << "    {\"file\": \"tail_normal_fp32_1023x1025_seed1234.tensor.bin\", "
          "\"distribution\": \"normal\", \"dtype\": \"fp32\", "
          "\"num_rows\": 1023, \"num_cols\": 1025, \"element_count\": 1048575, "
          "\"tail_mod_16\": 15, \"tail_mod_32\": 31},\n"
       << "    {\"file\": \"tail_normal_fp16_1023x1025_seed1234.tensor.bin\", "
          "\"distribution\": \"normal\", \"dtype\": \"fp16\", "
          "\"num_rows\": 1023, \"num_cols\": 1025, \"element_count\": 1048575, "
          "\"tail_mod_16\": 15, \"tail_mod_32\": 31}\n"
       << "  ]\n"
       << "}\n";
  if (!file) throw std::runtime_error("failed to write benchmark manifest");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const std::filesystem::path output = argc > 1
        ? std::filesystem::path(argv[1])
        : std::filesystem::path("input/benchmark");
    std::filesystem::create_directories(output);
    const std::size_t count = static_cast<std::size_t>(kRows * kCols);

    std::mt19937 uniform_engine(kSeed);
    std::uniform_real_distribution<float> uniform_distribution(
        -kUniformLimit, kUniformLimit);
    std::vector<float> uniform(count);
    for (float& value : uniform) value = uniform_distribution(uniform_engine);

    std::mt19937 normal_engine(kSeed);
    std::normal_distribution<float> normal_distribution(
        kNormalMean, kNormalStddev);
    std::vector<float> normal(count);
    for (float& value : normal) value = normal_distribution(normal_engine);

    std::vector<float> outliers = normal;
    for (std::size_t i = 0; i < outliers.size(); i += kOutlierInterval) {
      outliers[i] = (i / kOutlierInterval) % 2 == 0
          ? kOutlierMagnitude : -kOutlierMagnitude;
    }

    std::mt19937 sparse_engine(kSeed + 1);
    std::bernoulli_distribution keep(kNonzeroProbability);
    std::normal_distribution<float> sparse_values(kNormalMean, kNormalStddev);
    std::vector<float> zero_heavy(count, 0.0F);
    for (float& value : zero_heavy) {
      if (keep(sparse_engine)) value = sparse_values(sparse_engine);
    }

    std::vector<float> multiscale = normal;
    for (std::size_t i = 0; i < multiscale.size(); ++i) {
      const int power = static_cast<int>((i / 32) % 17) - 8;
      multiscale[i] = std::ldexp(multiscale[i], power);
    }

    const std::size_t tail_count =
        static_cast<std::size_t>(kTailRows * kTailCols);
    std::vector<float> tail_normal(normal.begin(), normal.begin() + tail_count);

    write_and_verify(output / filename("uniform"), kRows, kCols,
                     luxorion::TensorDType::FP32, uniform);
    write_and_verify(output / filename("normal"), kRows, kCols,
                     luxorion::TensorDType::FP32, normal);
    write_and_verify(output / filename("outliers"), kRows, kCols,
                     luxorion::TensorDType::FP32, outliers);
    write_and_verify(output / "zero_heavy_fp32_1024x1024_seed1234.tensor.bin",
                     kRows, kCols, luxorion::TensorDType::FP32, zero_heavy);
    write_and_verify(output / "multiscale_fp32_1024x1024_seed1234.tensor.bin",
                     kRows, kCols, luxorion::TensorDType::FP32, multiscale);
    write_and_verify(output / "normal_fp16_1024x1024_seed1234.tensor.bin",
                     kRows, kCols, luxorion::TensorDType::FP16, normal);
    write_and_verify(output / "uniform_fp16_1024x1024_seed1234.tensor.bin",
                     kRows, kCols, luxorion::TensorDType::FP16, uniform);
    write_and_verify(output / "outliers_fp16_1024x1024_seed1234.tensor.bin",
                     kRows, kCols, luxorion::TensorDType::FP16, outliers);
    write_and_verify(output / "zero_heavy_fp16_1024x1024_seed1234.tensor.bin",
                     kRows, kCols, luxorion::TensorDType::FP16, zero_heavy);
    write_and_verify(output / "multiscale_fp16_1024x1024_seed1234.tensor.bin",
                     kRows, kCols, luxorion::TensorDType::FP16, multiscale);
    write_and_verify(output / "tail_normal_fp32_1023x1025_seed1234.tensor.bin",
                     kTailRows, kTailCols, luxorion::TensorDType::FP32,
                     tail_normal);
    write_and_verify(output / "tail_normal_fp16_1023x1025_seed1234.tensor.bin",
                     kTailRows, kTailCols, luxorion::TensorDType::FP16,
                     tail_normal);
    write_manifest(output);
    std::cout << "Generated and verified 12 benchmark tensors in "
              << output.string() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Benchmark input generation failed: " << error.what() << '\n';
    return 1;
  }
}

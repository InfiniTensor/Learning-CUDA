#include "low_precision.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* expression, const char* file,
                int line) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) +
                             ": " + expression + " failed: " +
                             cudaGetErrorString(status));
  }
}

#define CUDA_CHECK(expression) \
  check_cuda((expression), #expression, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count_ > 0) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_),
                            count_ * sizeof(T)));
    }
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* data() { return data_; }
  const T* data() const { return data_; }
  std::size_t size() const { return count_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
};

struct MxEncoded {
  std::vector<std::uint8_t> values;
  std::vector<std::uint8_t> scales;
};

struct NvEncoded {
  std::vector<std::uint8_t> packed_values;
  std::vector<std::uint8_t> block_scales;
  float global_scale = 1.0F;
};

struct ErrorMetrics {
  double max_absolute = 0.0;
  double mean_absolute = 0.0;
  double rmse = 0.0;
  double max_relative = 0.0;
  double cosine_similarity = 1.0;
  double sqnr_db = std::numeric_limits<double>::infinity();
  std::size_t samples = 0;
};

float finite_amax(const float* values, std::size_t count) {
  float result = 0.0F;
  for (std::size_t i = 0; i < count; ++i) {
    if (std::isfinite(values[i])) {
      result = std::max(result, std::fabs(values[i]));
    }
  }
  return result;
}

MxEncoded quantize_mxfp8_cpu(const std::vector<float>& input,
                             lpq::MxFp8Kind kind) {
  MxEncoded encoded;
  encoded.values.resize(input.size());
  encoded.scales.resize(lpq::mx_scale_count(input.size()));
  for (std::size_t block = 0; block < encoded.scales.size(); ++block) {
    const std::size_t begin = block * lpq::kMxBlockSize;
    const std::size_t block_count =
        std::min<std::size_t>(lpq::kMxBlockSize, input.size() - begin);
    const std::uint8_t scale_code =
        lpq::choose_mx_scale(finite_amax(input.data() + begin, block_count),
                             kind);
    encoded.scales[block] = scale_code;
    const float scale = lpq::decode_e8m0(scale_code);
    for (std::size_t lane = 0; lane < block_count; ++lane) {
      const float scaled = input[begin + lane] / scale;
      encoded.values[begin + lane] =
          kind == lpq::MxFp8Kind::kE4M3 ? lpq::encode_e4m3(scaled)
                                        : lpq::encode_e5m2(scaled);
    }
  }
  return encoded;
}

std::vector<float> dequantize_mxfp8_cpu(const MxEncoded& encoded,
                                        lpq::MxFp8Kind kind) {
  std::vector<float> output(encoded.values.size());
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float value = kind == lpq::MxFp8Kind::kE4M3
                            ? lpq::decode_e4m3(encoded.values[i])
                            : lpq::decode_e5m2(encoded.values[i]);
    output[i] =
        value * lpq::decode_e8m0(encoded.scales[i / lpq::kMxBlockSize]);
  }
  return output;
}

NvEncoded quantize_nvfp4_cpu(const std::vector<float>& input) {
  NvEncoded encoded;
  encoded.packed_values.assign(lpq::nv_packed_count(input.size()), 0U);
  encoded.block_scales.resize(lpq::nv_scale_count(input.size()));

  const float global_amax = finite_amax(input.data(), input.size());
  encoded.global_scale =
      global_amax > 0.0F
          ? global_amax / (lpq::kE4M3Max * lpq::kE2M1Max)
          : 1.0F;

  for (std::size_t block = 0; block < encoded.block_scales.size(); ++block) {
    const std::size_t begin = block * lpq::kNvBlockSize;
    const std::size_t block_count =
        std::min<std::size_t>(lpq::kNvBlockSize, input.size() - begin);
    const float block_amax =
        finite_amax(input.data() + begin, block_count);
    const float raw_scale =
        block_amax > 0.0F
            ? (block_amax / lpq::kE2M1Max) / encoded.global_scale
            : 1.0F;
    const std::uint8_t scale_code = lpq::encode_e4m3_round_up(raw_scale);
    encoded.block_scales[block] = scale_code;
    const float combined_scale =
        encoded.global_scale * lpq::decode_e4m3(scale_code);

    for (std::size_t lane = 0; lane < block_count; ++lane) {
      const std::size_t index = begin + lane;
      const std::uint8_t code =
          combined_scale > 0.0F
              ? lpq::encode_e2m1(input[index] / combined_scale)
              : 0U;
      if ((index & 1U) == 0) {
        encoded.packed_values[index / 2] = code & 0xFU;
      } else {
        encoded.packed_values[index / 2] |=
            static_cast<std::uint8_t>((code & 0xFU) << 4);
      }
    }
  }
  return encoded;
}

std::vector<float> dequantize_nvfp4_cpu(const NvEncoded& encoded,
                                        std::size_t count) {
  std::vector<float> output(count);
  for (std::size_t i = 0; i < count; ++i) {
    const std::uint8_t packed = encoded.packed_values[i / 2];
    const std::uint8_t code =
        (i & 1U) == 0 ? packed & 0xFU : (packed >> 4) & 0xFU;
    output[i] =
        lpq::decode_e2m1(code) *
        lpq::decode_e4m3(encoded.block_scales[i / lpq::kNvBlockSize]) *
        encoded.global_scale;
  }
  return output;
}

ErrorMetrics calculate_metrics(const std::vector<float>& reference,
                               const std::vector<float>& reconstructed) {
  if (reference.size() != reconstructed.size()) {
    throw std::invalid_argument("metric vectors have different sizes");
  }
  ErrorMetrics metrics;
  double absolute_sum = 0.0;
  double squared_sum = 0.0;
  double signal_energy = 0.0;
  double noise_energy = 0.0;
  double dot = 0.0;
  double reference_norm = 0.0;
  double reconstructed_norm = 0.0;

  for (std::size_t i = 0; i < reference.size(); ++i) {
    if (!std::isfinite(reference[i]) || !std::isfinite(reconstructed[i])) {
      continue;
    }
    const double expected = reference[i];
    const double actual = reconstructed[i];
    const double error = actual - expected;
    const double absolute = std::fabs(error);
    metrics.max_absolute = std::max(metrics.max_absolute, absolute);
    metrics.max_relative =
        std::max(metrics.max_relative,
                 absolute / std::max(std::fabs(expected), 1.0e-12));
    absolute_sum += absolute;
    squared_sum += error * error;
    signal_energy += expected * expected;
    noise_energy += error * error;
    dot += expected * actual;
    reference_norm += expected * expected;
    reconstructed_norm += actual * actual;
    ++metrics.samples;
  }

  if (metrics.samples > 0) {
    metrics.mean_absolute = absolute_sum / metrics.samples;
    metrics.rmse = std::sqrt(squared_sum / metrics.samples);
  }
  if (reference_norm > 0.0 && reconstructed_norm > 0.0) {
    metrics.cosine_similarity =
        dot / std::sqrt(reference_norm * reconstructed_norm);
  }
  if (noise_energy > 0.0) {
    metrics.sqnr_db = 10.0 * std::log10(signal_energy / noise_energy);
  }
  return metrics;
}

std::uint32_t xorshift32(std::uint32_t& state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

std::vector<float> make_input(std::size_t count, bool special_values) {
  std::vector<float> input(count);
  std::uint32_t state = 0xC001D00DU;
  for (std::size_t i = 0; i < count; ++i) {
    const float unit =
        static_cast<float>(xorshift32(state) >> 8) * (1.0F / 16777216.0F);
    float value = (2.0F * unit - 1.0F) * 8.0F;
    if (i % 251 == 0) {
      value *= 32.0F;
    }
    if (i % 1021 == 0) {
      value = 0.0F;
    }
    if (i % 4093 == 0) {
      value *= 0.001F;
    }
    input[i] = value;
  }

  if (special_values) {
    const std::vector<float> specials = {
        0.0F,
        -0.0F,
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
        448.0F,
        -448.0F,
        57344.0F,
        -57344.0F,
        std::numeric_limits<float>::denorm_min(),
        std::numeric_limits<float>::min(),
        0.25F,
        0.75F,
        1.25F,
        1.75F,
        2.5F,
        3.5F,
    };
    for (std::size_t i = 0; i < std::min(count, specials.size()); ++i) {
      input[i] = specials[i];
    }
  }
  return input;
}

bool floats_match(float expected, float actual) {
  if (std::isnan(expected) || std::isnan(actual)) {
    return std::isnan(expected) && std::isnan(actual);
  }
  if (std::isinf(expected) || std::isinf(actual)) {
    return expected == actual;
  }
  if (expected == actual) {
    return true;
  }
  return std::fabs(expected - actual) <=
         std::max(1.0e-7F, std::fabs(expected) * 1.0e-6F);
}

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void test_scalar_formats() {
  const float e2m1_positive[8] = {0.0F, 0.5F, 1.0F, 1.5F,
                                  2.0F, 3.0F, 4.0F, 6.0F};
  for (int code = 0; code < 16; ++code) {
    const float magnitude = e2m1_positive[code & 7];
    const float expected = (code & 8) != 0 ? -magnitude : magnitude;
    const float decoded = lpq::decode_e2m1(static_cast<std::uint8_t>(code));
    require(decoded == expected && std::signbit(decoded) == std::signbit(expected),
            "E2M1 decode table mismatch at code " + std::to_string(code));
    require(lpq::encode_e2m1(decoded) == code,
            "E2M1 round-trip mismatch at code " + std::to_string(code));
  }

  const std::pair<float, std::uint8_t> tie_cases[] = {
      {0.25F, 0U}, {0.75F, 2U}, {1.25F, 2U},
      {1.75F, 4U}, {2.5F, 4U},  {3.5F, 6U},
  };
  for (const auto& [value, expected] : tie_cases) {
    require(lpq::encode_e2m1(value) == expected,
            "E2M1 ties-to-even mismatch for " + std::to_string(value));
  }

  const std::pair<float, std::uint8_t> e4m3_known[] = {
      {0.0F, 0x00U},       {0.001953125F, 0x01U},
      {0.015625F, 0x08U},  {1.0F, 0x38U},
      {1.5F, 0x3CU},       {448.0F, 0x7EU},
  };
  for (const auto& [value, code] : e4m3_known) {
    require(lpq::encode_e4m3(value) == code &&
                lpq::decode_e4m3(code) == value,
            "E4M3 known encoding mismatch for " + std::to_string(value));
  }
  const std::pair<float, std::uint8_t> e5m2_known[] = {
      {0.0F, 0x00U},          {0.0000152587890625F, 0x01U},
      {0.00006103515625F, 0x04U}, {1.0F, 0x3CU},
      {1.5F, 0x3EU},          {57344.0F, 0x7BU},
  };
  for (const auto& [value, code] : e5m2_known) {
    require(lpq::encode_e5m2(value) == code &&
                lpq::decode_e5m2(code) == value,
            "E5M2 known encoding mismatch for " + std::to_string(value));
  }
  require(lpq::encode_e4m3(1.0625F) == 0x38U &&
              lpq::encode_e4m3(1.1875F) == 0x3AU,
          "E4M3 ties-to-even mismatch");
  require(lpq::encode_e5m2(1.125F) == 0x3CU &&
              lpq::encode_e5m2(1.375F) == 0x3EU,
          "E5M2 ties-to-even mismatch");
  require(lpq::decode_e8m0(127U) == 1.0F &&
              lpq::decode_e8m0(128U) == 2.0F,
          "E8M0 known encoding mismatch");

  for (int code = 0; code < 256; ++code) {
    const int magnitude_code = code & 0x7F;
    if (magnitude_code != 0x7F) {
      const float decoded = lpq::decode_e4m3(static_cast<std::uint8_t>(code));
      require(lpq::encode_e4m3(decoded) == code,
              "E4M3 finite round-trip mismatch at code " +
                  std::to_string(code));
    }
    const int exponent = (code >> 2) & 0x1F;
    if (exponent != 0x1F) {
      const float decoded = lpq::decode_e5m2(static_cast<std::uint8_t>(code));
      require(lpq::encode_e5m2(decoded) == code,
              "E5M2 finite round-trip mismatch at code " +
                  std::to_string(code));
    }
  }

  require(lpq::decode_e4m3(0x7EU) == lpq::kE4M3Max,
          "E4M3 maximum mismatch");
  require(lpq::decode_e5m2(0x7BU) == lpq::kE5M2Max,
          "E5M2 maximum mismatch");
  require(lpq::choose_mx_scale(448.0F, lpq::MxFp8Kind::kE4M3) == 127U,
          "MX E4M3 unit scale mismatch");
  require(lpq::choose_mx_scale(896.0F, lpq::MxFp8Kind::kE4M3) == 128U,
          "MX E4M3 doubled scale mismatch");
  require(lpq::choose_mx_scale(57344.0F, lpq::MxFp8Kind::kE5M2) == 127U,
          "MX E5M2 unit scale mismatch");
}

void compare_float_vectors(const std::vector<float>& expected,
                           const std::vector<float>& actual,
                           const std::string& label) {
  require(expected.size() == actual.size(), label + " size mismatch");
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (!floats_match(expected[i], actual[i])) {
      throw std::runtime_error(
          label + " mismatch at index " + std::to_string(i) + ": expected " +
          std::to_string(expected[i]) + ", got " + std::to_string(actual[i]));
    }
  }
}

void compare_byte_vectors(const std::vector<std::uint8_t>& expected,
                          const std::vector<std::uint8_t>& actual,
                          const std::string& label) {
  require(expected.size() == actual.size(), label + " size mismatch");
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] != actual[i]) {
      throw std::runtime_error(
          label + " mismatch at byte " + std::to_string(i) + ": expected " +
          std::to_string(static_cast<unsigned>(expected[i])) + ", got " +
          std::to_string(static_cast<unsigned>(actual[i])));
    }
  }
}

void test_mxfp8_rounding_boundaries(lpq::MxFp8Kind kind) {
  const int maximum_code =
      kind == lpq::MxFp8Kind::kE4M3 ? 0x7E : 0x7B;
  const float maximum_value =
      kind == lpq::MxFp8Kind::kE4M3 ? lpq::kE4M3Max : lpq::kE5M2Max;
  const auto decode = [kind](std::uint8_t code) {
    return kind == lpq::MxFp8Kind::kE4M3 ? lpq::decode_e4m3(code)
                                         : lpq::decode_e5m2(code);
  };

  std::vector<float> boundary_values;
  boundary_values.reserve(static_cast<std::size_t>(maximum_code) * 6);
  for (int code = 0; code < maximum_code; ++code) {
    const float lower = decode(static_cast<std::uint8_t>(code));
    const float upper = decode(static_cast<std::uint8_t>(code + 1));
    const float midpoint = lower + (upper - lower) * 0.5F;
    const float candidates[] = {
        std::nextafter(midpoint, -std::numeric_limits<float>::infinity()),
        midpoint,
        std::nextafter(midpoint, std::numeric_limits<float>::infinity()),
    };
    for (const float candidate : candidates) {
      boundary_values.push_back(candidate);
      boundary_values.push_back(-candidate);
    }
  }

  constexpr std::size_t kCasesPerBlock = lpq::kMxBlockSize - 1;
  const std::size_t block_count =
      (boundary_values.size() + kCasesPerBlock - 1) / kCasesPerBlock;
  std::vector<float> input(block_count * lpq::kMxBlockSize, 0.0F);
  std::size_t case_index = 0;
  for (std::size_t block = 0; block < block_count; ++block) {
    for (std::size_t lane = 0;
         lane < kCasesPerBlock && case_index < boundary_values.size();
         ++lane) {
      input[block * lpq::kMxBlockSize + lane] =
          boundary_values[case_index++];
    }
    input[(block + 1) * lpq::kMxBlockSize - 1] = maximum_value;
  }

  const MxEncoded cpu_encoded = quantize_mxfp8_cpu(input, kind);
  const std::vector<float> cpu_output =
      dequantize_mxfp8_cpu(cpu_encoded, kind);
  DeviceBuffer<float> device_input(input.size());
  DeviceBuffer<std::uint8_t> device_values(input.size());
  DeviceBuffer<std::uint8_t> device_scales(
      lpq::mx_scale_count(input.size()));
  DeviceBuffer<float> device_output(input.size());
  CUDA_CHECK(cudaMemcpy(device_input.data(), input.data(),
                        input.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(lpq::quantize_mxfp8(
      device_input.data(), input.size(), device_values.data(),
      device_scales.data(), kind));
  CUDA_CHECK(lpq::dequantize_mxfp8(
      device_values.data(), device_scales.data(), input.size(),
      device_output.data(), kind));
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<std::uint8_t> gpu_values(input.size());
  std::vector<std::uint8_t> gpu_scales(lpq::mx_scale_count(input.size()));
  std::vector<float> gpu_output(input.size());
  CUDA_CHECK(cudaMemcpy(gpu_values.data(), device_values.data(),
                        gpu_values.size(), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gpu_scales.data(), device_scales.data(),
                        gpu_scales.size(), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gpu_output.data(), device_output.data(),
                        gpu_output.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  const std::string label =
      kind == lpq::MxFp8Kind::kE4M3 ? "MXFP8 E4M3 boundary"
                                     : "MXFP8 E5M2 boundary";
  compare_byte_vectors(cpu_encoded.values, gpu_values,
                       label + " value bytes");
  compare_byte_vectors(cpu_encoded.scales, gpu_scales,
                       label + " scale bytes");
  compare_float_vectors(cpu_output, gpu_output,
                        label + " dequantization");
}

void test_nvfp4_rounding_boundaries() {
  const float midpoints[] = {
      0.25F, 0.75F, 1.25F, 1.75F, 2.5F, 3.5F, 5.0F,
  };
  std::vector<float> boundary_values;
  boundary_values.reserve(42);
  for (const float midpoint : midpoints) {
    const float candidates[] = {
        std::nextafter(midpoint, -std::numeric_limits<float>::infinity()),
        midpoint,
        std::nextafter(midpoint, std::numeric_limits<float>::infinity()),
    };
    for (const float candidate : candidates) {
      boundary_values.push_back(candidate);
      boundary_values.push_back(-candidate);
    }
  }

  constexpr std::size_t kCasesPerBlock = lpq::kNvBlockSize - 1;
  const std::size_t case_blocks =
      (boundary_values.size() + kCasesPerBlock - 1) / kCasesPerBlock;
  std::vector<float> input(
      (case_blocks + 1) * lpq::kNvBlockSize, 0.0F);
  input[0] = lpq::kE4M3Max * lpq::kE2M1Max;
  std::size_t case_index = 0;
  for (std::size_t block = 1; block <= case_blocks; ++block) {
    for (std::size_t lane = 0;
         lane < kCasesPerBlock && case_index < boundary_values.size();
         ++lane) {
      input[block * lpq::kNvBlockSize + lane] =
          boundary_values[case_index++];
    }
    input[(block + 1) * lpq::kNvBlockSize - 1] = lpq::kE2M1Max;
  }

  const NvEncoded cpu_encoded = quantize_nvfp4_cpu(input);
  require(cpu_encoded.global_scale == 1.0F,
          "NVFP4 boundary global scale must be one");
  const std::vector<float> cpu_output =
      dequantize_nvfp4_cpu(cpu_encoded, input.size());
  DeviceBuffer<float> device_input(input.size());
  DeviceBuffer<std::uint8_t> device_packed(
      lpq::nv_packed_count(input.size()));
  DeviceBuffer<std::uint8_t> device_scales(
      lpq::nv_scale_count(input.size()));
  DeviceBuffer<float> device_global_scale(1);
  DeviceBuffer<float> device_output(input.size());
  CUDA_CHECK(cudaMemcpy(device_input.data(), input.data(),
                        input.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(lpq::quantize_nvfp4(
      device_input.data(), input.size(), device_packed.data(),
      device_scales.data(), device_global_scale.data()));
  CUDA_CHECK(lpq::dequantize_nvfp4(
      device_packed.data(), device_scales.data(), device_global_scale.data(),
      input.size(), device_output.data()));
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<std::uint8_t> gpu_packed(
      lpq::nv_packed_count(input.size()));
  std::vector<std::uint8_t> gpu_scales(
      lpq::nv_scale_count(input.size()));
  std::vector<float> gpu_output(input.size());
  float gpu_global_scale = 0.0F;
  CUDA_CHECK(cudaMemcpy(gpu_packed.data(), device_packed.data(),
                        gpu_packed.size(), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gpu_scales.data(), device_scales.data(),
                        gpu_scales.size(), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&gpu_global_scale, device_global_scale.data(),
                        sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gpu_output.data(), device_output.data(),
                        gpu_output.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  compare_byte_vectors(cpu_encoded.packed_values, gpu_packed,
                       "NVFP4 boundary packed bytes");
  compare_byte_vectors(cpu_encoded.block_scales, gpu_scales,
                       "NVFP4 boundary scale bytes");
  require(gpu_global_scale == 1.0F,
          "NVFP4 GPU boundary global scale must be one");
  compare_float_vectors(cpu_output, gpu_output,
                        "NVFP4 boundary dequantization");
}

void test_gpu_for_size(std::size_t count) {
  const std::vector<float> input = make_input(count, true);
  DeviceBuffer<float> device_input(count);
  if (count > 0) {
    CUDA_CHECK(cudaMemcpy(device_input.data(), input.data(),
                          count * sizeof(float), cudaMemcpyHostToDevice));
  }

  for (const lpq::MxFp8Kind kind :
       {lpq::MxFp8Kind::kE4M3, lpq::MxFp8Kind::kE5M2}) {
    const MxEncoded cpu_encoded = quantize_mxfp8_cpu(input, kind);
    const std::vector<float> cpu_output =
        dequantize_mxfp8_cpu(cpu_encoded, kind);

    DeviceBuffer<std::uint8_t> device_values(count);
    DeviceBuffer<std::uint8_t> device_scales(lpq::mx_scale_count(count));
    DeviceBuffer<float> device_output(count);
    CUDA_CHECK(lpq::quantize_mxfp8(
        device_input.data(), count, device_values.data(), device_scales.data(),
        kind));
    CUDA_CHECK(lpq::dequantize_mxfp8(
        device_values.data(), device_scales.data(), count, device_output.data(),
        kind));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<std::uint8_t> gpu_values(count);
    std::vector<std::uint8_t> gpu_scales(lpq::mx_scale_count(count));
    std::vector<float> gpu_output(count);
    if (count > 0) {
      CUDA_CHECK(cudaMemcpy(gpu_values.data(), device_values.data(), count,
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(gpu_scales.data(), device_scales.data(),
                            gpu_scales.size(), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(gpu_output.data(), device_output.data(),
                            count * sizeof(float), cudaMemcpyDeviceToHost));
    }
    const std::string label =
        kind == lpq::MxFp8Kind::kE4M3 ? "MXFP8 E4M3" : "MXFP8 E5M2";
    compare_byte_vectors(cpu_encoded.values, gpu_values,
                         label + " value bytes");
    compare_byte_vectors(cpu_encoded.scales, gpu_scales,
                         label + " scale bytes");
    compare_float_vectors(cpu_output, gpu_output, label + " dequantization");
  }

  const NvEncoded cpu_encoded = quantize_nvfp4_cpu(input);
  const std::vector<float> cpu_output =
      dequantize_nvfp4_cpu(cpu_encoded, count);
  DeviceBuffer<std::uint8_t> device_packed(lpq::nv_packed_count(count));
  DeviceBuffer<std::uint8_t> device_scales(lpq::nv_scale_count(count));
  DeviceBuffer<float> device_global_scale(1);
  DeviceBuffer<float> device_output(count);
  CUDA_CHECK(lpq::quantize_nvfp4(
      device_input.data(), count, device_packed.data(), device_scales.data(),
      device_global_scale.data()));
  CUDA_CHECK(lpq::dequantize_nvfp4(
      device_packed.data(), device_scales.data(), device_global_scale.data(),
      count, device_output.data()));
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<std::uint8_t> gpu_packed(lpq::nv_packed_count(count));
  std::vector<std::uint8_t> gpu_scales(lpq::nv_scale_count(count));
  std::vector<float> gpu_output(count);
  float gpu_global_scale = 0.0F;
  if (!gpu_packed.empty()) {
    CUDA_CHECK(cudaMemcpy(gpu_packed.data(), device_packed.data(),
                          gpu_packed.size(), cudaMemcpyDeviceToHost));
  }
  if (!gpu_scales.empty()) {
    CUDA_CHECK(cudaMemcpy(gpu_scales.data(), device_scales.data(),
                          gpu_scales.size(), cudaMemcpyDeviceToHost));
  }
  if (count > 0) {
    CUDA_CHECK(cudaMemcpy(gpu_output.data(), device_output.data(),
                          count * sizeof(float), cudaMemcpyDeviceToHost));
  }
  CUDA_CHECK(cudaMemcpy(&gpu_global_scale, device_global_scale.data(),
                        sizeof(float), cudaMemcpyDeviceToHost));

  compare_byte_vectors(cpu_encoded.packed_values, gpu_packed,
                       "NVFP4 packed value bytes");
  compare_byte_vectors(cpu_encoded.block_scales, gpu_scales,
                       "NVFP4 block scale bytes");
  require(floats_match(cpu_encoded.global_scale, gpu_global_scale),
          "NVFP4 global scale differs");
  compare_float_vectors(cpu_output, gpu_output, "NVFP4 dequantization");
}

void run_tests() {
  std::cout << "[test] scalar format encoders and decoders\n";
  test_scalar_formats();

  std::cout << "[test] zero-length API behavior\n";
  DeviceBuffer<float> global_scale(1);
  CUDA_CHECK(lpq::quantize_mxfp8(nullptr, 0, nullptr, nullptr,
                                 lpq::MxFp8Kind::kE4M3));
  CUDA_CHECK(lpq::dequantize_mxfp8(nullptr, nullptr, 0, nullptr,
                                   lpq::MxFp8Kind::kE4M3));
  CUDA_CHECK(
      lpq::quantize_nvfp4(nullptr, 0, nullptr, nullptr, global_scale.data()));
  CUDA_CHECK(cudaDeviceSynchronize());
  float zero_length_scale = 0.0F;
  CUDA_CHECK(cudaMemcpy(&zero_length_scale, global_scale.data(), sizeof(float),
                        cudaMemcpyDeviceToHost));
  require(zero_length_scale == 1.0F, "zero-length NVFP4 scale must be one");

  std::cout << "[test] MXFP8 GPU rounding boundaries\n";
  test_mxfp8_rounding_boundaries(lpq::MxFp8Kind::kE4M3);
  test_mxfp8_rounding_boundaries(lpq::MxFp8Kind::kE5M2);
  std::cout << "[test] NVFP4 GPU rounding boundaries\n";
  test_nvfp4_rounding_boundaries();

  const std::size_t sizes[] = {1,  2,  3,  4,  5,  15, 16,
                               17, 31, 32, 33, 257, 1000, 65539};
  for (const std::size_t size : sizes) {
    std::cout << "[test] CPU/GPU byte equivalence, n=" << size << "\n";
    test_gpu_for_size(size);
  }
  std::cout << "[test] all tests passed\n";
}

float time_cuda(const std::function<void()>& operation, int warmup,
                int iterations) {
  for (int i = 0; i < warmup; ++i) {
    operation();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    operation();
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0F;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed_ms / static_cast<float>(iterations);
}

double elapsed_ms(std::chrono::steady_clock::time_point begin,
                  std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

void print_metrics(const ErrorMetrics& metrics) {
  std::cout << std::scientific << std::setprecision(6)
            << "    max_abs=" << metrics.max_absolute
            << "  MAE=" << metrics.mean_absolute
            << "  RMSE=" << metrics.rmse
            << "  max_rel=" << metrics.max_relative << "\n"
            << std::fixed << std::setprecision(6)
            << "    cosine=" << metrics.cosine_similarity
            << "  SQNR=" << metrics.sqnr_db << " dB\n";
}

void print_performance(const std::string& label, std::size_t count,
                       std::size_t encoded_bytes, float gpu_quant_ms,
                       float gpu_dequant_ms, double cpu_quant_ms,
                       double cpu_dequant_ms, const ErrorMetrics& metrics) {
  const double input_bytes = static_cast<double>(count * sizeof(float));
  const double quant_traffic = input_bytes + encoded_bytes;
  const double dequant_traffic = input_bytes + encoded_bytes;
  const double compression =
      input_bytes / static_cast<double>(encoded_bytes);

  std::cout << "\n[" << label << "]\n"
            << std::fixed << std::setprecision(3)
            << "  encoded bytes: " << encoded_bytes
            << "  compression: " << compression << "x\n"
            << "  GPU quantize:   " << gpu_quant_ms << " ms  "
            << quant_traffic / (gpu_quant_ms * 1.0e6) << " GB/s\n"
            << "  GPU dequantize: " << gpu_dequant_ms << " ms  "
            << dequant_traffic / (gpu_dequant_ms * 1.0e6) << " GB/s\n"
            << "  CPU quantize:   " << cpu_quant_ms << " ms  speedup "
            << cpu_quant_ms / gpu_quant_ms << "x\n"
            << "  CPU dequantize: " << cpu_dequant_ms << " ms  speedup "
            << cpu_dequant_ms / gpu_dequant_ms << "x\n";
  print_metrics(metrics);
}

void benchmark_mxfp8(const std::vector<float>& input,
                     const DeviceBuffer<float>& device_input,
                     lpq::MxFp8Kind kind) {
  const std::size_t count = input.size();
  DeviceBuffer<std::uint8_t> values(count);
  DeviceBuffer<std::uint8_t> scales(lpq::mx_scale_count(count));
  DeviceBuffer<float> output(count);

  const auto cpu_quant_begin = std::chrono::steady_clock::now();
  const MxEncoded cpu_encoded = quantize_mxfp8_cpu(input, kind);
  const auto cpu_quant_end = std::chrono::steady_clock::now();
  const std::vector<float> cpu_output =
      dequantize_mxfp8_cpu(cpu_encoded, kind);
  const auto cpu_dequant_end = std::chrono::steady_clock::now();

  const float gpu_quant_ms = time_cuda(
      [&] {
        CUDA_CHECK(lpq::quantize_mxfp8(device_input.data(), count,
                                       values.data(), scales.data(), kind));
      },
      5, 50);
  const float gpu_dequant_ms = time_cuda(
      [&] {
        CUDA_CHECK(lpq::dequantize_mxfp8(values.data(), scales.data(), count,
                                         output.data(), kind));
      },
      5, 50);

  std::vector<float> gpu_output(count);
  CUDA_CHECK(cudaMemcpy(gpu_output.data(), output.data(), count * sizeof(float),
                        cudaMemcpyDeviceToHost));
  compare_float_vectors(cpu_output, gpu_output, "benchmark MXFP8");
  const ErrorMetrics metrics = calculate_metrics(input, gpu_output);
  const std::size_t encoded_bytes =
      count + lpq::mx_scale_count(count);
  print_performance(
      kind == lpq::MxFp8Kind::kE4M3 ? "MXFP8 E4M3" : "MXFP8 E5M2",
      count, encoded_bytes, gpu_quant_ms, gpu_dequant_ms,
      elapsed_ms(cpu_quant_begin, cpu_quant_end),
      elapsed_ms(cpu_quant_end, cpu_dequant_end), metrics);
}

void benchmark_nvfp4(const std::vector<float>& input,
                     const DeviceBuffer<float>& device_input) {
  const std::size_t count = input.size();
  DeviceBuffer<std::uint8_t> packed(lpq::nv_packed_count(count));
  DeviceBuffer<std::uint8_t> scales(lpq::nv_scale_count(count));
  DeviceBuffer<float> global_scale(1);
  DeviceBuffer<float> output(count);

  const auto cpu_quant_begin = std::chrono::steady_clock::now();
  const NvEncoded cpu_encoded = quantize_nvfp4_cpu(input);
  const auto cpu_quant_end = std::chrono::steady_clock::now();
  const std::vector<float> cpu_output =
      dequantize_nvfp4_cpu(cpu_encoded, count);
  const auto cpu_dequant_end = std::chrono::steady_clock::now();

  const float gpu_quant_ms = time_cuda(
      [&] {
        CUDA_CHECK(lpq::quantize_nvfp4(
            device_input.data(), count, packed.data(), scales.data(),
            global_scale.data()));
      },
      5, 50);
  const float gpu_dequant_ms = time_cuda(
      [&] {
        CUDA_CHECK(lpq::dequantize_nvfp4(
            packed.data(), scales.data(), global_scale.data(), count,
            output.data()));
      },
      5, 50);

  std::vector<float> gpu_output(count);
  CUDA_CHECK(cudaMemcpy(gpu_output.data(), output.data(), count * sizeof(float),
                        cudaMemcpyDeviceToHost));
  compare_float_vectors(cpu_output, gpu_output, "benchmark NVFP4");
  const ErrorMetrics metrics = calculate_metrics(input, gpu_output);
  const std::size_t encoded_bytes =
      lpq::nv_packed_count(count) + lpq::nv_scale_count(count) + sizeof(float);
  print_performance("NVFP4", count, encoded_bytes, gpu_quant_ms,
                    gpu_dequant_ms,
                    elapsed_ms(cpu_quant_begin, cpu_quant_end),
                    elapsed_ms(cpu_quant_end, cpu_dequant_end), metrics);
}

void run_benchmarks(std::size_t count) {
  cudaDeviceProp properties{};
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  std::cout << "\n[benchmark]\n"
            << "  device: " << properties.name << "  SM "
            << properties.major << "." << properties.minor << "\n"
            << "  elements: " << count << "  FP32 input: "
            << (count * sizeof(float) / (1024.0 * 1024.0)) << " MiB\n"
            << "  timings exclude host-device transfers\n";

  const std::vector<float> input = make_input(count, false);
  DeviceBuffer<float> device_input(count);
  CUDA_CHECK(cudaMemcpy(device_input.data(), input.data(),
                        count * sizeof(float), cudaMemcpyHostToDevice));

  benchmark_mxfp8(input, device_input, lpq::MxFp8Kind::kE4M3);
  benchmark_mxfp8(input, device_input, lpq::MxFp8Kind::kE5M2);
  benchmark_nvfp4(input, device_input);
}

struct Options {
  bool run_tests = true;
  bool run_benchmarks = true;
  std::size_t benchmark_size = 1U << 24;
};

Options parse_options(int argc, char** argv) {
  Options options;
  bool explicit_mode = false;
  options.run_tests = false;
  options.run_benchmarks = false;

  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--test") {
      options.run_tests = true;
      explicit_mode = true;
    } else if (argument == "--benchmark") {
      options.run_benchmarks = true;
      explicit_mode = true;
    } else if (argument == "--all") {
      options.run_tests = true;
      options.run_benchmarks = true;
      explicit_mode = true;
    } else if (argument == "--size" && i + 1 < argc) {
      options.benchmark_size = std::stoull(argv[++i]);
    } else if (argument == "--help") {
      std::cout << "Usage: quant_dequant [--test] [--benchmark] [--all] "
                   "[--size ELEMENTS]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown or incomplete option: " + argument);
    }
  }
  if (!explicit_mode) {
    options.run_tests = true;
    options.run_benchmarks = true;
  }
  require(options.benchmark_size > 0, "benchmark size must be positive");
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (options.run_tests) {
      run_tests();
    }
    if (options.run_benchmarks) {
      run_benchmarks(options.benchmark_size);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}

#include "low_precision.cuh"

#include "hw_path.cuh"

#ifdef LPQ_WITH_HADAMARD
// 方向三项目的公共 launch API，以相对路径 include，不复制源码。
#include "hadamard.cuh"
#endif

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// CUDA API 默认只返回错误码；CLI 在这里统一转成带调用表达式的异常，便于
// 文件读写、warmup 和计时流程复用同一套错误处理。
#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    const cudaError_t error__ = (call);                                      \
    if (error__ != cudaSuccess)                                              \
      throw std::runtime_error(std::string(#call) + ": " +                  \
                               cudaGetErrorString(error__));                 \
  } while (0)

struct TensorInput {
  // bytes 保留文件中的原始 little-endian payload，reference 用于 CPU 端误差统计。
  std::int64_t rows = 0;
  std::int64_t cols = 0;
  lpq::InputType type = lpq::InputType::kFp32;
  std::vector<std::uint8_t> bytes;
  std::vector<float> reference;
};

struct AppConfig {
  lpq::QuantConfig quant;
  lpq::OutputType output_type = lpq::OutputType::kFp16;
  std::string target_gpu = "unspecified";
#ifdef LPQ_WITH_HW_PATH
  // 硬件/库对照路径：off（默认，行为不变）| cvt（指令级编解码）| gemm（cuBLASLt）。
  std::string hardware_path = "off";
#endif
#ifdef LPQ_WITH_HADAMARD
  // Hadamard 旋转：off（默认，行为不变）| chunk128（量化前对每个 128 列 chunk
  // 做归一化 FWHT 旋转，反量化后再变换一次作为逆旋转）。
  std::string hadamard = "off";
#endif
};

struct Metrics {
  double max_abs_error = 0.0;
  double mae = 0.0;
  double mse = 0.0;
};

struct RunResult {
  // 这些字段分别对应量化文件的 payload、反量化输出和 CUDA event 计时结果。
  std::vector<std::uint8_t> packed;
  std::vector<std::uint8_t> scales;
  std::vector<std::uint8_t> output;
  std::vector<float> output_fp32;
  float global_scale = 1.0f;
  float quant_ms = 0.0f;
  float dequant_ms = 0.0f;
};

#pragma pack(push, 1)
// quantized.lpq 的固定 88-byte little-endian header。payload 顺序是 header、
// packed values、scale bytes；字段顺序必须与 tests/validate_outputs.py 保持一致。
struct QuantFileHeader {
  char magic[8];                 // 固定为 "LPQBIN1"，末尾补零。
  std::uint32_t version;         // 文件格式版本。
  std::uint32_t header_bytes;    // 当前值为 88，便于未来扩展头部。
  std::uint32_t format;          // 1=MXFP8，2=NVFP4。
  std::uint32_t scale_mode;      // 1=tensor，2=block。
  std::uint32_t rounding;        // 1=nearest，2=stochastic。
  std::uint32_t input_type;      // 输入 dtype 编号。
  std::uint64_t rows;            // 原始二维 shape。
  std::uint64_t cols;
  std::uint64_t num_elements;    // rows * cols。
  std::uint32_t block_size;      // 逻辑 scale 分组长度。
  std::uint32_t scale_encoding;  // 1=E8M0，2=E4M3。
  std::uint64_t packed_bytes;    // packed values 的字节数。
  std::uint64_t scale_count;     // scale byte 数量。
  float global_scale;            // NVFP4 第二级 scale，MXFP8 固定为 1。
  std::uint32_t reserved;        // 保留字段，当前写入 0。
};
#pragma pack(pop)

static_assert(sizeof(QuantFileHeader) == 88,
              "quantized file header must have a stable layout");

// 简单的 RAII device buffer，保证异常退出时也会释放 cudaMalloc 得到的内存。
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
    if (bytes > 0) CUDA_CHECK(cudaMalloc(&data_, bytes));
  }
  ~DeviceBuffer() {
    if (data_) cudaFree(data_);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  void* get() { return data_; }

 private:
  void* data_ = nullptr;
  std::size_t bytes_ = 0;
};

// CUDA event 只用于 kernel 区间计时，避免把 H2D/D2H、分配和文件 I/O 算进去。
class Event {
 public:
  Event() { CUDA_CHECK(cudaEventCreate(&event_)); }
  ~Event() { cudaEventDestroy(event_); }
  operator cudaEvent_t() const { return event_; }

 private:
  cudaEvent_t event_{};
};

std::string trim(std::string value) {
  // 输入格式很小，使用轻量字符串解析即可，不额外引入 TOML/二进制格式依赖。
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::string unquote(std::string value) {
  // 配置值允许使用单引号或双引号；去引号后统一按裸字符串处理。
  value = trim(value);
  if (value.size() >= 2 &&
      ((value.front() == '"' && value.back() == '"') ||
       (value.front() == '\'' && value.back() == '\'')))
    return value.substr(1, value.size() - 2);
  return value;
}

std::map<std::string, std::string> parse_key_values(const std::string& text,
                                                     char delimiter) {
  // 同一个解析器同时处理 tensor header 的 key: value 和 TOML 风格的 key=value。
  // 这里只读取项目需要的标量字段，方括号 section 名称直接跳过。
  std::map<std::string, std::string> values;
  std::istringstream stream(text);
  std::string line;
  while (std::getline(stream, line)) {
    const auto comment = line.find('#');
    if (comment != std::string::npos) line.resize(comment);
    line = trim(line);
    if (line.empty() || line.front() == '[') continue;
    const auto split = line.find(delimiter);
    if (split == std::string::npos) continue;
    values[trim(line.substr(0, split))] = unquote(line.substr(split + 1));
  }
  return values;
}

std::vector<std::uint8_t> read_file(const fs::path& path) {
  // 先一次性读入，后续既要解析文本头，也要把二进制 payload 原样送到 GPU。
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open input: " + path.string());
  input.seekg(0, std::ios::end);
  const auto size = input.tellg();
  if (size < 0) throw std::runtime_error("cannot determine file size");
  input.seekg(0);
  std::vector<std::uint8_t> result(static_cast<std::size_t>(size));
  if (!result.empty())
    input.read(reinterpret_cast<char*>(result.data()), result.size());
  if (!input) throw std::runtime_error("failed to read input: " + path.string());
  return result;
}

TensorInput load_tensor(const fs::path& path) {
  // 输入文件是 ASCII header + little-endian 原始数据。这里同时构造 GPU 使用的
  // bytes 和 CPU 误差统计使用的 FP32 reference，避免重复读取文件。
  const auto file = read_file(path);
  const std::string marker = "[data]";
  const auto marker_it = std::search(file.begin(), file.end(), marker.begin(),
                                     marker.end());
  if (marker_it == file.end())
    throw std::runtime_error("tensor file has no [data] marker");
  const auto marker_offset =
      static_cast<std::size_t>(std::distance(file.begin(), marker_it));
  auto payload_offset = marker_offset + marker.size();
  if (payload_offset < file.size() && file[payload_offset] == '\r')
    ++payload_offset;
  if (payload_offset >= file.size() || file[payload_offset] != '\n')
    throw std::runtime_error("[data] must be followed by a newline");
  ++payload_offset;

  const std::string header(reinterpret_cast<const char*>(file.data()),
                           marker_offset);
  const auto values = parse_key_values(header, ':');
  if (!values.count("num_rows") || !values.count("num_cols") ||
      !values.count("dtype"))
    throw std::runtime_error("tensor header requires num_rows, num_cols, dtype");

  TensorInput tensor;
  tensor.rows = std::stoll(values.at("num_rows"));
  tensor.cols = std::stoll(values.at("num_cols"));
  if (tensor.rows <= 0 || tensor.cols <= 0)
    throw std::runtime_error("tensor dimensions must be positive");
  const auto count = static_cast<std::uint64_t>(tensor.rows) *
                     static_cast<std::uint64_t>(tensor.cols);
  if (count > std::numeric_limits<std::size_t>::max())
    throw std::runtime_error("tensor is too large for this host");
  const auto dtype = values.at("dtype");
  if (dtype == "fp32")
    tensor.type = lpq::InputType::kFp32;
  else if (dtype == "fp16")
    tensor.type = lpq::InputType::kFp16;
  else
    throw std::runtime_error("dtype must be fp32 or fp16");
  const std::size_t element_size =
      tensor.type == lpq::InputType::kFp32 ? sizeof(float) : sizeof(__half);
  // header 中的 shape 决定 payload 的精确字节数，防止截断或多余数据被静默接受。
  const std::size_t expected = static_cast<std::size_t>(count) * element_size;
  if (file.size() - payload_offset != expected) {
    std::ostringstream message;
    message << "tensor payload has " << (file.size() - payload_offset)
            << " bytes, expected " << expected;
    throw std::runtime_error(message.str());
  }
  tensor.bytes.assign(file.begin() + payload_offset, file.end());
  tensor.reference.resize(count);
  if (tensor.type == lpq::InputType::kFp32) {
    std::memcpy(tensor.reference.data(), tensor.bytes.data(), expected);
  } else {
    // 文件里的 FP16 先转成 FP32 reference；GPU 仍保留原始 FP16 bytes 作为输入。
    for (std::size_t i = 0; i < count; ++i) {
      __half value;
      std::memcpy(&value, tensor.bytes.data() + i * sizeof(value),
                  sizeof(value));
      tensor.reference[i] = __half2float(value);
    }
  }
  for (const float value : tensor.reference) {
    // NaN/Inf 会让误差指标失去意义，因此在 launch 前明确拒绝。
    if (!std::isfinite(value))
      throw std::runtime_error("input contains NaN or infinity");
  }
  return tensor;
}

AppConfig load_config(const fs::path& path) {
  // 配置解析阶段把字符串枚举转换为公共 API 的强类型，后续 CUDA 路径不再处理
  // 拼写错误或默认值。
  const auto bytes = read_file(path);
  const std::string text(reinterpret_cast<const char*>(bytes.data()),
                         bytes.size());
  const auto values = parse_key_values(text, '=');
  auto required = [&](const std::string& key) -> const std::string& {
    const auto found = values.find(key);
    if (found == values.end())
      throw std::runtime_error("config is missing " + key);
    return found->second;
  };
  AppConfig config;
  const auto format = required("format");
  if (format == "mxfp8")
    config.quant.format = lpq::Format::kMxfp8;
  else if (format == "nvfp4")
    config.quant.format = lpq::Format::kNvfp4;
  else
    throw std::runtime_error("format must be mxfp8 or nvfp4");
  config.quant.block_size = values.count("block_size")
                                ? std::stoi(values.at("block_size"))
                                : (config.quant.format == lpq::Format::kMxfp8
                                       ? 32
                                       : 16);
  if (config.quant.block_size <= 0 || config.quant.block_size > 65536)
    throw std::runtime_error("block_size must be in [1, 65536]");

  const auto scale_mode = required("scale_mode");
  if (scale_mode == "block")
    config.quant.scale_mode = lpq::ScaleMode::kBlock;
  else if (scale_mode == "tensor")
    config.quant.scale_mode = lpq::ScaleMode::kTensor;
  else
    throw std::runtime_error("scale_mode must be block or tensor");

  const auto output_type = required("output_type");
  if (output_type == "fp16")
    config.output_type = lpq::OutputType::kFp16;
  else if (output_type == "bf16")
    config.output_type = lpq::OutputType::kBf16;
  else if (output_type == "fp32")
    config.output_type = lpq::OutputType::kFp32;
  else
    throw std::runtime_error("output_type must be fp16, bf16, or fp32");

  const auto rounding = required("rounding");
  if (rounding == "nearest")
    config.quant.rounding = lpq::Rounding::kNearest;
  else if (rounding == "stochastic")
    config.quant.rounding = lpq::Rounding::kStochastic;
  else
    throw std::runtime_error("rounding must be nearest or stochastic");
  if (values.count("seed"))
    config.quant.stochastic_seed = std::stoull(values.at("seed"));
  if (values.count("target_gpu")) config.target_gpu = values.at("target_gpu");
#ifdef LPQ_WITH_HW_PATH
  if (values.count("hardware_path")) {
    config.hardware_path = values.at("hardware_path");
    if (config.hardware_path != "off" && config.hardware_path != "cvt" &&
        config.hardware_path != "gemm")
      throw std::runtime_error("hardware_path must be off, cvt, or gemm");
  }
#endif
#ifdef LPQ_WITH_HADAMARD
  if (values.count("hadamard")) {
    config.hadamard = values.at("hadamard");
    if (config.hadamard != "off" && config.hadamard != "chunk128")
      throw std::runtime_error("hadamard must be off or chunk128");
  }
  if (config.hadamard == "chunk128" &&
      config.output_type != lpq::OutputType::kFp16)
    throw std::runtime_error("hadamard=chunk128 要求 output_type=fp16");
#else
  // 默认构建不识别该配置项时静默忽略会让用户误以为旋转已生效，因此显式报错。
  if (values.count("hadamard") && values.at("hadamard") != "off")
    throw std::runtime_error("hadamard=chunk128 需要 WITH_HADAMARD=1 构建");
#endif
  return config;
}

std::vector<float> output_to_float(const std::vector<std::uint8_t>& output,
                                   lpq::OutputType type, std::size_t count) {
  // 统一转为 FP32 计算 MAE/MSE；输出文件本身仍保留用户指定的 FP16/BF16/FP32
  // 存储格式。
  std::vector<float> result(count);
  if (type == lpq::OutputType::kFp32) {
    std::memcpy(result.data(), output.data(), count * sizeof(float));
  } else if (type == lpq::OutputType::kFp16) {
    for (std::size_t i = 0; i < count; ++i) {
      __half value;
      std::memcpy(&value, output.data() + i * sizeof(value), sizeof(value));
      result[i] = __half2float(value);
    }
  } else {
    // BF16 只需把 16-bit sign/exponent/mantissa 高位放回 FP32 的高 16 bit。
    for (std::size_t i = 0; i < count; ++i) {
      std::uint16_t bits;
      std::memcpy(&bits, output.data() + i * sizeof(bits), sizeof(bits));
      const std::uint32_t fp32_bits = static_cast<std::uint32_t>(bits) << 16;
      std::memcpy(&result[i], &fp32_bits, sizeof(fp32_bits));
    }
  }
  return result;
}

#ifdef LPQ_WITH_HADAMARD
RunResult run_cuda_hadamard(const TensorInput& tensor, const AppConfig& config,
                            int warmup, int iterations) {
  // hadamard=chunk128 的旋转路径。语义：量化前把每个 128 列 chunk 看作一行，
  // 用方向三的 launch_fast 做归一化 FWHT 旋转；反量化到 FP16 后再做一次同样
  // 的变换（归一化 Hadamard 正交且对称，逆变换 = 再变换一次）。中间表示固定
  // 为 FP16（launch API 只支持 FP16/BF16），FP32 输入先在 host 侧舍入。
  if (tensor.cols % 128 != 0)
    throw std::runtime_error("hadamard=chunk128 要求列数能被 128 整除");
  const std::size_t count = tensor.reference.size();
  const std::size_t chunk_rows =
      static_cast<std::size_t>(tensor.rows) *
      static_cast<std::size_t>(tensor.cols / 128);
  std::vector<std::uint8_t> input_bytes;
  if (tensor.type == lpq::InputType::kFp16) {
    input_bytes = tensor.bytes;
  } else {
    input_bytes.resize(count * sizeof(__half));
    for (std::size_t i = 0; i < count; ++i) {
      const __half value = __float2half(tensor.reference[i]);
      std::memcpy(input_bytes.data() + i * sizeof(value), &value,
                  sizeof(value));
    }
  }
  const std::size_t packed_size =
      lpq::packed_bytes(count, config.quant.format);
  const std::size_t scales_size = lpq::scale_count(count, config.quant);
  const std::size_t fp16_size = count * sizeof(__half);
  DeviceBuffer d_input(input_bytes.size());
  DeviceBuffer d_rotated(fp16_size);
  DeviceBuffer d_packed(packed_size);
  DeviceBuffer d_scales(scales_size);
  DeviceBuffer d_global(sizeof(float));
  DeviceBuffer d_scratch(sizeof(float));
  DeviceBuffer d_dequant(fp16_size);
  DeviceBuffer d_output(fp16_size);
  CUDA_CHECK(cudaMemcpy(d_input.get(), input_bytes.data(), input_bytes.size(),
                        cudaMemcpyHostToDevice));

  auto quantize = [&] {
    // 旋转计入量化阶段计时；sign_seed=0 表示纯 FWHT，不启用随机符号。
    CUDA_CHECK(hadamard::launch_fast(
        d_input.get(), d_rotated.get(), chunk_rows, 128,
        hadamard::DataType::kFloat16, 0, nullptr));
    CUDA_CHECK(lpq::launch_quantize(
        d_rotated.get(), lpq::InputType::kFp16,
        static_cast<std::uint8_t*>(d_packed.get()),
        static_cast<std::uint8_t*>(d_scales.get()),
        static_cast<float*>(d_global.get()),
        static_cast<float*>(d_scratch.get()), count, config.quant, nullptr));
  };
  auto dequantize = [&] {
    CUDA_CHECK(lpq::launch_dequantize(
        static_cast<const std::uint8_t*>(d_packed.get()),
        static_cast<const std::uint8_t*>(d_scales.get()),
        static_cast<const float*>(d_global.get()), d_dequant.get(),
        lpq::OutputType::kFp16, count, config.quant, nullptr));
    // 逆旋转计入反量化阶段计时。
    CUDA_CHECK(hadamard::launch_fast(
        d_dequant.get(), d_output.get(), chunk_rows, 128,
        hadamard::DataType::kFloat16, 0, nullptr));
  };
  for (int i = 0; i < warmup; ++i) {
    quantize();
    dequantize();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // 与 run_cuda 相同的 event 计时口径：warmup 后分别统计量化与反量化阶段。
  Event start;
  Event stop;
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) quantize();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float quant_total = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&quant_total, start, stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) dequantize();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float dequant_total = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&dequant_total, start, stop));

  RunResult result;
  result.packed.resize(packed_size);
  result.scales.resize(scales_size);
  result.output.resize(fp16_size);
  CUDA_CHECK(cudaMemcpy(result.packed.data(), d_packed.get(), packed_size,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(result.scales.data(), d_scales.get(), scales_size,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(result.output.data(), d_output.get(), fp16_size,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&result.global_scale, d_global.get(), sizeof(float),
                        cudaMemcpyDeviceToHost));
  result.output_fp32 =
      output_to_float(result.output, lpq::OutputType::kFp16, count);
  result.quant_ms = quant_total / iterations;
  result.dequant_ms = dequant_total / iterations;
  return result;
}
#endif  // LPQ_WITH_HADAMARD

RunResult run_cuda(const TensorInput& tensor, const AppConfig& config,
                   int warmup, int iterations) {
  // 一次 run 负责完整的 device 生命周期：分配缓冲区、预热、分别计时量化与
  // 反量化，最后拷回 payload 和结果。所有公共 launch 都使用默认 stream。
  if (warmup < 0 || iterations <= 0)
    throw std::runtime_error("warmup must be >= 0 and iterations must be > 0");
#ifdef LPQ_WITH_HADAMARD
  // chunk128 走旋转融合路径；off 时完全走原有软件路径。
  if (config.hadamard == "chunk128")
    return run_cuda_hadamard(tensor, config, warmup, iterations);
#endif
  const std::size_t count = tensor.reference.size();
  const std::size_t packed_size =
      lpq::packed_bytes(count, config.quant.format);
  const std::size_t scales_size = lpq::scale_count(count, config.quant);
  const std::size_t output_size =
      count * lpq::output_type_size(config.output_type);
  DeviceBuffer d_input(tensor.bytes.size());
  DeviceBuffer d_packed(packed_size);
  DeviceBuffer d_scales(scales_size);
  DeviceBuffer d_global(sizeof(float));
  DeviceBuffer d_scratch(sizeof(float));
  DeviceBuffer d_output(output_size);
  CUDA_CHECK(cudaMemcpy(d_input.get(), tensor.bytes.data(), tensor.bytes.size(),
                        cudaMemcpyHostToDevice));

  auto quantize = [&] {
    // lambda 让 warmup 和正式计时使用完全相同的 kernel 序列。
    CUDA_CHECK(lpq::launch_quantize(
        d_input.get(), tensor.type,
        static_cast<std::uint8_t*>(d_packed.get()),
        static_cast<std::uint8_t*>(d_scales.get()),
        static_cast<float*>(d_global.get()), static_cast<float*>(d_scratch.get()),
        count, config.quant, nullptr));
  };
  auto dequantize = [&] {
    CUDA_CHECK(lpq::launch_dequantize(
        static_cast<const std::uint8_t*>(d_packed.get()),
        static_cast<const std::uint8_t*>(d_scales.get()),
        static_cast<const float*>(d_global.get()), d_output.get(),
        config.output_type, count, config.quant, nullptr));
  };
  for (int i = 0; i < warmup; ++i) {
    quantize();
    dequantize();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // event 记录的是 GPU 时间；每次结果除以 iterations，得到单次平均 kernel 时间。
  Event start;
  Event stop;
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) quantize();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float quant_total = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&quant_total, start, stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) dequantize();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float dequant_total = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&dequant_total, start, stop));

  RunResult result;
  result.packed.resize(packed_size);
  result.scales.resize(scales_size);
  result.output.resize(output_size);
  CUDA_CHECK(cudaMemcpy(result.packed.data(), d_packed.get(), packed_size,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(result.scales.data(), d_scales.get(), scales_size,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(result.output.data(), d_output.get(), output_size,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&result.global_scale, d_global.get(), sizeof(float),
                        cudaMemcpyDeviceToHost));
  result.output_fp32 =
      output_to_float(result.output, config.output_type, count);
  result.quant_ms = quant_total / iterations;
  result.dequant_ms = dequant_total / iterations;
  return result;
}

#ifdef LPQ_WITH_HW_PATH
// 以下两个函数只在 WITH_HW_PATH 构建中存在：它们在正常软件路径完成之后追加
// 运行，把对照结果写入输出目录的 hw_path.json / hw_gemm.json，不改变软件路径
// 的任何既有输出。

// 这些工具函数定义在文件后部，这里提前声明。
std::string json_escape(const std::string& value);
std::string format_name(lpq::Format format);

struct HwCvtResult {
  std::size_t encode_mismatches = 0;
  std::size_t decode_mismatches = 0;
  float sw_encode_ms = 0.0f;
  float hw_encode_ms = 0.0f;
  float sw_decode_ms = 0.0f;
  float hw_decode_ms = 0.0f;
};

// MXFP8 元素级对照：软件二分查找编码 vs SM 原生 cvt 指令编码（以及对应解码）。
// scale 仍由软件路径算出（E8M0 没有对应的硬件指令），两条路径共享同一份 scale，
// 因此比较的纯粹是元素编解码 kernel 本身。
HwCvtResult run_hw_cvt(const TensorInput& tensor, const AppConfig& config,
                       int warmup, int iterations) {
  if (config.quant.format != lpq::Format::kMxfp8)
    throw std::runtime_error("hardware_path=cvt 只支持 mxfp8 格式");
  if (tensor.type != lpq::InputType::kFp32)
    throw std::runtime_error("hardware_path=cvt 需要 FP32 输入");
  const std::size_t count = tensor.reference.size();
  const std::size_t packed_size =
      lpq::packed_bytes(count, config.quant.format);
  const std::size_t scales_size = lpq::scale_count(count, config.quant);
  DeviceBuffer d_input(tensor.bytes.size());
  DeviceBuffer d_packed_sw(packed_size);
  DeviceBuffer d_packed_hw(packed_size);
  DeviceBuffer d_scales(scales_size);
  DeviceBuffer d_global(sizeof(float));
  DeviceBuffer d_scratch(sizeof(float));
  DeviceBuffer d_out_sw(count * sizeof(float));
  DeviceBuffer d_out_hw(count * sizeof(float));
  CUDA_CHECK(cudaMemcpy(d_input.get(), tensor.bytes.data(), tensor.bytes.size(),
                        cudaMemcpyHostToDevice));
  // 先用完整软件量化得到 scales 和参考 packed/dequant 输出。
  CUDA_CHECK(lpq::launch_quantize(
      d_input.get(), tensor.type,
      static_cast<std::uint8_t*>(d_packed_sw.get()),
      static_cast<std::uint8_t*>(d_scales.get()),
      static_cast<float*>(d_global.get()), static_cast<float*>(d_scratch.get()),
      count, config.quant, nullptr));
  CUDA_CHECK(lpq::launch_dequantize(
      static_cast<const std::uint8_t*>(d_packed_sw.get()),
      static_cast<const std::uint8_t*>(d_scales.get()),
      static_cast<const float*>(d_global.get()), d_out_sw.get(),
      lpq::OutputType::kFp32, count, config.quant, nullptr));
  const bool tensor_mode = config.quant.scale_mode == lpq::ScaleMode::kTensor;
  CUDA_CHECK(lpq::hw::launch_mx_encode_cvt(
      static_cast<const float*>(d_input.get()),
      static_cast<const std::uint8_t*>(d_scales.get()),
      static_cast<std::uint8_t*>(d_packed_hw.get()), count,
      config.quant.block_size, tensor_mode, nullptr));
  CUDA_CHECK(lpq::hw::launch_mx_decode_cvt(
      static_cast<const std::uint8_t*>(d_packed_hw.get()),
      static_cast<const std::uint8_t*>(d_scales.get()),
      static_cast<float*>(d_out_hw.get()), count, config.quant.block_size,
      tensor_mode, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());

  HwCvtResult result;
  {
    // 逐字节比对编码输出；解码输出按 FP32 位级比对（两条路径都应精确相等）。
    std::vector<std::uint8_t> sw(packed_size), hw(packed_size);
    CUDA_CHECK(cudaMemcpy(sw.data(), d_packed_sw.get(), packed_size,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hw.data(), d_packed_hw.get(), packed_size,
                          cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < packed_size; ++i)
      if (sw[i] != hw[i]) ++result.encode_mismatches;
    std::vector<float> out_sw(count), out_hw(count);
    CUDA_CHECK(cudaMemcpy(out_sw.data(), d_out_sw.get(), count * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out_hw.data(), d_out_hw.get(), count * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < count; ++i) {
      std::uint32_t a, b;
      std::memcpy(&a, &out_sw[i], sizeof(a));
      std::memcpy(&b, &out_hw[i], sizeof(b));
      if (a != b) ++result.decode_mismatches;
    }
  }

  auto time_kernel = [&](auto&& fn) {
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    Event start;
    Event stop;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total, start, stop));
    return total / iterations;
  };
  const auto* input_ptr = static_cast<const float*>(d_input.get());
  const auto* scales_ptr = static_cast<const std::uint8_t*>(d_scales.get());
  auto* packed_hw_ptr = static_cast<std::uint8_t*>(d_packed_hw.get());
  auto* packed_sw_ptr = static_cast<std::uint8_t*>(d_packed_sw.get());
  result.sw_encode_ms = time_kernel([&] {
    CUDA_CHECK(lpq::launch_mx_element_quantize(
        input_ptr, packed_sw_ptr, scales_ptr, count, config.quant.block_size,
        tensor_mode, nullptr));
  });
  result.hw_encode_ms = time_kernel([&] {
    CUDA_CHECK(lpq::hw::launch_mx_encode_cvt(input_ptr, scales_ptr,
                                             packed_hw_ptr, count,
                                             config.quant.block_size,
                                             tensor_mode, nullptr));
  });
  auto* out_sw_ptr = static_cast<float*>(d_out_sw.get());
  auto* out_hw_ptr = static_cast<float*>(d_out_hw.get());
  result.sw_decode_ms = time_kernel([&] {
    CUDA_CHECK(lpq::launch_dequantize(
        static_cast<const std::uint8_t*>(d_packed_sw.get()), scales_ptr,
        static_cast<const float*>(d_global.get()), out_sw_ptr,
        lpq::OutputType::kFp32, count, config.quant, nullptr));
  });
  result.hw_decode_ms = time_kernel([&] {
    CUDA_CHECK(lpq::hw::launch_mx_decode_cvt(
        static_cast<const std::uint8_t*>(d_packed_hw.get()), scales_ptr,
        out_hw_ptr, count, config.quant.block_size, tensor_mode, nullptr));
  });
  return result;
}

void save_hw_cvt_log(const fs::path& path, const TensorInput& tensor,
                     const AppConfig& config, const HwCvtResult& result,
                     int warmup, int iterations, const cudaDeviceProp& device) {
  // 与软件路径 metrics.json 相同的有效带宽口径：按最少输入/输出字节估算。
  const double count = static_cast<double>(tensor.reference.size());
  const double traffic =
      count * (sizeof(float) + 1.0) + lpq::scale_count(count, config.quant);
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot create " + path.string());
  output << std::setprecision(10)
         << "{\n"
         << "  \"mode\": \"cvt\",\n"
         << "  \"format\": \"mxfp8\",\n"
         << "  \"shape\": [" << tensor.rows << ", " << tensor.cols << "],\n"
         << "  \"actual_gpu\": \"" << json_escape(device.name) << "\",\n"
         << "  \"dependency\": \"cuda_fp8.h cvt 指令（sm_89+）\",\n"
         << "  \"encode_byte_mismatches\": " << result.encode_mismatches
         << ",\n"
         << "  \"decode_bit_mismatches\": " << result.decode_mismatches
         << ",\n"
         << "  \"warmup\": " << warmup << ",\n"
         << "  \"iterations\": " << iterations << ",\n"
         << "  \"sw_encode_kernel_ms\": " << result.sw_encode_ms << ",\n"
         << "  \"hw_encode_kernel_ms\": " << result.hw_encode_ms << ",\n"
         << "  \"sw_decode_kernel_ms\": " << result.sw_decode_ms << ",\n"
         << "  \"hw_decode_kernel_ms\": " << result.hw_decode_ms << ",\n"
         << "  \"sw_encode_effective_gbps\": "
         << traffic / (result.sw_encode_ms * 1.0e6) << ",\n"
         << "  \"hw_encode_effective_gbps\": "
         << traffic / (result.hw_encode_ms * 1.0e6) << ",\n"
         << "  \"sw_decode_effective_gbps\": "
         << traffic / (result.sw_decode_ms * 1.0e6) << ",\n"
         << "  \"hw_decode_effective_gbps\": "
         << traffic / (result.hw_decode_ms * 1.0e6) << ",\n"
         << "  \"encode_speedup\": "
         << result.sw_encode_ms / result.hw_encode_ms << ",\n"
         << "  \"decode_speedup\": "
         << result.sw_decode_ms / result.hw_decode_ms << "\n"
         << "}\n";
}

#ifdef LPQ_HW_CUBLAS
void save_hw_gemm_log(const fs::path& path, const TensorInput& tensor,
                      const AppConfig& config,
                      const lpq::hw::GemmBenchResult& result, int warmup,
                      int iterations, const cudaDeviceProp& device) {
  const double sw_total = result.sw_dequant_ms + result.sw_gemm_ms;
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot create " + path.string());
  output << std::setprecision(10)
         << "{\n"
         << "  \"mode\": \"gemm\",\n"
         << "  \"format\": \"" << format_name(config.quant.format) << "\",\n"
         << "  \"shape\": [" << tensor.rows << ", " << tensor.cols << "],\n"
         << "  \"actual_gpu\": \"" << json_escape(device.name) << "\",\n"
         << "  \"dependency\": \"cuBLASLt（第三方库）+ 低精度 Tensor Core\",\n"
         << "  \"hw_supported\": " << (result.hw_supported ? "true" : "false")
         << ",\n"
         << "  \"hw_impl\": \"" << json_escape(result.hw_impl) << "\",\n"
         << "  \"warmup\": " << warmup << ",\n"
         << "  \"iterations\": " << iterations << ",\n"
         << "  \"hw_gemm_ms\": " << result.hw_gemm_ms << ",\n"
         << "  \"sw_dequant_ms\": " << result.sw_dequant_ms << ",\n"
         << "  \"sw_gemm_ms\": " << result.sw_gemm_ms << ",\n"
         << "  \"sw_pipeline_ms\": " << sw_total << ",\n"
         << "  \"hw_tflops\": " << result.hw_tflops << ",\n"
         << "  \"sw_tflops\": " << result.sw_tflops << ",\n"
         << "  \"gemm_speedup\": "
         << (result.sw_gemm_ms > 0 ? result.sw_gemm_ms / result.hw_gemm_ms
                                   : 0.0)
         << ",\n"
         << "  \"pipeline_speedup\": "
         << (result.hw_gemm_ms > 0 ? sw_total / result.hw_gemm_ms : 0.0)
         << ",\n"
         << "  \"self_check_hw_rel_error\": " << result.self_check_hw_rel
         << ",\n"
         << "  \"self_check_sw_rel_error\": " << result.self_check_sw_rel
         << ",\n"
         << "  \"bench_hw_vs_sw_max_abs\": " << result.bench_hw_vs_sw_max_abs
         << ",\n"
         << "  \"bench_hw_vs_sw_max_rel\": " << result.bench_hw_vs_sw_max_rel
         << "\n"
         << "}\n";
}
#endif  // LPQ_HW_CUBLAS
#endif  // LPQ_WITH_HW_PATH

Metrics compute_metrics(const std::vector<float>& reference,
                        const std::vector<float>& actual) {  // 用 long double 累加，减少大张量上 MAE/MSE 的 CPU 累加误差；输出指标仍为
  // JSON 可读的 double。
  if (reference.size() != actual.size() || reference.empty())
    throw std::runtime_error("invalid arrays for error metrics");
  Metrics metrics;
  long double absolute_sum = 0.0;
  long double squared_sum = 0.0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double error = static_cast<double>(actual[i]) - reference[i];
    const double absolute = std::abs(error);
    metrics.max_abs_error = std::max(metrics.max_abs_error, absolute);
    absolute_sum += absolute;
    squared_sum += error * error;
  }
  metrics.mae = static_cast<double>(absolute_sum / reference.size());
  metrics.mse = static_cast<double>(squared_sum / reference.size());
  return metrics;
}

std::string format_name(lpq::Format format) {
  return format == lpq::Format::kMxfp8 ? "mxfp8" : "nvfp4";
}
std::string input_name(lpq::InputType type) {
  return type == lpq::InputType::kFp32 ? "fp32" : "fp16";
}
std::string output_name(lpq::OutputType type) {
  if (type == lpq::OutputType::kFp32) return "fp32";
  return type == lpq::OutputType::kFp16 ? "fp16" : "bf16";
}
std::string scale_mode_name(lpq::ScaleMode mode) {
  return mode == lpq::ScaleMode::kBlock ? "block" : "tensor";
}
std::string rounding_name(lpq::Rounding rounding) {
  return rounding == lpq::Rounding::kNearest ? "nearest" : "stochastic";
}

void write_binary(const fs::path& path, const void* data, std::size_t bytes) {
  // dequantized.bin 不带 header，调用方需根据 metrics.json 中的 output_type 解释。
  std::ofstream output(path, std::ios::binary);
  if (!output) throw std::runtime_error("cannot create " + path.string());
  output.write(static_cast<const char*>(data), bytes);
  if (!output) throw std::runtime_error("cannot write " + path.string());
}

void save_quantized(const fs::path& path, const TensorInput& tensor,
                    const AppConfig& config, const RunResult& result) {
  // 文件采用当前主机的 little-endian 布局，与测试脚本约定的 88-byte struct 对齐。
  // 写入时不包含 header 之外的 padding，便于按 packed_bytes/scale_count 定位。
  QuantFileHeader header{};
  std::memcpy(header.magic, "LPQBIN1", 7);
  header.version = 1;
  header.header_bytes = sizeof(header);
  header.format = static_cast<std::uint32_t>(config.quant.format);
  header.scale_mode = static_cast<std::uint32_t>(config.quant.scale_mode);
  header.rounding = static_cast<std::uint32_t>(config.quant.rounding);
  header.input_type = static_cast<std::uint32_t>(tensor.type);
  header.rows = tensor.rows;
  header.cols = tensor.cols;
  header.num_elements = tensor.reference.size();
  header.block_size = config.quant.block_size;
  header.scale_encoding =
      config.quant.format == lpq::Format::kMxfp8 ? 1 : 2;
  header.packed_bytes = result.packed.size();
  header.scale_count = result.scales.size();
  header.global_scale = result.global_scale;
  std::ofstream output(path, std::ios::binary);
  if (!output) throw std::runtime_error("cannot create " + path.string());
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));
  output.write(reinterpret_cast<const char*>(result.packed.data()),
               result.packed.size());
  output.write(reinterpret_cast<const char*>(result.scales.data()),
               result.scales.size());
  if (!output) throw std::runtime_error("cannot write " + path.string());
}

std::string json_escape(const std::string& value) {
  // metrics.json 当前只需要转义字符串中的双引号和反斜杠。
  std::string result;
  for (const char c : value) {
    if (c == '"' || c == '\\') result.push_back('\\');
    result.push_back(c);
  }
  return result;
}

void save_log(const fs::path& path, const TensorInput& tensor,
              const AppConfig& config, const RunResult& result,
              const Metrics& metrics, int warmup, int iterations,
              const cudaDeviceProp& device) {
  // 有效带宽按算法需要搬运的最少字节数估算，不等同于 profiler 报告的 DRAM 带宽。
  // NVFP4 的 global_scale 是额外的 4-byte payload，MXFP8 的 global_scale=1 不计入。
  const double input_bytes = tensor.bytes.size();
  const double payload_bytes = result.packed.size() + result.scales.size() +
                               (config.quant.format == lpq::Format::kNvfp4
                                    ? sizeof(float)
                                    : 0);
  const double quant_traffic = input_bytes + payload_bytes;
  const double dequant_traffic =
      payload_bytes + static_cast<double>(result.output.size());
  const double quant_gbps = quant_traffic / (result.quant_ms * 1.0e6);
  const double dequant_gbps = dequant_traffic / (result.dequant_ms * 1.0e6);
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot create " + path.string());
  output << std::setprecision(10)
         << "{\n"
         << "  \"format\": \"" << format_name(config.quant.format) << "\",\n"
         << "  \"shape\": [" << tensor.rows << ", " << tensor.cols << "],\n"
         << "  \"input_type\": \"" << input_name(tensor.type) << "\",\n"
         << "  \"output_type\": \"" << output_name(config.output_type)
         << "\",\n"
         << "  \"scale_mode\": \"" << scale_mode_name(config.quant.scale_mode)
         << "\",\n"
         << "  \"block_size\": " << config.quant.block_size << ",\n"
         << "  \"rounding\": \"" << rounding_name(config.quant.rounding)
         << "\",\n"
         << "  \"target_gpu\": \"" << json_escape(config.target_gpu) << "\",\n"
         << "  \"actual_gpu\": \"" << json_escape(device.name) << "\",\n"
         << "  \"software_simulation\": true,\n"
#ifdef LPQ_WITH_HADAMARD
         << "  \"hadamard\": \"" << json_escape(config.hadamard) << "\",\n"
#endif
         << "  \"packed_bytes\": " << result.packed.size() << ",\n"
         << "  \"scale_bytes\": " << result.scales.size() << ",\n"
         << "  \"global_scale\": " << result.global_scale << ",\n"
         << "  \"compression_ratio\": " << input_bytes / payload_bytes << ",\n"
         << "  \"max_abs_error\": " << metrics.max_abs_error << ",\n"
         << "  \"mae\": " << metrics.mae << ",\n"
         << "  \"mse\": " << metrics.mse << ",\n"
         << "  \"warmup\": " << warmup << ",\n"
         << "  \"iterations\": " << iterations << ",\n"
         << "  \"quant_kernel_ms\": " << result.quant_ms << ",\n"
         << "  \"dequant_kernel_ms\": " << result.dequant_ms << ",\n"
         << "  \"quant_effective_gbps\": " << quant_gbps << ",\n"
         << "  \"dequant_effective_gbps\": " << dequant_gbps << "\n"
         << "}\n";
}

TensorInput make_tensor(std::vector<float> values, std::int64_t rows,
                        std::int64_t cols) {
  // 自测数据已经是 FP32，直接复制为 device 输入字节并保留同一份 reference。
  TensorInput tensor;
  tensor.rows = rows;
  tensor.cols = cols;
  tensor.type = lpq::InputType::kFp32;
  tensor.reference = std::move(values);
  tensor.bytes.resize(tensor.reference.size() * sizeof(float));
  std::memcpy(tensor.bytes.data(), tensor.reference.data(), tensor.bytes.size());
  return tensor;
}

bool self_test_case(const std::string& distribution, lpq::Format format,
                    lpq::ScaleMode scale_mode,
                    const std::vector<float>& values) {
  // 覆盖一种分布、格式和 scale 模式，检查输出有限、packed 长度正确且误差在
  // 该格式的预期范围内。
  AppConfig config;
  config.quant.format = format;
  config.quant.scale_mode = scale_mode;
  config.quant.block_size = format == lpq::Format::kMxfp8 ? 32 : 16;
  config.output_type = lpq::OutputType::kFp32;
  const auto tensor = make_tensor(values, 7, values.size() / 7);
  const auto result = run_cuda(tensor, config, 1, 2);
  const auto metrics = compute_metrics(values, result.output_fp32);
  const double threshold = format == lpq::Format::kMxfp8 ? 0.08 : 0.65;
  const bool finite =
      std::all_of(result.output_fp32.begin(), result.output_fp32.end(),
                  [](float x) { return std::isfinite(x); });
  const bool packed_ok =
      result.packed.size() == lpq::packed_bytes(values.size(), format);
  const bool pass = finite && packed_ok && metrics.mae < threshold;
  std::cout << (pass ? "PASS" : "FAIL") << " distribution=" << distribution
            << " format=" << format_name(format)
            << " scale=" << scale_mode_name(scale_mode)
            << " mae=" << metrics.mae << " max=" << metrics.max_abs_error
            << " packed_bytes=" << result.packed.size() << '\n';
  return pass;
}

int self_test() {
  // 自测顺序：先验证格式常量，再验证固定 byte 布局，最后跑分布/模式矩阵和
  // stochastic rounding 的可复现性。
  if (lpq::decode_e4m3(0x7e) != 448.0f ||
      lpq::decode_e4m3(0x01) != std::ldexp(1.0f, -9) ||
      lpq::decode_e8m0(127) != 1.0f || lpq::decode_e2m1(7) != 6.0f ||
      lpq::decode_e2m1(15) != -6.0f) {
    std::cerr << "FAIL exact format decoding\n";
    return 1;
  }
  AppConfig mx_exact_config;
  mx_exact_config.quant.format = lpq::Format::kMxfp8;
  mx_exact_config.quant.scale_mode = lpq::ScaleMode::kBlock;
  mx_exact_config.quant.block_size = 32;
  mx_exact_config.output_type = lpq::OutputType::kFp32;
  std::vector<float> mx_exact_values(32, 0.0f);
  mx_exact_values[1] = 1.0f;
  mx_exact_values[2] = -1.0f;
  mx_exact_values[3] = 448.0f;
  mx_exact_values[4] = -448.0f;
  const auto mx_exact =
      run_cuda(make_tensor(mx_exact_values, 1, 32), mx_exact_config, 0, 1);
  // 32 元素 block 的 E8M0 scale=1，前几个 byte 对应 0、1、-1、448、-448。
  const bool mx_layout =
      mx_exact.scales == std::vector<std::uint8_t>{127} &&
      mx_exact.packed[0] == 0x00 && mx_exact.packed[1] == 0x38 &&
      mx_exact.packed[2] == 0xb8 && mx_exact.packed[3] == 0x7e &&
      mx_exact.packed[4] == 0xfe &&
      mx_exact.output_fp32 == mx_exact_values;
  std::cout << (mx_layout ? "PASS" : "FAIL")
            << " exact MXFP8 GPU encoding\n";

  AppConfig nv_exact_config;
  nv_exact_config.quant.format = lpq::Format::kNvfp4;
  nv_exact_config.quant.scale_mode = lpq::ScaleMode::kBlock;
  nv_exact_config.quant.block_size = 16;
  nv_exact_config.output_type = lpq::OutputType::kFp32;
  const std::vector<float> nv_exact_values = {
      0.0f, 224.0f,  448.0f,  672.0f,  896.0f,  1344.0f,
      1792.0f, 2688.0f, 0.0f, -224.0f, -448.0f, -672.0f,
      -896.0f, -1344.0f, -1792.0f, -2688.0f};
  const auto nv_exact =
      run_cuda(make_tensor(nv_exact_values, 1, 16), nv_exact_config, 0, 1);
  const std::vector<std::uint8_t> expected_nv_packed = {
      0x10, 0x32, 0x54, 0x76, 0x90, 0xba, 0xdc, 0xfe};
  // 16 个 NVFP4 元素按两个一组打包，正负 code 的 nibble 顺序也在这里锁定。
  const bool nv_layout =
      nv_exact.scales == std::vector<std::uint8_t>{0x7e} &&
      nv_exact.packed == expected_nv_packed &&
      nv_exact.global_scale == 1.0f &&
      nv_exact.output_fp32 == nv_exact_values;
  std::cout << (nv_layout ? "PASS" : "FAIL")
            << " exact packed NVFP4 GPU encoding\n";

  constexpr std::size_t count = 7 * 137;
  std::mt19937 generator(2026);
  std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::vector<float> uniform_values(count);
  std::vector<float> normal_values(count);
  std::vector<float> outlier_values(count);
  for (std::size_t i = 0; i < count; ++i) {
    uniform_values[i] = uniform(generator);
    normal_values[i] = normal(generator);
    outlier_values[i] = normal(generator) * 0.1f;
  }
  // 稀疏异常值用于观察 block scaling 是否把异常影响限制在局部分组内。
  for (std::size_t i = 0; i < count; i += 97)
    outlier_values[i] = (i & 1) == 0 ? 12.0f : -9.0f;
  bool pass = mx_layout && nv_layout;
  for (const auto format : {lpq::Format::kMxfp8, lpq::Format::kNvfp4}) {
    for (const auto mode : {lpq::ScaleMode::kBlock,
                            lpq::ScaleMode::kTensor}) {
      pass &= self_test_case("uniform", format, mode, uniform_values);
      pass &= self_test_case("normal", format, mode, normal_values);
      pass &= self_test_case("outlier", format, mode, outlier_values);
    }
    pass &= self_test_case("zero", format, lpq::ScaleMode::kBlock,
                           std::vector<float>(35, 0.0f));
  }
  AppConfig stochastic;
  stochastic.quant.format = lpq::Format::kNvfp4;
  stochastic.quant.scale_mode = lpq::ScaleMode::kBlock;
  stochastic.quant.block_size = 16;
  stochastic.quant.rounding = lpq::Rounding::kStochastic;
  stochastic.output_type = lpq::OutputType::kFp32;
  // 同一输入和 seed 连续运行两次，packed/scales 必须逐 byte 相同。
  const auto stochastic_tensor = make_tensor(uniform_values, 7, 137);
  const auto stochastic_a = run_cuda(stochastic_tensor, stochastic, 0, 1);
  const auto stochastic_b = run_cuda(stochastic_tensor, stochastic, 0, 1);
  const bool reproducible = stochastic_a.packed == stochastic_b.packed &&
                            stochastic_a.scales == stochastic_b.scales;
  std::cout << (reproducible ? "PASS" : "FAIL")
            << " stochastic rounding reproducibility\n";
  pass &= reproducible;
  std::cout << (pass ? "ALL TESTS PASSED\n" : "SELF TEST FAILED\n");
  return pass ? 0 : 1;
}

void usage(const char* program) {
  // CLI 只有两条路径：运行一次量化实验，或执行不写文件的内置自测。
  std::cout
      << "Usage:\n  " << program
      << " --input tensor.bin --config quant.toml --output-dir results "
         "[--warmup N] [--iters N]\n  "
      << program << " --self-test\n";
}

}  // namespace

int main(int argc, char** argv) {
  // main 只编排输入解析、CUDA 运行和结果落盘；量化算法位于 low_precision.cu。
  try {
    fs::path input_path;
    fs::path config_path;
    fs::path output_dir = "results/run";
    int warmup = 10;
    int iterations = 100;
    bool run_self_test = false;
    for (int i = 1; i < argc; ++i) {
      // 选项采用显式顺序解析，数值选项的缺失值在这里立即报错。
      const std::string argument = argv[i];
      auto value = [&](const char* option) {
        if (++i >= argc)
          throw std::runtime_error(std::string(option) + " requires a value");
        return std::string(argv[i]);
      };
      if (argument == "--input")
        input_path = value("--input");
      else if (argument == "--config")
        config_path = value("--config");
      else if (argument == "--output-dir")
        output_dir = value("--output-dir");
      else if (argument == "--warmup")
        warmup = std::stoi(value("--warmup"));
      else if (argument == "--iters")
        iterations = std::stoi(value("--iters"));
      else if (argument == "--self-test")
        run_self_test = true;
      else if (argument == "--help" || argument == "-h") {
        usage(argv[0]);
        return 0;
      } else {
        throw std::runtime_error("unknown argument: " + argument);
      }
    }
    if (run_self_test) return self_test();
    if (input_path.empty() || config_path.empty()) {
      usage(argv[0]);
      return 2;
    }

    const auto tensor = load_tensor(input_path);
    const auto config = load_config(config_path);
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp device{};
    CUDA_CHECK(cudaGetDeviceProperties(&device, device_id));
    const auto result = run_cuda(tensor, config, warmup, iterations);
    const auto metrics =
        compute_metrics(tensor.reference, result.output_fp32);
    fs::create_directories(output_dir);
    save_quantized(output_dir / "quantized.lpq", tensor, config, result);
    write_binary(output_dir / "dequantized.bin", result.output.data(),
                 result.output.size());
    save_log(output_dir / "metrics.json", tensor, config, result, metrics,
             warmup, iterations, device);

#ifdef LPQ_WITH_HW_PATH
    // 硬件/库对照路径：在软件路径输出全部写完后追加运行，互不影响。
    if (config.hardware_path == "cvt") {
      if (!lpq::hw::cvt_supported(device))
        throw std::runtime_error(
            "hardware_path=cvt 需要 sm_89+ 的 E4M3 cvt 指令");
      const auto hw = run_hw_cvt(tensor, config, warmup, iterations);
      save_hw_cvt_log(output_dir / "hw_path.json", tensor, config, hw, warmup,
                      iterations, device);
      std::cout << "hw_cvt encode_mismatches=" << hw.encode_mismatches
                << " decode_mismatches=" << hw.decode_mismatches
                << " encode_speedup=" << hw.sw_encode_ms / hw.hw_encode_ms
                << " decode_speedup=" << hw.sw_decode_ms / hw.hw_decode_ms
                << '\n';
    } else if (config.hardware_path == "gemm") {
#ifdef LPQ_HW_CUBLAS
      if (tensor.rows != tensor.cols)
        throw std::runtime_error("hardware_path=gemm 需要方阵输入（M=N=K）");
      if (tensor.type != lpq::InputType::kFp32)
        throw std::runtime_error("hardware_path=gemm 需要 FP32 输入");
      DeviceBuffer d_input(tensor.bytes.size());
      CUDA_CHECK(cudaMemcpy(d_input.get(), tensor.bytes.data(),
                            tensor.bytes.size(), cudaMemcpyHostToDevice));
      lpq::hw::GemmBenchResult gemm;
      CUDA_CHECK(lpq::hw::run_gemm_comparison(
          static_cast<const float*>(d_input.get()),
          static_cast<int>(tensor.rows), config.quant.format, warmup,
          iterations, &gemm));
      save_hw_gemm_log(output_dir / "hw_gemm.json", tensor, config, gemm,
                       warmup, iterations, device);
      std::cout << "hw_gemm supported=" << gemm.hw_supported
                << " hw_gemm_ms=" << gemm.hw_gemm_ms
                << " sw_pipeline_ms=" << gemm.sw_dequant_ms + gemm.sw_gemm_ms
                << '\n';
#else
      throw std::runtime_error(
          "hardware_path=gemm 需要构建时找到 cuBLASLt（-DCUBLAS_ROOT=...）");
#endif
    }
#endif  // LPQ_WITH_HW_PATH

    std::cout << std::setprecision(7) << "format="
              << format_name(config.quant.format) << " shape=" << tensor.rows
              << 'x' << tensor.cols << " output="
              << output_name(config.output_type) << '\n'
#ifdef LPQ_WITH_HADAMARD
              << "hadamard=" << config.hadamard << '\n'
#endif
              << "max_abs_error=" << metrics.max_abs_error
              << " mae=" << metrics.mae << " mse=" << metrics.mse << '\n'
              << "quant_ms=" << result.quant_ms
              << " dequant_ms=" << result.dequant_ms << '\n'
              << "wrote " << output_dir << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}

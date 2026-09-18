#include <cuda_runtime.h>
#include "../CUDACommon/output.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kBlockSize = 32;

// 统一检查 CUDA API，出错时给出可定位的异常信息。
void check_cuda(cudaError_t status, const char* expression, const char* file,
                int line) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(expression) + " failed at " + file + ":" +
                             std::to_string(line) + ": " + cudaGetErrorString(status));
  }
}

#define CUDA_CHECK(expression) check_cuda((expression), #expression, __FILE__, __LINE__)

// 主机端量化结果：每元素一个 E4M3 字节，每 32 个元素一个 E8M0 scale 字节。
struct QuantizedFile {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::vector<std::uint8_t> data;
  std::vector<std::uint8_t> scales;
};

struct FloatInput {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::vector<float> values;
};

// 读取 CPU reference 写出的 MXFP8Q1 文件。
QuantizedFile read_quantized(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open quantized file: " + path);
  char magic[8];
  std::uint32_t version = 0, block = 0;
  std::uint64_t rows = 0, cols = 0, data_bytes = 0, scale_bytes = 0;
  file.read(magic, 8);
  file.read(reinterpret_cast<char*>(&version), sizeof(version));
  file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
  file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
  file.read(reinterpret_cast<char*>(&block), sizeof(block));
  file.read(reinterpret_cast<char*>(&data_bytes), sizeof(data_bytes));
  file.read(reinterpret_cast<char*>(&scale_bytes), sizeof(scale_bytes));
  if (std::string(magic, 7) != "MXFP8Q1" || version != 1 || block != kBlockSize)
    throw std::runtime_error("invalid MXFP8Q1 header: " + path);
  if (data_bytes != rows * cols || scale_bytes != (data_bytes + kBlockSize - 1) / kBlockSize)
    throw std::runtime_error("invalid MXFP8Q1 payload sizes: " + path);
  QuantizedFile result{static_cast<std::size_t>(rows), static_cast<std::size_t>(cols),
                       std::vector<std::uint8_t>(static_cast<std::size_t>(data_bytes)),
                       std::vector<std::uint8_t>(static_cast<std::size_t>(scale_bytes))};
  file.read(reinterpret_cast<char*>(result.data.data()),
            static_cast<std::streamsize>(result.data.size()));
  file.read(reinterpret_cast<char*>(result.scales.data()),
            static_cast<std::streamsize>(result.scales.size()));
  if (!file) throw std::runtime_error("truncated MXFP8Q1 file: " + path);
  return result;
}

// 读取 CPU 反量化 golden 文件。
std::vector<float> read_dequant(const std::string& path, std::size_t& rows,
                                std::size_t& cols) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open dequant file: " + path);
  char magic[8];
  std::uint32_t version = 0;
  std::uint64_t row_count = 0, col_count = 0, count = 0;
  file.read(magic, 8);
  file.read(reinterpret_cast<char*>(&version), sizeof(version));
  file.read(reinterpret_cast<char*>(&row_count), sizeof(row_count));
  file.read(reinterpret_cast<char*>(&col_count), sizeof(col_count));
  file.read(reinterpret_cast<char*>(&count), sizeof(count));
  if (std::string(magic, 8) != "FP32DEQ1" || version != 1)
    throw std::runtime_error("invalid FP32DEQ1 header: " + path);
  rows = static_cast<std::size_t>(row_count);
  cols = static_cast<std::size_t>(col_count);
  if (count != rows * cols) throw std::runtime_error("invalid FP32DEQ1 payload: " + path);
  std::vector<float> values(static_cast<std::size_t>(count));
  file.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!file) throw std::runtime_error("truncated FP32DEQ1 file: " + path);
  return values;
}

// 读取 CPU reference 使用的固定 FP32INP1 输入文件。
FloatInput read_fp32_input(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open FP32 input: " + path);
  char magic[8];
  std::uint32_t version = 0;
  std::uint64_t rows = 0, cols = 0;
  file.read(magic, 8);
  file.read(reinterpret_cast<char*>(&version), sizeof(version));
  file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
  file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
  if (std::string(magic, 8) != "FP32INP1" || version != 1)
    throw std::runtime_error("invalid FP32INP1 header: " + path);
  FloatInput result{static_cast<std::size_t>(rows), static_cast<std::size_t>(cols),
                    std::vector<float>(static_cast<std::size_t>(rows * cols))};
  file.read(reinterpret_cast<char*>(result.values.data()),
            static_cast<std::streamsize>(result.values.size() * sizeof(float)));
  if (!file) throw std::runtime_error("truncated FP32INP1 file: " + path);
  return result;
}

// 写出与 CPU reference 完全相同的 MXFP8Q1 文件布局。
void write_quantized(const std::string& path, const QuantizedFile& q) {
  std::ofstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open CUDA quantized output: " + path);
  const std::uint32_t version = 1;
  const std::uint64_t rows = q.rows, cols = q.cols;
  const std::uint32_t block = static_cast<std::uint32_t>(kBlockSize);
  const std::uint64_t data_bytes = q.data.size(), scale_bytes = q.scales.size();
  file.write("MXFP8Q1", 8);
  file.write(reinterpret_cast<const char*>(&version), sizeof(version));
  file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
  file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
  file.write(reinterpret_cast<const char*>(&block), sizeof(block));
  file.write(reinterpret_cast<const char*>(&data_bytes), sizeof(data_bytes));
  file.write(reinterpret_cast<const char*>(&scale_bytes), sizeof(scale_bytes));
  file.write(reinterpret_cast<const char*>(q.data.data()),
             static_cast<std::streamsize>(q.data.size()));
  file.write(reinterpret_cast<const char*>(q.scales.data()),
             static_cast<std::streamsize>(q.scales.size()));
  if (!file) throw std::runtime_error("failed writing CUDA quantized output: " + path);
}


// E4M3FN 解码公式必须与 CPU reference 完全一致。
__device__ float decode_e4m3(std::uint8_t code) {
  const int sign = (code & 0x80u) ? -1 : 1;
  const int exponent = (code >> 3) & 0x0fu;
  const int mantissa = code & 0x07u;
  if (exponent == 0) return sign * scalbnf(static_cast<float>(mantissa), -9);
  if (exponent == 15 && mantissa == 7) return __int_as_float(0x7fc00000);
  return sign * scalbnf(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7);
}

// 枚举全部有限 E4M3 编码，采用与 CPU reference 相同的“误差更小才更新”规则。
// 等距时保留先枚举的编码，不是 FP16/BF16 输出转换使用的 nearest-even。
__device__ std::uint8_t encode_e4m3(float value) {
  if (isnan(value) || value == 0.0f) return 0;
  const float clipped = fminf(fmaxf(value, -448.0f), 448.0f);
  const float positive_infinity = __int_as_float(0x7f800000);
  float best_error = positive_infinity;
  std::uint8_t best = 0;
  for (int code = 0; code < 256; ++code) {
    const float candidate = decode_e4m3(static_cast<std::uint8_t>(code));
    if (!isfinite(candidate)) continue;
    const float error = fabsf(candidate - clipped);
    if (error < best_error) {
      best_error = error;
      best = static_cast<std::uint8_t>(code);
    }
  }
  return best;
}

// 每个线程负责一个元素，scale 由元素所属的 32 元素 block 决定。
template <typename T>
__global__ void mxfp8_dequant_kernel(const std::uint8_t* data,
                                     const std::uint8_t* scales, T* output,
                                     std::size_t count) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const std::size_t block = index / kBlockSize;
  const int exponent = static_cast<int>(scales[block]);
  const float scale = scalbnf(1.0f, exponent - 127);
  // 中间值按 FP32 计算，最后一次转换直接写入目标类型显存，不额外启动转换 kernel。
  output[index] = cuda_output::Format<T>::convert(decode_e4m3(data[index]) * scale);
}

// 在 GPU 上执行反量化，并使用 CUDA event 统计 kernel-only 时间。
template <typename T = float>
std::vector<T> dequantize_cuda(const QuantizedFile& input, float& kernel_ms) {
  const std::size_t count = input.data.size();
  if (count == 0) { kernel_ms = 0.0f; return {}; }
  std::uint8_t* device_data = nullptr;
  std::uint8_t* device_scales = nullptr;
  T* device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_data, input.data.size() * sizeof(std::uint8_t)));
  CUDA_CHECK(cudaMalloc(&device_scales, input.scales.size() * sizeof(std::uint8_t)));
  CUDA_CHECK(cudaMalloc(&device_output, count * sizeof(T)));
  try {
    CUDA_CHECK(cudaMemcpy(device_data, input.data.data(), input.data.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_scales, input.scales.data(), input.scales.size(), cudaMemcpyHostToDevice));
    const int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    mxfp8_dequant_kernel<<<blocks, threads>>>(device_data, device_scales, device_output, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::vector<T> output(count);
    CUDA_CHECK(cudaMemcpy(output.data(), device_output, count * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_data));
    CUDA_CHECK(cudaFree(device_scales));
    CUDA_CHECK(cudaFree(device_output));
    return output;
  } catch (...) {
    cudaFree(device_data);
    cudaFree(device_scales);
    cudaFree(device_output);
    throw;
  }
}

// 一个 CUDA block 对应一个 32 元素量化 block，先归约 amax，再并行编码 E4M3。
__global__ void mxfp8_quant_kernel(const float* input, std::uint8_t* data,
                                   std::uint8_t* scales, std::size_t count) {
  __shared__ float abs_values[kBlockSize];
  const std::size_t lane = threadIdx.x;
  const std::size_t block = static_cast<std::size_t>(blockIdx.x);
  const std::size_t index = block * kBlockSize + lane;
  // 尾部无效线程用零参与最大值归约，不能提前退出，否则会破坏后续块同步。
  const float value = index < count ? input[index] : 0.0f;
  abs_values[lane] = fabsf(value);
  __syncthreads();

  // 32 个线程在 shared memory 中做树形最大值归约。
  for (std::size_t stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (lane < stride) abs_values[lane] = fmaxf(abs_values[lane], abs_values[lane + stride]);
    __syncthreads();
  }

  // 仅线程 0 写共享 scale；同步后所有有效线程再用该 scale 编码。
  if (lane == 0) {
    const float max_abs = abs_values[0];
    int exponent = 127;
    if (max_abs > 0.0f && isfinite(max_abs)) {
      exponent = static_cast<int>(ceilf(log2f(max_abs / 448.0f))) + 127;
      exponent = max(0, min(254, exponent));
    } else if (isinf(max_abs)) {
      exponent = 254;
    }
    scales[block] = static_cast<std::uint8_t>(exponent);
  }
  __syncthreads();

  if (index < count) {
    const float scale = scalbnf(1.0f, static_cast<int>(scales[block]) - 127);
    data[index] = encode_e4m3(value / scale);
  }
}

// 在 GPU 上量化 FP32 输入，并返回 kernel-only 时间。
QuantizedFile quantize_cuda(const FloatInput& input, float& kernel_ms) {
  const std::size_t count = input.values.size();
  const std::size_t blocks = (count + kBlockSize - 1) / kBlockSize;
  QuantizedFile result{input.rows, input.cols, std::vector<std::uint8_t>(count),
                       std::vector<std::uint8_t>(blocks)};
  if (count == 0) {
    kernel_ms = 0.0f;
    return result;
  }

  float* device_input = nullptr;
  std::uint8_t* device_data = nullptr;
  std::uint8_t* device_scales = nullptr;
  CUDA_CHECK(cudaMalloc(&device_input, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_data, count * sizeof(std::uint8_t)));
  CUDA_CHECK(cudaMalloc(&device_scales, blocks * sizeof(std::uint8_t)));
  try {
    CUDA_CHECK(cudaMemcpy(device_input, input.values.data(), count * sizeof(float),
                          cudaMemcpyHostToDevice));
    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    mxfp8_quant_kernel<<<static_cast<unsigned int>(blocks), kBlockSize>>>(
        device_input, device_data, device_scales, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaMemcpy(result.data.data(), device_data, count * sizeof(std::uint8_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.scales.data(), device_scales,
                          blocks * sizeof(std::uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_data));
    CUDA_CHECK(cudaFree(device_scales));
    return result;
  } catch (...) {
    cudaFree(device_input);
    cudaFree(device_data);
    cudaFree(device_scales);
    throw;
  }
}

// 量化结果必须逐字节匹配 CPU golden，而不是只比较反量化后的浮点误差。
void compare_quantized(const QuantizedFile& actual, const QuantizedFile& expected) {
  if (actual.rows != expected.rows || actual.cols != expected.cols ||
      actual.data.size() != expected.data.size() ||
      actual.scales.size() != expected.scales.size())
    throw std::runtime_error("quantized shape mismatch");
  std::size_t data_mismatches = 0, scale_mismatches = 0;
  for (std::size_t i = 0; i < actual.data.size(); ++i)
    if (actual.data[i] != expected.data[i]) ++data_mismatches;
  for (std::size_t i = 0; i < actual.scales.size(); ++i)
    if (actual.scales[i] != expected.scales[i]) ++scale_mismatches;
  std::cout << "data_byte_mismatches=" << data_mismatches
            << " \nscale_byte_mismatches=" << scale_mismatches << '\n';
  if (data_mismatches != 0 || scale_mismatches != 0)
    throw std::runtime_error("CUDA/CPU quantized byte comparison failed");
}


// CLI 反量化流程：调用 GPU 封装、比较 CPU golden、按 T 写出文件。
template <typename T>
void run_dequant(const QuantizedFile& input, const std::vector<float>& golden,
                 const std::string& path) {
  float kernel_ms = 0.0f;
  const auto output = dequantize_cuda<T>(input, kernel_ms);
  cuda_output::compare(output, golden);
  cuda_output::write(path, input.rows, input.cols, output);
  std::cout << "cuda_dequant=pass kernel_ms=" << kernel_ms
            << " output_bytes=" << output.size() * sizeof(T) << " output=" << path << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string dtype = cuda_output::parse_dtype(argc, argv);
    if (argc > 1 && std::string(argv[1]) == "--quantize") {
      if (argc < 4 || argc > 5) {
        std::cerr << "usage: " << argv[0]
                  << " --quantize <input.fp32> <cuda_output.mxfp8> [cpu_golden.mxfp8]\n";
        return 2;
      }
      const FloatInput input = read_fp32_input(argv[2]);
      float kernel_ms = 0.0f;
      const QuantizedFile output = quantize_cuda(input, kernel_ms);
      if (argc == 5) compare_quantized(output, read_quantized(argv[4]));
      write_quantized(argv[3], output);
      const double bytes = static_cast<double>(input.values.size() * sizeof(float) +
                                               output.data.size() + output.scales.size());
      const double bandwidth = kernel_ms > 0.0f ? bytes / (kernel_ms * 1e6) : 0.0;
      std::cout << "cuda_quantize=pass \nkernel_ms=" << kernel_ms
                << " \neffective_bandwidth_GBps=" << bandwidth
                << " \noutput=" << argv[3] << '\n';
      return 0;
    }
    if (argc < 3 || argc > 4) {
      std::cerr << "usage: " << argv[0]
                << " <cpu_quantized.mxfp8> <cpu_dequant.fp32> [cuda_output] [--output-dtype fp32|fp16|bf16]\n"
                << "   or: " << argv[0]
                << " --quantize <input.fp32> <cuda_output.mxfp8> [cpu_golden.mxfp8]\n";
      return 2;
    }
    const std::string quantized_path = argv[1];
    const std::string golden_path = argv[2];
    const std::string output_path = argc > 3 ? argv[3] : "cuda_dequant." + dtype;
    const QuantizedFile quantized = read_quantized(quantized_path);
    std::size_t golden_rows = 0, golden_cols = 0;
    const auto golden = read_dequant(golden_path, golden_rows, golden_cols);
    if (golden_rows != quantized.rows || golden_cols != quantized.cols)
      throw std::runtime_error("CPU golden shape does not match quantized input");
    // 运行时字符串选择编译期模板实例，三种输出共用同一套反量化公式。
    if (dtype == "fp16") run_dequant<__half>(quantized, golden, output_path);
    else if (dtype == "bf16") run_dequant<__nv_bfloat16>(quantized, golden, output_path);
    else run_dequant<float>(quantized, golden, output_path);
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}

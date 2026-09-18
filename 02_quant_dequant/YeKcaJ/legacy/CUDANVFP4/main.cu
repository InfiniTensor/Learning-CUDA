#include <cuda_runtime.h>
#include "../CUDACommon/output.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kBlockSize = 16;
constexpr float kNvfp4Max = 6.0f;
constexpr float kE4m3Max = 448.0f;
constexpr float kGlobalScaleDenominator = kNvfp4Max * kE4m3Max;

// 将 CUDA API 错误转换为异常，同时保留调用位置和 CUDA 错误描述。
void check_cuda(cudaError_t status, const char* expression, const char* file,
                int line) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(expression) + " failed at " + file + ":" +
                             std::to_string(line) + ": " + cudaGetErrorString(status));
  }
}

#define CUDA_CHECK(expression) check_cuda((expression), #expression, __FILE__, __LINE__)

struct FloatInput {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::vector<float> values;
};

// 主机端量化结果：两元素共用一个 packed 字节，每 16 元素一个 E4M3 scale。
struct QuantizedFile {
  std::size_t rows = 0;
  std::size_t cols = 0;
  float global_scale = 1.0f;
  std::vector<std::uint8_t> packed;
  std::vector<std::uint8_t> block_scales;
};

// 读取冻结 reference 的 FP32INP1 文件；values 按行主序展开。
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
  if (rows > static_cast<std::uint64_t>(SIZE_MAX) ||
      cols > static_cast<std::uint64_t>(SIZE_MAX) || rows * cols > SIZE_MAX)
    throw std::runtime_error("FP32 input dimensions are too large: " + path);
  FloatInput result{static_cast<std::size_t>(rows), static_cast<std::size_t>(cols),
                    std::vector<float>(static_cast<std::size_t>(rows * cols))};
  file.read(reinterpret_cast<char*>(result.values.data()),
            static_cast<std::streamsize>(result.values.size() * sizeof(float)));
  if (!file) throw std::runtime_error("truncated FP32INP1 file: " + path);
  return result;
}

// 这里读取 CPU 的 FP32 反量化 golden，不是量化前的原始输入。
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
  if (row_count > static_cast<std::uint64_t>(SIZE_MAX) ||
      col_count > static_cast<std::uint64_t>(SIZE_MAX) || count > SIZE_MAX ||
      row_count * col_count != count)
    throw std::runtime_error("invalid FP32DEQ1 dimensions: " + path);
  rows = static_cast<std::size_t>(row_count);
  cols = static_cast<std::size_t>(col_count);
  std::vector<float> values(static_cast<std::size_t>(count));
  file.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!file) throw std::runtime_error("truncated FP32DEQ1 file: " + path);
  return values;
}

// 按 NVFP4Q1 布局读取全局 scale、packed 数据和 block scales，并校验长度。
QuantizedFile read_quantized(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open NVFP4 file: " + path);
  char magic[8];
  std::uint32_t version = 0, block = 0;
  std::uint64_t rows = 0, cols = 0, data_bytes = 0, scale_bytes = 0;
  float global_scale = 1.0f;
  file.read(magic, 8);
  file.read(reinterpret_cast<char*>(&version), sizeof(version));
  file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
  file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
  file.read(reinterpret_cast<char*>(&block), sizeof(block));
  file.read(reinterpret_cast<char*>(&global_scale), sizeof(global_scale));
  file.read(reinterpret_cast<char*>(&data_bytes), sizeof(data_bytes));
  file.read(reinterpret_cast<char*>(&scale_bytes), sizeof(scale_bytes));
  if (std::string(magic, 7) != "NVFP4Q1" || version != 1 || block != kBlockSize)
    throw std::runtime_error("invalid NVFP4Q1 header: " + path);
  if (rows > static_cast<std::uint64_t>(SIZE_MAX) ||
      cols > static_cast<std::uint64_t>(SIZE_MAX) || rows * cols > SIZE_MAX)
    throw std::runtime_error("NVFP4 dimensions are too large: " + path);
  const std::uint64_t count = rows * cols;
  if (data_bytes != (count + 1) / 2 || scale_bytes != (count + kBlockSize - 1) / kBlockSize)
    throw std::runtime_error("invalid NVFP4 payload sizes: " + path);
  QuantizedFile result{static_cast<std::size_t>(rows), static_cast<std::size_t>(cols), global_scale,
                       std::vector<std::uint8_t>(static_cast<std::size_t>(data_bytes)),
                       std::vector<std::uint8_t>(static_cast<std::size_t>(scale_bytes))};
  file.read(reinterpret_cast<char*>(result.packed.data()),
            static_cast<std::streamsize>(result.packed.size()));
  file.read(reinterpret_cast<char*>(result.block_scales.data()),
            static_cast<std::streamsize>(result.block_scales.size()));
  if (!file) throw std::runtime_error("truncated NVFP4Q1 file: " + path);
  return result;
}

// 保持 CPU reference 的文件布局和版本号，便于完整文件逐字节比较。
void write_quantized(const std::string& path, const QuantizedFile& q) {
  std::ofstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open CUDA NVFP4 output: " + path);
  const std::uint32_t version = 1;
  const std::uint64_t rows = q.rows, cols = q.cols;
  const std::uint32_t block = static_cast<std::uint32_t>(kBlockSize);
  const std::uint64_t data_bytes = q.packed.size(), scale_bytes = q.block_scales.size();
  file.write("NVFP4Q1", 8);
  file.write(reinterpret_cast<const char*>(&version), sizeof(version));
  file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
  file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
  file.write(reinterpret_cast<const char*>(&block), sizeof(block));
  file.write(reinterpret_cast<const char*>(&q.global_scale), sizeof(q.global_scale));
  file.write(reinterpret_cast<const char*>(&data_bytes), sizeof(data_bytes));
  file.write(reinterpret_cast<const char*>(&scale_bytes), sizeof(scale_bytes));
  file.write(reinterpret_cast<const char*>(q.packed.data()),
             static_cast<std::streamsize>(q.packed.size()));
  file.write(reinterpret_cast<const char*>(q.block_scales.data()),
             static_cast<std::streamsize>(q.block_scales.size()));
  if (!file) throw std::runtime_error("failed writing CUDA NVFP4 output: " + path);
}


// E4M3 在 NVFP4 中表示局部 scale；指数为零时按 subnormal 公式解码。
__device__ float decode_e4m3(std::uint8_t code) {
  const int sign = (code & 0x80u) ? -1 : 1;
  const int exponent = (code >> 3) & 0x0fu;
  const int mantissa = code & 0x07u;
  if (exponent == 0) return sign * scalbnf(static_cast<float>(mantissa), -9);
  if (exponent == 15 && mantissa == 7) return __int_as_float(0x7fc00000);
  return sign * scalbnf(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7);
}

// 枚举有限编码，等距时保留先出现者；与冻结 CPU 规则一致，不是 nearest-even。
__device__ std::uint8_t encode_e4m3(float value) {
  if (isnan(value) || value == 0.0f) return 0;
  const float clipped = fminf(fmaxf(value, -kE4m3Max), kE4m3Max);
  float best_error = __int_as_float(0x7f800000);
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

// nibble 的高一位是符号，低三位索引 E2M1 幅值表。
__device__ float decode_e2m1(std::uint8_t nibble) {
  constexpr float magnitudes[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                   2.0f, 3.0f, 4.0f, 6.0f};
  const float magnitude = magnitudes[nibble & 0x7u];
  return (nibble & 0x8u) ? -magnitude : magnitude;
}

// 选最近幅值，中点保留较小幅值；signbit 保留负零符号。
__device__ std::uint8_t encode_e2m1(float value) {
  constexpr float magnitudes[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                   2.0f, 3.0f, 4.0f, 6.0f};
  const bool negative = signbit(value);
  const float magnitude = fabsf(value);
  int best_index = 0;
  float best_error = __int_as_float(0x7f800000);
  for (int i = 0; i < 8; ++i) {
    const float error = fabsf(magnitudes[i] - magnitude);
    if (error < best_error) {
      best_error = error;
      best_index = i;
    }
  }
  return static_cast<std::uint8_t>(best_index | (negative ? 0x8 : 0));
}

// 一个线程块扫描输入并归约全局最大绝对值；结果回到 CPU 后计算 global_scale。
__global__ void nvfp4_global_max_kernel(const float* input, float* output,
                                         std::size_t count) {
  __shared__ float values[256];
  float local_max = 0.0f;
  for (std::size_t index = threadIdx.x; index < count; index += blockDim.x)
    local_max = fmaxf(local_max, fabsf(input[index]));
  values[threadIdx.x] = local_max;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) values[threadIdx.x] = fmaxf(values[threadIdx.x], values[threadIdx.x + stride]);
    __syncthreads();
  }
  if (threadIdx.x == 0) *output = values[0];
}

// 一个 CUDA block 对应 16 个 NVFP4 元素：归约 block max、编码 scale，并打包两个 nibble。
__global__ void nvfp4_quant_kernel(const float* input, std::uint8_t* packed,
                                   std::uint8_t* block_scales, float global_scale,
                                   std::size_t count) {
  __shared__ float abs_values[kBlockSize];
  const std::size_t lane = threadIdx.x;
  const std::size_t block = static_cast<std::size_t>(blockIdx.x);
  const std::size_t base = block * kBlockSize;
  const std::size_t index = base + lane;
  // 尾部线程补零并继续参与同步，待完成归约后再限制实际读写范围。
  const float value = index < count ? input[index] : 0.0f;
  abs_values[lane] = fabsf(value);
  __syncthreads();
  for (std::size_t stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (lane < stride) abs_values[lane] = fmaxf(abs_values[lane], abs_values[lane + stride]);
    __syncthreads();
  }
  if (lane == 0) {
    const float normalized_scale = abs_values[0] > 0.0f
                                       ? abs_values[0] / (kNvfp4Max * global_scale)
                                       : 0.0f;
    block_scales[block] = encode_e4m3(normalized_scale);
  }
  __syncthreads();

  // 前 8 个线程各写一个完整字节，避免两个线程分别改写同一字节的高低位。
  if (lane < 8) {
    const std::size_t even_index = base + lane * 2;
    const std::size_t odd_index = even_index + 1;
    // 此处已结束 block 同步；无有效字节的线程可直接退出。
    if (even_index >= count) return;
    const float effective_scale = global_scale * decode_e4m3(block_scales[block]);
    // 与 CPU 一致：零 scale 编码正零，避免 0/0，也避免保留负零符号。
    const std::uint8_t low = effective_scale > 0.0f
                                 ? encode_e2m1(input[even_index] / effective_scale)
                                 : 0;
    const std::uint8_t high = odd_index < count && effective_scale > 0.0f
                                  ? encode_e2m1(input[odd_index] / effective_scale)
                                  : 0;
    packed[even_index / 2] = static_cast<std::uint8_t>(low | (high << 4));
  }
}

// 主机封装负责分配、传输和启动两个 kernel；global_scale 当前在 CPU 计算。
QuantizedFile quantize_cuda(const FloatInput& input, float& kernel_ms) {
  const std::size_t count = input.values.size();
  const std::size_t blocks = (count + kBlockSize - 1) / kBlockSize;
  QuantizedFile result{input.rows, input.cols, 1.0f,
                       std::vector<std::uint8_t>((count + 1) / 2),
                       std::vector<std::uint8_t>(blocks)};
  if (count == 0) {
    kernel_ms = 0.0f;
    return result;
  }

  float* device_input = nullptr;
  float* device_max = nullptr;
  std::uint8_t* device_packed = nullptr;
  std::uint8_t* device_scales = nullptr;
  CUDA_CHECK(cudaMalloc(&device_input, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_max, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_packed, result.packed.size()));
  CUDA_CHECK(cudaMalloc(&device_scales, result.block_scales.size()));
  try {
    CUDA_CHECK(cudaMemcpy(device_input, input.values.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    // 此计时跨越全局归约、最大值回传、CPU scale 计算和量化，不是纯 kernel 时间。
    CUDA_CHECK(cudaEventRecord(start));
    nvfp4_global_max_kernel<<<1, 256>>>(device_input, device_max, count);
    CUDA_CHECK(cudaGetLastError());
    float max_abs = 0.0f;
    CUDA_CHECK(cudaMemcpy(&max_abs, device_max, sizeof(float), cudaMemcpyDeviceToHost));
    result.global_scale = max_abs > 0.0f ? max_abs / kGlobalScaleDenominator : 1.0f;
    nvfp4_quant_kernel<<<static_cast<unsigned int>(blocks), kBlockSize>>>(
        device_input, device_packed, device_scales, result.global_scale, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaMemcpy(result.packed.data(), device_packed, result.packed.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.block_scales.data(), device_scales, result.block_scales.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_max));
    CUDA_CHECK(cudaFree(device_packed));
    CUDA_CHECK(cudaFree(device_scales));
    return result;
  } catch (...) {
    cudaFree(device_input);
    cudaFree(device_max);
    cudaFree(device_packed);
    cudaFree(device_scales);
    throw;
  }
}

// 每线程恢复一个元素；相邻线程读取同一 packed 字节的不同 nibble。
template <typename T>
__global__ void nvfp4_dequant_kernel(const std::uint8_t* packed,
                                     const std::uint8_t* block_scales,
                                     float global_scale, T* output,
                                     std::size_t count) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const std::uint8_t byte = packed[index / 2];
  const std::uint8_t nibble = (index & 1u) ? static_cast<std::uint8_t>(byte >> 4)
                                           : static_cast<std::uint8_t>(byte & 0x0fu);
  const std::size_t block = index / kBlockSize;
  const float scale = global_scale * decode_e4m3(block_scales[block]);
  // FP32 完成解码与缩放，最后一次转换到 T，直接写入目标类型显存。
  output[index] = cuda_output::Format<T>::convert(decode_e2m1(nibble) * scale);
}

// 此处 event 只包围反量化 kernel，不包含前后的显存分配及数据传输。
template <typename T = float>
std::vector<T> dequantize_cuda(const QuantizedFile& input, float& kernel_ms) {
  const std::size_t count = input.rows * input.cols;
  std::vector<T> output(count);
  if (count == 0) {
    kernel_ms = 0.0f;
    return output;
  }
  std::uint8_t* device_packed = nullptr;
  std::uint8_t* device_scales = nullptr;
  T* device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_packed, input.packed.size()));
  CUDA_CHECK(cudaMalloc(&device_scales, input.block_scales.size()));
  CUDA_CHECK(cudaMalloc(&device_output, count * sizeof(T)));
  try {
    CUDA_CHECK(cudaMemcpy(device_packed, input.packed.data(), input.packed.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_scales, input.block_scales.data(), input.block_scales.size(), cudaMemcpyHostToDevice));
    const int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    nvfp4_dequant_kernel<<<blocks, threads>>>(device_packed, device_scales,
                                               input.global_scale, device_output, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaMemcpy(output.data(), device_output, count * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_packed));
    CUDA_CHECK(cudaFree(device_scales));
    CUDA_CHECK(cudaFree(device_output));
    return output;
  } catch (...) {
    cudaFree(device_packed);
    cudaFree(device_scales);
    cudaFree(device_output);
    throw;
  }
}

// global_scale 比 FP32 位模式，packed 和 block_scales 比字节，不使用浮点容差。
void compare_quantized(const QuantizedFile& actual, const QuantizedFile& expected) {
  if (actual.rows != expected.rows || actual.cols != expected.cols ||
      actual.packed.size() != expected.packed.size() ||
      actual.block_scales.size() != expected.block_scales.size())
    throw std::runtime_error("quantized shape mismatch");
  std::uint32_t actual_bits = 0, expected_bits = 0;
  std::memcpy(&actual_bits, &actual.global_scale, sizeof(actual_bits));
  std::memcpy(&expected_bits, &expected.global_scale, sizeof(expected_bits));
  std::size_t packed_mismatches = 0, scale_mismatches = 0;
  for (std::size_t i = 0; i < actual.packed.size(); ++i)
    if (actual.packed[i] != expected.packed[i]) ++packed_mismatches;
  for (std::size_t i = 0; i < actual.block_scales.size(); ++i)
    if (actual.block_scales[i] != expected.block_scales[i]) ++scale_mismatches;
  const std::size_t global_mismatches = actual_bits == expected_bits ? 0 : 1;
  std::cout << "global_scale_mismatches=" << global_mismatches
            << " packed_byte_mismatches=" << packed_mismatches
            << " block_scale_byte_mismatches=" << scale_mismatches << '\n';
  if (actual.rows != expected.rows || actual.cols != expected.cols ||
      global_mismatches != 0 || packed_mismatches != 0 || scale_mismatches != 0)
    throw std::runtime_error("CUDA/CPU NVFP4 byte comparison failed");
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
                  << " --quantize <input.fp32> <output.nvfp4> [cpu_golden.nvfp4]\n";
        return 2;
      }
      const FloatInput input = read_fp32_input(argv[2]);
      float kernel_ms = 0.0f;
      const QuantizedFile output = quantize_cuda(input, kernel_ms);
      if (argc == 5) compare_quantized(output, read_quantized(argv[4]));
      write_quantized(argv[3], output);
      const double bytes = static_cast<double>(input.values.size() * sizeof(float) +
                                               output.packed.size() + output.block_scales.size());
      const double bandwidth = kernel_ms > 0.0f ? bytes / (kernel_ms * 1e6) : 0.0;
      std::cout << "cuda_quantize=pass kernel_ms=" << kernel_ms
                << " effective_bandwidth_GBps=" << bandwidth
                << " global_scale=" << output.global_scale
                << " output=" << argv[3] << '\n';
      return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--dequantize") {
      if (argc < 4 || argc > 5) {
        std::cerr << "usage: " << argv[0]
                  << " --dequantize <input.nvfp4> <cpu_golden.dequant.fp32> [cuda_output] [--output-dtype fp32|fp16|bf16]\n";
        return 2;
      }
      const QuantizedFile input = read_quantized(argv[2]);
      std::size_t rows = 0, cols = 0;
      const auto expected = read_dequant(argv[3], rows, cols);
      if (rows != input.rows || cols != input.cols)
        throw std::runtime_error("CPU golden shape does not match NVFP4 input");
      const std::string output_path = argc == 5 ? argv[4] : "cuda_dequant." + dtype;
      // 运行时选输出类型，编译期模板让三种输出共用同一套反量化公式。
      if (dtype == "fp16") run_dequant<__half>(input, expected, output_path);
      else if (dtype == "bf16") run_dequant<__nv_bfloat16>(input, expected, output_path);
      else run_dequant<float>(input, expected, output_path);
      return 0;
    }
    std::cerr << "usage: " << argv[0] << " --quantize ... | --dequantize ...\n";
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}

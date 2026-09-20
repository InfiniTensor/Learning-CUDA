/*
 * 模块用途：MXFP8 CUDA baseline，包含量化与反量化。
 * 量化映射：一个32线程CUDA block处理一个MXFP8块，共享内存归约amax，
 *             线程0选择E8M0 scale，随后每个有效线程编码一个E4M3元素。
 * 反量化映射：一个CUDA线程处理一个tensor元素；i/32找到共享scale。
 * benchmark接口另以CUDA event统计kernel，并保持输入、编码和恢复结果驻留显存；
 * 当前不包含warp shuffle、向量化或原生FP8指令。
 */
#include "mxfp8_cuda.h"
#include "benchmark.h"
#include "benchmark_helpers.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace luxorion {
namespace {

constexpr unsigned kThreadsPerBlock = 256;
constexpr std::size_t kQuantBlockSize = 32;

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": "
                             + cudaGetErrorString(status));
  }
}

// 简单RAII显存缓冲区：函数因异常提前结束时也会释放已经分配的显存。
template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)),
               "cudaMalloc failed");
  }

  ~DeviceBuffer() {
    if (pointer_ != nullptr) cudaFree(pointer_);
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* get() { return pointer_; }
  const T* get() const { return pointer_; }

 private:
  T* pointer_ = nullptr;
};

// 设备端只实现本kernel需要的decode。所有运算均为精确的二进制幂缩放，
// 因而其有限结果与CPU reference中先用double再转float的结果一致。
__device__ float decode_e4m3_device(std::uint8_t code) {
  const unsigned exponent_field = (code >> 3) & 0xFU;
  const unsigned mantissa = code & 0x7U;
  const float magnitude = exponent_field == 0
      ? ldexpf(static_cast<float>(mantissa), -9)
      : ldexpf(1.0F + static_cast<float>(mantissa) / 8.0F,
               static_cast<int>(exponent_field) - 7);
  return copysignf(magnitude, (code & 0x80U) ? -1.0F : 1.0F);
}

__device__ float decode_e8m0_device(std::uint8_t code) {
  return ldexpf(1.0F, static_cast<int>(code) - 127);
}

// 清晰优先的baseline编码器：枚举全部有限正E4M3编码，按距离选最近值。
// 与CPU reference一样用double比较距离，并在中点选择偶数编码。
__device__ std::uint8_t encode_e4m3_device(float value) {
  const unsigned sign = (__float_as_uint(value) >> 31) ? 0x80U : 0U;
  const double magnitude = fabs(static_cast<double>(value));
  if (magnitude >= 448.0) {
    return static_cast<std::uint8_t>(sign | 0x7EU);
  }

  unsigned best = 0;
  double best_distance = 1.0e300;
  for (unsigned code = 0; code <= 0x7EU; ++code) {
    const double candidate = static_cast<double>(
        decode_e4m3_device(static_cast<std::uint8_t>(code)));
    const double distance = fabs(magnitude - candidate);
    if (distance < best_distance
        || (distance == best_distance && (code & 1U) == 0U)) {
      best = code;
      best_distance = distance;
    }
  }
  return static_cast<std::uint8_t>(sign | best);
}

// 枚举E8M0的全部有限scale，选择第一个能覆盖amax的编码。
// 直接用double比较边界，避免极小输入先除以448而下溢。
__device__ std::uint8_t select_scale_device(float amax) {
  if (amax == 0.0F) return 127U;
  for (unsigned code = 0; code <= 254U; ++code) {
    const double capacity = ldexp(448.0, static_cast<int>(code) - 127);
    if (static_cast<double>(amax) <= capacity) {
      return static_cast<std::uint8_t>(code);
    }
  }
  return 254U;
}

__device__ float load_input(float value) { return value; }
__device__ float load_input(__half value) { return __half2float(value); }

template <typename InputT>
__global__ void mxfp8_quantize_kernel(const InputT* input,
                                      std::uint8_t* data,
                                      std::uint8_t* scales,
                                      std::size_t element_count) {
  __shared__ float magnitudes[kQuantBlockSize];
  __shared__ float scale;

  const unsigned lane = threadIdx.x;
  const std::size_t i = static_cast<std::size_t>(blockIdx.x)
                      * kQuantBlockSize + lane;
  const float value = i < element_count ? load_input(input[i]) : 0.0F;
  magnitudes[lane] = fabsf(value);
  __syncthreads();

  // 32 -> 16 -> ... -> 1的共享内存树形归约；尾块的无效线程贡献0。
  for (unsigned offset = kQuantBlockSize / 2; offset != 0; offset >>= 1) {
    if (lane < offset) {
      magnitudes[lane] = fmaxf(magnitudes[lane], magnitudes[lane + offset]);
    }
    __syncthreads();
  }

  if (lane == 0) {
    const std::uint8_t code = select_scale_device(magnitudes[0]);
    scales[blockIdx.x] = code;
    scale = decode_e8m0_device(code);
  }
  __syncthreads();

  if (i < element_count) {
    data[i] = encode_e4m3_device(value / scale);
  }
}

__global__ void mxfp8_dequantize_kernel(const std::uint8_t* data,
                                        const std::uint8_t* scales,
                                        float* output,
                                        std::size_t element_count) {
  // CUDA block/thread只决定执行编号；quantization block由元素下标i/32决定。
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x
                      + threadIdx.x;
  if (i >= element_count) return;  // 保护最后一个不完整CUDA block。

  const std::size_t quant_block = i / kQuantBlockSize;
  const float element = decode_e4m3_device(data[i]);
  const float scale = decode_e8m0_device(scales[quant_block]);
  output[i] = element * scale;
}

std::size_t quant_block_count(std::size_t size) {
  return size / kQuantBlockSize + (size % kQuantBlockSize != 0);
}

void validate_input(const MXFP8Result& q) {
  if (q.data.empty() || q.scales.size() != quant_block_count(q.data.size())) {
    throw std::invalid_argument(
        "MXFP8 CUDA data length and scale count are inconsistent");
  }
  for (std::uint8_t code : q.scales) {
    if (code == 0xFFU) {
      throw std::invalid_argument("MXFP8 CUDA NaN scale is not supported");
    }
  }
  for (std::uint8_t code : q.data) {
    if ((code & 0x7FU) == 0x7FU) {
      throw std::invalid_argument("MXFP8 CUDA NaN element is not supported");
    }
  }
}

void validate_source(const std::vector<float>& input) {
  if (input.empty()) {
    throw std::invalid_argument("MXFP8 CUDA quantization requires a nonempty input");
  }
  for (float value : input) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("MXFP8 CUDA quantization requires finite inputs");
    }
  }
}

}  // namespace

MXFP8Result mxfp8_quantize_cuda(const std::vector<float>& input) {
  validate_source(input);

  MXFP8Result result;
  result.data.resize(input.size());
  result.scales.resize(quant_block_count(input.size()));
  if (result.scales.size()
      > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::length_error("MXFP8 CUDA input is too large for a 1D grid");
  }

  DeviceBuffer<float> device_input(input.size());
  DeviceBuffer<std::uint8_t> device_data(result.data.size());
  DeviceBuffer<std::uint8_t> device_scales(result.scales.size());
  check_cuda(cudaMemcpy(device_input.get(), input.data(),
                        input.size() * sizeof(float), cudaMemcpyHostToDevice),
             "copying MXFP8 input to GPU failed");

  mxfp8_quantize_kernel<<<static_cast<unsigned>(result.scales.size()),
                           static_cast<unsigned>(kQuantBlockSize)>>>(
      device_input.get(), device_data.get(), device_scales.get(), input.size());
  check_cuda(cudaGetLastError(), "launching MXFP8 quantize kernel failed");

  check_cuda(cudaMemcpy(result.data.data(), device_data.get(), result.data.size(),
                        cudaMemcpyDeviceToHost),
             "copying MXFP8 data to CPU failed");
  check_cuda(cudaMemcpy(result.scales.data(), device_scales.get(),
                        result.scales.size(), cudaMemcpyDeviceToHost),
             "copying MXFP8 scales to CPU failed");
  return result;
}

std::vector<float> mxfp8_dequantize_cuda(const MXFP8Result& q) {
  validate_input(q);

  DeviceBuffer<std::uint8_t> device_data(q.data.size());
  DeviceBuffer<std::uint8_t> device_scales(q.scales.size());
  DeviceBuffer<float> device_output(q.data.size());

  check_cuda(cudaMemcpy(device_data.get(), q.data.data(), q.data.size(),
                        cudaMemcpyHostToDevice),
             "copying MXFP8 data to GPU failed");
  check_cuda(cudaMemcpy(device_scales.get(), q.scales.data(), q.scales.size(),
                        cudaMemcpyHostToDevice),
             "copying MXFP8 scales to GPU failed");

  const std::size_t grid_size = q.data.size() / kThreadsPerBlock
                              + (q.data.size() % kThreadsPerBlock != 0);
  if (grid_size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::length_error("MXFP8 CUDA input is too large for a 1D grid");
  }
  mxfp8_dequantize_kernel<<<static_cast<unsigned>(grid_size),
                             kThreadsPerBlock>>>(
      device_data.get(), device_scales.get(), device_output.get(), q.data.size());
  check_cuda(cudaGetLastError(), "launching MXFP8 dequantize kernel failed");

  // D2H拷贝会等待kernel完成，因此第一版无需单独cudaDeviceSynchronize。
  std::vector<float> output(q.data.size());
  check_cuda(cudaMemcpy(output.data(), device_output.get(),
                        output.size() * sizeof(float), cudaMemcpyDeviceToHost),
             "copying MXFP8 output to CPU failed");

  for (float value : output) {
    if (!std::isfinite(value)) {
      throw std::overflow_error("MXFP8 CUDA reconstruction exceeds the FP32 range");
    }
  }
  return output;
}

template <typename DeviceInput>
MXFP8BenchmarkResult benchmark_mxfp8_cuda_impl(
    const void* host_input, std::size_t host_input_bytes,
    const std::vector<float>& input, unsigned warmup, unsigned repeats) {
  validate_source(input);
  MXFP8BenchmarkResult result;
  result.quantized.data.resize(input.size());
  result.quantized.scales.resize(quant_block_count(input.size()));
  result.restored.resize(input.size());
  const std::size_t quant_blocks = result.quantized.scales.size();
  const std::size_t dequant_blocks = input.size() / kThreadsPerBlock
      + (input.size() % kThreadsPerBlock != 0);
  if (quant_blocks > static_cast<std::size_t>(std::numeric_limits<int>::max())
      || dequant_blocks > static_cast<std::size_t>(
          std::numeric_limits<int>::max())) {
    throw std::length_error("MXFP8 benchmark input is too large for a 1D grid");
  }

  DeviceBuffer<DeviceInput> device_input(input.size());
  DeviceBuffer<std::uint8_t> device_data(input.size());
  DeviceBuffer<std::uint8_t> device_scales(quant_blocks);
  DeviceBuffer<float> device_output(input.size());
  check_cuda(cudaMemcpy(device_input.get(), host_input, host_input_bytes,
                        cudaMemcpyHostToDevice),
             "copying MXFP8 benchmark input to GPU failed");

  const auto launch_quantize = [&] {
    mxfp8_quantize_kernel<<<static_cast<unsigned>(quant_blocks),
                             static_cast<unsigned>(kQuantBlockSize)>>>(
        device_input.get(), device_data.get(), device_scales.get(), input.size());
    check_cuda(cudaGetLastError(), "launching benchmark MXFP8 quantize failed");
  };
  const auto launch_dequantize = [&] {
    mxfp8_dequantize_kernel<<<static_cast<unsigned>(dequant_blocks),
                               kThreadsPerBlock>>>(
        device_data.get(), device_scales.get(), device_output.get(), input.size());
    check_cuda(cudaGetLastError(), "launching benchmark MXFP8 dequantize failed");
  };
  const auto launch_pipeline = [&] {
    launch_quantize();
    launch_dequantize();
  };

  result.timings.warmup = warmup;
  result.timings.repeats = repeats;
  result.timings.quantize = benchmark_detail::measure_cuda_events(
      warmup, repeats, launch_quantize);
  launch_quantize();
  check_cuda(cudaDeviceSynchronize(), "preparing MXFP8 dequantize benchmark failed");
  result.timings.dequantize = benchmark_detail::measure_cuda_events(
      warmup, repeats, launch_dequantize);
  result.timings.device_pipeline = benchmark_detail::measure_cuda_events(
      warmup, repeats, launch_pipeline);

  const auto run_end_to_end = [&] {
    check_cuda(cudaMemcpy(device_input.get(), host_input, host_input_bytes,
                          cudaMemcpyHostToDevice),
               "MXFP8 benchmark H2D failed");
    launch_pipeline();
    check_cuda(cudaMemcpy(result.restored.data(), device_output.get(),
                          result.restored.size() * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "MXFP8 benchmark D2H failed");
  };
  result.timings.end_to_end = benchmark_detail::measure_host(
      warmup, repeats, run_end_to_end);

  check_cuda(cudaMemcpy(result.quantized.data.data(), device_data.get(),
                        result.quantized.data.size(), cudaMemcpyDeviceToHost),
             "copying benchmark MXFP8 data to CPU failed");
  check_cuda(cudaMemcpy(result.quantized.scales.data(), device_scales.get(),
                        result.quantized.scales.size(), cudaMemcpyDeviceToHost),
             "copying benchmark MXFP8 scales to CPU failed");
  for (float value : result.restored) {
    if (!std::isfinite(value)) {
      throw std::overflow_error("MXFP8 benchmark reconstruction exceeds FP32");
    }
  }

  const std::size_t quantize_bytes = host_input_bytes + input.size()
                                   + quant_blocks;
  const std::size_t dequantize_bytes = input.size() + quant_blocks
                                     + input.size() * sizeof(float);
  result.timings.quantize_effective_gbps = benchmark_detail::effective_gbps(
      quantize_bytes, result.timings.quantize.median_ms);
  result.timings.dequantize_effective_gbps = benchmark_detail::effective_gbps(
      dequantize_bytes, result.timings.dequantize.median_ms);
  return result;
}

MXFP8BenchmarkResult benchmark_mxfp8_cuda(
    const std::vector<float>& input, unsigned warmup, unsigned repeats) {
  return benchmark_mxfp8_cuda_impl<float>(
      input.data(), input.size() * sizeof(float), input, warmup, repeats);
}

MXFP8BenchmarkResult benchmark_mxfp8_cuda_fp16(
    const std::vector<std::uint16_t>& input_bits,
    const std::vector<float>& decoded_input,
    unsigned warmup, unsigned repeats) {
  if (input_bits.size() != decoded_input.size()) {
    throw std::invalid_argument("MXFP8 FP16 bit/value counts differ");
  }
  return benchmark_mxfp8_cuda_impl<__half>(
      input_bits.data(), input_bits.size() * sizeof(std::uint16_t),
      decoded_input, warmup, repeats);
}

}  // namespace luxorion

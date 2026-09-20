/*
 * 模块用途：NVFP4 CUDA baseline，覆盖GPU全局scale、块量化/打包和反量化。
 * 量化流程：两级GPU归约得到global scale；一个16线程CUDA block处理一个
 *            NVFP4块，共享内存归约局部amax并广播实际local scale。
 * 反量化流程：一个线程恢复一个有效元素，通过i/16与i/2定位scale和packed字节。
 * baseline优先保证流程清晰和CPU逐位一致，不包含shuffle、向量化或快速编码器。
 */
#include "nvfp4_cuda.h"
#include "benchmark.h"
#include "benchmark_helpers.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace luxorion {
namespace {

constexpr unsigned kReductionThreads = 256;
constexpr unsigned kDequantThreads = 256;
constexpr unsigned kMaxReductionBlocks = 1024;
constexpr std::size_t kQuantBlockSize = 16;
constexpr std::size_t kBytesPerQuantBlock = 8;

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": "
                             + cudaGetErrorString(status));
  }
}

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

std::size_t quant_block_count(std::size_t size) {
  return size / kQuantBlockSize + (size % kQuantBlockSize != 0);
}

__device__ float decode_e4m3_device(std::uint8_t code) {
  const unsigned magnitude_code = code & 0x7FU;
  const unsigned exponent_field = magnitude_code >> 3;
  const unsigned mantissa = magnitude_code & 0x7U;
  const float magnitude = exponent_field == 0
      ? ldexpf(static_cast<float>(mantissa), -9)
      : ldexpf(1.0F + static_cast<float>(mantissa) / 8.0F,
               static_cast<int>(exponent_field) - 7);
  return copysignf(magnitude, (code & 0x80U) ? -1.0F : 1.0F);
}

__device__ float decode_e2m1_device(std::uint8_t code) {
  const float magnitudes[8] = {0.0F, 0.5F, 1.0F, 1.5F,
                               2.0F, 3.0F, 4.0F, 6.0F};
  return copysignf(magnitudes[code & 0x7U],
                   (code & 0x8U) ? -1.0F : 1.0F);
}

// 枚举有限候选的baseline编码器；double距离和偶数编码中点规则与CPU一致。
__device__ std::uint8_t encode_e4m3_device(float value) {
  const unsigned sign = (__float_as_uint(value) >> 31) ? 0x80U : 0U;
  const double magnitude = fabs(static_cast<double>(value));
  if (magnitude >= 448.0) {
    return static_cast<std::uint8_t>(sign | 0x7EU);
  }
  unsigned best = 0;
  double best_distance = 1.0e300;
  for (unsigned code = 0; code <= 0x7EU; ++code) {
    const double distance = fabs(
        magnitude - static_cast<double>(decode_e4m3_device(code)));
    if (distance < best_distance
        || (distance == best_distance && (code & 1U) == 0U)) {
      best = code;
      best_distance = distance;
    }
  }
  return static_cast<std::uint8_t>(sign | best);
}

__device__ std::uint8_t encode_e2m1_device(float value) {
  const unsigned sign = (__float_as_uint(value) >> 31) ? 0x8U : 0U;
  const double magnitude = fabs(static_cast<double>(value));
  if (magnitude >= 6.0) {
    return static_cast<std::uint8_t>(sign | 0x7U);
  }
  unsigned best = 0;
  double best_distance = 1.0e300;
  for (unsigned code = 0; code <= 0x7U; ++code) {
    const double distance = fabs(
        magnitude - static_cast<double>(decode_e2m1_device(code)));
    if (distance < best_distance
        || (distance == best_distance && (code & 1U) == 0U)) {
      best = code;
      best_distance = distance;
    }
  }
  return static_cast<std::uint8_t>(sign | best);
}

__device__ float load_input(float value) { return value; }
__device__ float load_input(__half value) { return __half2float(value); }

template <typename InputT>
__global__ void global_amax_partial_kernel(const InputT* input,
                                           std::size_t element_count,
                                           float* partial) {
  __shared__ float scratch[kReductionThreads];
  float local = 0.0F;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x
                         + threadIdx.x;
       i < element_count;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    local = fmaxf(local, fabsf(load_input(input[i])));
  }
  scratch[threadIdx.x] = local;
  __syncthreads();
  for (unsigned offset = kReductionThreads / 2; offset != 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x],
                                   scratch[threadIdx.x + offset]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) partial[blockIdx.x] = scratch[0];
}

__global__ void global_scale_kernel(const float* partial,
                                    unsigned partial_count,
                                    float* global_scale) {
  __shared__ float scratch[kReductionThreads];
  float local = 0.0F;
  for (unsigned i = threadIdx.x; i < partial_count; i += blockDim.x) {
    local = fmaxf(local, partial[i]);
  }
  scratch[threadIdx.x] = local;
  __syncthreads();
  for (unsigned offset = kReductionThreads / 2; offset != 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x],
                                   scratch[threadIdx.x + offset]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *global_scale = scratch[0] == 0.0F
        ? 1.0F
        : static_cast<float>(static_cast<double>(scratch[0]) / 2688.0);
  }
}

template <typename InputT>
__global__ void nvfp4_quantize_kernel(const InputT* input,
                                      std::uint8_t* packed_data,
                                      std::uint8_t* block_scales,
                                      const float* global_scale,
                                      std::size_t element_count) {
  __shared__ float magnitudes[kQuantBlockSize];
  __shared__ std::uint8_t codes[kQuantBlockSize];
  __shared__ float effective_scale;

  const unsigned lane = threadIdx.x;
  const std::size_t i = static_cast<std::size_t>(blockIdx.x)
                      * kQuantBlockSize + lane;
  const float value = i < element_count ? load_input(input[i]) : 0.0F;
  magnitudes[lane] = fabsf(value);
  __syncthreads();
  for (unsigned offset = kQuantBlockSize / 2; offset != 0; offset >>= 1) {
    if (lane < offset) {
      magnitudes[lane] = fmaxf(magnitudes[lane], magnitudes[lane + offset]);
    }
    __syncthreads();
  }

  if (lane == 0) {
    const float global = *global_scale;
    const float ideal = magnitudes[0] == 0.0F
        ? 1.0F
        : static_cast<float>((static_cast<double>(magnitudes[0]) / 6.0)
                             / static_cast<double>(global));
    std::uint8_t code = encode_e4m3_device(ideal);
    if (magnitudes[0] != 0.0F && code == 0) code = 1;
    block_scales[blockIdx.x] = code;
    effective_scale = global * decode_e4m3_device(code);
  }
  __syncthreads();

  codes[lane] = i < element_count
      ? encode_e2m1_device(value / effective_scale)
      : std::uint8_t{0};
  __syncthreads();
  if ((lane & 1U) == 0U) {
    // 本项目约定偶数元素在高四位、奇数元素在低四位。
    packed_data[static_cast<std::size_t>(blockIdx.x) * kBytesPerQuantBlock
                + lane / 2] = static_cast<std::uint8_t>(
        (static_cast<unsigned>(codes[lane]) << 4) | codes[lane + 1]);
  }
}

__global__ void nvfp4_dequantize_kernel(const std::uint8_t* packed_data,
                                        const std::uint8_t* block_scales,
                                        const float* global_scale,
                                        float* output,
                                        std::size_t element_count) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x
                      + threadIdx.x;
  if (i >= element_count) return;
  const std::uint8_t packed = packed_data[i / 2];
  const std::uint8_t code = static_cast<std::uint8_t>(
      i % 2 == 0 ? packed >> 4 : packed & 0xFU);
  const float scale = *global_scale * decode_e4m3_device(block_scales[i / 16]);
  output[i] = decode_e2m1_device(code) * scale;
}

void validate_quantized_input(const NVFP4Result& q) {
  const std::size_t blocks = quant_block_count(q.element_count);
  if (q.element_count == 0 || q.block_scales.size() != blocks
      || q.packed_data.size() != blocks * kBytesPerQuantBlock) {
    throw std::invalid_argument(
        "NVFP4 CUDA element count and payload sizes are inconsistent");
  }
  if (!std::isfinite(q.global_scale) || q.global_scale <= 0.0F) {
    throw std::invalid_argument(
        "NVFP4 CUDA global scale must be positive and finite");
  }
  for (std::uint8_t code : q.block_scales) {
    if (code == 0 || code > 0x7EU) {
      throw std::invalid_argument(
          "NVFP4 CUDA block scales must be positive and finite");
    }
  }
  const std::size_t used_bytes = q.element_count / 2
                               + (q.element_count % 2 != 0);
  if (q.element_count % 2 != 0
      && (q.packed_data[used_bytes - 1] & 0xFU) != 0) {
    throw std::invalid_argument("NVFP4 CUDA tail padding must be zero");
  }
  for (std::size_t i = used_bytes; i < q.packed_data.size(); ++i) {
    if (q.packed_data[i] != 0) {
      throw std::invalid_argument("NVFP4 CUDA tail padding must be zero");
    }
  }
}

void validate_source(const std::vector<float>& input) {
  if (input.empty()) {
    throw std::invalid_argument("NVFP4 CUDA quantization requires a nonempty input");
  }
  for (float value : input) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("NVFP4 CUDA quantization requires finite inputs");
    }
  }
}

}  // namespace

NVFP4Result nvfp4_quantize_cuda(const std::vector<float>& input) {
  validate_source(input);

  NVFP4Result result;
  result.element_count = input.size();
  const std::size_t blocks = quant_block_count(input.size());
  if (blocks > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::length_error("NVFP4 CUDA input is too large for a 1D grid");
  }
  result.block_scales.resize(blocks);
  result.packed_data.resize(blocks * kBytesPerQuantBlock);

  DeviceBuffer<float> device_input(input.size());
  DeviceBuffer<std::uint8_t> device_data(result.packed_data.size());
  DeviceBuffer<std::uint8_t> device_scales(result.block_scales.size());
  DeviceBuffer<float> device_global(1);
  const std::size_t blocks_needed = input.size() / kReductionThreads
                                  + (input.size() % kReductionThreads != 0);
  const unsigned reduction_blocks = static_cast<unsigned>(
      std::min<std::size_t>(blocks_needed, kMaxReductionBlocks));
  DeviceBuffer<float> device_partial(reduction_blocks);

  check_cuda(cudaMemcpy(device_input.get(), input.data(),
                        input.size() * sizeof(float), cudaMemcpyHostToDevice),
             "copying NVFP4 input to GPU failed");
  global_amax_partial_kernel<<<reduction_blocks, kReductionThreads>>>(
      device_input.get(), input.size(), device_partial.get());
  check_cuda(cudaGetLastError(), "launching NVFP4 partial amax kernel failed");
  global_scale_kernel<<<1, kReductionThreads>>>(
      device_partial.get(), reduction_blocks, device_global.get());
  check_cuda(cudaGetLastError(), "launching NVFP4 global scale kernel failed");
  check_cuda(cudaMemcpy(&result.global_scale, device_global.get(), sizeof(float),
                        cudaMemcpyDeviceToHost),
             "copying NVFP4 global scale to CPU failed");
  if (result.global_scale == 0.0F) {
    throw std::underflow_error("NVFP4 CUDA global scale underflows to zero");
  }

  nvfp4_quantize_kernel<<<static_cast<unsigned>(blocks),
                           static_cast<unsigned>(kQuantBlockSize)>>>(
      device_input.get(), device_data.get(), device_scales.get(),
      device_global.get(), input.size());
  check_cuda(cudaGetLastError(), "launching NVFP4 quantize kernel failed");
  check_cuda(cudaMemcpy(result.packed_data.data(), device_data.get(),
                        result.packed_data.size(), cudaMemcpyDeviceToHost),
             "copying NVFP4 packed data to CPU failed");
  check_cuda(cudaMemcpy(result.block_scales.data(), device_scales.get(),
                        result.block_scales.size(), cudaMemcpyDeviceToHost),
             "copying NVFP4 block scales to CPU failed");
  return result;
}

std::vector<float> nvfp4_dequantize_cuda(const NVFP4Result& q) {
  validate_quantized_input(q);
  DeviceBuffer<std::uint8_t> device_data(q.packed_data.size());
  DeviceBuffer<std::uint8_t> device_scales(q.block_scales.size());
  DeviceBuffer<float> device_global(1);
  DeviceBuffer<float> device_output(q.element_count);
  check_cuda(cudaMemcpy(device_data.get(), q.packed_data.data(), q.packed_data.size(),
                        cudaMemcpyHostToDevice),
             "copying NVFP4 packed data to GPU failed");
  check_cuda(cudaMemcpy(device_scales.get(), q.block_scales.data(),
                        q.block_scales.size(), cudaMemcpyHostToDevice),
             "copying NVFP4 block scales to GPU failed");
  check_cuda(cudaMemcpy(device_global.get(), &q.global_scale, sizeof(float),
                        cudaMemcpyHostToDevice),
             "copying NVFP4 global scale to GPU failed");

  const std::size_t grid_size = q.element_count / kDequantThreads
                              + (q.element_count % kDequantThreads != 0);
  if (grid_size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::length_error("NVFP4 CUDA input is too large for a 1D grid");
  }
  nvfp4_dequantize_kernel<<<static_cast<unsigned>(grid_size), kDequantThreads>>>(
      device_data.get(), device_scales.get(), device_global.get(),
      device_output.get(), q.element_count);
  check_cuda(cudaGetLastError(), "launching NVFP4 dequantize kernel failed");

  std::vector<float> output(q.element_count);
  check_cuda(cudaMemcpy(output.data(), device_output.get(),
                        output.size() * sizeof(float), cudaMemcpyDeviceToHost),
             "copying NVFP4 output to CPU failed");
  for (float value : output) {
    if (!std::isfinite(value)) {
      throw std::overflow_error("NVFP4 CUDA reconstruction exceeds the FP32 range");
    }
  }
  return output;
}

template <typename DeviceInput>
NVFP4BenchmarkResult benchmark_nvfp4_cuda_impl(
    const void* host_input, std::size_t host_input_bytes,
    const std::vector<float>& input, unsigned warmup, unsigned repeats) {
  validate_source(input);
  NVFP4BenchmarkResult result;
  result.quantized.element_count = input.size();
  const std::size_t quant_blocks = quant_block_count(input.size());
  const std::size_t dequant_blocks = input.size() / kDequantThreads
      + (input.size() % kDequantThreads != 0);
  if (quant_blocks > static_cast<std::size_t>(std::numeric_limits<int>::max())
      || dequant_blocks > static_cast<std::size_t>(
          std::numeric_limits<int>::max())) {
    throw std::length_error("NVFP4 benchmark input is too large for a 1D grid");
  }
  result.quantized.block_scales.resize(quant_blocks);
  result.quantized.packed_data.resize(quant_blocks * kBytesPerQuantBlock);
  result.restored.resize(input.size());

  DeviceBuffer<DeviceInput> device_input(input.size());
  DeviceBuffer<std::uint8_t> device_data(result.quantized.packed_data.size());
  DeviceBuffer<std::uint8_t> device_scales(quant_blocks);
  DeviceBuffer<float> device_global(1);
  DeviceBuffer<float> device_output(input.size());
  const std::size_t blocks_needed = input.size() / kReductionThreads
      + (input.size() % kReductionThreads != 0);
  const unsigned reduction_blocks = static_cast<unsigned>(
      std::min<std::size_t>(blocks_needed, kMaxReductionBlocks));
  DeviceBuffer<float> device_partial(reduction_blocks);
  check_cuda(cudaMemcpy(device_input.get(), host_input, host_input_bytes,
                        cudaMemcpyHostToDevice),
             "copying NVFP4 benchmark input to GPU failed");

  const auto launch_quantize = [&] {
    global_amax_partial_kernel<<<reduction_blocks, kReductionThreads>>>(
        device_input.get(), input.size(), device_partial.get());
    check_cuda(cudaGetLastError(), "launching benchmark NVFP4 partial amax failed");
    global_scale_kernel<<<1, kReductionThreads>>>(
        device_partial.get(), reduction_blocks, device_global.get());
    check_cuda(cudaGetLastError(), "launching benchmark NVFP4 global scale failed");
    nvfp4_quantize_kernel<<<static_cast<unsigned>(quant_blocks),
                             static_cast<unsigned>(kQuantBlockSize)>>>(
        device_input.get(), device_data.get(), device_scales.get(),
        device_global.get(), input.size());
    check_cuda(cudaGetLastError(), "launching benchmark NVFP4 quantize failed");
  };
  const auto launch_dequantize = [&] {
    nvfp4_dequantize_kernel<<<static_cast<unsigned>(dequant_blocks),
                               kDequantThreads>>>(
        device_data.get(), device_scales.get(), device_global.get(),
        device_output.get(), input.size());
    check_cuda(cudaGetLastError(), "launching benchmark NVFP4 dequantize failed");
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
  check_cuda(cudaDeviceSynchronize(), "preparing NVFP4 dequantize benchmark failed");
  result.timings.dequantize = benchmark_detail::measure_cuda_events(
      warmup, repeats, launch_dequantize);
  result.timings.device_pipeline = benchmark_detail::measure_cuda_events(
      warmup, repeats, launch_pipeline);

  const auto run_end_to_end = [&] {
    check_cuda(cudaMemcpy(device_input.get(), host_input, host_input_bytes,
                          cudaMemcpyHostToDevice),
               "NVFP4 benchmark H2D failed");
    launch_pipeline();
    check_cuda(cudaMemcpy(result.restored.data(), device_output.get(),
                          result.restored.size() * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "NVFP4 benchmark D2H failed");
  };
  result.timings.end_to_end = benchmark_detail::measure_host(
      warmup, repeats, run_end_to_end);

  check_cuda(cudaMemcpy(result.quantized.packed_data.data(), device_data.get(),
                        result.quantized.packed_data.size(), cudaMemcpyDeviceToHost),
             "copying benchmark NVFP4 data to CPU failed");
  check_cuda(cudaMemcpy(result.quantized.block_scales.data(), device_scales.get(),
                        result.quantized.block_scales.size(), cudaMemcpyDeviceToHost),
             "copying benchmark NVFP4 scales to CPU failed");
  check_cuda(cudaMemcpy(&result.quantized.global_scale, device_global.get(),
                        sizeof(float), cudaMemcpyDeviceToHost),
             "copying benchmark NVFP4 global scale to CPU failed");
  if (result.quantized.global_scale == 0.0F) {
    throw std::underflow_error("NVFP4 benchmark global scale underflows to zero");
  }
  for (float value : result.restored) {
    if (!std::isfinite(value)) {
      throw std::overflow_error("NVFP4 benchmark reconstruction exceeds FP32");
    }
  }

  const std::size_t quantize_bytes = host_input_bytes * 2
      + result.quantized.packed_data.size() + result.quantized.block_scales.size()
      + static_cast<std::size_t>(reduction_blocks) * sizeof(float) * 2
      + sizeof(float);
  const std::size_t dequantize_bytes = result.quantized.packed_data.size()
      + result.quantized.block_scales.size() + sizeof(float)
      + input.size() * sizeof(float);
  result.timings.quantize_effective_gbps = benchmark_detail::effective_gbps(
      quantize_bytes, result.timings.quantize.median_ms);
  result.timings.dequantize_effective_gbps = benchmark_detail::effective_gbps(
      dequantize_bytes, result.timings.dequantize.median_ms);
  return result;
}

NVFP4BenchmarkResult benchmark_nvfp4_cuda(
    const std::vector<float>& input, unsigned warmup, unsigned repeats) {
  return benchmark_nvfp4_cuda_impl<float>(
      input.data(), input.size() * sizeof(float), input, warmup, repeats);
}

NVFP4BenchmarkResult benchmark_nvfp4_cuda_fp16(
    const std::vector<std::uint16_t>& input_bits,
    const std::vector<float>& decoded_input,
    unsigned warmup, unsigned repeats) {
  if (input_bits.size() != decoded_input.size()) {
    throw std::invalid_argument("NVFP4 FP16 bit/value counts differ");
  }
  return benchmark_nvfp4_cuda_impl<__half>(
      input_bits.data(), input_bits.size() * sizeof(std::uint16_t),
      decoded_input, warmup, repeats);
}

}  // namespace luxorion

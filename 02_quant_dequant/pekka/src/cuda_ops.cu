#include "qd/cuda_ops.hpp"

#include "qd/formats.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace qd {
namespace {

void check_cuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count != 0) check_cuda(cudaMalloc(&pointer_, count * sizeof(T)), "cudaMalloc");
  }
  ~DeviceBuffer() { if (pointer_ != nullptr) cudaFree(pointer_); }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  T *get() { return pointer_; }
  const T *get() const { return pointer_; }
  std::size_t bytes() const { return count_ * sizeof(T); }
private:
  T *pointer_ = nullptr;
  std::size_t count_ = 0;
};

class Event {
public:
  Event() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
  ~Event() { cudaEventDestroy(event_); }
  cudaEvent_t get() const { return event_; }
private:
  cudaEvent_t event_{};
};

__device__ float finite_amax_range(const float *values, std::size_t begin,
                                   std::size_t end) {
  float maximum = 0.0F;
  for (std::size_t index = begin; index < end; ++index) {
    if (isfinite(values[index])) maximum = fmaxf(maximum, fabsf(values[index]));
  }
  return maximum;
}

// One thread deliberately owns one scale group.  This baseline favors a
// transparent, deterministic reduction; a warp-reduction version can replace
// it later without changing the file or host APIs.
__global__ void mxfp8_scale_kernel(const float *values, std::size_t count,
                                   std::uint32_t block_size, bool tensor_mode,
                                   std::uint8_t *scales, std::size_t groups) {
  const std::size_t group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= groups) return;
  const std::size_t begin = tensor_mode ? 0 : group * block_size;
  const std::size_t end = tensor_mode ? count : min(count, begin + block_size);
  const float amax = finite_amax_range(values, begin, end);
  scales[group] = formats::encode_e8m0_ceil(amax == 0.0F ? 1.0F : amax / 448.0F);
}

// Every thread scans a grid-stride slice and contributes one non-negative
// local maximum.  Positive IEEE-754 floats have the same ordering as their
// unsigned bit patterns, so atomicMax is valid here and avoids a serial tensor
// scan.  NaN/Inf are excluded consistently with the CPU reference.
__global__ void tensor_amax_kernel(const float *values, std::size_t count,
                                   float *global_amax) {
  float local = 0.0F;
  const std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += stride) {
    if (isfinite(values[index])) local = fmaxf(local, fabsf(values[index]));
  }
  atomicMax(reinterpret_cast<unsigned int *>(global_amax), __float_as_uint(local));
}

__global__ void nvfp4_global_scale_kernel(const float *global_amax,
                                          float *global_scale) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *global_scale = *global_amax == 0.0F ? 1.0F : *global_amax / (6.0F * 448.0F);
  }
}

__global__ void nvfp4_local_scale_kernel(const float *values, std::size_t count,
                                         std::uint32_t block_size,
                                         bool tensor_mode,
                                         const float *global_scale,
                                         std::uint8_t *scales,
                                         std::size_t groups) {
  const std::size_t group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= groups) return;
  const std::size_t begin = tensor_mode ? 0 : group * block_size;
  const std::size_t end = tensor_mode ? count : min(count, begin + block_size);
  const float amax = finite_amax_range(values, begin, end);
  const float ideal = amax == 0.0F ? 1.0F : amax / (6.0F * *global_scale);
  scales[group] = formats::encode_e4m3(ideal);
}

__global__ void mxfp8_quantize_kernel(const float *values, std::size_t count,
                                      std::uint32_t block_size, bool tensor_mode,
                                      const std::uint8_t *scales, int rounding,
                                      std::uint64_t seed,
                                      std::uint8_t *packed) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const std::size_t group = tensor_mode ? 0 : index / block_size;
  const float scale = formats::decode_e8m0(scales[group]);
  const float normalized = values[index] / scale;
  packed[index] = rounding == static_cast<int>(Rounding::kStochastic)
                      ? formats::encode_e4m3_stochastic(normalized, seed, index)
                      : formats::encode_e4m3(normalized);
}

// A thread owns a complete byte, so two FP4 elements can never race on a
// read-modify-write.  For odd tensors the unused high nibble remains zero.
__global__ void nvfp4_quantize_kernel(const float *values, std::size_t count,
                                      std::uint32_t block_size, bool tensor_mode,
                                      const std::uint8_t *scales,
                                      const float *global_scale,
                                      int rounding, std::uint64_t seed,
                                      std::uint8_t *packed) {
  const std::size_t byte_index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t first = byte_index * 2U;
  if (first >= count) return;
  const auto encode = [&](std::size_t index) {
    const std::size_t group = tensor_mode ? 0 : index / block_size;
    const float denominator = *global_scale * formats::decode_e4m3(scales[group]);
    const float normalized = values[index] / denominator;
    return rounding == static_cast<int>(Rounding::kStochastic)
               ? formats::encode_e2m1_stochastic(normalized, seed, index)
               : formats::encode_e2m1(normalized);
  };
  std::uint8_t byte = encode(first);
  if (first + 1U < count) byte |= static_cast<std::uint8_t>(encode(first + 1U) << 4U);
  packed[byte_index] = byte;
}

__global__ void dequantize_kernel(const std::uint8_t *packed,
                                  const std::uint8_t *scales,
                                  std::size_t count, std::uint32_t block_size,
                                  bool tensor_mode, int format,
                                  float global_scale, int output_type,
                                  float *fp32_output, std::uint8_t *raw_output) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const std::size_t group = tensor_mode ? 0 : index / block_size;
  float value = 0.0F;
  if (format == static_cast<int>(Format::kMxfp8)) {
    value = formats::decode_e4m3(packed[index]) * formats::decode_e8m0(scales[group]);
  } else {
    const std::uint8_t byte = packed[index / 2U];
    const std::uint8_t code = (index & 1U) == 0 ? byte & 0x0FU : byte >> 4U;
    value = formats::decode_e2m1(code) * formats::decode_e4m3(scales[group]) *
            global_scale;
  }
  fp32_output[index] = value;
  if (output_type == static_cast<int>(DataType::kFp32)) {
    reinterpret_cast<float *>(raw_output)[index] = value;
  } else if (output_type == static_cast<int>(DataType::kFp16)) {
    reinterpret_cast<__half *>(raw_output)[index] = __float2half_rn(value);
  } else {
    reinterpret_cast<__nv_bfloat16 *>(raw_output)[index] = __float2bfloat16_rn(value);
  }
}

float elapsed_average(const Event &start, const Event &stop, int iterations) {
  check_cuda(cudaEventSynchronize(stop.get()), "cudaEventSynchronize");
  float milliseconds = 0.0F;
  check_cuda(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()),
             "cudaEventElapsedTime");
  return milliseconds / iterations;
}

std::size_t groups_for(std::size_t count, const Config &config) {
  return config.scale_mode == ScaleMode::kTensor
             ? 1U
             : (count + config.block_size - 1U) / config.block_size;
}

} // namespace

GpuQuantizeResult quantize_gpu(const Matrix &matrix, const Config &config,
                               int iterations) {
  if (iterations <= 0) throw std::runtime_error("iterations must be positive");
  const std::size_t count = matrix.values.size();
  const std::size_t groups = groups_for(count, config);
  const std::size_t packed_count = config.format == Format::kMxfp8 ? count : (count + 1U) / 2U;
  DeviceBuffer<float> values(count);
  DeviceBuffer<std::uint8_t> scales(groups);
  DeviceBuffer<std::uint8_t> packed(packed_count);
  DeviceBuffer<float> global_amax(1);
  DeviceBuffer<float> global_scale(1);
  check_cuda(cudaMemcpy(values.get(), matrix.values.data(), values.bytes(),
                        cudaMemcpyHostToDevice), "copy input to GPU");

  constexpr int threads = 256;
  const int group_blocks = static_cast<int>((groups + threads - 1U) / threads);
  const int value_blocks = static_cast<int>((count + threads - 1U) / threads);
  const int packed_blocks = static_cast<int>((packed_count + threads - 1U) / threads);
  const int reduction_blocks = std::min(value_blocks, 1024);
  const bool tensor_mode = config.scale_mode == ScaleMode::kTensor;
  Event start;
  Event stop;
  check_cuda(cudaEventRecord(start.get()), "record quantize start");
  for (int iteration = 0; iteration < iterations; ++iteration) {
    if (config.format == Format::kMxfp8) {
      mxfp8_scale_kernel<<<group_blocks, threads>>>(
          values.get(), count, config.block_size, tensor_mode, scales.get(), groups);
      mxfp8_quantize_kernel<<<value_blocks, threads>>>(
          values.get(), count, config.block_size, tensor_mode, scales.get(),
          static_cast<int>(config.rounding), config.seed, packed.get());
    } else {
      check_cuda(cudaMemsetAsync(global_amax.get(), 0, sizeof(float)),
                 "clear global amax");
      tensor_amax_kernel<<<reduction_blocks, threads>>>(values.get(), count,
                                                        global_amax.get());
      nvfp4_global_scale_kernel<<<1, 1>>>(global_amax.get(), global_scale.get());
      nvfp4_local_scale_kernel<<<group_blocks, threads>>>(
          values.get(), count, config.block_size, tensor_mode, global_scale.get(),
          scales.get(), groups);
      nvfp4_quantize_kernel<<<packed_blocks, threads>>>(
          values.get(), count, config.block_size, tensor_mode, scales.get(),
          global_scale.get(), static_cast<int>(config.rounding), config.seed,
          packed.get());
    }
  }
  check_cuda(cudaGetLastError(), "quantize kernel launch");
  check_cuda(cudaEventRecord(stop.get()), "record quantize stop");

  QuantizedData data{config.format, matrix.original_type, config.scale_mode,
                     config.rounding, matrix.rows, matrix.cols, config.block_size,
                     std::vector<std::uint8_t>(packed_count),
                     std::vector<std::uint8_t>(groups), 1.0F};
  check_cuda(cudaMemcpy(data.packed_values.data(), packed.get(), packed.bytes(),
                        cudaMemcpyDeviceToHost), "copy packed values from GPU");
  check_cuda(cudaMemcpy(data.local_scales.data(), scales.get(), scales.bytes(),
                        cudaMemcpyDeviceToHost), "copy scales from GPU");
  if (config.format == Format::kNvfp4) {
    check_cuda(cudaMemcpy(&data.global_scale, global_scale.get(), sizeof(float),
                          cudaMemcpyDeviceToHost), "copy global scale from GPU");
  }
  return {std::move(data), elapsed_average(start, stop, iterations)};
}

GpuDequantizeResult dequantize_gpu(const QuantizedData &data,
                                   DataType output_type, int iterations) {
  if (iterations <= 0) throw std::runtime_error("iterations must be positive");
  const std::size_t count = element_count(data.rows, data.cols);
  const std::size_t raw_bytes = count * data_type_size(output_type);
  DeviceBuffer<std::uint8_t> packed(data.packed_values.size());
  DeviceBuffer<std::uint8_t> scales(data.local_scales.size());
  DeviceBuffer<float> fp32_output(count);
  DeviceBuffer<std::uint8_t> raw_output(raw_bytes);
  check_cuda(cudaMemcpy(packed.get(), data.packed_values.data(), packed.bytes(),
                        cudaMemcpyHostToDevice), "copy packed values to GPU");
  check_cuda(cudaMemcpy(scales.get(), data.local_scales.data(), scales.bytes(),
                        cudaMemcpyHostToDevice), "copy scales to GPU");

  constexpr int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1U) / threads);
  Event start;
  Event stop;
  check_cuda(cudaEventRecord(start.get()), "record dequantize start");
  for (int iteration = 0; iteration < iterations; ++iteration) {
    dequantize_kernel<<<blocks, threads>>>(
        packed.get(), scales.get(), count, data.block_size,
        data.scale_mode == ScaleMode::kTensor, static_cast<int>(data.format),
        data.global_scale, static_cast<int>(output_type), fp32_output.get(),
        raw_output.get());
  }
  check_cuda(cudaGetLastError(), "dequantize kernel launch");
  check_cuda(cudaEventRecord(stop.get()), "record dequantize stop");

  GpuDequantizeResult result;
  result.fp32_values.resize(count);
  result.output_payload.resize(raw_bytes);
  check_cuda(cudaMemcpy(result.fp32_values.data(), fp32_output.get(), fp32_output.bytes(),
                        cudaMemcpyDeviceToHost), "copy float output from GPU");
  check_cuda(cudaMemcpy(result.output_payload.data(), raw_output.get(), raw_output.bytes(),
                        cudaMemcpyDeviceToHost), "copy typed output from GPU");
  result.milliseconds = elapsed_average(start, stop, iterations);
  return result;
}

} // namespace qd

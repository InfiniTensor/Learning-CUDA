#include "low_precision/cuda_quant.hpp"

#include "low_precision/ascend_runtime.hpp"

#include "low_precision_ascend_kernels/aclrtlaunch_mxfp8_dequant.h"
#include "low_precision_ascend_kernels/aclrtlaunch_mxfp8_quant.h"
#include "low_precision_ascend_kernels/aclrtlaunch_nvfp4_block_scales.h"
#include "low_precision_ascend_kernels/aclrtlaunch_nvfp4_dequant.h"
#include "low_precision_ascend_kernels/aclrtlaunch_nvfp4_global_scale.h"
#include "low_precision_ascend_kernels/aclrtlaunch_nvfp4_quant.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace low_precision {
namespace {

constexpr uint32_t kLaunchCores = 1;

void check(aclError status, const char* operation) {
    if (status != ACL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": " + ascend::error_string(status));
    }
}

void check_launch(uint32_t status, const char* operation) {
    check(static_cast<aclError>(status), operation);
}

void check_shape(std::size_t rows, std::size_t cols) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("matrix dimensions must be nonzero");
    }
}

uint64_t blocks_per_row(std::size_t cols, uint64_t block_size) {
    return (static_cast<uint64_t>(cols) + block_size - 1u) / block_size;
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        check(cudaMalloc(&pointer_, count * sizeof(T)), "Ascend device allocation");
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() { return pointer_; }

private:
    T* pointer_ = nullptr;
};

void synchronize(const char* operation) {
    check(cudaDeviceSynchronize(), operation);
}

void quantize_mxfp8(const void* input,
                    std::uint8_t* values,
                    std::uint8_t* scales,
                    std::size_t rows,
                    std::size_t cols,
                    uint32_t input_type,
                    Rounding rounding,
                    std::uint64_t seed) {
    check_shape(rows, cols);
    check(ascend::initialize(), "Ascend runtime initialization");
    check_launch(ACLRT_LAUNCH_KERNEL(mxfp8_quant)(
                     kLaunchCores, ascend::stream(), const_cast<void*>(input), values, scales,
                     static_cast<uint64_t>(rows), static_cast<uint64_t>(cols), input_type,
                     rounding == Rounding::Stochastic ? 1u : 0u, seed),
                 "MXFP8 quantize launch");
    synchronize("MXFP8 quantize");
}

void dequantize_mxfp8(const std::uint8_t* values,
                      const std::uint8_t* scales,
                      void* output,
                      std::size_t rows,
                      std::size_t cols,
                      uint32_t output_type) {
    check_shape(rows, cols);
    check(ascend::initialize(), "Ascend runtime initialization");
    check_launch(ACLRT_LAUNCH_KERNEL(mxfp8_dequant)(
                     kLaunchCores, ascend::stream(), const_cast<std::uint8_t*>(values),
                     const_cast<std::uint8_t*>(scales), output, static_cast<uint64_t>(rows),
                     static_cast<uint64_t>(cols), output_type),
                 "MXFP8 dequantize launch");
    synchronize("MXFP8 dequantize");
}

void quantize_nvfp4(const void* input,
                    std::uint8_t* values,
                    std::uint8_t* block_scales,
                    std::size_t rows,
                    std::size_t cols,
                    float* global_scale,
                    uint32_t input_type,
                    Rounding rounding,
                    std::uint64_t seed) {
    check_shape(rows, cols);
    check(ascend::initialize(), "Ascend runtime initialization");
    DeviceBuffer<float> device_global_scale(1);
    const uint64_t elements = static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols);

    check_launch(ACLRT_LAUNCH_KERNEL(nvfp4_global_scale)(
                     1u, ascend::stream(), const_cast<void*>(input), device_global_scale.get(),
                     elements, input_type),
                 "NVFP4 global scale launch");
    check_launch(ACLRT_LAUNCH_KERNEL(nvfp4_block_scales)(
                     kLaunchCores, ascend::stream(), const_cast<void*>(input),
                     device_global_scale.get(), block_scales, static_cast<uint64_t>(rows),
                     static_cast<uint64_t>(cols), input_type,
                     rounding == Rounding::Stochastic ? 1u : 0u, seed),
                 "NVFP4 block scale launch");
    check_launch(ACLRT_LAUNCH_KERNEL(nvfp4_quant)(
                     kLaunchCores, ascend::stream(), const_cast<void*>(input),
                     device_global_scale.get(), block_scales, values, static_cast<uint64_t>(rows),
                     static_cast<uint64_t>(cols), input_type,
                     rounding == Rounding::Stochastic ? 1u : 0u, seed),
                 "NVFP4 quantize launch");
    synchronize("NVFP4 quantize");
    if (global_scale != nullptr) {
        check(cudaMemcpy(global_scale, device_global_scale.get(), sizeof(float),
                         cudaMemcpyDeviceToHost),
              "NVFP4 global scale copy");
    }
}

void dequantize_nvfp4(const std::uint8_t* values,
                      const std::uint8_t* block_scales,
                      float global_scale,
                      void* output,
                      std::size_t rows,
                      std::size_t cols,
                      uint32_t output_type) {
    check_shape(rows, cols);
    check(ascend::initialize(), "Ascend runtime initialization");
    check_launch(ACLRT_LAUNCH_KERNEL(nvfp4_dequant)(
                     kLaunchCores, ascend::stream(), const_cast<std::uint8_t*>(values),
                     const_cast<std::uint8_t*>(block_scales), output, global_scale,
                     static_cast<uint64_t>(rows), static_cast<uint64_t>(cols), output_type),
                 "NVFP4 dequantize launch");
    synchronize("NVFP4 dequantize");
}

}  // namespace

void cuda_quantize_mxfp8(const float* input,
                         std::uint8_t* values,
                         std::uint8_t* scales,
                         std::size_t rows,
                         std::size_t cols,
                         Rounding rounding,
                         std::uint64_t seed) {
    quantize_mxfp8(input, values, scales, rows, cols, 0u, rounding, seed);
}

void cuda_quantize_mxfp8(const Fp16* input,
                         std::uint8_t* values,
                         std::uint8_t* scales,
                         std::size_t rows,
                         std::size_t cols,
                         Rounding rounding,
                         std::uint64_t seed) {
    quantize_mxfp8(input, values, scales, rows, cols, 1u, rounding, seed);
}

void cuda_dequantize_mxfp8(const std::uint8_t* values,
                           const std::uint8_t* scales,
                           float* output,
                           std::size_t rows,
                           std::size_t cols) {
    dequantize_mxfp8(values, scales, output, rows, cols, 0u);
}

void cuda_dequantize_mxfp8(const std::uint8_t* values,
                           const std::uint8_t* scales,
                           Fp16* output,
                           std::size_t rows,
                           std::size_t cols) {
    dequantize_mxfp8(values, scales, output, rows, cols, 1u);
}

void cuda_dequantize_mxfp8(const std::uint8_t* values,
                           const std::uint8_t* scales,
                           Bf16* output,
                           std::size_t rows,
                           std::size_t cols) {
    dequantize_mxfp8(values, scales, output, rows, cols, 2u);
}

void cuda_quantize_nvfp4(const float* input,
                         std::uint8_t* values,
                         std::uint8_t* block_scales,
                         std::size_t rows,
                         std::size_t cols,
                         float* global_scale,
                         Rounding rounding,
                         std::uint64_t seed) {
    quantize_nvfp4(input, values, block_scales, rows, cols, global_scale, 0u, rounding, seed);
}

void cuda_quantize_nvfp4(const Fp16* input,
                         std::uint8_t* values,
                         std::uint8_t* block_scales,
                         std::size_t rows,
                         std::size_t cols,
                         float* global_scale,
                         Rounding rounding,
                         std::uint64_t seed) {
    quantize_nvfp4(input, values, block_scales, rows, cols, global_scale, 1u, rounding, seed);
}

void cuda_dequantize_nvfp4(const std::uint8_t* values,
                           const std::uint8_t* block_scales,
                           float global_scale,
                           float* output,
                           std::size_t rows,
                           std::size_t cols) {
    dequantize_nvfp4(values, block_scales, global_scale, output, rows, cols, 0u);
}

void cuda_dequantize_nvfp4(const std::uint8_t* values,
                           const std::uint8_t* block_scales,
                           float global_scale,
                           Fp16* output,
                           std::size_t rows,
                           std::size_t cols) {
    dequantize_nvfp4(values, block_scales, global_scale, output, rows, cols, 1u);
}

void cuda_dequantize_nvfp4(const std::uint8_t* values,
                           const std::uint8_t* block_scales,
                           float global_scale,
                           Bf16* output,
                           std::size_t rows,
                           std::size_t cols) {
    dequantize_nvfp4(values, block_scales, global_scale, output, rows, cols, 2u);
}

}  // namespace low_precision

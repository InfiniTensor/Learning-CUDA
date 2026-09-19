#include "hadamard_api.h"
#include "aclrtlaunch_hadamard_transform_fp16.h"
#include "aclrtlaunch_hadamard_transform_bf16.h"
#include "aclrtlaunch_hadamard_quantize_fp16.h"
#include "aclrtlaunch_hadamard_quantize_bf16.h"
#include "aclrtlaunch_hadamard_fused_fp16.h"
#include "aclrtlaunch_hadamard_fused_bf16.h"

#include <cmath>
#include <limits>

namespace hadamard::ascend {
namespace {

bool valid_scalars(StorageType storage, Method method, std::uint32_t n,
                   float scale, std::uint32_t blocks) {
    return (storage == StorageType::FP16 || storage == StorageType::BF16)
        && (method == Method::ScalarButterfly || method == Method::VectorGather)
        && n >= 1 && n <= 256 && (n & (n - 1)) == 0
        && std::isfinite(scale) && scale > 0.0f && blocks >= 1 && blocks <= 32;
}

bool valid_range(const void* pointer, std::size_t bytes, std::size_t alignment) {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    return pointer && address % alignment == 0
        && bytes <= std::numeric_limits<std::uintptr_t>::max() - address;
}

bool overlap(const void* lhs, std::size_t lhs_bytes, const void* rhs, std::size_t rhs_bytes) {
    const auto a = reinterpret_cast<std::uintptr_t>(lhs);
    const auto b = reinterpret_cast<std::uintptr_t>(rhs);
    return a < b + rhs_bytes && b < a + lhs_bytes;
}

bool valid_buffers(const std::uint16_t* input, std::uint16_t* output,
                   std::uint8_t* packed, float* scales, std::size_t rows,
                   std::uint32_t n, bool quantize) {
    constexpr auto maximum = std::numeric_limits<std::size_t>::max();
    const std::size_t row_bytes = std::size_t(n) * sizeof(std::uint16_t);
    if (rows > maximum / row_bytes) return false;
    const std::size_t input_bytes = rows * row_bytes;
    if (!valid_range(input, input_bytes, alignof(std::uint16_t))) return false;
    if (!quantize) {
        return valid_range(output, input_bytes, alignof(std::uint16_t))
            && (input == output || !overlap(input, input_bytes, output, input_bytes));
    }
    if (rows > maximum / sizeof(float)) return false;
    const std::size_t packed_bytes = rows * ((n + 1) / 2);
    const std::size_t scale_bytes = rows * sizeof(float);
    return valid_range(packed, packed_bytes, alignof(std::uint8_t))
        && valid_range(scales, scale_bytes, alignof(float))
        && !overlap(input, input_bytes, packed, packed_bytes)
        && !overlap(input, input_bytes, scales, scale_bytes)
        && !overlap(packed, packed_bytes, scales, scale_bytes);
}

std::uint8_t* raw(const void* pointer) {
    return reinterpret_cast<std::uint8_t*>(const_cast<void*>(pointer));
}

}  // namespace

aclError transform(const std::uint16_t* input, std::uint16_t* output,
                   std::size_t rows, std::uint32_t n, float scale,
                   StorageType storage, Method method, aclrtStream stream,
                   std::uint32_t block_dim) {
    if (!valid_scalars(storage, method, n, scale, block_dim)) return ACL_ERROR_INVALID_PARAM;
    if (!rows) return ACL_SUCCESS;
    if (!valid_buffers(input, output, nullptr, nullptr, rows, n, false)) return ACL_ERROR_INVALID_PARAM;
    const auto count = static_cast<std::uint64_t>(rows);
    const auto implementation = static_cast<std::uint32_t>(method);
    if (storage == StorageType::FP16)
        return ACLRT_LAUNCH_KERNEL(hadamard_transform_fp16)(block_dim, stream, raw(input), raw(output), count, n, scale, implementation);
    return ACLRT_LAUNCH_KERNEL(hadamard_transform_bf16)(block_dim, stream, raw(input), raw(output), count, n, scale, implementation);
}

aclError quantize_int4(const std::uint16_t* input, std::uint8_t* packed, float* scales,
                       std::size_t rows, std::uint32_t n,
                       StorageType storage, Method method, aclrtStream stream,
                       std::uint32_t block_dim) {
    if (!valid_scalars(storage, method, n, 1.0f, block_dim)) return ACL_ERROR_INVALID_PARAM;
    if (!rows) return ACL_SUCCESS;
    if (!valid_buffers(input, nullptr, packed, scales, rows, n, true)) return ACL_ERROR_INVALID_PARAM;
    const auto count = static_cast<std::uint64_t>(rows);
    // 两种变换方法共用同一量化 kernel，使分步/融合对照只改变变换与融合方式。
    if (storage == StorageType::FP16)
        return ACLRT_LAUNCH_KERNEL(hadamard_quantize_fp16)(block_dim, stream, raw(input), packed, raw(scales), count, n);
    return ACLRT_LAUNCH_KERNEL(hadamard_quantize_bf16)(block_dim, stream, raw(input), packed, raw(scales), count, n);
}

aclError transform_int4(const std::uint16_t* input, std::uint8_t* packed, float* scales,
                        std::size_t rows, std::uint32_t n, float scale,
                        StorageType storage, Method method, aclrtStream stream,
                        std::uint32_t block_dim) {
    if (!valid_scalars(storage, method, n, scale, block_dim)) return ACL_ERROR_INVALID_PARAM;
    if (!rows) return ACL_SUCCESS;
    if (!valid_buffers(input, nullptr, packed, scales, rows, n, true)) return ACL_ERROR_INVALID_PARAM;
    const auto count = static_cast<std::uint64_t>(rows);
    const auto implementation = static_cast<std::uint32_t>(method);
    if (storage == StorageType::FP16)
        return ACLRT_LAUNCH_KERNEL(hadamard_fused_fp16)(block_dim, stream, raw(input), packed, raw(scales), count, n, scale, implementation);
    return ACLRT_LAUNCH_KERNEL(hadamard_fused_bf16)(block_dim, stream, raw(input), packed, raw(scales), count, n, scale, implementation);
}

}  // namespace hadamard::ascend

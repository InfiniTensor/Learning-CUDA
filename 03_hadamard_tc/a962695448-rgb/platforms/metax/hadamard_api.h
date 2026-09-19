// 算法来源：本项目天数后端固定提交 a387db3332c6f9b01f128dd681848260c9691281。
// 沐曦后端必须独立完成正确性与性能验证，不继承其他平台的验收结果。
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstddef>
#include <cstdint>

namespace hadamard::metax {

enum class Method { Baseline, Optimized, Warp64 };

// Warp64 为沐曦 64-lane warp 专用候选，须使用
// -DHADAMARD_METAX_WARP64 且由 MACA/cu-bridge (__MACACC__) 编译；其他构建的
// 非空操作返回 cudaErrorNotSupported。调用方必须预先确认当前设备 warpSize=64；
// API 不在发射路径中查询设备属性。rows=0 仍遵循下方的无操作约定。

// 连续设备内存：[rows, n]，n 为 1..256 的二次幂。scale 必须有限且为正数。
// 支持 FP16/BF16 存储，内部 FP32；输入应有限，变换后的值应在输出类型范围内。
// 所有操作仅在调用方的 stream 上发射，不分配内存、不复制、不等待。
// rows=0 时只检查 n、scale、method，允许空指针，不发射内核。
// 输入和变换输出只要求 2 字节对齐；scales 要求 float 对齐。
// transform 允许 input==output 原位变换，其余重叠均返回 cudaErrorInvalidValue。
// 不检查分配容量/设备归属；调用方须提供当前设备有效且足够大的设备缓冲区。
// 返回参数/发射错误；异步执行错误由调用方同步 stream 时检查。
cudaError_t transform(const __half* input, __half* output, std::size_t rows,
                      int n, float scale, cudaStream_t stream,
                      Method method = Method::Optimized);
cudaError_t transform(const __nv_bfloat16* input, __nv_bfloat16* output,
                      std::size_t rows, int n, float scale, cudaStream_t stream,
                      Method method = Method::Optimized);

// 每行 ceil(n/2) 字节；偶数元素在低 4 位，奇数在高 4 位，n=1 的高位为 0。
// q=clamp(round_to_nearest_even(x/s), -7, 7)，s=max(abs(x))/7；全零行 s=1。
// scales 为 rows 个 float。所有输入/输出缓冲区必须互不重叠。
cudaError_t quantize_int4(const __half* input, std::uint8_t* packed, float* scales,
                          std::size_t rows, int n, cudaStream_t stream,
                          Method method = Method::Optimized);
cudaError_t quantize_int4(const __nv_bfloat16* input, std::uint8_t* packed,
                          float* scales, std::size_t rows, int n,
                          cudaStream_t stream, Method method = Method::Optimized);

// 融合路径先以最近偶数规则舍入到公开的 FP16/BF16 输出类型，再计算 INT4，
// 因而与 transform 后调用 quantize_int4 的字节和 scales 语义一致。
cudaError_t transform_int4(const __half* input, std::uint8_t* packed, float* scales,
                           std::size_t rows, int n, float scale,
                           cudaStream_t stream, Method method = Method::Optimized);
cudaError_t transform_int4(const __nv_bfloat16* input, std::uint8_t* packed,
                           float* scales, std::size_t rows, int n, float scale,
                           cudaStream_t stream, Method method = Method::Optimized);

}  // namespace hadamard::metax

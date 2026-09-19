// Ported from the verified MetaX implementation at cb82ea6.
// MUSA results require independent validation on Moore Threads hardware.
#pragma once

#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>
#include <cstddef>
#include <cstdint>

namespace hadamard::moore_control {

enum class Method { Baseline, Optimized, Shuffle32 };

// Shuffle32 is an explicit MUSA candidate enabled by HADAMARD_MOORE_SHUFFLE32.
// Select only after the native 32-lane shuffle probe passes. On the tested S4000,
// host device properties report 128, but device-code warpSize is 32.
// The asynchronous API does not query device properties on each launch.
// Nonempty Shuffle32 calls return musaErrorNotSupported when it is not compiled.

// 连续设备内存：[rows, n]，n 为 1..256 的二次幂。scale 必须有限且为正数。
// 支持 FP16/BF16 存储，内部 FP32；输入应有限，变换后的值应在输出类型范围内。
// 所有操作仅在调用方的 stream 上发射，不分配内存、不复制、不等待。
// rows=0 时只检查 n、scale、method，允许空指针，不发射内核。
// 输入和变换输出只要求 2 字节对齐；scales 要求 float 对齐。
// transform 允许 input==output 原位变换，其余重叠均返回 musaErrorInvalidValue。
// 不检查分配容量/设备归属；调用方须提供当前设备有效且足够大的设备缓冲区。
// 返回参数/发射错误；异步执行错误由调用方同步 stream 时检查。
musaError_t transform(const __half* input, __half* output, std::size_t rows,
                      int n, float scale, musaStream_t stream,
                      Method method = Method::Optimized);
musaError_t transform(const __mt_bfloat16* input, __mt_bfloat16* output,
                      std::size_t rows, int n, float scale, musaStream_t stream,
                      Method method = Method::Optimized);

// 每行 ceil(n/2) 字节；偶数元素在低 4 位，奇数在高 4 位，n=1 的高位为 0。
// q=clamp(round_to_nearest_even(x/s), -7, 7)，s=max(abs(x))/7；全零行 s=1。
// scales 为 rows 个 float。所有输入/输出缓冲区必须互不重叠。
musaError_t quantize_int4(const __half* input, std::uint8_t* packed, float* scales,
                          std::size_t rows, int n, musaStream_t stream,
                          Method method = Method::Optimized);
musaError_t quantize_int4(const __mt_bfloat16* input, std::uint8_t* packed,
                          float* scales, std::size_t rows, int n,
                          musaStream_t stream, Method method = Method::Optimized);

// 融合路径先以最近偶数规则舍入到公开的 FP16/BF16 输出类型，再计算 INT4，
// 因而与 transform 后调用 quantize_int4 的字节和 scales 语义一致。
musaError_t transform_int4(const __half* input, std::uint8_t* packed, float* scales,
                           std::size_t rows, int n, float scale,
                           musaStream_t stream, Method method = Method::Optimized);
musaError_t transform_int4(const __mt_bfloat16* input, std::uint8_t* packed,
                           float* scales, std::size_t rows, int n, float scale,
                           musaStream_t stream, Method method = Method::Optimized);

}  // namespace hadamard::moore_control


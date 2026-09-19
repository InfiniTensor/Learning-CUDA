#pragma once

#include <acl/acl.h>
#include <cstddef>
#include <cstdint>

namespace hadamard::ascend {

enum class StorageType : std::uint32_t { FP16 = 0, BF16 = 1 };
enum class Method : std::uint32_t { ScalarButterfly = 0, VectorGather = 1 };

// 当前实现面向 Ascend910B1。两种方法使用同一 block_dim，首版允许 1..32，默认 1。
// 该参数范围仍须由对应发布版本的实机 grid 验证支撑，不代表性能最优值。
// 连续 [rows,n]，n 为 1..256 的二次幂；16 位指针承载 FP16/BF16 原始位模式。
// input/output 至少 2B 对齐，scales 至少 4B 对齐，packed 无额外地址对齐要求。
// 所有计算均在 NPU 完成，只发射到 caller stream，不分配/复制 host 或 GM 张量，不同步。
// rows=0 时验证枚举、n、scale、block_dim 后直接返回成功，允许空指针/空 stream。
// 非空时调用者须保证缓冲区属于当前设备、容量足够、stream 有效且异步执行前不释放。
// 输入及 FP32 中间值应有限，结果须在目标类型可表示范围；API 不同步扫描设备数据。
// transform 允许 input==output 完全原位；其余部分重叠及量化缓冲之间的重叠均拒绝。
// 返回参数/发射状态；完成性及异步执行错误由调用者同步 stream 或事件确认。
aclError transform(const std::uint16_t* input, std::uint16_t* output,
                   std::size_t rows, std::uint32_t n, float scale,
                   StorageType storage, Method method, aclrtStream stream,
                   std::uint32_t block_dim = 1);

// 每行 ceil(n/2) 字节，偶数元素位于低 nibble，n=1 的高 nibble 为 0。
// s=max(abs(x))/7（全零行 s=1），q=clamp(RNE(x/s),-7,7)，scales 为 rows 个 float。
// NPU 标量 / 遵循本项目数值探针；不使用已发现中点误差的矢量 Div/近似倒数。
aclError quantize_int4(const std::uint16_t* input, std::uint8_t* packed, float* scales,
                       std::size_t rows, std::uint32_t n,
                       StorageType storage, Method method, aclrtStream stream,
                       std::uint32_t block_dim = 1);

// 融合路径先 RNE 到公开 FP16/BF16 类型，再读回 FP32 量化，保持分步语义。
aclError transform_int4(const std::uint16_t* input, std::uint8_t* packed, float* scales,
                        std::size_t rows, std::uint32_t n, float scale,
                        StorageType storage, Method method, aclrtStream stream,
                        std::uint32_t block_dim = 1);

}  // namespace hadamard::ascend

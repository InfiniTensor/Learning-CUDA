#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace hadamard {

// 输入和输出使用两字节存储类型；实际蝶形计算在 kernel 内提升到 FP32。
enum class DataType { kFloat16, kBFloat16 };

// 量化结果的存储格式。kNone 表示只执行 Hadamard 变换。
enum class QuantType { kNone, kFp8E4M3, kInt8, kInt4 };

struct Shape {
  int batch;
  int sequence;
  int heads;
  int head_dim;

  // 把 [batch, sequence, heads] 展平为独立处理的行数。
  std::size_t rows() const {
    return static_cast<std::size_t>(batch) * sequence * heads;
  }

  // 连续张量中的总元素数；最后一维必须是连续的 head_dim。
  std::size_t elements() const { return rows() * head_dim; }
};

// 所有 launch 都在 stream 上异步提交。变换乘以 1/sqrt(head_dim) 做归一化，
// 因而是正交变换。sign_seed == 0 表示不启用随机符号；非零时先对每个列
// 乘以一个由 seed 确定的 ±1，再执行 Hadamard 变换。
cudaError_t launch_fast(const void* input, void* output, std::size_t rows,
                        int head_dim, DataType dtype, std::uint32_t sign_seed,
                        cudaStream_t stream);

// 共享内存逐 stage 同步版本，仅作为正确性和性能基线。
cudaError_t launch_baseline(const void* input, void* output, std::size_t rows,
                            int head_dim, DataType dtype,
                            std::uint32_t sign_seed, cudaStream_t stream);

// 融合 Hadamard 和逐行动态量化，不写回中间的 FP16/BF16 结果。
// FP8 使用 E4M3（有限最大值 448）；INT4 使用有符号二补码 nibble 打包：
// 维度 2j 放低 4 位，维度 2j+1 放高 4 位。scales 每行保存一个 FP32 scale。
cudaError_t launch_fused_quant(const void* input, void* quantized,
                               float* scales, std::size_t rows, int head_dim,
                               DataType dtype, QuantType quant,
                               std::uint32_t sign_seed, cudaStream_t stream);

// 独立的逐行量化 kernel，用于验证“先 Hadamard、再量化”与融合路径的语义一致性。
cudaError_t launch_quantize(const void* transformed, void* quantized,
                            float* scales, std::size_t rows, int head_dim,
                            DataType dtype, QuantType quant,
                            cudaStream_t stream);

const char* dtype_name(DataType dtype);
const char* quant_name(QuantType quant);

}  // namespace hadamard

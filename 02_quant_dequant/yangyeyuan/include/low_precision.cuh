#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace lpq {

// 文件格式编号与 host 端配置一一对应；不要随意调整已有编号，否则会破坏
// quantized.lpq 的兼容性。
enum class Format : std::uint32_t { kMxfp8 = 1, kNvfp4 = 2 };
enum class InputType : std::uint32_t { kFp16 = 1, kFp32 = 2 };
enum class OutputType : std::uint32_t { kFp16 = 1, kBf16 = 2, kFp32 = 3 };
enum class ScaleMode : std::uint32_t { kTensor = 1, kBlock = 2 };
enum class Rounding : std::uint32_t { kNearest = 1, kStochastic = 2 };

struct QuantConfig {
  Format format = Format::kMxfp8;
  ScaleMode scale_mode = ScaleMode::kBlock;
  Rounding rounding = Rounding::kNearest;
  // 最后一个不足 block_size 的分组也按实际存在的元素计算 amax。
  int block_size = 32;
  // 随机舍入使用 seed 和元素线性下标生成确定性随机数。
  std::uint64_t stochastic_seed = 2026;
};

// 返回低精度元素本身占用的字节数。MXFP8 每个元素 1 字节；NVFP4 每两个元素
// 共用 1 字节，元素 2*i 放低 nibble，元素 2*i+1 放高 nibble。
std::size_t packed_bytes(std::size_t num_elements, Format format);

// 返回 scale byte 的数量。tensor 模式始终只有一个局部 scale，block 模式按
// ceil(num_elements / block_size) 分配。
std::size_t scale_count(std::size_t num_elements, const QuantConfig& config);

// 返回反量化输出中每个元素的字节数：FP16/BF16 为 2，FP32 为 4。
std::size_t output_type_size(OutputType type);

// scratch_amax 是 tensor-wide 归约使用的单个 device float；调用方需要为它分配
// 临时空间。scales 对 MXFP8 保存原始 E8M0 byte，对 NVFP4 保存原始 E4M3 byte。
// global_scale 对 MXFP8 写入 1，对 NVFP4 写入第二级 FP32 scale。
// 所有函数只把 kernel 排入 stream，不主动同步；异步错误由调用方检查。
cudaError_t launch_quantize(const void* input, InputType input_type,
                            std::uint8_t* packed, std::uint8_t* scales,
                            float* global_scale, float* scratch_amax,
                            std::size_t num_elements,
                            const QuantConfig& config, cudaStream_t stream);

cudaError_t launch_dequantize(const std::uint8_t* packed,
                              const std::uint8_t* scales,
                              const float* global_scale, void* output,
                              OutputType output_type,
                              std::size_t num_elements,
                              const QuantConfig& config,
                              cudaStream_t stream);

// 这些 host/device 辅助函数用于逐 byte 精确测试和文件检查工具。NaN 编码解码为
// NaN；有限值溢出时量化器饱和到对应格式的最大有限值。
__host__ __device__ float decode_e4m3(std::uint8_t bits);
__host__ __device__ float decode_e8m0(std::uint8_t bits);
__host__ __device__ float decode_e2m1(std::uint8_t bits);

#ifdef LPQ_WITH_HW_PATH
// 仅供硬件对照路径（WITH_HW_PATH）单独计时软件 MXFP8 元素编码 kernel 使用；
// 默认构建不包含此符号，launch_quantize 的行为不受影响。
cudaError_t launch_mx_element_quantize(const float* input, std::uint8_t* packed,
                                       const std::uint8_t* scales,
                                       std::size_t num_elements, int block_size,
                                       bool tensor_mode, cudaStream_t stream);
#endif

}  // namespace lpq

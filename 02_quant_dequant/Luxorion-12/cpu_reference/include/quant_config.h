#pragma once

#include "formats.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace luxorion {

enum class ScaleMode { Tensor, Block };
enum class OutputType { FP16, BF16, FP32 };
enum class RoundingMode { Nearest, Stochastic };

struct QuantConfig {
  QuantFormat format = QuantFormat::Mxfp8;
  std::size_t block_size = 32;
  ScaleMode scale_mode = ScaleMode::Block;
  OutputType output_type = OutputType::FP32;
  RoundingMode rounding = RoundingMode::Nearest;
  std::string target_gpu;
  std::uint32_t seed = 1234;
};

// 严格解析题面定义的量化参数文件；缺失、重复、未知字段或非法取值均抛异常。
QuantConfig read_quant_config(const std::string &path);

} // namespace luxorion

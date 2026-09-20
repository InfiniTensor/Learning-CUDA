#pragma once

// 固定20字节tensor文件协议：
// [0,8) rows int64 LE；[8,16) cols int64 LE；[16,20) ASCII fp32/fp16；
// data从偏移20开始，按行主序连续存储。读取后统一展开为FP32供CPU/GPU计算。

#include <cstdint>
#include <string>
#include <vector>

namespace luxorion {

enum class TensorDType {
  FP32,
  FP16,
};

struct TensorData {
  std::int64_t num_rows = 0;
  std::int64_t num_cols = 0;
  // 表示文件data区的原始类型；values在内存中始终为FP32。
  TensorDType dtype = TensorDType::FP32;
  std::vector<float> values;
  // FP16文件保留原始binary16位模式，供GPU原生half路径直接H2D；FP32时为空。
  std::vector<std::uint16_t> fp16_bits;
};

// 读取并严格校验完整文件；尺寸、dtype、文件大小或数值非法时抛出异常。
TensorData read_tensor_file(const std::string& path);
// 按dtype写出固定20字节header与行主序data；FP16范围外输入会被拒绝。
void write_tensor_file(const std::string& path, const TensorData& tensor);

}  // namespace luxorion

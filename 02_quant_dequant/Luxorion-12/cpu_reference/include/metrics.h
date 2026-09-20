#pragma once

// 模块接口：统一实验的误差指标和量化负载压缩率，供演示、测试及后续GPU验证使用。
// 调用者提供数值数组或字节数；计算实现见src/metrics.cpp。

#include <cstddef>
#include <vector>

namespace luxorion {

struct Metrics {
  // 最大绝对误差、平均绝对误差、均方误差；均使用double保存。
  double max_abs = 0.0;
  double mae = 0.0;
  double mse = 0.0;
};

// reference 是实际输入数值；数组须非空、等长且有限，否则 invalid_argument。
Metrics calculate_metrics(const std::vector<float>& reference,
                          const std::vector<float>& restored);

struct CompressionStats {
  // 实际输入数据负载，不是vector<float>容器本身占用的内存。
  std::size_t original_bytes = 0;
  // 低精度元素、全部scale以及尾块填充的实际存储字节数，不含header。
  std::size_t quantized_payload_bytes = 0;
  // original_bytes / quantized_payload_bytes；越大表示负载缩小越多。
  double compression_ratio = 0.0;
};

// 调用者统计实际负载：包含填充、scale 和必要的 global scale，不含 header。
// 两个字节数都须大于零；比值小于 1 表示量化负载比原数据更大。
CompressionStats calculate_compression(std::size_t original_bytes,
                                       std::size_t quantized_payload_bytes);

}  // namespace luxorion

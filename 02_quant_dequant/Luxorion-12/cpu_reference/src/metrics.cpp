/*
 * 模块用途：评估一次量化实验的数值损失和存储收益，独立于具体低精度格式。
 * 解决的问题：单看编码或几个恢复值无法判断整体误差，也不能准确反映存储成本。
 * 输入：原始/恢复浮点数组，或调用者统计好的原始/量化负载字节数。
 * 输出：Metrics（MaxAbs/MAE/MSE）或CompressionStats（字节数和压缩率）。
 * 主流程位置：量化并反量化之后，用原始数组与恢复数组计算误差；
 *              用量化结果的实际字段大小计算压缩率。CPU/GPU实验可共用本模块。
 * 职责边界：不选择scale、不转换编码、不做文件IO，也不测量CUDA执行时间。
 */
#include "metrics.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace luxorion {
// 这里比较「实际输入的FP32数值」与「反量化得到的FP32数值」。
// 该模块不接触低精度编码，不判断量化格式，也不统计补零元素的误差。

// 用途：用三个指标概括恢复值相对原输入的误差，便于验证和比较实验。
// 输入：非空、等长且有限的原始/恢复数组；输出：最大/平均绝对误差和均方误差。
// 主流程关系：由演示或测试调用，数组长度只包含有效元素，不含存储补零。
Metrics calculate_metrics(const std::vector<float>& reference,
                          const std::vector<float>& restored) {
  if (reference.empty() || reference.size() != restored.size()) {
    throw std::invalid_argument("Metrics require nonempty arrays of equal length");
  }
  Metrics result;
  double absolute_sum = 0.0;
  double squared_sum = 0.0;
  // error_i = reference[i] - restored[i]。
  // MaxAbs=max(|error_i|)，MAE=sum(|error_i|)/N，MSE=sum(error_i^2)/N。
  // 用double累积，使指标计算本身的精度损失小于被观察的低精度误差。
  for (std::size_t i = 0; i < reference.size(); ++i) {
    if (!std::isfinite(reference[i]) || !std::isfinite(restored[i])) {
      throw std::invalid_argument("Metrics require finite input and restored values");
    }
    // 必须先转 double 再减，避免两个有限 FP32 数的差在 FP32 中溢出。
    const double error = static_cast<double>(reference[i])
                       - static_cast<double>(restored[i]);
    const double absolute_error = std::fabs(error);
    result.max_abs = std::max(result.max_abs, absolute_error);
    absolute_sum += absolute_error;
    squared_sum += error * error;
  }
  const double count = static_cast<double>(reference.size());
  result.mae = absolute_sum / count;
  result.mse = squared_sum / count;
  // MSE强调较大的误差，但单位是原数据单位的平方；不要把它当作MAE。
  return result;
}

// 用途：量化存储收益，避免只看元素位宽而漏算scale或尾块补零。
// 输入：两项正的负载字节数；输出：原始字节数、量化字节数及其比值。
// 主流程关系：调用者先统计量化结果结构的负载，本函数统一执行比值计算。
CompressionStats calculate_compression(std::size_t original_bytes,
                                       std::size_t quantized_payload_bytes) {
  // 字节数由调用者提供：模块无需猜测输入是FP32还是FP16，也无需知道格式。
  // FP32原始负载=N*4；MXFP8量化负载=data.size()+scales.size()；
  // NVFP4量化负载=packed_data.size()+block_scales.size()+4（global scale）。
  // NVFP4的packed_data.size()包含尾块补零，不能直接用ceil(N/2)代替。
  if (original_bytes == 0 || quantized_payload_bytes == 0) {
    throw std::invalid_argument("Compression requires positive byte counts");
  }
  // 比值为4表示量化负载是原来的1/4；小于1表示这份输入量化后反而更大。
  // 这是量化负载的压缩率，不是CSV文件或带完整header的二进制文件压缩率。
  return {original_bytes, quantized_payload_bytes,
          static_cast<double>(original_bytes)
          / static_cast<double>(quantized_payload_bytes)};
}
}  // namespace luxorion

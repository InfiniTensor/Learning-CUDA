#pragma once

// 模块接口：定义MXFP8量化结果以及CPU量化/反量化入口。
// 调用者只需提供FP32数组，接收编码结果或恢复数组，不必自行处理分块循环。
// MXFP8Result是量化到反量化之间的数据契约；实现见src/mxfp8_cpu.cpp。

#include <cstdint>
#include <vector>

namespace luxorion {

struct MXFP8Result {
  // 行主序连续分块，每 32 个元素一块，尾块 data 不补齐。
  // data[i] 是第i个元素的E4M3编码；读取其浮点值需decode_e4m3。
  std::vector<std::uint8_t> data;
  // scales[b] 是第b块的E8M0编码；data[i]使用scales[i/32]。
  // N个元素对应N个data字节与ceil(N/32)个scale字节，长度可由data推知。
  std::vector<std::uint8_t> scales;
};

// 向上选择 E8M0 scale；全零块取 1，极小块取 2^-127。
// 空输入或非有限输入抛出 std::invalid_argument。
MXFP8Result mxfp8_quantize_cpu(const std::vector<float>& input);
// 原始长度由 data.size() 决定；拒绝错误 scale 数量和 NaN 编码。
// FP32 恢复溢出抛出 std::overflow_error。
std::vector<float> mxfp8_dequantize_cpu(const MXFP8Result& q);

}  // namespace luxorion

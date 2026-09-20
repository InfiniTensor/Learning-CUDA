#pragma once

// GPU baseline接口与CUDA实现独立放置；结果结构复用CPU reference的数据契约，
// 便于逐层比较scale、编码和恢复结果。
#include "mxfp8.h"

namespace luxorion {

// 一个32线程CUDA block处理一个MXFP8量化块，共享内存归约amax。
MXFP8Result mxfp8_quantize_cuda(const std::vector<float>& input);
// 一个CUDA线程恢复一个元素，通过i/32读取所属量化块的scale。
std::vector<float> mxfp8_dequantize_cuda(const MXFP8Result& q);

}  // namespace luxorion

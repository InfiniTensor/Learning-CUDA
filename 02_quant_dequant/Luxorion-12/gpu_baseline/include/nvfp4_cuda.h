#pragma once

// GPU baseline接口复用CPU reference的NVFP4Result，CUDA实现不进入CPU目录。
#include "nvfp4.h"

namespace luxorion {

// GPU先归约global amax，再由一个16线程CUDA block处理一个NVFP4量化块。
NVFP4Result nvfp4_quantize_cuda(const std::vector<float>& input);
// 一个CUDA线程恢复一个有效元素，尾部填充不进入输出。
std::vector<float> nvfp4_dequantize_cuda(const NVFP4Result& q);

}  // namespace luxorion

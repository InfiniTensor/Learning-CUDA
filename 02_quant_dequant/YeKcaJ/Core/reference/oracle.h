#pragma once

// CPU 独立对照接口；GPU runtime 和 kernels 不包含此文件。
#include "common/types.h"
#include "reference/frozen.h"

namespace pipeline {

unsigned reference_encode(float v, bool four, const Options& o, std::size_t i);
Packed reference_quantize(const Tensor& t, const Options& o);
std::vector<float> reference_dequantize(const Packed& q);
void compare_packed(const Packed& a, const Packed& b);

}  // namespace pipeline

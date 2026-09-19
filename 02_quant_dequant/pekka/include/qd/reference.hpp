#pragma once

#include "qd/common.hpp"

#include <vector>

namespace qd {

QuantizedData quantize_reference(const Matrix &matrix, const Config &config);
std::vector<float> dequantize_reference(const QuantizedData &data);

} // namespace qd

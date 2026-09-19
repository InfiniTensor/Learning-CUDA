#include "qd/reference.hpp"

#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
} // namespace

int main() {
  qd::Matrix matrix{1, 5, qd::DataType::kFp32,
                    {0.0F, 0.5F, -1.0F, 3.0F, 6.0F}};
  qd::Config fp4;
  fp4.format = qd::Format::kNvfp4;
  fp4.block_size = 16;
  fp4.scale_mode = qd::ScaleMode::kBlock;
  const qd::QuantizedData packed = qd::quantize_reference(matrix, fp4);
  require(packed.packed_values.size() == 3, "FP4 packed byte count");
  require((packed.packed_values.back() & 0xF0U) == 0, "FP4 odd tail");
  require(packed.local_scales.size() == 1, "FP4 scale count");
  require(qd::dequantize_reference(packed).size() == matrix.values.size(),
          "FP4 output count");

  qd::Config fp8;
  fp8.format = qd::Format::kMxfp8;
  fp8.block_size = 32;
  fp8.scale_mode = qd::ScaleMode::kTensor;
  const qd::QuantizedData tensor = qd::quantize_reference(matrix, fp8);
  require(tensor.packed_values.size() == matrix.values.size(), "FP8 packed count");
  require(tensor.local_scales.size() == 1, "tensor scale count");

  qd::Matrix zeros{1, 33, qd::DataType::kFp32, std::vector<float>(33, 0.0F)};
  fp8.scale_mode = qd::ScaleMode::kBlock;
  const qd::QuantizedData zero_result = qd::quantize_reference(zeros, fp8);
  require(zero_result.local_scales.size() == 2, "tail block scale count");
  for (std::uint8_t code : zero_result.packed_values) {
    require(code == 0, "zero block encoding");
  }
  std::cout << "reference tests passed\n";
}

#include "qd/io.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
} // namespace

int main() {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "qd_io_test";
  std::filesystem::create_directories(root);

  // Half values: +0, 1, and the minimum subnormal 2^-24.  The latter catches
  // the exponent off-by-one that often appears in handwritten half readers.
  const std::vector<std::uint16_t> half_bits{0x0000, 0x3C00, 0x0001};
  std::vector<std::uint8_t> payload(half_bits.size() * sizeof(std::uint16_t));
  std::memcpy(payload.data(), half_bits.data(), payload.size());
  const auto qdat_path = root / "known.qdat";
  qd::write_qdat(qdat_path, 1, 3, qd::DataType::kFp16, payload);
  const qd::Matrix matrix = qd::read_qdat(qdat_path);
  require(matrix.values[0] == 0.0F, "half zero conversion");
  require(matrix.values[1] == 1.0F, "half one conversion");
  require(matrix.values[2] == std::ldexp(1.0F, -24), "half subnormal conversion");

  qd::QuantizedData data;
  data.format = qd::Format::kNvfp4;
  data.original_type = qd::DataType::kFp32;
  data.scale_mode = qd::ScaleMode::kBlock;
  data.rounding = qd::Rounding::kNearest;
  data.rows = 1;
  data.cols = 3;
  data.block_size = 16;
  data.packed_values = {0x21, 0x03};
  data.local_scales = {0x38};
  data.global_scale = 0.5F;
  const auto qlow_path = root / "roundtrip.qlow";
  qd::write_qlow(qlow_path, data);
  const qd::QuantizedData loaded = qd::read_qlow(qlow_path);
  require(loaded.packed_values == data.packed_values, "QLOW packed round trip");
  require(loaded.local_scales == data.local_scales, "QLOW scale round trip");
  require(loaded.global_scale == data.global_scale, "QLOW global scale round trip");

  std::filesystem::remove(qdat_path);
  std::filesystem::remove(qlow_path);
  std::filesystem::remove(root);
}

#pragma once

#include "qd/common.hpp"

#include <filesystem>
#include <vector>

namespace qd {

Metrics calculate_metrics(const std::vector<float> &original,
                          const std::vector<float> &dequantized);
void write_metrics_json(const std::filesystem::path &path, const Matrix &matrix,
                        const Config &config, const QuantizedData &quantized,
                        const Metrics &metrics, const Timings &timings,
                        std::uint64_t qlow_bytes);

} // namespace qd

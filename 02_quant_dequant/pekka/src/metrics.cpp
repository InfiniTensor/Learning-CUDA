#include "qd/metrics.hpp"

#include "qd/io.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace qd {

Metrics calculate_metrics(const std::vector<float> &original,
                          const std::vector<float> &dequantized) {
  if (original.size() != dequantized.size()) {
    throw std::runtime_error("metric inputs have different lengths");
  }
  Metrics metrics;
  long double absolute_sum = 0.0;
  long double squared_sum = 0.0;
  for (std::size_t index = 0; index < original.size(); ++index) {
    if (!std::isfinite(original[index])) {
      ++metrics.non_finite_input_count;
      continue;
    }
    const double error = std::fabs(static_cast<double>(original[index]) -
                                   static_cast<double>(dequantized[index]));
    metrics.max_abs_error = std::max(metrics.max_abs_error, error);
    absolute_sum += error;
    squared_sum += error * error;
    ++metrics.finite_count;
  }
  if (metrics.finite_count != 0) {
    metrics.mean_abs_error = static_cast<double>(absolute_sum / metrics.finite_count);
    metrics.mean_squared_error = static_cast<double>(squared_sum / metrics.finite_count);
  }
  return metrics;
}

void write_metrics_json(const std::filesystem::path &path, const Matrix &matrix,
                        const Config &config, const QuantizedData &quantized,
                        const Metrics &metrics, const Timings &timings,
                        std::uint64_t qlow_bytes) {
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot create metrics JSON: " + path.string());
  const double original_bytes = static_cast<double>(matrix.values.size() *
                                                     data_type_size(matrix.original_type));
  const double quantized_payload = static_cast<double>(
      quantized.packed_values.size() + quantized.local_scales.size() + sizeof(float));
  const double quant_seconds = timings.quantize_ms / 1000.0;
  const double dequant_seconds = timings.dequantize_ms / 1000.0;
  const double quant_logical_bytes = original_bytes + quantized_payload;
  const double dequant_logical_bytes = quantized_payload +
      matrix.values.size() * data_type_size(config.output_type);

  output << std::setprecision(10) << "{\n"
         << "  \"format\": \"" << to_string(config.format) << "\",\n"
         << "  \"scale_mode\": \"" << to_string(config.scale_mode) << "\",\n"
         << "  \"rounding\": \"" << to_string(config.rounding) << "\",\n"
         << "  \"input_dtype\": \"" << to_string(matrix.original_type) << "\",\n"
         << "  \"output_dtype\": \"" << to_string(config.output_type) << "\",\n"
         << "  \"target_gpu_label\": \"" << config.target_gpu << "\",\n"
         << "  \"rows\": " << matrix.rows << ",\n"
         << "  \"cols\": " << matrix.cols << ",\n"
         << "  \"elements\": " << matrix.values.size() << ",\n"
         << "  \"max_abs_error\": " << metrics.max_abs_error << ",\n"
         << "  \"mae\": " << metrics.mean_abs_error << ",\n"
         << "  \"mse\": " << metrics.mean_squared_error << ",\n"
         << "  \"finite_count\": " << metrics.finite_count << ",\n"
         << "  \"non_finite_input_count\": " << metrics.non_finite_input_count << ",\n"
         << "  \"payload_compression_ratio\": " << original_bytes / quantized_payload << ",\n"
         << "  \"file_compression_ratio\": " << original_bytes / qlow_bytes << ",\n"
         << "  \"qlow_file_bytes\": " << qlow_bytes << ",\n"
         << "  \"benchmark_iterations\": " << timings.iterations << ",\n"
         << "  \"quantize_ms\": " << timings.quantize_ms << ",\n"
         << "  \"dequantize_ms\": " << timings.dequantize_ms << ",\n"
         << "  \"quantize_effective_gbps\": "
         << (quant_seconds > 0.0 ? quant_logical_bytes / quant_seconds / 1.0e9 : 0.0) << ",\n"
         << "  \"dequantize_effective_gbps\": "
         << (dequant_seconds > 0.0 ? dequant_logical_bytes / dequant_seconds / 1.0e9 : 0.0) << "\n"
         << "}\n";
}

} // namespace qd

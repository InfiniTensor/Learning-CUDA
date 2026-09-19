#include "qd/config.hpp"
#include "qd/cuda_ops.hpp"
#include "qd/io.hpp"
#include "qd/metrics.hpp"
#include "qd/reference.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Arguments {
  std::filesystem::path input;
  std::filesystem::path config;
  std::filesystem::path quantized;
  std::filesystem::path output;
  std::filesystem::path metrics;
  int warmup_iterations = 3;
  int benchmark_iterations = 10;
  bool verify_reference = true;
};

void print_usage(const char *program) {
  std::cerr
      << "Usage: " << program << " --input matrix.qdat --config format.toml\n"
      << "       --quantized matrix.qlow --output reconstructed.qdat\n"
      << "       --metrics metrics.json [--warmup N] [--iterations N]\n"
      << "       [--no-reference-check]\n";
}

Arguments parse_arguments(int argc, char **argv) {
  Arguments arguments;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    if (option == "--no-reference-check") {
      arguments.verify_reference = false;
      continue;
    }
    if (index + 1 >= argc) throw std::runtime_error("missing value after " + option);
    const std::string value = argv[++index];
    if (option == "--input") arguments.input = value;
    else if (option == "--config") arguments.config = value;
    else if (option == "--quantized") arguments.quantized = value;
    else if (option == "--output") arguments.output = value;
    else if (option == "--metrics") arguments.metrics = value;
    else if (option == "--warmup") arguments.warmup_iterations = std::stoi(value);
    else if (option == "--iterations") arguments.benchmark_iterations = std::stoi(value);
    else throw std::runtime_error("unknown option: " + option);
  }
  if (arguments.input.empty() || arguments.config.empty() ||
      arguments.quantized.empty() || arguments.output.empty() ||
      arguments.metrics.empty()) {
    throw std::runtime_error("all five path options are required");
  }
  if (arguments.warmup_iterations < 0 || arguments.benchmark_iterations <= 0) {
    throw std::runtime_error("warmup must be non-negative and iterations must be positive");
  }
  return arguments;
}

void check_reference(const qd::Matrix &matrix, const qd::Config &config,
                     const qd::QuantizedData &gpu) {
  const qd::QuantizedData cpu = qd::quantize_reference(matrix, config);
  if (cpu.packed_values != gpu.packed_values ||
      cpu.local_scales != gpu.local_scales ||
      cpu.global_scale != gpu.global_scale) {
    throw std::runtime_error("CPU/GPU quantization mismatch");
  }
  const std::vector<float> cpu_values = qd::dequantize_reference(cpu);
  // The GPU dequantization comparison is performed by the caller after QLOW
  // reload; checking the quantized bytes here gives a more useful root cause.
  if (cpu_values.size() != matrix.values.size()) {
    throw std::runtime_error("CPU reference produced an invalid output length");
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 1) {
      print_usage(argv[0]);
      return 2;
    }
    const Arguments arguments = parse_arguments(argc, argv);
    const qd::Config config = qd::read_config(arguments.config);
    const qd::Matrix matrix = qd::read_qdat(arguments.input);

    if (arguments.warmup_iterations > 0) {
      (void)qd::quantize_gpu(matrix, config, arguments.warmup_iterations);
    }
    qd::GpuQuantizeResult quantized =
        qd::quantize_gpu(matrix, config, arguments.benchmark_iterations);
    if (arguments.verify_reference) check_reference(matrix, config, quantized.data);

    qd::write_qlow(arguments.quantized, quantized.data);
    // Reloading is intentional: dequantization validates the on-disk artifact,
    // not merely the in-memory object that was just produced.
    qd::QuantizedData reloaded = qd::read_qlow(arguments.quantized);
    if (arguments.warmup_iterations > 0) {
      (void)qd::dequantize_gpu(reloaded, config.output_type,
                              arguments.warmup_iterations);
    }
    qd::GpuDequantizeResult dequantized = qd::dequantize_gpu(
        reloaded, config.output_type, arguments.benchmark_iterations);
    qd::write_qdat(arguments.output, matrix.rows, matrix.cols,
                   config.output_type, dequantized.output_payload);

    if (arguments.verify_reference) {
      const std::vector<float> cpu = qd::dequantize_reference(reloaded);
      if (cpu != dequantized.fp32_values) {
        throw std::runtime_error("CPU/GPU dequantization mismatch");
      }
    }
    const qd::Metrics metrics =
        qd::calculate_metrics(matrix.values, dequantized.fp32_values);
    const qd::Timings timings{quantized.milliseconds, dequantized.milliseconds,
                              arguments.benchmark_iterations};
    qd::write_metrics_json(arguments.metrics, matrix, config, reloaded, metrics,
                           timings, qd::qlow_file_size(reloaded));

    std::cout << "Completed " << qd::to_string(config.format) << " pipeline for "
              << matrix.rows << "x" << matrix.cols << " "
              << qd::to_string(matrix.original_type) << " input.\n"
              << "  QLOW: " << arguments.quantized << "\n"
              << "  reconstructed QDAT: " << arguments.output << " ("
              << qd::to_string(config.output_type) << ")\n"
              << "  max_abs_error=" << metrics.max_abs_error
              << ", MAE=" << metrics.mean_abs_error
              << ", MSE=" << metrics.mean_squared_error << "\n"
              << "  quantize=" << timings.quantize_ms
              << " ms, dequantize=" << timings.dequantize_ms << " ms\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    print_usage(argv[0]);
    return 1;
  }
}

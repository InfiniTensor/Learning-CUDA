#include "benchmark.h"
#include "io.h"
#include "metrics.h"
#include "mxfp8.h"
#include "nvfp4.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

using luxorion::CompressionStats;
using luxorion::GpuBaselineTimings;
using luxorion::Metrics;

struct ResultRow {
  std::string dataset;
  std::string format;
  std::string input_dtype;
  std::size_t input_payload_bytes = 0;
  std::size_t elements = 0;
  GpuBaselineTimings timings;
  Metrics error;
  CompressionStats compression;
};

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": "
                             + cudaGetErrorString(status));
  }
}

std::string json_escape(const std::string& value) {
  std::ostringstream out;
  for (const unsigned char ch : value) {
    switch (ch) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (ch < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<unsigned>(ch) << std::dec << std::setfill(' ');
        } else {
          out << ch;
        }
    }
  }
  return out.str();
}

void require_same_float(float actual, float expected, const char* label,
                        std::size_t index) {
  if (actual != expected || std::signbit(actual) != std::signbit(expected)) {
    std::ostringstream message;
    message << label << " mismatch at index " << index << ": GPU=" << actual
            << ", CPU=" << expected;
    throw std::runtime_error(message.str());
  }
}

void validate_mxfp8(const luxorion::MXFP8BenchmarkResult& gpu,
                    const std::vector<float>& input) {
  const auto cpu_q = luxorion::mxfp8_quantize_cpu(input);
  if (gpu.quantized.data != cpu_q.data || gpu.quantized.scales != cpu_q.scales) {
    throw std::runtime_error("MXFP8 GPU quantized payload differs from CPU oracle");
  }
  const auto cpu_restored = luxorion::mxfp8_dequantize_cpu(cpu_q);
  if (gpu.restored.size() != cpu_restored.size()) {
    throw std::runtime_error("MXFP8 GPU restored length differs from CPU oracle");
  }
  for (std::size_t i = 0; i < cpu_restored.size(); ++i) {
    require_same_float(gpu.restored[i], cpu_restored[i], "MXFP8 restored", i);
  }
}

void validate_nvfp4(const luxorion::NVFP4BenchmarkResult& gpu,
                    const std::vector<float>& input) {
  const auto cpu_q = luxorion::nvfp4_quantize_cpu(input);
  if (gpu.quantized.element_count != cpu_q.element_count
      || gpu.quantized.packed_data != cpu_q.packed_data
      || gpu.quantized.block_scales != cpu_q.block_scales) {
    throw std::runtime_error("NVFP4 GPU quantized payload differs from CPU oracle");
  }
  require_same_float(gpu.quantized.global_scale, cpu_q.global_scale,
                     "NVFP4 global scale", 0);
  const auto cpu_restored = luxorion::nvfp4_dequantize_cpu(cpu_q);
  if (gpu.restored.size() != cpu_restored.size()) {
    throw std::runtime_error("NVFP4 GPU restored length differs from CPU oracle");
  }
  for (std::size_t i = 0; i < cpu_restored.size(); ++i) {
    require_same_float(gpu.restored[i], cpu_restored[i], "NVFP4 restored", i);
  }
}

void write_csv(const fs::path& path, const std::vector<ResultRow>& rows) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot create CSV: " + path.string());
  out << "dataset,format,input_dtype,input_payload_bytes,elements,warmup,repeats,"
         "quantize_median_ms,quantize_p95_ms,quantize_effective_gbps,"
         "dequantize_median_ms,dequantize_p95_ms,dequantize_effective_gbps,"
         "device_pipeline_median_ms,device_pipeline_p95_ms,"
         "end_to_end_median_ms,end_to_end_p95_ms,"
         "max_abs,mae,mse,original_bytes,quantized_payload_bytes,compression_ratio\n";
  out << std::setprecision(10);
  for (const auto& row : rows) {
    const auto& t = row.timings;
    out << row.dataset << ',' << row.format << ',' << row.input_dtype << ','
        << row.input_payload_bytes << ',' << row.elements << ','
        << t.warmup << ',' << t.repeats << ','
        << t.quantize.median_ms << ',' << t.quantize.p95_ms << ','
        << t.quantize_effective_gbps << ','
        << t.dequantize.median_ms << ',' << t.dequantize.p95_ms << ','
        << t.dequantize_effective_gbps << ','
        << t.device_pipeline.median_ms << ',' << t.device_pipeline.p95_ms << ','
        << t.end_to_end.median_ms << ',' << t.end_to_end.p95_ms << ','
        << row.error.max_abs << ',' << row.error.mae << ',' << row.error.mse << ','
        << row.compression.original_bytes << ','
        << row.compression.quantized_payload_bytes << ','
        << row.compression.compression_ratio << '\n';
  }
}

void write_timing_json(std::ostream& out, const GpuBaselineTimings& t,
                       const std::string& indent) {
  out << indent << "\"warmup\": " << t.warmup << ",\n"
      << indent << "\"repeats\": " << t.repeats << ",\n"
      << indent << "\"quantize\": {\"median_ms\": " << t.quantize.median_ms
      << ", \"p95_ms\": " << t.quantize.p95_ms
      << ", \"effective_gbps\": " << t.quantize_effective_gbps << "},\n"
      << indent << "\"dequantize\": {\"median_ms\": " << t.dequantize.median_ms
      << ", \"p95_ms\": " << t.dequantize.p95_ms
      << ", \"effective_gbps\": " << t.dequantize_effective_gbps << "},\n"
      << indent << "\"device_pipeline\": {\"median_ms\": "
      << t.device_pipeline.median_ms << ", \"p95_ms\": "
      << t.device_pipeline.p95_ms << "},\n"
      << indent << "\"end_to_end\": {\"median_ms\": "
      << t.end_to_end.median_ms << ", \"p95_ms\": "
      << t.end_to_end.p95_ms << "}\n";
}

void write_json(const fs::path& path, const std::vector<ResultRow>& rows,
                const cudaDeviceProp& properties, int runtime_version,
                int driver_version, unsigned warmup, unsigned repeats) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot create JSON: " + path.string());
  out << std::setprecision(10);
  out << "{\n"
      << "  \"protocol_version\": 1,\n"
      << "  \"environment\": {\n"
      << "    \"device\": \"" << json_escape(properties.name) << "\",\n"
      << "    \"compute_capability\": \"" << properties.major << '.'
      << properties.minor << "\",\n"
      << "    \"cuda_runtime_version\": " << runtime_version << ",\n"
      << "    \"cuda_driver_version\": " << driver_version << "\n"
      << "  },\n"
      << "  \"protocol\": {\n"
      << "    \"warmup\": " << warmup << ",\n"
      << "    \"repeats\": " << repeats << ",\n"
      << "    \"percentiles\": \"median and nearest-rank p95\",\n"
      << "    \"kernel_clock\": \"CUDA events on the default stream\",\n"
      << "    \"end_to_end_clock\": \"host steady_clock\",\n"
      << "    \"device_pipeline\": \"quantize + dequantize; data remains on device\",\n"
      << "    \"end_to_end\": \"H2D + quantize + dequantize + D2H; excludes allocation, file IO, and validation\",\n"
      << "    \"blocking\": \"row-major flattened continuous blocks: MXFP8=32, NVFP4=16\"\n"
      << "  },\n"
      << "  \"results\": [\n";
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto& row = rows[i];
    out << "    {\n"
        << "      \"dataset\": \"" << json_escape(row.dataset) << "\",\n"
        << "      \"format\": \"" << row.format << "\",\n"
        << "      \"input_dtype\": \"" << row.input_dtype << "\",\n"
        << "      \"input_payload_bytes\": " << row.input_payload_bytes << ",\n"
        << "      \"elements\": " << row.elements << ",\n"
        << "      \"timing\": {\n";
    write_timing_json(out, row.timings, "        ");
    out << "      },\n"
        << "      \"error\": {\"max_abs\": " << row.error.max_abs
        << ", \"mae\": " << row.error.mae << ", \"mse\": "
        << row.error.mse << "},\n"
        << "      \"storage\": {\"original_bytes\": "
        << row.compression.original_bytes
        << ", \"quantized_payload_bytes\": "
        << row.compression.quantized_payload_bytes
        << ", \"compression_ratio\": " << row.compression.compression_ratio
        << "}\n"
        << "    }" << (i + 1 == rows.size() ? "\n" : ",\n");
  }
  out << "  ]\n}\n";
}

unsigned parse_count(const char* text, const char* label, bool allow_zero) {
  const std::string value(text);
  std::size_t parsed = 0;
  const unsigned long number = std::stoul(value, &parsed);
  if (parsed != value.size() || (!allow_zero && number == 0)) {
    throw std::invalid_argument(std::string(label) + " is invalid: " + value);
  }
  return static_cast<unsigned>(number);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc > 5) {
      throw std::invalid_argument(
          "usage: gpu_baseline_benchmark [input_dir] [output_dir] [warmup] [repeats]");
    }
    const fs::path input_dir = argc > 1 ? argv[1] : "input/benchmark";
    const fs::path output_dir = argc > 2 ? argv[2] : "output/gpu_baseline";
    const unsigned warmup = argc > 3 ? parse_count(argv[3], "warmup", true) : 10;
    const unsigned repeats = argc > 4 ? parse_count(argv[4], "repeats", false) : 100;

    int device = 0;
    int runtime_version = 0;
    int driver_version = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device), "cudaGetDevice failed");
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties failed");
    check_cuda(cudaRuntimeGetVersion(&runtime_version),
               "cudaRuntimeGetVersion failed");
    check_cuda(cudaDriverGetVersion(&driver_version),
               "cudaDriverGetVersion failed");

    const std::vector<std::pair<std::string, std::string>> datasets = {
        {"uniform_fp32_1024x1024_seed1234.tensor.bin", "uniform_fp32"},
        {"normal_fp32_1024x1024_seed1234.tensor.bin", "normal_fp32"},
        {"outliers_fp32_1024x1024_seed1234.tensor.bin", "outliers_fp32"},
        {"zero_heavy_fp32_1024x1024_seed1234.tensor.bin", "zero_heavy_fp32"},
        {"multiscale_fp32_1024x1024_seed1234.tensor.bin", "multiscale_fp32"},
        {"normal_fp16_1024x1024_seed1234.tensor.bin", "normal_fp16"},
        {"uniform_fp16_1024x1024_seed1234.tensor.bin", "uniform_fp16"},
        {"outliers_fp16_1024x1024_seed1234.tensor.bin", "outliers_fp16"},
        {"zero_heavy_fp16_1024x1024_seed1234.tensor.bin", "zero_heavy_fp16"},
        {"multiscale_fp16_1024x1024_seed1234.tensor.bin", "multiscale_fp16"},
        {"tail_normal_fp32_1023x1025_seed1234.tensor.bin", "tail_normal_fp32"},
        {"tail_normal_fp16_1023x1025_seed1234.tensor.bin", "tail_normal_fp16"},
    };
    std::vector<ResultRow> rows;
    rows.reserve(datasets.size() * 2);

    std::cout << "GPU: " << properties.name << ", warmup=" << warmup
              << ", repeats=" << repeats << std::endl;
    for (const auto& entry : datasets) {
      const auto& filename = entry.first;
      const auto tensor = luxorion::read_tensor_file((input_dir / filename).string());
      const std::string& dataset = entry.second;
      const std::string input_dtype = tensor.dtype == luxorion::TensorDType::FP16
          ? "fp16" : "fp32";
      const std::size_t input_payload_bytes = tensor.values.size()
          * (tensor.dtype == luxorion::TensorDType::FP16 ? 2U : 4U);
      std::cout << '[' << dataset << "] MXFP8..." << std::flush;
      const auto mxfp8 = tensor.dtype == luxorion::TensorDType::FP16
          ? luxorion::benchmark_mxfp8_cuda_fp16(
                tensor.fp16_bits, tensor.values, warmup, repeats)
          : luxorion::benchmark_mxfp8_cuda(
                tensor.values, warmup, repeats);
      validate_mxfp8(mxfp8, tensor.values);
      rows.push_back({
          dataset,
          "MXFP8",
          input_dtype,
          input_payload_bytes,
          tensor.values.size(),
          mxfp8.timings,
          luxorion::calculate_metrics(tensor.values, mxfp8.restored),
          luxorion::calculate_compression(
              input_payload_bytes,
              mxfp8.quantized.data.size() + mxfp8.quantized.scales.size()),
      });
      std::cout << " verified" << std::endl;

      std::cout << '[' << dataset << "] NVFP4..." << std::flush;
      const auto nvfp4 = tensor.dtype == luxorion::TensorDType::FP16
          ? luxorion::benchmark_nvfp4_cuda_fp16(
                tensor.fp16_bits, tensor.values, warmup, repeats)
          : luxorion::benchmark_nvfp4_cuda(
                tensor.values, warmup, repeats);
      validate_nvfp4(nvfp4, tensor.values);
      rows.push_back({
          dataset,
          "NVFP4",
          input_dtype,
          input_payload_bytes,
          tensor.values.size(),
          nvfp4.timings,
          luxorion::calculate_metrics(tensor.values, nvfp4.restored),
          luxorion::calculate_compression(
              input_payload_bytes,
              nvfp4.quantized.packed_data.size()
                  + nvfp4.quantized.block_scales.size() + sizeof(float)),
      });
      std::cout << " verified" << std::endl;
    }

    fs::create_directories(output_dir);
    const fs::path csv_path = output_dir / "baseline_results.csv";
    const fs::path json_path = output_dir / "baseline_results.json";
    write_csv(csv_path, rows);
    write_json(json_path, rows, properties, runtime_version, driver_version,
               warmup, repeats);
    std::cout << "CSV:  " << fs::absolute(csv_path).string() << '\n'
              << "JSON: " << fs::absolute(json_path).string() << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "gpu_baseline_benchmark failed: " << error.what() << '\n';
    return 1;
  }
}

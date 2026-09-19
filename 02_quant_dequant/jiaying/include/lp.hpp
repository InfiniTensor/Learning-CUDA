#pragma once
#include "codecs.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lp {
enum class DType : uint32_t { FP32 = 0, FP16 = 1, BF16 = 2 };
enum class Format : uint32_t { MXFP8 = 0, NVFP4 = 1 };
struct Config {
    Format format = Format::MXFP8;
    uint32_t block_size = 32;
    bool tensor_scale = false;
    DType output_type = DType::FP16;
    bool stochastic = false;
    uint64_t seed = 42;
    std::string target_gpu = "unspecified";
};
struct Tensor {
    uint64_t rows = 0, cols = 0;
    DType dtype = DType::FP32;
    std::vector<uint8_t> data;
    uint64_t size() const { return rows * cols; }
};
struct Weights {
    uint64_t rows = 0, cols = 0;
    Config config;
    DType input_type = DType::FP32;
    float global_scale = 1;
    std::vector<uint8_t> data, scales;
    uint64_t row_bytes() const { return config.format == Format::MXFP8 ? cols : (cols + 1) / 2; }
    uint64_t blocks_per_row() const { return (cols + config.block_size - 1) / config.block_size; }
};
struct Timings {
    double quant_ms = 0, dequant_ms = 0, transfer_pipeline_ms = 0;
    double quant_bytes = 0, dequant_bytes = 0;
    std::string device = "CPU single thread";
    std::string compute_capability;
    int runtime_version = 0, driver_version = 0;
};
struct Metrics {
    double max_abs = 0, mae = 0, mse = 0, rmse = 0;
    uint64_t nonfinite_output = 0;
};
size_t dtype_bytes(DType type);
std::string dtype_name(DType type);
DType parse_dtype(const std::string &name);
void validate_config(const Config &c);
size_t checked_size(uint64_t rows, uint64_t cols, size_t bytes);
Config read_config(const std::string &path);
Tensor read_tensor(const std::string &path, bool input_only = true);
void write_tensor(const std::string &path, const Tensor &tensor);
Weights read_weights(const std::string &path);
void write_weights(const std::string &path, const Weights &weights);
Weights make_weights(const Tensor &tensor, const Config &config);
Tensor make_output(const Weights &weights, DType type);
float tensor_value(const Tensor &tensor, uint64_t index);
void quantize_cpu(const Tensor &input, Weights &weights);
void dequantize_cpu(const Weights &weights, Tensor &output);
Timings run_cpu(const Tensor &input, Weights &weights, Tensor &output, int repeats);
Timings run_cuda(const Tensor &input, Weights &weights, Tensor &output, int warmup, int repeats);
Timings decode_cuda(const Weights &weights, Tensor &output, int warmup, int repeats);
Metrics measure_error(const Tensor &input, const Tensor &output);
double elapsed_ms(double start);
double now_ms();
void write_log(const std::string &path, const Tensor &input, const Weights &weights,
               const Metrics &error, const Timings &gpu, const Timings &cpu, double io_ms,
               double total_ms, int warmup, int repeats, const std::string &backend);
} // namespace lp

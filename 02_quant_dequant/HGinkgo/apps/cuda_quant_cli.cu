#include "low_precision/cuda_quant.hpp"
#include "low_precision/io.hpp"

#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum class Mode { Quantize, Dequantize };
enum class OutputType { Fp32, Fp16, Bf16 };

struct Options {
    Mode mode = Mode::Quantize;
    std::string input_path;
    std::string output_path;
    std::size_t rows = 0;
    std::size_t cols = 0;
    low_precision::InputType input_type = low_precision::InputType::Fp32;
    OutputType output_type = OutputType::Fp32;
    low_precision::Format format = low_precision::Format::MxFp8;
    low_precision::Rounding rounding = low_precision::Rounding::NearestEven;
    std::uint64_t seed = 0;
    bool has_rows = false;
    bool has_cols = false;
    bool has_input_type = false;
    bool has_output_type = false;
    bool has_format = false;
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message +
        "\nusage: cuda_quant_cli quantize --input IN --output OUT "
        "--rows R --cols C --input-type fp32|fp16 "
        "--format mxfp8|nvfp4 --rounding nearest|stochastic [--seed N]\n"
        "       cuda_quant_cli dequantize --input IN --output OUT "
        "--output-type fp32|fp16|bf16");
}

std::string require_value(int& index, int argc, char** argv) {
    if (index + 1 >= argc) {
        usage_error("missing value for " + std::string(argv[index]));
    }
    return argv[++index];
}

std::size_t parse_size(const std::string& value, const char* name) {
    try {
        std::size_t consumed = 0;
        const unsigned long long parsed = std::stoull(value, &consumed);
        if (consumed != value.size() || parsed == 0 ||
            parsed > std::numeric_limits<std::size_t>::max()) {
            usage_error(std::string("invalid ") + name);
        }
        return static_cast<std::size_t>(parsed);
    } catch (const std::exception&) {
        usage_error(std::string("invalid ") + name);
    }
}

std::uint64_t parse_seed(const std::string& value) {
    try {
        std::size_t consumed = 0;
        const unsigned long long parsed = std::stoull(value, &consumed);
        if (consumed != value.size()) {
            usage_error("invalid seed");
        }
        return static_cast<std::uint64_t>(parsed);
    } catch (const std::exception&) {
        usage_error("invalid seed");
    }
}

Options parse_options(int argc, char** argv) {
    if (argc < 2) {
        usage_error("missing command");
    }
    Options options;
    const std::string command = argv[1];
    if (command == "quantize") {
        options.mode = Mode::Quantize;
    } else if (command == "dequantize") {
        options.mode = Mode::Dequantize;
    } else if (command == "--help" || command == "-h") {
        std::cout << "cuda_quant_cli quantize|dequantize ...\n";
        std::exit(0);
    } else {
        usage_error("unknown command: " + command);
    }

    for (int index = 2; index < argc; ++index) {
        const std::string option = argv[index];
        const std::string value = require_value(index, argc, argv);
        if (option == "--input") {
            options.input_path = value;
        } else if (option == "--output") {
            options.output_path = value;
        } else if (option == "--rows") {
            options.rows = parse_size(value, "rows");
            options.has_rows = true;
        } else if (option == "--cols") {
            options.cols = parse_size(value, "cols");
            options.has_cols = true;
        } else if (option == "--input-type" && options.mode == Mode::Quantize) {
            if (value == "fp32") {
                options.input_type = low_precision::InputType::Fp32;
            } else if (value == "fp16") {
                options.input_type = low_precision::InputType::Fp16;
            } else {
                usage_error("invalid input type");
            }
            options.has_input_type = true;
        } else if (option == "--output-type" && options.mode == Mode::Dequantize) {
            if (value == "fp32") {
                options.output_type = OutputType::Fp32;
            } else if (value == "fp16") {
                options.output_type = OutputType::Fp16;
            } else if (value == "bf16") {
                options.output_type = OutputType::Bf16;
            } else {
                usage_error("invalid output type");
            }
            options.has_output_type = true;
        } else if (option == "--format" && options.mode == Mode::Quantize) {
            if (value == "mxfp8") {
                options.format = low_precision::Format::MxFp8;
            } else if (value == "nvfp4") {
                options.format = low_precision::Format::NvFp4;
            } else {
                usage_error("invalid quantization format");
            }
            options.has_format = true;
        } else if (option == "--rounding" && options.mode == Mode::Quantize) {
            if (value == "nearest") {
                options.rounding = low_precision::Rounding::NearestEven;
            } else if (value == "stochastic") {
                options.rounding = low_precision::Rounding::Stochastic;
            } else {
                usage_error("invalid rounding mode");
            }
        } else if (option == "--seed" && options.mode == Mode::Quantize) {
            options.seed = parse_seed(value);
        } else {
            usage_error("unknown or misplaced option: " + option);
        }
    }
    if (options.input_path.empty() || options.output_path.empty()) {
        usage_error("input and output paths are required");
    }
    if (options.mode == Mode::Quantize &&
        (!options.has_rows || !options.has_cols || !options.has_input_type ||
         !options.has_format)) {
        usage_error("quantize requires rows, cols, input-type and format");
    }
    if (options.mode == Mode::Dequantize && !options.has_output_type) {
        usage_error("dequantize requires output-type");
    }
    return options;
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        check_cuda(cudaMalloc(&pointer_, count * sizeof(T)), "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() { return pointer_; }

private:
    T* pointer_ = nullptr;
};

std::vector<std::uint8_t> read_bytes(const std::string& path, std::size_t byte_count) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open input file: " + path);
    }
    std::vector<std::uint8_t> bytes(byte_count);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input || input.get() != std::char_traits<char>::eof()) {
        throw std::runtime_error("input file size does not match rows, cols and input type");
    }
    return bytes;
}

void write_bytes(const std::string& path, const void* data, std::size_t byte_count) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot open output file: " + path);
    }
    output.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(byte_count));
    if (!output) {
        throw std::runtime_error("failed to write output file");
    }
}

template <typename T>
std::vector<T> decode_raw(const std::vector<std::uint8_t>& bytes, std::size_t count) {
    if (bytes.size() != count * sizeof(T)) {
        throw std::runtime_error("raw input size mismatch");
    }
    std::vector<T> values(count);
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
}

template <typename T>
void write_vector(const std::string& path, const std::vector<T>& values) {
    write_bytes(path, values.data(), values.size() * sizeof(T));
}

low_precision::QuantizedFile quantize(const Options& options) {
    const std::size_t count = options.rows * options.cols;
    const std::size_t element_size = options.input_type == low_precision::InputType::Fp32
                                         ? sizeof(float)
                                         : sizeof(low_precision::Fp16);
    const auto bytes = read_bytes(options.input_path, count * element_size);
    low_precision::QuantizedFile file;
    file.format = options.format;
    file.input_type = options.input_type;
    file.rounding = options.rounding;
    file.rows = options.rows;
    file.cols = options.cols;
    file.block_size = options.format == low_precision::Format::MxFp8 ? 32u : 16u;
    file.scale_mode = options.format == low_precision::Format::MxFp8
                          ? low_precision::ScaleMode::Block
                          : low_precision::ScaleMode::TensorAndBlock;

    DeviceBuffer<std::uint8_t> d_values(
        options.format == low_precision::Format::MxFp8 ? count : (count + 1) / 2);
    const std::size_t scale_count = options.format == low_precision::Format::MxFp8
                                        ? options.rows * ((options.cols + 31) / 32)
                                        : options.rows * ((options.cols + 15) / 16);
    DeviceBuffer<std::uint8_t> d_scales(scale_count);
    if (options.input_type == low_precision::InputType::Fp32) {
        const auto input = decode_raw<float>(bytes, count);
        DeviceBuffer<float> d_input(count);
        check_cuda(cudaMemcpy(d_input.get(), input.data(), bytes.size(), cudaMemcpyHostToDevice),
                   "copy FP32 input");
        if (options.format == low_precision::Format::MxFp8) {
            low_precision::cuda_quantize_mxfp8(d_input.get(), d_values.get(), d_scales.get(),
                                                options.rows, options.cols, options.rounding,
                                                options.seed);
        } else {
            float global_scale = 0.0f;
            low_precision::cuda_quantize_nvfp4(d_input.get(), d_values.get(), d_scales.get(),
                                                options.rows, options.cols, &global_scale,
                                                options.rounding, options.seed);
            file.global_scale = global_scale;
        }
    } else {
        const auto input = decode_raw<low_precision::Fp16>(bytes, count);
        DeviceBuffer<low_precision::Fp16> d_input(count);
        check_cuda(cudaMemcpy(d_input.get(), input.data(), bytes.size(), cudaMemcpyHostToDevice),
                   "copy FP16 input");
        if (options.format == low_precision::Format::MxFp8) {
            low_precision::cuda_quantize_mxfp8(d_input.get(), d_values.get(), d_scales.get(),
                                                options.rows, options.cols, options.rounding,
                                                options.seed);
        } else {
            float global_scale = 0.0f;
            low_precision::cuda_quantize_nvfp4(d_input.get(), d_values.get(), d_scales.get(),
                                                options.rows, options.cols, &global_scale,
                                                options.rounding, options.seed);
            file.global_scale = global_scale;
        }
    }
    file.values.resize((count + (options.format == low_precision::Format::NvFp4 ? 1 : 0)) /
                       (options.format == low_precision::Format::NvFp4 ? 2 : 1));
    file.scales.resize(scale_count);
    check_cuda(cudaMemcpy(file.values.data(), d_values.get(), file.values.size(),
                          cudaMemcpyDeviceToHost),
               "copy quantized values");
    check_cuda(cudaMemcpy(file.scales.data(), d_scales.get(), file.scales.size(),
                          cudaMemcpyDeviceToHost),
               "copy quantized scales");
    return file;
}

void dequantize(const Options& options) {
    const auto file = low_precision::read_quantized_file(options.input_path);
    const std::size_t count = file.rows * file.cols;
    DeviceBuffer<std::uint8_t> d_values(file.values.size());
    DeviceBuffer<std::uint8_t> d_scales(file.scales.size());
    check_cuda(cudaMemcpy(d_values.get(), file.values.data(), file.values.size(),
                          cudaMemcpyHostToDevice),
               "copy quantized values");
    check_cuda(cudaMemcpy(d_scales.get(), file.scales.data(), file.scales.size(),
                          cudaMemcpyHostToDevice),
               "copy quantized scales");

    if (file.format == low_precision::Format::MxFp8) {
        if (options.output_type == OutputType::Fp32) {
            DeviceBuffer<float> d_output(count);
            low_precision::cuda_dequantize_mxfp8(d_values.get(), d_scales.get(), d_output.get(),
                                                 file.rows, file.cols);
            std::vector<float> output(count);
            check_cuda(cudaMemcpy(output.data(), d_output.get(), output.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost),
                       "copy FP32 output");
            write_vector(options.output_path, output);
        } else if (options.output_type == OutputType::Fp16) {
            DeviceBuffer<low_precision::Fp16> d_output(count);
            low_precision::cuda_dequantize_mxfp8(d_values.get(), d_scales.get(), d_output.get(),
                                                 file.rows, file.cols);
            std::vector<low_precision::Fp16> output(count);
            check_cuda(cudaMemcpy(output.data(), d_output.get(), output.size() * sizeof(output[0]),
                                  cudaMemcpyDeviceToHost),
                       "copy FP16 output");
            write_vector(options.output_path, output);
        } else {
            DeviceBuffer<low_precision::Bf16> d_output(count);
            low_precision::cuda_dequantize_mxfp8(d_values.get(), d_scales.get(), d_output.get(),
                                                 file.rows, file.cols);
            std::vector<low_precision::Bf16> output(count);
            check_cuda(cudaMemcpy(output.data(), d_output.get(), output.size() * sizeof(output[0]),
                                  cudaMemcpyDeviceToHost),
                       "copy BF16 output");
            write_vector(options.output_path, output);
        }
    } else if (options.output_type == OutputType::Fp32) {
        DeviceBuffer<float> d_output(count);
        low_precision::cuda_dequantize_nvfp4(d_values.get(), d_scales.get(), file.global_scale,
                                             d_output.get(), file.rows, file.cols);
        std::vector<float> output(count);
        check_cuda(cudaMemcpy(output.data(), d_output.get(), output.size() * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "copy FP32 output");
        write_vector(options.output_path, output);
    } else if (options.output_type == OutputType::Fp16) {
        DeviceBuffer<low_precision::Fp16> d_output(count);
        low_precision::cuda_dequantize_nvfp4(d_values.get(), d_scales.get(), file.global_scale,
                                             d_output.get(), file.rows, file.cols);
        std::vector<low_precision::Fp16> output(count);
        check_cuda(cudaMemcpy(output.data(), d_output.get(), output.size() * sizeof(output[0]),
                              cudaMemcpyDeviceToHost),
                   "copy FP16 output");
        write_vector(options.output_path, output);
    } else {
        DeviceBuffer<low_precision::Bf16> d_output(count);
        low_precision::cuda_dequantize_nvfp4(d_values.get(), d_scales.get(), file.global_scale,
                                             d_output.get(), file.rows, file.cols);
        std::vector<low_precision::Bf16> output(count);
        check_cuda(cudaMemcpy(output.data(), d_output.get(), output.size() * sizeof(output[0]),
                              cudaMemcpyDeviceToHost),
                   "copy BF16 output");
        write_vector(options.output_path, output);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        int device_count = 0;
        const cudaError_t device_status = cudaGetDeviceCount(&device_count);
        if (device_status == cudaErrorNoDevice ||
            device_status == cudaErrorInsufficientDriver || device_count == 0) {
            std::cerr << "cuda_quant_cli: no usable CUDA device; skipped\n";
            return 77;
        }
        check_cuda(device_status, "cudaGetDeviceCount");
        const Options options = parse_options(argc, argv);
        if (options.mode == Mode::Quantize) {
            low_precision::write_quantized_file(options.output_path, quantize(options));
        } else {
            dequantize(options);
        }
    } catch (const std::exception& error) {
        std::cerr << "cuda_quant_cli: " << error.what() << '\n';
        return 1;
    }
}

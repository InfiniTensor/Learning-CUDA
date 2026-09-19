#include "kernels.cuh"
#include "contiguous256.cuh"
#include "packed_rows.cuh"
#include "row_policy.hpp"
#include "reference.hpp"

#include <array>
#include <charconv>
#include <chrono>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>

namespace {

void check_cuda(cudaError_t result, const char* expression, int line) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(expression) + " at line " +
                                 std::to_string(line) + ": " + cudaGetErrorString(result));
}
#define CUDA_CHECK(expression) check_cuda((expression), #expression, __LINE__)

void cleanup_cuda(cudaError_t result, const char* name) noexcept {
    if (result != cudaSuccess)
        std::cerr << "CUDA cleanup error (" << name << "): " << cudaGetErrorString(result) << '\n';
}

std::size_t product(std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::invalid_argument("shape or allocation size overflows size_t");
    return a * b;
}

std::size_t divide_up(std::size_t size, std::size_t divisor) {
    return (size - 1) / divisor + 1;
}

template <class T> class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count_) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), product(count_, sizeof(T))));
    }
    ~DeviceBuffer() { if (data_) cleanup_cuda(cudaFree(data_), "cudaFree"); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* data() { return data_; }
    const T* data() const { return data_; }
    std::size_t bytes() const { return product(count_, sizeof(T)); }
    void upload(const std::vector<T>& values) {
        if (values.size() != count_) throw std::logic_error("upload size mismatch");
        CUDA_CHECK(cudaMemcpy(data_, values.data(), bytes(), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> values(count_);
        CUDA_CHECK(cudaMemcpy(values.data(), data_, bytes(), cudaMemcpyDeviceToHost));
        return values;
    }
private:
    T* data_ = nullptr;
    std::size_t count_;
};

class Event {
public:
    Event() { CUDA_CHECK(cudaEventCreate(&event_)); }
    ~Event() { cleanup_cuda(cudaEventDestroy(event_), "cudaEventDestroy"); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    operator cudaEvent_t() const { return event_; }
private:
    cudaEvent_t event_{};
};

enum class Method { Naive, Warp, TensorCore, SplitInt4, FusedInt4 };
const char* method_name(Method method) {
    switch (method) {
        case Method::Naive: return "naive_global";
        case Method::Warp: return "warp";
        case Method::TensorCore: return "tensor_core";
        case Method::SplitInt4: return "split_int4";
        case Method::FusedInt4: return "fused_int4";
    }
    throw std::logic_error("unknown method");
}

struct Options {
    bool self_test = false, benchmark = false, normalized = false;
    std::size_t batch = 4, seq = 128, heads = 8, dim = 256;
    std::string dtype = "fp16", csv, fused_layout = "original", row_layout = "original";
    int repetitions = 200, warmup = 20, block_threads = 128;
    std::size_t rows() const { return product(product(batch, seq), heads); }
    std::size_t elements() const { return product(rows(), dim); }
    float scale() const { return normalized ? 1.0f / std::sqrt(static_cast<float>(dim)) : 1.0f; }
};

void help() {
    std::cout << "Hadamard CUDA benchmark; shape [batch, seq, heads, dim], last-axis transform.\n"
              << "Usage: hadamard [--self-test] [--benchmark] [options]\n"
              << "  --batch B --seq S --heads H   Positive dimensions (defaults 4,128,8)\n"
              << "  --dim N                      Power of two in [1,256], default 256\n"
              << "  --dtype fp16|bf16            Default fp16\n"
              << "  --scale 1|normalized         Default 1; normalized means 1/sqrt(N)\n"
              << "  --normalize                  Alias for --scale normalized\n"
              << "  --repetitions R --warmup W    Default 200,20; R positive, W nonnegative\n"
              << "  --block-threads 128|256       Warp transform/quantize/split/fused; default 128\n"
              << "                               Naive, Tensor Core and CPU paths are unchanged\n"
              << "  --fused-layout original|contiguous256  Fused INT4 only; default original\n"
              << "                               contiguous256 requires N256 and block_threads=128\n"
              << "                               Self-test: only N256 fused uses it; other N remain original\n"
              << "  --row-layout original|packed|auto  N<=16 row mapping; default original\n"
              << "                               auto may choose 256 threads on validated models\n"
              << "                               self-test: other N use original; incompatible with contiguous256\n"
              << "  --csv FILE                   Append measured rows, with header if empty\n"
              << "No mode selects --self-test. Self-tests always cover both dtypes/scales.\n"
              << "INT4: rowwise symmetric [-7,7], scale=max(abs(y))/7 (zero row:1),\n"
              << "ties-to-even; even element low nibble; y is rounded to output dtype first.\n";
}

std::size_t unsigned_number(const std::string& text, const std::string& argument, bool zero_ok = false) {
    std::size_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || (!zero_ok && !value))
        throw std::invalid_argument(argument + " requires a valid " + (zero_ok ? "nonnegative" : "positive") + " integer");
    return value;
}

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto next = [&]() -> std::string {
            if (++i == argc) throw std::invalid_argument("missing value for " + argument);
            return argv[i];
        };
        if (argument == "--self-test") options.self_test = true;
        else if (argument == "--benchmark") options.benchmark = true;
        else if (argument == "--normalize") options.normalized = true;
        else if (argument == "--batch") options.batch = unsigned_number(next(), argument);
        else if (argument == "--seq") options.seq = unsigned_number(next(), argument);
        else if (argument == "--heads") options.heads = unsigned_number(next(), argument);
        else if (argument == "--dim") options.dim = unsigned_number(next(), argument);
        else if (argument == "--dtype") options.dtype = next();
        else if (argument == "--csv") options.csv = next();
        else if (argument == "--fused-layout") options.fused_layout = next();
        else if (argument == "--row-layout") options.row_layout = next();
        else if (argument == "--block-threads") {
            const auto value = unsigned_number(next(), argument);
            if (value != 128 && value != 256)
                throw std::invalid_argument("--block-threads must be 128 or 256");
            options.block_threads = static_cast<int>(value);
        }
        else if (argument == "--scale") {
            const auto value = next();
            if (value == "1" || value == "1.0") options.normalized = false;
            else if (value == "normalized") options.normalized = true;
            else throw std::invalid_argument("--scale must be 1 or normalized");
        } else if (argument == "--repetitions" || argument == "--warmup") {
            const auto value = unsigned_number(next(), argument, argument == "--warmup");
            if (value > 100000) throw std::invalid_argument(argument + " must not exceed 100000");
            if (argument == "--warmup") options.warmup = static_cast<int>(value);
            else options.repetitions = static_cast<int>(value);
        } else throw std::invalid_argument("unknown option: " + argument);
    }
    if (!hadamard::power_of_two(options.dim) || options.dim > 256)
        throw std::invalid_argument("--dim must be a power of two in [1,256]");
    if (options.dtype != "fp16" && options.dtype != "bf16")
        throw std::invalid_argument("--dtype must be fp16 or bf16");
    if (options.fused_layout != "original" && options.fused_layout != "contiguous256")
        throw std::invalid_argument("--fused-layout must be original or contiguous256");
    const auto row_layout = hadamard::parse_row_layout(options.row_layout);
    if (row_layout != hadamard::RowLayout::Original && options.fused_layout != "original")
        throw std::invalid_argument("--row-layout cannot be combined with contiguous256");
    if (row_layout == hadamard::RowLayout::Packed && options.benchmark && options.dim > 16)
        throw std::invalid_argument("--benchmark with --row-layout packed requires --dim <= 16");
    if (options.fused_layout == "contiguous256") {
        if (options.block_threads != 128)
            throw std::invalid_argument("--fused-layout contiguous256 requires --block-threads 128");
        if (options.benchmark && options.dim != 256)
            throw std::invalid_argument("--benchmark with --fused-layout contiguous256 requires --dim 256");
    }
    // Validate all products before allocating or touching a CUDA device.
    product(options.elements(), sizeof(float));
    if (!options.self_test && !options.benchmark) options.self_test = true;
    return options;
}

template <class T> class Runner {
public:
    Runner(std::size_t row_count, int dimension, float transform_scale, int warp_block_threads = 128,
           bool use_contiguous256_fused = false,
           hadamard::RowLayout row_layout = hadamard::RowLayout::Original)
        : rows(row_count), dim(dimension), scale(transform_scale), block_threads(warp_block_threads),
          contiguous256_fused(use_contiguous256_fused), requested_row_layout(row_layout), count(product(rows, dim)),
          input(count), output(count), scratch_a(count), scratch_b(count),
          matrix(dim >= 16 ? dim * dim : 0),
          packed(product(rows, (dim + 1) / 2)), quant_scales(rows) {
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
        row_device_family = hadamard::row_device(properties.name);
        if (divide_up(count, 256) > static_cast<std::size_t>(properties.maxGridSize[0]) ||
            divide_up(rows, block_threads / 32) > static_cast<std::size_t>(properties.maxGridSize[0]))
            throw std::invalid_argument("shape exceeds CUDA launch grid limits");
        if (dim >= 16) {
            std::vector<T> h(dim * dim);
            for (int r = 0; r < dim; ++r)
                for (int c = 0; c < dim; ++c) {
                    int bits = r & c, sign = 1;
                    while (bits) { sign = -sign; bits &= bits - 1; }
                    h[r * dim + c] = hadamard::as_storage<T>(static_cast<float>(sign));
                }
            matrix.upload(h);
        }
    }

    void run(Method method) {
        if (method == Method::Naive) {
            const auto blocks = static_cast<unsigned int>(divide_up(count, 256));
            hadamard::to_float_kernel<<<blocks, 256>>>(input.data(), scratch_a.data(), count);
            CUDA_CHECK(cudaGetLastError());
            float* current = scratch_a.data();
            float* next = scratch_b.data();
            for (int stride = 1; stride < dim; stride *= 2) {
                hadamard::butterfly_stage<<<blocks, 256>>>(current, next, count, stride);
                CUDA_CHECK(cudaGetLastError());
                std::swap(current, next);
            }
            hadamard::from_float_kernel<<<blocks, 256>>>(current, output.data(), count, scale);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        switch (dim) {
#define DIM_CASE(n) case n: specialized<n>(method); break
            DIM_CASE(1); DIM_CASE(2); DIM_CASE(4); DIM_CASE(8); DIM_CASE(16);
            DIM_CASE(32); DIM_CASE(64); DIM_CASE(128); DIM_CASE(256);
#undef DIM_CASE
            default: throw std::logic_error("unsupported dimension");
        }
    }

    std::size_t rows;
    int dim;
    float scale;
    int block_threads;
    bool contiguous256_fused;
    hadamard::RowLayout requested_row_layout;
    hadamard::RowDevice row_device_family = hadamard::RowDevice::Unknown;
    std::size_t count;
    DeviceBuffer<T> input, output;
    DeviceBuffer<float> scratch_a, scratch_b;
    DeviceBuffer<T> matrix;
    DeviceBuffer<std::uint8_t> packed;
    DeviceBuffer<float> quant_scales;

private:
    template <int N> void specialized(Method method) {
        const auto blocks = static_cast<unsigned int>(divide_up(rows, block_threads / 32));
        if constexpr (N <= 16) {
            const auto choice = hadamard::choose_rows(requested_row_layout, row_device_family, rows, N,
                                                      method == Method::FusedInt4, block_threads);
            if (choice.packed && method != Method::TensorCore) {
                const auto packed_blocks = static_cast<unsigned int>(divide_up(rows, choice.threads / N));
                if (method == Method::FusedInt4) {
                    hadamard::packed_rows_kernel<T, N, true, true><<<packed_blocks, choice.threads>>>(
                        input.data(), nullptr, packed.data(), quant_scales.data(), rows, scale);
                    CUDA_CHECK(cudaGetLastError());
                } else {
                    hadamard::packed_rows_kernel<T, N, true, false><<<packed_blocks, choice.threads>>>(
                        input.data(), output.data(), nullptr, nullptr, rows, scale);
                    CUDA_CHECK(cudaGetLastError());
                    if (method == Method::SplitInt4) {
                        hadamard::launch_warp<T, N, false, true>(
                            output.data(), nullptr, packed.data(), quant_scales.data(), rows, 1, block_threads);
                        CUDA_CHECK(cudaGetLastError());
                    }
                }
                return;
            }
        }
        if (method == Method::TensorCore) {
            if constexpr (N >= 16) {
                const dim3 grid(static_cast<unsigned int>(divide_up(rows, 16)), N / 16);
                hadamard::tensor_core_kernel<T, N><<<grid, 128>>>(input.data(), matrix.data(), output.data(), rows, scale);
                CUDA_CHECK(cudaGetLastError());
            } else throw std::invalid_argument("tensor_core requires dim >= 16");
        } else if (method == Method::FusedInt4) {
            if constexpr (N == 256) {
                if (contiguous256_fused) {
                    hadamard::contiguous256_kernel<T, true, true><<<blocks, block_threads>>>(
                        input.data(), nullptr, packed.data(), quant_scales.data(), rows, scale);
                    CUDA_CHECK(cudaGetLastError());
                    return;
                }
            }
            hadamard::launch_warp<T, N, true, true>(input.data(), nullptr, packed.data(), quant_scales.data(), rows, scale, block_threads);
            CUDA_CHECK(cudaGetLastError());
        } else {
            hadamard::launch_warp<T, N, true, false>(input.data(), output.data(), nullptr, nullptr, rows, scale, block_threads);
            CUDA_CHECK(cudaGetLastError());
            if (method == Method::SplitInt4) {
                hadamard::launch_warp<T, N, false, true>(output.data(), nullptr, packed.data(), quant_scales.data(), rows, 1, block_threads);
                CUDA_CHECK(cudaGetLastError());
            }
        }
    }
};

template <class T> std::vector<float> to_float(const std::vector<T>& input) {
    std::vector<float> result(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) result[i] = hadamard::as_float(input[i]);
    return result;
}

template <class T> std::vector<T> make_input(std::size_t rows, int dim, const std::string& pattern, std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> distribution(-128, 128);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    std::normal_distribution<float> normal(0.0f, 0.5f);
    std::vector<T> result(product(rows, dim));
    for (std::size_t row = 0; row < rows; ++row)
        for (int j = 0; j < dim; ++j) {
            float value = 0;
            if (pattern == "random") value = static_cast<float>(distribution(generator)) / 128.0f;
            else if (pattern == "uniform") value = uniform(generator);
            else if (pattern == "normal") value = normal(generator);
            else if (pattern == "outlier")
                value = j == static_cast<int>((row * 7) % dim) ? 8.0f : uniform(generator) * 0.001f;
            else if (pattern == "impulse") value = j == static_cast<int>((row * 7) % dim) ? 0.75f : 0.0f;
            else if (pattern == "alternating") value = (j % 2 ? -1.0f : 1.0f) * (row % 2 ? 0.5f : 1.0f);
            else if (pattern != "zeros") throw std::logic_error("unknown input pattern");
            result[row * dim + j] = hadamard::as_storage<T>(value);
        }
    return result;
}

struct Validation {
    double max_error = 0;
    std::size_t dense_rows = 0;
    std::size_t rounded_warp_mismatches = 0;
    std::size_t dense_quant_byte_mismatches = 0;
    std::size_t dense_quant_scale_mismatches = 0;
};

template <class T>
Validation validate(Runner<T>& runner, const std::vector<T>& input, bool all_rows) {
    runner.input.upload(input);
    const auto floats = to_float(input);
    std::vector<std::size_t> indices;
    if (all_rows || runner.rows <= 32) {
        for (std::size_t row = 0; row < runner.rows; ++row) indices.push_back(row);
    } else {
        for (std::size_t i = 0; i < 32; ++i) indices.push_back(i * (runner.rows - 1) / 31);
    }
    std::vector<float> sample;
    sample.reserve(product(indices.size(), runner.dim));
    for (const auto row : indices)
        sample.insert(sample.end(), floats.begin() + row * runner.dim, floats.begin() + (row + 1) * runner.dim);
    const auto dense = hadamard::dense_reference(sample, runner.dim, runner.scale);
    std::vector<float> expected(dense.size());
    for (std::size_t i = 0; i < dense.size(); ++i)
        expected[i] = hadamard::as_float(hadamard::as_storage<T>(static_cast<float>(dense[i])));
    Validation validation{0, indices.size()};
    const double tolerance = std::is_same<T, __half>::value ? 1e-2 : 5e-2;
    std::vector<float> warp_values;
    std::vector<bool> warp_row_exact(indices.size(), true);
    for (const auto method : {Method::Naive, Method::Warp, Method::TensorCore}) {
        if (method == Method::TensorCore && runner.dim < 16) continue;
        runner.run(method);
        const auto actual = to_float(runner.output.download());
        for (std::size_t s = 0; s < indices.size(); ++s)
            for (int j = 0; j < runner.dim; ++j) {
                const auto index = indices[s] * runner.dim + j;
                const double difference = std::abs(static_cast<double>(actual[index]) - expected[s * runner.dim + j]);
                validation.max_error = std::max(validation.max_error, difference);
                if (method == Method::Warp && difference != 0) {
                    ++validation.rounded_warp_mismatches;
                    warp_row_exact[s] = false;
                }
                if (!std::isfinite(actual[index]) || !(difference < tolerance)) {
                    std::ostringstream message;
                    message << method_name(method) << " oracle mismatch row=" << indices[s] << " column=" << j
                            << " actual=" << actual[index] << " expected=" << expected[s * runner.dim + j]
                            << " abs_error=" << difference << " tolerance(strict)=" << tolerance;
                    throw std::runtime_error(message.str());
                }
            }
        if (method == Method::Warp) warp_values = actual;
    }
    // Check the full quantization contract, even when dense transform checking is sampled.
    const auto cpu_quant = hadamard::quantize_int4(warp_values, runner.dim);
    const auto dense_quant = hadamard::quantize_int4(expected, runner.dim);
    runner.run(Method::SplitInt4);
    const auto split_packed = runner.packed.download();
    const auto split_scales = runner.quant_scales.download();
    runner.run(Method::FusedInt4);
    const auto fused_packed = runner.packed.download();
    const auto fused_scales = runner.quant_scales.download();
    if (split_packed != fused_packed || split_packed != cpu_quant.packed)
        throw std::runtime_error("CPU/split/fused INT4 packed bytes mismatch");
    for (std::size_t row = 0; row < runner.rows; ++row)
        if (split_scales[row] != fused_scales[row] || split_scales[row] != cpu_quant.scales[row])
            throw std::runtime_error("CPU/split/fused INT4 scales mismatch at row " + std::to_string(row));
    const int bytes = (runner.dim + 1) / 2;
    for (std::size_t s = 0; s < indices.size(); ++s) {
        if (split_scales[indices[s]] != dense_quant.scales[s]) {
            if (warp_row_exact[s]) throw std::runtime_error("dense-oracle INT4 scale mismatch for exact transform row");
            ++validation.dense_quant_scale_mismatches;
        }
        for (int j = 0; j < bytes; ++j)
            if (split_packed[indices[s] * bytes + j] != dense_quant.packed[s * bytes + j]) {
                if (warp_row_exact[s]) throw std::runtime_error("dense-oracle INT4 packed bytes mismatch for exact transform row");
                ++validation.dense_quant_byte_mismatches;
            }
    }
    return validation;
}

template <class T> void self_test_dtype(const char* dtype, std::size_t& cases, Validation& totals,
                                      int block_threads, bool contiguous256_fused, hadamard::RowLayout row_layout) {
    auto record = [&](const Validation& validation) {
        totals.max_error = std::max(totals.max_error, validation.max_error);
        totals.dense_rows += validation.dense_rows;
        totals.rounded_warp_mismatches += validation.rounded_warp_mismatches;
        totals.dense_quant_byte_mismatches += validation.dense_quant_byte_mismatches;
        totals.dense_quant_scale_mismatches += validation.dense_quant_scale_mismatches;
        ++cases;
    };
    for (int dim = 1; dim <= 256; dim *= 2) {
        for (const bool normalized : {false, true})
            for (const std::size_t rows : {1u, 3u, 17u, 65u}) {
                const float scale = normalized ? 1.0f / std::sqrt(static_cast<float>(dim)) : 1.0f;
                Runner<T> runner(rows, dim, scale, block_threads, contiguous256_fused && dim == 256, row_layout);
                auto test = [&](const char* pattern, std::uint32_t seed) {
                    const auto input = make_input<T>(rows, dim, pattern, seed);
                    try {
                        record(validate(runner, input, true));
                    } catch (const std::exception& error) {
                        throw std::runtime_error(std::string(dtype) + " dim=" + std::to_string(dim) +
                            " rows=" + std::to_string(rows) + " pattern=" + pattern +
                            " seed=" + std::to_string(seed) +
                            " normalized=" + std::to_string(normalized) + ": " + error.what());
                    }
                };
                for (const auto* pattern : {"random", "zeros", "impulse", "alternating"})
                    test(pattern, 2026 + dim * 17 + rows);
                for (const auto* pattern : {"uniform", "normal", "outlier"})
                    for (const std::uint32_t seed : {2026u, 95811u, 314159u}) test(pattern, seed);
            }
        std::cout << "SELF_TEST " << dtype << " dim=" << dim << " PASS fused_layout="
                  << (contiguous256_fused && dim == 256 ? "contiguous256" : "original") << std::endl;
    }
    // Larger, non-multiple batch: full split/fused/CPU-quant comparison, 32 dense rows.
    for (const bool normalized : {false, true}) {
        Runner<T> runner(1025, 256, normalized ? 1.0f / 16.0f : 1.0f, block_threads, contiguous256_fused, row_layout);
        record(validate(runner, make_input<T>(1025, 256, "random", 95811), false));
    }
}

double event_microseconds(const std::function<void()>& launch, int warmup, int repetitions) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    Event start, stop;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repetitions; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    return static_cast<double>(milliseconds) * 1000.0 / repetitions;
}

std::string csv_quote(const std::string& value) {
    std::string result = "\"";
    for (const char c : value) { if (c == '\"') result += '\"'; result += c; }
    return result + '\"';
}

std::string timestamp() {
    const auto now = std::time(nullptr);
    const std::tm* utc = std::gmtime(&now);
    std::ostringstream value;
    value << std::put_time(utc, "%Y-%m-%dT%H:%M:%SZ");
    return value.str();
}

struct Measurement { std::string method, scope; int repetitions; double microseconds; };

void report(const Options& options, const cudaDeviceProp& gpu, const std::vector<Measurement>& measurements,
            double max_error, std::size_t dense_rows) {
    int runtime = 0, driver = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    std::ofstream csv;
    std::string csv_header = "timestamp_utc,gpu,compute_capability,cuda_runtime,cuda_driver,batch,seq,heads,dim,dtype,scale,method,scope,repetitions,mean_us,input_elements_per_second,max_abs_error,dense_oracle_rows,warp_block_threads,mean_ms,fused_layout";
    const bool detailed_rows = options.row_layout != "original";
    if (detailed_rows) csv_header += ",requested_row_layout,resolved_row_layout,quantize_block_threads";
    bool header = false;
    if (!options.csv.empty()) {
        const std::filesystem::path path(options.csv);
        if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
        header = !std::filesystem::exists(path) || std::filesystem::file_size(path) == 0;
        if (!header) {
            std::ifstream existing(path);
            std::string first_line;
            std::getline(existing, first_line);
            if (!first_line.empty() && first_line.back() == '\r') first_line.pop_back();
            if (first_line != csv_header)
                throw std::runtime_error("CSV header differs; choose a new file to preserve existing results");
        }
        csv.open(path, std::ios::app);
        if (!csv) throw std::runtime_error("cannot open CSV file: " + options.csv);
        if (header)
            csv << csv_header << '\n';
    }
    std::cout << "BENCHMARK shape=[" << options.batch << ',' << options.seq << ',' << options.heads << ','
              << options.dim << "] dtype=" << options.dtype << " scale=" << options.scale()
              << " rows=" << options.rows() << " dense_oracle_rows=" << dense_rows
              << (options.row_layout == "auto" ? " fallback_block_threads=" : " warp_block_threads=") << options.block_threads
              << " fused_layout=" << options.fused_layout << " row_layout=" << options.row_layout << '\n';
    std::cout << "Timing: kernel_only=CUDA events, allocations/H2D/matrix setup excluded;\n"
              << "cpu_compute=FP32 FWHT host wall time, input reset excluded;\n"
              << "host_e2e=pageable H2D + warp transform + D2H, preallocated buffers.\n"
              << "Throughput=input elements / elapsed seconds; it is not FLOP/s or physical memory bandwidth.\n";
    for (const auto& measured : measurements) {
        const double throughput = static_cast<double>(options.elements()) * 1e6 / measured.microseconds;
        std::cout << std::left << std::setw(20) << measured.method << std::setw(16) << measured.scope
                  << std::right << std::fixed << std::setprecision(3) << std::setw(12) << measured.microseconds
                  << " us (" << std::setprecision(6) << measured.microseconds / 1000.0 << " ms)  "
                  << std::scientific << std::setprecision(4) << throughput << " elements/s\n";
        if (csv) {
            const bool uses_warp = measured.method == "warp" || measured.method == "split_int4" ||
                                   measured.method == "fused_int4" || measured.method == "warp_h2d_d2h";
            csv << timestamp() << ',' << csv_quote(gpu.name) << ',' << gpu.major << gpu.minor << ','
                << runtime << ',' << driver << ',' << options.batch << ',' << options.seq << ',' << options.heads
                << ',' << options.dim << ',' << options.dtype << ',' << std::setprecision(9) << options.scale()
                << ',' << measured.method << ',' << measured.scope << ',' << measured.repetitions << ','
                << std::setprecision(12) << measured.microseconds << ',' << throughput << ',' << max_error << ',' << dense_rows << ',';
            const auto choice = hadamard::choose_rows(hadamard::parse_row_layout(options.row_layout),
                hadamard::row_device(gpu.name), options.rows(), static_cast<int>(options.dim),
                measured.method == "fused_int4", options.block_threads);
            if (uses_warp) csv << choice.threads;
            csv << ',' << measured.microseconds / 1000.0 << ',';
            if (measured.method == "fused_int4") csv << options.fused_layout;
            if (detailed_rows) {
                csv << ',' << options.row_layout << ',';
                if (uses_warp) csv << (choice.packed ? "packed" : "original");
                csv << ',';
                if (measured.method == "split_int4") csv << options.block_threads;
            }
            csv << '\n';
        }
    }
    if (csv) { csv.flush(); if (!csv) throw std::runtime_error("failed writing CSV"); }
}

template <class T> void benchmark(const Options& options, const cudaDeviceProp& gpu) {
    Runner<T> runner(options.rows(), static_cast<int>(options.dim), options.scale(), options.block_threads,
                     options.fused_layout == "contiguous256", hadamard::parse_row_layout(options.row_layout));
    const auto input = make_input<T>(runner.rows, runner.dim, "random", 20260905);
    const auto validation = validate(runner, input, false);
    std::vector<Measurement> measurements;
    for (const auto method : {Method::Naive, Method::Warp, Method::TensorCore, Method::SplitInt4, Method::FusedInt4}) {
        if (method == Method::TensorCore && options.dim < 16) continue;
        const double duration = event_microseconds([&]() { runner.run(method); }, options.warmup, options.repetitions);
        measurements.push_back({method_name(method), "kernel_only", options.repetitions, duration});
    }
    using Clock = std::chrono::steady_clock;
    const int host_repetitions = std::min(options.repetitions, 20);
    const auto source = to_float(input);
    std::vector<float> cpu(source.size());
    double cpu_microseconds = 0;
    volatile float cpu_result = 0;
    for (int i = 0; i < host_repetitions + 1; ++i) {
        cpu = source;
        const auto begin = Clock::now();
        hadamard::fwht(cpu.data(), runner.rows, options.dim, options.scale());
        const auto finish = Clock::now();
        cpu_result = cpu.front();
        if (i) cpu_microseconds += std::chrono::duration<double, std::micro>(finish - begin).count();
    }
    (void)cpu_result;
    measurements.push_back({"cpu_fp32_fwht", "cpu_compute", host_repetitions, cpu_microseconds / host_repetitions});
    std::vector<T> host_output(input.size());
    double e2e_microseconds = 0;
    for (int i = 0; i < host_repetitions + 1; ++i) {
        const auto begin = Clock::now();
        runner.input.upload(input);
        runner.run(Method::Warp);
        CUDA_CHECK(cudaMemcpy(host_output.data(), runner.output.data(), runner.output.bytes(), cudaMemcpyDeviceToHost));
        const auto finish = Clock::now();
        if (i) e2e_microseconds += std::chrono::duration<double, std::micro>(finish - begin).count();
    }
    measurements.push_back({"warp_h2d_d2h", "host_e2e", host_repetitions, e2e_microseconds / host_repetitions});
    report(options, gpu, measurements, validation.max_error, validation.dense_rows);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        for (int i = 1; i < argc; ++i)
            if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") { help(); return 0; }
        const auto options = parse(argc, argv);
        int devices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (!devices) throw std::runtime_error("no CUDA device is available");
        CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceProp gpu{};
        CUDA_CHECK(cudaGetDeviceProperties(&gpu, 0));
        if (gpu.major < 8) throw std::runtime_error("this build requires an sm80+ GPU for BF16 Tensor Cores");
        std::cout << "GPU=" << gpu.name << " sm=" << gpu.major << gpu.minor << '\n';
        if (options.self_test) {
            std::size_t cases = 0;
            Validation totals;
            const bool contiguous256_fused = options.fused_layout == "contiguous256";
            self_test_dtype<__half>("fp16", cases, totals, options.block_threads, contiguous256_fused, hadamard::parse_row_layout(options.row_layout));
            self_test_dtype<__nv_bfloat16>("bf16", cases, totals, options.block_threads, contiguous256_fused, hadamard::parse_row_layout(options.row_layout));
            std::cout << "SELF_TEST PASS cases=" << cases << " max_abs_error=" << totals.max_error
                      << " CPU/split/fused_INT4_bytes=exact scales=exact"
                      << " rounded_warp_vs_dense_elements=" << totals.rounded_warp_mismatches
                      << " dense_quant_differing_bytes=" << totals.dense_quant_byte_mismatches
                      << " dense_quant_differing_scales=" << totals.dense_quant_scale_mismatches
                      << (options.row_layout == "auto" ? " fallback_block_threads=" : " warp_block_threads=") << options.block_threads
                      << " row_layout=" << options.row_layout
                      << " fused_layout=" << options.fused_layout
                      << " fused_layout_scope=" << (contiguous256_fused ? "N256_only_other_N_original" : "all_N_original") << '\n';
        }
        if (options.benchmark) {
            if (options.dtype == "fp16") benchmark<__half>(options, gpu);
            else benchmark<__nv_bfloat16>(options, gpu);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        return 0;
    } catch (const std::invalid_argument& error) {
        std::cerr << "Argument error: " << error.what() << "\nUse --help for usage.\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}

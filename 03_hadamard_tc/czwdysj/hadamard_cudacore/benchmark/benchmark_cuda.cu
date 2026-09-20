#include "include/hadamard.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t error, const char* expression, const char* file,
                int line) {
    if (error == cudaSuccess) {
        return;
    }
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) +
                             " " + expression + ": " +
                             cudaGetErrorString(error));
}

#define CUDA_CHECK(expression) \
    check_cuda((expression), #expression, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer {
   public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count != 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_),
                                  count * sizeof(T)));
        }
    }
    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* data() { return data_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

   private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

template <typename T>
T from_float(float value);
template <>
float from_float<float>(float value) {
    return value;
}
template <>
__half from_float<__half>(float value) {
    return __float2half_rn(value);
}
template <>
__nv_bfloat16 from_float<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

template <typename T>
float to_float(T value);
template <>
float to_float<float>(float value) {
    return value;
}
template <>
float to_float<__half>(__half value) {
    return __half2float(value);
}
template <>
float to_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
constexpr hadamard::DType dtype_of();
template <>
constexpr hadamard::DType dtype_of<float>() {
    return hadamard::DType::kFloat32;
}
template <>
constexpr hadamard::DType dtype_of<__half>() {
    return hadamard::DType::kFloat16;
}
template <>
constexpr hadamard::DType dtype_of<__nv_bfloat16>() {
    return hadamard::DType::kBFloat16;
}

struct Options {
    std::vector<hadamard::KernelVersion> kernels{
        hadamard::KernelVersion::kV0, hadamard::KernelVersion::kV1,
        hadamard::KernelVersion::kV2};
    std::vector<std::int64_t> batches{1, 32, 128, 1024, 4096};
    std::vector<int> dimensions{128, 256, 512, 1024, 2048, 4096, 8192};
    std::string dtype = "fp16";
    int warmup = 100;
    int repeat = 1000;
    bool profile = false;
    bool kernel_was_set = false;
    bool batch_was_set = false;
    bool dim_was_set = false;
    std::string csv_path = "results_cuda.csv";
};

struct Result {
    std::string implementation;
    std::int64_t batch = 0;
    int head_dim = 0;
    std::string dtype;
    double latency_us = 0.0;
    double speedup_vs_v0 = std::numeric_limits<double>::quiet_NaN();
    bool correct = false;
    double max_abs_error = 0.0;
    double max_relative_error = 0.0;
};

long long parse_positive(const char* text, const char* name) {
    char* end = nullptr;
    const long long value = std::strtoll(text, &end, 10);
    if (end == text || *end != '\0' || value <= 0) {
        throw std::invalid_argument(std::string("invalid ") + name + ": " + text);
    }
    return value;
}

hadamard::KernelVersion parse_kernel(const std::string& value) {
    if (value == "v0" || value == "naive") return hadamard::KernelVersion::kV0;
    if (value == "v1" || value == "shared") return hadamard::KernelVersion::kV1;
    if (value == "v2" || value == "optimized") return hadamard::KernelVersion::kV2;
    throw std::invalid_argument("--kernel must be v0, v1, v2, or all");
}

void print_help(const char* program) {
    std::cout
        << "Usage: " << program
        << " [--kernel v0|v1|v2|all] [--batch N] [--dim N]\n"
           "       [--dtype fp16|bf16|fp32|all] [--warmup N] [--repeat N]\n"
           "       [--csv PATH] [--profile]\n\n"
           "Defaults: full shape grid, fp16, warmup=100, repeat=1000.\n"
           "--profile requires one explicit kernel, batch and dim; it keeps the\n"
           "normal warmup and records 3 measured operations.\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--kernel" && index + 1 < argc) {
            const std::string value = argv[++index];
            if (value == "all") {
                options.kernels = {hadamard::KernelVersion::kV0,
                                   hadamard::KernelVersion::kV1,
                                   hadamard::KernelVersion::kV2};
            } else {
                options.kernels = {parse_kernel(value)};
            }
            options.kernel_was_set = true;
        } else if (argument == "--batch" && index + 1 < argc) {
            options.batches = {parse_positive(argv[++index], "batch")};
            options.batch_was_set = true;
        } else if (argument == "--dim" && index + 1 < argc) {
            options.dimensions = {
                static_cast<int>(parse_positive(argv[++index], "dimension"))};
            options.dim_was_set = true;
        } else if (argument == "--dtype" && index + 1 < argc) {
            options.dtype = argv[++index];
            if (options.dtype != "fp16" && options.dtype != "bf16" &&
                options.dtype != "fp32" && options.dtype != "all") {
                throw std::invalid_argument(
                    "--dtype must be fp16, bf16, fp32, or all");
            }
        } else if (argument == "--warmup" && index + 1 < argc) {
            options.warmup =
                static_cast<int>(parse_positive(argv[++index], "warmup"));
        } else if (argument == "--repeat" && index + 1 < argc) {
            options.repeat =
                static_cast<int>(parse_positive(argv[++index], "repeat"));
        } else if (argument == "--csv" && index + 1 < argc) {
            options.csv_path = argv[++index];
        } else if (argument == "--profile") {
            options.profile = true;
        } else if (argument == "--help" || argument == "-h") {
            print_help(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown or incomplete argument: " +
                                        argument);
        }
    }
    if (options.profile) {
        if (!options.kernel_was_set || options.kernels.size() != 1 ||
            !options.batch_was_set || !options.dim_was_set) {
            throw std::invalid_argument(
                "--profile requires one explicit --kernel, --batch and --dim");
        }
        options.repeat = 3;
    }
    return options;
}

void cpu_reference(std::vector<float>& values, int rows, int head_dim) {
    for (int row = 0; row < rows; ++row) {
        float* data = values.data() + static_cast<std::size_t>(row) * head_dim;
        for (int stride = 1; stride < head_dim; stride <<= 1) {
            for (int base = 0; base < head_dim; base += 2 * stride) {
                for (int offset = 0; offset < stride; ++offset) {
                    const float a = data[base + offset];
                    const float b = data[base + offset + stride];
                    data[base + offset] = a + b;
                    data[base + offset + stride] = a - b;
                }
            }
        }
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        for (int col = 0; col < head_dim; ++col) data[col] *= scale;
    }
}

template <typename T>
Result run_case(hadamard::KernelVersion kernel, std::int64_t batch,
                int head_dim, const Options& options) {
    Result result;
    result.implementation = hadamard::kernel_name(kernel);
    result.batch = batch;
    result.head_dim = head_dim;
    result.dtype = hadamard::dtype_name(dtype_of<T>());

    if (!hadamard::is_power_of_two(head_dim) || head_dim < 32 ||
        head_dim > 8192) {
        std::cout << "Kernel: " << result.implementation << " Shape: [" << batch
                  << ", " << head_dim << "] dtype: " << result.dtype
                  << " unsupported\n";
        return result;
    }

    const std::size_t elements = static_cast<std::size_t>(batch) * head_dim;
    std::mt19937 generator(20260918 + head_dim);
    std::normal_distribution<float> distribution(0.0f, 0.5f);
    std::vector<T> host_input(elements);
    for (T& value : host_input) value = from_float<T>(distribution(generator));

    DeviceBuffer<T> input(elements);
    DeviceBuffer<T> output(elements);
    const std::size_t workspace_bytes = hadamard::workspace_bytes(
        kernel, batch, head_dim, dtype_of<T>());
    DeviceBuffer<unsigned char> workspace(workspace_bytes);
    CUDA_CHECK(cudaMemcpy(input.data(), host_input.data(), input.bytes(),
                          cudaMemcpyHostToDevice));

    const auto launch = [&] {
        CUDA_CHECK(hadamard::launch(
            kernel, input.data(), output.data(), workspace.data(), batch,
            head_dim, dtype_of<T>()));
    };

    // Correctness uses the first at most two rows and is outside the timer.
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    const int checked_rows = static_cast<int>(std::min<std::int64_t>(batch, 2));
    const std::size_t checked_elements =
        static_cast<std::size_t>(checked_rows) * head_dim;
    std::vector<T> host_output(checked_elements);
    CUDA_CHECK(cudaMemcpy(host_output.data(), output.data(),
                          checked_elements * sizeof(T), cudaMemcpyDeviceToHost));
    std::vector<float> expected(checked_elements);
    for (std::size_t index = 0; index < checked_elements; ++index) {
        expected[index] = to_float(host_input[index]);
    }
    cpu_reference(expected, checked_rows, head_dim);
    double max_abs = 0.0;
    double max_rel = 0.0;
    for (std::size_t index = 0; index < checked_elements; ++index) {
        const double actual = to_float(host_output[index]);
        const double absolute = std::abs(actual - expected[index]);
        const double relative =
            absolute /
            std::max(static_cast<double>(std::abs(expected[index])), 1e-6);
        max_abs = std::max(max_abs, absolute);
        max_rel = std::max(max_rel, relative);
    }
    const double atol = dtype_of<T>() == hadamard::DType::kFloat32
                            ? 2e-5
                            : (dtype_of<T>() == hadamard::DType::kFloat16
                                   ? 1e-2
                                   : (kernel == hadamard::KernelVersion::kV0
                                          ? 8e-2
                                          : 5e-2));
    result.correct = max_abs <= atol;
    result.max_abs_error = max_abs;
    result.max_relative_error = max_rel;

    for (int index = 0; index < options.warmup; ++index) launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    if (options.profile) CUDA_CHECK(cudaProfilerStart());
    CUDA_CHECK(cudaEventRecord(start));
    for (int index = 0; index < options.repeat; ++index) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    if (options.profile) CUDA_CHECK(cudaProfilerStop());
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    result.latency_us = elapsed_ms * 1000.0 / options.repeat;

    std::cout << "Kernel: " << result.implementation << '\n'
              << "Shape: [" << batch << ", " << head_dim << "]\n"
              << "dtype: " << result.dtype << '\n'
              << "Warmup: " << options.warmup << '\n'
              << "Repeat: " << options.repeat << '\n'
              << "Average latency: " << std::fixed << std::setprecision(3)
              << result.latency_us << " us\n"
              << "Correctness: " << (result.correct ? "PASS" : "FAILED")
              << " max_abs_error=" << std::scientific << result.max_abs_error
              << " max_relative_error=" << result.max_relative_error << "\n\n";
    return result;
}

template <typename T>
void run_dtype(const Options& options, std::vector<Result>& results) {
    for (const std::int64_t batch : options.batches) {
        for (const int dimension : options.dimensions) {
            for (const auto kernel : options.kernels) {
                try {
                    results.push_back(
                        run_case<T>(kernel, batch, dimension, options));
                } catch (const std::exception& error) {
                    std::cout << "Kernel: " << hadamard::kernel_name(kernel)
                              << " Shape: [" << batch << ", " << dimension
                              << "] dtype: "
                              << hadamard::dtype_name(dtype_of<T>())
                              << " unsupported/error: " << error.what() << "\n";
                }
            }
        }
    }
}

void compute_v0_speedups(std::vector<Result>& results) {
    for (Result& result : results) {
        const auto baseline = std::find_if(
            results.begin(), results.end(), [&](const Result& candidate) {
                return candidate.implementation == "hadamard_v0" &&
                       candidate.batch == result.batch &&
                       candidate.head_dim == result.head_dim &&
                       candidate.dtype == result.dtype &&
                       candidate.latency_us > 0.0;
            });
        if (baseline != results.end() && result.latency_us > 0.0) {
            result.speedup_vs_v0 = baseline->latency_us / result.latency_us;
        }
    }
}

void write_csv(const std::string& path, const std::vector<Result>& results) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot open CSV: " + path);
    output << "measurement_context,implementation,batch,head_dim,dtype,"
              "latency_us,speedup_vs_v0,speedup_vs_naive,speedup_vs_dao,speed_vs_dao,"
              "latency_ratio_vs_dao,percent_latency_gap,ratio_vs_clone,"
              "comparison_valid,correct,max_abs_error,max_relative_error,"
              "my_vs_reference_max_abs_error,"
              "my_vs_reference_max_relative_error,"
              "dao_vs_reference_max_abs_error,"
              "dao_vs_reference_max_relative_error,"
              "my_vs_dao_max_abs_error,my_vs_dao_max_relative_error\n";
    output << std::setprecision(10);
    for (const Result& result : results) {
        output << "native_cpp," << result.implementation << ',' << result.batch << ','
               << result.head_dim << ',' << result.dtype << ','
               << result.latency_us << ',';
        if (!std::isnan(result.speedup_vs_v0)) output << result.speedup_vs_v0;
        output << ',';
        if (!std::isnan(result.speedup_vs_v0)) output << result.speedup_vs_v0;
        // Cross-library ratios stay blank here: Dao is intentionally timed in
        // the same Python process before any such ratio is considered fair.
        output << ",,,,,,," << (result.correct ? "true" : "false")
               << ',' << result.max_abs_error << ','
               << result.max_relative_error << ',' << result.max_abs_error
               << ',' << result.max_relative_error << ",,,,\n";
    }
}

void print_summary(const std::vector<Result>& results) {
    std::cout << "\nSummary (speedup_vs_v0 = v0/current; >1 is faster)\n";
    std::cout << std::left << std::setw(24) << "implementation" << std::right
              << std::setw(8) << "batch" << std::setw(8) << "dim"
              << std::setw(8) << "dtype" << std::setw(14) << "latency_us"
              << std::setw(14) << "vs_v0" << std::setw(10) << "correct"
              << '\n';
    for (const Result& result : results) {
        std::cout << std::left << std::setw(24) << result.implementation
                  << std::right << std::setw(8) << result.batch << std::setw(8)
                  << result.head_dim << std::setw(8) << result.dtype
                  << std::setw(14) << std::fixed << std::setprecision(3)
                  << result.latency_us;
        if (std::isnan(result.speedup_vs_v0)) {
            std::cout << std::setw(14) << "n/a";
        } else {
            std::cout << std::setw(13) << std::setprecision(3)
                      << result.speedup_vs_v0 << 'x';
        }
        std::cout << std::setw(10) << (result.correct ? "PASS" : "FAILED")
                  << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
        std::cout << "GPU: " << properties.name << " (SM " << properties.major
                  << '.' << properties.minor << ")\n\n";
        std::vector<Result> results;
        if (options.dtype == "fp16" || options.dtype == "all")
            run_dtype<__half>(options, results);
        if (options.dtype == "bf16" || options.dtype == "all")
            run_dtype<__nv_bfloat16>(options, results);
        if (options.dtype == "fp32" || options.dtype == "all")
            run_dtype<float>(options, results);
        compute_v0_speedups(results);
        print_summary(results);
        write_csv(options.csv_path, results);
        std::cout << "CSV: " << options.csv_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}

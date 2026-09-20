#include "hadamard.cuh"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// 统一把 CUDA/cuBLAS 错误转换成异常，便于命令行测试直接报告失败位置。
#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    const cudaError_t error = (expr);                                           \
    if (error != cudaSuccess) {                                                 \
      throw std::runtime_error(std::string("CUDA failure at ") + __FILE__ +     \
                               ":" + std::to_string(__LINE__) + ": " +         \
                               cudaGetErrorString(error));                      \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(expr)                                                     \
  do {                                                                         \
    const cublasStatus_t status = (expr);                                       \
    if (status != CUBLAS_STATUS_SUCCESS) {                                      \
      throw std::runtime_error(std::string("cuBLAS failure at ") + __FILE__ +   \
                               ":" + std::to_string(__LINE__) + ": status " +  \
                               std::to_string(static_cast<int>(status)));       \
    }                                                                          \
  } while (0)

using hadamard::DataType;
using hadamard::QuantType;

struct Options {
  // 默认形状对应报告中的主要 benchmark：4×128×32×128。
  int batch = 4;
  int sequence = 128;
  int heads = 32;
  int head_dim = 128;
  int warmup = 20;
  int iterations = 200;
  int rounds = 5;
  DataType dtype = DataType::kFloat16;
  std::uint32_t sign_seed = 0;
  QuantType quant = QuantType::kInt4;
  bool tensor_core = true;
  bool self_test = false;
  std::string csv_path;
};

// 打印命令行参数说明；具体参数解析在 parse_options 中完成。
void usage(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --batch N --seq N --heads N --dim {64,128,256}\n"
      << "  --dtype {fp16,bf16} --quant {none,fp8,int8,int4}\n"
      << "  --seed N             Enable deterministic random signs (0=off)\n"
      << "  --warmup N --iters N --rounds N --csv PATH\n"
      << "  --no-tensor-core     Skip cuBLAS Tensor Core comparison\n"
      << "  --self-test          Test all dimensions, dtypes, and quantizers\n";
}

Options parse_options(int argc, char** argv) {
  Options options;
  auto value = [&](int& i) -> std::string {
    if (++i >= argc) throw std::invalid_argument("missing option value");
    return argv[i];
  };
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--batch") options.batch = std::stoi(value(i));
    else if (arg == "--seq") options.sequence = std::stoi(value(i));
    else if (arg == "--heads") options.heads = std::stoi(value(i));
    else if (arg == "--dim") options.head_dim = std::stoi(value(i));
    else if (arg == "--warmup") options.warmup = std::stoi(value(i));
    else if (arg == "--iters") options.iterations = std::stoi(value(i));
    else if (arg == "--rounds") options.rounds = std::stoi(value(i));
    else if (arg == "--seed") options.sign_seed = std::stoul(value(i));
    else if (arg == "--csv") options.csv_path = value(i);
    else if (arg == "--dtype") {
      const std::string name = value(i);
      if (name == "fp16") options.dtype = DataType::kFloat16;
      else if (name == "bf16") options.dtype = DataType::kBFloat16;
      else throw std::invalid_argument("dtype must be fp16 or bf16");
    } else if (arg == "--quant") {
      const std::string name = value(i);
      if (name == "none") options.quant = QuantType::kNone;
      else if (name == "fp8" || name == "fp8_e4m3")
        options.quant = QuantType::kFp8E4M3;
      else if (name == "int8") options.quant = QuantType::kInt8;
      else if (name == "int4") options.quant = QuantType::kInt4;
      else throw std::invalid_argument(
          "quant must be none, fp8, int8, or int4");
    } else if (arg == "--no-tensor-core") options.tensor_core = false;
    else if (arg == "--self-test") options.self_test = true;
    else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown option: " + arg);
    }
  }
  // kernel 使用编译期维度特化，因此这里只接受三种支持的 head_dim。
  if (options.batch <= 0 || options.sequence <= 0 || options.heads <= 0)
    throw std::invalid_argument("batch, seq, and heads must be positive");
  if (options.head_dim != 64 && options.head_dim != 128 &&
      options.head_dim != 256)
    throw std::invalid_argument("dim must be 64, 128, or 256");
  if (options.warmup < 0 || options.iterations <= 0)
    throw std::invalid_argument("warmup must be non-negative; iters positive");
  if (options.rounds <= 0)
    throw std::invalid_argument("rounds must be positive");
  return options;
}

std::uint32_t mix32(std::uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  return x ^ (x >> 16);
}

float sign_for(int column, std::uint32_t seed) {
  if (seed == 0) return 1.0f;
  return (mix32(static_cast<std::uint32_t>(column) ^ seed) & 1u) ? -1.0f
                                                                 : 1.0f;
}

void cpu_fht(std::vector<float>& values, std::size_t rows, int dim,
             std::uint32_t sign_seed) {
  // CPU 参考使用完全相同的 Sylvester 蝶形顺序，输入已经先舍入到
  // FP16/BF16，因此比较时只衡量 GPU kernel 本身的误差。
  for (std::size_t row = 0; row < rows; ++row) {
    float* x = values.data() + row * dim;
    for (int column = 0; column < dim; ++column)
      x[column] *= sign_for(column, sign_seed);
    for (int stride = 1; stride < dim; stride <<= 1) {
      for (int base = 0; base < dim; base += 2 * stride) {
        for (int j = 0; j < stride; ++j) {
          const float lower = x[base + j];
          const float upper = x[base + j + stride];
          x[base + j] = lower + upper;
          x[base + j + stride] = lower - upper;
        }
      }
    }
    const float normalization = 1.0f / std::sqrt(static_cast<float>(dim));
    for (int column = 0; column < dim; ++column) x[column] *= normalization;
  }
}

template <typename T>
std::vector<T> convert_input(const std::vector<float>& input,
                             std::vector<float>* rounded) {
  // 同时返回转换后的 FP32 副本，供 CPU 参考和误差比较使用。
  std::vector<T> result(input.size());
  rounded->resize(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    result[i] = T(input[i]);
    (*rounded)[i] = static_cast<float>(result[i]);
  }
  return result;
}

template <>
std::vector<__half> convert_input(const std::vector<float>& input,
                                  std::vector<float>* rounded) {
  std::vector<__half> result(input.size());
  rounded->resize(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    result[i] = __float2half(input[i]);
    (*rounded)[i] = __half2float(result[i]);
  }
  return result;
}

template <>
std::vector<__nv_bfloat16> convert_input(const std::vector<float>& input,
                                         std::vector<float>* rounded) {
  std::vector<__nv_bfloat16> result(input.size());
  rounded->resize(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    result[i] = __float2bfloat16(input[i]);
    (*rounded)[i] = __bfloat162float(result[i]);
  }
  return result;
}

template <typename T>
float output_value(T value) {
  return static_cast<float>(value);
}

template <>
float output_value(__half value) {
  return __half2float(value);
}

template <>
float output_value(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
double maximum_error(const std::vector<T>& actual,
                     const std::vector<float>& expected) {
  double error = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    error = std::max(error, std::abs(static_cast<double>(output_value(actual[i])) -
                                     expected[i]));
  }
  return error;
}

float time_cuda(const std::function<void()>& launch, int warmup,
                int iterations) {
  // 只对 GPU 工作计时：预热后用 CUDA event 包住多次异步 launch。
  for (int i = 0; i < warmup; ++i) launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));
  CUDA_CHECK(cudaEventRecord(begin));
  for (int i = 0; i < iterations; ++i) launch();
  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));
  float elapsed = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed, begin, end));
  CUDA_CHECK(cudaEventDestroy(begin));
  CUDA_CHECK(cudaEventDestroy(end));
  return elapsed / iterations;
}

float time_cuda_median(const std::function<void()>& launch, int warmup,
                       int iterations, int rounds) {
  // 正式计时使用多轮独立测量并取中位数，抑制 boost clock、温度和其他进程
  // 造成的单次离群值；rounds=1 时退化为单轮平均。
  std::vector<float> samples(static_cast<std::size_t>(rounds));
  for (int round = 0; round < rounds; ++round)
    samples[static_cast<std::size_t>(round)] =
        time_cuda(launch, warmup, iterations);
  std::sort(samples.begin(), samples.end());
  return samples[static_cast<std::size_t>(rounds / 2)];
}

class TensorCoreTransform {
 public:
  TensorCoreTransform(int dim, std::size_t rows, DataType dtype)
      : dim_(dim), rows_(rows), dtype_(dtype) {
    // 这里只为对照路径构造稠密的 H/sqrt(D) 矩阵；矩阵构造和拷贝不计入 kernel 计时。
    CUBLAS_CHECK(cublasCreate(&handle_));
    CUBLAS_CHECK(cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH));
    const std::size_t bytes = static_cast<std::size_t>(dim) * dim * 2;
    CUDA_CHECK(cudaMalloc(&matrix_, bytes));
    std::vector<float> matrix_float(static_cast<std::size_t>(dim) * dim);
    const float normalization = 1.0f / std::sqrt(static_cast<float>(dim));
    for (int row = 0; row < dim; ++row) {
      for (int column = 0; column < dim; ++column) {
        // Sylvester 构造：元素符号为 (-1)^popcount(row & column)。
        // 用 XOR 折叠求奇偶，避免 GCC 专有内建，保持 MSVC 可移植。
        unsigned parity_bits = static_cast<unsigned>(row & column);
        parity_bits ^= parity_bits >> 16;
        parity_bits ^= parity_bits >> 8;
        parity_bits ^= parity_bits >> 4;
        parity_bits ^= parity_bits >> 2;
        parity_bits ^= parity_bits >> 1;
        matrix_float[static_cast<std::size_t>(row) * dim + column] =
            ((parity_bits & 1u) ? -1.0f : 1.0f) * normalization;
      }
    }
    std::vector<float> unused;
    if (dtype == DataType::kFloat16) {
      auto matrix = convert_input<__half>(matrix_float, &unused);
      CUDA_CHECK(cudaMemcpy(matrix_, matrix.data(), bytes,
                            cudaMemcpyHostToDevice));
    } else {
      auto matrix = convert_input<__nv_bfloat16>(matrix_float, &unused);
      CUDA_CHECK(cudaMemcpy(matrix_, matrix.data(), bytes,
                            cudaMemcpyHostToDevice));
    }
  }

  ~TensorCoreTransform() {
    if (matrix_) cudaFree(matrix_);
    if (handle_) cublasDestroy(handle_);
  }

  void launch(const void* input, void* output) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cudaDataType_t data_type = dtype_ == DataType::kFloat16
                                         ? CUDA_R_16F
                                         : CUDA_R_16BF;
    // cuBLAS 按列主序解释内存：row-major 的 X 等价于 column-major 的 X^T。
    // 由于 H 对称，计算 H * X^T 后得到的内存布局正好对应 row-major 的 X * H。
    CUBLAS_CHECK(cublasGemmEx(
        handle_, CUBLAS_OP_N, CUBLAS_OP_N, dim_, static_cast<int>(rows_), dim_,
        &alpha, matrix_, data_type, dim_, input, data_type, dim_, &beta, output,
        data_type, dim_, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }

 private:
  int dim_;
  std::size_t rows_;
  DataType dtype_;
  cublasHandle_t handle_ = nullptr;
  void* matrix_ = nullptr;
};

struct DeviceBuffers {
  // 测试程序统一管理输入、变换输出、量化输出和两套 scale 缓冲区。
  void* input = nullptr;
  void* transformed = nullptr;
  void* quantized = nullptr;
  void* quantized_reference = nullptr;
  float* scales = nullptr;
  float* scales_reference = nullptr;

  ~DeviceBuffers() {
    cudaFree(input);
    cudaFree(transformed);
    cudaFree(quantized);
    cudaFree(quantized_reference);
    cudaFree(scales);
    cudaFree(scales_reference);
  }
};

std::vector<float> make_input(std::size_t elements) {
  // 固定随机种子保证不同 kernel/不同运行之间使用同一组输入；
  // 周期性放大少量元素，用来覆盖量化时的 outlier 情况。
  std::mt19937 generator(2026);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::vector<float> input(elements);
  for (std::size_t i = 0; i < elements; ++i) {
    input[i] = normal(generator);
    if (i % 997 == 0) input[i] *= 20.0f;  // 注入确定性的稀疏异常值
  }
  return input;
}

void allocate_buffers(DeviceBuffers* buffers, std::size_t elements,
                      std::size_t rows, int dim, QuantType quant) {
  // FP16/BF16 都是 2 字节；INT4 每两个元素占一个 byte，其他量化格式每元素一个 byte。
  CUDA_CHECK(cudaMalloc(&buffers->input, elements * 2));
  CUDA_CHECK(cudaMalloc(&buffers->transformed, elements * 2));
  if (quant != QuantType::kNone) {
    const std::size_t quant_bytes =
        quant == QuantType::kInt4 ? elements / 2 : elements;
    CUDA_CHECK(cudaMalloc(&buffers->quantized, quant_bytes));
    CUDA_CHECK(cudaMalloc(&buffers->quantized_reference, quant_bytes));
    CUDA_CHECK(cudaMalloc(&buffers->scales, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers->scales_reference, rows * sizeof(float)));
  }
}

template <typename T>
void upload_input(DeviceBuffers* buffers, const std::vector<float>& input,
                  std::vector<float>* rounded) {
  const auto converted = convert_input<T>(input, rounded);
  CUDA_CHECK(cudaMemcpy(buffers->input, converted.data(),
                        converted.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
double download_error(void* device_output, const std::vector<float>& reference) {
  std::vector<T> output(reference.size());
  CUDA_CHECK(cudaMemcpy(output.data(), device_output, output.size() * sizeof(T),
                        cudaMemcpyDeviceToHost));
  return maximum_error(output, reference);
}

void check_fused_quant(DeviceBuffers* buffers, std::size_t rows, int dim,
                       DataType dtype, QuantType quant, std::uint32_t seed) {
  // 同时运行融合路径和显式的“fast transform + standalone quantize”，
  // 比较量化 bytes 及每行 scale，验证融合没有改变语义。
  const std::size_t quant_bytes =
      quant == QuantType::kInt4 ? rows * dim / 2 : rows * dim;
  CUDA_CHECK(hadamard::launch_fused_quant(
      buffers->input, buffers->quantized, buffers->scales, rows, dim, dtype,
      quant, seed, nullptr));
  CUDA_CHECK(hadamard::launch_fast(buffers->input, buffers->transformed, rows,
                                   dim, dtype, seed, nullptr));
  CUDA_CHECK(hadamard::launch_quantize(
      buffers->transformed, buffers->quantized_reference,
      buffers->scales_reference, rows, dim, dtype, quant, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<std::uint8_t> fused(quant_bytes);
  std::vector<std::uint8_t> separate(quant_bytes);
  std::vector<float> fused_scales(rows);
  std::vector<float> separate_scales(rows);
  CUDA_CHECK(cudaMemcpy(fused.data(), buffers->quantized, quant_bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(separate.data(), buffers->quantized_reference,
                        quant_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(fused_scales.data(), buffers->scales,
                        rows * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(separate_scales.data(), buffers->scales_reference,
                        rows * sizeof(float), cudaMemcpyDeviceToHost));
  if (fused != separate)
    throw std::runtime_error("fused quantized bytes differ from H-then-quantize");
  float scale_error = 0.0f;
  for (std::size_t i = 0; i < rows; ++i)
    scale_error = std::max(scale_error,
                           std::abs(fused_scales[i] - separate_scales[i]));
  if (scale_error != 0.0f)
    throw std::runtime_error("fused scales differ from H-then-quantize");
}

void append_csv(const std::string& path, const std::string& kernel,
                const Options& options, std::size_t rows, float milliseconds,
                double bandwidth) {
  if (path.empty()) return;
  const bool header = !std::filesystem::exists(path) ||
                      std::filesystem::file_size(path) == 0;
  std::ofstream output(path, std::ios::app);
  if (!output) throw std::runtime_error("cannot open CSV: " + path);
  if (header)
    output << "kernel,dtype,quant,batch,seq,heads,head_dim,rows,ms,effective_GBps\n";
  output << kernel << ',' << hadamard::dtype_name(options.dtype) << ','
         << hadamard::quant_name(options.quant) << ',' << options.batch << ','
         << options.sequence << ',' << options.heads << ',' << options.head_dim
         << ',' << rows << ',' << std::fixed << std::setprecision(6)
         << milliseconds << ',' << std::setprecision(3) << bandwidth << '\n';
}

double effective_bandwidth(std::size_t input_bytes, std::size_t output_bytes,
                           float milliseconds) {
  // 这是按理论最少输入+输出字节数计算的 effective 带宽，不是 profiler 的 DRAM 带宽。
  return static_cast<double>(input_bytes + output_bytes) /
         (milliseconds * 1.0e6);
}

void run_case(const Options& options, bool verbose) {
  // 一个 case 同时负责：准备输入、正确性验证、性能计时和可选 CSV 记录。
  const std::size_t rows = static_cast<std::size_t>(options.batch) *
                           options.sequence * options.heads;
  const std::size_t elements = rows * options.head_dim;
  const auto input = make_input(elements);
  std::vector<float> rounded;
  DeviceBuffers buffers;
  allocate_buffers(&buffers, elements, rows, options.head_dim, options.quant);
  if (options.dtype == DataType::kFloat16)
    upload_input<__half>(&buffers, input, &rounded);
  else
    upload_input<__nv_bfloat16>(&buffers, input, &rounded);
  auto reference = rounded;
  cpu_fht(reference, rows, options.head_dim, options.sign_seed);

  // 先验证快速 warp FHT，再验证共享内存基线。
  CUDA_CHECK(hadamard::launch_fast(buffers.input, buffers.transformed, rows,
                                   options.head_dim, options.dtype,
                                   options.sign_seed, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  const double fast_error = options.dtype == DataType::kFloat16
                                ? download_error<__half>(buffers.transformed,
                                                         reference)
                                : download_error<__nv_bfloat16>(
                                      buffers.transformed, reference);
  const double tolerance =
      options.dtype == DataType::kFloat16 ? 1.0e-2 : 5.0e-2;
  if (fast_error >= tolerance)
    throw std::runtime_error("fast transform exceeded error tolerance: " +
                             std::to_string(fast_error));

  CUDA_CHECK(hadamard::launch_baseline(
      buffers.input, buffers.transformed, rows, options.head_dim, options.dtype,
      options.sign_seed, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  const double baseline_error =
      options.dtype == DataType::kFloat16
          ? download_error<__half>(buffers.transformed, reference)
          : download_error<__nv_bfloat16>(buffers.transformed, reference);
  if (baseline_error >= tolerance)
    throw std::runtime_error("baseline exceeded error tolerance: " +
                             std::to_string(baseline_error));

  if (options.quant != QuantType::kNone)
    check_fused_quant(&buffers, rows, options.head_dim, options.dtype,
                      options.quant, options.sign_seed);

  // 变换路径的计时只包含 kernel launch，不包含分配、拷贝和 CPU 参考计算。
  const std::size_t tensor_bytes = elements * 2;
  const float fast_ms = time_cuda_median(
      [&] {
        CUDA_CHECK(hadamard::launch_fast(
            buffers.input, buffers.transformed, rows, options.head_dim,
            options.dtype, options.sign_seed, nullptr));
      },
      options.warmup, options.iterations, options.rounds);
  const float baseline_ms = time_cuda_median(
      [&] {
        CUDA_CHECK(hadamard::launch_baseline(
            buffers.input, buffers.transformed, rows, options.head_dim,
            options.dtype, options.sign_seed, nullptr));
      },
      options.warmup, options.iterations, options.rounds);

  const double fast_gbps =
      effective_bandwidth(tensor_bytes, tensor_bytes, fast_ms);
  const double baseline_gbps =
      effective_bandwidth(tensor_bytes, tensor_bytes, baseline_ms);
  if (verbose) {
    std::cout << "shape=[" << options.batch << ',' << options.sequence << ','
              << options.heads << ',' << options.head_dim << "] dtype="
              << hadamard::dtype_name(options.dtype) << " rows=" << rows
              << " seed=" << options.sign_seed << '\n';
    std::cout << std::fixed << std::setprecision(6)
              << "  warp_fht       " << fast_ms << " ms  "
              << std::setprecision(2) << fast_gbps << " effective GB/s"
              << "  max_abs_error=" << std::scientific << fast_error << '\n'
              << std::fixed << std::setprecision(6)
              << "  shared_baseline " << baseline_ms << " ms  "
              << std::setprecision(2) << baseline_gbps << " effective GB/s"
              << "  speedup=" << baseline_ms / fast_ms << "x"
              << "  max_abs_error=" << std::scientific << baseline_error
              << '\n';
  }
  append_csv(options.csv_path, "warp_fht", options, rows, fast_ms, fast_gbps);
  append_csv(options.csv_path, "shared_baseline", options, rows, baseline_ms,
             baseline_gbps);

  if (options.quant != QuantType::kNone) {
    const std::size_t quant_bytes =
        options.quant == QuantType::kInt4 ? elements / 2 : elements;
    const float fused_ms = time_cuda_median(
        [&] {
          CUDA_CHECK(hadamard::launch_fused_quant(
              buffers.input, buffers.quantized, buffers.scales, rows,
              options.head_dim, options.dtype, options.quant,
              options.sign_seed, nullptr));
        },
        options.warmup, options.iterations, options.rounds);
    const double fused_gbps = effective_bandwidth(
        tensor_bytes, quant_bytes + rows * sizeof(float), fused_ms);
    if (verbose)
      std::cout << std::fixed << std::setprecision(6) << "  fused_"
                << hadamard::quant_name(options.quant) << "     " << fused_ms
                << " ms  " << std::setprecision(2) << fused_gbps
                << " effective GB/s  exact_match=yes\n";
    append_csv(options.csv_path,
               std::string("fused_") + hadamard::quant_name(options.quant),
               options, rows, fused_ms, fused_gbps);
  }

  // 随机符号是额外的对角操作；稠密 Tensor Core 对照只实现不带随机符号的纯 Hadamard。
  if (options.tensor_core && options.sign_seed == 0) {
    TensorCoreTransform tensor_core(options.head_dim, rows, options.dtype);
    tensor_core.launch(buffers.input, buffers.transformed);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double tensor_error =
        options.dtype == DataType::kFloat16
            ? download_error<__half>(buffers.transformed, reference)
            : download_error<__nv_bfloat16>(buffers.transformed, reference);
    if (tensor_error >= tolerance)
      throw std::runtime_error("Tensor Core result exceeded tolerance: " +
                               std::to_string(tensor_error));
    const float tensor_ms = time_cuda_median(
        [&] { tensor_core.launch(buffers.input, buffers.transformed); },
        options.warmup, options.iterations, options.rounds);
    const double tensor_gbps =
        effective_bandwidth(tensor_bytes, tensor_bytes, tensor_ms);
    if (verbose)
      std::cout << std::fixed << std::setprecision(6) << "  tensor_core_gemm "
                << tensor_ms << " ms  TC/warp_time=" << tensor_ms / fast_ms
                << "x  winner=" << (fast_ms < tensor_ms ? "warp_fht" : "tensor_core")
                << "  max_abs_error=" << std::scientific << tensor_error
                << '\n';
    append_csv(options.csv_path, "tensor_core_gemm", options, rows, tensor_ms,
               tensor_gbps);
  }
}

void self_test(const Options& base) {
  // 用 37 行刻意覆盖非 8 倍数的尾部 block，检查越界 warp 是否正确退出。
  int cases = 0;
  for (const DataType dtype : {DataType::kFloat16, DataType::kBFloat16}) {
    for (const int dim : {64, 128, 256}) {
      for (const QuantType quant : {QuantType::kFp8E4M3, QuantType::kInt8,
                                    QuantType::kInt4}) {
        Options options = base;
        options.batch = 1;
        options.sequence = 37;
        options.heads = 1;
        options.head_dim = dim;
        options.dtype = dtype;
        options.quant = quant;
        options.sign_seed = 12345;
        options.tensor_core = false;
        options.warmup = 1;
        options.iterations = 2;
        options.csv_path.clear();
        run_case(options, false);
        ++cases;
        std::cout << "PASS dtype=" << hadamard::dtype_name(dtype)
                  << " dim=" << dim << " quant=" << hadamard::quant_name(quant)
                  << '\n';
      }
    }
  }
  std::cout << "All " << cases << " correctness cases passed.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (options.self_test)
      self_test(options);
    else
      run_case(options, true);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}

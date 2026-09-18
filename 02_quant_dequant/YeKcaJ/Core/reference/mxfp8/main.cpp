#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kBlockSize = 32;
constexpr float kFp8Max = 448.0f;

// E4M3FN: 1 位符号、4 位指数、3 位尾数；exp=15,mantissa=7 保留为 NaN。
float decode_e4m3(std::uint8_t code) {
  const int sign = (code & 0x80u) ? -1 : 1;
  const int exponent = (code >> 3) & 0x0fu;
  const int mantissa = code & 0x07u;
  if (exponent == 0) return sign * std::ldexp(static_cast<float>(mantissa), -9);
  if (exponent == 15 && mantissa == 7)
    return std::numeric_limits<float>::quiet_NaN();
  return sign * std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f,
                           exponent - 7);
}

// 参考实现直接枚举候选编码，保证舍入规则清晰且便于 CUDA 对照。
// 等距时保留先枚举的编码，不是 nearest-even；NaN 和正负零在此编码为正零。
std::uint8_t encode_e4m3(float value) {
  if (std::isnan(value) || value == 0.0f) return 0;
  const float clipped = std::clamp(value, -kFp8Max, kFp8Max);
  float best_error = std::numeric_limits<float>::infinity();
  std::uint8_t best = 0;
  for (int code = 0; code < 256; ++code) {
    const float candidate = decode_e4m3(static_cast<std::uint8_t>(code));
    if (!std::isfinite(candidate)) continue;
    const float error = std::fabs(candidate - clipped);
    if (error < best_error) {
      best_error = error;
      best = static_cast<std::uint8_t>(code);
    }
  }
  return best;
}

struct Quantized {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::vector<std::uint8_t> data;
  std::vector<std::uint8_t> scales;  // 每个 32 元素 block 一个 E8M0 指数。
};

// 按行主序展开后每 32 个元素分组，量化 block 可以跨越矩阵行边界。
Quantized quantize(const std::vector<float>& input, std::size_t rows,
                  std::size_t cols) {
  assert(input.size() == rows * cols);
  Quantized out{rows, cols, std::vector<std::uint8_t>(input.size()), {}};
  const std::size_t blocks = (input.size() + kBlockSize - 1) / kBlockSize;
  out.scales.resize(blocks);
  for (std::size_t block = 0; block < blocks; ++block) {
    const std::size_t begin = block * kBlockSize;
    const std::size_t end = std::min(begin + kBlockSize, input.size());
    float max_abs = 0.0f;
    for (std::size_t i = begin; i < end; ++i)
      max_abs = std::max(max_abs, std::fabs(input[i]));

    // 选择 2 的幂 scale，使 block 最大值缩放后不超过 E4M3 最大值 448。
    int exponent = 127;
    if (max_abs > 0.0f) {
      exponent = static_cast<int>(std::ceil(std::log2(max_abs / kFp8Max))) + 127;
      exponent = std::clamp(exponent, 0, 254);
    }
    out.scales[block] = static_cast<std::uint8_t>(exponent);
    const float scale = std::ldexp(1.0f, exponent - 127);
    for (std::size_t i = begin; i < end; ++i)
      out.data[i] = encode_e4m3(input[i] / scale);
  }
  return out;
}

// E8M0 字节保存带 127 偏置的指数，恢复为 2^(code-127) 后乘解码值。
std::vector<float> dequantize(const Quantized& q) {
  std::vector<float> output(q.rows * q.cols);
  for (std::size_t block = 0; block < q.scales.size(); ++block) {
    const float scale = std::ldexp(1.0f, static_cast<int>(q.scales[block]) - 127);
    const std::size_t begin = block * kBlockSize;
    const std::size_t end = std::min(begin + kBlockSize, output.size());
    for (std::size_t i = begin; i < end; ++i)
      output[i] = decode_e4m3(q.data[i]) * scale;
  }
  return output;
}

// 文件布局：magic、版本号、矩阵尺寸、block size、区段长度、data、scales。
void write_binary(const std::string& path, const Quantized& q) {
  std::ofstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open output: " + path);
  const std::uint64_t rows = q.rows, cols = q.cols;
  const std::uint32_t block = static_cast<std::uint32_t>(kBlockSize);
  const std::uint64_t data_bytes = q.data.size(), scale_bytes = q.scales.size();
  file.write("MXFP8Q1", 8);
  const std::uint32_t version = 1;
  file.write(reinterpret_cast<const char*>(&version), sizeof(version));
  file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
  file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
  file.write(reinterpret_cast<const char*>(&block), sizeof(block));
  file.write(reinterpret_cast<const char*>(&data_bytes), sizeof(data_bytes));
  file.write(reinterpret_cast<const char*>(&scale_bytes), sizeof(scale_bytes));
  file.write(reinterpret_cast<const char*>(q.data.data()), static_cast<std::streamsize>(q.data.size()));
  file.write(reinterpret_cast<const char*>(q.scales.data()), static_cast<std::streamsize>(q.scales.size()));
}

// 保存固定 FP32 输入，供后续 CUDA 与 CPU 使用完全相同的数据。
void write_fp32_input(const std::string& path, std::size_t rows, std::size_t cols,
                      const std::vector<float>& values) {
  std::ofstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open input output: " + path);
  const std::uint32_t version = 1;
  const std::uint64_t row_count = rows, col_count = cols;
  file.write("FP32INP1", 8);
  file.write(reinterpret_cast<const char*>(&version), sizeof(version));
  file.write(reinterpret_cast<const char*>(&row_count), sizeof(row_count));
  file.write(reinterpret_cast<const char*>(&col_count), sizeof(col_count));
  file.write(reinterpret_cast<const char*>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(float)));
}

// 读取固定 FP32 输入并校验 magic、版本和矩阵尺寸。
std::vector<float> read_fp32_input(const std::string& path, std::size_t& rows,
                                   std::size_t& cols) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open FP32 input: " + path);
  char magic[8];
  std::uint32_t version = 0;
  std::uint64_t row_count = 0, col_count = 0;
  file.read(magic, 8);
  file.read(reinterpret_cast<char*>(&version), sizeof(version));
  file.read(reinterpret_cast<char*>(&row_count), sizeof(row_count));
  file.read(reinterpret_cast<char*>(&col_count), sizeof(col_count));
  if (std::string(magic, 8) != "FP32INP1" || version != 1)
    throw std::runtime_error("invalid FP32 input header: " + path);
  rows = static_cast<std::size_t>(row_count);
  cols = static_cast<std::size_t>(col_count);
  std::vector<float> values(rows * cols);
  file.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!file) throw std::runtime_error("truncated FP32 input: " + path);
  return values;
}

// 保存 FP32 反量化结果，作为逐元素比较的 golden 文件。
void write_dequant(const std::string& path, std::size_t rows, std::size_t cols,
                   const std::vector<float>& values) {
  std::ofstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open dequant output: " + path);
  const std::uint32_t version = 1;
  const std::uint64_t row_count = rows, col_count = cols, count = values.size();
  file.write("FP32DEQ1", 8);
  file.write(reinterpret_cast<const char*>(&version), sizeof(version));
  file.write(reinterpret_cast<const char*>(&row_count), sizeof(row_count));
  file.write(reinterpret_cast<const char*>(&col_count), sizeof(col_count));
  file.write(reinterpret_cast<const char*>(&count), sizeof(count));
  file.write(reinterpret_cast<const char*>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(float)));
}

// 读取量化文件，检查版本、block size 和数据长度。
Quantized read_binary(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open quantized file: " + path);
  char magic[8];
  std::uint32_t version = 0, block = 0;
  std::uint64_t rows = 0, cols = 0, data_bytes = 0, scale_bytes = 0;
  file.read(magic, 8);
  file.read(reinterpret_cast<char*>(&version), sizeof(version));
  file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
  file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
  file.read(reinterpret_cast<char*>(&block), sizeof(block));
  file.read(reinterpret_cast<char*>(&data_bytes), sizeof(data_bytes));
  file.read(reinterpret_cast<char*>(&scale_bytes), sizeof(scale_bytes));
  if (std::string(magic, 7) != "MXFP8Q1" || version != 1 || block != kBlockSize)
    throw std::runtime_error("invalid MXFP8 quantized header: " + path);
  Quantized q{static_cast<std::size_t>(rows), static_cast<std::size_t>(cols),
              std::vector<std::uint8_t>(static_cast<std::size_t>(data_bytes)),
              std::vector<std::uint8_t>(static_cast<std::size_t>(scale_bytes))};
  file.read(reinterpret_cast<char*>(q.data.data()), static_cast<std::streamsize>(q.data.size()));
  file.read(reinterpret_cast<char*>(q.scales.data()), static_cast<std::streamsize>(q.scales.size()));
  if (!file) throw std::runtime_error("truncated MXFP8 quantized file: " + path);
  return q;
}

// 读取 FP32 反量化 golden 文件。
std::vector<float> read_dequant(const std::string& path, std::size_t& rows,
                                std::size_t& cols) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open dequant file: " + path);
  char magic[8];
  std::uint32_t version = 0;
  std::uint64_t row_count = 0, col_count = 0, count = 0;
  file.read(magic, 8);
  file.read(reinterpret_cast<char*>(&version), sizeof(version));
  file.read(reinterpret_cast<char*>(&row_count), sizeof(row_count));
  file.read(reinterpret_cast<char*>(&col_count), sizeof(col_count));
  file.read(reinterpret_cast<char*>(&count), sizeof(count));
  if (std::string(magic, 8) != "FP32DEQ1" || version != 1)
    throw std::runtime_error("invalid dequant header: " + path);
  rows = static_cast<std::size_t>(row_count);
  cols = static_cast<std::size_t>(col_count);
  if (count != rows * cols) throw std::runtime_error("invalid dequant size: " + path);
  std::vector<float> values(static_cast<std::size_t>(count));
  file.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!file) throw std::runtime_error("truncated dequant file: " + path);
  return values;
}

// 生成固定测试集，写出输入/golden，并回读执行字节级和元素级校验。
// freeze 写入固定输入和 golden；verify 只读取它们进行比较并写验证报告。
void run_reference_suite(bool regenerate) {
  namespace fs = std::filesystem;
  fs::create_directories("tests/data");
  fs::create_directories("tests/golden");
  std::ofstream report(regenerate ? "tests/FREEZE_TEST_RESULTS.txt"
                                  : "tests/VERIFY_TEST_RESULTS.txt");
  if (!report) throw std::runtime_error("cannot create freeze report");
  report << "format_version=1\nalgorithm=MXFP8_E4M3FN_E8M0_BLOCK32\n"
         << "mode=" << (regenerate ? "freeze" : "verify") << "\n";
  struct TestCase { std::string name; std::size_t rows; std::size_t cols; std::vector<float> values; };
  // 固定种子和矩阵形状覆盖零值、正常值、离群值和尾部不足整块。
  std::vector<TestCase> cases;
  cases.push_back({"zeros", 2, 17, std::vector<float>(34, 0.0f)});
  std::vector<float> basic(32);
  for (std::size_t i = 0; i < basic.size(); ++i) basic[i] = static_cast<float>(i) - 15.5f;
  cases.push_back({"basic", 2, 16, basic});
  std::mt19937 generator(1234);
  std::normal_distribution<float> distribution(0.0f, 3.0f);
  std::vector<float> outlier(185);
  for (float& value : outlier) value = distribution(generator);
  outlier[0] = 1000.0f;
  outlier.back() = -0.0f;
  cases.push_back({"outlier_tail", 5, 37, outlier});
  std::vector<float> negative_tail(33);
  for (std::size_t i = 0; i < negative_tail.size(); ++i)
    negative_tail[i] = -0.25f * static_cast<float>(i);
  cases.push_back({"tail_block", 1, 33, negative_tail});
  std::mt19937 random_generator(4321);
  std::normal_distribution<float> random_distribution(0.0f, 1.0f);
  std::vector<float> random_values(7 * 64);
  for (float& value : random_values) value = random_distribution(random_generator);
  cases.push_back({"random", 7, 64, random_values});
  for (const auto& test : cases) {
    const std::string input_path = "tests/data/" + test.name + ".fp32";
    const std::string quant_path = "tests/golden/" + test.name + ".mxfp8";
    const std::string dequant_path = "tests/golden/" + test.name + ".dequant.fp32";
    if (regenerate) write_fp32_input(input_path, test.rows, test.cols, test.values);
    const Quantized quantized = quantize(test.values, test.rows, test.cols);
    const auto dequantized = dequantize(quantized);
    if (regenerate) {
      write_binary(quant_path, quantized);
      write_dequant(dequant_path, test.rows, test.cols, dequantized);
    }
    std::size_t input_rows = 0, input_cols = 0;
    const auto loaded_input = read_fp32_input(input_path, input_rows, input_cols);
    if (input_rows != test.rows || input_cols != test.cols || loaded_input != test.values)
      throw std::runtime_error("input file comparison failed: " + test.name);
    const Quantized loaded_quantized = read_binary(quant_path);
    if (loaded_quantized.rows != quantized.rows || loaded_quantized.cols != quantized.cols ||
        loaded_quantized.data != quantized.data || loaded_quantized.scales != quantized.scales)
      throw std::runtime_error("quantized byte comparison failed: " + test.name);
    std::size_t loaded_rows = 0, loaded_cols = 0;
    const auto loaded_dequantized = read_dequant(dequant_path, loaded_rows, loaded_cols);
    if (loaded_rows != test.rows || loaded_cols != test.cols || loaded_dequantized.size() != dequantized.size())
      throw std::runtime_error("dequant shape comparison failed: " + test.name);
    float max_diff = 0.0f;
    for (std::size_t i = 0; i < dequantized.size(); ++i)
      max_diff = std::max(max_diff, std::fabs(dequantized[i] - loaded_dequantized[i]));
    if (max_diff > 1e-6f) throw std::runtime_error("dequant element comparison failed: " + test.name);
    // 此处统计原始输入到反量化结果的损失，与前面的 golden 一致性误差不同。
    double mae = 0.0, mse = 0.0;
    float max_error = 0.0f;
    for (std::size_t i = 0; i < test.values.size(); ++i) {
      const float error = std::fabs(test.values[i] - dequantized[i]);
      max_error = std::max(max_error, error);
      mae += error;
      mse += static_cast<double>(error) * error;
    }
    mae /= test.values.size();
    mse /= test.values.size();
    report << test.name << " rows=" << test.rows << " cols=" << test.cols
           << " blocks=" << quantized.scales.size() << " max_dequant_diff=" << max_diff
           << " max_abs_error=" << max_error << " mae=" << mae << " mse=" << mse << "\n";
  }
  report << "status=PASS\n";
}

void print_error(const std::vector<float>& input, const std::vector<float>& output) {
  double mae = 0.0, mse = 0.0;
  float max_error = 0.0f;
  for (std::size_t i = 0; i < input.size(); ++i) {
    const float error = std::fabs(input[i] - output[i]);
    max_error = std::max(max_error, error);
    mae += error;
    mse += static_cast<double>(error) * error;
  }
  mae /= input.size();
  mse /= input.size();
  std::cout << std::setprecision(8) << "max_abs_error=" << max_error
            << " mae=" << mae << " mse=" << mse << '\n';
}

void self_test() {
  // 5x37 不是 32 的整数倍，用于覆盖尾部 block；同时加入异常值和负零。
  constexpr std::size_t rows = 5, cols = 37;
  std::mt19937 generator(1234);
  std::normal_distribution<float> distribution(0.0f, 3.0f);
  std::vector<float> input(rows * cols);
  for (float& value : input) value = distribution(generator);
  input[0] = 1000.0f;
  input.back() = -0.0f;
  const Quantized q = quantize(input, rows, cols);
  assert(q.data.size() == input.size());
  assert(q.scales.size() == (input.size() + kBlockSize - 1) / kBlockSize);
  const auto output = dequantize(q);
  for (float value : output) {
    if (!std::isfinite(value)) throw std::runtime_error("non-finite MXFP8 output");
  }
  print_error(input, output);
  std::cout << "self_test=pass blocks=" << q.scales.size() << '\n';
}

}  // namespace

// 无测试选项时生成指定尺寸的正态分布矩阵，保存量化文件并打印损失。
int main(int argc, char** argv) {
  try {
    if (argc > 1 && std::string(argv[1]) == "--self-test") {
      self_test();
      return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--freeze") {
      run_reference_suite(true);
      std::cout << "freeze_reference=pass\n";
      return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--verify") {
      run_reference_suite(false);
      std::cout << "verify_reference=pass\n";
      return 0;
    }
    const std::size_t rows = argc > 1 ? std::stoull(argv[1]) : 128;
    const std::size_t cols = argc > 2 ? std::stoull(argv[2]) : 128;
    const std::string output_path = argc > 3 ? argv[3] : "mxfp8.bin";
    std::mt19937 generator(1234);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    std::vector<float> input(rows * cols);
    for (float& value : input) value = distribution(generator);
    const Quantized q = quantize(input, rows, cols);
    const auto output = dequantize(q);
    write_binary(output_path, q);
    print_error(input, output);
    std::cout << "wrote=" << output_path << " rows=" << rows << " cols=" << cols
              << " blocks=" << q.scales.size() << '\n';
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}

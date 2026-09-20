// 测试模块用途：保证实验报告中的误差与压缩率可信，避免算法正确但统计错误。
// 输入：可手算数组、极值、字节数和真实CPU量化结果；输出：指标检查结论。
// 主流程关系：覆盖计算后的评估环节，不测试量化格式的编解码实现。
#include "metrics.h"
#include "mxfp8.h"
#include "nvfp4.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
// 先用可手算数组验证公式，再用真实量化结果验证负载统计。
// close使用小容差比较非整数double指标；编码测试通常要求字节完全相等。
unsigned checks = 0;
void check(bool condition, const std::string& message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}

bool close(double actual, double expected) {
  return std::isfinite(actual) &&
         std::fabs(actual - expected) <= 1.0e-12 * std::max(1.0, std::fabs(expected));
}

template <typename Function>
void expect_invalid(Function function, const std::string& message) {
  bool rejected = false;
  try { function(); } catch (const std::invalid_argument&) { rejected = true; }
  check(rejected, message);
}

void test_manual_errors() {
  const auto result = luxorion::calculate_metrics({1, 2, 3}, {1, 1, 5});
  check(result.max_abs == 2.0, "manual MaxAbs = 2");
  check(result.mae == 1.0, "manual MAE = 1");
  check(close(result.mse, 5.0 / 3.0), "manual MSE = 5/3");
  const auto identical = luxorion::calculate_metrics({-3, -0.0F, 0, 4}, {-3, 0, -0.0F, 4});
  check(identical.max_abs == 0 && identical.mae == 0 && identical.mse == 0,
        "identical numerical values and opposite zero signs have zero error");
  const auto single = luxorion::calculate_metrics({-1}, {2});
  check(single.max_abs == 3 && single.mae == 3 && single.mse == 9, "single element error");
  const auto fractional = luxorion::calculate_metrics({0.5F, -0.5F}, {0, 0});
  check(fractional.max_abs == 0.5 && fractional.mae == 0.5 && fractional.mse == 0.25,
        "fractional errors");
}

void test_large_errors() {
  const float maximum = std::numeric_limits<float>::max();
  const auto result = luxorion::calculate_metrics({maximum}, {-maximum});
  const double expected = 2.0 * static_cast<double>(maximum);
  check(std::isfinite(result.max_abs) && std::isfinite(result.mae) && std::isfinite(result.mse),
        "finite FP32 inputs produce finite double error metrics");
  check(result.max_abs == expected && result.mae == expected, "subtraction takes place in double");
  check(close(result.mse, expected * expected), "squared error takes place in double");
}

void test_compression_and_real_payloads() {
  const auto manual = luxorion::calculate_compression(68, 22);
  check(manual.original_bytes == 68 && manual.quantized_payload_bytes == 22,
        "compression stores actual byte counts");
  check(close(manual.compression_ratio, 68.0 / 22.0), "manual NVFP4 ratio");
  const auto expansion = luxorion::calculate_compression(4, 13);
  check(close(expansion.compression_ratio, 4.0 / 13.0) && expansion.compression_ratio < 1,
        "small payload may expand instead of compress");

  std::vector<float> input(17, 0.0F);
  input[0] = 2688.0F;
  input[16] = 6.0F;
  const auto nv = luxorion::nvfp4_quantize_cpu(input);
  const std::size_t nv_payload = nv.packed_data.size() + nv.block_scales.size() + 4;
  const auto nv_stats = luxorion::calculate_compression(input.size() * 4, nv_payload);
  check(nv.packed_data.size() == 16 && nv.block_scales.size() == 2 && nv_payload == 22,
        "NVFP4 payload includes padded block and FP32 global scale");
  check(close(nv_stats.compression_ratio, 68.0 / 22.0), "FP32 NVFP4 actual ratio");
  const auto half_stats = luxorion::calculate_compression(input.size() * 2, nv_payload);
  check(close(half_stats.compression_ratio, 34.0 / 22.0), "FP16 input uses two bytes per element");
  const auto nv_error = luxorion::calculate_metrics(input, luxorion::nvfp4_dequantize_cpu(nv));
  check(nv_error.max_abs == 0 && nv_error.mae == 0 && nv_error.mse == 0,
        "padded elements excluded from real pipeline errors");

  input.assign(17, 1.0F);
  const auto mx = luxorion::mxfp8_quantize_cpu(input);
  const auto mx_stats = luxorion::calculate_compression(input.size() * 4,
                                                       mx.data.size() + mx.scales.size());
  check(mx_stats.quantized_payload_bytes == 18, "MXFP8 has 17 data bytes and one scale byte");
  check(close(mx_stats.compression_ratio, 68.0 / 18.0), "MXFP8 actual ratio");
  const auto mx_error = luxorion::calculate_metrics(input, luxorion::mxfp8_dequantize_cpu(mx));
  check(mx_error.max_abs == 0 && mx_error.mae == 0 && mx_error.mse == 0,
        "real MXFP8 exact restoration");
}

void test_invalid_inputs() {
  expect_invalid([] { luxorion::calculate_metrics({}, {}); }, "empty arrays");
  expect_invalid([] { luxorion::calculate_metrics({1}, {}); }, "empty restored array");
  expect_invalid([] { luxorion::calculate_metrics({1}, {1, 2}); }, "mismatched lengths");
  for (float bad : {std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity()}) {
    expect_invalid([bad] { luxorion::calculate_metrics({bad}, {0}); }, "nonfinite reference");
    expect_invalid([bad] { luxorion::calculate_metrics({0}, {bad}); }, "nonfinite restoration");
  }
  expect_invalid([] { luxorion::calculate_compression(0, 22); }, "zero original bytes");
  expect_invalid([] { luxorion::calculate_compression(68, 0); }, "zero payload bytes");
  expect_invalid([] { luxorion::calculate_compression(0, 0); }, "both byte counts zero");
}
}  // namespace

int main() {
  try {
    test_manual_errors();
    test_large_errors();
    test_compression_and_real_payloads();
    test_invalid_inputs();
    std::cout << "Metrics: " << checks << " checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Metrics test failed: " << error.what() << '\n';
    return 1;
  }
}

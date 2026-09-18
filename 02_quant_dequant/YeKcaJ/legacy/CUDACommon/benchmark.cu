#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// CMake 用 BENCH_NVFP4 选择格式；直接包含实际算子源码，不另写测试版算法。
// 临时改名原 main，避免与本文件的性能测试入口冲突。
#ifdef BENCH_NVFP4
#define main cuda_application_main
#include "../CUDANVFP4/main.cu"
#undef main
#else
#define main cuda_application_main
#include "../CUDAMXFP8/main.cu"
#undef main
#endif

namespace {

// 偶数个样本时取排序后中间偏大的一个值。
template <typename F>
double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

// 当前 P95 取 floor(0.95 * (n - 1)) 位置，不做插值。
template <typename F>
double p95(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t index = static_cast<std::size_t>(0.95 * (values.size() - 1));
  return values[index];
}

// 测量完整主机调用，包含显存分配、传输、同步和释放，不是纯 kernel 时间。
template <typename F>
void run_one(const char* op, std::size_t count, int repeats, F&& call,
             double bytes_per_run) {
  std::vector<double> samples;
  samples.reserve(repeats);
  // 预热不计入样本，减小首次调用初始化对结果的影响。
  for (int i = 0; i < 3; ++i) call();
  for (int i = 0; i < repeats; ++i) {
    const auto begin = std::chrono::steady_clock::now();
    call();
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double med = median<F>(samples);
  const double tail = p95<F>(samples);
  // 按逻辑输入输出字节数计算有效吞吐，不代表实际显存总流量。
  const double gbps = bytes_per_run / (med * 1.0e6);
  std::cout << "op=" << op << " elements=" << count << " repeats=" << repeats
            << " median_ms=" << std::setprecision(8) << med
            << " p95_ms=" << tail << " effective_GBps=" << gbps << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    // 第一个参数为重复次数，使用时应传正整数；输入生成在计时区间之外。
    const int repeats = argc > 1 ? std::stoi(argv[1]) : 20;
    const std::vector<std::size_t> sizes = {1u << 20, 4u << 20, 16u << 20};
    // 固定种子生成正态分布；三个规模依次使用同一个随机数序列。
    std::mt19937 generator(20260908);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    for (const std::size_t count : sizes) {
      std::vector<float> values(count);
      for (float& value : values) value = distribution(generator);
#ifdef BENCH_NVFP4
      FloatInput input{1, count, std::move(values)};
      QuantizedFile quantized;
      run_one("quantize", count, repeats, [&] {
        float ignored_ms = 0.0f;
        quantized = quantize_cuda(input, ignored_ms);
      }, static_cast<double>(count * sizeof(float) + (count + 1) / 2 +
                             (count + kBlockSize - 1) / kBlockSize));
      float ignored_ms = 0.0f;
      quantized = quantize_cuda(input, ignored_ms);
      run_one("dequantize_fp32", count, repeats, [&] {
        float ignored_ms2 = 0.0f;
        (void)dequantize_cuda<float>(quantized, ignored_ms2);
      }, static_cast<double>((count + 1) / 2 + (count + kBlockSize - 1) / kBlockSize + count * sizeof(float)));
#else
      FloatInput input{1, count, std::move(values)};
      QuantizedFile quantized;
      run_one("quantize", count, repeats, [&] {
        float ignored_ms = 0.0f;
        quantized = quantize_cuda(input, ignored_ms);
      }, static_cast<double>(count * sizeof(float) + count + (count + kBlockSize - 1) / kBlockSize));
      float ignored_ms = 0.0f;
      quantized = quantize_cuda(input, ignored_ms);
      run_one("dequantize_fp32", count, repeats, [&] {
        float ignored_ms2 = 0.0f;
        (void)dequantize_cuda<float>(quantized, ignored_ms2);
      }, static_cast<double>(count + (count + kBlockSize - 1) / kBlockSize + count * sizeof(float)));
#endif
    }
  } catch (const std::exception& error) {
    std::cerr << "benchmark=FAIL " << error.what() << '\n';
    return 1;
  }
  return 0;
}

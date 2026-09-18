// FP32 驻留 GPU、host_api 和 CPU 基准；独立于普通文件任务。
#include "app/commands.h"
#include "runtime/workspace.cuh"
#include "reference/oracle.h"
#include "common/timing.h"
#include "io/tensor_io.h"
#include <filesystem>
#include <fstream>
#include <cmath>
#include <iostream>
#include <random>

namespace pipeline {

// 调用方提供非空样本；中位数取中间值/均值，其他分位数使用 nearest-rank。
double percentile(std::vector<double> x, double p) {
  std::sort(x.begin(), x.end());
  if (p == .5 && x.size() % 2 == 0)
    return (x[x.size() / 2 - 1] + x[x.size() / 2]) / 2;
  return x[static_cast<std::size_t>(std::ceil(p * x.size())) - 1];
}

// 输出一条机器可读的性能记录；bytes 是逻辑读写量，不是实测 DRAM 流量。
void stats(const char* op, const char* scope, std::size_t n, const std::vector<double>& samples,
           double bytes) {
  const double med = percentile(samples, .5);
  std::cout << "{\"format\":\"" << format_name << "\",\"op\":\"" << op << "\",\"scope\":\"" << scope
            << "\",\"elements\":" << n << ",\"repeats\":" << samples.size()
            << ",\"median_ms\":" << med << ",\"p95_ms\":" << percentile(samples, .95)
            << ",\"logical_GBps\":" << (med > 0 ? bytes / (med * 1e6) : 0) << "}\n";
}

// 单一输出类型的驻留 GPU 测试：预热 3 次，复用显存，仅计反量化 kernel。
template <typename T>
void bench_dequant(Workspace& w, int repeats) {
  for (int i = 0; i < 3; ++i)
    w.events.measure([&] { w.launch_dequant<T>(); });
  std::vector<double> times;
  for (int i = 0; i < repeats; ++i)
    times.push_back(w.events.measure([&] { w.launch_dequant<T>(); }));
  const std::string op = std::string("dequant_") + cuda_output::Format<T>::name;
  stats(op.c_str(), "resident_gpu", w.n, times,
        w.n * sizeof(T) + data_size(w.n) + w.groups + (nvfp4 ? 4 : 0));
}

// 固定 FP32 正态输入与 seed，测试 block + nearest；n 和重复次数来自命令行。
// quant_enumeration 是内部算法对照，quant_optimized 才是当前默认实现。
// 做后续优化时，应比较修改前后 quant_optimized，而非把内部对照当作当前 B0。
Tensor benchmark_input(std::size_t n) {
  if (!n || n > (1u << 26)) throw std::runtime_error("invalid benchmark input size");
  Tensor t{1, n, 4, std::vector<float>(n)};
  std::mt19937 rng(20260909);
  std::normal_distribution<float> normal;
  for (auto& v : t.values) v = normal(rng);
  return t;
}

// 导出旧基准使用的正态输入；正式测试随后读取文件并由 Python 记录 SHA256。
void export_benchmark_input(std::size_t n, const std::string& path) {
  if (std::filesystem::exists(path)) throw std::runtime_error("benchmark input already exists");
  const auto t = benchmark_input(n);
  std::ofstream f(path, std::ios::binary);
  f.write("FP32INP1", 8);
  write_scalar<std::uint32_t>(f, 1);
  f.write(reinterpret_cast<const char*>(&t.rows), 8);
  f.write(reinterpret_cast<const char*>(&t.cols), 8);
  f.write(reinterpret_cast<const char*>(t.values.data()), n * sizeof(float));
  if (!f) throw std::runtime_error("failed to export benchmark input");
}

void benchmark(std::size_t n, int repeats, const std::string& input_file) {
  if (!n || n > (1u << 26) || repeats < 1 || repeats > 1000)
    throw std::runtime_error("benchmark: n must be 1..2^26, repeats 1..1000");
  Tensor t = input_file.empty() ? benchmark_input(n) : read_input(input_file);
  if (t.element_bytes != 4 || t.rows != 1 || t.cols != n)
    throw std::runtime_error("benchmark requires FP32 input with shape 1 x elements");
  Options o;
  Workspace w(n, o);
  w.upload(t.values);
  for (int i = 0; i < 3; ++i) {
    w.events.measure([&] { w.launch_quant(true); });
    w.events.measure([&] { w.launch_quant(); });
  }
  std::vector<double> baseline, optimized;
  // 交替顺序减小 GPU 温度、时钟漂移对基线和优化版本比较的影响。
  for (int i = 0; i < repeats; ++i) {
    if (i % 2) {
      optimized.push_back(w.events.measure([&] { w.launch_quant(); }));
      baseline.push_back(w.events.measure([&] { w.launch_quant(true); }));
    } else {
      baseline.push_back(w.events.measure([&] { w.launch_quant(true); }));
      optimized.push_back(w.events.measure([&] { w.launch_quant(); }));
    }
  }
  const double bytes = 4 * n + data_size(n) + w.groups + (nvfp4 ? 4 : 0);
  stats("quant_enumeration", "resident_gpu", n, baseline, bytes);
  stats("quant_optimized", "resident_gpu", n, optimized, bytes);
  // 先确认两条量化路径逐字节一致，再用现有结果测试三种反量化输出。
  w.events.measure([&] { w.launch_quant(); });
  auto q = w.download(t);
  w.events.measure([&] { w.launch_quant(true); });
  compare_packed(q, w.download(t));
  bench_dequant<float>(w, repeats);
  bench_dequant<__half>(w, repeats);
  bench_dequant<__nv_bfloat16>(w, repeats);
  // 两条 host 路径均包含上传、量化、下载和主机结果分配/释放。
  // reused 在计时外申请显存和 event；交替测量顺序，减少时钟漂移的影响。
  Workspace reused(n, o);
  std::vector<double> wall, reused_wall;
  auto fresh_call = [&] {
    const auto start = Clock::now();
    {
      Workspace fresh(n, o);
      fresh.upload(t.values);
      fresh.events.measure([&] { fresh.launch_quant(); });
      auto result = fresh.download(t);
      if (result.data.size() != data_size(n))
        throw std::runtime_error("bad result");
    }
    return elapsed(start);
  };
  auto reused_call = [&] {
    const auto start = Clock::now();
    {
      reused.upload(t.values);
      reused.events.measure([&] { reused.launch_quant(); });
      auto result = reused.download(t);
      if (result.data.size() != data_size(n))
        throw std::runtime_error("bad reused result");
    }
    return elapsed(start);
  };
  for (int i = -3; i < repeats; ++i) {
    double fresh_ms, reused_ms;
    if (i % 2) {
      reused_ms = reused_call();
      fresh_ms = fresh_call();
    } else {
      fresh_ms = fresh_call();
      reused_ms = reused_call();
    }
    if (i >= 0) {
      wall.push_back(fresh_ms);
      reused_wall.push_back(reused_ms);
    }
  }
  stats("quant_optimized", "host_api", n, wall, bytes);
  stats("quant_reused_workspace", "host_api", n, reused_wall, bytes);
  // 完整 packed 比较放在计时外，包含 scale 和 NVFP4 global_scale。
  compare_packed(q, reused.download(t));
  // CPU 对照使用冻结的枚举编码规则，仅在 1M 预跑一次、计时三次。
  if (n == (1u << 20)) {
    auto cpu = reference_quantize(t, o);
    compare_packed(q, cpu);
    std::vector<double> quant, dequant;
    for (int i = 0; i < 3; ++i) {
      auto start = Clock::now();
      cpu = reference_quantize(t, o);
      quant.push_back(elapsed(start));
      start = Clock::now();
      auto y = reference_dequantize(cpu);
      dequant.push_back(elapsed(start));
      if (y.size() != n)
        throw std::runtime_error("CPU output length mismatch");
    }
    stats("quant_reference", "cpu", n, quant, bytes);
    stats("dequant_fp32", "cpu", n, dequant, bytes);
  }
}

}  // namespace pipeline

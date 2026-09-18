// 当前 pipeline 的主机端入口：负责流程调度、文件输出、性能测试和自测。
// 正式量化/反量化 kernel 位于 pipeline.cuh，本文件仅额外定义编码自测 kernel。
// 编码辅助函数已独立到 codec.cuh，不再包含旧版 CUDA main.cu。
#include "pipeline.cuh"

namespace pipeline {

// ===== 1. 计时统计辅助函数 =====

using Clock = std::chrono::steady_clock;

// 主机墙钟耗时（毫秒），与 CUDA event 测得的设备序列时间区分。
double elapsed(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

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

// ===== 2. 反量化、结果文件与误差日志 =====

// 完整 run 流程的后半段：反量化到 T -> CPU 校验 -> 写张量 -> 输出 JSON。
// CPU 一致性检查与量化损失不同：后者比较原始输入和最终低精度舍入后的输出。
template <typename T>
void finish(Workspace& w, const Tensor& t, const Packed& q, const std::string& path, float quant_ms,
            double gpu_wall, bool quant_verified) {
  const float dequant_ms = w.n ? w.events.measure([&] { w.launch_dequant<T>(); }) : 0;
  const auto output = w.download_output<T>();

  // 验证实际 GPU 结果，同时保留“原始输入到目标 dtype”的量化损失统计。
  const auto expected = reference_dequantize(q);
  cuda_output::compare(output, expected);
  if (!fs::path(path).parent_path().empty())
    fs::create_directories(fs::path(path).parent_path());
  cuda_output::write(path, t.rows, t.cols, output);

  double max_abs = 0, mae = 0, mse = 0;
  std::size_t nonfinite = 0;
  for (std::size_t i = 0; i < output.size(); ++i) {
    const double value = static_cast<float>(output[i]);
    if (!std::isfinite(value)) {
      ++nonfinite;
      continue;
    }
    const double diff = std::fabs(value - t.values[i]);
    max_abs = std::max(max_abs, diff);
    mae += diff;
    mse += diff * diff;
  }
  if (w.n) {
    mae /= w.n;
    mse /= w.n;
  }

  // payload 不计文件头，但包含 NVFP4 的全局 scale；file 压缩率计入双方文件头。
  const auto payload = q.data.size() + q.scales.size() + (nvfp4 ? 4 : 0);
  std::cout << "{\"format\":\"" << format_name << "\",\"input_dtype\":\"fp" << t.element_bytes * 8
            << "\",\"output_dtype\":\"" << cuda_output::Format<T>::name << "\",\"scale_mode\":\""
            << (q.options.tensor ? "tensor" : "block") << "\",\"rounding\":\""
            << (q.options.stochastic ? "stochastic" : "nearest") << "\",\"seed\":" << q.options.seed
            << ",\"elements\":" << w.n << ",\"nonfinite_output\":" << nonfinite;
  // FP16 输出可能溢出；此时不把有限子集的误差伪装成完整张量误差。
  if (nonfinite)
    std::cout << ",\"max_abs_error\":null,\"mae\":null,\"mse\":null";
  else
    std::cout << ",\"max_abs_error\":" << max_abs << ",\"mae\":" << mae << ",\"mse\":" << mse;
  std::cout << ",\"compression_payload\":"
            << (payload ? double(w.n * t.element_bytes) / payload : 0) << ",\"compression_file\":"
            << double(28 + w.n * t.element_bytes) / (72 + q.data.size() + q.scales.size())
            << ",\"quant_kernel_ms\":" << quant_ms
            << ",\"quant_setup_upload_compute_download_ms\":" << gpu_wall
            << ",\"dequant_kernel_ms\":" << dequant_ms << ",\"quant_logical_GBps\":"
            << (quant_ms > 0 ? double(4 * w.n + payload) / (quant_ms * 1e6) : 0)
            << ",\"dequant_logical_GBps\":"
            << (dequant_ms > 0 ? double(sizeof(T) * w.n + payload) / (dequant_ms * 1e6) : 0)
            << ",\"cpu_dequant_match\":true,\"cpu_quant_match\":"
            << (quant_verified ? "true" : "null") << "}\n";
}

// --dequant-file 路径：直接上传已有 packed 权重，不重复执行量化。
template <typename T>
void restore(const Packed& q, const std::string& path) {
  Workspace w(checked_count(q.rows, q.cols), q.options);
  w.upload(q);
  const float ms = w.n ? w.events.measure([&] { w.launch_dequant<T>(); }) : 0;
  auto output = w.download_output<T>();
  cuda_output::compare(output, reference_dequantize(q));
  if (!fs::path(path).parent_path().empty())
    fs::create_directories(fs::path(path).parent_path());
  cuda_output::write(path, q.rows, q.cols, output);
  std::cout << "dequant_kernel_ms=" << ms << '\n';
}

// ===== 3. 正式性能测试（不读取 TOML 配置） =====

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
void benchmark(std::size_t n, int repeats) {
  if (!n || n > (1u << 26) || repeats < 1 || repeats > 1000)
    throw std::runtime_error("benchmark: n must be 1..2^26, repeats 1..1000");
  Tensor t{1, n, 4, std::vector<float>(n)};
  std::mt19937 rng(20260909);
  std::normal_distribution<float> normal;
  for (auto& v : t.values)
    v = normal(rng);
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
  // 单次完整量化调用包括申请/释放与传输；与驻留 GPU 指标分开记录。
  std::vector<double> wall;
  for (int i = -3; i < repeats; ++i) {
    const auto start = Clock::now();
    {
      Workspace fresh(n, o);
      fresh.upload(t.values);
      fresh.events.measure([&] { fresh.launch_quant(); });
      auto result = fresh.download(t);
      if (result.data.size() != data_size(n))
        throw std::runtime_error("bad result");
    }
    if (i >= 0)
      wall.push_back(elapsed(start));
  }
  stats("quant_optimized", "host_api", n, wall, bytes);
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

// ===== 4. 编码边界与完整流程自测 =====

// 自测 kernel：每线程独立编码一个值，输出编码字节，不计算 scale 或 packed 数据。
// 用于隔离检查编码器的舍入边界与随机种子；不参与正常量化或 benchmark。
__global__ void encoder_probe(const float* values, std::uint8_t* codes, std::size_t n,
                              bool stochastic, std::uint32_t seed) {
  const auto i = blockIdx.x * 256ull + threadIdx.x;
  if (i < n)
    codes[i] = encode(values[i], nvfp4, stochastic, seed, i);
}

// 构造相邻编码中点及其两侧、正负零、随机值，与 CPU 编码器逐字节对照。
void check_encoder() {
  std::vector<float> values{0, -0.0f, 448, -448};
  const unsigned last = nvfp4 ? 7 : 126;
  auto mag = [](unsigned code) { return nvfp4 ? magnitude2(code) : cpu_decode4(code); };
  for (unsigned code = 0; code < last; ++code) {
    const float mid = (mag(code) + mag(code + 1)) / 2;
    for (float s : {1.0f, -1.0f}) {
      values.push_back(s * std::nextafter(mid, 0.0f));
      values.push_back(s * mid);
      values.push_back(s * std::nextafter(mid, INFINITY));
    }
  }
  std::mt19937 rng(17);
  std::uniform_real_distribution<float> dist(-600, 600);
  for (unsigned i = 0; i < 10000; ++i)
    values.push_back(dist(rng));
  // 随机 FP32 位模式覆盖极小数、大指数和正负号，不只测试普通正态数。
  for (unsigned i = 0; !nvfp4 && i < 100000; ++i) {
    const std::uint32_t bits = rng();
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    if (std::isfinite(value)) values.push_back(value);
  }
  DeviceBuffer<float> input(values.size());
  DeviceBuffer<std::uint8_t> result(values.size());
  CUDA_CHECK(cudaMemcpy(input.ptr, values.data(), values.size() * 4, cudaMemcpyHostToDevice));
  for (bool stochastic : {false, true}) {
    Options o;
    o.stochastic = stochastic;
    encoder_probe<<<static_cast<unsigned>(ceil_div(values.size(), 256)), 256>>>(
        input.ptr, result.ptr, values.size(), stochastic, o.seed);
    CUDA_CHECK(cudaGetLastError());
    std::vector<std::uint8_t> codes(values.size());
    CUDA_CHECK(cudaMemcpy(codes.data(), result.ptr, codes.size(), cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < values.size(); ++i)
      if (codes[i] != reference_encode(values[i], nvfp4, o, i))
        throw std::runtime_error("encoder boundary mismatch at " + std::to_string(i));
  }
  // 中点上下舍入概率应各约 1/2；检查随机模式确实生效且更换 seed 会改变结果。
  const float mid = nvfp4 ? 1.25f : 1.0625f;
  std::fill(values.begin(), values.end(), mid);
  CUDA_CHECK(cudaMemcpy(input.ptr, values.data(), values.size() * 4, cudaMemcpyHostToDevice));
  std::vector<std::uint8_t> first(values.size()), second(values.size());
  for (unsigned seed : {1, 2}) {
    encoder_probe<<<static_cast<unsigned>(ceil_div(values.size(), 256)), 256>>>(
        input.ptr, result.ptr, values.size(), true, seed);
    CUDA_CHECK(cudaGetLastError());
    auto& codes = seed == 1 ? first : second;
    CUDA_CHECK(cudaMemcpy(codes.data(), result.ptr, codes.size(), cudaMemcpyDeviceToHost));
  }
  std::size_t high = 0;
  for (auto code : first)
    if ((nvfp4 ? magnitude2(code & 7) : cpu_decode4(code)) > mid)
      ++high;
  const double fraction = double(high) / first.size();
  if (fraction < .45 || fraction > .55 || first == second)
    throw std::runtime_error("stochastic distribution/seed failure");
  std::cout << "encoder_samples=" << values.size() << " stochastic_high_fraction=" << fraction
            << " status=PASS\n";
}

// 覆盖两种 scale 模式、两种舍入、尾部长度、三种输出类型和重复执行一致性。
// 另外读取冻结输入与 CPU reference 对照，并检查全零及极端有限值。
void self_test(const std::string& cpu_dir) {
  std::size_t checked = 0;
  for (bool tensor : {false, true})
    for (bool stochastic : {false, true}) {
      Options o;
      o.tensor = tensor;
      o.stochastic = stochastic;
      std::vector<std::size_t> sizes;
      for (unsigned i = 0; i <= 65; ++i)
        sizes.push_back(i);
      for (unsigned i : {255, 256, 257, 1023, 1024, 1025})
        sizes.push_back(i);
      for (auto n : sizes) {
        Tensor t{1, n, 4, std::vector<float>(n)};
        for (std::size_t i = 0; i < n; ++i)
          t.values[i] = (static_cast<int>(i % 17) - 8) * .375f;
        if (n)
          t.values.back() = -0.0f;
        Workspace w(n, o);
        w.upload(t.values);
        w.events.measure([&] { w.launch_quant(); });
        const auto q = w.download(t);
        compare_packed(q, reference_quantize(t, o));
        const auto y = reference_dequantize(q);
        w.events.measure([&] { w.launch_dequant<float>(); });
        cuda_output::compare(w.download_output<float>(), y);
        w.events.measure([&] { w.launch_dequant<__half>(); });
        cuda_output::compare(w.download_output<__half>(), y);
        w.events.measure([&] { w.launch_dequant<__nv_bfloat16>(); });
        cuda_output::compare(w.download_output<__nv_bfloat16>(), y);
        w.events.measure([&] { w.launch_quant(); });
        compare_packed(q, w.download(t));
        ++checked;
      }
    }
  for (const auto& name : {"zeros", "basic", "outlier_tail", "tail_block", "random"}) {
    const auto t = read_input(cpu_dir + "/tests/data/" + name + ".fp32");
    const auto frozen = cpu_reference(t.values);
    Options o;
    Workspace w(t.values.size(), o);
    w.upload(t.values);
    w.events.measure([&] { w.launch_quant(); });
    const auto q = w.download(t);
    if (q.data != frozen.data || q.scales != frozen.scales ||
        std::memcmp(&q.global, &frozen.global_scale, 4))
      throw std::runtime_error("legacy reference mismatch");
    ++checked;
  }
  // 极小有限值在全局 scale 下溢时使用最小正 subnormal，不产生 0/0。
  for (float v : {0.0f, -0.0f, 1e-38f, 1e-9f, 1e30f, std::numeric_limits<float>::denorm_min()}) {
    Tensor t{1, 33, 4, std::vector<float>(33, v)};
    Options o;
    Workspace w(33, o);
    w.upload(t.values);
    w.events.measure([&] { w.launch_quant(); });
    compare_packed(w.download(t), reference_quantize(t, o));
    ++checked;
  }
  check_encoder();
  if (!nvfp4) {
    // 相邻 warp 使用跨度很大的 scale，检查融合广播、次正规数与饱和边界。
    Tensor t{1, 0, 4, {}};
    for (int exponent = -149; exponent <= 119; ++exponent) {
      for (unsigned lane = 0; lane < 32; ++lane) {
        const float value = std::ldexp(lane == 0 ? 448.0f : (lane - 16.0f), exponent);
        t.values.push_back(value);
      }
    }
    t.values.push_back(-0.0f);
    t.cols = t.values.size();
    Options o;
    Workspace w(t.values.size(), o);
    w.upload(t.values);
    w.events.measure([&] { w.launch_quant(); });
    const auto fused = w.download(t);
    compare_packed(fused, reference_quantize(t, o));
    w.events.measure([&] { w.launch_quant(true); });
    compare_packed(fused, w.download(t));
    std::cout << "mxfp8_fused_scale_cases=270 status=PASS\n";
  }
  std::cout << "pipeline_cases=" << checked << " status=PASS\n";
}
}  // namespace pipeline

// ===== 5. 程序入口与命令分发 =====

// 支持 --benchmark、--self-test、--dequant-file，以及普通量化+反量化流程。
// tools/quantize.py 会把 TOML 配置转换为普通流程所需的位置参数。
int main(int argc, char** argv) {
  using namespace pipeline;
  try {
    std::cout << std::setprecision(10);
    if (argc == 4 && std::string(argv[1]) == "--benchmark") {
      benchmark(std::stoull(argv[2]), std::stoi(argv[3]));
      return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--self-test") {
      self_test(argv[2]);
      return 0;
    }
    if (argc == 5 && std::string(argv[1]) == "--dequant-file") {
      const auto q = read_packed(argv[2]);
      const std::string dtype = argv[4];
      if (dtype == "fp32")
        restore<float>(q, argv[3]);
      else if (dtype == "fp16")
        restore<__half>(q, argv[3]);
      else if (dtype == "bf16")
        restore<__nv_bfloat16>(q, argv[3]);
      else
        throw std::runtime_error("invalid output dtype");
      return 0;
    }

    // 普通流程第一步：检查参数、固定 block 大小，并防止输入输出路径互相覆盖。
    if (argc != 10)
      throw std::runtime_error(
          "usage: pipeline input quantized output block|tensor nearest|stochastic fp32|fp16|bf16 block_size seed verify|noverify");
    const std::string mode = argv[4], rounding = argv[5], dtype = argv[6], verification = argv[9];
    if ((mode != "block" && mode != "tensor") ||
        (rounding != "nearest" && rounding != "stochastic") ||
        (dtype != "fp32" && dtype != "fp16" && dtype != "bf16") ||
        (verification != "verify" && verification != "noverify"))
      throw std::runtime_error("invalid pipeline option");
    const auto seed = std::stoull(argv[8]);
    if (seed > UINT32_MAX)
      throw std::runtime_error("seed exceeds uint32");
    Options o{mode == "tensor", rounding == "stochastic", std::stoull(argv[7]),
              static_cast<std::uint32_t>(seed)};
    validate(o);
    if (fs::absolute(argv[1]).lexically_normal() == fs::absolute(argv[2]).lexically_normal() ||
        fs::absolute(argv[1]).lexically_normal() == fs::absolute(argv[3]).lexically_normal() ||
        fs::absolute(argv[2]).lexically_normal() == fs::absolute(argv[3]).lexically_normal())
      throw std::runtime_error("input/output paths must differ");

    // 第二步：读取输入，再计时显存分配、上传、量化和下载。
    // wall 不包含读文件、CPU 校验、写文件或 Workspace 析构释放。
    const auto t = read_input(argv[1]);
    const auto start = Clock::now();
    Workspace w(t.values.size(), o);
    w.upload(t.values);
    const float quant_ms = w.n ? w.events.measure([&] { w.launch_quant(); }) : 0;
    auto q = w.download(t);
    const double wall = elapsed(start);

    // 第三步：按需校验 CPU 量化结果，并写入/回读 .lpq 检查文件往返一致性。
    if (verification == "verify")
      compare_packed(q, reference_quantize(t, o));
    write_packed(argv[2], q);
    compare_packed(q, read_packed(argv[2]));

    // 第四步：按输出 dtype 实例化反量化流程，保存张量并报告误差与性能。
    if (dtype == "fp16")
      finish<__half>(w, t, q, argv[3], quant_ms, wall, verification == "verify");
    else if (dtype == "bf16")
      finish<__nv_bfloat16>(w, t, q, argv[3], quant_ms, wall, verification == "verify");
    else
      finish<float>(w, t, q, argv[3], quant_ms, wall, verification == "verify");
  } catch (const std::exception& e) {
    std::cerr << "pipeline=FAIL " << e.what() << '\n';
    return 1;
  }
}

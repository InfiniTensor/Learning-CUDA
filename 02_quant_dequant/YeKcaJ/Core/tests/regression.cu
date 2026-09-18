// 正确性自测与编码探针；只在 BUILD_TESTING=ON 时参与构建。
#include "app/commands.h"
#include "runtime/workspace.cuh"
#include "reference/oracle.h"
#include "reference/output_compare.cuh"
#include "io/tensor_io.h"
#include "kernels/codec.cuh"
#include <random>

namespace pipeline {

// 自测时临时替换输出显存，前后各填哨兵；析构恢复 Workspace 原始指针。
// 这能检测越界写，但不能替代平台工具对越界读和数据竞争的检查。
template <typename T>
struct GuardedOutput {
  DeviceBuffer<T>& target;
  DeviceBuffer<T> storage;
  std::vector<std::uint8_t> bytes;
  GuardedOutput(DeviceBuffer<T>& buffer, std::size_t count)
      : target(buffer), storage(count + 64), bytes((count + 64) * sizeof(T), 0xa5) {
    CUDA_CHECK(cudaMemcpy(storage.ptr, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
    std::swap(target.ptr, storage.ptr);
    target.ptr += 32;  // 偏移保持向量读写所需的对齐；storage.ptr 保存原始显存。
  }
  ~GuardedOutput() { target.ptr -= 32; std::swap(target.ptr, storage.ptr); }
  void check(std::size_t written_bytes) {
    CUDA_CHECK(cudaMemcpy(bytes.data(), target.ptr - 32, bytes.size(), cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < bytes.size(); ++i)
      if ((i < 32 * sizeof(T) || i >= 32 * sizeof(T) + written_bytes) && bytes[i] != 0xa5)
        throw std::runtime_error("output front/back guard overwritten");
  }
};

template <typename T>
void check_guarded_dequant(Workspace& w, const std::vector<float>& expected) {
  GuardedOutput<float> guard(w.output, w.n);
  w.events.measure([&] { w.launch_dequant<T>(); });
  cuda_output::compare(w.download_output<T>(), expected);
  guard.check(w.n * sizeof(T));
}

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
        GuardedOutput<std::uint8_t> data_guard(w.data, data_size(n));
        GuardedOutput<std::uint8_t> scale_guard(w.scales, w.groups);
        w.upload(t.values);
        w.events.measure([&] { w.launch_quant(); });
        const auto q = w.download(t);
        compare_packed(q, reference_quantize(t, o));
        const auto y = reference_dequantize(q);
        check_guarded_dequant<float>(w, y);
        check_guarded_dequant<__half>(w, y);
        check_guarded_dequant<__nv_bfloat16>(w, y);
        w.events.measure([&] { w.launch_quant(); });
        compare_packed(q, w.download(t));
        data_guard.check(data_size(n));
        scale_guard.check(w.groups);
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
    try {
      compare_packed(w.download(t), reference_quantize(t, o));
    } catch (const std::exception&) {
      std::cerr << "extreme input=" << v << '\n';
      throw;
    }
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
  std::cout << "guarded_cases=288 guards=front,back outputs=data,scales,fp32,fp16,bf16 status=PASS\n";
}

}  // namespace pipeline

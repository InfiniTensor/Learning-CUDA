// 包含实际 CUDA 算子，临时改名其 main，让本文件提供回归测试入口。
// TEST_NVFP4 决定测试哪种格式；CPU reference 在独立编译单元中链接进来。
#define main cuda_application_main
#ifdef TEST_NVFP4
#include "../../CUDANVFP4/main.cu"
#else
#include "../../CUDAMXFP8/main.cu"
#endif
#undef main
#include "reference.h"
#include <filesystem>
#include <limits>
#include <sstream>

namespace fs = std::filesystem;
std::size_t cases_checked = 0;

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error(what);
}

// 用 double 的二进制缩放和 ties-to-even 建立独立的 16 位转换 oracle。
// 不调用 CUDA half/bfloat16 转换，以检查设备转换和主机校验逻辑。
std::uint16_t expected16(float value, bool bf16) {
  const std::uint16_t sign = std::signbit(value) ? 0x8000 : 0;
  const int fraction = bf16 ? 7 : 10;
  const int bias = bf16 ? 127 : 15;
  const int min_exp = 1 - bias;
  const int max_exp = bias;
  const std::uint16_t infinity = bf16 ? 0x7f80 : 0x7c00;
  if (std::isnan(value)) return infinity | (1 << (fraction - 1));
  if (std::isinf(value)) return sign | infinity;
  if (value == 0) return sign;
  const double magnitude = std::fabs(static_cast<double>(value));
  int exponent = 0;
  std::frexp(magnitude, &exponent);
  --exponent;
  if (exponent < min_exp)
    return sign | static_cast<std::uint16_t>(std::nearbyint(std::ldexp(magnitude, fraction - min_exp)));
  int significand = static_cast<int>(std::nearbyint(std::ldexp(magnitude, fraction - exponent)));
  if (significand == (1 << (fraction + 1))) { significand >>= 1; ++exponent; }
  if (exponent > max_exp) return sign | infinity;
  return sign | ((exponent + bias) << fraction) | (significand - (1 << fraction));
}

// 回归测试比 CLI 更严格：FP32 也逐位比较；16 位使用独立舍入 oracle。
template <typename T>
void check_values(const std::vector<T>& actual, const std::vector<float>& expected,
                  const std::string& name) {
  require(actual.size() == expected.size(), name + ": size mismatch");
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (std::isnan(expected[i])) {
      require(std::isnan(static_cast<float>(actual[i])), name + ": NaN mismatch");
    } else if constexpr (std::is_same_v<T, float>) {
      require(std::memcmp(&actual[i], &expected[i], sizeof(float)) == 0,
              name + ": fp32 bit mismatch at " + std::to_string(i));
    } else {
      std::uint16_t bits;
      std::memcpy(&bits, &actual[i], 2);
      const auto wanted = expected16(expected[i], std::is_same_v<T, __nv_bfloat16>);
      require(bits == wanted, name + ": 16-bit mismatch at " + std::to_string(i));
    }
  }
}

template <typename T>
void check_dequant(const QuantizedFile& q, const std::vector<float>& expected,
                   const std::string& name) {
  float ms;
  const auto actual = dequantize_cuda<T>(q, ms);
  check_values(actual, expected, name + "/" + cuda_output::Format<T>::name);
}

void check_all_outputs(const QuantizedFile& q, const std::vector<float>& expected,
                      const std::string& name) {
  check_dequant<float>(q, expected, name);
  check_dequant<__half>(q, expected, name);
  check_dequant<__nv_bfloat16>(q, expected, name);
}

// 新增输入实时调用冻结 CPU 算法，不写入 golden；先比量化字节，再比三种输出。
void check_case(const std::string& name, const std::vector<float>& values) {
  const auto expected = cpu_reference(values);
  float ms;
  const auto actual = quantize_cuda(FloatInput{1, values.size(), values}, ms);
#ifdef TEST_NVFP4
  require(std::memcmp(&actual.global_scale, &expected.global_scale, 4) == 0, name + ": global scale");
  require(actual.packed == expected.data, name + ": packed mismatch");
  require(actual.block_scales == expected.scales, name + ": block scale mismatch");
  if (values.size() % 2) require((actual.packed.back() & 0xf0) == 0, name + ": high nibble padding");
#else
  require(actual.data == expected.data, name + ": data mismatch");
  require(actual.scales == expected.scales, name + ": scale mismatch");
#endif
  check_all_outputs(actual, expected.dequant, name);
  ++cases_checked;
}

// 测试探针只调用生产编码函数，单独检查舍入，排除 scale 计算的影响。
__global__ void encoding_probe(const float* input, std::uint8_t* output4,
                                std::uint8_t* output2, std::size_t n) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  output4[i] = encode_e4m3(input[i]);
#ifdef TEST_NVFP4
  output2[i] = encode_e2m1(input[i]);
#endif
}

void check_rounding() {
  std::vector<float> samples{0.0f, -0.0f};
  // 每个相邻可表示值的中点，取中点及两侧相邻 FP32，并覆盖正负两侧。
  const auto append = [&](float mid) {
    for (float sign : {1.0f, -1.0f}) {
      samples.push_back(sign * std::nextafter(mid, 0.0f));
      samples.push_back(sign * mid);
      samples.push_back(sign * std::nextafter(mid, INFINITY));
    }
  };
  for (int code = 0; code < 126; ++code)
    append((cpu_decode4(code) + cpu_decode4(code + 1)) / 2.0f);
  const float magnitudes[] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
  for (int i = 0; i < 7; ++i) append((magnitudes[i] + magnitudes[i + 1]) / 2.0f);
  float* input;
  std::uint8_t *out4, *out2;
  CUDA_CHECK(cudaMalloc(&input, samples.size() * 4));
  CUDA_CHECK(cudaMalloc(&out4, samples.size()));
  CUDA_CHECK(cudaMalloc(&out2, samples.size()));
  CUDA_CHECK(cudaMemcpy(input, samples.data(), samples.size() * 4, cudaMemcpyHostToDevice));
  encoding_probe<<<(samples.size() + 255) / 256, 256>>>(input, out4, out2, samples.size());
  CUDA_CHECK(cudaGetLastError());
  std::vector<std::uint8_t> actual(samples.size());
  CUDA_CHECK(cudaMemcpy(actual.data(), out4, actual.size(), cudaMemcpyDeviceToHost));
  for (std::size_t i = 0; i < samples.size(); ++i)
    require(actual[i] == cpu_encode4(samples[i]), "E4M3 rounding mismatch");
#ifdef TEST_NVFP4
  CUDA_CHECK(cudaMemcpy(actual.data(), out2, actual.size(), cudaMemcpyDeviceToHost));
  for (std::size_t i = 0; i < samples.size(); ++i)
    require(actual[i] == cpu_encode2(samples[i]), "E2M1 rounding mismatch");
#endif
  CUDA_CHECK(cudaFree(input));
  CUDA_CHECK(cudaFree(out4));
  CUDA_CHECK(cudaFree(out2));
  std::cout << "rounding_samples=" << samples.size() << " status=PASS\n";
}

template <typename T>
__global__ void conversion_probe(const float* input, T* output, std::size_t n) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) output[i] = cuda_output::Format<T>::convert(input[i]);
}

template <typename T>
void check_conversion() {
  // 包含偶数/奇数尾数中点、subnormal、下溢、最大有限值、溢出和非有限值。
  std::vector<float> values{0, -0.0f, 1.00048828125f, 1.00146484375f,
      1.00390625f, 1.01171875f, 65504, 65520,
      std::ldexp(1.0f, -25), std::ldexp(1.0f, -24), std::ldexp(1.0f, -14),
      std::ldexp(1.0f, -134), std::ldexp(1.0f, -133), std::ldexp(1.0f, -126),
      std::numeric_limits<float>::max(), std::numeric_limits<float>::denorm_min(),
      INFINITY, -INFINITY, NAN};
  const auto n = values.size();
  for (std::size_t i = 2; i < n; ++i) {
    values.push_back(-values[i]);
    values.push_back(std::nextafter(values[i], 0.0f));
    values.push_back(std::nextafter(values[i], INFINITY));
  }
  float* input;
  T* output;
  CUDA_CHECK(cudaMalloc(&input, values.size() * 4));
  CUDA_CHECK(cudaMalloc(&output, values.size() * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(input, values.data(), values.size() * 4, cudaMemcpyHostToDevice));
  conversion_probe<<<1, 256>>>(input, output, values.size());
  CUDA_CHECK(cudaGetLastError());
  std::vector<T> actual(values.size());
  CUDA_CHECK(cudaMemcpy(actual.data(), output, actual.size() * sizeof(T), cudaMemcpyDeviceToHost));
  check_values(actual, values, "conversion");
  cuda_output::compare(actual, values);
  const fs::path path = fs::path("tests/results") / (std::string("conversion.") + cuda_output::Format<T>::name);
  fs::create_directories(path.parent_path());
  cuda_output::write(path.string(), 1, actual.size(), actual);
  require(fs::file_size(path) == 36 + actual.size() * sizeof(T), "output file byte length");
  std::ifstream file(path, std::ios::binary);
  char magic[8];
  file.read(magic, 8);
  require(std::string(magic, 8) == cuda_output::Format<T>::magic, "output file magic");
  std::uint32_t version;
  std::uint64_t rows, cols, count;
  file.read(reinterpret_cast<char*>(&version), 4);
  file.read(reinterpret_cast<char*>(&rows), 8);
  file.read(reinterpret_cast<char*>(&cols), 8);
  file.read(reinterpret_cast<char*>(&count), 8);
  require(version == 1 && rows == 1 && cols == actual.size() && count == actual.size(),
          "output file metadata");
  std::vector<T> loaded(actual.size());
  file.read(reinterpret_cast<char*>(loaded.data()), loaded.size() * sizeof(T));
  require(bool(file), "output file truncated");
  require(std::memcmp(loaded.data(), actual.data(), actual.size() * sizeof(T)) == 0, "output file payload");
  // 主动篡改一个输出，确认比较器会拒绝错误结果，而非无条件通过。
  auto bad = actual;
  bad[0] = cuda_output::Format<T>::convert(1.0f);
  bool rejected = false;
  std::ostringstream ignored;
  auto* previous = std::cout.rdbuf(ignored.rdbuf());
  try { cuda_output::compare(bad, values); } catch (const std::runtime_error&) { rejected = true; }
  std::cout.rdbuf(previous);
  require(rejected, "incorrect output was accepted");
  CUDA_CHECK(cudaFree(input));
  CUDA_CHECK(cudaFree(output));
}

// 直接构造编码与 scale，覆盖量化输入未必能产生的反量化边界组合。
void check_encoded_inputs() {
#ifdef TEST_NVFP4
  const float globals[] = {1, 1.00390625f, 1.00048828125f, 1e-8f, 1e5f, 1e-37f};
  for (float global : globals) {
    QuantizedFile q{1, 16 * 127, global, {}, {}};
    std::vector<float> expected;
    const float magnitudes[] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
    for (int scale = 0; scale <= 126; ++scale) {
      q.block_scales.push_back(scale);
      const float effective = global * cpu_decode4(scale);
      for (int code = 0; code < 16; ++code) {
        if (code % 2 == 0) q.packed.push_back(code | ((code + 1) << 4));
        expected.push_back((code & 8 ? -magnitudes[code & 7] : magnitudes[code]) * effective);
      }
    }
    check_all_outputs(q, expected, "encoded NVFP4");
  }
#else
  for (int scale : {0, 1, 100, 110, 120, 127, 135, 145, 200, 254}) {
    QuantizedFile q{1, 256, std::vector<std::uint8_t>(256), std::vector<std::uint8_t>(8, scale)};
    std::vector<float> expected;
    for (int code = 0; code < 256; ++code) {
      q.data[code] = code;
      expected.push_back(cpu_decode4(code) * std::ldexp(1.0f, scale - 127));
    }
    check_all_outputs(q, expected, "encoded MXFP8");
  }
#endif
}

int main(int argc, char** argv) {
  try {
    require(argc == 2, "usage: regression <CPU directory>");
    const fs::path cpu = argv[1];
    for (const auto& name : {"zeros", "basic", "outlier_tail", "tail_block", "random"}) {
      const auto in = read_fp32_input((cpu / "tests/data" / (std::string(name) + ".fp32")).string());
      float ms;
      const auto q = quantize_cuda(in, ms);
#ifdef TEST_NVFP4
      const auto expected = read_quantized((cpu / "tests/golden" / (std::string(name) + ".nvfp4")).string());
#else
      const auto expected = read_quantized((cpu / "tests/golden" / (std::string(name) + ".mxfp8")).string());
#endif
      compare_quantized(q, expected);
      std::size_t rows, cols;
      const auto golden = read_dequant((cpu / "tests/golden" / (std::string(name) + ".dequant.fp32")).string(), rows, cols);
      check_all_outputs(q, golden, name);
    }
    // 穷举小长度，覆盖空输入、奇数 packed 尾部和不足一个量化 block。
    for (std::size_t n = 0; n <= 65; ++n) {
      std::vector<float> values(n);
      for (std::size_t i = 0; i < n; ++i) values[i] = (static_cast<int>(i % 17) - 8) * 0.375f;
      check_case("tail_" + std::to_string(n), values);
      check_case("zero_" + std::to_string(n), std::vector<float>(n, -0.0f));
    }
    // 检查反量化线程块边界附近的长度，防止尾部线程越界。
    for (std::size_t n : {255, 256, 257, 1023, 1024, 1025}) {
      std::vector<float> values(n, -0.0f);
      values.front() = 6;
      values.back() = -3;
      check_case("launch_tail_" + std::to_string(n), values);
    }
    std::vector<float> signed_zeros(33, -0.0f);
    signed_zeros[1] = 6;
    check_case("signed_zeros_positive_and_zero_scale", signed_zeros);
#ifdef TEST_NVFP4
    std::vector<float> tiny(33, -1e-9f);
    tiny[0] = 6;
    check_case("rounded_zero_block_scale", tiny);
    std::vector<float> midpoint_values;
    for (float mid : {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f}) {
      std::vector<float> block(16, 0);
      block[0] = 6;
      block[1] = std::nextafter(mid, 0.0f);
      block[2] = mid;
      block[3] = std::nextafter(mid, INFINITY);
      for (int i = 1; i <= 3; ++i) block[i + 3] = -block[i];
      midpoint_values.insert(midpoint_values.end(), block.begin(), block.end());
    }
    check_case("E2M1_midpoints_end_to_end", midpoint_values);
#endif
    check_rounding();
    check_conversion<__half>();
    check_conversion<__nv_bfloat16>();
    check_encoded_inputs();
    std::cout << "golden_cases=5 generated_cases=" << cases_checked
              << " output_types=fp32,fp16,bf16 regression=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "regression=FAIL " << e.what() << '\n';
    return 1;
  }
}

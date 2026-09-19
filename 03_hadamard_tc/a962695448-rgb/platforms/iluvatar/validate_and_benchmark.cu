#include "hadamard_api.h"
#include "reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace api = hadamard::iluvatar;
#define CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) \
    throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(e_)); } while (0)

template<class T> float read(T);
template<> float read(__half x) { return __half2float(x); }
template<> float read(__nv_bfloat16 x) { return __bfloat162float(x); }
template<class T> T rounded(float);
template<> __half rounded(float x) { return __float2half_rn(x); }
template<> __nv_bfloat16 rounded(float x) { return __float2bfloat16_rn(x); }

std::vector<api::Method> active_methods() {
    std::vector<api::Method> methods{api::Method::Baseline, api::Method::Optimized};
#if defined(HADAMARD_ILUVATAR_WARP64) && defined(__ILUVATAR__)
    methods.push_back(api::Method::Warp64);
#endif
    return methods;
}

const char* method_name(api::Method method) {
    if (method == api::Method::Baseline) return "baseline";
    if (method == api::Method::Optimized) return "optimized";
    return "warp64";
}

// 每个区间都有前后哨兵。17 个元素的偏移同时覆盖仅 2 字节对齐的 FP16/BF16 指针。
template<class T> class Guarded {
    static constexpr std::size_t guard = 17;
    T* raw_ = nullptr;
    std::size_t count_;
    std::vector<unsigned char> initial_;
    std::string name_;
public:
    explicit Guarded(std::size_t count, cudaStream_t stream, const char* name = "unnamed")
        : count_(count), initial_((count + 2 * guard) * sizeof(T), 0xa5), name_(name) {
        CHECK(cudaMalloc(reinterpret_cast<void**>(&raw_), initial_.size()));
        // 初始化、上传、kernel、回读使用同一个非阻塞 stream。默认 stream 上的
        // pageable H2D 即使通过 cudaMemcpy 调用，也不保证返回时最终 DMA 已结束。
        // 按字节 memset 避免异步初始化读取随后被 upload 更新的主机 shadow。
        CHECK(cudaMemsetAsync(raw_, 0xa5, initial_.size(), stream));
    }
    ~Guarded() { if (raw_) cudaFree(raw_); }
    Guarded(const Guarded&) = delete;
    Guarded& operator=(const Guarded&) = delete;
    T* data() { return raw_ + guard; }
    void upload(const std::vector<T>& values, cudaStream_t stream) {
        if (values.size() != count_) throw std::runtime_error("upload size mismatch");
        std::memcpy(initial_.data() + guard * sizeof(T), values.data(), count_ * sizeof(T));
        CHECK(cudaMemcpyAsync(raw_, initial_.data(), initial_.size(), cudaMemcpyHostToDevice, stream));
    }
    std::vector<T> download(cudaStream_t stream, bool unchanged = false, const char* phase = "readback") {
        std::vector<unsigned char> bytes(initial_.size());
        CHECK(cudaMemcpyAsync(bytes.data(), raw_, bytes.size(), cudaMemcpyDeviceToHost, stream));
        CHECK(cudaStreamSynchronize(stream));
        const std::size_t prefix = guard * sizeof(T), end = prefix + count_ * sizeof(T);
        for (std::size_t i = 0; i < bytes.size(); ++i) {
            if (bytes[i] == initial_[i] || (!unchanged && i >= prefix && i < end)) continue;
            const bool is_guard = i < prefix || i >= end;
            throw std::runtime_error(std::string(is_guard ? "device buffer guard overwritten" : "read-only input modified")
                + " buffer=" + name_ + " phase=" + phase + " region=" + (i < prefix ? "prefix" : (i >= end ? "suffix" : "payload"))
                + " byte_from_payload=" + std::to_string(static_cast<long long>(i) - static_cast<long long>(prefix))
                + " expected=" + std::to_string(static_cast<unsigned>(initial_[i]))
                + " actual=" + std::to_string(static_cast<unsigned>(bytes[i]))
                + " elements=" + std::to_string(count_) + " element_bytes=" + std::to_string(sizeof(T)));
        }
        std::vector<T> result(count_);
        std::memcpy(result.data(), bytes.data() + prefix, count_ * sizeof(T));
        return result;
    }
};

struct Options {
    bool validate = false, benchmark = false, custom_shape = false, quick = false;
    std::size_t batch = 1, seq = 1, heads = 1;
    int dim = 128, repeats = 100, groups = 5;
    std::string dtype = "both", csv = "iluvatar_benchmark.csv", json = "iluvatar_validation.json";
};

std::size_t positive(const std::string& text, const char* name) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    std::size_t used = 0;
    const auto value = std::stoull(text, &used);
    if (!value || value > std::numeric_limits<std::size_t>::max() || used != text.size())
        throw std::invalid_argument(std::string(name) + " is outside its supported range");
    return static_cast<std::size_t>(value);
}

std::size_t multiply(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::invalid_argument("shape product overflows size_t");
    return a * b;
}

std::size_t checked_shape(std::size_t b, std::size_t s, std::size_t h, int n) {
    if (n < 1 || n > 256 || !hadamard::power_of_two(n))
        throw std::invalid_argument("dim must be a power of two in [1,256]");
    const auto rows = multiply(multiply(b, s), h);
    if (!rows || rows > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("rows outside supported grid range");
    multiply(multiply(rows, static_cast<std::size_t>(n)), sizeof(__half));
    return rows;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--validate") o.validate = true;
        else if (key == "--benchmark") o.benchmark = true;
        else if (key == "--quick") o.quick = true;
        else {
            if (i + 1 == argc) throw std::invalid_argument("missing value for " + key);
            const std::string value = argv[++i];
            if (key == "--dtype") o.dtype = value;
            else if (key == "--csv") o.csv = value;
            else if (key == "--json") o.json = value;
            else if (key == "--batch" || key == "--seq" || key == "--heads" || key == "--dim") {
                const auto v = positive(value, key.c_str());
                o.custom_shape = true;
                if (key == "--batch") o.batch = v;
                else if (key == "--seq") o.seq = v;
                else if (key == "--heads") o.heads = v;
                else {
                    if (v > 256) throw std::invalid_argument("dim is greater than 256");
                    o.dim = static_cast<int>(v);
                }
            } else if (key == "--repeats" || key == "--groups") {
                const auto v = positive(value, key.c_str());
                if (v > 10000) throw std::invalid_argument("repeats/groups exceed 10000");
                if (key == "--repeats") o.repeats = static_cast<int>(v);
                else o.groups = static_cast<int>(v);
            } else throw std::invalid_argument("unknown argument " + key);
        }
    }
    if (!o.validate && !o.benchmark) throw std::invalid_argument("specify --validate and/or --benchmark");
    if (o.dtype != "both" && o.dtype != "fp16" && o.dtype != "bf16")
        throw std::invalid_argument("dtype must be fp16, bf16 or both");
    checked_shape(o.batch, o.seq, o.heads, o.dim);
    return o;
}

template<class T> std::vector<T> make_input(std::size_t rows, int n, int pattern, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    std::normal_distribution<float> normal(0.0f, 0.5f);
    std::vector<T> result(rows * n);
    for (std::size_t i = 0; i < result.size(); ++i) {
        float x = 0;
        if (pattern == 0) x = uniform(rng);
        else if (pattern == 1) x = normal(rng);
        else if (pattern == 2) x = i % n == (i / n + seed) % n ? 8.0f : uniform(rng);
        else if (pattern == 4) x = i % n == (i / n) % n ? 1.0f : 0.0f;
        result[i] = rounded<T>(x);
    }
    return result;
}

struct Summary {
    std::size_t cases = 0, elements = 0, exact_transform_elements = 0, contract_checks = 0;
    std::size_t warp64_cases = 0, exact_warp64_elements = 0;
    std::size_t warp64_grid_stride_cases = 0, warp64_grid_stride_elements = 0, unsupported_warp64_checks = 0;
    double max_rounded_error = 0, max_unrounded_error = 0;
};

template<class T> void one_case(cudaStream_t stream, std::size_t rows, int n, float scale,
                               int pattern, unsigned seed, const char* dtype, Summary& summary) {
    const auto input = make_input<T>(rows, n, pattern, seed);
    const std::size_t size = input.size(), bytes = rows * ((n + 1) / 2);
    Guarded<T> x(size, stream), baseline(size, stream), optimized(size, stream), inplace(size, stream);
    Guarded<std::uint8_t> base_split(bytes, stream), opt_split(bytes, stream), base_fused(bytes, stream), opt_fused(bytes, stream);
    Guarded<float> bs(rows, stream), os(rows, stream), bfs(rows, stream), ofs(rows, stream);
    x.upload(input, stream);
    inplace.upload(input, stream);
    CHECK(api::transform(x.data(), baseline.data(), rows, n, scale, stream, api::Method::Baseline));
    CHECK(api::transform(x.data(), optimized.data(), rows, n, scale, stream, api::Method::Optimized));
    CHECK(api::transform(inplace.data(), inplace.data(), rows, n, scale, stream, api::Method::Optimized));
    CHECK(api::quantize_int4(baseline.data(), base_split.data(), bs.data(), rows, n, stream, api::Method::Baseline));
    CHECK(api::quantize_int4(optimized.data(), opt_split.data(), os.data(), rows, n, stream, api::Method::Optimized));
    CHECK(api::transform_int4(x.data(), base_fused.data(), bfs.data(), rows, n, scale, stream, api::Method::Baseline));
    CHECK(api::transform_int4(x.data(), opt_fused.data(), ofs.data(), rows, n, scale, stream, api::Method::Optimized));
    CHECK(cudaStreamSynchronize(stream));
    x.download(stream, true);
    const auto a = baseline.download(stream), b = optimized.download(stream), ip = inplace.download(stream);
    const std::string context = std::string(dtype) + " rows=" + std::to_string(rows) + " n=" + std::to_string(n)
        + " pattern=" + std::to_string(pattern) + " seed=" + std::to_string(seed) + " scale=" + std::to_string(scale);
    if (std::memcmp(a.data(), b.data(), size * sizeof(T)) || std::memcmp(b.data(), ip.data(), size * sizeof(T)))
        throw std::runtime_error("baseline/optimized/in-place transform not bitwise identical: " + context);
    std::vector<float> fx(size), actual(size);
    for (std::size_t i = 0; i < size; ++i) { fx[i] = read(input[i]); actual[i] = read(b[i]); }
    // 全部元素使用 O(N^2) FP64 稠密矩阵公式，不调用任何 FWHT 作为正确性 oracle。
    const auto dense = hadamard::dense_reference(fx, n, static_cast<double>(scale));
    const double tolerance = std::string(dtype) == "fp16" ? 1e-2 : 5e-2;
    for (std::size_t i = 0; i < size; ++i) {
        const double expected = read(rounded<T>(static_cast<float>(dense[i])));
        const double error = std::abs(static_cast<double>(actual[i]) - expected);
        if (!std::isfinite(actual[i]) || !(error < tolerance))
            throw std::runtime_error("rounded FP64 dense oracle mismatch: " + context + " index=" + std::to_string(i)
                                     + " error=" + std::to_string(error));
        summary.max_rounded_error = std::max(summary.max_rounded_error, error);
        summary.max_unrounded_error = std::max(summary.max_unrounded_error, std::abs(actual[i] - dense[i]));
    }
    const auto expected_q = hadamard::quantize_int4(actual, n);
    if (base_split.download(stream) != expected_q.packed || opt_split.download(stream) != expected_q.packed ||
        base_fused.download(stream) != expected_q.packed || opt_fused.download(stream) != expected_q.packed ||
        bs.download(stream) != expected_q.scales || os.download(stream) != expected_q.scales ||
        bfs.download(stream) != expected_q.scales || ofs.download(stream) != expected_q.scales)
        throw std::runtime_error("CPU/baseline/optimized split/fused INT4 bytes or scales mismatch: " + context);
#if defined(HADAMARD_ILUVATAR_WARP64) && defined(__ILUVATAR__)
    Guarded<T> warp_output(size, stream, "warp64.output"), warp_inplace(size, stream, "warp64.inplace");
    Guarded<std::uint8_t> warp_split(bytes, stream, "warp64.split"), warp_fused(bytes, stream, "warp64.fused");
    Guarded<float> warp_split_scales(rows, stream, "warp64.split-scales"), warp_fused_scales(rows, stream, "warp64.fused-scales");
    warp_inplace.upload(input, stream);
    CHECK(api::transform(x.data(), warp_output.data(), rows, n, scale, stream, api::Method::Warp64));
    CHECK(api::transform(warp_inplace.data(), warp_inplace.data(), rows, n, scale, stream, api::Method::Warp64));
    CHECK(api::quantize_int4(warp_output.data(), warp_split.data(), warp_split_scales.data(), rows, n, stream, api::Method::Warp64));
    CHECK(api::transform_int4(x.data(), warp_fused.data(), warp_fused_scales.data(), rows, n, scale, stream, api::Method::Warp64));
    const auto warp = warp_output.download(stream), warp_ip = warp_inplace.download(stream);
    if (std::memcmp(a.data(), warp.data(), size * sizeof(T)) || std::memcmp(a.data(), warp_ip.data(), size * sizeof(T)))
        throw std::runtime_error("baseline/Warp64/in-place transform not bitwise identical: " + context);
    if (warp_split.download(stream) != expected_q.packed || warp_fused.download(stream) != expected_q.packed ||
        warp_split_scales.download(stream) != expected_q.scales || warp_fused_scales.download(stream) != expected_q.scales)
        throw std::runtime_error("CPU/baseline/Warp64 split/fused INT4 bytes or scales mismatch: " + context);
    x.download(stream, true, "after-warp64");
    ++summary.warp64_cases;
    summary.exact_warp64_elements += size;
#endif
    ++summary.cases;
    summary.elements += size;
    summary.exact_transform_elements += size;
}

template<class T> void contract_tests(cudaStream_t stream, Summary& summary) {
    Guarded<T> input(64, stream, "contract.input"), output(64, stream, "contract.output");
    Guarded<std::uint8_t> packed(32, stream, "contract.packed");
    Guarded<float> scales(8, stream, "contract.scales");
    input.download(stream, true, "initialized");
    output.download(stream, true, "initialized");
    packed.download(stream, true, "initialized");
    scales.download(stream, true, "initialized");
    auto reject = [&](cudaError_t status) {
        if (status != cudaErrorInvalidValue) throw std::runtime_error("invalid API input did not return cudaErrorInvalidValue");
        ++summary.contract_checks;
    };
    auto success = [&](cudaError_t status) { CHECK(status); ++summary.contract_checks; };
    for (const auto method : active_methods()) {
        std::cout << "CONTRACT_PROGRESS method=" << method_name(method) << " phase=invalid-parameters" << std::endl;
        for (int n : {0, 3, 512}) {
            reject(api::transform(input.data(), output.data(), 1, n, 1, stream, method));
            reject(api::quantize_int4(input.data(), packed.data(), scales.data(), 1, n, stream, method));
            reject(api::transform_int4(input.data(), packed.data(), scales.data(), 1, n, 1, stream, method));
        }
        for (float scale : {0.0f, -1.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
            reject(api::transform(input.data(), output.data(), 1, 8, scale, stream, method));
            reject(api::transform_int4(input.data(), packed.data(), scales.data(), 1, 8, scale, stream, method));
        }
        reject(api::transform(static_cast<const T*>(nullptr), output.data(), 1, 8, 1, stream, method));
        reject(api::transform(input.data(), static_cast<T*>(nullptr), 1, 8, 1, stream, method));
        reject(api::transform(input.data(), input.data() + 1, 1, 8, 1, stream, method));
        reject(api::transform(input.data(), output.data(), std::numeric_limits<std::size_t>::max(), 256, 1, stream, method));
        reject(api::quantize_int4(input.data(), reinterpret_cast<std::uint8_t*>(input.data()), scales.data(), 1, 8, stream, method));
        reject(api::transform_int4(input.data(), packed.data(), nullptr, 1, 8, 1, stream, method));
        auto* odd = reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(input.data()) + 1);
        reject(api::transform(odd, output.data(), 1, 8, 1, stream, method));
        auto* bad_scale = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(scales.data()) + 2);
        reject(api::quantize_int4(input.data(), packed.data(), bad_scale, 1, 8, stream, method));
        success(api::transform(static_cast<const T*>(nullptr), static_cast<T*>(nullptr), 0, 8, 1, stream, method));
        success(api::quantize_int4(static_cast<const T*>(nullptr), nullptr, nullptr, 0, 8, stream, method));
        success(api::transform_int4(static_cast<const T*>(nullptr), nullptr, nullptr, 0, 8, 1, stream, method));
        input.download(stream, true, "after-invalid-and-zero-rows");
        output.download(stream, true, "after-invalid-and-zero-rows");
        packed.download(stream, false, "after-invalid-and-zero-rows");
        scales.download(stream, false, "after-invalid-and-zero-rows");
        // 正负半整数：预期手写，避免舍入测试仅复用 CPU 参考实现。
        const std::vector<float> ties{7, -7, .5f, 1.5f, 2.5f, -.5f, -1.5f, -2.5f};
        std::vector<T> t(64, rounded<T>(0));
        for (std::size_t i = 0; i < ties.size(); ++i) t[i] = rounded<T>(ties[i]);
        std::cout << "CONTRACT_PROGRESS method=" << method_name(method) << " phase=ties-upload" << std::endl;
        input.upload(t, stream);
        input.download(stream, true, "after-ties-upload");
        std::cout << "CONTRACT_PROGRESS method=" << method_name(method) << " phase=ties-quantize" << std::endl;
        CHECK(api::quantize_int4(input.data(), packed.data(), scales.data(), 1, 8, stream, method));
        const auto q = packed.download(stream, false, "after-ties-quantize");
        const auto s = scales.download(stream, false, "after-ties-quantize");
        const std::array<std::uint8_t, 4> expected{{0x97, 0x20, 0x02, 0xee}};
        if (!std::equal(expected.begin(), expected.end(), q.begin()) || s[0] != 1.0f)
            throw std::runtime_error("positive/negative ties-to-even test failed");
        input.download(stream, true, "after-ties-quantize");
        ++summary.contract_checks;
    }
    const auto invalid = static_cast<api::Method>(-1);
    reject(api::transform(input.data(), output.data(), 1, 8, 1, stream, invalid));
    reject(api::quantize_int4(input.data(), packed.data(), scales.data(), 1, 8, stream, invalid));
    reject(api::transform_int4(input.data(), packed.data(), scales.data(), 1, 8, 1, stream, invalid));
#if !defined(HADAMARD_ILUVATAR_WARP64) || !defined(__ILUVATAR__)
    auto unsupported = [&](cudaError_t status) {
        if (status != cudaErrorNotSupported) throw std::runtime_error("uncompiled Warp64 path did not return cudaErrorNotSupported");
        ++summary.unsupported_warp64_checks;
    };
    unsupported(api::transform(input.data(), output.data(), 1, 8, 1, stream, api::Method::Warp64));
    unsupported(api::quantize_int4(input.data(), packed.data(), scales.data(), 1, 8, stream, api::Method::Warp64));
    unsupported(api::transform_int4(input.data(), packed.data(), scales.data(), 1, 8, 1, stream, api::Method::Warp64));
    CHECK(api::transform(static_cast<const T*>(nullptr), static_cast<T*>(nullptr), 0, 8, 1, stream, api::Method::Warp64));
    CHECK(api::quantize_int4(static_cast<const T*>(nullptr), nullptr, nullptr, 0, 8, stream, api::Method::Warp64));
    CHECK(api::transform_int4(static_cast<const T*>(nullptr), nullptr, nullptr, 0, 8, 1, stream, api::Method::Warp64));
    summary.unsupported_warp64_checks += 3;
#endif
    CHECK(cudaStreamSynchronize(stream));
    output.download(stream, true);
}


template<class T> Summary validate(cudaStream_t stream, const char* dtype, const Options& options) {
    Summary result;
    const std::vector<int> dims = options.custom_shape ? std::vector<int>{options.dim}
        : (options.quick ? std::vector<int>{1, 64, 256} : std::vector<int>{1, 2, 4, 8, 16, 32, 64, 128, 256});
    const std::vector<std::size_t> rows = options.custom_shape
        ? std::vector<std::size_t>{checked_shape(options.batch, options.seq, options.heads, options.dim)}
        : (options.quick ? std::vector<std::size_t>{3} : std::vector<std::size_t>{1, 3, 17, 257});
    for (int n : dims) {
        for (auto r : rows) {
            for (int normalized = 0; normalized < (n == 1 ? 1 : 2); ++normalized) {
                const float scale = normalized ? 1.0f / std::sqrt(static_cast<float>(n)) : 1.0f;
                for (int pattern = 0; pattern < 5; ++pattern) {
                    // 零值/脉冲无随机性，只计一次；N=1 的两种 scale 相同，也只计一次。
                    const int seeds = pattern < 3 && !options.quick ? 3 : 1;
                    for (int seed = 0; seed < seeds; ++seed)
                        one_case<T>(stream, r, n, scale, pattern, 123 + 7919 * seed, dtype, result);
                }
            }
        }
        std::cout << "VALIDATION_PROGRESS dtype=" << dtype << " n=" << n << " cases=" << result.cases << std::endl;
    }
    if (!options.quick && !options.custom_shape) {
        // 超过 65535 个 block 的网格上限，让同一个 block 必须处理下一行。
        for (int n : {1, 2}) one_case<T>(stream, 65537, n, 1.0f, 0, 1847, dtype, result);
#if defined(HADAMARD_ILUVATAR_WARP64) && defined(__ILUVATAR__)
        // 每个 Warp64 CTA 处理 4 行，262145 同时覆盖网格复用和不满 CTA 的尾行。
        for (int n : {1, 2}) {
            one_case<T>(stream, 262145, n, 1.0f, 0, 4541, dtype, result);
            ++result.warp64_grid_stride_cases;
            result.warp64_grid_stride_elements += 262145u * n;
        }
#endif
    }
    contract_tests<T>(stream, result);
    std::cout << "VALIDATION_PASS dtype=" << dtype << " cases=" << result.cases << " elements=" << result.elements
              << " max_rounded_error=" << std::setprecision(12) << result.max_rounded_error
              << " max_unrounded_error=" << result.max_unrounded_error << " contract_checks=" << result.contract_checks
              << " warp64_cases=" << result.warp64_cases << " exact_warp64_elements=" << result.exact_warp64_elements
              << " warp64_grid_stride_cases=" << result.warp64_grid_stride_cases
              << " unsupported_warp64_checks=" << result.unsupported_warp64_checks << std::endl;
    return result;
}

template<class T> void benchmark(cudaStream_t stream, const char* dtype, const Options& o, std::ofstream& csv) {
    struct Shape { std::size_t b, s, h; int n; };
    std::vector<Shape> shapes;
    if (o.custom_shape) shapes.push_back({o.batch, o.seq, o.heads, o.dim});
    else for (int n : {64, 128, 256}) for (auto rows : {1, 17, 257, 4096, 16384})
        shapes.push_back({rows >= 4096 ? static_cast<std::size_t>(rows / 1024) : 1,
                          rows >= 4096 ? 64 : static_cast<std::size_t>(rows), rows >= 4096 ? 16u : 1u, n});
    struct Configuration { api::Method method; int operation; const char* name; };
    // 无 Warp64 宏时保留原来的 6 路顺序；启用后追加 3 路，独立输出新的原始样本。
    std::vector<Configuration> configurations{
        {api::Method::Baseline, 0, "baseline_transform"}, {api::Method::Optimized, 0, "optimized_transform"},
        {api::Method::Baseline, 1, "baseline_split"}, {api::Method::Optimized, 1, "optimized_split"},
        {api::Method::Baseline, 2, "baseline_fused"}, {api::Method::Optimized, 2, "optimized_fused"}};
#if defined(HADAMARD_ILUVATAR_WARP64) && defined(__ILUVATAR__)
    configurations.push_back({api::Method::Warp64, 0, "warp64_transform"});
    configurations.push_back({api::Method::Warp64, 1, "warp64_split"});
    configurations.push_back({api::Method::Warp64, 2, "warp64_fused"});
#endif
    const int configuration_count = static_cast<int>(configurations.size());
    for (const auto shape : shapes) {
        const auto rows = checked_shape(shape.b, shape.s, shape.h, shape.n), count = rows * shape.n;
        const auto input = make_input<T>(rows, shape.n, 0, 2909);
        Guarded<T> x(count, stream), y(count, stream);
        Guarded<std::uint8_t> q(rows * ((shape.n + 1) / 2), stream);
        Guarded<float> s(rows, stream);
        x.upload(input, stream);
        const float scale = 1.0f;
        auto launch = [&](int which) {
            const auto cfg = configurations[which];
            if (cfg.operation < 2) CHECK(api::transform(x.data(), y.data(), rows, shape.n, scale, stream, cfg.method));
            if (cfg.operation == 1) CHECK(api::quantize_int4(y.data(), q.data(), s.data(), rows, shape.n, stream, cfg.method));
            if (cfg.operation == 2) CHECK(api::transform_int4(x.data(), q.data(), s.data(), rows, shape.n, scale, stream, cfg.method));
        };
        for (int which = 0; which < configuration_count; ++which) for (int i = 0; i < 10; ++i) launch(which);
        CHECK(cudaStreamSynchronize(stream));
        cudaEvent_t begin, end;
        CHECK(cudaEventCreate(&begin)); CHECK(cudaEventCreate(&end));
        for (int group = 0; group < o.groups; ++group) {
            // 各组轮换方法顺序；两端事件之间无分配、CPU 参考或主机设备复制。
            for (int order = 0; order < configuration_count; ++order) {
                const int which = (order + group) % configuration_count;
                const auto cfg = configurations[which];
                CHECK(cudaEventRecord(begin, stream));
                for (int i = 0; i < o.repeats; ++i) launch(which);
                CHECK(cudaEventRecord(end, stream));
                CHECK(cudaEventSynchronize(end));
                float elapsed = 0;
                CHECK(cudaEventElapsedTime(&elapsed, begin, end));
                const double us = static_cast<double>(elapsed) * 1000.0 / o.repeats;
                if (!(us > 0) || !std::isfinite(us)) throw std::runtime_error("invalid event timing");
                const std::size_t logical_bytes = cfg.operation == 0 ? count * sizeof(T) * 2
                    : (cfg.operation == 1 ? count * sizeof(T) * 3 : count * sizeof(T)) + rows * ((shape.n + 1) / 2) + rows * sizeof(float);
                csv << dtype << ',' << shape.b << ',' << shape.s << ',' << shape.h << ',' << shape.n << ',' << rows
                    << ',' << cfg.name << ',' << group << ',' << order << ',' << o.repeats << ',' << std::setprecision(12) << us
                    << ',' << logical_bytes << ',' << logical_bytes / us / 1000.0 << ',' << count * sizeof(T)
                    << ",2909,true,1\n";
                csv.flush();
            }
        }
        CHECK(cudaEventDestroy(begin)); CHECK(cudaEventDestroy(end));
        x.download(stream, true);
        const auto output = y.download(stream);
        const auto packed = q.download(stream);
        const auto scales = s.download(stream);
        std::vector<float> actual(count);
        for (std::size_t i = 0; i < count; ++i) actual[i] = read(output[i]);
        const auto quantized = hadamard::quantize_int4(actual, shape.n);
        if (packed != quantized.packed || scales != quantized.scales)
            throw std::runtime_error("benchmark-size fused INT4 differs from CPU quantization");
        for (std::size_t row : {std::size_t(0), rows / 2, rows - 1}) {
            std::vector<float> sample(shape.n);
            for (int i = 0; i < shape.n; ++i) sample[i] = read(input[row * shape.n + i]);
            const auto expected = hadamard::dense_reference(sample, shape.n, scale);
            for (int i = 0; i < shape.n; ++i) {
                const double error = std::abs(actual[row * shape.n + i] - read(rounded<T>(static_cast<float>(expected[i]))));
                if (!(error < (std::string(dtype) == "fp16" ? .01 : .05)))
                    throw std::runtime_error("benchmark-size sampled dense oracle mismatch");
            }
        }
        std::cout << "BENCHMARK_PROGRESS dtype=" << dtype << " rows=" << rows << " n=" << shape.n << std::endl;
    }
}

void write_summary(std::ostream& f, const char* dtype, const Summary& s) {
    f << '"' << dtype << "\":{\"cases\":" << s.cases << ",\"elements\":" << s.elements
      << ",\"exact_baseline_optimized_elements\":" << s.exact_transform_elements
      << ",\"warp64_cases\":" << s.warp64_cases << ",\"exact_baseline_warp64_elements\":" << s.exact_warp64_elements
      << ",\"warp64_grid_stride_cases\":" << s.warp64_grid_stride_cases
      << ",\"warp64_grid_stride_elements\":" << s.warp64_grid_stride_elements
      << ",\"unsupported_warp64_checks\":" << s.unsupported_warp64_checks
      << ",\"max_abs_error_rounded_fp64\":" << std::setprecision(15) << s.max_rounded_error
      << ",\"max_abs_error_unrounded_fp64\":" << s.max_unrounded_error
      << ",\"api_contract_checks\":" << s.contract_checks << '}';
}

int main(int argc, char** argv) {
    Options options;
    try { options = parse(argc, argv); }
    catch (const std::exception& e) { std::cerr << "INVALID_ARGUMENT " << e.what() << '\n'; return 2; }
    try {
        CHECK(cudaSetDevice(0));
        cudaDeviceProp prop{};
        CHECK(cudaGetDeviceProperties(&prop, 0));
#if defined(HADAMARD_ILUVATAR_WARP64) && defined(__ILUVATAR__)
        if (prop.warpSize != 64) throw std::runtime_error("Warp64 build requires a real device reporting warpSize=64");
#endif
        int runtime = 0, driver = 0;
        CHECK(cudaRuntimeGetVersion(&runtime)); CHECK(cudaDriverGetVersion(&driver));
        std::cout << "DEVICE name=" << prop.name << " warp=" << prop.warpSize << " runtime=" << runtime << " driver=" << driver << std::endl;
        cudaStream_t stream;
        CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        Summary fp16, bf16;
        if (options.validate) {
            if (options.dtype != "bf16") fp16 = validate<__half>(stream, "fp16", options);
            if (options.dtype != "fp16") bf16 = validate<__nv_bfloat16>(stream, "bf16", options);
            std::ofstream json(options.json);
            if (!json) throw std::runtime_error("cannot create validation JSON " + options.json);
            json << "{\"status\":\"PASS\",\"full_matrix\":" << ((!options.quick && !options.custom_shape && options.dtype == "both") ? "true" : "false")
#if defined(HADAMARD_ILUVATAR_WARP64) && defined(__ILUVATAR__)
                 << ",\"warp64_enabled\":true,\"methods\":[\"baseline\",\"optimized\",\"warp64\"]"
#else
                 << ",\"warp64_enabled\":false,\"methods\":[\"baseline\",\"optimized\"]"
#endif
                 << ",\"oracle\":\"all-element FP64 dense, rounded to output dtype\",\"fp16_tolerance_strict\":0.01,\"bf16_tolerance_strict\":0.05,"
                 << "\"warmup_not_counted\":true,\"dtypes\":{";
            bool comma = false;
            if (options.dtype != "bf16") { write_summary(json, "fp16", fp16); comma = true; }
            if (options.dtype != "fp16") { if (comma) json << ','; write_summary(json, "bf16", bf16); }
            json << "}}\n";
            if (!json) throw std::runtime_error("failed writing validation JSON");
        }
        if (options.benchmark) {
            std::ofstream csv(options.csv);
            if (!csv) throw std::runtime_error("cannot create benchmark CSV " + options.csv);
            csv << "dtype,batch,seq,heads,dim,rows,method,group,order,repeats,kernel_us,logical_io_bytes,logical_GBs,input_working_set_bytes,seed,input_read_only,scale\n";
            if (options.dtype != "bf16") benchmark<__half>(stream, "fp16", options, csv);
            if (options.dtype != "fp16") benchmark<__nv_bfloat16>(stream, "bf16", options, csv);
            if (!csv) throw std::runtime_error("failed writing benchmark CSV");
        }
        CHECK(cudaStreamDestroy(stream));
        std::cout << "PASS requested validation/benchmark operations completed" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << std::endl;
        return 1;
    }
}

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

#ifndef HADAMARD_ASCEND_VECTOR_SCALE
#define HADAMARD_ASCEND_VECTOR_SCALE 0
#endif
constexpr bool kVectorScaleEnabled = HADAMARD_ASCEND_VECTOR_SCALE != 0;

namespace api = hadamard::ascend;
#define CHECK(call) do { aclError e_ = (call); if (e_ != ACL_SUCCESS) \
    throw std::runtime_error(std::string(#call) + " returned ACL error " + std::to_string(e_)); } while (0)

// 主机只处理标准 16 位存储位模式，不使用 SDK half/BF16 构造器或 PyTorch。
struct FP16Bits { std::uint16_t bits; };
struct BF16Bits { std::uint16_t bits; };
static_assert(sizeof(FP16Bits) == 2 && sizeof(BF16Bits) == 2, "16-bit storage required");
std::uint32_t configured_blocks = 1;

std::uint32_t float_bits(float x) { std::uint32_t b; std::memcpy(&b, &x, 4); return b; }
float bits_float(std::uint32_t b) { float x; std::memcpy(&x, &b, 4); return x; }
std::uint32_t rshift_rne(std::uint32_t x, unsigned shift) {
    const auto high = x >> shift, low = x & ((1u << shift) - 1u), midpoint = 1u << (shift - 1);
    return high + (low > midpoint || (low == midpoint && (high & 1u)));
}

std::uint16_t encode_half(float x) {
    const auto b = float_bits(x), fraction = b & 0x7fffffu;
    const auto sign = static_cast<std::uint16_t>((b >> 16) & 0x8000u);
    const int exponent_bits = (b >> 23) & 255;
    const int exponent = exponent_bits - 127;
    if (exponent_bits == 255) return sign | (fraction ? 0x7e00u : 0x7c00u);
    if (exponent > 15) return sign | 0x7c00u;
    if (exponent < -25) return sign;
    if (exponent < -14) return sign | rshift_rne(fraction | 0x800000u, -exponent - 1);
    return sign | (((exponent + 15) << 10) + rshift_rne(fraction, 13));
}

std::uint16_t encode_bfloat(float x) {
    const auto b = float_bits(x), high = b >> 16, low = b & 0xffffu;
    if ((b & 0x7f800000u) == 0x7f800000u)
        return static_cast<std::uint16_t>(high | ((b & 0x7fffffu) ? 0x40u : 0u));
    return static_cast<std::uint16_t>(high + (low > 0x8000u || (low == 0x8000u && (high & 1u))));
}

float decode_half(std::uint16_t b) {
    const int exponent = (b >> 10) & 31, fraction = b & 1023;
    if (exponent == 31) return fraction ? std::numeric_limits<float>::quiet_NaN()
        : ((b & 0x8000u) ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity());
    const float x = exponent ? std::ldexp(static_cast<float>(1024 + fraction), exponent - 25)
                             : std::ldexp(static_cast<float>(fraction), -24);
    return b & 0x8000u ? -x : x;
}

template<class T> float read(T);
template<> float read(FP16Bits x) { return decode_half(x.bits); }
template<> float read(BF16Bits x) { return bits_float(static_cast<std::uint32_t>(x.bits) << 16); }
template<class T> T rounded(float);
template<> FP16Bits rounded(float x) { return {encode_half(x)}; }
template<> BF16Bits rounded(float x) { return {encode_bfloat(x)}; }
template<class T> api::StorageType storage();
template<> api::StorageType storage<FP16Bits>() { return api::StorageType::FP16; }
template<> api::StorageType storage<BF16Bits>() { return api::StorageType::BF16; }

template<class T> aclError launch_transform(const T* x, T* y, std::size_t rows, int n, float scale,
                                           aclrtStream stream, api::Method method, std::uint32_t blocks = configured_blocks) {
    return api::transform(reinterpret_cast<const std::uint16_t*>(x), reinterpret_cast<std::uint16_t*>(y),
                          rows, static_cast<std::uint32_t>(n), scale, storage<T>(), method, stream, blocks);
}
template<class T> aclError launch_quantize(const T* x, std::uint8_t* q, float* scales, std::size_t rows, int n,
                                          aclrtStream stream, api::Method method, std::uint32_t blocks = configured_blocks) {
    return api::quantize_int4(reinterpret_cast<const std::uint16_t*>(x), q, scales, rows,
                              static_cast<std::uint32_t>(n), storage<T>(), method, stream, blocks);
}
template<class T> aclError launch_fused(const T* x, std::uint8_t* q, float* scales, std::size_t rows, int n, float scale,
                                       aclrtStream stream, api::Method method, std::uint32_t blocks = configured_blocks) {
    return api::transform_int4(reinterpret_cast<const std::uint16_t*>(x), q, scales, rows,
                               static_cast<std::uint32_t>(n), scale, storage<T>(), method, stream, blocks);
}

// 每个区间都有前后哨兵。17 个元素的偏移同时覆盖仅 2 字节对齐的 FP16/BF16 指针。
template<class T> class Guarded {
    static constexpr std::size_t guard = 17;
    T* raw_ = nullptr;
    void* pinned_ = nullptr;
    aclrtStream stream_;
    std::size_t count_;
    std::vector<unsigned char> initial_;
    std::string name_;
public:
    explicit Guarded(std::size_t count, aclrtStream stream, std::string name = "unnamed")
        : stream_(stream), count_(count), name_(name) {
        if (count > (std::numeric_limits<std::size_t>::max() - 63) / sizeof(T) - 2 * guard)
            throw std::runtime_error("guarded allocation size overflow");
        const auto total = ((count + 2 * guard) * sizeof(T) + 63) / 64 * 64;
        initial_.assign(total, 0xa5);
        try {
            CHECK(aclrtMallocHost(&pinned_, total));
            CHECK(aclrtMalloc(reinterpret_cast<void**>(&raw_), total, ACL_MEM_MALLOC_HUGE_FIRST));
            CHECK(aclrtMemsetAsync(raw_, total, 0xa5, total, stream));
        } catch (...) {
            if (raw_) aclrtFree(raw_);
            if (pinned_) aclrtFreeHost(pinned_);
            throw;
        }
    }
    ~Guarded() {
        if (stream_) aclrtSynchronizeStream(stream_);
        if (raw_) aclrtFree(raw_);
        if (pinned_) aclrtFreeHost(pinned_);
    }
    Guarded(const Guarded&) = delete;
    Guarded& operator=(const Guarded&) = delete;
    T* data() { return raw_ + guard; }
    void upload(const std::vector<T>& values, aclrtStream stream) {
        if (values.size() != count_) throw std::runtime_error("upload size mismatch");
        std::memcpy(initial_.data() + guard * sizeof(T), values.data(), count_ * sizeof(T));
        std::memcpy(pinned_, initial_.data(), initial_.size());
        CHECK(aclrtMemcpyAsync(raw_, initial_.size(), pinned_, initial_.size(), ACL_MEMCPY_HOST_TO_DEVICE, stream));
    }
    std::vector<T> download(aclrtStream stream, bool unchanged = false, const char* phase = "readback") {
        CHECK(aclrtMemcpyAsync(pinned_, initial_.size(), raw_, initial_.size(), ACL_MEMCPY_DEVICE_TO_HOST, stream));
        CHECK(aclrtSynchronizeStream(stream));
        const auto* bytes = static_cast<const unsigned char*>(pinned_);
        const std::size_t prefix = guard * sizeof(T), end = prefix + count_ * sizeof(T);
        for (std::size_t i = 0; i < initial_.size(); ++i) {
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
        std::memcpy(result.data(), bytes + prefix, count_ * sizeof(T));
        return result;
    }
};

struct Options {
    bool validate = false, benchmark = false, custom_shape = false, quick = false;
    bool skip_stress = false;
    std::size_t batch = 1, seq = 1, heads = 1;
    int dim = 128, repeats = 5, groups = 5, warmup = 3;
    std::uint32_t block_dim = 1;
    std::string dtype = "both", csv = "ascend_benchmark.csv", json = "ascend_validation.json";
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
    if (!rows) throw std::invalid_argument("rows must be positive for CLI tensors");
    multiply(multiply(rows, static_cast<std::size_t>(n)), sizeof(FP16Bits));
    multiply(rows, sizeof(float));
    return rows;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--validate") o.validate = true;
        else if (key == "--benchmark") o.benchmark = true;
        else if (key == "--quick") o.quick = true;
        else if (key == "--skip-stress") o.skip_stress = true;
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
            } else if (key == "--block-dim") {
                const auto v = positive(value, key.c_str());
                if (v > 32) throw std::invalid_argument("block-dim must be in [1,32]");
                o.block_dim = static_cast<std::uint32_t>(v);
            } else if (key == "--repeats" || key == "--groups" || key == "--warmup") {
                const auto v = (key == "--warmup" && value == "0") ? 0 : positive(value, key.c_str());
                if (v > 10000) throw std::invalid_argument("repeats/groups/warmup exceed 10000");
                if (key == "--repeats") o.repeats = static_cast<int>(v);
                else if (key == "--groups") o.groups = static_cast<int>(v);
                else o.warmup = static_cast<int>(v);
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
    std::size_t grid_cases = 0, grid_elements = 0, large_m_cases = 0, large_m_elements = 0;
    double grid_max_rounded_error = 0, grid_max_unrounded_error = 0;
    double max_rounded_error = 0, max_unrounded_error = 0;
};

template<class T> void one_case(aclrtStream stream, std::size_t rows, int n, float scale,
                               int pattern, unsigned seed, const char* dtype, Summary& summary) {
    const auto input = make_input<T>(rows, n, pattern, seed);
    const std::size_t size = input.size(), bytes = rows * ((n + 1) / 2);
    const std::string context = std::string(dtype) + " rows=" + std::to_string(rows) + " n=" + std::to_string(n)
        + " pattern=" + std::to_string(pattern) + " seed=" + std::to_string(seed) + " scale=" + std::to_string(scale)
        + " block_dim=" + std::to_string(configured_blocks);
    Guarded<T> x(size, stream, context + " input"), baseline(size, stream, context + " scalar"),
               optimized(size, stream, context + " vector"), inplace(size, stream, context + " vector-inplace"),
               scalar_inplace(size, stream, context + " scalar-inplace");
    Guarded<std::uint8_t> base_split(bytes, stream, context + " scalar-split"), opt_split(bytes, stream, context + " vector-split"),
                          base_fused(bytes, stream, context + " scalar-fused"), opt_fused(bytes, stream, context + " vector-fused");
    Guarded<float> bs(rows, stream, context + " scalar-split-scales"), os(rows, stream, context + " vector-split-scales"),
                   bfs(rows, stream, context + " scalar-fused-scales"), ofs(rows, stream, context + " vector-fused-scales");
    x.upload(input, stream);
    inplace.upload(input, stream);
    scalar_inplace.upload(input, stream);
    CHECK(launch_transform(x.data(), baseline.data(), rows, n, scale, stream, api::Method::ScalarButterfly));
    CHECK(launch_transform(x.data(), optimized.data(), rows, n, scale, stream, api::Method::VectorGather));
    CHECK(launch_transform(inplace.data(), inplace.data(), rows, n, scale, stream, api::Method::VectorGather));
    CHECK(launch_transform(scalar_inplace.data(), scalar_inplace.data(), rows, n, scale, stream, api::Method::ScalarButterfly));
    CHECK(launch_quantize(baseline.data(), base_split.data(), bs.data(), rows, n, stream, api::Method::ScalarButterfly));
    CHECK(launch_quantize(optimized.data(), opt_split.data(), os.data(), rows, n, stream, api::Method::VectorGather));
    CHECK(launch_fused(x.data(), base_fused.data(), bfs.data(), rows, n, scale, stream, api::Method::ScalarButterfly));
    CHECK(launch_fused(x.data(), opt_fused.data(), ofs.data(), rows, n, scale, stream, api::Method::VectorGather));
    CHECK(aclrtSynchronizeStream(stream));
    x.download(stream, true);
    const auto a = baseline.download(stream), b = optimized.download(stream), ip = inplace.download(stream), sip = scalar_inplace.download(stream);
    if (std::memcmp(a.data(), b.data(), size * sizeof(T)) || std::memcmp(b.data(), ip.data(), size * sizeof(T))
        || std::memcmp(a.data(), sip.data(), size * sizeof(T)))
        throw std::runtime_error("scalar/vector/in-place transform not bitwise identical: " + context);
    std::vector<float> fx(size), actual(size);
    for (std::size_t i = 0; i < size; ++i) { fx[i] = read(input[i]); actual[i] = read(b[i]); }
    auto fwht = fx;
    hadamard::fwht(fwht.data(), rows, n, scale);
    for (std::size_t i = 0; i < size; ++i) {
        if (rounded<T>(fwht[i]).bits != b[i].bits)
            throw std::runtime_error("rounded FP32 CPU FWHT mismatch: " + context + " index=" + std::to_string(i));
    }
    // 第二项独立参考使用 O(N^2) FP64 稠密矩阵公式，不共享蝶形代码。
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
        throw std::runtime_error("CPU/scalar/vector split/fused INT4 bytes or scales mismatch: " + context);
    ++summary.cases;
    summary.elements += size;
    summary.exact_transform_elements += size;
}

template<class T> void contract_tests(aclrtStream stream, Summary& summary) {
    Guarded<T> input(64, stream, "contract.input"), output(64, stream, "contract.output");
    Guarded<std::uint8_t> packed(32, stream, "contract.packed");
    Guarded<float> scales(8, stream, "contract.scales");
    input.download(stream, true, "initialized");
    output.download(stream, true, "initialized");
    packed.download(stream, true, "initialized");
    scales.download(stream, true, "initialized");
    auto reject = [&](aclError status) {
        if (status != ACL_ERROR_INVALID_PARAM) throw std::runtime_error("invalid API input did not return ACL_ERROR_INVALID_PARAM");
        ++summary.contract_checks;
    };
    auto success = [&](aclError status) { CHECK(status); ++summary.contract_checks; };
    for (const auto method : {api::Method::ScalarButterfly, api::Method::VectorGather}) {
        const char* method_name = method == api::Method::ScalarButterfly ? "scalar_butterfly" : "vector_gather";
        std::cout << "CONTRACT_PROGRESS method=" << method_name << " phase=invalid-parameters" << std::endl;
        for (int n : {0, 3, 512}) {
            reject(launch_transform(input.data(), output.data(), 1, n, 1, stream, method));
            reject(launch_quantize(input.data(), packed.data(), scales.data(), 1, n, stream, method));
            reject(launch_fused(input.data(), packed.data(), scales.data(), 1, n, 1, stream, method));
        }
        for (float scale : {0.0f, -1.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
            reject(launch_transform(input.data(), output.data(), 1, 8, scale, stream, method));
            reject(launch_fused(input.data(), packed.data(), scales.data(), 1, 8, scale, stream, method));
        }
        reject(launch_transform(static_cast<const T*>(nullptr), output.data(), 1, 8, 1, stream, method));
        reject(launch_transform(input.data(), static_cast<T*>(nullptr), 1, 8, 1, stream, method));
        reject(launch_transform(input.data(), input.data() + 1, 1, 8, 1, stream, method));
        reject(launch_transform(input.data(), output.data(), std::numeric_limits<std::size_t>::max(), 256, 1, stream, method));
        reject(launch_quantize(input.data(), reinterpret_cast<std::uint8_t*>(input.data()), scales.data(), 1, 8, stream, method));
        reject(launch_fused(input.data(), packed.data(), nullptr, 1, 8, 1, stream, method));
        auto* odd = reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(input.data()) + 1);
        reject(launch_transform(odd, output.data(), 1, 8, 1, stream, method));
        auto* bad_scale = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(scales.data()) + 2);
        reject(launch_quantize(input.data(), packed.data(), bad_scale, 1, 8, stream, method));
        success(launch_transform(static_cast<const T*>(nullptr), static_cast<T*>(nullptr), 0, 8, 1, stream, method));
        success(launch_quantize(static_cast<const T*>(nullptr), nullptr, nullptr, 0, 8, stream, method));
        success(launch_fused(static_cast<const T*>(nullptr), nullptr, nullptr, 0, 8, 1, stream, method));
        success(launch_transform(static_cast<const T*>(nullptr), static_cast<T*>(nullptr), 0, 8, 1, nullptr, method));
        success(launch_quantize(static_cast<const T*>(nullptr), nullptr, nullptr, 0, 8, nullptr, method));
        success(launch_fused(static_cast<const T*>(nullptr), nullptr, nullptr, 0, 8, 1, nullptr, method));
        for (std::uint32_t blocks : {0u, 33u}) {
            reject(launch_transform(input.data(), output.data(), 1, 8, 1, stream, method, blocks));
            reject(launch_quantize(input.data(), packed.data(), scales.data(), 1, 8, stream, method, blocks));
            reject(launch_fused(input.data(), packed.data(), scales.data(), 1, 8, 1, stream, method, blocks));
            reject(launch_transform(static_cast<const T*>(nullptr), static_cast<T*>(nullptr), 0, 8, 1, nullptr, method, blocks));
        }
        const auto bad_storage = static_cast<api::StorageType>(99);
        const auto* raw_input = reinterpret_cast<const std::uint16_t*>(input.data());
        auto* raw_output = reinterpret_cast<std::uint16_t*>(output.data());
        reject(api::transform(raw_input, raw_output, 1, 8, 1, bad_storage, method, stream));
        reject(api::quantize_int4(raw_input, packed.data(), scales.data(), 1, 8, bad_storage, method, stream));
        reject(api::transform_int4(raw_input, packed.data(), scales.data(), 1, 8, 1, bad_storage, method, stream));
        reject(api::transform(nullptr, nullptr, 0, 8, 1, bad_storage, method, nullptr));
        input.download(stream, true, "after-invalid-and-zero-rows");
        output.download(stream, true, "after-invalid-and-zero-rows");
        packed.download(stream, false, "after-invalid-and-zero-rows");
        scales.download(stream, false, "after-invalid-and-zero-rows");
        // 正负半整数：预期手写，避免舍入测试仅复用 CPU 参考实现。
        const std::vector<float> ties{7, -7, .5f, 1.5f, 2.5f, -.5f, -1.5f, -2.5f};
        std::vector<T> t(64, rounded<T>(0));
        for (std::size_t i = 0; i < ties.size(); ++i) t[i] = rounded<T>(ties[i]);
        std::cout << "CONTRACT_PROGRESS method=" << method_name << " phase=ties-upload" << std::endl;
        input.upload(t, stream);
        input.download(stream, true, "after-ties-upload");
        std::cout << "CONTRACT_PROGRESS method=" << method_name << " phase=ties-quantize" << std::endl;
        CHECK(launch_quantize(input.data(), packed.data(), scales.data(), 1, 8, stream, method));
        const auto q = packed.download(stream, false, "after-ties-quantize");
        const auto s = scales.download(stream, false, "after-ties-quantize");
        const std::array<std::uint8_t, 4> expected{{0x97, 0x20, 0x02, 0xee}};
        if (!std::equal(expected.begin(), expected.end(), q.begin()) || s[0] != 1.0f)
            throw std::runtime_error("positive/negative ties-to-even test failed");
        input.download(stream, true, "after-ties-quantize");
        ++summary.contract_checks;

        // 已观测的除法中点回归：1/(FP32(1/7))与0.5/(FP32(1/7))应量化为7、3。
        // 近似除法可能把第二项抬至3.5，错误地产生0x47。预期完全固定，不从actual生成。
        Guarded<T> chain_input(2, stream, "division-chain.input"),
                   chain_transform(2, stream, "division-chain.transform");
        Guarded<std::uint8_t> chain_split(1, stream, "division-chain.split"),
                              chain_fused(1, stream, "division-chain.fused");
        Guarded<float> chain_split_scale(1, stream, "division-chain.split_scale"),
                       chain_fused_scale(1, stream, "division-chain.fused_scale");
        const std::vector<T> chain_values{rounded<T>(0.75f), rounded<T>(0.25f)};
        const std::array<T, 2> expected_transform{{rounded<T>(1.0f), rounded<T>(0.5f)}};
        constexpr std::uint8_t expected_packed = 0x37;
        constexpr std::uint32_t expected_scale_bits = 0x3e124925u;
        chain_input.upload(chain_values, stream);
        std::cout << "CONTRACT_PROGRESS method=" << method_name << " phase=division-chain" << std::endl;
        CHECK(launch_transform(chain_input.data(), chain_transform.data(), 1, 2, 1.0f, stream, method));
        CHECK(launch_quantize(chain_transform.data(), chain_split.data(), chain_split_scale.data(), 1, 2, stream, method));
        CHECK(launch_fused(chain_input.data(), chain_fused.data(), chain_fused_scale.data(), 1, 2, 1.0f, stream, method));
        chain_input.download(stream, true, "division-chain-complete");
        const auto chain_y = chain_transform.download(stream, false, "division-chain-complete");
        const auto chain_qs = chain_split.download(stream, false, "division-chain-complete");
        const auto chain_qf = chain_fused.download(stream, false, "division-chain-complete");
        const auto chain_ss = chain_split_scale.download(stream, false, "division-chain-complete");
        const auto chain_sf = chain_fused_scale.download(stream, false, "division-chain-complete");
        if (std::memcmp(chain_y.data(), expected_transform.data(), 2 * sizeof(T))
            || chain_qs[0] != expected_packed || chain_qf[0] != expected_packed
            || float_bits(chain_ss[0]) != expected_scale_bits || float_bits(chain_sf[0]) != expected_scale_bits) {
            std::cerr << "DIVISION_CHAIN_FAIL method=" << method_name
                      << " split_packed=" << unsigned(chain_qs[0]) << " fused_packed=" << unsigned(chain_qf[0])
                      << " split_scale_bits=" << float_bits(chain_ss[0]) << " fused_scale_bits=" << float_bits(chain_sf[0]) << std::endl;
            throw std::runtime_error("fixed transform/quantize/fused division-chain regression failed");
        }
        ++summary.contract_checks;  // 每个dtype/Method计一个串联回归，不把其中三次API发射算成三个用例。
        std::cout << "DIVISION_CHAIN_PASS dtype=" << (storage<T>() == api::StorageType::FP16 ? "fp16" : "bf16")
                  << " method=" << method_name
                  << " input=[0.75,0.25] transform=[1,0.5] scale_bits=0x3e124925 packed=0x37 split_fused_exact=true guards=true contract_increment=1"
                  << std::endl;
    }
    const auto invalid = static_cast<api::Method>(-1);
    reject(launch_transform(input.data(), output.data(), 1, 8, 1, stream, invalid));
    reject(launch_quantize(input.data(), packed.data(), scales.data(), 1, 8, stream, invalid));
    reject(launch_fused(input.data(), packed.data(), scales.data(), 1, 8, 1, stream, invalid));
    reject(launch_transform(static_cast<const T*>(nullptr), static_cast<T*>(nullptr), 0, 8, 1, nullptr, invalid));
    CHECK(aclrtSynchronizeStream(stream));
    output.download(stream, true);
}

template<class T> void large_m_check(aclrtStream stream, Summary& summary) {
    constexpr std::size_t rows = 262145;
    constexpr std::uint32_t blocks = 32;
    const auto input = make_input<T>(rows, 1, 0, 1847);
    Guarded<T> x(rows, stream, "large-m.input"), scalar(rows, stream, "large-m.scalar"),
               vector(rows, stream, "large-m.vector"), inplace(rows, stream, "large-m.inplace");
    x.upload(input, stream); inplace.upload(input, stream);
    std::cout << "LARGE_M_START rows=" << rows << " n=1 block_dim=" << blocks << " operations=transform,inplace" << std::endl;
    CHECK(launch_transform(x.data(), scalar.data(), rows, 1, 1, stream, api::Method::ScalarButterfly, blocks));
    CHECK(launch_transform(x.data(), vector.data(), rows, 1, 1, stream, api::Method::VectorGather, blocks));
    CHECK(launch_transform(inplace.data(), inplace.data(), rows, 1, 1, stream, api::Method::VectorGather, blocks));
    const auto a = scalar.download(stream), b = vector.download(stream), c = inplace.download(stream);
    if (std::memcmp(input.data(), a.data(), rows * sizeof(T)) || std::memcmp(input.data(), b.data(), rows * sizeof(T))
        || std::memcmp(input.data(), c.data(), rows * sizeof(T))) throw std::runtime_error("large-M N1 identity mismatch");
    x.download(stream, true);
    summary.large_m_cases = 1;
    summary.large_m_elements = rows;
    std::cout << "LARGE_M_PASS rows=" << rows << " n=1 block_dim=" << blocks << std::endl;
}

template<class T> Summary validate(aclrtStream stream, const char* dtype, const Options& options) {
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
    if (!options.custom_shape) {
        const auto original_blocks = configured_blocks;
        Summary grid;
        for (std::uint32_t blocks = 1; blocks <= 32; ++blocks) {
            if (options.quick && blocks != original_blocks) continue;
            configured_blocks = blocks;
            for (int n : {1, 256}) one_case<T>(stream, 33, n, 1.0f, 0, 1741, dtype, grid);
            std::cout << "GRID_PROGRESS dtype=" << dtype << " rows=33 block_dim=" << blocks << std::endl;
        }
        configured_blocks = original_blocks;
        result.grid_cases = grid.cases;
        result.grid_elements = grid.elements;
        result.grid_max_rounded_error = grid.max_rounded_error;
        result.grid_max_unrounded_error = grid.max_unrounded_error;
    }
    if (!options.quick && !options.custom_shape && !options.skip_stress) {
        large_m_check<T>(stream, result);
    }
    contract_tests<T>(stream, result);
    std::cout << "VALIDATION_PASS dtype=" << dtype << " cases=" << result.cases << " elements=" << result.elements
              << " max_rounded_error=" << std::setprecision(12) << result.max_rounded_error
              << " max_unrounded_error=" << result.max_unrounded_error << " contract_checks=" << result.contract_checks << std::endl;
    return result;
}

template<class T> void benchmark(aclrtStream stream, const char* dtype, const Options& o, std::ofstream& csv) {
    struct Shape { std::size_t b, s, h; int n; };
    std::vector<Shape> shapes;
    if (o.custom_shape) shapes.push_back({o.batch, o.seq, o.heads, o.dim});
    else for (int n : {64, 128, 256}) for (auto rows : {1, 17, 257})
        shapes.push_back({1, static_cast<std::size_t>(rows), 1, n});
    const std::array<const char*, 7> names{{"scalar_transform", "vector_transform", "scalar_split",
                                         "vector_split", "scalar_fused", "vector_fused", "quant_only"}};
    for (const auto shape : shapes) {
        const auto rows = checked_shape(shape.b, shape.s, shape.h, shape.n), count = rows * shape.n;
        const auto input = make_input<T>(rows, shape.n, 0, 2909);
        Guarded<T> x(count, stream), y(count, stream);
        Guarded<std::uint8_t> q(rows * ((shape.n + 1) / 2), stream);
        Guarded<float> s(rows, stream);
        x.upload(input, stream);
        const float scale = 1.0f;
        // quant_only 的输入在计时前生成；两方法共享同一量化实现，只报告一条量化基准。
        CHECK(launch_transform(x.data(), y.data(), rows, shape.n, scale, stream, api::Method::ScalarButterfly));
        auto launch = [&](int which) {
            const auto method = which % 2 ? api::Method::VectorGather : api::Method::ScalarButterfly;
            if (which < 4) CHECK(launch_transform(x.data(), y.data(), rows, shape.n, scale, stream, method));
            if (which >= 2 && which < 4) CHECK(launch_quantize(y.data(), q.data(), s.data(), rows, shape.n, stream, method));
            if (which >= 4 && which < 6) CHECK(launch_fused(x.data(), q.data(), s.data(), rows, shape.n, scale, stream, method));
            if (which == 6) CHECK(launch_quantize(y.data(), q.data(), s.data(), rows, shape.n, stream, api::Method::ScalarButterfly));
        };
        for (int which = 0; which < 7; ++which) for (int i = 0; i < o.warmup; ++i) launch(which);
        CHECK(aclrtSynchronizeStream(stream));
        for (int group = 0; group < o.groups; ++group) {
            // 各组轮换方法顺序；两端事件之间无分配、CPU 参考或主机设备复制。
            for (int order = 0; order < 7; ++order) {
                const int which = (order + group) % 7;
                aclrtEvent begin = nullptr, end = nullptr;
                // 每个样本新建时间线 event，避免假设旧 event 的 reset/reuse 语义。
                CHECK(aclrtCreateEventWithFlag(&begin, ACL_EVENT_TIME_LINE));
                try { CHECK(aclrtCreateEventWithFlag(&end, ACL_EVENT_TIME_LINE)); }
                catch (...) { aclrtDestroyEvent(begin); throw; }
                float elapsed = 0;
                try {
                CHECK(aclrtRecordEvent(begin, stream));
                for (int i = 0; i < o.repeats; ++i) launch(which);
                CHECK(aclrtRecordEvent(end, stream));
                CHECK(aclrtSynchronizeStream(stream));
                CHECK(aclrtEventElapsedTime(&elapsed, begin, end));
                } catch (...) {
                    aclrtSynchronizeStream(stream); aclrtDestroyEvent(begin); aclrtDestroyEvent(end); throw;
                }
                CHECK(aclrtDestroyEvent(begin)); CHECK(aclrtDestroyEvent(end));
                const double us = static_cast<double>(elapsed) * 1000.0 / o.repeats;
                if (!(us > 0) || !std::isfinite(us)) throw std::runtime_error("invalid event timing");
                const std::size_t logical_bytes = which < 2 ? count * sizeof(T) * 2
                    : (which < 4 ? count * sizeof(T) * 3 : count * sizeof(T)) + rows * ((shape.n + 1) / 2) + rows * sizeof(float);
                csv << dtype << ',' << shape.b << ',' << shape.s << ',' << shape.h << ',' << shape.n << ',' << rows
                    << ',' << names[which] << ',' << group << ',' << order << ',' << o.repeats << ',' << std::setprecision(12) << us
                    << ',' << logical_bytes << ',' << logical_bytes / us / 1000.0 << ',' << count * sizeof(T)
                    << ",2909,true,1," << configured_blocks << ',' << o.warmup << ",acl_timeline_event_ms,"
                    << (kVectorScaleEnabled ? "true" : "false") << ',' << us / 1000.0 << '\n';
                csv.flush();
            }
        }
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
      << ",\"exact_scalar_vector_elements\":" << s.exact_transform_elements
      << ",\"fp32_fwht_output_bits_exact\":true"
      << ",\"max_abs_error_rounded_fp64\":" << std::setprecision(15) << s.max_rounded_error
      << ",\"max_abs_error_unrounded_fp64\":" << s.max_unrounded_error
      << ",\"api_contract_checks\":" << s.contract_checks
      << ",\"grid_cases\":" << s.grid_cases << ",\"grid_elements\":" << s.grid_elements
      << ",\"grid_max_abs_error_rounded_fp64\":" << s.grid_max_rounded_error
      << ",\"grid_max_abs_error_unrounded_fp64\":" << s.grid_max_unrounded_error
      << ",\"large_m_cases\":" << s.large_m_cases << ",\"large_m_elements\":" << s.large_m_elements << '}';
}

int main(int argc, char** argv) {
    Options options;
    try { options = parse(argc, argv); }
    catch (const std::exception& e) { std::cerr << "INVALID_ARGUMENT " << e.what() << '\n'; return 2; }
    aclrtStream stream = nullptr;
    bool initialized = false, device_set = false;
    try {
        configured_blocks = options.block_dim;
        CHECK(aclInit(nullptr)); initialized = true;
        CHECK(aclrtSetDevice(0)); device_set = true;
        CHECK(aclrtCreateStream(&stream));
        const char* soc = aclrtGetSocName();
        if (!soc || std::string(soc) != "Ascend910B1") throw std::runtime_error("this build requires observed SOC Ascend910B1");
        std::cout << "DEVICE soc=" << soc << " execution=npu block_dim=" << configured_blocks
                  << " vector_scale_enabled=" << (kVectorScaleEnabled ? "true" : "false") << std::endl;
        Summary fp16, bf16;
        if (options.validate) {
            if (options.dtype != "bf16") fp16 = validate<FP16Bits>(stream, "fp16", options);
            if (options.dtype != "fp16") bf16 = validate<BF16Bits>(stream, "bf16", options);
            std::ofstream json(options.json);
            if (!json) throw std::runtime_error("cannot create validation JSON " + options.json);
            json << "{\"status\":\"PASS\",\"full_matrix\":" << ((!options.quick && !options.custom_shape && options.dtype == "both") ? "true" : "false")
                 << ",\"full_suite_complete\":" << ((!options.quick && !options.custom_shape && !options.skip_stress && options.dtype == "both") ? "true" : "false")
                 << ",\"execution\":\"npu\",\"soc\":\"Ascend910B1\",\"main_block_dim\":" << configured_blocks
                 << ",\"methods\":[\"scalar_butterfly\",\"vector_gather\"]"
                 << ",\"vector_scale_enabled\":" << (kVectorScaleEnabled ? "true" : "false")
                 << ",\"oracle\":\"FP32 CPU FWHT exact output bits plus all-element dtype-rounded FP64 dense\",\"fp16_tolerance_strict\":0.01,\"bf16_tolerance_strict\":0.05,"
                 << "\"large_m_definition\":\"262145 rows, N1, both transforms and vector in-place, block_dim32; no large-M INT4 claim; does not exercise indices above 2^32\","
                 << "\"large_m_skipped\":" << ((options.quick || options.custom_shape || options.skip_stress) ? "true" : "false") << ','
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
            csv << "dtype,batch,seq,heads,dim,rows,method,group,order,repeats,kernel_us,logical_io_bytes,logical_GBs,input_working_set_bytes,seed,input_read_only,scale,block_dim,warmup,timer,vector_scale_enabled,kernel_ms\n";
            if (options.dtype != "bf16") benchmark<FP16Bits>(stream, "fp16", options, csv);
            if (options.dtype != "fp16") benchmark<BF16Bits>(stream, "bf16", options, csv);
            if (!csv) throw std::runtime_error("failed writing benchmark CSV");
        }
        CHECK(aclrtDestroyStream(stream)); stream = nullptr;
        CHECK(aclrtResetDevice(0)); device_set = false;
        CHECK(aclFinalize()); initialized = false;
        std::cout << "PASS requested validation/benchmark operations completed" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << std::endl;
        if (stream) { aclrtSynchronizeStream(stream); aclrtDestroyStream(stream); }
        if (device_set) aclrtResetDevice(0);
        if (initialized) aclFinalize();
        return 1;
    }
}

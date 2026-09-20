#include "project.hpp"
#include <algorithm>
#include <limits>
#include <random>
namespace lp {
uint64_t checked_size(uint64_t r, uint64_t c) {
    if (!r || !c || r > INT64_MAX || c > INT64_MAX || r > uint64_t(SIZE_MAX) / 4 / c)
        throw std::runtime_error("invalid/overflow shape");
    return r * c;
}
void validate(const Config &c) {
    if (c.format < 0 || c.format > 2 || c.block != (c.format == 2 ? 16 : 32) || c.mode < 0 ||
        c.mode > 1 || c.out < 0 || c.out > 2 || c.round < 0 || c.round > 1)
        throw std::runtime_error("invalid configuration");
}
Tensor generate(uint64_t r, uint64_t c, int dtype, const std::string &dist, uint64_t seed) {
    uint64_t n = checked_size(r, c);
    if (dtype < 0 || dtype > 1)
        throw std::runtime_error("input dtype must be fp32/fp16");
    if (dist != "uniform" && dist != "normal" && dist != "outliers")
        throw std::runtime_error("invalid distribution");
    Tensor t{r, c, dtype, {}};
    t.data.resize(n * (dtype ? 2 : 4));
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0, 1);
    std::uniform_real_distribution<float> uni(-1, 1);
    for (uint64_t i = 0; i < n; ++i) {
        float x = dist == "uniform" ? uni(rng) : normal(rng);
        if (dist == "outliers" && i % 100 == 0)
            x *= 100;
        store_value(t.data.data(), i, dtype, output_bits(x, dtype));
    }
    return t;
}
Packed quant_cpu(const Tensor &t, const Config &c) {
    validate(c);
    uint64_t n = checked_size(t.rows, t.cols);
    if (t.dtype < 0 || t.dtype > 1 || t.data.size() != n * (t.dtype ? 2 : 4))
        throw std::runtime_error("invalid input tensor");
    Packed p{t.rows, t.cols, t.dtype, c, 1, {}, {}};
    // 沿行方向分块：每行 bpr 个块，共 rows * bpr 个块
    uint64_t bpr = (t.cols + c.block - 1) / c.block;
    uint64_t blocks = c.mode ? 1 : t.rows * bpr;
    p.scales.resize(blocks);
    p.data.resize(c.format == 2 ? (n + 1) / 2 : n);
    double all = 0;
    for (uint64_t i = 0; i < n; ++i) {
        float x = input_value(t.data.data(), i, t.dtype);
        if (!std::isfinite(x))
            throw std::runtime_error("nonfinite input");
        all = std::max(all, std::abs(double(x)));
    }
    if (c.format == 2)
        p.global = global_scale(all);
    for (uint64_t b = 0; b < blocks; ++b) {
        double a = 0;
        if (c.mode) {
            // 全局模式：扫描整个张量
            for (uint64_t i = 0; i < n; ++i)
                a = std::max(a, std::abs(double(input_value(t.data.data(), i, t.dtype))));
        } else {
            // 行方向块 b 对应第 r 行第 cb 块
            uint64_t r = b / bpr;
            uint64_t cb = b % bpr;
            uint64_t start = r * t.cols + cb * uint64_t(c.block);
            uint64_t end = std::min(r * t.cols + t.cols, start + c.block);
            for (uint64_t i = start; i < end; ++i)
                a = std::max(a, std::abs(double(input_value(t.data.data(), i, t.dtype))));
        }
        p.scales[b] = c.format == 2 ? nvscale(a, p.global) : mxscale(a, c.format);
    }
    for (uint64_t j = 0; j < p.data.size(); ++j) {
        unsigned byte = 0;
        int count = c.format == 2 ? 2 : 1;
        for (int k = 0; k < count; ++k) {
            uint64_t i = j * count + k;
            if (i >= n)
                break;
            // 元素 i 所在的行方向块号
            uint64_t blk = c.mode ? 0 : (i / t.cols) * bpr + (i % t.cols) / c.block;
            double s = scale_value(p.scales[blk], c.format, p.global);
            uint8_t q = encode(double(input_value(t.data.data(), i, t.dtype)) / s, c.format,
                               c.round, c.seed, i);
            byte |= unsigned(q) << (k * 4);
        }
        p.data[j] = byte;
    }
    return p;
}
void validate_packed(const Packed &p) {
    validate(p.c);
    uint64_t n = checked_size(p.rows, p.cols);
    uint64_t bpr = (p.cols + p.c.block - 1) / p.c.block;
    if (p.input_dtype < 0 || p.input_dtype > 1 || !std::isfinite(p.global) || p.global <= 0 ||
        (p.c.format != 2 && p.global != 1) ||
        p.scales.size() != (p.c.mode ? 1 : p.rows * bpr) ||
        p.data.size() != (p.c.format == 2 ? (n + 1) / 2 : n))
        throw std::runtime_error("inconsistent packed metadata");
    for (auto s : p.scales)
        if (p.c.format == 2 ? (s == 0 || s > 126) : (s == 255))
            throw std::runtime_error("invalid scale");
    if (p.c.format == 2 && (n & 1) && (p.data.back() & 240))
        throw std::runtime_error("nonzero padding nibble");
}
Tensor dequant_cpu(const Packed &p) {
    validate_packed(p);
    Tensor t{p.rows, p.cols, p.c.out, {}};
    t.data.resize(p.size() * (t.dtype ? 2 : 4));
    uint64_t bpr = (p.cols + p.c.block - 1) / p.c.block;
    for (uint64_t i = 0; i < p.size(); ++i) {
        uint8_t q = p.c.format == 2 ? (p.data[i / 2] >> ((i & 1) * 4)) & 15 : p.data[i];
        uint64_t blk = p.c.mode ? 0 : (i / p.cols) * bpr + (i % p.cols) / p.c.block;
        double x = decode(q, p.c.format) *
                   scale_value(p.scales[blk], p.c.format, p.global);
        store_value(t.data.data(), i, t.dtype, output_bits(x, t.dtype));
    }
    return t;
}
void equal(const Packed &a, const Packed &b) {
    if (a.data != b.data || a.scales != b.scales || bits(a.global) != bits(b.global))
        throw std::runtime_error("CPU/GPU packed/scale mismatch");
}
void equal(const Tensor &a, const Tensor &b) {
    if (a.dtype != b.dtype || a.rows != b.rows || a.cols != b.cols || a.data != b.data)
        throw std::runtime_error("CPU/GPU output bit mismatch");
}
static double value(const Tensor &t, uint64_t i) {
    if (t.dtype == 2) {
        uint16_t h;
        __builtin_memcpy(&h, t.data.data() + i * 2, 2);
        return frombits(uint32_t(h) << 16);
    }
    return input_value(t.data.data(), i, t.dtype);
}
Error error(const Tensor &a, const Tensor &b) {
    if (a.size() != b.size())
        throw std::runtime_error("error shape mismatch");
    Error e;
    long double sum = 0, sq = 0;
    for (uint64_t i = 0; i < a.size(); ++i) {
        long double d = fabsl(value(a, i) - value(b, i));
        e.max = std::max(e.max, double(d));
        sum += d;
        sq += d * d;
    }
    e.mae = double(sum / a.size());
    e.mse = double(sq / a.size());
    return e;
}
} // namespace lp
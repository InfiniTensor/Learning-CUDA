// io.cpp - 配置 / 张量 / 量化输出 的读写实现
#include "io.h"
#include "fp8.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>

namespace hd {

// ======================= 名称 =======================
const char* dtype_name(Dtype d) {
    switch (d) {
    case Dtype::FP32:
        return "fp32";
    case Dtype::FP16:
        return "fp16";
    default:
        return "bf16";
    }
}
const char* quant_name(QuantFormat q) {
    switch (q) {
    case QuantFormat::FP8_E4M3:
        return "fp8_e4m3";
    case QuantFormat::FP8_E5M2:
        return "fp8_e5m2";
    default:
        return "none";
    }
}

// ======================= 半精度 / bf16 软件转换 =======================
static uint16_t f32_to_f16_bits(float f) {
    uint32_t u = f32_to_bits(f);
    uint32_t sign = (u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((u >> 23) & 0xFFu) - 127;
    uint32_t mant = u & 0x7FFFFFu;
    if (exp == 128)
        return (uint16_t)(mant ? (sign | 0x7E00u) : (sign | 0x7C00u));
    int32_t he = exp + 15;
    if (he >= 31)
        return (uint16_t)(sign | 0x7C00u);
    if (he <= 0) {
        if (he < -10)
            return (uint16_t)sign;
        mant |= 0x800000u;
        int shift = 14 - he;
        uint32_t hm = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1u);
        uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (hm & 1u)))
            hm++;
        return (uint16_t)(sign | hm);
    }
    uint32_t hm = mant >> 13;
    uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u))) {
        hm++;
        if (hm == 0x400u) {
            hm = 0;
            he++;
            if (he >= 31)
                return (uint16_t)(sign | 0x7C00u);
        }
    }
    return (uint16_t)(sign | ((uint32_t)he << 10) | hm);
}
static float f16_bits_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            while (!(mant & 0x400u)) {
                mant <<= 1;
                exp--;
            }
            exp += 1;
            mant &= 0x3FFu;
            bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    return bits_to_f32(bits);
}
static uint16_t f32_to_bf16_bits(float f) {
    uint32_t u = f32_to_bits(f);
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb; // 最近偶数舍入
    return (uint16_t)(u >> 16);
}
static float bf16_bits_to_f32(uint16_t b) {
    return bits_to_f32((uint32_t)b << 16);
}

// ======================= 字符串工具 =======================
static std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a]))
        a++;
    while (b > a && std::isspace((unsigned char)s[b - 1]))
        b--;
    return s.substr(a, b - a);
}
static std::string strip_quotes(std::string s) {
    if (s.size() >= 2 &&
        ((s.front() == '"' && s.back() == '"') || (s.front() == '\'' && s.back() == '\'')))
        return s.substr(1, s.size() - 2);
    return s;
}
static std::string lower(std::string s) {
    for (char& c : s)
        c = (char)std::tolower((unsigned char)c);
    return s;
}

// ======================= 配置 =======================
Config load_config(const std::string& path) {
    Config cfg;
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "[io] 无法打开配置文件: %s\n", path.c_str());
        return cfg;
    }
    std::string line;
    while (std::getline(f, line)) {
        size_t hash = line.find('#');
        if (hash != std::string::npos)
            line = line.substr(0, hash);
        line = trim(line);
        if (line.empty())
            continue;
        size_t eq = line.find('=');
        if (eq == std::string::npos)
            continue;
        std::string key = lower(trim(line.substr(0, eq)));
        std::string val = strip_quotes(trim(line.substr(eq + 1)));
        if (key == "hadamard_size")
            cfg.hadamard_size = std::atoi(val.c_str());
        else if (key == "scale")
            cfg.scale = (float)std::atof(val.c_str());
        else if (key == "quantize_enable")
            cfg.quantize_enable = (lower(val) == "true" || val == "1");
        else if (key == "quant_format")
            cfg.quant_format =
                (lower(val) == "fp8_e5m2") ? QuantFormat::FP8_E5M2 : QuantFormat::FP8_E4M3;
        else if (key == "block_size")
            cfg.block_size = std::atoi(val.c_str());
        else if (key == "threads")
            cfg.threads = std::atoi(val.c_str());
    }
    if (cfg.block_size <= 0)
        cfg.block_size = 32;
    if (cfg.threads <= 0)
        cfg.threads = 256;
    return cfg;
}

// ======================= 张量 =======================
Tensor load_tensor(const std::string& path) {
    Tensor t;
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[io] 无法打开数据文件: %s\n", path.c_str());
        return t;
    }
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    size_t dpos = content.find("[data]");
    if (dpos == std::string::npos) {
        std::fprintf(stderr, "[io] 数据文件缺少 [data] 段\n");
        return t;
    }
    std::istringstream hs(content.substr(0, dpos));
    std::string line;
    while (std::getline(hs, line)) {
        line = trim(line);
        size_t colon = line.find(':');
        if (colon == std::string::npos)
            continue;
        std::string key = lower(trim(line.substr(0, colon)));
        std::string val = trim(line.substr(colon + 1));
        if (key == "rows")
            t.rows = (int32_t)std::atoi(val.c_str());
        else if (key == "cols")
            t.cols = (int32_t)std::atoi(val.c_str());
        else if (key == "dtype") {
            std::string v = lower(val);
            t.dtype = (v == "fp16") ? Dtype::FP16 : (v == "bf16" ? Dtype::BF16 : Dtype::FP32);
        }
    }
    size_t nl = content.find('\n', dpos);
    if (nl == std::string::npos) {
        std::fprintf(stderr, "[io] [data] 段后无数据\n");
        return t;
    }
    size_t off = nl + 1;
    const char* p = content.data() + off;
    size_t avail = content.size() - off;
    int64_t n = (int64_t)t.rows * t.cols;
    t.data.resize((size_t)std::max<int64_t>(n, 0));
    if (t.dtype == Dtype::FP32) {
        if ((int64_t)avail < n * 4) {
            std::fprintf(stderr, "[io] fp32 数据不足\n");
            t.data.clear();
            return t;
        }
        std::memcpy(t.data.data(), p, (size_t)(n * 4));
    } else {
        if ((int64_t)avail < n * 2) {
            std::fprintf(stderr, "[io] fp16/bf16 数据不足\n");
            t.data.clear();
            return t;
        }
        const uint16_t* hp = reinterpret_cast<const uint16_t*>(p);
        for (int64_t i = 0; i < n; i++)
            t.data[(size_t)i] =
                (t.dtype == Dtype::FP16) ? f16_bits_to_f32(hp[i]) : bf16_bits_to_f32(hp[i]);
    }
    return t;
}

void save_tensor(const std::string& path, const Tensor& t, Dtype out_dtype) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[io] 无法写出数据文件: %s\n", path.c_str());
        return;
    }
    f << "[header]\n";
    f << "rows: " << t.rows << "\n";
    f << "cols: " << t.cols << "\n";
    f << "layout: row_major\n";
    f << "dtype: " << dtype_name(out_dtype) << "\n";
    f << "apply_dim: last\n\n";
    f << "[data]\n";
    int64_t n = t.numel();
    if (out_dtype == Dtype::FP32) {
        f.write(reinterpret_cast<const char*>(t.data.data()), (std::streamsize)(n * 4));
    } else {
        std::vector<uint16_t> tmp((size_t)n);
        for (int64_t i = 0; i < n; i++)
            tmp[(size_t)i] = (out_dtype == Dtype::FP16) ? f32_to_f16_bits(t.data[(size_t)i])
                                                        : f32_to_bf16_bits(t.data[(size_t)i]);
        f.write(reinterpret_cast<const char*>(tmp.data()), (std::streamsize)(n * 2));
    }
}

// ======================= 合成数据 =======================
Tensor gen_tensor(int64_t rows, int64_t cols, const std::string& kind, uint32_t seed, Dtype dtype) {
    Tensor t;
    t.rows = (int32_t)rows;
    t.cols = (int32_t)cols;
    t.dtype = dtype;
    int64_t n = rows * cols;
    t.data.resize((size_t)std::max<int64_t>(n, 0));
    std::mt19937 rng(seed);
    std::string k = lower(kind);
    if (k == "normal") {
        std::normal_distribution<float> d(0.0f, 1.0f);
        for (int64_t i = 0; i < n; i++)
            t.data[(size_t)i] = d(rng);
    } else if (k == "outlier") {
        // 含离群值：模拟激活中的异常值，用于观察 Hadamard 旋转的抑制效果
        std::normal_distribution<float> d(0.0f, 1.0f);
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        std::uniform_real_distribution<float> sg(-1.0f, 1.0f);
        for (int64_t i = 0; i < n; i++) {
            float v = d(rng);
            if (u(rng) < 0.001f)
                v = sg(rng) * (50.0f + 450.0f * u(rng));
            t.data[(size_t)i] = v;
        }
    } else {
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        for (int64_t i = 0; i < n; i++)
            t.data[(size_t)i] = d(rng);
    }
    return t;
}

// ======================= 量化输出 =======================
void save_quant_file(const std::string& path, const QuantHeader& h, const QuantResult& qr) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[io] 无法写出量化文件: %s\n", path.c_str());
        return;
    }
    f.write(reinterpret_cast<const char*>(&h), sizeof(QuantHeader));
    if (!qr.qdata.empty())
        f.write(reinterpret_cast<const char*>(qr.qdata.data()), (std::streamsize)qr.qdata.size());
    if (!qr.scales.empty())
        f.write(reinterpret_cast<const char*>(qr.scales.data()), (std::streamsize)qr.scales.size());
}

bool load_quant_file(const std::string& path, QuantHeader& h, QuantResult& qr) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[io] 无法打开量化文件: %s\n", path.c_str());
        return false;
    }
    f.read(reinterpret_cast<char*>(&h), sizeof(QuantHeader));
    if (std::memcmp(h.magic, "HQ01", 4) != 0) {
        std::fprintf(stderr, "[io] 量化文件 magic 不匹配\n");
        return false;
    }
    qr.qdata.resize((size_t)h.data_bytes);
    qr.scales.resize((size_t)h.scale_bytes);
    qr.num_blocks = (int64_t)h.scale_bytes;
    if (h.data_bytes > 0)
        f.read(reinterpret_cast<char*>(qr.qdata.data()), (std::streamsize)h.data_bytes);
    if (h.scale_bytes > 0)
        f.read(reinterpret_cast<char*>(qr.scales.data()), (std::streamsize)h.scale_bytes);
    return true;
}

// ======================= 误差 =======================
ErrMetrics compare_error(const std::vector<float>& ref, const std::vector<float>& got) {
    ErrMetrics m;
    int64_t n = (int64_t)std::min(ref.size(), got.size());
    if (n == 0)
        return m;
    double sum_abs = 0, sum_sq = 0, mx = 0, ref_sq = 0;
    for (int64_t i = 0; i < n; i++) {
        double r = (double)ref[(size_t)i];
        double d = std::fabs(r - (double)got[(size_t)i]);
        sum_abs += d;
        sum_sq += d * d;
        ref_sq += r * r;
        mx = std::max(mx, d);
    }
    m.max_abs = mx;
    m.mae = sum_abs / (double)n;
    m.mse = sum_sq / (double)n;
    m.rel_l2 = (ref_sq > 0) ? std::sqrt(sum_sq / ref_sq) : 0.0;
    return m;
}

} // namespace hd

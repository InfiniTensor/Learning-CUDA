// io.cpp - 配置/张量读写、压缩权重文件与反量化输出的实现
#include "io.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>

namespace lowp {

// ======================= 主机端半精度/bf16 软件转换 =======================
static uint16_t f32_to_f16_bits(float f) {
    uint32_t u = f32_to_bits(f);
    uint32_t sign = (u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((u >> 23) & 0xFFu) - 127;
    uint32_t mant = u & 0x7FFFFFu;
    if (exp == 128)
        return mant ? (sign | 0x7E00u) : (sign | 0x7C00u);
    int32_t he = exp + 15;
    if (he >= 31)
        return sign | 0x7C00u;
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
    if (exp == 0) {
        if (mant == 0)
            return bits_to_f32(sign);
        while (!(mant & 0x400u)) {
            mant <<= 1;
            exp--;
        }
        exp += 1;
        mant &= 0x3FFu;
        return bits_to_f32(sign | ((exp + 127u - 15u) << 23) | (mant << 13));
    }
    if (exp == 31)
        return bits_to_f32(sign | 0x7F800000u | (mant << 13));
    return bits_to_f32(sign | ((exp + 112u) << 23) | (mant << 13));
}
static uint16_t f32_to_bf16_bits(float f) {
    uint32_t u = f32_to_bits(f);
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb; // round-to-nearest-even
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

// ======================= 配置解析 =======================
Config load_config(const std::string& path) {
    Config cfg;
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "[io] 无法打开配置文件: %s\n", path.c_str());
        return cfg;
    }

    std::string line;
    bool have_bs = false;
    while (std::getline(f, line)) {
        // 去掉行尾注释
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

        if (key == "format")
            cfg.format = (lower(val) == "nvfp4") ? LowFormat::NVFP4 : LowFormat::MXFP8;
        else if (key == "elem_format")
            cfg.elem = (lower(val) == "e5m2") ? ElemFmt::E5M2 : ElemFmt::E4M3;
        else if (key == "block_size") {
            cfg.block_size = std::atoi(val.c_str());
            have_bs = true;
        } else if (key == "scale_mode")
            cfg.scale_mode = (lower(val) == "tensor") ? ScaleMode::TENSOR : ScaleMode::BLOCK;
        else if (key == "output_type") {
            std::string v = lower(val);
            cfg.out_dtype =
                (v == "bf16") ? OutDtype::BF16 : (v == "fp32" ? OutDtype::FP32 : OutDtype::FP16);
        } else if (key == "rounding")
            cfg.round = (lower(val) == "stochastic") ? RoundMode::STOCHASTIC : RoundMode::NEAREST;
        else if (key == "seed")
            cfg.seed = (uint32_t)std::strtoul(val.c_str(), nullptr, 10);
        else if (key == "target_gpu")
            cfg.target_gpu = val;
        else if (key == "no_cpu_ref")
            cfg.no_cpu_ref = (lower(val) == "true" || val == "1");
    }
    // 各格式默认 block 粒度
    if (!have_bs)
        cfg.block_size = (cfg.format == LowFormat::MXFP8) ? 32 : 16;
    return cfg;
}

// ======================= 张量读写 =======================
Tensor load_tensor(const std::string& path) {
    Tensor t;
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[io] 无法打开张量文件: %s\n", path.c_str());
        return t;
    }
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    size_t dpos = content.find("[data]");
    if (dpos == std::string::npos) {
        std::fprintf(stderr, "[io] 张量文件缺少 [data] 段: %s\n", path.c_str());
        return t;
    }
    // 解析头部文本
    std::istringstream hs(content.substr(0, dpos));
    std::string line;
    while (std::getline(hs, line)) {
        line = trim(line);
        size_t colon = line.find(':');
        if (colon == std::string::npos)
            continue;
        std::string key = lower(trim(line.substr(0, colon)));
        std::string val = trim(line.substr(colon + 1));
        if (key == "num_rows")
            t.rows = std::strtoll(val.c_str(), nullptr, 10);
        else if (key == "num_cols")
            t.cols = std::strtoll(val.c_str(), nullptr, 10);
        else if (key == "dtype")
            t.src_fp16 = (lower(val) == "fp16");
    }
    // 数据段从 [data] 所在行结束后开始
    size_t nl = content.find('\n', dpos);
    if (nl == std::string::npos) {
        std::fprintf(stderr, "[io] [data] 段后无数据\n");
        return t;
    }
    size_t off = nl + 1;
    int64_t n = t.rows * t.cols;
    t.data.resize((size_t)n);
    const char* p = content.data() + off;
    size_t avail = content.size() - off;
    if (t.src_fp16) {
        int64_t need = n * 2;
        if ((int64_t)avail < need) {
            std::fprintf(stderr, "[io] fp16 数据不足: 需要 %lld 字节, 实际 %zu\n", (long long)need,
                         avail);
            t.data.clear();
            return t;
        }
        const uint16_t* hp = reinterpret_cast<const uint16_t*>(p);
        for (int64_t i = 0; i < n; i++)
            t.data[(size_t)i] = f16_bits_to_f32(hp[i]);
    } else {
        int64_t need = n * 4;
        if ((int64_t)avail < need) {
            std::fprintf(stderr, "[io] fp32 数据不足: 需要 %lld 字节, 实际 %zu\n", (long long)need,
                         avail);
            t.data.clear();
            return t;
        }
        std::memcpy(t.data.data(), p, (size_t)need);
    }
    return t;
}

void save_tensor(const std::string& path, const Tensor& t) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[io] 无法写出张量: %s\n", path.c_str());
        return;
    }
    f << "[header]\n";
    f << "num_rows: " << t.rows << "\n";
    f << "num_cols: " << t.cols << "\n";
    f << "dtype: " << (t.src_fp16 ? "fp16" : "fp32") << "\n\n";
    f << "[data]\n";
    int64_t n = t.numel();
    if (t.src_fp16) {
        std::vector<uint16_t> tmp((size_t)n);
        for (int64_t i = 0; i < n; i++)
            tmp[(size_t)i] = f32_to_f16_bits(t.data[(size_t)i]);
        f.write(reinterpret_cast<const char*>(tmp.data()), (std::streamsize)(n * 2));
    } else {
        f.write(reinterpret_cast<const char*>(t.data.data()), (std::streamsize)(n * 4));
    }
}

Tensor gen_tensor(const std::string& kind, int64_t rows, int64_t cols, uint32_t seed,
                  bool as_fp16) {
    Tensor t;
    t.rows = rows;
    t.cols = cols;
    t.src_fp16 = as_fp16;
    int64_t n = rows * cols;
    t.data.resize((size_t)n);
    std::mt19937 rng(seed);
    std::string k = lower(kind);
    if (k == "normal") {
        std::normal_distribution<float> d(0.0f, 1.0f);
        for (int64_t i = 0; i < n; i++)
            t.data[(size_t)i] = d(rng);
    } else if (k == "outlier") {
        // 正态分布 + 少量大离群值(模拟激活异常值)
        std::normal_distribution<float> d(0.0f, 1.0f);
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        std::uniform_real_distribution<float> sgn(-1.0f, 1.0f);
        for (int64_t i = 0; i < n; i++) {
            float v = d(rng);
            if (u(rng) < 0.001f)
                v = sgn(rng) * (50.0f + 450.0f * u(rng)); // 0.1% 异常值
            t.data[(size_t)i] = v;
        }
    } else if (k == "zeros") {
        // 全零张量：用于检验缩放因子为 0 的边界路径
        std::fill(t.data.begin(), t.data.end(), 0.0f);
    } else {
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        for (int64_t i = 0; i < n; i++)
            t.data[(size_t)i] = d(rng);
    }
    return t;
}

// ======================= 压缩权重文件 =======================
void save_quant_file(const std::string& path, const QuantHeader& h, const uint8_t* packed,
                     const uint8_t* scales) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[io] 无法写出权重文件: %s\n", path.c_str());
        return;
    }
    f.write(reinterpret_cast<const char*>(&h), sizeof(QuantHeader));
    if (h.packed_bytes > 0)
        f.write(reinterpret_cast<const char*>(packed), (std::streamsize)h.packed_bytes);
    if (h.scale_bytes > 0)
        f.write(reinterpret_cast<const char*>(scales), (std::streamsize)h.scale_bytes);
}

bool load_quant_file(const std::string& path, QuantHeader& h, std::vector<uint8_t>& packed,
                     std::vector<uint8_t>& scales) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[io] 无法打开权重文件: %s\n", path.c_str());
        return false;
    }
    f.read(reinterpret_cast<char*>(&h), sizeof(QuantHeader));
    if (std::memcmp(h.magic, "LPQ1", 4) != 0) {
        std::fprintf(stderr, "[io] 权重文件 magic 不匹配\n");
        return false;
    }
    packed.resize((size_t)h.packed_bytes);
    scales.resize((size_t)h.scale_bytes);
    if (h.packed_bytes > 0)
        f.read(reinterpret_cast<char*>(packed.data()), (std::streamsize)h.packed_bytes);
    if (h.scale_bytes > 0)
        f.read(reinterpret_cast<char*>(scales.data()), (std::streamsize)h.scale_bytes);
    return true;
}

// ======================= 反量化张量输出 =======================
void save_dequant_tensor(const std::string& path, const std::vector<float>& data, OutDtype dt) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[io] 无法写出反量化张量: %s\n", path.c_str());
        return;
    }
    if (dt == OutDtype::FP32) {
        f.write(reinterpret_cast<const char*>(data.data()),
                (std::streamsize)(data.size() * sizeof(float)));
    } else if (dt == OutDtype::FP16) {
        std::vector<uint16_t> tmp(data.size());
        for (size_t i = 0; i < data.size(); i++)
            tmp[i] = f32_to_f16_bits(data[i]);
        f.write(reinterpret_cast<const char*>(tmp.data()), (std::streamsize)(tmp.size() * 2));
    } else {
        std::vector<uint16_t> tmp(data.size());
        for (size_t i = 0; i < data.size(); i++)
            tmp[i] = f32_to_bf16_bits(data[i]);
        f.write(reinterpret_cast<const char*>(tmp.data()), (std::streamsize)(tmp.size() * 2));
    }
}

std::vector<float> load_dequant_tensor(const std::string& path, OutDtype dt, int64_t numel) {
    std::vector<float> out((size_t)numel);
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[io] 无法打开反量化张量: %s\n", path.c_str());
        return out;
    }
    if (dt == OutDtype::FP32) {
        f.read(reinterpret_cast<char*>(out.data()), (std::streamsize)(numel * 4));
    } else if (dt == OutDtype::FP16) {
        std::vector<uint16_t> tmp((size_t)numel);
        f.read(reinterpret_cast<char*>(tmp.data()), (std::streamsize)(numel * 2));
        for (int64_t i = 0; i < numel; i++)
            out[(size_t)i] = f16_bits_to_f32(tmp[(size_t)i]);
    } else {
        std::vector<uint16_t> tmp((size_t)numel);
        f.read(reinterpret_cast<char*>(tmp.data()), (std::streamsize)(numel * 2));
        for (int64_t i = 0; i < numel; i++)
            out[(size_t)i] = bf16_bits_to_f32(tmp[(size_t)i]);
    }
    return out;
}

float round_to_out_dtype(float v, OutDtype dt) {
    if (dt == OutDtype::FP32)
        return v;
    if (dt == OutDtype::FP16)
        return f16_bits_to_f32(f32_to_f16_bits(v));
    return bf16_bits_to_f32(f32_to_bf16_bits(v));
}

// ======================= 名称 =======================
const char* out_dtype_name(OutDtype d) {
    switch (d) {
    case OutDtype::FP16:
        return "fp16";
    case OutDtype::BF16:
        return "bf16";
    default:
        return "fp32";
    }
}
const char* scale_mode_name(ScaleMode s) {
    return s == ScaleMode::TENSOR ? "tensor" : "block";
}
const char* round_mode_name(RoundMode r) {
    return r == RoundMode::NEAREST ? "nearest" : "stochastic";
}

} // namespace lowp

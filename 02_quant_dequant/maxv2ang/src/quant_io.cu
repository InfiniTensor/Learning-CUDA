// ============================================================================
// 选题二「MXFP8 / NVFP4 低精度模拟与反量化」—— 主机侧实现
// ============================================================================
// 纯主机代码：量化参数解析、张量/低精度权重文件 I/O、主机端参考实现（对拍基准）、
// 误差与压缩率指标。不含任何 __global__ kernel，也不进设备编译。
// 公开接口见 include/quant_io.h；格式语义见 include/quant_format.h。
// ============================================================================

#include "quant_io.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <sys/stat.h>
#include <sys/types.h> // mkdir

// ============================================================================
// 3. 4bit 打包 / 解包（元素对打包：偶数下标占低 4bit，奇数下标占高 4bit）
// ============================================================================

std::vector<uint8_t> pack_nibbles(std::vector<uint8_t> codes) {
    for (auto &c : codes)
        c &= 0x0F;
    if (codes.size() % 2)
        codes.push_back(0);
    std::vector<uint8_t> out(codes.size() / 2);
    for (size_t i = 0; i < out.size(); i++)
        out[i] = (uint8_t)(codes[2 * i] | (codes[2 * i + 1] << 4));
    return out;
}
std::vector<uint8_t> unpack_nibbles(const std::vector<uint8_t> &packed, size_t n) {
    std::vector<uint8_t> out(n);
    for (size_t i = 0; i < n; i++) {
        uint8_t b = packed[i / 2];
        out[i] = (i % 2 == 0) ? (uint8_t)(b & 0x0F) : (uint8_t)(b >> 4);
    }
    return out;
}

// ============================================================================
// 4. 量化参数（KEY = VALUE 文本）
// ============================================================================

static std::string qc_trim(const std::string &s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos)
        return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}
static std::string qc_strip_quotes(const std::string &s) {
    if (s.size() >= 2 && (s.front() == '"' || s.front() == '\'') && s.back() == s.front())
        return s.substr(1, s.size() - 2);
    return s;
}

Config read_config(const std::string &path) {
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("无法打开配置文件: " + path);
    Config cfg;
    std::string line;
    while (std::getline(f, line)) {
        line = qc_trim(line);
        if (line.empty() || line[0] == '#')
            continue;
        size_t eq = line.find('=');
        if (eq == std::string::npos)
            continue;
        std::string key = qc_trim(line.substr(0, eq));
        std::string val = qc_strip_quotes(qc_trim(line.substr(eq + 1)));
        if (key == "format")
            cfg.format = val;
        else if (key == "block_size")
            cfg.block_size = std::stoi(val);
        else if (key == "scale_mode")
            cfg.scale_mode = val;
        else if (key == "output_type")
            cfg.output_type = val;
        else if (key == "rounding")
            cfg.rounding = val;
        else if (key == "target_gpu")
            cfg.target_gpu = val;
        else if (key == "elem_format")
            cfg.elem_format = val;
        else if (key == "seed")
            cfg.seed = std::stoi(val);
    }
    if (cfg.format != "mxfp8" && cfg.format != "nvfp4")
        throw std::runtime_error("format 必须是 mxfp8/nvfp4");
    if (cfg.scale_mode != "tensor" && cfg.scale_mode != "block")
        throw std::runtime_error("scale_mode 必须是 tensor/block");
    if (cfg.output_type != "fp16" && cfg.output_type != "bf16" && cfg.output_type != "fp32")
        throw std::runtime_error("output_type 必须是 fp16/bf16/fp32");
    if (cfg.rounding != "nearest" && cfg.rounding != "stochastic")
        throw std::runtime_error("rounding 必须是 nearest/stochastic");
    if (cfg.elem_format != "e4m3" && cfg.elem_format != "e5m2")
        throw std::runtime_error("elem_format 必须是 e4m3/e5m2");
    return cfg;
}

// ============================================================================
// 5. 张量文件 I/O（格式与设备端一致：
//    int64 rows | int64 cols | int32 len | ASCII dtype | 行主序数据）
// ============================================================================

// ---- 小端序列化（显式按字节，不依赖主机字节序）----

static uint16_t qc_rd_u16(const uint8_t *p) {
    return (uint16_t)(p[0] | (p[1] << 8));
}
static int64_t qc_rd_i64(const uint8_t *p) {
    uint64_t lo =
        (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    uint64_t hi =
        (uint32_t)p[4] | ((uint32_t)p[5] << 8) | ((uint32_t)p[6] << 16) | ((uint32_t)p[7] << 24);
    return (int64_t)(lo | (hi << 32));
}
static float qc_rd_f32(const uint8_t *p) {
    uint32_t u =
        (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static void qc_wr_u16(std::vector<uint8_t> &b, uint16_t v) {
    b.push_back((uint8_t)v);
    b.push_back((uint8_t)(v >> 8));
}
static void qc_wr_i32(std::vector<uint8_t> &b, int32_t v) {
    uint32_t u = (uint32_t)v;
    for (int i = 0; i < 4; i++)
        b.push_back((uint8_t)(u >> (8 * i)));
}
static void qc_wr_i64(std::vector<uint8_t> &b, int64_t v) {
    uint64_t u = (uint64_t)v;
    for (int i = 0; i < 8; i++)
        b.push_back((uint8_t)(u >> (8 * i)));
}
static void qc_wr_f32(std::vector<uint8_t> &b, float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    for (int i = 0; i < 4; i++)
        b.push_back((uint8_t)(u >> (8 * i)));
}
static std::vector<uint8_t> qc_read_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
        throw std::runtime_error("无法打开文件: " + path);
    std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf((size_t)n);
    if (n > 0)
        f.read((char *)buf.data(), n);
    return buf;
}
static void qc_write_file(const std::string &path, const std::vector<uint8_t> &buf) {
    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("无法写文件: " + path);
    f.write((const char *)buf.data(), (std::streamsize)buf.size());
}
// 递归创建目录（等价 mkdir -p）。
// 单层 ::mkdir 在父目录不存在时会失败，且失败极易被忽略 —— 要等到后续写文件
// 才以「无法写文件」的形式暴露，很难定位到是目录没建出来。这里逐层创建。
void qc_mkdir(const std::string &path) {
    if (path.empty())
        return;
    for (size_t i = 1; i <= path.size(); i++) {
        if (i < path.size() && path[i] != '/')
            continue;
        std::string sub = path.substr(0, i);
        if (sub.empty() || sub == "/")
            continue;
        ::mkdir(sub.c_str(), 0755); // 已存在则忽略（EEXIST）
    }
}

// 张量文件（输入）：i64 rows | i64 cols | char[4] dtype | 行主序数据（小端）。
// dtype 固定 4 字节（"fp32"/"fp16"/"bf16"），无长度前缀。
TensorFile read_tensor(const std::string &path) {
    std::vector<uint8_t> buf = qc_read_file(path);
    size_t off = 0;
    auto need = [&](size_t k) {
        if (off + k > buf.size())
            throw std::runtime_error("张量文件损坏: " + path);
    };
    need(20);
    TensorFile t;
    t.rows = qc_rd_i64(&buf[0]);
    t.cols = qc_rd_i64(&buf[8]);
    off = 16;
    t.dtype.assign((const char *)&buf[off], 4);
    off += 4;
    int64_t n = t.rows * t.cols;
    if (t.dtype == "fp32") {
        need(4 * (size_t)n);
        t.data.resize((size_t)n);
        for (int64_t i = 0; i < n; i++)
            t.data[(size_t)i] = qc_rd_f32(&buf[off + 4 * i]);
    } else if (t.dtype == "fp16") {
        need(2 * (size_t)n);
        t.data.resize((size_t)n);
        for (int64_t i = 0; i < n; i++)
            t.data[(size_t)i] = f16_bits_to_f32(qc_rd_u16(&buf[off + 2 * i]));
    } else if (t.dtype == "bf16") {
        need(2 * (size_t)n);
        t.data.resize((size_t)n);
        for (int64_t i = 0; i < n; i++)
            t.data[(size_t)i] = bf16_bits_to_f32(qc_rd_u16(&buf[off + 2 * i]));
    } else {
        throw std::runtime_error("未知 dtype: " + t.dtype);
    }
    return t;
}

// 张量文件（输入与反量化输出同格式）：
//   i64 rows | i64 cols | char[4] dtype | 行主序数据（小端）
void write_tensor(const std::string &path, int64_t rows, int64_t cols,
                         const std::string &dtype, const std::vector<float> &data) {
    if (dtype.size() != 4)
        throw std::runtime_error("dtype 必须是 4 字符: " + dtype);
    std::vector<uint8_t> buf;
    qc_wr_i64(buf, rows);
    qc_wr_i64(buf, cols);
    buf.insert(buf.end(), dtype.begin(), dtype.end());
    if (dtype == "fp32") {
        for (float v : data)
            qc_wr_f32(buf, v);
    } else if (dtype == "fp16") {
        for (float v : data)
            qc_wr_u16(buf, f64_to_f16_bits((double)v));
    } else if (dtype == "bf16") {
        for (float v : data)
            qc_wr_u16(buf, f32_to_bf16_bits(v));
    } else {
        throw std::runtime_error("未知 dtype: " + dtype);
    }
    qc_write_file(path, buf);
}

// ============================================================================
// 6. 低精度权重文件（LPW1）
//    magic | u8 fmt | u8 elem | u8 smode | u8 保留 | i64 rows | i64 cols
//    | i32 block_size | i64 num_blocks | f32 global_scale | scales | packed
// ============================================================================

static const char QC_MAGIC[5] = "LPW1";

void write_quantized(const std::string &path, const Quantized &q, int64_t rows,
                            int64_t cols) {
    uint8_t fmt_id = q.format == "mxfp8" ? 0 : 1;
    uint8_t smode_id = q.scale_mode == "tensor" ? 0 : 1;
    std::vector<uint8_t> buf;
    buf.insert(buf.end(), QC_MAGIC, QC_MAGIC + 4);
    buf.push_back(fmt_id);
    buf.push_back((uint8_t)q.elem_id);
    buf.push_back(smode_id);
    buf.push_back(0); // 保留
    qc_wr_i64(buf, rows);
    qc_wr_i64(buf, cols);
    qc_wr_i32(buf, q.block_size);
    qc_wr_i64(buf, q.num_blocks);
    qc_wr_f32(buf, q.global_scale);
    buf.insert(buf.end(), q.scale_bytes.begin(), q.scale_bytes.end());
    buf.insert(buf.end(), q.packed.begin(), q.packed.end());
    qc_write_file(path, buf);
}

// ============================================================================
// 7. 量化 / 反量化 —— 主机端实现（与设备端共用同一份语义）
//    输入统一为 FP32 向量（与 read_tensor / GPU 上传的数据形态一致）
// ============================================================================

void qc_validate_finite(const std::vector<float> &w) {
    for (float x : w)
        if (!std::isfinite(x))
            throw std::runtime_error("输入含 NaN/Inf，请先清洗");
}

// ---- MXFP8 ----

Quantized quantize_mxfp8_host(const std::vector<float> &w32, int block_size, int elem_id,
                                    const std::string &scale_mode, int mode, uint64_t seed) {
    qc_validate_finite(w32);
    int p = elem_id == FMT_E4M3 ? 3 : 2; // MX 约定: scale = 2^(floor(log2 amax) - p)
    size_t n = w32.size();

    Quantized q;
    q.format = "mxfp8";
    q.elem_id = elem_id;
    q.scale_mode = scale_mode;
    q.num_elems = (int64_t)n;
    q.global_scale = 1.0f; // MXFP8 不用全局缩放

    if (scale_mode == "block") {
        size_t nblk = (n + block_size - 1) / block_size;
        size_t padded = nblk * block_size;
        q.block_size = block_size;
        q.num_blocks = (int64_t)nblk;
        q.scale_bytes.assign(nblk, 0);
        std::vector<uint8_t> codes(padded, 0);
        for (size_t b = 0; b < nblk; b++) {
            double amax = 0.0; // 块内 amax（补零元素贡献 0）
            for (size_t i = b * block_size; i < (b + 1) * block_size; i++) {
                double a = fabs((double)(i < n ? w32[i] : 0.0f));
                if (a > amax)
                    amax = a;
            }
            long long e = mx_scale_exponent(amax, p);
            q.scale_bytes[b] = (uint8_t)(e + 127);
            double scale = ldexp(1.0, (int)e);
            for (size_t i = b * block_size; i < (b + 1) * block_size; i++) {
                double v = (double)(i < n ? w32[i] : 0.0f) / scale;
                codes[i] = fmt_encode(elem_id, v, mode, elem_rng(seed, (int64_t)i));
            }
        }
        q.packed = std::move(codes); // MXFP8 无需打包：1 字节 = 1 码点
    } else if (scale_mode == "tensor") {
        double amax = 0.0;
        for (float x : w32) {
            double a = fabs((double)x);
            if (a > amax)
                amax = a;
        }
        long long e = mx_scale_exponent(amax, p);
        q.scale_bytes.assign(1, (uint8_t)(e + 127));
        double scale = ldexp(1.0, (int)e);
        q.num_blocks = 1;
        q.packed.resize(n);
        for (size_t i = 0; i < n; i++)
            q.packed[i] =
                fmt_encode(elem_id, (double)w32[i] / scale, mode, elem_rng(seed, (int64_t)i));
    } else {
        throw std::runtime_error("未知 scale_mode: " + scale_mode);
    }
    return q;
}

std::vector<double> dequantize_mxfp8_host(const Quantized &q) {
    size_t n = (size_t)q.num_elems;
    std::vector<double> out(n);
    if (q.scale_bytes.size() == 1) { // tensor 模式
        double scale = ldexp(1.0, (int)q.scale_bytes[0] - 127);
        for (size_t i = 0; i < n; i++)
            out[i] = fmt_decode(q.elem_id, q.packed[i]) * scale;
    } else { // block 模式
        int bs = q.block_size;
        for (size_t i = 0; i < n; i++)
            out[i] =
                fmt_decode(q.elem_id, q.packed[i]) * ldexp(1.0, (int)q.scale_bytes[i / bs] - 127);
    }
    return out;
}

// ---- NVFP4 ----

Quantized quantize_nvfp4_host(const std::vector<float> &w32, int block_size,
                                    const std::string &scale_mode, int mode, uint64_t seed) {
    qc_validate_finite(w32);
    size_t n = w32.size();
    double amax_t = 0.0;
    for (float x : w32) {
        double a = fabs((double)x);
        if (a > amax_t)
            amax_t = a;
    }

    Quantized q;
    q.format = "nvfp4";
    q.elem_id = FMT_E2M1;
    q.scale_mode = scale_mode;
    q.num_elems = (int64_t)n;

    if (scale_mode == "block") {
        // 全局缩放 FP32：= amax/(448×6)，让最大块的理想块缩放恰好 = 448
        q.global_scale = amax_t == 0.0 ? 1.0f : (float)(amax_t / (E4M3_MAX * E2M1_MAX));
        double gs = q.global_scale; // 用 FP32 舍入后的值参与后续运算
        size_t nblk = (n + block_size - 1) / block_size;
        size_t padded = nblk * block_size;
        q.block_size = block_size;
        q.num_blocks = (int64_t)nblk;
        q.scale_bytes.assign(nblk, 0);
        std::vector<uint8_t> codes(padded, 0);
        for (size_t b = 0; b < nblk; b++) {
            double amax_b = 0.0;
            for (size_t i = b * block_size; i < (b + 1) * block_size; i++) {
                double a = fabs((double)(i < n ? w32[i] : 0.0f));
                if (a > amax_b)
                    amax_b = a;
            }
            // 理想块缩放 = amax_block/(6×gs)；向上取整到 E4M3（保证不溢出 ±6）
            double ideal = amax_b / (E2M1_MAX * gs);
            int s_idx = round_up_idx(e4m3_mags(), e4m3_mags_k(), ideal);
            q.scale_bytes[b] = (uint8_t)s_idx; // 正数侧码点 = 下标
            double denom = e4m3_mags()[s_idx] * gs;
            if (!(denom > 0.0))
                denom = 1.0; // 零块防 0/0
            for (size_t i = b * block_size; i < (b + 1) * block_size; i++) {
                double v = (double)(i < n ? w32[i] : 0.0f) / denom;
                codes[i] = fmt_encode(FMT_E2M1, v, mode, elem_rng(seed, (int64_t)i));
            }
        }
        q.packed = pack_nibbles(std::move(codes)); // 每字节 2 个 4bit 码点
    } else if (scale_mode == "tensor") {
        q.global_scale = amax_t == 0.0 ? 1.0f : (float)(amax_t / E2M1_MAX);
        q.num_blocks = 1;
        q.scale_bytes.assign(1, 0x38); // E4M3 的 1.0
        double gs = q.global_scale;
        std::vector<uint8_t> codes(n);
        for (size_t i = 0; i < n; i++)
            codes[i] = fmt_encode(FMT_E2M1, (double)w32[i] / gs, mode, elem_rng(seed, (int64_t)i));
        q.packed = pack_nibbles(std::move(codes));
    } else {
        throw std::runtime_error("未知 scale_mode: " + scale_mode);
    }
    return q;
}

std::vector<double> dequantize_nvfp4_host(const Quantized &q) {
    size_t n = (size_t)q.num_elems;
    double gs = q.global_scale;
    std::vector<double> out(n);
    if (q.scale_bytes.size() == 1) { // tensor 模式（s_b 恒 1.0）
        double sb = fmt_decode(FMT_E4M3, q.scale_bytes[0]);
        std::vector<uint8_t> codes = unpack_nibbles(q.packed, n);
        for (size_t i = 0; i < n; i++)
            out[i] = fmt_decode(FMT_E2M1, codes[i]) * sb * gs;
    } else { // block 模式：解包到补齐后的完整块，再按块乘缩放
        int bs = q.block_size;
        size_t nb = (size_t)q.num_blocks;
        std::vector<uint8_t> codes = unpack_nibbles(q.packed, nb * bs);
        for (size_t i = 0; i < n; i++) {
            // 乘法顺序固定为 (v × s_b) × gs —— 设备端保持一致
            double t = fmt_decode(FMT_E2M1, codes[i]) * fmt_decode(FMT_E4M3, q.scale_bytes[i / bs]);
            out[i] = t * gs;
        }
    }
    return out;
}

Quantized quantize_host(const std::vector<float> &w, const Config &cfg) {
    int mode = cfg.rounding == "stochastic" ? MODE_STOCHASTIC : MODE_NEAREST;
    int elem_id = cfg.elem_format == "e5m2" ? FMT_E5M2 : FMT_E4M3;
    if (cfg.format == "mxfp8")
        return quantize_mxfp8_host(w, cfg.block_size, elem_id, cfg.scale_mode, mode,
                                  (uint64_t)cfg.seed);
    return quantize_nvfp4_host(w, cfg.block_size, cfg.scale_mode, mode, (uint64_t)cfg.seed);
}

std::vector<double> dequantize_host(const Quantized &q) {
    return q.format == "mxfp8" ? dequantize_mxfp8_host(q) : dequantize_nvfp4_host(q);
}

// ============================================================================
// 8. 误差 / 压缩率指标 / 输出类型转换
// ============================================================================

ErrStats error_stats(const std::vector<double> &orig, const std::vector<float> &deq) {
    ErrStats s;
    double sum_abs = 0, sum_sq = 0;
    for (size_t i = 0; i < orig.size(); i++) {
        double d = orig[i] - (double)deq[i];
        double ad = fabs(d);
        if (ad > s.max_abs)
            s.max_abs = ad;
        sum_abs += ad;
        sum_sq += d * d;
    }
    double n = (double)orig.size();
    s.mae = sum_abs / n;
    s.mse = sum_sq / n;
    s.rmse = sqrt(s.mse);
    return s;
}

CompStats compression_stats(int src_bits, int64_t num_elems, size_t packed_bytes,
                                   size_t scale_bytes, int global_scale_bits) {
    double total_bits = 8.0 * ((double)packed_bytes + (double)scale_bytes) + global_scale_bits;
    CompStats c;
    c.src_bits = src_bits;
    c.dst_bits = total_bits / (double)num_elems;
    c.ratio = src_bits / c.dst_bits;
    return c;
}

// 反量化结果(float64) → 输出类型（fp16/bf16 的舍入是预期行为）
std::vector<float> cast_output(const std::vector<double> &in, int out_type_id_) {
    std::vector<float> out(in.size());
    if (out_type_id_ == OUT_FP32) {
        for (size_t i = 0; i < in.size(); i++)
            out[i] = (float)in[i];
    } else if (out_type_id_ == OUT_FP16) {
        for (size_t i = 0; i < in.size(); i++)
            out[i] = f16_bits_to_f32(f64_to_f16_bits(in[i]));
    } else { // bf16: 先舍到 FP32 再 RNE 截断（两步）
        for (size_t i = 0; i < in.size(); i++)
            out[i] = bf16_bits_to_f32(f32_to_bf16_bits((float)in[i]));
    }
    return out;
}

int out_type_id_of(const Config &cfg) {
    return cfg.output_type == "fp32" ? OUT_FP32 : cfg.output_type == "bf16" ? OUT_BF16 : OUT_FP16;
}
size_t out_type_bytes(int oti) {
    return oti == OUT_FP32 ? 4 : 2;
}


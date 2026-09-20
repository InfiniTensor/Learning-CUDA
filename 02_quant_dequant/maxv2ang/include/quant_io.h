// ============================================================================
// 选题二「MXFP8 / NVFP4 低精度模拟与反量化」—— 主机侧 API
// ============================================================================
// 主机端（非设备）功能的公开接口；实现在 src/quant_io.cu。
//
// 内容：量化参数解析、张量文件与低精度权重文件（LPW1）读写、4bit 打包/解包、
//       主机端参考实现（对拍基准）、误差与压缩率指标、输出类型转换。
//
// 这里的代码不进设备编译：需要 __device__ 的格式语义全在 quant_format.h。
// ============================================================================

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "quant_format.h"

// ---- 4bit 打包 / 解包（元素对打包：偶数下标占低 4bit，奇数下标占高 4bit）----
std::vector<uint8_t> pack_nibbles(std::vector<uint8_t> codes);
std::vector<uint8_t> unpack_nibbles(const std::vector<uint8_t> &packed, size_t n);

// ---- 量化参数（KEY = VALUE 文本）----
struct Config {
    std::string format = "mxfp8";     // mxfp8 / nvfp4
    int block_size = 32;              // mxfp8 默认 32，nvfp4 默认 16
    std::string scale_mode = "block"; // tensor / block
    std::string output_type = "fp16"; // fp16 / bf16 / fp32
    std::string rounding = "nearest"; // nearest / stochastic
    std::string target_gpu = "T4";    // 仅用于报告说明
    std::string elem_format = "e4m3"; // mxfp8 的元素格式： e4m3 / e5m2
    int seed = 42;
};
Config read_config(const std::string &path);

// ---- 张量文件 I/O ----
struct TensorFile {
    int64_t rows = 0, cols = 0;
    std::string dtype;       // fp32 / fp16 / bf16
    std::vector<float> data; // 统一转成 FP32（fp16/bf16 解码后）
};
void qc_mkdir(const std::string &path);
TensorFile read_tensor(const std::string &path);
void write_tensor(const std::string &path, int64_t rows, int64_t cols,
                         const std::string &dtype, const std::vector<float> &data);

// ---- 低精度权重文件（LPW1）----
struct Quantized {
    std::string format; // "mxfp8" / "nvfp4"
    int elem_id = FMT_E4M3;
    std::string scale_mode;
    int block_size = 0;
    int64_t num_blocks = 0;
    int64_t num_elems = 0;
    float global_scale = 1.0f;
    std::vector<uint8_t> scale_bytes;
    std::vector<uint8_t> packed;
};
void write_quantized(const std::string &path, const Quantized &q, int64_t rows,
                            int64_t cols);

// ---- 主机端参考实现（对拍基准；与设备端共用 quant_format.h 的舍入语义）----
void qc_validate_finite(const std::vector<float> &w);
Quantized quantize_mxfp8_host(const std::vector<float> &w32, int block_size, int elem_id,
                                    const std::string &scale_mode, int mode, uint64_t seed);
std::vector<double> dequantize_mxfp8_host(const Quantized &q);
Quantized quantize_nvfp4_host(const std::vector<float> &w32, int block_size,
                                    const std::string &scale_mode, int mode, uint64_t seed);
std::vector<double> dequantize_nvfp4_host(const Quantized &q);
Quantized quantize_host(const std::vector<float> &w, const Config &cfg);
std::vector<double> dequantize_host(const Quantized &q);

// ---- 误差 / 压缩率指标 / 输出类型转换 ----
struct ErrStats {
    double max_abs = 0, mae = 0, mse = 0, rmse = 0;
};
ErrStats error_stats(const std::vector<double> &orig, const std::vector<float> &deq);
struct CompStats {
    double src_bits = 0, dst_bits = 0, ratio = 0;
};
CompStats compression_stats(int src_bits, int64_t num_elems, size_t packed_bytes,
                                   size_t scale_bytes, int global_scale_bits);
std::vector<float> cast_output(const std::vector<double> &in, int out_type_id_);
int out_type_id_of(const Config &cfg);
size_t out_type_bytes(int oti);

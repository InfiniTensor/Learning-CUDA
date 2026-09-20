// io.h - 数据文件 / 参数文件 / 输出文件 的读写
#pragma once
#include "types.h"
#include <string>
#include <vector>

namespace hd {

// ======================= 配置 =======================
// 参数文件（key = value）：
//   hadamard_size = 128
//   scale = 1.0
//   quantize_enable = true
//   quant_format = "fp8_e4m3"
//   block_size = 32
Config load_config(const std::string& path);

// ======================= 张量 =======================
// 数据文件（文本头 + 二进制数据段）：
//   [header]
//   rows: int32
//   cols: int32
//   layout: row_major
//   dtype: fp32 | fp16 | bf16
//   apply_dim: last
//
//   [data]
//   <rows*cols 个元素，按 dtype 紧密排列，行主序>
Tensor load_tensor(const std::string& path);
void save_tensor(const std::string& path, const Tensor& t, Dtype out_dtype);

// 生成合成数据：kind = uniform | normal | outlier
Tensor gen_tensor(int64_t rows, int64_t cols, const std::string& kind, uint32_t seed, Dtype dtype);

// ======================= 量化输出文件 =======================
// 布局：[QuantHeader][fp8 数据 rows*cols 字节][E8M0 缩放 bytes]
#pragma pack(push, 1)
struct QuantHeader {
    char magic[4]; // "HQ01"
    int32_t rows;
    int32_t cols;
    int32_t block_size;
    int32_t quant_format; // 0 = e4m3, 1 = e5m2
    float scale;          // 变换侧的乘性缩放
    int64_t data_bytes;
    int64_t scale_bytes;
};
#pragma pack(pop)

void save_quant_file(const std::string& path, const QuantHeader& h, const QuantResult& qr);
bool load_quant_file(const std::string& path, QuantHeader& h, QuantResult& qr);

// ======================= 误差统计 =======================
struct ErrMetrics {
    double max_abs = 0.0;
    double mae = 0.0;
    double mse = 0.0;
    double rel_l2 = 0.0; // ||a-b||_2 / ||a||_2
};
ErrMetrics compare_error(const std::vector<float>& ref, const std::vector<float>& got);

} // namespace hd

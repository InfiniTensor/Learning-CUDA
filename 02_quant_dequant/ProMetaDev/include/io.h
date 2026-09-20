// io.h - 张量/配置读写、压缩权重文件与反量化输出的 I/O 接口
#pragma once
#include "lowp.h"
#include <cstdint>
#include <string>
#include <vector>

namespace lowp {

// ======================= 量化配置 =======================
struct Config {
    LowFormat format = LowFormat::MXFP8;
    ElemFmt elem = ElemFmt::E4M3; // 仅 mxfp8 使用
    int block_size = 32;
    ScaleMode scale_mode = ScaleMode::BLOCK;
    OutDtype out_dtype = OutDtype::FP16;
    RoundMode round = RoundMode::NEAREST;
    uint32_t seed = 1234;
    std::string target_gpu = "T4";
    bool no_cpu_ref = false; // 跳过 CPU 对照
};

// ======================= 张量 =======================
struct Tensor {
    int64_t rows = 0, cols = 0;
    bool src_fp16 = false;   // 输入原始类型
    std::vector<float> data; // 统一以 fp32 保存
    int64_t numel() const {
        return rows * cols;
    }
};

// ======================= 压缩权重文件头 =======================
#pragma pack(push, 1)
struct QuantHeader {
    char magic[4];  // "LPQ1"
    int32_t format; // 0 = mxfp8, 1 = nvfp4
    int32_t elem;   // mxfp8: 0 = e4m3, 1 = e5m2
    int64_t rows;
    int64_t cols;
    int32_t block_size; // 实际使用的 block 粒度(tensor 模式下等于元素总数)
    int32_t scale_mode; // 0 = tensor, 1 = block
    int32_t out_dtype;  // 反量化输出类型
    int32_t round_mode;
    int64_t num_blocks;
    float global_scale;   // nvfp4 全局缩放；mxfp8 恒为 1.0
    int64_t packed_bytes; // 打包后的低精度数据字节数
    int64_t scale_bytes;  // 缩放因子字节数
};
#pragma pack(pop)

// 配置解析
Config load_config(const std::string& path);

// 张量读写（文本头 + 二进制数据段）
Tensor load_tensor(const std::string& path);
void save_tensor(const std::string& path, const Tensor& t);

// 生成测试数据：kind = random | normal | outlier
Tensor gen_tensor(const std::string& kind, int64_t rows, int64_t cols, uint32_t seed, bool as_fp16);

// 压缩权重文件
void save_quant_file(const std::string& path, const QuantHeader& h, const uint8_t* packed,
                     const uint8_t* scales);
bool load_quant_file(const std::string& path, QuantHeader& h, std::vector<uint8_t>& packed,
                     std::vector<uint8_t>& scales);

// 反量化张量（裸二进制，行主序，类型由 out_dtype 指定）
void save_dequant_tensor(const std::string& path, const std::vector<float>& data, OutDtype dt);
std::vector<float> load_dequant_tensor(const std::string& path, OutDtype dt, int64_t numel);

// 将 fp32 值按输出类型做一次舍入（用于保证 CPU 参考与 GPU 输出逐位一致）
float round_to_out_dtype(float v, OutDtype dt);

// 元素类型名（用于日志）
const char* out_dtype_name(OutDtype d);
const char* scale_mode_name(ScaleMode s);
const char* round_mode_name(RoundMode r);

} // namespace lowp

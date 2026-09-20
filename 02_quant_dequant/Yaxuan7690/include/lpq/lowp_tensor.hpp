// =============================================================================
//  lowp_tensor.hpp -- 输入张量文件 / 低精度权重文件 / 反量化输出文件 的读写
//
//  张量输入文件(题目「张量数据文件」的落地定义,小端):
//      int64  num_rows
//      int64  num_cols
//      uint8  dtype_len
//      char   dtype[dtype_len]        // "fp32" / "fp16"
//      <data> num_rows*num_cols 个元素,按行主序(row-major)紧排
//  读取时兼容 dtype 串后按 8 字节对齐的变体,避免因生成脚本差异导致无法读取.
//
//  低精度权重文件(题目「低精度权重文件」):
//      128 字节固定头 LowpHeader + packed 数据 + scale 数据
//      4bit 格式严格按每字节 2 个元素打包(低半字节在前).
//
//  反量化输出文件:按题目要求为「纯二进制,行主序」,不带任何头.
// =============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "lpq/config.hpp"

namespace lpq {

// -----------------------------------------------------------------------------
// 宿主侧张量(内容为原始位模式,按输入 dtype 解释)
// -----------------------------------------------------------------------------
struct HostTensor {
    int64_t rows = 0;
    int64_t cols = 0;
    InputType dtype = InputType::FP32;
    std::vector<uint8_t> data;

    int64_t numel() const { return rows * cols; }
    size_t elem_size() const { return dtype == InputType::FP32 ? 4u : 2u; }
};

HostTensor read_tensor_file(const std::string& path);
// 将张量写成同样格式(用于测试中生成 fp16 输入等)
void write_tensor_file(const std::string& path, const HostTensor& t);

// -----------------------------------------------------------------------------
// 低精度权重文件头(128 字节,小端;写入时先 memcpy 后整体转小端无必要,
// 因为目标平台均为小端,且文件不做跨端交换)
// -----------------------------------------------------------------------------
#pragma pack(push, 1)
struct LowpHeader {
    char magic[8];              // "LPQLOWP\0"
    uint32_t version;           // = 1
    uint32_t format;            // 0 = mxfp8, 1 = nvfp4, 2 = mxfp4
    uint32_t elem_bits;         // 8 / 4
    uint32_t elem_kind;         // 0 = e4m3, 1 = e5m2, 2 = e2m1
    uint32_t scale_kind;        // 0 = e8m0, 1 = e4m3
    uint32_t scale_mode;        // 0 = tensor, 1 = block
    uint32_t rounding;          // 0 = nearest, 1 = stochastic
    uint32_t has_global_scale;  // nvfp4 为 1,mxfp8 / mxfp4 为 0
    int64_t num_rows;
    int64_t num_cols;
    int64_t block_size;  // 每个缩放因子覆盖的元素数(tensor 模式下等于 numel)
    int64_t num_blocks;
    int64_t packed_bytes;
    int64_t scale_bytes;
    float global_scale;      // NVFP4 全张量缩放;MXFP8 / MXFP4 固定为 1.0
    float global_scale_inv;  // 1 / global_scale
    uint64_t seed;           // 随机舍入种子
    uint8_t reserved[24];
};
#pragma pack(pop)
static_assert(sizeof(LowpHeader) == 128, "LowpHeader 必须为 128 字节");

struct LowpFile {
    LowpHeader header{};
    std::vector<uint8_t> packed;
    std::vector<uint8_t> scales;
};

void write_lowp_file(const std::string& path, const LowpFile& f);
LowpFile read_lowp_file(const std::string& path);

// 纯二进制写出(反量化张量)
void write_raw_file(const std::string& path, const void* data, size_t bytes);

// 依据格式返回元素编码名(写入日志用)
const char* elem_kind_name(uint32_t elem_kind);
const char* scale_kind_name(uint32_t scale_kind);

}  // namespace lpq
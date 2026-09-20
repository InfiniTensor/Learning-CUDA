// quant.h - 量化/反量化流程的统一接口（GPU 实现与 CPU 参考实现）
#pragma once
#include "io.h"
#include "lowp.h"
#include <cstdint>
#include <string>
#include <vector>

namespace lowp {

// 一次完整量化 + 反量化流程的结果
struct QuantResult {
    std::vector<uint8_t> packed; // 低精度打包数据
    std::vector<uint8_t> scales; // 缩放因子（E8M0 或 E4M3 字节）
    float global_scale = 1.0f;   // nvfp4 全局缩放；mxfp8 恒为 1
    int64_t num_blocks = 0;
    int block_size = 0;         // 实际 block 粒度（tensor 模式下等于 n）
    double quant_ms = 0.0;      // 量化 kernel 时间
    double dequant_ms = 0.0;    // 反量化 kernel 时间
    double cpu_ms = 0.0;        // CPU 参考实现耗时(仅 run_cpu 填充)
    std::vector<float> dequant; // 反量化结果（fp32 域，长度 n）
};

// GPU 全流程（量化 + 反量化）。成功返回 true，否则 err 给出原因。
bool run_gpu(const std::vector<float>& in, int64_t n, const Config& cfg, QuantResult& out,
             std::string& err);

// CPU 参考全流程（标量实现，用于正确性对照）
QuantResult run_cpu(const std::vector<float>& in, int64_t n, const Config& cfg);

// 从已保存的压缩权重（packed + scales + global_scale）直接用 GPU kernel 反量化
bool run_gpu_dequant(const std::vector<uint8_t>& packed, const std::vector<uint8_t>& scales,
                     float gscale, int64_t n, int64_t bs, int is_nvfp4, int elem_fmt, OutDtype dt,
                     std::vector<float>& out, double* ms, std::string& err);

// 主机端 **fp32 域** 反量化（不经过输出类型舍入）。
// 用于把「纯量化误差」与「端到端误差(含 fp16/bf16 输出量程与舍入)」区分开。
std::vector<float> host_dequant_fp32(const std::vector<uint8_t>& packed,
                                     const std::vector<uint8_t>& scales, float gscale, int64_t n,
                                     int64_t bs, bool is_nvfp4, ElemFmt elem);

} // namespace lowp

// =============================================================================
//  reference.hpp -- 完全独立的 CPU 参考实现
//
//  独立性的来源:
//    1) 格式转换采用「枚举全部可表示值 + 二分查找取值区间」,而不是 formats.hpp
//       的尾数截断位运算;
//    2) 共享指数用「反复折半/加倍计数」求 floor(log2()),而不是 IEEE 位域提取;
//    3) 流水线是完全串行的标量实现,与 GPU 的多线程/向量化访存无关.
//  因此 CPU 与 GPU 结果一致可以作为较强的正确性证据.
//
//  注:随机舍入(SR)两端使用随机数的方式不同(GPU 端按低位掩码比较,参考端按
//  浮点概率比较),因此 SR 只做统计意义下的交叉验证;RNE 要求逐位一致.
// =============================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "lpq/config.hpp"
#include "lpq/formats.hpp"

namespace lpq {
namespace ref {

struct RefFmt {
    int ebits = 4;
    int mbits = 3;
    int bias = 7;
    bool has_inf = false;
    bool has_nan = true;
};

// kind: 0 = e4m3, 1 = e5m2, 2 = e2m1
RefFmt ref_fmt_of(int kind);

struct RefTable {
    RefFmt f;
    std::vector<double> vals;     // 升序排列的正幅值(不含 0,用于二分定位区间)
    std::vector<uint32_t> codes;  // 与 vals 对应的幅值编码
    std::vector<double> decode;   // 编码 -> 值(NaN/Inf 编码为 NaN/Inf)
    uint32_t max_code = 0;
    uint32_t inf_code = 0;
    uint32_t qnan_code = 0;
};

RefTable build_table(const RefFmt& f);
// 与 formats.hpp::f32_to_fmt 语义一致:饱和 + RNE(ties-to-even)或随机舍入
uint32_t table_quantize(const RefTable& t, float x, bool stochastic, uint64_t rnd);
// 编码 -> 值(含符号位)
double table_value(const RefTable& t, uint32_t code);

// 用「反复折半/加倍」求 floor(log2(v)),与 OCP 公式一致且无浮点对数误差
int exact_log2_floor(double v);

struct CpuQuantized {
    std::vector<uint8_t> packed;
    std::vector<uint8_t> scales;
    float global_scale = 1.0f;
    float global_scale_inv = 1.0f;
    int64_t block_size = 0;
    int64_t num_blocks = 0;
};

CpuQuantized cpu_quantize(const float* x, int64_t numel, const QuantConfig& cfg);
std::vector<float> cpu_dequantize(const CpuQuantized& q, int64_t numel, const QuantConfig& cfg);

}  // namespace ref
}  // namespace lpq
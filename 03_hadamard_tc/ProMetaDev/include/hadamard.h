// hadamard.h - Hadamard 变换的 CPU 参考与 GPU 接口
#pragma once
#include "types.h"
#include <string>
#include <vector>

namespace hd {

// ======================= CPU 参考 =======================
// 快速 Walsh-Hadamard 变换（O(n log n)），逐行独立，乘性缩放 scale
void cpu_fht(const Tensor& in, Tensor& out, float scale);

// 朴素参考：直接构造 H_n 并做矩阵乘（O(n^2)），用于验证 FWHT 本身的正确性
void cpu_matmul_ref(const Tensor& in, Tensor& out, float scale);

// 从量化结果反量化回 fp32（用于验证量化链路）
void cpu_dequant(const QuantResult& qr, int32_t rows, int32_t cols, int32_t block_size,
                 QuantFormat fmt, Tensor& out);

// ======================= GPU 接口 =======================
// 单独的 FHT kernel
bool gpu_fht(const Tensor& in, const Config& cfg, Tensor& out, double* ms, std::string& err);

// 单独的量化 kernel（对已变换的数据做 FP8 block 量化）
bool gpu_quantize(const Tensor& in, const Config& cfg, QuantResult& out, double* ms,
                  std::string& err);

// 融合：FHT + 量化，一次完成（中间结果只留在寄存器/共享内存，不落显存）
bool gpu_fht_quant_fused(const Tensor& in, const Config& cfg, QuantResult& out, double* ms,
                         std::string& err);

// 朴素矩阵乘基线：y = scale * H_n @ x（O(n^2)，不含任何快速算法）。
// 既是「手动实现」的性能对照基线，也用于大数据规模下替代 O(n^2) 的 CPU 参考做正确性校验。
bool gpu_matmul_baseline(const Tensor& in, float scale, Tensor& out, double* ms, std::string& err);

// 有效带宽（GB/s）：读写字节数 / 时间
double bandwidth_gbps(double bytes, double ms);

} // namespace hd

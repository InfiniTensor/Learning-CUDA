// types.h - Hadamard 变换加速的基础类型
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace hd {

enum class Dtype { FP32 = 0, FP16 = 1, BF16 = 2 };
enum class QuantFormat { NONE = 0, FP8_E4M3 = 1, FP8_E5M2 = 2 };

// ======================= 配置 =======================
struct Config {
    int hadamard_size = 128; // n ∈ {64, 128, 256}
    float scale = 1.0f;      // 输出乘性缩放，如 1/sqrt(n)
    bool quantize_enable = false;
    QuantFormat quant_format = QuantFormat::FP8_E4M3;
    int block_size = 32; // 量化块大小
    int threads = 256;   // 线程块大小
};

// ======================= 张量 =======================
// 统一以 fp32 在内存中存储；[rows, cols] 每行沿最后一维做 Hadamard 变换
struct Tensor {
    int32_t rows = 0;
    int32_t cols = 0;
    Dtype dtype = Dtype::FP32; // 输入文件中的原始类型
    std::vector<float> data;   // rows * cols，行主序
    int64_t numel() const {
        return (int64_t)rows * cols;
    }
};

// ======================= 量化结果 =======================
struct QuantResult {
    std::vector<uint8_t> qdata;  // FP8 字节，rows*cols
    std::vector<uint8_t> scales; // E8M0 缩放因子，每 block 一个
    int64_t num_blocks = 0;
};

// ======================= 性能统计 =======================
struct PerfStats {
    double fht_ms = 0.0;   // 单独的 FHT kernel 时间
    double quant_ms = 0.0; // 单独的量化 kernel 时间
    double fused_ms = 0.0; // 融合 FHT+量化 kernel 时间
    double fht_bw = 0.0;   // FHT 有效带宽 (GB/s)
    double quant_bw = 0.0;
    double fused_bw = 0.0;
    double fusion_speedup = 0.0; // (fht_ms + quant_ms) / fused_ms
    double torch_ms = 0.0;       // PyTorch 基线（矩阵乘）时间，由外部填入
    double vs_torch_speedup = 0.0;
};

const char* dtype_name(Dtype d);
const char* quant_name(QuantFormat q);

} // namespace hd

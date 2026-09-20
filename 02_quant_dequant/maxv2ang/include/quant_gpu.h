// ============================================================================
// 选题二「MXFP8 / NVFP4 低精度模拟与反量化」—— 库 API（GPU 侧）
// ============================================================================
// GPU 实现的公开接口；实现在 src/quant_gpu.cu。
//
// 本库不含 main、不含命令行解析、不产出可执行文件。使用方式：
//   1. 调 gpu_init() 选择设备；
//   2. 用 quantize_gpu() / dequant_gpu() 逐 kernel 调用，或用 run_one() 端到端跑
//      一个输入（读张量 → 量化 → 反量化 → 误差/压缩率 → 落盘）。
//
// 格式语义（编解码、缩放、舍入、文件 I/O、配置解析、指标统计）在
// quant_format.h 中，主机端与设备端共用同一份实现 —— verify=true 时 GPU 输出会
// 与主机端逐字节对拍。
//
// 硬件约束（题目硬性要求）：
//   - 不依赖 Hopper/Blackwell/Ampere+ 特性；无 FP8/FP4 Tensor Core；
//   - grid-stride + warp shuffle + atomicMax(ull) 都是 sm_35 即有的原语。
// ============================================================================

#pragma once

#include <cuda_runtime.h>

#include "quant_format.h"
#include "quant_io.h"

#include <cstdint>
#include <cstdlib>  // CUDA_CHECK 里的 std::exit
#include <iostream> // CUDA_CHECK 里的 std::cerr
#include <string>
#include <vector>

// CUDA 运行时错误检查（出错即打印并退出）
#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t e_ = (call);                                                                   \
        if (e_ != cudaSuccess) {                                                                   \
            std::cerr << "CUDA error: " << cudaGetErrorString(e_) << " (" #call << ") at "         \
                      << __FILE__ << ":" << __LINE__ << "\n";                                      \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

// ---------------------------------------------------------------- 设备

// 选择设备并缓存 SM 数。调用任何其它 GPU 接口前先调用一次。
void gpu_init(int device = 0);

// ---------------------------------------------------------------- 逐 kernel

struct GpuQuantResult {
    Quantized q;            // GPU 量化的结果（已拷回主机）
    double quantize_ms = 0; // 含 amax kernel（如有），多次重复取最小
    bool verify_match = false;
};

struct GpuDequantResult {
    std::vector<float> out; // 已按 output_type 舍入、转回 FP32 表示（写文件用）
    double dequant_ms = 0;
    bool verify_match = false;
};

GpuQuantResult quantize_gpu(const std::vector<float> &w, const Config &cfg, int repeat, bool verify);

GpuDequantResult dequant_gpu(const Quantized &q, int out_type_id, int repeat, bool verify);

// ---------------------------------------------------------------- 端到端

// 一个（格式 × 矩阵）组合的完整结果
struct RunMetrics {
    std::string format, matrix, scale_mode, output_type;
    std::string in_dtype, rounding; // 输入矩阵 dtype / 舍入模式（题目要求的两个维度）
    int64_t num_elems = 0;
    int rows = 0, cols = 0;
    double quantize_ms = 0, dequant_ms = 0, quant_gbps = 0, dequant_gbps = 0;
    double max_abs = 0, mae = 0, mse = 0, rmse = 0;
    double compression_ratio = 0, dst_bits = 0;
    bool verify_quant = false, verify_dequant = false, verify_disabled = false;
    int repeat = 0;
};

// 端到端跑一个输入：读张量 → 量化 → 反量化 → 误差/压缩率 → 落盘 → 指标
//   input_path  输入张量文件
//   cfg         量化参数
//   outdir      输出父目录，产出 outdir/<输入名>/ 下的 weights.bin、dequant.bin、
//               metrics.json、metrics.log
//   repeat      计时重复次数（取最小值）
//   verify      是否与主机端实现逐字节对拍
//   row         非空则填入该组合的指标
// 返回 0 成功；-1 表示 verify 对拍不一致；其它非 0 为错误
int run_one(const std::string &input_path, const Config &cfg, const std::string &outdir, int repeat,
            bool verify, RunMetrics *row);

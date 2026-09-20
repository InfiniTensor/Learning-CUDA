#pragma once

// 硬件/库对照路径（训练营加分项）的公共接口。
// 只有在 CMake 打开 WITH_HW_PATH 后才会编译 src/hw_path.cu 并定义
// LPQ_WITH_HW_PATH；默认构建不包含本文件声明的任何实现，CLI 行为与之前一致。
//
// 依赖说明：
// - cvt 路径使用 cuda_fp8.h 的 SM 原生 E4M3 转换指令，要求 sm_89+
//   （Ada/Hopper/Blackwell）；本机 sm_120 实测可用。
// - gemm 路径额外要求 LPQ_HW_CUBLAS（构建期找到 cuBLASLt），运行时能否执行
//   由 cublasLtMatmulAlgoGetHeuristic 探测决定，不可用时在结果中如实标注。

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "low_precision.cuh"

#ifdef LPQ_WITH_HW_PATH

namespace lpq::hw {

// cvt 指令路径需要 sm_89 及以上的 E4M3 转换指令。
bool cvt_supported(const cudaDeviceProp& device);

// MXFP8 元素级硬件编码：value = input[i] / decode_e8m0(scale)，随后用
// cvt.rn.satfinite.e4m3 指令编码。语义与软件 encode_e4m3_nearest 一致
// （SATFINITE 饱和 + round-to-nearest-even）；唯一的已知差异是硬件指令保留
// -0.0 的符号位，kernel 内已把 -0.0 归一化为 +0.0，使输出与软件逐字节相同。
// 输入只支持 FP32（基准输入均为 FP32）。
cudaError_t launch_mx_encode_cvt(const float* input, const std::uint8_t* scales,
                                 std::uint8_t* packed, std::size_t count,
                                 int block_size, bool tensor_mode,
                                 cudaStream_t stream);

// MXFP8 元素级硬件解码：E4M3 byte 经 cvt 指令转 half 再转 FP32，乘 E8M0
// scale 得到 FP32 输出。E4M3 全部有限值在 half 中精确可表示，因此结果与软件
// decode_e4m3 位级一致（NaN 编码 0x7f/0xff 仍为 NaN）。
cudaError_t launch_mx_decode_cvt(const std::uint8_t* packed,
                                 const std::uint8_t* scales, float* output,
                                 std::size_t count, int block_size,
                                 bool tensor_mode, cudaStream_t stream);

#ifdef LPQ_HW_CUBLAS

// GEMM 对照结果：推理线性层 Y = X·Wᵀ，X/W 为同一 side×side 输入。
// 软件路径 = 反量化到 FP16 + cublasGemmEx；硬件路径 = cuBLASLt 原生
// FP8（E4M3 + VEC32_UE8M0 块缩放）或 FP4（E2M1 + VEC16_UE4M3 块缩放）GEMM。
struct GemmBenchResult {
  bool hw_supported = false;   // cuBLAS 是否接受该类型/缩放/布局组合
  std::string hw_impl;         // 实际使用的实现描述，便于报告如实标注
  int side = 0;                // M = N = K
  double hw_gemm_ms = 0.0;
  double sw_dequant_ms = 0.0;  // X 和 W 两次反量化 kernel 合计
  double sw_gemm_ms = 0.0;
  double hw_tflops = 0.0;
  double sw_tflops = 0.0;      // 仅 FP16 GEMM 本身
  // 正确性：256³ 上对 CPU double 参考的相对误差，以及基准尺寸上
  // 硬件 GEMM 与软件 FP16 GEMM 输出之间的最大相对差。
  double self_check_hw_rel = -1.0;
  double self_check_sw_rel = -1.0;
  double bench_hw_vs_sw_max_rel = -1.0;
  double bench_hw_vs_sw_max_abs = -1.0;
};

// 运行一次 GEMM 对照。d_input 是 side×side 的 FP32 device 输入；format 决定
// 走 FP8（kMxfp8）还是 FP4（kNvfp4）路径。量化仍由软件路径完成（两种对照共用
// 同一份量化结果，比较的只是「反量化+FP16 GEMM」与「原生低精度 GEMM」）。
// warmup/iterations 与 CLI 其余计时语义一致。
cudaError_t run_gemm_comparison(const float* d_input, int side, Format format,
                                int warmup, int iterations,
                                GemmBenchResult* result);

#endif  // LPQ_HW_CUBLAS

}  // namespace lpq::hw

#endif  // LPQ_WITH_HW_PATH

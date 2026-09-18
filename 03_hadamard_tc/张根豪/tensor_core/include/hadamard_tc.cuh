// hadamard_tc.cuh — 基于 Tensor Core (WMMA) 的 Hadamard 变换 host 接口
//
// ===========================================================================
// 算法：为什么 Hadamard 变换可以变成两次 16x16 矩阵乘
// ===========================================================================
//
// 直接用稠密 H_d @ x 走 Tensor Core 是错误方向：复杂度会从 O(d log d) 退化成
// O(d^2)。本模块用 Sylvester 矩阵的 Kronecker 分解避开这一点：
//
//     H_d = H_{d/16} (kron) H_16          (d 为 2 的幂且 d >= 64)
//
// 把 token 内下标写成 j = 16*rr + c（rr = j/16, c = j%16），则
//
//     y[16*rr+c] = sum_{rr',c'} H_{d/16}[rr,rr'] * H_16[c,c'] * x[16*rr'+c']
//
// 即：把 token 的 d 个元素按 row-major 摆成 (d/16) x 16 的矩阵 X，则
//
//     Y = H_{d/16} @ X @ H_16^T ,   而 H_16 对称，H_16^T = H_16
//
// 为了让每次 WMMA 都吃满 m16n16k16 的形状，本实现固定“每个 warp 处理 256 个
// 连续元素（一个 16x16 tile）”：
//
//   d = 256 : 一个 tile 恰好一个 token；
//   d = 128 : 一个 tile 装 2 个 token，行 0-7 属 token0、行 8-15 属 token1；
//   d = 64  : 一个 tile 装 4 个 token，每 4 行一个 token；
//
// 于是左乘矩阵统一写成块对角常量矩阵
//
//     A = I_k (kron) H_m ,  m = min(d,256)/16 ,  k = 16/m
//
// 最终每个 tile 只需要两条 mma：
//
//     P = X @ H_16   (第 1 次 WMMA，X 是原始低精度输入，FP32 累加仍可能舍入)
//     Y = A @ P      (第 2 次 WMMA)
//
// d = 512/1024 时一个 token 跨 kTiles = d/256 个 tile：先在每个 tile 内做上面
// 两次 mma，再对 accumulator 逐元素做 log2(kTiles) 级蝶形（对应
// H_d = H_{kTiles} (kron) H_256），完全在寄存器里完成。
//
// ===========================================================================
// 两种精度模式
// ===========================================================================
//
// WMMA 的 accumulator 是 FP32，但 operand 必须是 FP16/BF16。第 1 次 mma 的输入
// 就是原始张量；累加仍可能舍入，且中间结果 P 必须转回低精度才能喂给第 2 次 mma。
//
//   - TC_FAST  : P 直接舍入成 FP16/BF16，2 次 mma。最快，但引入一次额外舍入。
//   - TC_SPLIT : 把 P 拆成 P = P_hi + P_lo（P_hi = round(P)，
//                P_lo = round(P - P_hi)），用 3 次 mma 累加到同一个 FP32
//                accumulator。等效尾数从 11/8 bit 提升到约 22/16 bit，
//                显著减小中间舍入误差，但不保证与 FP32 FHT 普遍逐位一致。
//
// 两条路径都保持与 FHT 相同的写回语义：FP32 结果乘可选 1/sqrt(d) 后一次舍入。
#pragma once

#include "hadamard.cuh"

#include <cuda_runtime.h>

// Tensor Core 精度模式
enum class TcMode {
  Fast,   // 2 次 mma，中间结果单份低精度
  Split,  // 3 次 mma，中间结果 hi/lo 双份低精度
};

// Tensor Core Hadamard 变换。
//
// 参数与 launch_hadamard 完全一致，额外的 mode 选择精度/速度折中。
// head_dim 支持 64/128/256/512/1024（必须 >= 64，因为一个 tile 是 256 元素）。
// 若 total_tokens 不能被每 tile 的 token 数整除，尾部 token 会自动回落到
// launch_hadamard_warp，保证结果完整且语义一致。
//
// 返回值：0 成功；-1 不支持的 head_dim；-2 不支持的 dtype。
int launch_hadamard_tc(const void* input,
                       void* output,
                       int batch_size,
                       int seq_len,
                       int num_heads,
                       int head_dim,
                       DataType dtype,
                       bool normalize,
                       TcMode mode,
                       cudaStream_t stream);

// Tensor Core Hadamard + per-token symmetric INT4 融合。
// 量化协议与 launch_hadamard_fused_quant_int4 完全相同：
//   scale = max|x|/7（全零 token 用 1），round-to-nearest-even 后 clamp 到
//   [-7,7]，相邻两元素打包进一个 byte 的低/高 nibble。
// 仅支持 head_dim <= 256（此时一个 tile 内的 token 边界与 lane 分组对齐，
// per-token max 可以用一次 sub-warp shuffle 归约完成）。
//
// 不足一个 tile 的尾部 token 自动回退到 warp 融合，与变换接口的尾部语义一致。
// 返回值：0 成功；-1 不支持的 head_dim；-2 不支持的 dtype。
int launch_hadamard_tc_fused_quant_int4(const void* input,
                                        unsigned char* packed_output,
                                        float* scales,
                                        int batch_size,
                                        int seq_len,
                                        int num_heads,
                                        int head_dim,
                                        DataType dtype,
                                        bool normalize,
                                        TcMode mode,
                                        cudaStream_t stream);

// 工具：模式名（日志/CSV 用）
const char* tc_mode_name(TcMode mode);

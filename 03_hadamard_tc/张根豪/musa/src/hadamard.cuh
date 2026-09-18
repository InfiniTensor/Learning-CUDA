// hadamard.cuh — 快速 Hadamard 变换 CUDA kernel 的 host 侧接口
//
// 语义：
//   输入张量逻辑形状 [batch_size, seq_len, num_heads, head_dim]，
//   展平为 [total_tokens, head_dim]，total_tokens = batch_size*seq_len*num_heads。
//   每个 token 独立地对最后一维做 Hadamard 变换：
//     非归一化: y = H x
//     归一化:   y = H x / sqrt(head_dim)        (默认，正交旋转，误差阈值友好)
//
// 设计时预留的扩展位（Week3+ 融合量化 / Tensor Core）：
//   - dtype / normalize 已经是运行时参数；
//   - 后续可在此接口旁新增 launch_hadamard_fused_quant(..., QuantType quant_type, ...)
//     复用同一套 (dtype, head_dim) 分发，而不必改动本接口。
#pragma once

#include <musa_runtime.h>

// 支持的数据类型。 Week1/Week2 仅要求 FP16 / BF16。
enum class DataType {
  FP16,
  BF16,
};

// host 侧统一入口。
//
// 参数：
//   input/output  设备指针，元素类型为 dtype 对应的 __half / __nv_bfloat16
//   head_dim      2 的幂；当前实例化支持 32/64/128/256/512/1024（64/128/256 为核心）
//   normalize     true: y = Hx/sqrt(head_dim)；false: y = Hx
//   stream        调用方提供的 CUDA stream
//
// 返回值：
//   0  成功（kernel 已入队，不代表已执行完）
//   -1 不支持的 head_dim
//   -2 不支持的 dtype
//
// 注意：CUDA 调用错误在本函数内部用 CUDA_CHECK 直接报错退出，不隐藏错误。
int launch_hadamard(const void* input,
                    void* output,
                    int batch_size,
                    int seq_len,
                    int num_heads,
                    int head_dim,
                    DataType dtype,
                    bool normalize,
                    cudaStream_t stream);

// Week3 优化入口。数学与数值语义和 launch_hadamard 完全一致，但使用：
//   1. warp shuffle 完成 warp 内蝶形；
//   2. half2/bfloat162 相邻元素向量化 load/store；
//   3. d=64 每 block 4 tokens、d=128 每 block 2 tokens。
int launch_hadamard_optimized(const void* input,
                              void* output,
                              int batch_size,
                              int seq_len,
                              int num_heads,
                              int head_dim,
                              DataType dtype,
                              bool normalize,
                              cudaStream_t stream);

// 对低精度 Hadamard 输出做 per-token symmetric INT4 量化。
// q = clamp(round_to_nearest_even(x / scale), -7, 7), scale=max(abs(x))/7。
// packed_output 的低/高 nibble 依次保存相邻两个元素的 4-bit two's-complement；
// 每 token 占 head_dim/2 bytes，scales 每 token 一个 FP32。
int launch_quantize_int4(const void* input,
                         unsigned char* packed_output,
                         float* scales,
                         int total_tokens,
                         int head_dim,
                         DataType dtype,
                         cudaStream_t stream);

// 融合 optimized Hadamard + INT4。为严格匹配 unfused 路径，Hadamard FP32
// 中间结果在寄存器中先模拟转换到输入 dtype，再计算 scale/INT4，避免写回中间张量。
int launch_hadamard_fused_quant_int4(const void* input,
                                     unsigned char* packed_output,
                                     float* scales,
                                     int batch_size,
                                     int seq_len,
                                     int num_heads,
                                     int head_dim,
                                     DataType dtype,
                                     bool normalize,
                                     cudaStream_t stream);

// 工具：数据类型名称（日志/CSV 用）
const char* dtype_name(DataType dtype);

// ---------------------------------------------------------------------------
// warp-per-token FHT（optimized 的进一步优化版本）
// ---------------------------------------------------------------------------
//
// 与 launch_hadamard / launch_hadamard_optimized 数学语义完全一致，但把每线程
// 负责的元素数从 2 提升到 head_dim/32，于是：
//   1. 一个 warp 恰好处理一个 token，蝶形的 log2(head_dim/32) 个低阶 stage 在
//      寄存器数组内完成，高 5 个 stage 全部用 __shfl_xor_sync；
//   2. 完全不使用 shared memory，也不需要任何 __syncthreads()；
//   3. global load/store 宽度提升到 head_dim*2/32 字节（d=256 即 128-bit）。
// 要求 head_dim >= 64（否则一个 warp 装不满 2 个元素/线程）。
int launch_hadamard_warp(const void* input,
                         void* output,
                         int batch_size,
                         int seq_len,
                         int num_heads,
                         int head_dim,
                         DataType dtype,
                         bool normalize,
                         cudaStream_t stream);

// warp-per-token FHT + per-token symmetric INT4 融合。量化协议与
// launch_hadamard_fused_quant_int4 完全相同（含写回前的 dtype 舍入模拟），
// 但 per-token max 归约用纯 warp shuffle，没有 shared memory 与 barrier。
int launch_hadamard_fused_quant_int4_warp(const void* input,
                                          unsigned char* packed_output,
                                          float* scales,
                                          int batch_size,
                                          int seq_len,
                                          int num_heads,
                                          int head_dim,
                                          DataType dtype,
                                          bool normalize,
                                          cudaStream_t stream);

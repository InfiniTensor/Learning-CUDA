#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace mxfp8 {

constexpr int kBlockSize = 32;

// BF16 使用软件转换，不依赖原生 BF16 指令。
struct BF16 {
    uint16_t bits;
};

// 返回 n 个元素需要的 E8M0 scale 数量。
inline int num_scales(int n) {
    return (n + kBlockSize - 1) / kBlockSize;
}

// FP32 / FP16 -> MXFP8。
void quantize(const float* input,
              uint8_t* quantized,
              uint8_t* scales,
              int n,
              cudaStream_t stream = 0);

void quantize(const __half* input,
              uint8_t* quantized,
              uint8_t* scales,
              int n,
              cudaStream_t stream = 0);

// MXFP8 -> FP32 / FP16 / BF16。
void dequantize(const uint8_t* quantized,
                const uint8_t* scales,
                float* output,
                int n,
                cudaStream_t stream = 0);

void dequantize(const uint8_t* quantized,
                const uint8_t* scales,
                __half* output,
                int n,
                cudaStream_t stream = 0);

void dequantize(const uint8_t* quantized,
                const uint8_t* scales,
                BF16* output,
                int n,
                cudaStream_t stream = 0);

}  // namespace mxfp8

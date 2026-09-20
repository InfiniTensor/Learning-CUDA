// Public API of the Hadamard transform implementations.
//
// Convention used everywhere in this project (see docs/learning_notes.md):
//     y = H_d * x / sqrt(d)
// with H_d the Sylvester matrix, H_1 = [1] and H_2n = [[H_n, H_n], [H_n, -H_n]].
// The 1/sqrt(d) factor makes the transform orthonormal: scaled twice it is the
// identity. Confirmed against Dao-AILab/fast-hadamard-transform in M2 (same scale
// convention, bit-exact on every case tested).
#ifndef HADAMARD_HADAMARD_H_
#define HADAMARD_HADAMARD_H_

#include <cuda_runtime.h>

#include <cstdint>

#include "hadamard/common.h"

namespace hadamard {

// ---------------------------------------------------------------------------
// CPU FP32 reference implementations (ground truth for the correctness harness)
// ---------------------------------------------------------------------------

// Explicit H_d matrix multiply, O(d^2) per row. Slow on purpose: it pins down the
// sign/matrix convention independently of the butterfly implementation.
void ref_matmul_fp32(const float* x, float* y, long rows, int d);

// Butterfly fast Walsh-Hadamard transform, O(d log2 d) per row.
void ref_fwht_fp32(const float* x, float* y, long rows, int d);

// ---------------------------------------------------------------------------
// Host-side precision helpers: build the inputs the GPU kernels actually consume
// and dequantize their outputs before comparing.
// ---------------------------------------------------------------------------
void pack_fp32(const float* src, long n, uint16_t* dst, DType dtype);
void unpack_fp32(const uint16_t* src, long n, float* dst, DType dtype);

// ---------------------------------------------------------------------------
// FP8 E4M3 quantization (M4)
//
// The rotated activation is quantized per token, i.e. one scale per row of the
// flattened [B, S, H, head_dim] tensor. Rounding is round-to-nearest-even with
// saturation (__nv_cvt_float_to_fp8 with __NV_SATFINITE), shared between the
// reference and the kernels so a fused kernel can be compared byte for byte
// against the two-stage pipeline.
// ---------------------------------------------------------------------------

// Per-token scale, used by the host reference and by the device epilogues so the two
// cannot drift apart. An all-zero row would make the reciprocal that the quantizer
// multiplies with infinite (and 0 * inf a NaN), so it maps to scale = 1 instead.
__host__ __device__ inline float fp8_e4m3_scale(float amax) {
    return amax == 0.0f ? 1.0f : amax / kFp8E4M3Max;
}

// Decode one E4M3 code (S1E4M3, bias 7, subnormals, 0x7f/0xff are NaN).
// Implemented by hand on purpose: it is the independent check that the header's host
// conversion and the device instruction agree.
float fp8_e4m3_dequant(uint8_t code);

// scales[row] = fp8_e4m3_scale(max |y[row][:]|)
void ref_quant_scales_fp8(const float* y, float* scales, long rows, int d);
// q[row][i] = fp8_e4m3(y[row][i] / scales[row])
void ref_quantize_fp8(const float* y, const float* scales, uint8_t* q, long rows, int d);
// y[row][i] = fp8_e4m3_dequant(q[row][i]) * scales[row]
void ref_dequantize_fp8(const uint8_t* q, const float* scales, long rows, int d, float* y);

// ---------------------------------------------------------------------------
// GPU entry points
//
// Both launchers return false only when the shape or element type is unsupported, in
// which case the harness reports SKIP instead of FAIL:
//   launch_fwht_baseline -> M2 (register butterfly + warp shuffle, no Tensor Core)
//   launch_hadamard_tc   -> M3 (WMMA GEMM against the constant H_d matrix; head_dim
//                           below 16 cannot be tiled along K and falls back to M2)
// ---------------------------------------------------------------------------
bool launch_fwht_baseline(const void* x, void* y, const Shape& shape, DType dtype,
                          cudaStream_t stream = nullptr);

bool launch_hadamard_tc(const void* x, void* y, const Shape& shape, DType dtype,
                        cudaStream_t stream = nullptr);

// ---------------------------------------------------------------------------
// Fused FP8 entry points (M4)
//
// All three take the unquantized activation of `dtype`, write `shape.elems()` bytes
// of FP8 E4M3 and one FP32 scale per row. head_dim 16..512 is the range of the
// Tensor Core version; the butterfly version covers every head_dim the rest of the
// project supports.
//   launch_quantize_fp8        -> the second half of the two-stage pipeline
//   launch_fwht_baseline_fp8   -> M2 butterfly with the quantizer in its epilogue
//   launch_hadamard_tc_fp8     -> M3 Tensor Core with the quantizer in its epilogue
// ---------------------------------------------------------------------------
bool launch_quantize_fp8(const void* y, uint8_t* q, float* scales, const Shape& shape, DType dtype,
                         cudaStream_t stream = nullptr);

bool launch_fwht_baseline_fp8(const void* x, uint8_t* q, float* scales, const Shape& shape,
                              DType dtype, cudaStream_t stream = nullptr);

bool launch_hadamard_tc_fp8(const void* x, uint8_t* q, float* scales, const Shape& shape,
                            DType dtype, cudaStream_t stream = nullptr);

// Toy kernel proving the toolchain works end to end (M0 deliverable).
void launch_hello(int* out, int value, cudaStream_t stream = nullptr);

// Tensor Core probe: computes C = A * I on 16x16 fragments and checks the result
// equals A. Verifies that the WMMA path for `dtype` is usable on this GPU.
// Returns false when the element type is unsupported on the current device.
bool launch_wmma_identity_probe(DType dtype, bool* correct, cudaStream_t stream = nullptr);

}  // namespace hadamard

#endif  // HADAMARD_HADAMARD_H_

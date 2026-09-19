#pragma once

#include "kernels.cuh"

namespace hadamard {

// Each thread owns adjacent input values and writes their complete output byte.
// Scalar input loads retain the public contract for legal two-byte offsets.
template <class T, int N>
__global__ void quantize_pairs_kernel(const T* input, std::uint8_t* packed,
                                      float* scales, std::size_t rows) {
    static_assert(N >= 2 && N <= 16 && (N & (N - 1)) == 0);
    constexpr int Lanes = N / 2;
    const std::size_t pair = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t row = pair / Lanes;
    const int lane = threadIdx.x % Lanes;
    const bool valid = row < rows;
    const float low = valid ? as_float(input[pair * 2]) : 0.0f;
    const float high = valid ? as_float(input[pair * 2 + 1]) : 0.0f;
    float magnitude = fmaxf(fabsf(low), fabsf(high));
#pragma unroll
    for (int stride = Lanes / 2; stride > 0; stride /= 2)
        magnitude = fmaxf(magnitude, __shfl_xor_sync(0xffffffff, magnitude, stride, Lanes));
    const float scale = magnitude == 0 ? 1.0f : magnitude / 7.0f;
    const int q_low = max(-7, min(7, __float2int_rn(low / scale)));
    const int q_high = max(-7, min(7, __float2int_rn(high / scale)));
    if (valid) {
        packed[pair] = static_cast<std::uint8_t>((q_low & 15) | ((q_high & 15) << 4));
        if (lane == 0) scales[row] = scale;
    }
}

}  // namespace hadamard

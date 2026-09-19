#pragma once

#include "kernels.cuh"

namespace hadamard {

template <class T, int N, bool Transform, bool Quantize>
__global__ void packed_rows_kernel(const T* input, T* output, std::uint8_t* packed,
                                  float* scales, std::size_t rows, float transform_scale) {
    static_assert(N >= 1 && N <= 16 && (N & (N - 1)) == 0);
    const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t row = element / N;
    const int lane = threadIdx.x % N;
    const bool valid = row < rows;
    // Every lane reaches all shuffles, including complete masked subgroups at the tail.
    float value = valid ? as_float(input[element]) : 0.0f;
    if constexpr (Transform) {
#pragma unroll
        for (int stride = 1; stride < N; stride *= 2) {
            const float peer = __shfl_xor_sync(0xffffffff, value, stride, N);
            value = (lane & stride) ? peer - value : value + peer;
        }
        value = as_float(as_storage<T>(value * transform_scale));
    }
    if constexpr (!Quantize) {
        if (valid) output[element] = as_storage<T>(value);
    } else {
        float magnitude = fabsf(value);
#pragma unroll
        for (int stride = N / 2; stride > 0; stride /= 2)
            magnitude = fmaxf(magnitude, __shfl_xor_sync(0xffffffff, magnitude, stride, N));
        const float scale = magnitude == 0 ? 1.0f : magnitude / 7.0f;
        if (valid && lane == 0) scales[row] = scale;
        const int q = max(-7, min(7, __float2int_rn(value / scale)));
        int next = 0;
        if constexpr (N > 1) next = __shfl_down_sync(0xffffffff, q, 1, N);
        if (valid && !(lane & 1)) {
            const int high = N > 1 ? (next & 15) : 0;
            packed[row * ((N + 1) / 2) + lane / 2] = static_cast<std::uint8_t>((q & 15) | (high << 4));
        }
    }
}

}  // namespace hadamard

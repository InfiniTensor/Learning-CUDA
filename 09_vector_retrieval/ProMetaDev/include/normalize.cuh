// normalize.cuh - 行归一化（cosine 度量在检索前统一归一化，从而等价于内积）
#pragma once
#include <cuda_runtime.h>

namespace vs {

// 把每行向量归一化为单位长度
static __global__ void k_normalize(float* __restrict__ data, int64_t n, int32_t dim) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    float* v = data + i * dim;
    float s = 0.0f;
    for (int32_t d = 0; d < dim; d++)
        s += v[d] * v[d];
    float inv = 1.0f / (sqrtf(s) + 1e-12f);
    for (int32_t d = 0; d < dim; d++)
        v[d] *= inv;
}

} // namespace vs

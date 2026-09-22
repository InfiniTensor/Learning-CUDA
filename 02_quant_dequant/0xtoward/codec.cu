#include "formats.cuh"
#include <cuda_fp16.h>

// dtype ids: 0=FP32, 1=FP16, 2=BF16. BF16 is encoded in software for SM75.
__device__ float load_value(const void *x, int64_t i, int dtype) {
  if (dtype == 0)
    return static_cast<const float *>(x)[i];
  if (dtype == 1)
    return __half2float(static_cast<const __half *>(x)[i]);
  return __uint_as_float(unsigned(static_cast<const uint16_t *>(x)[i]) << 16);
}
__device__ void store_value(void *y, int64_t i, int dtype, float value) {
  if (dtype == 0)
    static_cast<float *>(y)[i] = value;
  else if (dtype == 1)
    static_cast<__half *>(y)[i] = __float2half_rn(value);
  else {
    unsigned u = __float_as_uint(value);
    static_cast<uint16_t *>(y)[i] =
        isnan(value) ? 0x7fc0 : uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16);
  }
}
__device__ float warp_max(float x) {
  for (int d = 16; d; d >>= 1)
    x = fmaxf(x, __shfl_down_sync(~0u, x, d));
  return __shfl_sync(~0u, x, 0);
}
__device__ float warp_sum(float x) {
  for (int d = 16; d; d >>= 1)
    x += __shfl_down_sync(~0u, x, d);
  return __shfl_sync(~0u, x, 0);
}

// One warp handles one 32-element MXFP8 or 16-element NVFP4 block.
__global__ void quantize_naive(const void *x, uint8_t *q, uint8_t *scales,
                               const float *amax_tensor, float *global_scale,
                               int rows, int cols, int fp4, int dtype,
                               int tensor, int random_rounding, unsigned seed,
                               int rceil, int four_over_six, float fp8_bound) {
  int lane = threadIdx.x, size = fp4 ? 16 : 32;
  int groups = (cols + size - 1) / size, group = blockIdx.x;
  int row = group / groups, col = (group % groups) * size + lane;
  bool valid = lane < size && col < cols;
  int64_t index = int64_t(row) * cols + col;
  float xi = valid ? load_value(x, index, dtype) : 0;
  bool bad = __ballot_sync(~0u, !isfinite(xi)) != 0;
  float amax = warp_max(fabsf(xi));
  if (tensor)
    amax = *amax_tensor;
  int scale_index = tensor ? 0 : group;
  if (!fp4) {
    int e = amax > 0 ? ilogbf(amax) - 8 : -127;
    // Increase only when amax would exceed 448; avoids log2 boundary rounding.
    if (rceil && amax > 0 && ldexpf(amax, -e) > 448)
      ++e;
    e = max(-127, min(127, e));
    if (!lane && (!tensor || !group))
      scales[scale_index] = bad ? 255 : e + 127;
    if (valid) {
      float z = ldexpf(xi, -e);
      q[index] = bad               ? 127
                 : random_rounding ? stochastic(z, false, random01(index, seed))
                                   : e4_encode(z);
    }
    if (!group && !lane)
      *global_scale = 1;
    return;
  }
  float encode_scale =
      *amax_tensor > 0 ? __fdiv_rn(6 * fp8_bound, *amax_tensor) : 1;
  float g = *amax_tensor > 0 ? __fdiv_rn(1.0f, encode_scale) : 1;
  g = fmaxf(g, 0x1p-126f);
  if (!group && !lane)
    *global_scale = g;
  float scale6 = __fdiv_rn(amax, 6.0f) * encode_scale;
  uint8_t s6 = e4_encode(scale6);
  float d6 = e4_decode(s6) * g;
  float v6 = d6 > 0 ? xi * __fdiv_rn(1.0f, d6) : 0;
  uint8_t q6 = e2_encode(v6), selected_scale = s6;
  if (four_over_six && !tensor) {
    uint8_t s4 = e4_encode(scale6 * 1.5f);
    float d4 = e4_decode(s4) * g;
    uint8_t q4 = e2_encode(d4 > 0 ? xi * __fdiv_rn(1.0f, d4) : 0);
    float e6 =
        valid ? xi - __fdiv_rn((e2_decode(q6) * e4_decode(s6)) * *amax_tensor,
                               6 * fp8_bound)
              : 0;
    float e4 =
        valid ? xi - __fdiv_rn((e2_decode(q4) * e4_decode(s4)) * *amax_tensor,
                               6 * fp8_bound)
              : 0;
    // Score the representations after E4M3 scale rounding, not ideal scales.
    if (warp_sum(e4 * e4) < warp_sum(e6 * e6)) {
      q6 = q4;
      selected_scale = s4;
      d6 = d4;
    }
  }
  if (random_rounding)
    q6 = stochastic(d6 > 0 ? xi / d6 : 0, true, random01(index, seed));
  if (bad) {
    selected_scale = 127;
    q6 = 0;
  }
  if (!lane && (!tensor || !group))
    scales[scale_index] = selected_scale;
  unsigned next = __shfl_down_sync(~0u, unsigned(q6), 1);
  // Every even lane owns one whole byte. Odd tail's high nibble is zero.
  if (valid && !(lane & 1))
    q[int64_t(row) * ((cols + 1) / 2) + col / 2] =
        q6 | ((col + 1 < cols ? next : 0) << 4);
}

__global__ void dequantize_naive(const uint8_t *q, const uint8_t *scales,
                                 const float *global_scale, void *y, int rows,
                                 int cols, int fp4, int dtype, int tensor) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= int64_t(rows) * cols)
    return;
  int row = i / cols, col = i % cols, size = fp4 ? 16 : 32;
  int si = tensor ? 0 : row * ((cols + size - 1) / size) + col / size;
  uint8_t s = scales[si];
  float value;
  if (fp4) {
    uint8_t packed = q[int64_t(row) * ((cols + 1) / 2) + col / 2];
    uint8_t nibble = (packed >> ((col & 1) * 4)) & 15;
    value = (e2_decode(nibble) * e4_decode(s)) * *global_scale;
  } else
    value = s == 255 ? nanf("") : ldexpf(e4_decode(q[i]), int(s) - 127);
  store_value(y, i, dtype, value);
}

extern "C" int quantize(const void *x, uint8_t *q, uint8_t *s, const float *a,
                        float *g, int rows, int cols, int fp4, int dtype,
                        int tensor, int sr, unsigned seed, int rceil, int four6,
                        float bound, void *stream) {
  int block = fp4 ? 16 : 32, groups = rows * ((cols + block - 1) / block);
  quantize_naive<<<groups, 32, 0, static_cast<cudaStream_t>(stream)>>>(
      x, q, s, a, g, rows, cols, fp4, dtype, tensor, sr, seed, rceil, four6,
      bound);
  return int(cudaGetLastError());
}
extern "C" int dequantize(const uint8_t *q, const uint8_t *s, const float *g,
                          void *y, int rows, int cols, int fp4, int dtype,
                          int tensor, void *stream) {
  int64_t count = int64_t(rows) * cols;
  dequantize_naive<<<(count + 255) / 256, 256, 0,
                     static_cast<cudaStream_t>(stream)>>>(
      q, s, g, y, rows, cols, fp4, dtype, tensor);
  return int(cudaGetLastError());
}

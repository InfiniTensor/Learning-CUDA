// Optional CUDA Math API variant. Group-32 MXFP8, FLOOR, finite FP32 input.
#include <cmath>
#include <cstdint>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

__global__ void compress_native(const float *x, __nv_fp8_e4m3 *q, uint8_t *s,
                                int rows, int cols) {
  int lane = threadIdx.x, groups = (cols + 31) / 32, group = blockIdx.x;
  int row = group / groups, col = (group % groups) * 32 + lane;
  float v = col < cols ? x[row * cols + col] : 0, a = fabsf(v);
  bool bad = __ballot_sync(~0u, !isfinite(v)) != 0;
  for (int d = 16; d; d >>= 1)
    a = fmaxf(a, __shfl_down_sync(~0u, a, d));
  int e = a > 0 && !bad ? max(-127, min(127, ilogbf(a) - 8)) : -127;
  e = __shfl_sync(~0u, e, 0);
  if (!lane)
    s[group] = bad ? 255 : e + 127;
  if (col < cols)
    q[row * cols + col] = __nv_fp8_e4m3(bad ? nanf("") : ldexpf(v, -e));
}
__global__ void decompress_native(const __nv_fp8_e4m3 *q, const uint8_t *s,
                                  float *y, int rows, int cols) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * cols)
    return;
  int row = i / cols, col = i % cols;
  uint8_t code = s[row * ((cols + 31) / 32) + col / 32];
  y[i] = code == 255 ? nanf("") : ldexpf(float(q[i]), int(code) - 127);
}
extern "C" void solve(const float *x, unsigned char *q, unsigned char *s, int M,
                      int N) {
  compress_native<<<M *((N + 31) / 32), 32>>>(
      x, reinterpret_cast<__nv_fp8_e4m3 *>(q), s, M, N);
}
extern "C" void solve_decompress(const unsigned char *q, const unsigned char *s,
                                 float *y, int M, int N) {
  decompress_native<<<(M * N + 255) / 256, 256>>>(
      reinterpret_cast<const __nv_fp8_e4m3 *>(q), s, y, M, N);
}

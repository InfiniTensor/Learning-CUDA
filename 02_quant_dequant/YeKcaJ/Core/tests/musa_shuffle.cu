// 验证 MUSA SDK 的逻辑子组 shuffle，不能只根据设备报告的 warpSize 推断语义。
#include "runtime/cuda_utils.cuh"
#include <algorithm>
#include <iostream>
#include <vector>

template <unsigned Width>
__global__ void shuffle_probe(float* result, unsigned* source, unsigned n) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  float value = i < n ? float((i * 73u) % 997u) : 0.0f;
  for (unsigned step = Width / 2; step; step >>= 1)
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, step, Width));
  // 全部线程参与，包含尾部填零线程；跨 32/128/256 边界检查子组隔离。
  result[i] = __shfl_sync(0xffffffffu, value, 0, Width);
  source[i] = __shfl_sync(0xffffffffu, i, 0, Width);
}

template <unsigned Width>
void check() {
  using namespace pipeline;
  constexpr unsigned count = 768;
  DeviceBuffer<float> result(count);
  DeviceBuffer<unsigned> source(count);
  std::vector<float> got(count);
  std::vector<unsigned> ids(count);
  for (unsigned n : {0u, 1u, 31u, 32u, 33u, 127u, 128u, 129u, 255u, 256u, 257u, 767u, 768u}) {
    shuffle_probe<Width><<<3, 256>>>(result.ptr, source.ptr, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(got.data(), result.ptr, count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ids.data(), source.ptr, count * sizeof(unsigned), cudaMemcpyDeviceToHost));
    for (unsigned i = 0; i < count; ++i) {
      const unsigned start = i / Width * Width;
      float expected = 0;
      for (unsigned j = start; j < start + Width && j < n; ++j)
        expected = std::max(expected, float((j * 73u) % 997u));
      if (got[i] != expected || ids[i] != start)
        throw std::runtime_error("shuffle mismatch: width=" + std::to_string(Width) +
                                 " n=" + std::to_string(n) + " i=" + std::to_string(i));
    }
  }
}

int main() {
  try {
    check<4>(); check<8>(); check<16>(); check<32>();
    std::cout << "musa_shuffle_cases=52 lane_checks=39936 PASS\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

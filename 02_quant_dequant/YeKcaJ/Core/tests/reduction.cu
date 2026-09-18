// 独立检查全局归约，不通过相同归约实现生成期望值。
#include "kernels/reduce.cuh"
#include "runtime/cuda_utils.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

int main() {
  using namespace pipeline;
  try {
    unsigned cases = 0;
    for (std::size_t n : {1u, 31u, 32u, 33u, 255u, 256u, 257u, 2049u,
                          65537u, 1048579u, 4194307u}) {
      const unsigned blocks = static_cast<unsigned>(std::min<std::size_t>(4096, ceil_div(n, 256)));
      DeviceBuffer<float> input(n), partial(blocks), state(2);
      // 将唯一峰值放在 warp 边界、最后一个 warp，以及跨步扫描的尾部。
      for (std::size_t peak : {std::size_t(0), std::min(n - 1, std::size_t(31)),
                               std::min(n - 1, std::size_t(255)), n - 1}) {
        for (float value : {0.0f, -0.0f, -17.0f, std::numeric_limits<float>::denorm_min(),
                            std::numeric_limits<float>::max()}) {
          std::vector<float> x(n, 0.0f), got(blocks), expected(blocks, 0.0f);
          x[peak] = value;
          float expected_max = 0;
          for (std::size_t i = 0; i < n; ++i) {
            const float magnitude = std::fabs(x[i]);
            expected[(i / 256) % blocks] = std::max(expected[(i / 256) % blocks], magnitude);
            expected_max = std::max(expected_max, magnitude);
          }
          CUDA_CHECK(cudaMemcpy(input.ptr, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
          maximum<<<blocks, 256>>>(input.ptr, partial.ptr, n);
          finalize_max<<<1, 256>>>(partial.ptr, state.ptr, blocks);
          CUDA_CHECK(cudaGetLastError());
          CUDA_CHECK(cudaMemcpy(got.data(), partial.ptr, blocks * sizeof(float), cudaMemcpyDeviceToHost));
          float actual[2];
          CUDA_CHECK(cudaMemcpy(actual, state.ptr, sizeof(actual), cudaMemcpyDeviceToHost));
          const float expected_scale = expected_max > 0
              ? std::max(expected_max / 2688.0f, std::numeric_limits<float>::denorm_min()) : 1.0f;
          if (got != expected || actual[0] != expected_max ||
              std::memcmp(&actual[1], &expected_scale, sizeof(float)))
            throw std::runtime_error("reduction mismatch: n=" + std::to_string(n));
          ++cases;
        }
      }
    }
    std::cout << "reduction_cases=" << cases << " PASS\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

// 将设备 FP32 运算与独立主机运算逐位比较，重点覆盖 MUSA 次正规数兼容路径。
#include "common/device_math.cuh"
#include "runtime/cuda_utils.cuh"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

__global__ void arithmetic_probe(const float* a, const float* b, float* y, std::size_t n) {
  const std::size_t i = blockIdx.x * 256ull + threadIdx.x;
  if (i < n) {
    y[2 * i] = pipeline::divide_rn(a[i], b[i]);
    y[2 * i + 1] = pipeline::multiply_rn(a[i], b[i]);
  }
}

int main() {
  using namespace pipeline;
  try {
    std::vector<float> a, b;
    auto from_bits = [](std::uint32_t bits) { float v; std::memcpy(&v, &bits, 4); return v; };
    const std::uint32_t edges[] = {0u, 0x80000000u, 1u, 0x80000001u, 0x007fffffu,
        0x00800000u, 0x00400000u, 0x3f800000u, 0x40000000u, 0x7f7fffffu, 0xff7fffffu};
    for (auto x : edges) for (auto z : edges) { a.push_back(from_bits(x)); b.push_back(from_bits(z)); }
    // 跨指数、尾数边界、次正规数中点：除以 2 时奇数 subnormal 位模式恰在中点。
    for (unsigned e = 0; e < 255; ++e) {
      for (unsigned m : {0u, 1u, 0x3fffffu, 0x400000u, 0x7ffffeu, 0x7fffffu}) {
        for (unsigned sign : {0u, 0x80000000u}) {
          const float value = from_bits(sign | (e << 23) | m);
          for (float denominator : {0.5f, 2.0f, 3.0f, 448.0f, 2688.0f,
                 from_bits(0x3f7fffffu), from_bits(0x3f800001u),
                 from_bits(1u), from_bits(0x007fffffu), from_bits(0x00800000u)}) {
            a.push_back(value); b.push_back(denominator);
          }
        }
      }
    }
    std::mt19937 rng(9051);
    while (a.size() < 1048576) {
      const float x = from_bits(rng()), z = from_bits(rng());
      if (std::isfinite(x) && std::isfinite(z)) { a.push_back(x); b.push_back(z); }
    }
    DeviceBuffer<float> da(a.size()), db(b.size()), dy(a.size() * 2);
    CUDA_CHECK(cudaMemcpy(da.ptr, a.data(), a.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db.ptr, b.data(), b.size() * 4, cudaMemcpyHostToDevice));
    arithmetic_probe<<<static_cast<unsigned>((a.size() + 255) / 256), 256>>>(da.ptr, db.ptr, dy.ptr, a.size());
    CUDA_CHECK(cudaGetLastError());
    std::vector<float> y(a.size() * 2);
    CUDA_CHECK(cudaMemcpy(y.data(), dy.ptr, y.size() * 4, cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < a.size(); ++i) {
      volatile float x = a[i], z = b[i];
      const float expected[] = {x / z, x * z};
      for (unsigned op = 0; op < 2; ++op) {
        if (std::isnan(expected[op]) && std::isnan(y[2 * i + op])) continue;
        if (std::memcmp(&expected[op], &y[2 * i + op], 4))
          throw std::runtime_error("device math mismatch: sample=" + std::to_string(i) +
              " op=" + std::to_string(op));
      }
    }
    std::cout << "device_math_pairs=" << a.size() << " operations=" << y.size() << " PASS\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

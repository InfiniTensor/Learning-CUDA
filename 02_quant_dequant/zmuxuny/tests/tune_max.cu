#include "../include/host.hpp"
#include <random>
using namespace lp;
__global__ void copy_vector(const uint4 *x, uint4 *y, size_t n) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += size_t(blockDim.x) * gridDim.x)
    y[i] = x[i];
}
template <int TYPE, int V>
void variant(Device &x, Device &a, size_t n, float expected, int threads, int blocks,
             int trial) {
  auto f = [&] {
    check(cudaMemsetAsync(a.ptr, 0, 4));
    maximum_vector<TYPE, V><<<blocks, threads>>>(x.ptr, n, a.as<float>());
  };
  double ms = elapsed(f, n > 16000000 ? 40 : 150);
  float actual;
  a.download(&actual, 4);
  if (actual != expected)
    throw std::runtime_error("amax mismatch");
  std::cout << n << ',' << TYPE << ',' << V << ',' << threads << ',' << blocks << ','
            << trial << ',' << ms << '\n';
}
template <int TYPE> void shape(size_t n) {
  std::vector<uint8_t> h(n * width(TYPE));
  std::mt19937 gen(42);
  std::normal_distribution<float> dist;
  float expected = 0;
  for (size_t i = 0; i < n; ++i) {
    save(h.data(), i, dist(gen), TYPE);
    expected = std::max(expected, fabsf(load(h.data(), i, TYPE)));
  }
  Device x(h.size()), a(4), copy(h.size());
  x.upload(h.data(), h.size());
  for (int trial = 0; trial < 3; ++trial) {
    double ms = elapsed(
        [&] {
          check(cudaMemsetAsync(a.ptr, 0, 4));
          maximum<<<std::min(size_t(256), (n + 255) / 256), 256>>>(x.ptr, n, TYPE,
                                                                   a.as<float>());
        },
        n > 16000000 ? 40 : 150);
    std::cout << n << ',' << TYPE << ",1,256,256," << trial << ',' << ms << '\n';
    for (int threads : {128, 256, 512})
      for (int blocks : (trial & 1 ? std::vector<int>{1024, 512, 256, 128}
                                   : std::vector<int>{128, 256, 512, 1024})) {
        variant<TYPE, 4>(x, a, n, expected, threads, blocks, trial);
        variant<TYPE, 8>(x, a, n, expected, threads, blocks, trial);
      }
  }
  double sm = elapsed(
      [&] {
        copy_vector<<<1024, 256>>>(x.as<uint4>(), copy.as<uint4>(), h.size() / 16);
      },
      100);
  double ce = elapsed(
      [&] {
        check(cudaMemcpyAsync(copy.ptr, x.ptr, h.size(), cudaMemcpyDeviceToDevice));
      },
      100);
  std::cerr << "copy," << n << ',' << TYPE << ',' << sm << ',' << ce << ','
            << 2 * h.size() / sm / 1e6 << ',' << 2 * h.size() / ce / 1e6 << '\n';
}
int main() {
  try {
    std::cout << "count,dtype,vector,threads,blocks,trial,amax_with_clear_ms\n";
    for (size_t n : {size_t(1024 * 1024), size_t(32) * 1024 * 1024}) {
      shape<FP32>(n);
      shape<FP16>(n);
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

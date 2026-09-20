#include "../include/host.hpp"
#include "../src/tensor_core.cuh"
#include <random>
using namespace lp;

template <int D, int TYPE, int FMT, int W>
void run_variant(Device &x, Device &y, Device &data, Device &scales, Device &amax,
                 const std::vector<uint8_t> &reference,
                 const std::vector<uint8_t> &refdata,
                 const std::vector<uint8_t> &refscales, Layout p, int trial) {
  constexpr int ITEMS = D > 256 ? D : 256;
  size_t n = p.count(), blocks = (n + W * ITEMS - 1) / (W * ITEMS);
  int repeats = n > 16000000 ? 30 : 150;
  auto h = [&] {
    hadamard_mma_kernel<D, TYPE, 0, FMT, W>
        <<<blocks, W * MMA_LANES>>>(x.as<uint16_t>(), y.as<uint16_t>(), n, true, false, 7,
                             nullptr, nullptr, nullptr, p);
  };
  auto f = [&] {
    if constexpr (FMT == NVFP4) {
      check(cudaMemsetAsync(amax.ptr, 0, 4));
      hadamard_mma_kernel<D, TYPE, 1, FMT, W>
          <<<blocks, W * MMA_LANES>>>(x.as<uint16_t>(), nullptr, n, true, false, 7, nullptr,
                               nullptr, amax.as<float>(), p);
    }
    hadamard_mma_kernel<D, TYPE, 2, FMT, W><<<blocks, W * MMA_LANES>>>(
        x.as<uint16_t>(), nullptr, n, true, false, 7, data.as<uint8_t>(),
        scales.as<uint8_t>(), amax.as<float>(), p);
  };
  double hm = elapsed(h, repeats), fm = elapsed(f, repeats);
  std::vector<uint8_t> a(reference.size()), b(refdata.size()), c(refscales.size());
  y.download(a.data(), a.size());
  data.download(b.data(), b.size());
  scales.download(c.data(), c.size());
  if (a != reference || b != refdata || c != refscales)
    throw std::runtime_error("MMA tuning output mismatch");
  std::cout << p.rows << ',' << D << ',' << TYPE << ',' << FMT << ',' << W << ','
            << trial << ',' << hm << ',' << fm << '\n';
}
template <int D, int TYPE, int FMT> void shape(size_t rows) {
  Layout p{rows, D, TYPE, FMT, FMT == MXFP8 ? 32 : 16, false, false, 42};
  std::vector<uint16_t> host(p.count());
  std::mt19937 gen(42);
  std::normal_distribution<float> dist;
  for (size_t i = 0; i < host.size(); ++i)
    save(host.data(), i, dist(gen), TYPE);
  Device x(host.size() * 2), y(host.size() * 2), data(p.bytes()), scales(p.groups()),
      amax(4);
  x.upload(host.data(), host.size() * 2);
  launch_had_mma(x.ptr, y.ptr, rows, D, TYPE, true, false, 7);
  fused_had_mma(x.ptr, rows, D, TYPE, true, false, 7, data.as<uint8_t>(),
                scales.as<uint8_t>(), amax.as<float>(), p);
  std::vector<uint8_t> a(host.size() * 2), b(p.bytes()), c(p.groups());
  y.download(a.data(), a.size());
  data.download(b.data(), b.size());
  scales.download(c.data(), c.size());
#define RUN(W) run_variant<D, TYPE, FMT, W>(x, y, data, scales, amax, a, b, c, p, trial)
  for (int trial = 0; trial < 3; ++trial) {
    if (trial & 1) {
#if !defined(__MUSACC__)
      RUN(16);
#endif
#if !defined(__MUSACC__)
      RUN(8);
#endif
      RUN(4);
      RUN(2);
      RUN(1);
    } else {
      RUN(1);
      RUN(2);
      RUN(4);
#if !defined(__MUSACC__)
      RUN(8);
#endif
#if !defined(__MUSACC__)
      RUN(16);
#endif
    }
  }
#undef RUN
}
template <int D> void dimensions(size_t rows) {
  shape<D, FP16, MXFP8>(rows);
  shape<D, FP16, NVFP4>(rows);
  shape<D, BF16, MXFP8>(rows);
  shape<D, BF16, NVFP4>(rows);
}
int main() {
  try {
    std::cout << std::unitbuf;
    std::cout << "rows,dim,dtype,format,warps,trial,had_ms,fused_ms\n";
    for (size_t rows : {size_t(32), size_t(8192)}) {
      dimensions<64>(rows);
      dimensions<128>(rows);
      dimensions<256>(rows);
      dimensions<512>(rows);
      dimensions<1024>(rows);
    }
    dimensions<1024>(65536);
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

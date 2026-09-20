#include "../include/host.hpp"
#include "../include/vector.cuh"
#include <random>
using namespace lp;
template <int FMT, int TYPE, int V>
void variant(Device &in, Device &data, Device &scales, Device &amax, Layout p,
             const std::vector<uint8_t> &ref, const std::vector<uint8_t> &refs,
             int threads, int trial) {
  auto f = [&] {
    quant_vector_kernel<FMT, TYPE, V>
        <<<(p.count() + threads * V - 1) / (threads * V), threads>>>(
            in.ptr, data.as<uint8_t>(), scales.as<uint8_t>(), amax.as<float>(), p);
  };
  double ms = elapsed(f, p.count() > 16000000 ? 40 : 200);
  std::vector<uint8_t> a(ref.size()), b(refs.size());
  data.download(a.data(), a.size());
  scales.download(b.data(), b.size());
  if (a != ref || b != refs)
    throw std::runtime_error("Vector quantization mismatch");
  std::cout << p.rows << ',' << TYPE << ',' << FMT << ',' << V << ',' << threads << ','
            << trial << ',' << ms << '\n';
}
template <int FMT, int TYPE> void shape(size_t rows) {
  Layout p{rows, 1024, TYPE, FMT, FMT == MXFP8 ? 32 : 16, false, false, 42};
  std::vector<uint8_t> host(p.count() * width(TYPE));
  std::mt19937 gen(42);
  std::normal_distribution<float> dist;
  for (size_t i = 0; i < p.count(); ++i)
    save(host.data(), i, dist(gen), TYPE);
  Device in(host.size()), data(p.bytes()), scales(p.groups()), amax(4);
  in.upload(host.data(), host.size());
  quantize(in.ptr, data.as<uint8_t>(), scales.as<uint8_t>(), amax.as<float>(), p);
  std::vector<uint8_t> ref(p.bytes()), refs(p.groups());
  data.download(ref.data(), ref.size());
  scales.download(refs.data(), refs.size());
  for (int trial = 0; trial < 3; ++trial) {
    auto base = [&] {
      quant_blocks_kernel<FMT, true><<<(p.count() + 255) / 256, 256>>>(
          in.ptr, data.as<uint8_t>(), scales.as<uint8_t>(), amax.as<float>(), p);
    };
    double ms = elapsed(base, p.count() > 16000000 ? 40 : 200);
    std::cout << rows << ',' << TYPE << ',' << FMT << ",1,256," << trial << ',' << ms
              << '\n';
    for (int threads : (trial & 1 ? std::vector<int>{512, 256, 128}
                                  : std::vector<int>{128, 256, 512})) {
      variant<FMT, TYPE, 2>(in, data, scales, amax, p, ref, refs, threads, trial);
      variant<FMT, TYPE, 4>(in, data, scales, amax, p, ref, refs, threads, trial);
      variant<FMT, TYPE, 8>(in, data, scales, amax, p, ref, refs, threads, trial);
    }
  }
}
int main() {
  try {
    std::cout << "rows,dtype,format,vector,threads,trial,kernel_ms\n";
    for (size_t rows : {size_t(32), size_t(1024), size_t(32768)}) {
      shape<MXFP8, FP32>(rows);
      shape<NVFP4, FP32>(rows);
      shape<MXFP8, FP16>(rows);
      shape<NVFP4, FP16>(rows);
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

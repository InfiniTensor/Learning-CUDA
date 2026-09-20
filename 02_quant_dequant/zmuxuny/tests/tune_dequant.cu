#include "../include/host.hpp"
#include <random>
using namespace lp;
template <int FMT, int TYPE, int V>
void variant(Device &data, Device &scales, Device &out, float global, Layout p,
             const std::vector<uint8_t> &ref, int threads, int trial) {
  auto f = [&] {
    dequant_vector_kernel<FMT, TYPE, V>
        <<<(p.count() + threads * V - 1) / (threads * V), threads>>>(
            data.as<uint8_t>(), scales.as<uint8_t>(), global, out.ptr, p);
  };
  double ms = elapsed(f, p.count() > 16000000 ? 40 : 200);
  std::vector<uint8_t> a(ref.size());
  out.download(a.data(), a.size());
  if (a != ref)
    throw std::runtime_error("Vector dequantization mismatch");
  std::cout << p.rows << ',' << TYPE << ',' << FMT << ',' << V << ',' << threads << ','
            << trial << ',' << ms << '\n';
}
template <int FMT, int TYPE> void shape(size_t rows) {
  Layout p{rows, 1024, FP32, FMT, FMT == MXFP8 ? 32 : 16, false, false, 42};
  std::vector<float> host(p.count());
  std::mt19937 gen(42);
  std::normal_distribution<float> dist;
  for (float &v : host)
    v = dist(gen);
  Device in(host.size() * 4), data(p.bytes()), scales(p.groups()), amax(4),
      out(p.count() * width(TYPE));
  in.upload(host.data(), host.size() * 4);
  quantize(in.ptr, data.as<uint8_t>(), scales.as<uint8_t>(), amax.as<float>(), p);
  float m;
  amax.download(&m, 4);
  float global = FMT == NVFP4 ? global_scale(m) : 1;
  dequantize(data.as<uint8_t>(), scales.as<uint8_t>(), global, out.ptr, TYPE, p);
  std::vector<uint8_t> ref(p.count() * width(TYPE));
  out.download(ref.data(), ref.size());
  for (int trial = 0; trial < 3; ++trial) {
    auto base = [&] {
      dequant_blocks_kernel<FMT, true><<<(p.bytes() + 255) / 256, 256>>>(
          data.as<uint8_t>(), scales.as<uint8_t>(), global, out.ptr, TYPE, p);
    };
    double ms = elapsed(base, p.count() > 16000000 ? 40 : 200);
    std::cout << rows << ',' << TYPE << ',' << FMT << ",1,256," << trial << ',' << ms
              << '\n';
    for (int threads : (trial & 1 ? std::vector<int>{512, 256, 128}
                                  : std::vector<int>{128, 256, 512})) {
      variant<FMT, TYPE, 4>(data, scales, out, global, p, ref, threads, trial);
      variant<FMT, TYPE, 8>(data, scales, out, global, p, ref, threads, trial);
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
      shape<MXFP8, BF16>(rows);
      shape<NVFP4, BF16>(rows);
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

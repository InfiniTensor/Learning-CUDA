#include "../include/host.hpp"
#include <limits>
using namespace lp;
__global__ void convert(const float *x, uint16_t *y, unsigned *packed, size_t n) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = bf16(x[i]);
  if (2 * i + 1 < n)
    packed[i] = pack_pair<BF16>(x[2 * i], x[2 * i + 1]);
}
int main() {
  try {
    std::vector<float> x;
    for (unsigned hi = 0; hi < 65536; ++hi) {
      if ((hi & 0x7f80) == 0x7f80)
        continue;
      for (unsigned lo : {0U, 0x7fffU, 0x8000U, 0x8001U, 0xffffU}) {
        union {
          unsigned u;
          float f;
        } v{(hi << 16) | lo};
        x.push_back(v.f);
      }
    }
    if (x.size() & 1)
      x.push_back(0);
    Device in(x.size() * 4), out(x.size() * 2), packed(x.size() * 2);
    in.upload(x.data(), x.size() * 4);
    convert<<<(x.size() + 255) / 256, 256>>>(in.as<float>(), out.as<uint16_t>(),
                                             packed.as<unsigned>(), x.size());
    std::vector<uint16_t> a(x.size()), b(x.size());
    out.download(a.data(), a.size() * 2);
    packed.download(b.data(), b.size() * 2);
    for (size_t i = 0; i < x.size(); ++i)
      if (a[i] != bf16(x[i]) || b[i] != bf16(x[i]))
        throw std::runtime_error("BF16 native/host mismatch");
    std::cout << "{\"status\":\"PASS\",\"float32_patterns\":" << x.size()
              << ",\"scalar_and_pair_equal\":true}\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

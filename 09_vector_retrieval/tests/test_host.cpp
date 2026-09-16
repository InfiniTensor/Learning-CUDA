#include "vsearch/binary_io.hpp"
#include "vsearch/config.hpp"
#include "vsearch/cpu_reference.hpp"
#include "vsearch/metric.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

// 极简断言计数器：打印 PASS/FAIL 并累计失败数，main 返回非零表示有失败。
int failures = 0;

void check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) ++failures;
}

}  // namespace

// 主机侧单元测试：覆盖
//   1) fp32 向量文件写入/读回一致；
//   2) fp16 round-trip 在容差内；
//   3) 参数文件解析；
//   4) CPU 参考的确定性 tie-break 与 L2 距离语义。
// 这些测试不依赖 GPU，可在任何平台先验证主机侧逻辑。
int main() {
  using namespace vs;
  const std::string base = "";
  const i64 n = 300;
  const int dim = 8;
  FloatVec data(static_cast<size_t>(n) * dim);
  unsigned s = 77u;
  for (auto& v : data) {
    s = s * 1664525u + 1013904223u;
    v = static_cast<float>((s >> 8) % 20000) / 1000.f - 10.f;
  }

  const std::string fp32Path = base + "test_fp32.bin";
  writeVectorFile(fp32Path, n, dim, DataType::Fp32, Metric::L2, data);
  Dataset back;
  readVectorFile(fp32Path, &back);
  bool ioOk = back.n == n && back.dim == dim && back.metric == Metric::L2;
  if (ioOk) {
    for (size_t i = 0; i < data.size(); ++i)
      if (back.data[i] != data[i]) {
        ioOk = false;
        break;
      }
  }
  check(ioOk, "fp32 file round-trip");

  const std::string fp16Path = base + "test_fp16.bin";
  writeVectorFile(fp16Path, n, dim, DataType::Fp16, Metric::InnerProduct, data);
  Dataset back16;
  readVectorFile(fp16Path, &back16);
  double maxErr = 0.0;
  for (size_t i = 0; i < data.size(); ++i)
    maxErr = std::max(
        maxErr, std::fabs(static_cast<double>(back16.data[i]) - data[i]));
  check(maxErr < 1e-2 && back16.dtype == DataType::Fp16,
        "fp16 round-trip within tolerance");

  const std::string cfgText =
      "top_k = 50\n"
      "search_mode = \"ivf_pq\"\n"
      "batch_size = 128\n"
      "nlist = 2048\n"
      "nprobe = 32\n"
      "pq_m = 8\n";
  const SearchConfig c = parseParamString(cfgText);
  check(c.top_k == 50 && c.search_mode == "ivf_pq" && c.nlist == 2048 &&
            c.nprobe == 32 && c.pq_m == 8,
        "parameter parser");

  // CPU top-K tie behavior: duplicated vectors resolve by id ascending.
  Dataset dup;
  dup.n = 4;
  dup.dim = 2;
  dup.metric = Metric::L2;
  const float rows[] = {1, 2, 1, 2, 5, 5, 1, 2};
  dup.data.assign(rows, rows + 8);
  const float q[] = {1, 2};
  std::vector<i32> ids;
  FloatVec scores;
  cpuExactSearch(dup, 1, q, 3, &ids, &scores);
  check(ids.size() == 3 && ids[0] == 0 && ids[1] == 1 && ids[2] == 3,
        "CPU reference deterministic tie ordering");
  check(std::fabs(scores[0] - 0.f) < 1e-6 &&
            std::fabs(scores[2] - 0.f) < 1e-6,
        "CPU reference L2 distances");

  std::printf("%s\n", failures == 0 ? "ALL HOST TESTS PASSED"
                                    : "HOST TESTS FAILED");
  return failures == 0 ? 0 : 1;
}

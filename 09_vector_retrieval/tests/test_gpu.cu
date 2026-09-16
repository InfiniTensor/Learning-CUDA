#include "vsearch/binary_io.hpp"
#include "vsearch/common.hpp"
#include "vsearch/cpu_reference.hpp"
#include "vsearch/engine.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// 断言计数器：任何一项失败都会让进程返回非零，便于 CI / 脚本判断。
int failures = 0;

// CoreX (Iluvatar) devices emulate FP64 with reduced mantissa width, so the
// device scoring path intentionally accumulates in float. Both orders and IDs
// must still match the CPU oracle; the value tolerance is relaxed to float
// precision (~1e-3) instead of the strict 1e-4 used on NVIDIA.
#ifdef VSEARCH_COREX
constexpr double kScoreTol = 2e-3;
#else
constexpr double kScoreTol = 1e-4;
#endif

// 分平台的距离容差：NVIDIA 上 GPU 与 CPU 都用 double 累计，可收紧到 1e-4；
// CoreX 的 device double 精度不足、打分路径改用 float，故放宽到 2e-3。
// 无论哪种平台，id 都必须与 CPU reference 逐位一致。
void check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) ++failures;
}

}  // namespace

// GPU 集成测试：在分簇合成数据上验证
//   1) exact 的 id/分数与 CPU 参考一致；
//   2) IVF-Flat 相对 exact 的 recall 达标；
//   3) IVF-Flat 索引保存->加载后结果可复现（round-trip）；
//   4) IVF-PQ 返回合法向量 id。
// 这是发布前在两个平台上都必须通过的核心正确性回归。
int main() {
  using namespace vs;
  try {
    constexpr i64 n = 20000;
    constexpr int dim = 64;
    constexpr i64 nq = 128;
    constexpr int topK = 20;

    FloatVec data(static_cast<size_t>(n) * dim);
    FloatVec queries(static_cast<size_t>(nq) * dim);
    unsigned s = 20260701u;
    auto next = [&]() {
      s = s * 1664525u + 1013904223u;
      return static_cast<float>((s >> 8) & 0xffffff) /
             static_cast<float>(0x1000000);
    };
    // 100 well separated clusters make IVF recall reproducible.
    const int nCluster = 100;
    std::vector<float> ctr(nCluster * dim);
    for (auto& v : ctr) v = next() * 100.f - 50.f;
    for (i64 i = 0; i < n; ++i) {
      const int c = static_cast<int>(i % nCluster);
      float* row = data.data() + i * dim;
      for (int j = 0; j < dim; ++j)
        row[j] = ctr[static_cast<size_t>(c) * dim + j] + (next() - 0.5f) * 2.f;
    }
    for (i64 i = 0; i < nq; ++i) {
      const i64 base = (i * 37) % n;
      float* row = queries.data() + i * dim;
      for (int j = 0; j < dim; ++j)
        row[j] = data[static_cast<size_t>(base) * dim + j] + (next() - 0.5f) * 0.1f;
    }

    Dataset ds;
    ds.n = n;
    ds.dim = dim;
    ds.metric = Metric::L2;
    ds.data = std::move(data);

    std::vector<i32> cpuIds;
    FloatVec cpuScores;
    cpuExactSearch(ds, nq, queries.data(), topK, &cpuIds, &cpuScores);

    SearchConfig cfg;
    cfg.top_k = topK;
    cfg.batch_size = 32;
    cfg.nlist = 256;
    cfg.nprobe = 8;
    cfg.kmeans_sample = 4096;
    cfg.kmeans_iters = 6;
    GpuEngine gpu(ds, cfg);

    std::vector<i32> exactIds;
    FloatVec exactScores;
    SearchStats st;
    gpu.searchAll(SearchMode::Exact, queries.data(), nq, topK, &exactIds,
                  &exactScores, &st);
    bool exactOk = true;
    int printed = 0;
    for (size_t i = 0; i < exactIds.size(); ++i) {
      if (exactIds[i] != cpuIds[i] ||
          std::fabs(static_cast<double>(exactScores[i] - cpuScores[i])) >
              kScoreTol) {
        exactOk = false;
        if (printed < 8) {
          const i64 qi = static_cast<i64>(i / topK);
          const int k = static_cast<int>(i % topK);
          std::printf("mismatch q=%lld k=%d gpu=(%d,%.7g) cpu=(%d,%.7g)\n",
                      static_cast<long long>(qi), k, exactIds[i],
                      static_cast<double>(exactScores[i]), cpuIds[i],
                      static_cast<double>(cpuScores[i]));
          ++printed;
        }
      }
    }
    check(exactOk, "GPU exact matches CPU reference");

    const IndexBuildStats bs = gpu.buildIndex(SearchMode::IvfFlat);
    std::vector<i32> ivfIds;
    FloatVec ivfScores;
    SearchStats ivfSt;
    gpu.searchAll(SearchMode::IvfFlat, queries.data(), nq, topK, &ivfIds,
                  &ivfScores, &ivfSt);
    int matched = 0;
    for (i64 qi = 0; qi < nq; ++qi)
      for (int k = 0; k < topK; ++k)
        if (exactIds[static_cast<size_t>(qi) * topK + k] ==
            ivfIds[static_cast<size_t>(qi) * topK + k])
          ++matched;
    const double recall = static_cast<double>(matched) /
                          static_cast<double>(nq * topK);
    std::printf("ivf_flat build_ms=%.1f recall=%.4f\n", bs.buildMs, recall);
    check(recall > 0.95, "IVF-Flat recall on clustered synthetic data");

    // Index save/load round trip must reproduce results exactly.
    const std::string indexPath = "test_index_tmp.idx";
    gpu.saveIndex(indexPath);
    GpuEngine loaded(ds, cfg);
    loaded.loadIndex(indexPath);
    std::vector<i32> reloadIds;
    FloatVec reloadScores;
    SearchStats rst;
    loaded.searchAll(SearchMode::IvfFlat, queries.data(), nq, topK,
                     &reloadIds, &reloadScores, &rst);
    check(reloadIds == ivfIds, "loaded IVF-Flat index reproduces search");
    std::printf("mem before pq build: free=%llu\n",
                static_cast<unsigned long long>(gpu.deviceFreeBytes()));

    cfg.nprobe = 256;  // probe all lists => IVF-PQ itself must be self-consistent
    cfg.pq_m = 16;
    std::printf("building IVF-PQ...\n");
    GpuEngine pqGpu(ds, cfg);
    const IndexBuildStats pbs = pqGpu.buildIndex(SearchMode::IvfPq);
    std::printf("pq built build_ms=%.1f\n", pbs.buildMs);
    std::vector<i32> pqIds;
    FloatVec pqScores;
    SearchStats pqSt;
    pqGpu.searchAll(SearchMode::IvfPq, queries.data(), nq, topK, &pqIds,
                    &pqScores, &pqSt);
    std::printf("ivf_pq build_ms=%.1f encode_ms=%.1f first_result=(%d,%.4f)\n",
                pbs.buildMs, pbs.pqEncodeMs, pqIds.empty() ? -1 : pqIds[0],
                pqScores.empty() ? 0.f : pqScores[0]);
    bool finiteOk = pqIds.size() == static_cast<size_t>(nq) * topK;
    for (int v : pqIds)
      if (v < 0 || v >= n) finiteOk = false;
    check(finiteOk, "IVF-PQ returns valid vector ids");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "GPU test exception: %s\n", e.what());
    return 2;
  }
  std::printf("%s\n", failures == 0 ? "ALL GPU TESTS PASSED"
                                    : "GPU TESTS FAILED");
  return failures == 0 ? 0 : 1;
}

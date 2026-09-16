#include "vsearch/binary_io.hpp"
#include "vsearch/config.hpp"
#include "vsearch/cpu_reference.hpp"
#include "vsearch/engine.hpp"
#include "vsearch/metric.hpp"
#include "vsearch/result_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace vs;

namespace {

struct CliArgs {
  std::string command;
  std::map<std::string, std::string> opts;
};

CliArgs parseCli(int argc, char** argv) {
  CliArgs a;
  if (argc < 2) throwRuntime("usage: vsearch <build|search|bench|validate> [--key=value ...]");
  a.command = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string s = argv[i];
    const auto eq = s.find('=');
    if (s.rfind("--", 0) != 0) throwRuntime("expected --key=value argument: " + s);
    if (eq != std::string::npos) {
      a.opts[s.substr(2, eq - 2)] = s.substr(eq + 1);
    } else {
      if (i + 1 >= argc) throwRuntime("missing value after " + s);
      a.opts[s.substr(2)] = argv[++i];
    }
  }
  return a;
}

std::string opt(const CliArgs& a, const std::string& key,
                const std::string& def = "") {
  auto it = a.opts.find(key);
  return it == a.opts.end() ? def : it->second;
}

void overrideConfig(const CliArgs& a, vs::SearchConfig* c) {
  const std::map<std::string, void*> map = {
      {"top_k", &c->top_k},       {"batch_size", &c->batch_size},
      {"nlist", &c->nlist},       {"nprobe", &c->nprobe},
      {"pq_m", &c->pq_m},         {"kmeans_iters", &c->kmeans_iters},
      {"nthreads", &c->nthreads}, {"exact_memory_mb", &c->exact_memory_mb},
  };
  for (const auto& [key, ptr] : map) {
    const auto it = a.opts.find(key);
    if (it == a.opts.end()) continue;
    if (key == "exact_memory_mb") {
      vs::i64 v = std::stoll(it->second);
      *static_cast<vs::i64*>(ptr) = v;
    } else if (key == "kmeans_iters") {
      *static_cast<int*>(ptr) = std::stoi(it->second);
    } else {
      *static_cast<int*>(ptr) = std::stoi(it->second);
    }
  }
  if (a.opts.count("search_mode")) c->search_mode = a.opts.at("search_mode");
  if (a.opts.count("index_path")) c->index_path = a.opts.at("index_path");
  if (a.opts.count("result_path")) c->result_path = a.opts.at("result_path");
  if (a.opts.count("perf_log_path")) c->perf_log_path = a.opts.at("perf_log_path");
  if (a.opts.count("quality_log_path"))
    c->quality_log_path = a.opts.at("quality_log_path");
}

vs::Dataset loadDataset(const CliArgs& a) {
  const std::string path = opt(a, "vectors");
  if (path.empty()) throwRuntime("missing --vectors");
  vs::Dataset ds;
  vs::readVectorFile(path, &ds);
  if (ds.metric == vs::Metric::Cosine) vs::cosineNormalize(&ds);
  return ds;
}

vs::Dataset loadQueries(const CliArgs& a) {
  const std::string path = opt(a, "queries");
  if (path.empty()) throwRuntime("missing --queries");
  vs::Dataset q;
  vs::readVectorFile(path, &q);
  if (q.dim <= 0) throwRuntime("query file has no dimension");
  return q;
}

std::vector<vs::SearchConfig> configFromArgs(const CliArgs& a) {
  vs::SearchConfig c;
  if (a.opts.count("params")) c = vs::parseParamFile(a.opts.at("params"));
  overrideConfig(a, &c);
  if (a.opts.count("ref_query_limit"))
    c.ref_query_limit = std::stoll(a.opts.at("ref_query_limit"));
  if (a.opts.count("force_rebuild"))
    c.force_rebuild = a.opts.at("force_rebuild") == "1";
  return {c};
}

vs::SearchResult toResult(i64 nq, int topK, vs::Metric metric,
                          const std::string& mode,
                          const std::vector<vs::i32>& ids,
                          const vs::FloatVec& scores) {
  vs::SearchResult r;
  r.nq = nq;
  r.topK = topK;
  r.metric = metric;
  r.mode = mode;
  r.ids = ids;
  r.scores = scores;
  return r;
}

vs::PerfRecord makePerf(const std::string& mode, i64 nq, i64 n, int dim,
                        int topK, const vs::SearchStats& s,
                        double cpuMs = 0.0) {
  vs::PerfRecord r;
  r.mode = mode;
  r.nq = nq;
  r.n = n;
  r.dim = dim;
  r.topK = topK;
  r.searchMs = s.wallMs;
  r.qps = s.qps;
  r.p50Ms = s.p50BatchMs;
  r.p99Ms = s.p99BatchMs;
  r.meanBatchMs = s.meanBatchMs;
  r.gpuUsedBytes = s.deviceUsedBytes;
  r.cpuExactMs = cpuMs;
  r.speedupVsCpu = cpuMs > 0.0 && s.wallMs > 0.0 ? cpuMs / s.wallMs : 0.0;
  return r;
}

// Average absolute distance difference for ids present in both result sets,
// plus the number of rank positions whose ids differ.
//
// 质量指标计算：对预测结果与 GPU exact 参考（gold）逐 rank 对比，
//   * avgError：两边都出现的同一 id 的绝对距离差均值（衡量分数一致性）；
//   * mismatchRanks：id 逐位不同的位置数（recall 由 1 - mismatch/total 推出）。
void qualityError(const vs::SearchResult& pred, const vs::SearchResult& gold,
                  double* avgError, int* mismatchRanks) {
  double sum = 0.0;
  int cnt = 0;
  int mismatch = 0;
  for (i64 qi = 0; qi < pred.nq; ++qi) {
    const std::size_t base = static_cast<std::size_t>(qi) * pred.topK;
    std::map<vs::i32, float> predMap;
    for (int k = 0; k < pred.topK; ++k)
      predMap.emplace(pred.ids[base + k], pred.scores[base + k]);
    for (int k = 0; k < pred.topK; ++k) {
      const auto it = predMap.find(gold.ids[base + k]);
      if (it != predMap.end()) {
        sum += std::fabs(static_cast<double>(it->second - gold.scores[base + k]));
        ++cnt;
      }
      if (pred.ids[base + k] != gold.ids[base + k]) ++mismatch;
    }
  }
  *avgError = cnt ? sum / cnt : 0.0;
  *mismatchRanks = mismatch;
}

// 自检命令：在内存构造的小规模数据上验证
//   1) GPU exact 与 CPU 参考的 id 完全一致、分数在容差内；
//   2) IVF-Flat（nprobe=全部桶）recall 正常、索引可构建。
// 不需要任何输入文件，适合在陌生环境快速确认 GPU 与工具链可用。
int validateCommand(const vs::Dataset& ds, i64 nq, const float* queries,
                    int topK, int metricCode) {
  vs::SearchConfig cfg;
  cfg.top_k = topK;
  cfg.batch_size = 32;
  cfg.nlist = 128;
  cfg.nprobe = 128;
  cfg.kmeans_sample = 3000;
  cfg.kmeans_iters = 6;
  vs::GpuEngine gpu(ds, cfg);

  std::vector<vs::i32> gpuIds;
  vs::FloatVec gpuScores;
  vs::SearchStats st;
  gpu.searchAll(vs::SearchMode::Exact, queries, nq, topK, &gpuIds,
                &gpuScores, &st);

  std::vector<vs::i32> cpuIds;
  vs::FloatVec cpuScores;
  vs::cpuExactSearch(ds, nq, queries, topK, &cpuIds, &cpuScores);

  int bad = 0;
  for (i64 qi = 0; qi < nq; ++qi)
    for (int k = 0; k < topK; ++k) {
      const size_t off = static_cast<size_t>(qi) * topK + k;
      const bool idOk = gpuIds[off] == cpuIds[off];
      // 分数按相对误差比较：id 必须逐位一致，分数允许浮点级偏差。
      const bool scoreOk =
          std::fabs(static_cast<double>(gpuScores[off] - cpuScores[off])) <
          1e-4 *
          std::max(1.0, std::fabs(static_cast<double>(cpuScores[off])));
      if (!idOk || !scoreOk) {
        ++bad;
        std::fprintf(stderr, "mismatch q=%lld k=%d gpu=(%d,%.7g) cpu=(%d,%.7g)\n",
                     static_cast<long long>(qi), k, gpuIds[off],
                     static_cast<double>(gpuScores[off]), cpuIds[off],
                     static_cast<double>(cpuScores[off]));
      }
    }

  auto stats = gpu.buildIndex(vs::SearchMode::IvfFlat);
  std::vector<vs::i32> ivfIds;
  vs::FloatVec ivfScores;
  vs::SearchStats ivfSt;
  gpu.searchAll(vs::SearchMode::IvfFlat, queries, nq, topK, &ivfIds,
                &ivfScores, &ivfSt);
  int matched = 0;
  for (i64 qi = 0; qi < nq; ++qi)
    for (int k = 0; k < topK; ++k)
      if (gpuIds[static_cast<size_t>(qi) * topK + k] ==
          ivfIds[static_cast<size_t>(qi) * topK + k])
        ++matched;
  std::printf("exact cpu/gpu mismatches: %d\n", bad);
  std::printf("ivf_flat recall@%d: %.4f (build %.1f ms)\n", topK,
              static_cast<double>(matched) /
                  std::max(1.0, static_cast<double>(nq) * topK),
              stats.buildMs);
  (void)metricCode;
  return bad == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CliArgs a = parseCli(argc, argv);
    std::vector<vs::SearchConfig> cfgs = configFromArgs(a);
    const std::string cmd = a.command;

    // `validate`：完全离线自检，不读取任何数据文件（见 validateCommand）。
    if (cmd == "validate") {
      // Deterministic in-memory dataset; no files required.
      constexpr i64 kN = 5000;
      constexpr int kDim = 64;
      constexpr i64 kNq = 128;
      const int topK = 10;
      vs::FloatVec data(static_cast<size_t>(kN) * kDim);
      unsigned seed = 12345u;
      auto rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<float>((seed >> 8) & 0xffffff) /
               static_cast<float>(0x1000000);
      };
      for (auto& v : data) v = rnd();
      vs::FloatVec queries(static_cast<size_t>(kNq) * kDim);
      for (auto& v : queries) v = rnd();
      vs::Dataset ds;
      ds.n = kN;
      ds.dim = kDim;
      ds.metric = vs::Metric::L2;
      ds.data = std::move(data);
      return validateCommand(ds, kNq, queries.data(), topK, 0);
    }

    vs::Dataset ds = loadDataset(a);
    vs::Dataset qs = loadQueries(a);
    if (ds.metric == vs::Metric::Cosine)
      vs::cosineNormalize(&qs);
    if (qs.dim != ds.dim)
      throwRuntime("query/database dimension mismatch");
    if (qs.n <= 0) throwRuntime("empty query set");

    vs::SearchConfig cfg = cfgs.front();
    const auto mode = vs::parseSearchMode(cfg.search_mode);
    const std::string indexPath = opt(a, "index", cfg.index_path);
    const int topK = std::min(cfg.top_k, static_cast<int>(ds.n));

    // `build`：只构建并保存近似索引，不做查询。
    if (cmd == "build") {
      if (mode == vs::SearchMode::Exact)
        throwRuntime("'build' requires an approximate search mode");
      vs::GpuEngine engine(ds, cfg);
      const vs::IndexBuildStats stats = engine.buildIndex(mode);
      if (indexPath.empty()) throwRuntime("'build' requires --index=...");
      engine.saveIndex(indexPath);
      std::printf("index built: mode=%s nlist=%d build_ms=%.1f train_ms=%.1f "
                  "assign_ms=%.1f device_bytes=%zu\n",
                  vs::modeName(mode), cfg.nlist, stats.buildMs, stats.trainMs,
                  stats.assignMs, stats.deviceBytes);
      return 0;
    }

    // `search` / `bench`：先算 GPU exact 作为基准（和 recall 的 gold）。
    // bench 额外跑 CPU 单线程精确检索以获得加速比，并输出性能/质量日志。
    if (cmd == "search" || cmd == "bench") {
      const i64 nq = qs.n;
      vs::GpuEngine engine(ds, cfg);

      // Reference baseline: GPU exact (used for recall) + CPU exact on a
      // bounded subset when --bench / bench command.
      std::vector<vs::i32> exactIds;
      vs::FloatVec exactScores;
      vs::SearchStats exactStats;
      engine.searchAll(vs::SearchMode::Exact, qs.data.data(), nq, topK,
                       &exactIds, &exactScores, &exactStats);

      i64 refNq = std::min<i64>(nq, std::max<i64>(1, cfg.ref_query_limit));
      double cpuMs = 0.0;
      std::vector<vs::i32> cpuIds;
      vs::FloatVec cpuScores;
      vs::SearchStats exactRefStats;
      if (cmd == "bench") {
        // CPU 参考只跑前 ref_query_limit 个 query（默认 200），控制耗时；
        // GPU exact 用同一子集重测，保证加速比口径一致。
        cpuMs = vs::cpuExactSearch(ds, refNq, qs.data.data(), topK, &cpuIds,
                                   &cpuScores);
        if (refNq < nq)
          engine.searchAll(vs::SearchMode::Exact, qs.data.data(), refNq, topK,
                           &cpuIds, &cpuScores, &exactRefStats);
        else
          exactRefStats = exactStats;
        std::printf("cpu exact reference: nq=%lld ms=%.1f\n",
                    static_cast<long long>(refNq), cpuMs);
      }

      if (cmd == "search") {
        std::vector<vs::i32> ids;
        vs::FloatVec scores;
        vs::SearchStats stats;
        if (mode != vs::SearchMode::Exact) {
          // 优先复用已保存的索引；缺失或强制重建时才重新 buildIndex。
          if (cfg.force_rebuild || indexPath.empty() || !std::ifstream(indexPath).good()) {
            const auto bs = engine.buildIndex(mode);
            std::printf("index built: build_ms=%.1f device_bytes=%zu\n",
                        bs.buildMs, bs.deviceBytes);
            if (!indexPath.empty()) engine.saveIndex(indexPath);
          } else {
            engine.loadIndex(indexPath);
          }
        }
        engine.searchAll(mode, qs.data.data(), nq, topK, &ids, &scores,
                         &stats);
        const auto res = toResult(nq, topK, ds.metric, vs::modeName(mode), ids,
                                  scores);
        if (!cfg.result_path.empty()) vs::writeResultText(cfg.result_path, res);
        const auto perf = makePerf(vs::modeName(mode), nq, ds.n, ds.dim, topK,
                                   stats, cpuMs);
        if (!cfg.perf_log_path.empty()) {
          std::vector<vs::PerfRecord> rows{perf};
          vs::writePerfLog(cfg.perf_log_path, rows);
        }
        if (!cfg.quality_log_path.empty()) {
          // exact result comparison (search command can also emit exact mode)
          const auto gold = toResult(nq, topK, ds.metric, "exact", exactIds,
                                     exactScores);
          double avgErr = 0.0;
          int mismatch = 0;
          qualityError(res, gold, &avgErr, &mismatch);
          vs::writeQualityLog(cfg.quality_log_path, nq, topK, ds.metric, res,
                              gold, avgErr, mismatch);
        }
        return 0;
      }

      // bench: approximate quality + cross-backend speedup log.
      // gold 始终是 GPU exact 结果，近似检索的 recall/误差都以它为参照。
      const auto gold = toResult(nq, topK, ds.metric, "exact", exactIds,
                                 exactScores);
      std::vector<vs::PerfRecord> perfRows;
      const auto& gpuExactForCompare =
          cmd == "bench" ? exactRefStats : exactStats;
      perfRows.push_back(makePerf("exact_gpu",
                                  cmd == "bench" ? refNq : nq, ds.n, ds.dim,
                                  topK, gpuExactForCompare, cpuMs));
      if (mode != vs::SearchMode::Exact) {
        if (cfg.force_rebuild || indexPath.empty() || !std::ifstream(indexPath).good()) {
          const auto bs = engine.buildIndex(mode);
          if (!indexPath.empty()) engine.saveIndex(indexPath);
          std::printf("index built: mode=%s build_ms=%.1f\n",
                      vs::modeName(mode), bs.buildMs);
        } else {
          engine.loadIndex(indexPath);
        }
        std::vector<vs::i32> ids;
        vs::FloatVec scores;
        vs::SearchStats stats;
        engine.searchAll(mode, qs.data.data(), nq, topK, &ids, &scores,
                         &stats);
        const auto pred =
            toResult(nq, topK, ds.metric, vs::modeName(mode), ids, scores);
        if (!cfg.result_path.empty()) vs::writeResultText(cfg.result_path, pred);
        perfRows.push_back(makePerf(vs::modeName(mode), nq, ds.n, ds.dim, topK,
                                    stats, cpuMs));
        if (!cfg.quality_log_path.empty()) {
          double avgErr = 0.0;
          int mismatch = 0;
          qualityError(pred, gold, &avgErr, &mismatch);
          vs::writeQualityLog(cfg.quality_log_path, nq, topK, ds.metric, pred,
                              gold, avgErr, mismatch);
        }
      } else {
        if (!cfg.quality_log_path.empty())
          vs::writeQualityLog(cfg.quality_log_path, nq, topK, ds.metric, gold,
                              gold, 0.0, 0);
      }
      if (!cfg.perf_log_path.empty())
        vs::writePerfLog(cfg.perf_log_path, perfRows);
      std::printf("bench done: mode=%s nq=%lld qps=%.1f\n",
                  vs::modeName(mode), static_cast<long long>(nq),
                  perfRows.back().qps);
      return 0;
    }

    throwRuntime("unknown command: " + cmd);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "vsearch: %s\n", e.what());
    return 1;
  }
}

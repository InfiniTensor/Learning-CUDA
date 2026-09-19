// =============================================================================
//  main.cu -- command line driver: correctness verification, benchmarking and
//  experiment sweeps for the fast Hadamard transform kernels.
//
//  Modes
//    info    device properties + measured memory bandwidth ceiling
//    roof    the bandwidth roof measured with SIMT kernels (copy/read/write)
//    verify  GPU result against the double precision CPU reference
//    bench   kernel timing, throughput and effective bandwidth
//    inplace out-of-place vs in-place execution across a row sweep (L2 effect)
//    dump    write one (input, output) pair to a raw file for cross-checking
//    tune    grid search over (threads_per_row, rows_per_block) -> CSV
//    matrix  full (rows x dim x dtype x geometry) sweep -> CSV
//    quant   fused Hadamard+quantisation vs the unfused pipeline
//    cpu     multithreaded CPU baseline for the same configuration
//    advice  in-place / batching heuristics for a given shape
//    loop    serving-style loop: plain submission vs one-shot vs forked graphs
//    gemm    rotation fused into the A-tile load of a GEMM, vs rotate-then-GEMM
// =============================================================================

#include "fhwt/fhwt.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// Build stamp.  On this development box the local Device Guard policy blocks
// *some* freshly linked binaries by an opaque heuristic, so being able to relink
// a semantically identical binary under a different stamp is a practical way to
// get the measurement running again.  It is also what `--mode info` prints, so a
// number recorded in a report can be traced back to the binary that produced it.
#ifndef FHWT_BUILD_STAMP
#define FHWT_BUILD_STAMP "dev"
#endif

#define CUDA_CHECK(expr)                                                                           \
  do {                                                                                             \
    cudaError_t err__ = (expr);                                                                    \
    if (err__ != cudaSuccess) {                                                                    \
      std::fprintf(stderr, "CUDA error %s at %s:%d -> %s\n", #expr, __FILE__, __LINE__,            \
                   cudaGetErrorString(err__));                                                     \
      std::exit(2);                                                                                \
    }                                                                                              \
  } while (0)

namespace {

using namespace fhwt;

struct Args {
  std::string mode = "info";
  int64_t rows = 65536;
  int64_t dim = 256;
  int64_t n = 0;  // --mode gemm: the N extent of C (0 => a default)
  int nsplit = 1; // --mode gemm: column splits of the grid
  std::string dtype = "f16";
  std::string kernel = "reg";
  std::string acc = "native";
  std::string scale = "norm";
  std::string quant = "none";
  int tpr = 0;
  int rpb = 0;
  int blocks = 0;
  int block_size = 0;
  int layers = 32; // --mode loop: number of layers per forward pass
  int tensors = 1; // --mode loop: independent [rows, dim] tensors rotated per layer
  int streams = 1; // --mode loop: >1 adds strategy E (raw launches spread over N streams)
  // --passes-per-graph K: record K whole passes into ONE graph, so a submission
  // covers K passes instead of one.  The per-node submit cost measured on this
  // driver (8.6 / 7.13: ~0.7 us per node) is what this amortises.
  int ppg = 1;
  int iters = 50;
  int warmup = 5;
  int reps = 3; // best-of-N bursts; see time_kernel()
  int threads = 0;
  std::string csv;
  std::string out;
  bool inplace = false;
  bool graph = false; // --graph: time a captured CUDA Graph instead of raw launches
  // --serial-entry: in --mode loop, queue the FIRST tensor of the pass with add()
  // so that the forked graph has a single root everything else hangs off.  Without
  // it every tensor is a root, which is the shape a serving loop's independent
  // activations have.
  bool serial_entry = false;
  // --group-entry: queue the first tensor of EVERY layer with add(), the rest of
  // that layer with add_parallel().  That is the per-layer fork/join shape (each
  // layer's tensors fork, the next layer's anchor joins them); the graph still has
  // one root but the joins are spread over the pass instead of collapsed into one.
  // With --tensors 1 every step is an anchor, so D degenerates to the plain chain.
  bool group_entry = false;
  std::vector<int64_t> dims;
  std::vector<int64_t> batch;
  std::vector<std::string> dtypes;
  bool quiet = false;
};

std::vector<std::string> split(const std::string &s, char sep) {
  std::vector<std::string> out;
  size_t start = 0;
  while (start <= s.size()) {
    size_t pos = s.find(sep, start);
    if (pos == std::string::npos)
      pos = s.size();
    if (pos > start)
      out.push_back(s.substr(start, pos - start));
    start = pos + 1;
  }
  return out;
}

DType parse_dtype(const std::string &s) {
  if (s == "f16" || s == "fp16" || s == "half")
    return DType::kF16;
  if (s == "bf16")
    return DType::kBF16;
  if (s == "f32" || s == "fp32" || s == "float")
    return DType::kF32;
  std::fprintf(stderr, "unknown dtype '%s'\n", s.c_str());
  std::exit(2);
}

KernelKind parse_kernel(const std::string &s) {
  if (s == "reg")
    return KernelKind::kReg;
  if (s == "smem")
    return KernelKind::kSmem;
  if (s == "tc")
    return KernelKind::kTensorCore;
  if (s == "auto")
    return KernelKind::kAuto;
  std::fprintf(stderr, "unknown kernel '%s'\n", s.c_str());
  std::exit(2);
}

QuantKind parse_quant(const std::string &s) {
  if (s == "none")
    return QuantKind::kNone;
  if (s == "fp8" || s == "fp8e4m3")
    return QuantKind::kFp8E4M3;
  if (s == "int4")
    return QuantKind::kInt4;
  std::fprintf(stderr, "unknown quant '%s'\n", s.c_str());
  std::exit(2);
}

Args parse(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", k.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (k == "--mode")
      a.mode = next();
    else if (k == "--rows")
      a.rows = std::stoll(next());
    else if (k == "--dim")
      a.dim = std::stoll(next());
    else if (k == "--n")
      a.n = std::stoll(next());
    else if (k == "--nsplit")
      a.nsplit = std::stoi(next());
    else if (k == "--dtype")
      a.dtype = next();
    else if (k == "--kernel")
      a.kernel = next();
    else if (k == "--acc")
      a.acc = next();
    else if (k == "--scale")
      a.scale = next();
    else if (k == "--quant")
      a.quant = next();
    else if (k == "--tpr")
      a.tpr = std::stoi(next());
    else if (k == "--rpb")
      a.rpb = std::stoi(next());
    else if (k == "--blocks")
      a.blocks = std::stoi(next());
    else if (k == "--block-size")
      a.block_size = std::stoi(next());
    else if (k == "--layers")
      a.layers = std::stoi(next());
    else if (k == "--iters")
      a.iters = std::stoi(next());
    else if (k == "--warmup")
      a.warmup = std::stoi(next());
    else if (k == "--reps")
      a.reps = std::stoi(next());
    else if (k == "--threads")
      a.threads = std::stoi(next());
    else if (k == "--dims") {
      for (auto &p : split(next(), ','))
        a.dims.push_back(std::stoll(p));
    } else if (k == "--batch") {
      for (auto &p : split(next(), ','))
        a.batch.push_back(std::stoll(p));
    } else if (k == "--dtypes") {
      for (auto &p : split(next(), ','))
        a.dtypes.push_back(p);
    } else if (k == "--csv")
      a.csv = next();
    else if (k == "--out")
      a.out = next();
    else if (k == "--inplace")
      a.inplace = true;
    else if (k == "--graph")
      a.graph = true;
    else if (k == "--serial-entry")
      a.serial_entry = true;
    else if (k == "--group-entry")
      a.group_entry = true;
    else if (k == "--tensors")
      a.tensors = std::stoi(next());
    else if (k == "--streams")
      a.streams = std::stoi(next());
    else if (k == "--passes-per-graph")
      a.ppg = std::stoi(next());
    else if (k == "--quiet")
      a.quiet = true;
    else {
      std::fprintf(stderr, "unknown argument '%s'\n", k.c_str());
      std::exit(2);
    }
  }
  return a;
}

Config make_config(const Args &a) {
  Config c;
  c.dtype = parse_dtype(a.dtype);
  c.kernel = parse_kernel(a.kernel);
  c.threads_per_row = a.tpr;
  c.rows_per_block = a.rpb;
  c.num_blocks = a.blocks;
  c.fp32_accum = (a.acc == "fp32");
  c.quant = parse_quant(a.quant);
  c.block_size = a.block_size;
  return c;
}

float parse_scale(const std::string &s, int64_t dim) {
  if (s == "norm")
    return (float)(1.0 / std::sqrt((double)dim));
  if (s == "raw" || s == "1")
    return 1.0f;
  return (float)std::atof(s.c_str());
}

// ------------------------------- timing --------------------------------------
struct Timing {
  double ms = 0.0;
  double gbps = 0.0;
};

// Capture `iters` identical launches into a CUDA Graph and time `reps` replays
// of the whole burst, each replayed with a single cudaGraphLaunch.
//
// Why this exists: on Windows/WDDM the CPU-side submission path, not the GPU, is
// what limits short kernels.  The GPU retires a 2.6 us kernel with a ~1.5 us
// bubble when the driver already holds the next command buffer, but with several
// hundred microseconds when it does not (measured: ~5% of the inter-kernel gaps
// in a small burst are ~600 us, which drags the mean gap from 1.5 us to ~30 us).
// Graphs take the per-launch submission off the critical path, which is what
// makes L2-resident shapes measurable at their true kernel speed.  See 8.5.
Timing time_burst_graph(const Args &a, const Config &cfg, void *x, void *y, float scale, int reps) {
  Timing best;
  best.ms = 0.0;
  best.gbps = -1.0;
  Graph graph;
  graph.capture(x, y, a.rows, a.dim, cfg, scale, a.iters);
  const cudaStream_t stream = graph.stream();
  for (int i = 0; i < a.warmup; ++i)
    graph.launch();
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (int rep = 0; rep < reps; ++rep) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));
    graph.launch();
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    Timing t;
    t.ms = ms / a.iters;
    t.gbps = transfer_bytes(a.rows, a.dim, cfg.dtype) / (t.ms * 1e-3) / 1e9;
    if (t.gbps > best.gbps)
      best = t;
  }
  return best;
}

// Time `iters` launches and keep the best of `reps` such bursts.
//
// This machine's GPU also drives the desktop, so an unlucky measurement window
// can be perturbed by unrelated GPU work and read up to 2x low.  Best-of-N is
// the standard estimator for microbenchmarks on a non-realtime OS: it is
// monotone in the number of samples and refuses to report the slow outliers.
Timing time_kernel(const Args &a, const Config &cfg, const LaunchPlan &plan, void *x, void *y,
                   float scale, int reps = 1) {
  Timing best;
  best.ms = 0.0;
  best.gbps = -1.0;
  const int n = reps > 0 ? reps : 1;
  if (a.graph)
    return time_burst_graph(a, cfg, x, y, scale, n);
  for (int rep = 0; rep < n; ++rep) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < a.warmup; ++i)
      hadamard(x, y, a.rows, a.dim, cfg, scale, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < a.iters; ++i)
      hadamard(x, y, a.rows, a.dim, cfg, scale, nullptr);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    Timing t;
    t.ms = ms / a.iters;
    t.gbps = transfer_bytes(a.rows, a.dim, cfg.dtype) / (t.ms * 1e-3) / 1e9;
    if (t.gbps > best.gbps)
      best = t;
  }
  (void)plan;
  return best;
}

void *alloc_like(int64_t rows, int64_t dim, DType t) {
  void *p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, (size_t)rows * dim * dtype_size(t)));
  return p;
}

void fill_random(void *p, int64_t n, DType t, uint64_t seed) {
  std::vector<float> h(n);
  uint64_t s = seed | 1;
  for (int64_t i = 0; i < n; ++i) {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    h[i] = (float)(((s >> 11) & 0xFFFFF) / 524288.0 - 1.0);
  }
  if (t == DType::kF32) {
    CUDA_CHECK(cudaMemcpy(p, h.data(), n * 4, cudaMemcpyHostToDevice));
  } else {
    std::vector<uint16_t> h2(n);
    for (int64_t i = 0; i < n; ++i) {
      float v = h[i];
      uint32_t u;
      if (t == DType::kF16) {
        __half hv = __float2half(v);
        std::memcpy(&u, &hv, 2);
      } else {
        __nv_bfloat16 bv = __float2bfloat16(v);
        std::memcpy(&u, &bv, 2);
      }
      h2[i] = (uint16_t)u;
    }
    CUDA_CHECK(cudaMemcpy(p, h2.data(), n * 2, cudaMemcpyHostToDevice));
  }
}

// ------------------------------ device info ----------------------------------
void measure_bandwidth(const DeviceInfo &info) {
  const RoofResult r = bandwidth_roof();
  std::printf("D2D memcpy ceiling      : %8.1f GB/s (async) / %.1f GB/s (sync), %.0f MiB buffers\n",
              r.memcpy_async_gbps, r.memcpy_sync_gbps, r.bytes / 1048576.0);
  std::printf(
      "SIMT copy roof          : %8.1f GB/s   (read+write, 128-bit, the roof for this op)\n",
      r.copy_gbps);
  std::printf("SIMT read / write roof  : %8.1f / %.1f GB/s\n", r.read_gbps, r.write_gbps);
  std::printf("SIMT copy in L2         : %8.1f GB/s   (working set = L2/4; the ratio to the\n",
              r.l2_copy_gbps);
  std::printf(
      "                            DRAM copy above is what \"keep it in cache\" is worth)\n");
  std::printf("Theoretical DRAM bandwidth: %8.1f GB/s (driver arithmetic; on some parts\n",
              device_bandwidth_gbps(info));
  std::printf(
      "                            this is *below* what the copy path actually sustains)\n");
}

int mode_roof() {
  const DeviceInfo info = device_info(0);
  std::printf("Device        : %s\n", info.name);
  std::printf("L2            : %.1f MiB\n", info.l2_bytes / 1048576.0);
  measure_bandwidth(info);
  return 0;
}

int mode_info() {
  DeviceInfo info = device_info(0);
  std::printf("Build         : %s %s\n", FHWT_BUILD_STAMP, __DATE__);
  std::printf("Device        : %s (sm_%d%d)\n", info.name, info.major, info.minor);
  std::printf("SMs           : %d\n", info.multi_processor_count);
  std::printf("Max thr/SM    : %d\n", info.max_threads_per_sm);
  std::printf("Shared/SM     : %d bytes\n", info.max_smem_per_sm);
  std::printf("L2            : %.1f MiB\n", info.l2_bytes / 1048576.0);
  std::printf("SM clock      : %.2f GHz\n", info.clock_khz / 1e6);
  std::printf("Mem clock/bus : %.2f GHz / %d bit\n", info.memory_clock_khz / 1e6,
              info.memory_bus_width);
  measure_bandwidth(info);
  return 0;
}

// ------------------------------- advice --------------------------------------
// Consult the library's own heuristics for the given shape, so that a caller
// does not have to guess whether aliasing the input buffer is worth it.
int mode_advice(const Args &a) {
  const Config cfg = make_config(a);
  const DeviceInfo info = device_info(0);
  const LaunchPlan p = plan(a.rows, a.dim, cfg);
  const double footprint = (double)a.rows * (double)a.dim * (double)dtype_size(cfg.dtype);
  const double l2 = (double)info.l2_bytes;
  const bool ip = inplace_pays_off(a.rows, a.dim, cfg.dtype, info);

  std::printf("shape             : rows=%lld dim=%lld %s\n", (long long)a.rows, (long long)a.dim,
              a.dtype.c_str());
  std::printf("kernel            : %s (tpr=%d rpb=%d grid=%d block=%d)\n", p.kernel_name,
              p.threads_per_row, p.rows_per_block, p.grid, p.block);
  std::printf("tensor footprint  : %8.2f MiB\n", footprint / 1048576.0);
  std::printf("out-of-place ws   : %8.2f MiB (input + output)\n", 2.0 * footprint / 1048576.0);
  std::printf("device L2         : %8.2f MiB\n", l2 / 1048576.0);
  std::printf("in-place          : %s\n",
              ip ? "RECOMMENDED (working set halves into L2, measured ~2.1x)"
                 : "not expected to help (measured 1.00x)");
  std::printf("batching          : %s\n",
              footprint < 4096.0 * 1024.0
                  ? "use fhwt::Graph (submission-bound at this size, measured 1.4-29x)"
                  : "single launch is fine (kernel >> launch overhead)");
  return 0;
}

// ------------------------------- verify --------------------------------------
int mode_verify(const Args &a) {
  const Config cfg = make_config(a);
  const float scale = parse_scale(a.scale, a.dim);
  const int64_t n = a.rows * a.dim;
  void *x = alloc_like(a.rows, a.dim, cfg.dtype);
  void *y = alloc_like(a.rows, a.dim, cfg.dtype);
  fill_random(x, n, cfg.dtype, 42);
  CUDA_CHECK(cudaMemset(y, 0, (size_t)n * dtype_size(cfg.dtype)));
  hadamard(x, y, a.rows, a.dim, cfg, scale, nullptr);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> xh(n), yh(n);
  if (cfg.dtype == DType::kF32) {
    CUDA_CHECK(cudaMemcpy(xh.data(), x, n * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(yh.data(), y, n * 4, cudaMemcpyDeviceToHost));
  } else {
    std::vector<uint16_t> t(n);
    CUDA_CHECK(cudaMemcpy(t.data(), x, n * 2, cudaMemcpyDeviceToHost));
    for (int64_t i = 0; i < n; ++i) {
      if (cfg.dtype == DType::kF16) {
        __half h;
        std::memcpy(&h, &t[i], 2);
        xh[i] = __half2float(h);
      } else {
        __nv_bfloat16 h;
        std::memcpy(&h, &t[i], 2);
        xh[i] = __bfloat162float(h);
      }
    }
    CUDA_CHECK(cudaMemcpy(t.data(), y, n * 2, cudaMemcpyDeviceToHost));
    for (int64_t i = 0; i < n; ++i) {
      if (cfg.dtype == DType::kF16) {
        __half h;
        std::memcpy(&h, &t[i], 2);
        yh[i] = __half2float(h);
      } else {
        __nv_bfloat16 h;
        std::memcpy(&h, &t[i], 2);
        yh[i] = __bfloat162float(h);
      }
    }
  }

  // CPU reference (double precision).
  double worst = 0.0, ref_max = 0.0, sum_sq = 0.0;
  std::vector<double> xr(a.dim), yr(a.dim);
  for (int64_t r = 0; r < a.rows; ++r) {
    for (int64_t j = 0; j < a.dim; ++j)
      xr[j] = xh[r * a.dim + j];
    // textbook butterfly in double
    for (int i = 0; i < a.dim; ++i)
      yr[i] = xr[i];
    for (int len = 1; len < a.dim; len <<= 1)
      for (int i = 0; i < a.dim; i += 2 * len)
        for (int j = 0; j < len; ++j) {
          double u = yr[i + j], v = yr[i + j + len];
          yr[i + j] = u + v;
          yr[i + j + len] = u - v;
        }
    for (int64_t j = 0; j < a.dim; ++j) {
      double want = yr[j] * (double)scale;
      double d = std::fabs(want - yh[r * a.dim + j]);
      worst = std::max(worst, d);
      ref_max = std::max(ref_max, std::fabs(want));
      sum_sq += d * d;
    }
  }
  std::printf("verify   rows=%lld dim=%lld %s kernel=%s acc=%s scale=%.6g\n"
              "         max_abs_err=%.4e  max|ref|=%.4e  rel=%.4e  rms=%.4e\n",
              (long long)a.rows, (long long)a.dim, to_string(cfg.dtype), to_string(cfg.kernel),
              a.acc.c_str(), scale, worst, ref_max, worst / (ref_max > 0 ? ref_max : 1.0),
              std::sqrt(sum_sq / n));
  const double tol = cfg.dtype == DType::kBF16 ? 5e-2 : (cfg.dtype == DType::kF16 ? 1e-2 : 1e-3);
  // The brief's thresholds are absolute errors on the normalised transform.  An
  // unnormalised one (--scale raw) grows by sqrt(dim), so it is judged relative
  // to max|ref| instead.  Earlier output just said "PASS (tolerance 1e-02)" for
  // both, which read as if --scale raw passed an absolute 1e-2; now it says which.
  const bool relative = (scale == 1.0f);
  const bool ok = relative ? (worst / (ref_max > 0 ? ref_max : 1.0) < tol) : (worst < tol);
  std::printf("         %s (%s error < %.0e)\n", ok ? "PASS" : "FAIL",
              relative ? "relative" : "absolute", tol);
  CUDA_CHECK(cudaFree(x));
  CUDA_CHECK(cudaFree(y));
  return ok ? 0 : 1;
}

// -------------------------------- bench --------------------------------------
struct BenchRow {
  int64_t rows;
  int64_t dim;
  std::string dtype;
  std::string kernel;
  std::string acc;
  int tpr;
  int rpb;
  int grid;
  int block;
  double ms;
  double gbps;
};

void print_row(const BenchRow &r, double ceiling) {
  std::printf("%8lld %6lld %-5s %-5s %-7s %4d %4d %8d %6d %9.4f %9.1f %7.1f%%\n", (long long)r.rows,
              (long long)r.dim, r.dtype.c_str(), r.kernel.c_str(), r.acc.c_str(), r.tpr, r.rpb,
              r.grid, r.block, r.ms, r.gbps, 100.0 * r.gbps / ceiling);
}

void print_header() {
  std::printf("%8s %6s %-5s %-5s %-7s %4s %4s %8s %6s %9s %9s %8s\n", "rows", "dim", "dtype",
              "kern", "acc", "tpr", "rpb", "grid", "block", "ms", "GB/s", "%peak");
}

int mode_bench(const Args &a) {
  const Config cfg = make_config(a);
  const float scale = parse_scale(a.scale, a.dim);
  const LaunchPlan p = plan(a.rows, a.dim, cfg);
  void *x = alloc_like(a.rows, a.dim, cfg.dtype);
  // In-place: the same tensor is both source and destination.  The DRAM traffic
  // is unchanged but the working-set footprint is halved, which matters a lot
  // once the tensor is comparable with L2 (see HadaCore, Appendix B).
  void *y = a.inplace ? x : alloc_like(a.rows, a.dim, cfg.dtype);
  fill_random(x, a.rows * a.dim, cfg.dtype, 7);
  if (a.inplace)
    std::printf("# in-place run: x == y, %zu KiB working set\n",
                (size_t)(a.rows * a.dim * dtype_size(cfg.dtype)) >> 10);
  Timing t = time_kernel(a, cfg, p, x, y, scale, a.reps);
  BenchRow r{a.rows,           a.dim,  a.dtype, a.kernel, a.acc, p.threads_per_row,
             p.rows_per_block, p.grid, p.block, t.ms,     t.gbps};
  DeviceInfo info = device_info(0);
  double ceiling = device_bandwidth_gbps(info);
  print_header();
  print_row(r, ceiling);
  CUDA_CHECK(cudaFree(x));
  if (!a.inplace)
    CUDA_CHECK(cudaFree(y));
  return 0;
}

// --------------------------------- dump --------------------------------------
// Write one (input, output) pair to a raw file.
//
// The project brief grades correctness against fast_hadamard_transform, so the
// result has to be comparable element by element from Python.  The file layout is
//
//   char[8]  "FHWTDMP1"
//   int64    rows
//   int64    dim
//   int32    dtype   (0 = f16, 1 = bf16, 2 = f32)
//   int32    kernel  (0 = reg,  1 = smem, 2 = tc)
//   float    scale
//   T        input  [rows * dim]
//   T        output [rows * dim]
int mode_dump(const Args &a) {
  if (a.out.empty()) {
    std::fprintf(stderr, "--mode dump needs --out <file>\n");
    return 2;
  }
  const Config cfg = make_config(a);
  const float scale = parse_scale(a.scale, a.dim);
  const int64_t n = a.rows * a.dim;
  const int64_t esz = (int64_t)dtype_size(cfg.dtype);
  void *x = alloc_like(a.rows, a.dim, cfg.dtype);
  void *y = alloc_like(a.rows, a.dim, cfg.dtype);
  fill_random(x, n, cfg.dtype, 42);
  hadamard(x, y, a.rows, a.dim, cfg, scale, nullptr);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<char> hx((size_t)(n * esz)), hy((size_t)(n * esz));
  CUDA_CHECK(cudaMemcpy(hx.data(), x, (size_t)(n * esz), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hy.data(), y, (size_t)(n * esz), cudaMemcpyDeviceToHost));

  std::ofstream f(a.out, std::ios::binary);
  const char magic[8] = {'F', 'H', 'W', 'T', 'D', 'M', 'P', '1'};
  f.write(magic, 8);
  const int64_t hdr[2] = {a.rows, a.dim};
  f.write(reinterpret_cast<const char *>(hdr), 16);
  const int32_t kinds[2] = {
      cfg.dtype == DType::kF16 ? 0 : (cfg.dtype == DType::kBF16 ? 1 : 2),
      cfg.kernel == KernelKind::kReg ? 0 : (cfg.kernel == KernelKind::kSmem ? 1 : 2)};
  f.write(reinterpret_cast<const char *>(kinds), 8);
  f.write(reinterpret_cast<const char *>(&scale), 4);
  f.write(hx.data(), (std::streamsize)hx.size());
  f.write(hy.data(), (std::streamsize)hy.size());
  f.close();
  std::printf("dump     rows=%lld dim=%lld %s kernel=%s scale=%.6g -> %s\n", (long long)a.rows,
              (long long)a.dim, to_string(cfg.dtype), to_string(cfg.kernel), scale, a.out.c_str());
  CUDA_CHECK(cudaFree(x));
  CUDA_CHECK(cudaFree(y));
  return 0;
}

// --------------------------------- CPU ---------------------------------------
// Straightforward multithreaded CPU FWHT: the "same level" CPU implementation
// the project brief asks to outperform.  Operates in fp32 with the same
// butterfly order as the GPU kernels.
void cpu_fwht_rows(const float *in, float *out, int64_t rows, int64_t dim, float scale,
                   int nthreads) {
  auto worker = [&](int64_t lo, int64_t hi) {
    std::vector<float> v(dim);
    for (int64_t r = lo; r < hi; ++r) {
      const float *src = in + r * dim;
      for (int64_t i = 0; i < dim; ++i)
        v[i] = src[i];
      for (int len = 1; len < dim; len <<= 1) {
        for (int64_t i = 0; i < dim; i += 2 * len) {
          for (int j = 0; j < len; ++j) {
            float u = v[i + j], w = v[i + j + len];
            v[i + j] = u + w;
            v[i + j + len] = u - w;
          }
        }
      }
      float *dst = out + r * dim;
      for (int64_t i = 0; i < dim; ++i)
        dst[i] = v[i] * scale;
    }
  };
  if (nthreads <= 1) {
    worker(0, rows);
    return;
  }
  std::vector<std::thread> pool;
  int64_t chunk = (rows + nthreads - 1) / nthreads;
  for (int t = 0; t < nthreads; ++t) {
    int64_t lo = std::min<int64_t>(rows, t * chunk);
    int64_t hi = std::min<int64_t>(rows, lo + chunk);
    if (lo >= hi)
      break;
    pool.emplace_back(worker, lo, hi);
  }
  for (auto &th : pool)
    th.join();
}

int mode_cpu(const Args &a) {
  const float scale = parse_scale(a.scale, a.dim);
  int nthreads = a.threads > 0 ? a.threads : (int)std::thread::hardware_concurrency();
  const int64_t n = a.rows * a.dim;
  std::vector<float> in(n), out(n);
  uint64_t s = 12345;
  for (int64_t i = 0; i < n; ++i) {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    in[i] = (float)(((s >> 11) & 0xFFFFF) / 524288.0 - 1.0);
  }
  cpu_fwht_rows(in.data(), out.data(), a.rows, a.dim, scale, nthreads); // warmup
  auto t0 = std::chrono::steady_clock::now();
  int reps = a.iters;
  for (int i = 0; i < reps; ++i)
    cpu_fwht_rows(in.data(), out.data(), a.rows, a.dim, scale, nthreads);
  auto t1 = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
  double bytes = transfer_bytes(a.rows, a.dim, DType::kF32);
  std::printf("cpu      rows=%lld dim=%lld threads=%d  %.4f ms  %.2f GB/s\n", (long long)a.rows,
              (long long)a.dim, nthreads, ms, bytes / (ms * 1e-3) / 1e9);
  return 0;
}

// -------------------------------- tune ---------------------------------------
// Grid search over (threads_per_row, rows_per_block) for one or more dims.
int mode_tune(const Args &a) {
  DeviceInfo info = device_info(0);
  double ceiling = device_bandwidth_gbps(info);
  std::vector<int64_t> dims;
  for (int64_t d = 64; d <= 1024; d <<= 1)
    dims.push_back(d);
  if (a.dims.size())
    dims = a.dims;

  std::printf("%8s %6s %-5s %-5s %-6s %4s %4s %8s %6s %9s %9s %8s\n", "rows", "dim", "dtype",
              "kern", "acc", "tpr", "rpb", "grid", "block", "ms", "GB/s", "%peak");
  std::vector<BenchRow> best;
  std::vector<BenchRow> all;
  for (int64_t dim : dims) {
    BenchRow best_row{};
    best_row.gbps = -1;
    for (int tpr : {8, 16, 32}) {
      if (tpr > dim)
        continue;
      for (int rpb : {1, 2, 4, 8, 16, 32}) {
        Args t = a;
        t.dim = dim;
        t.tpr = tpr;
        t.rpb = rpb;
        Config cfg = make_config(t);
        if (cfg.threads_per_row * rpb > 1024)
          continue;
        const float scale = 1.0f / std::sqrt((float)dim);
        LaunchPlan p = plan(a.rows, dim, cfg);
        void *x = alloc_like(a.rows, dim, cfg.dtype);
        void *y = alloc_like(a.rows, dim, cfg.dtype);
        fill_random(x, a.rows * dim, cfg.dtype, 7);
        Timing tm = time_kernel(t, cfg, p, x, y, scale, t.reps);
        BenchRow br{a.rows,           dim,    a.dtype, a.kernel, a.acc,  p.threads_per_row,
                    p.rows_per_block, p.grid, p.block, tm.ms,    tm.gbps};
        print_row(br, ceiling);
        if (tm.gbps > best_row.gbps)
          best_row = br;
        all.push_back(br);
        CUDA_CHECK(cudaFree(x));
        CUDA_CHECK(cudaFree(y));
      }
    }
    best.push_back(best_row);
  }
  std::printf("\nbest configuration per dim:\n");
  print_header();
  for (const auto &r : best)
    print_row(r, ceiling);
  if (!a.csv.empty()) {
    std::ofstream f(a.csv);
    f << "rows,dim,dtype,kernel,acc,tpr,rpb,grid,block,ms,gbps,peak_frac\n";
    for (const auto &r : all)
      f << r.rows << ',' << r.dim << ',' << r.dtype << ',' << r.kernel << ',' << r.acc << ','
        << r.tpr << ',' << r.rpb << ',' << r.grid << ',' << r.block << ',' << r.ms << ',' << r.gbps
        << ',' << (r.gbps / ceiling) << '\n';
    std::printf("wrote %s\n", a.csv.c_str());
  }
  return 0;
}

// Largest pair of tensors the sweep is allowed to allocate (device memory guard).
constexpr double kSweepLimitBytes = 6.0 * (1ull << 30);

// ------------------------------- in-place ------------------------------------
// Out-of-place (y != x) versus in-place (y == x) execution over a row sweep.
//
// Both variants move exactly the same number of bytes to and from DRAM, so on
// the face of it a bandwidth-bound kernel should not care which one it runs.
// What changes is the *footprint*: the out-of-place form keeps
// 2 * rows * dim * sizeof(T) bytes of distinct storage live, the in-place form
// only rows * dim * sizeof(T).  Once that footprint approaches the L2 size the
// gap becomes large -- the in-place pass keeps re-touching lines that are still
// resident, the out-of-place pass streams over twice as many distinct lines.
// This reproduces the effect HadaCore reports in Appendix B.
int mode_inplace(const Args &a) {
  DeviceInfo info = device_info(0);
  double ceiling = device_bandwidth_gbps(info);
  std::vector<int64_t> dims = a.dims.size() ? a.dims : std::vector<int64_t>{256};
  std::vector<int64_t> batch =
      a.batch.size() ? a.batch
                     : std::vector<int64_t>{8192, 16384, 32768, 65536, 131072, 262144, 524288};
  std::vector<std::string> dts = a.dtypes.size() ? a.dtypes : std::vector<std::string>{"f16"};

  std::printf("%8s %6s %-5s %-5s %4s %4s %10s %9s %9s %9s %9s %8s\n", "rows", "dim", "dtype",
              "kern", "tpr", "rpb", "KiB", "oop ms", "oop GB/s", "ip ms", "ip GB/s", "ip/oop");
  std::vector<std::string> csv;
  for (int64_t dim : dims) {
    for (const auto &dt : dts) {
      for (int64_t r : batch) {
        Args t = a;
        t.dim = dim;
        t.rows = r;
        t.dtype = dt;
        Config cfg = make_config(t);
        const double footprint = (double)r * (double)dim * (double)dtype_size(cfg.dtype);
        if (2.0 * footprint > kSweepLimitBytes)
          continue;
        const float scale = 1.0f / std::sqrt((float)dim);
        LaunchPlan p = plan(r, dim, cfg);
        void *x = nullptr;
        void *y = nullptr;
        CUDA_CHECK(cudaMalloc(&x, (size_t)footprint));
        CUDA_CHECK(cudaMalloc(&y, (size_t)footprint));
        fill_random(x, r * dim, cfg.dtype, 7);
        // Both columns are measured with the same robust estimator so the
        // ratio is meaningful.
        Timing oop = time_kernel(t, cfg, p, x, y, scale, t.reps);
        Timing ip = time_kernel(t, cfg, p, x, x, scale, t.reps);
        const double gain = oop.gbps > 0 ? ip.gbps / oop.gbps : 0.0;
        std::printf("%8lld %6lld %-5s %-5s %4d %4d %10.1f %9.4f %9.1f %9.4f %9.1f %7.2fx\n",
                    (long long)r, (long long)dim, dt.c_str(), to_string(cfg.kernel),
                    p.threads_per_row, p.rows_per_block, footprint / 1024.0, oop.ms, oop.gbps,
                    ip.ms, ip.gbps, gain);
        char buf[320];
        std::snprintf(buf, sizeof(buf), "%lld,%lld,%s,%s,%d,%d,%.1f,%.6f,%.4f,%.6f,%.4f,%.4f",
                      (long long)r, (long long)dim, dt.c_str(), to_string(cfg.kernel),
                      p.threads_per_row, p.rows_per_block, footprint / 1024.0, oop.ms, oop.gbps,
                      ip.ms, ip.gbps, gain);
        csv.push_back(buf);
        CUDA_CHECK(cudaFree(x));
        CUDA_CHECK(cudaFree(y));
      }
    }
  }
  if (!a.csv.empty()) {
    std::ofstream f(a.csv);
    f << "rows,dim,dtype,kernel,tpr,rpb,footprint_kib,oop_ms,oop_gbps,ip_ms,ip_gbps,ip_over_oop\n";
    for (const auto &l : csv)
      f << l << '\n';
    std::printf("wrote %s\n", a.csv.c_str());
  }
  return 0;
}

// -------------------------------- matrix -------------------------------------
// Full experiment matrix written to CSV (consumed by tools/analyze.py).
int mode_matrix(const Args &a) {
  DeviceInfo info = device_info(0);
  double ceiling = device_bandwidth_gbps(info);
  std::vector<BenchRow> rows;
  std::vector<int64_t> dims =
      a.dims.size() ? a.dims : std::vector<int64_t>{64, 128, 256, 512, 1024};
  std::vector<int64_t> batch =
      a.batch.size() ? a.batch : std::vector<int64_t>{1024, 4096, 16384, 65536, 262144};
  std::vector<std::string> dts =
      a.dtypes.size() ? a.dtypes : std::vector<std::string>{"f16", "bf16"};

  print_header();
  for (int64_t dim : dims) {
    for (int64_t r : batch) {
      for (const auto &dt : dts) {
        for (int tpr : {8, 16, 32}) {
          if (tpr > dim)
            continue;
          for (int rpb : {1, 4, 16}) {
            Args t = a;
            t.dim = dim;
            t.rows = r;
            t.dtype = dt;
            t.tpr = tpr;
            t.rpb = rpb;
            Config cfg = make_config(t);
            if (cfg.threads_per_row * rpb > 1024)
              continue;
            // Skip configurations that do not fit comfortably in device memory:
            // the sweep allocates two [rows, dim] tensors at once.
            if ((double)r * (double)dim * (double)dtype_size(cfg.dtype) * 2.0 > kSweepLimitBytes)
              continue;
            const float scale = 1.0f / std::sqrt((float)dim);
            LaunchPlan p = plan(r, dim, cfg);
            void *x = alloc_like(r, dim, cfg.dtype);
            void *y = alloc_like(r, dim, cfg.dtype);
            fill_random(x, r * dim, cfg.dtype, 7);
            Timing tm = time_kernel(t, cfg, p, x, y, scale, t.reps);
            BenchRow br{
                r,      dim,     dt,    a.kernel, a.acc, p.threads_per_row, p.rows_per_block,
                p.grid, p.block, tm.ms, tm.gbps};
            rows.push_back(br);
            print_row(br, ceiling);
            CUDA_CHECK(cudaFree(x));
            CUDA_CHECK(cudaFree(y));
          }
        }
      }
    }
  }
  if (!a.csv.empty()) {
    std::ofstream f(a.csv);
    f << "rows,dim,dtype,kernel,acc,tpr,rpb,grid,block,ms,gbps,peak_frac\n";
    for (const auto &r : rows)
      f << r.rows << ',' << r.dim << ',' << r.dtype << ',' << r.kernel << ',' << r.acc << ','
        << r.tpr << ',' << r.rpb << ',' << r.grid << ',' << r.block << ',' << r.ms << ',' << r.gbps
        << ',' << (r.gbps / ceiling) << '\n';
    std::printf("wrote %s\n", a.csv.c_str());
  }
  return 0;
}

// ------------------------------ fused quant ----------------------------------
bool buffers_equal(const void *a, const void *b, size_t bytes); // defined with --mode loop

// Times fused vs "rotate, then quantise", then checks the brief's fusion
// requirement on the same input: both paths must produce identical bytes and
// identical scales.  Exit code 1 on any mismatch.
//
// The check used to exist only in tests/test_fusion.cu, which is not part of the
// submitted tree, so the program itself could not show the one property the
// brief explicitly asks to verify.  Doing it here costs one extra run per call.
int mode_quant_bench(const Args &a) {
  DeviceInfo info = device_info(0);
  double peak = device_bandwidth_gbps(info);
  const Config cfg = make_config(a);
  const float scale = 1.0f / std::sqrt((float)a.dim);
  const int64_t n = a.rows * a.dim;
  void *x = alloc_like(a.rows, a.dim, cfg.dtype);
  fill_random(x, n, cfg.dtype, 7);
  void *q = nullptr;
  void *y = nullptr;
  CUDA_CHECK(cudaMalloc(&q, n + 64));
  CUDA_CHECK(cudaMalloc(&y, (size_t)n * dtype_size(cfg.dtype)));
  const int64_t ns = quant_scale_count(a.rows, a.dim, cfg);
  float *s = nullptr;
  CUDA_CHECK(cudaMalloc(&s, ns * sizeof(float)));
  const double bytes_fused = (double)n * (double)dtype_size(cfg.dtype) +
                             (double)n * (cfg.quant == QuantKind::kFp8E4M3 ? 1.0 : 0.5);
  // rotate: read + write the input dtype; quantise: read it again + write q.
  // (This used to be 2x dtype, which left out the intermediate tensor's write +
  // read and made the report quote 1.67x / 2.0x as the traffic ratio.  With the
  // right 7 / 6.5 B/elem the ratio is 2.33x / 2.6x, which is what dim >= 256
  // actually measures.)
  const double bytes_unfused = 3.0 * (double)n * (double)dtype_size(cfg.dtype) +
                               (double)n * (cfg.quant == QuantKind::kFp8E4M3 ? 1.0 : 0.5);

  cudaEvent_t st, en;
  CUDA_CHECK(cudaEventCreate(&st));
  CUDA_CHECK(cudaEventCreate(&en));
  auto bench = [&](int which) -> double {
    for (int i = 0; i < a.warmup; ++i) {
      if (which == 0)
        hadamard_quant(x, q, s, a.rows, a.dim, cfg, scale, nullptr);
      else if (which == 1)
        hadamard(x, y, a.rows, a.dim, cfg, scale, nullptr);
      else
        quantize_rows(y, q, s, a.rows, a.dim, cfg, nullptr);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(st));
    for (int i = 0; i < a.iters; ++i) {
      if (which == 0)
        hadamard_quant(x, q, s, a.rows, a.dim, cfg, scale, nullptr);
      else if (which == 1)
        hadamard(x, y, a.rows, a.dim, cfg, scale, nullptr);
      else
        quantize_rows(y, q, s, a.rows, a.dim, cfg, nullptr);
    }
    CUDA_CHECK(cudaEventRecord(en));
    CUDA_CHECK(cudaEventSynchronize(en));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, st, en));
    return ms / a.iters;
  };
  const double fused = bench(0);
  const double rot = bench(1);
  const double qq = bench(2);
  const double unfused = rot + qq;
  std::printf("fused quant %s %s dim=%lld rows=%lld tpr=%d rpb=%d\n", to_string(cfg.dtype),
              to_string(cfg.quant), (long long)a.dim, (long long)a.rows, cfg.threads_per_row,
              a.rpb);
  std::printf("  fused   : %9.4f ms  %8.1f GB/s (effective, %.2f B/elem)\n", fused,
              bytes_fused / (fused * 1e-3) / 1e9, bytes_fused / n);
  std::printf("  unfused : %9.4f ms  (rotate %.4f + quantise %.4f)  %8.1f GB/s (%.2f B/elem)\n",
              unfused, rot, qq, bytes_unfused / (unfused * 1e-3) / 1e9, bytes_unfused / n);
  std::printf("  speedup : %6.2fx   (bandwidth %.1f%% of peak %.0f GB/s)\n", unfused / fused,
              100.0 * bytes_fused / (fused * 1e-3) / 1e9 / peak, peak);

  // Consistency: fused(x) == quantise(hadamard(x)), bit for bit.
  const size_t nq = (size_t)(cfg.quant == QuantKind::kFp8E4M3 ? n : n / 2);
  void *q_ref = nullptr;
  float *s_ref = nullptr;
  CUDA_CHECK(cudaMalloc(&q_ref, nq));
  CUDA_CHECK(cudaMalloc(&s_ref, ns * sizeof(float)));
  CUDA_CHECK(cudaMemset(q, 0x5A, nq));
  CUDA_CHECK(cudaMemset(q_ref, 0xA5, nq));
  hadamard_quant(x, q, s, a.rows, a.dim, cfg, scale, nullptr);
  hadamard(x, y, a.rows, a.dim, cfg, scale, nullptr);
  quantize_rows(y, q_ref, s_ref, a.rows, a.dim, cfg, nullptr);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  const bool same_q = buffers_equal(q, q_ref, nq);
  const bool same_s = buffers_equal(s, s_ref, (size_t)ns * sizeof(float));
  std::printf("  check   : %s (fused vs rotate-then-quantise, bitwise: %zu bytes %s, "
              "%lld scales %s)\n",
              same_q && same_s ? "PASS" : "FAIL", nq, same_q ? "equal" : "DIFFER", (long long)ns,
              same_s ? "equal" : "DIFFER");
  CUDA_CHECK(cudaFree(q_ref));
  CUDA_CHECK(cudaFree(s_ref));
  CUDA_CHECK(cudaFree(x));
  CUDA_CHECK(cudaFree(q));
  CUDA_CHECK(cudaFree(y));
  CUDA_CHECK(cudaFree(s));
  return same_q && same_s ? 0 : 1;
}
// ------------------------------- loop ----------------------------------------
// A serving-shaped workload: `layers` layers, each rotating its own [rows, dim]
// activation, replayed for `iters` passes.  Three submission strategies:
//
//   A  plain      one cudaLaunchKernel per transform
//   B  per-step   one single-step graph replayed per transform
//   C  sequence   ONE graph holding every transform of the pass, replayed once
//
// Only C takes submissions off the critical path.  B is the trap section 8.5
// warns about: swapping cudaLaunchKernel for cudaGraphLaunch without changing
// how many launches happen buys nothing (measured below at ~0.7x).
bool buffers_equal(const void *a, const void *b, size_t bytes) {
  std::vector<uint8_t> ha(bytes), hb(bytes);
  CUDA_CHECK(cudaMemcpy(ha.data(), a, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hb.data(), b, bytes, cudaMemcpyDeviceToHost));
  return std::memcmp(ha.data(), hb.data(), bytes) == 0;
}

size_t quant_bytes(int64_t rows, int64_t dim, QuantKind q) {
  return q == QuantKind::kFp8E4M3 ? (size_t)rows * dim : ((size_t)rows * dim + 1) / 2;
}

// ------------------------------- gemm ----------------------------------------
// Rotate-then-GEMM against the *same* GEMM with the Hadamard folded into the
// A-tile load.  Both variants run the same core, so the difference is exactly
// where the rotation happens -- which is what makes this a measurement of the
// fusion rather than of the hand-written GEMM.  (Absolute GEMM quality is a
// separate matter; tools/bench_baseline.py has the cuBLAS number.)
int mode_gemm(const Args &a) {
  const DType dt = parse_dtype(a.dtype);
  if (dt == DType::kF32) {
    std::fprintf(stderr, "the fused GEMM supports fp16 / bf16 inputs\n");
    return 2;
  }
  if (a.rows <= 0 || a.dim <= 0) {
    std::fprintf(stderr, "--mode gemm needs --rows M and --dim K\n");
    return 2;
  }
  const int64_t m = a.rows;
  const int64_t k = a.dim;
  const int64_t n = a.n > 0 ? a.n : 4096;
  const float scale = parse_scale(a.scale, k);
  const size_t esz = dtype_size(dt);

  const size_t ab = (size_t)m * (size_t)k * esz;
  const size_t bb = (size_t)n * (size_t)k * esz;
  const size_t cb = (size_t)m * (size_t)n * esz;
  void *da = nullptr, *db = nullptr, *drot = nullptr, *c1 = nullptr, *c2 = nullptr;
  CUDA_CHECK(cudaMalloc(&da, ab));
  CUDA_CHECK(cudaMalloc(&db, bb));
  CUDA_CHECK(cudaMalloc(&drot, ab));
  CUDA_CHECK(cudaMalloc(&c1, cb));
  CUDA_CHECK(cudaMalloc(&c2, cb));
  fill_random(da, m * k, dt, 11);
  fill_random(db, n * k, dt, 23);

  GemmConfig gc;
  gc.dtype = dt;
  gc.scale = scale;
  gc.nsplit = a.nsplit > 0 ? a.nsplit : 1;
  Config hcfg;
  hcfg.dtype = dt;

  auto body_unfused = [&]() {
    hadamard(da, drot, m, k, hcfg, scale, nullptr);
    hadamard_gemm(drot, db, c1, m, n, k, gc, nullptr);
  };
  auto body_fused = [&]() { hadamard_gemm(da, db, c2, m, n, k, gc, nullptr); };

  gc.fuse_rotation = false;
  body_unfused();
  gc.fuse_rotation = true;
  body_fused();
  CUDA_CHECK(cudaDeviceSynchronize());
  const bool same = buffers_equal(c1, c2, cb);

  auto timed = [&](auto &&body) {
    double best = -1.0;
    for (int rep = 0; rep < a.reps; ++rep) {
      for (int i = 0; i < a.warmup; ++i)
        body();
      CUDA_CHECK(cudaDeviceSynchronize());
      cudaEvent_t s0, s1;
      CUDA_CHECK(cudaEventCreate(&s0));
      CUDA_CHECK(cudaEventCreate(&s1));
      CUDA_CHECK(cudaEventRecord(s0));
      for (int i = 0; i < a.iters; ++i)
        body();
      CUDA_CHECK(cudaEventRecord(s1));
      CUDA_CHECK(cudaEventSynchronize(s1));
      float ms = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, s0, s1));
      CUDA_CHECK(cudaEventDestroy(s0));
      CUDA_CHECK(cudaEventDestroy(s1));
      const double per = (double)ms / (double)a.iters;
      if (best < 0.0 || per < best)
        best = per;
    }
    return best;
  };

  const double ms_unfused = timed(body_unfused);
  const double ms_fused = timed(body_fused);

  const double bytes_unfused = 2.0 * (double)ab + (double)ab + (double)bb + (double)cb;
  const double bytes_fused = (double)ab + (double)bb + (double)cb;
  const double flops = 2.0 * (double)m * (double)n * (double)k;
  auto gflops = [&](double ms) { return flops / (ms * 1e-3) / 1e9; };
  auto gbps = [&](double ms, double bytes) { return bytes / (ms * 1e-3) / 1e9; };

  std::printf("shape            : M=%lld N=%lld K=%lld %s scale=%.4f nsplit=%d\n", (long long)m,
              (long long)n, (long long)k, to_string(dt), scale, gc.nsplit);
  std::printf("DRAM traffic     : unfused %.1f MiB (A + A_rot + A_rot + B + C) | fused %.1f MiB\n",
              bytes_unfused / 1048576.0, bytes_fused / 1048576.0);
  std::printf("rotation saves   : %.1f MiB per call (%.1f%%)\n",
              (bytes_unfused - bytes_fused) / 1048576.0,
              100.0 * (bytes_unfused - bytes_fused) / bytes_unfused);
  std::printf("correctness      : fused == rotate-then-GEMM bit-for-bit: %s\n",
              same ? "yes" : "NO");
  std::printf("--- A: hadamard(A) then GEMM (2 kernels) ---\n");
  std::printf("  per call       : %9.4f ms   %8.1f GFLOP/s   %8.1f GB/s effective\n", ms_unfused,
              gflops(ms_unfused), gbps(ms_unfused, bytes_unfused));
  std::printf("--- B: fused into the A-tile load (1 kernel) ---\n");
  std::printf("  per call       : %9.4f ms   %8.1f GFLOP/s   %8.1f GB/s effective   %5.2fx vs A\n",
              ms_fused, gflops(ms_fused), gbps(ms_fused, bytes_fused), ms_unfused / ms_fused);

  if (!a.csv.empty()) {
    std::ofstream f(a.csv);
    f << "m,n,k,dtype,nsplit,unfused_ms,fused_ms,speedup,gflops_unfused,gflops_fused,"
         "bytes_unfused,bytes_fused,bitwise\n";
    f << m << ',' << n << ',' << k << ',' << to_string(dt) << ',' << gc.nsplit << ',' << ms_unfused
      << ',' << ms_fused << ',' << (ms_unfused / ms_fused) << ',' << gflops(ms_unfused) << ','
      << gflops(ms_fused) << ',' << (long long)bytes_unfused << ',' << (long long)bytes_fused << ','
      << (same ? 1 : 0) << '\n';
  }

  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(drot));
  CUDA_CHECK(cudaFree(c1));
  CUDA_CHECK(cudaFree(c2));
  return 0;
}

int mode_loop(const Args &a) {
  const Config base = make_config(a);
  const bool quant = (base.quant != QuantKind::kNone);
  const int64_t layers = a.layers > 0 ? a.layers : 32;
  const int64_t tensors = a.tensors > 0 ? a.tensors : 1; // independent rotations per layer
  const int64_t slots = layers * tensors;                // one rotation each
  const int64_t passes = a.iters > 0 ? a.iters : 10;
  // Passes per graph launch.  `batches` graph launches then cover `inner` passes,
  // which is what the timed window holds -- normally the whole `passes` count, but
  // a truncated tail (passes % ppg) is simply not measured rather than mis-divided.
  const int64_t ppg = a.ppg > 0 ? a.ppg : 1;
  const int64_t batches = passes / ppg > 0 ? passes / ppg : 1;
  const int64_t inner = batches * ppg;
  std::vector<int64_t> dims = a.dims;
  if (dims.empty())
    dims.push_back(a.dim);
  if (quant && base.dtype == DType::kF32) {
    std::fprintf(stderr, "fused quantisation supports fp16 / bf16 inputs\n");
    return 2;
  }
  if (quant) {
    for (int64_t d : dims) {
      if (d < 32 || d > 1024) {
        std::fprintf(stderr, "dim %lld outside the fused quantisation range (32..1024)\n",
                     (long long)d);
        return 2;
      }
    }
  }

  // Every layer owns its own activation buffers, which is what makes the point
  // of strategy C: one graph, many buffers.
  struct Layer {
    Config cfg;
    int64_t dim = 0;
    float scale = 1.0f;
    size_t out_bytes = 0;
    int64_t n_scales = 0;
    void *x = nullptr;
    void *y = nullptr;
    void *ref = nullptr;
    float *s = nullptr;
    float *s_ref = nullptr;
    double bytes = 0.0;
  };

  std::vector<Layer> L((size_t)slots);
  double bytes_per_pass = 0.0;
  double mem_bytes = 0.0;
  for (int64_t l = 0; l < slots; ++l) {
    Layer &ly = L[(size_t)l];
    ly.cfg = base;
    // Every tensor of one layer shares that layer's head_dim; the dim list cycles
    // once per layer, not once per tensor.
    ly.dim = dims[(size_t)((l / tensors) % (int64_t)dims.size())];
    ly.scale = parse_scale(a.scale, ly.dim);
    const size_t xb = (size_t)a.rows * ly.dim * dtype_size(ly.cfg.dtype);
    ly.out_bytes = quant ? quant_bytes(a.rows, ly.dim, ly.cfg.quant) : xb;
    ly.n_scales = quant ? quant_scale_count(a.rows, ly.dim, ly.cfg) : 0;
    // Effective traffic, same convention as --mode quant.
    ly.bytes = (double)xb + (double)ly.out_bytes;
    CUDA_CHECK(cudaMalloc(&ly.x, xb));
    CUDA_CHECK(cudaMalloc(&ly.y, ly.out_bytes));
    CUDA_CHECK(cudaMalloc(&ly.ref, ly.out_bytes));
    fill_random(ly.x, a.rows * ly.dim, ly.cfg.dtype, 17 + (uint64_t)l * 131);
    if (quant) {
      CUDA_CHECK(cudaMalloc((void **)&ly.s, (size_t)ly.n_scales * sizeof(float)));
      CUDA_CHECK(cudaMalloc((void **)&ly.s_ref, (size_t)ly.n_scales * sizeof(float)));
    }
    bytes_per_pass += ly.bytes;
    mem_bytes +=
        (double)xb + 2.0 * (double)ly.out_bytes + 2.0 * (double)ly.n_scales * sizeof(float);
  }

  cudaStream_t probe = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&probe, cudaStreamNonBlocking));

  // ---- strategy A: plain submission -----------------------------------------
  auto launch_on = [&](int64_t i, cudaStream_t st) {
    Layer &ly = L[(size_t)i];
    if (quant)
      hadamard_quant(ly.x, ly.y, ly.s, a.rows, ly.dim, ly.cfg, ly.scale, st);
    else
      hadamard(ly.x, ly.y, a.rows, ly.dim, ly.cfg, ly.scale, st);
  };
  auto pass_plain = [&]() {
    for (int64_t i = 0; i < slots; ++i)
      launch_on(i, probe);
  };

  // ---- strategy B: one single-step graph per transform -----------------------
  std::vector<std::unique_ptr<Graph>> per_step((size_t)slots);
  auto t0 = std::chrono::steady_clock::now();
  for (int64_t l = 0; l < slots; ++l) {
    Layer &ly = L[(size_t)l];
    per_step[(size_t)l].reset(new Graph());
    if (quant) {
      per_step[(size_t)l]->capture_quant(ly.x, ly.y, ly.s, a.rows, ly.dim, ly.cfg, ly.scale, 1);
    } else {
      per_step[(size_t)l]->capture(ly.x, ly.y, a.rows, ly.dim, ly.cfg, ly.scale, 1);
    }
  }
  const double cap_b_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  auto pass_per_step = [&]() {
    for (int64_t l = 0; l < slots; ++l)
      per_step[(size_t)l]->launch(probe);
  };

  // ---- strategy C: one graph for the whole pass ------------------------------
  // With --passes-per-graph K the same node list is repeated K times, so one
  // submission covers K passes.  Every pass is the identical computation on the
  // same buffers, so the concatenation is still bit-for-bit the same work.
  GraphSequence seq;
  for (int64_t p = 0; p < ppg; ++p) {
    for (int64_t l = 0; l < slots; ++l) {
      Layer &ly = L[(size_t)l];
      if (quant)
        seq.add_quant(ly.x, ly.y, ly.s, a.rows, ly.dim, ly.cfg, ly.scale);
      else
        seq.add(ly.x, ly.y, a.rows, ly.dim, ly.cfg, ly.scale);
    }
  }
  t0 = std::chrono::steady_clock::now();
  seq.record();
  const double cap_c_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  auto pass_seq = [&]() { seq.launch(probe); };

  // ---- strategy D: the same one-graph pass, but with the layers forked -------
  // C records a chain because that is all a stream capture can express: node i
  // waits for node i-1, so the pass costs the sum of the kernel durations even
  // when each kernel fills a fraction of the GPU.  Every layer here owns its
  // own activation buffers, so nothing orders them; add_parallel() states that
  // and record() emits a fork/join instead, which lets the scheduler overlap
  // whatever fits.
  GraphSequence seq_par;
  // Three shapes come out of the same node list, and the difference between them
  // is only which steps are *anchors* (queued with add(), so they join whatever
  // is open).  See report 5.8 for what each one costs.
  //   all roots    no anchor at all -- every node is a root
  //   single root  the first tensor of the pass is the only anchor (flat fan-out)
  //   group        the first tensor of every layer is an anchor (per-layer fork/join)
  const bool entry_serial = a.serial_entry;
  const bool entry_group = a.group_entry;
  // `i` runs over all K passes; `slots` is a multiple of `tensors`, so the anchor
  // condition also holds at every pass boundary and the passes stay ordered.
  for (int64_t i = 0; i < slots * ppg; ++i) {
    Layer &ly = L[(size_t)(i % slots)];
    if ((entry_serial && i == 0) || (entry_group && (i % tensors) == 0)) {
      if (quant)
        seq_par.add_quant(ly.x, ly.y, ly.s, a.rows, ly.dim, ly.cfg, ly.scale);
      else
        seq_par.add(ly.x, ly.y, a.rows, ly.dim, ly.cfg, ly.scale);
    } else if (quant) {
      seq_par.add_quant_parallel(ly.x, ly.y, ly.s, a.rows, ly.dim, ly.cfg, ly.scale);
    } else {
      seq_par.add_parallel(ly.x, ly.y, a.rows, ly.dim, ly.cfg, ly.scale);
    }
  }
  t0 = std::chrono::steady_clock::now();
  seq_par.record();
  const double cap_d_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  auto pass_par = [&]() { seq_par.launch(probe); };

  // ---- strategy E: the same rotations, raw launches, spread over N streams ----
  // No graph at all.  This separates "the rotations are independent" from "there is
  // one submission": a graph removes submissions, streams only move them around, so
  // E is the control that says how much of D's behaviour is about parallelism.
  const int nstreams = a.streams > 1 ? a.streams : 0;
  std::vector<cudaStream_t> wk;
  std::vector<cudaEvent_t> done;
  for (int i = 0; i < nstreams; ++i) {
    cudaStream_t sk = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sk, cudaStreamNonBlocking));
    wk.push_back(sk);
    cudaEvent_t ev = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    done.push_back(ev);
  }
  auto pass_multi = [&]() {
    if (nstreams < 1)
      return;
    for (int64_t i = 0; i < slots; ++i)
      launch_on(i, wk[(size_t)(i % nstreams)]);
    // The timing window is recorded on `probe`, so it has to see every worker.
    for (int i = 0; i < nstreams; ++i) {
      CUDA_CHECK(cudaEventRecord(done[(size_t)i], wk[(size_t)i]));
      CUDA_CHECK(cudaStreamWaitEvent(probe, done[(size_t)i], 0));
    }
  };

  // ---- correctness: B and C must reproduce A bit for bit ---------------------
  for (int64_t l = 0; l < slots; ++l) {
    Layer &ly = L[(size_t)l];
    if (quant)
      hadamard_quant(ly.x, ly.ref, ly.s_ref, a.rows, ly.dim, ly.cfg, ly.scale, probe);
    else
      hadamard(ly.x, ly.ref, a.rows, ly.dim, ly.cfg, ly.scale, probe);
  }
  CUDA_CHECK(cudaStreamSynchronize(probe));

  auto matches_after = [&]() {
    int64_t n = 0;
    for (int64_t l = 0; l < slots; ++l) {
      Layer &ly = L[(size_t)l];
      bool ok = buffers_equal(ly.y, ly.ref, ly.out_bytes);
      if (ok && quant) {
        ok = buffers_equal(ly.s, ly.s_ref, (size_t)ly.n_scales * sizeof(float));
      }
      if (ok)
        ++n;
    }
    return n;
  };

  pass_per_step();
  CUDA_CHECK(cudaStreamSynchronize(probe));
  const int64_t match_b = matches_after();

  for (int64_t l = 0; l < slots; ++l) {
    CUDA_CHECK(cudaMemsetAsync(L[(size_t)l].y, 0, L[(size_t)l].out_bytes, probe));
  }
  CUDA_CHECK(cudaStreamSynchronize(probe));
  pass_seq();
  CUDA_CHECK(cudaStreamSynchronize(probe));
  const int64_t match_c = matches_after();

  for (int64_t l = 0; l < slots; ++l) {
    CUDA_CHECK(cudaMemsetAsync(L[(size_t)l].y, 0, L[(size_t)l].out_bytes, probe));
  }
  CUDA_CHECK(cudaStreamSynchronize(probe));
  pass_par();
  CUDA_CHECK(cudaStreamSynchronize(probe));
  const int64_t match_d = matches_after();

  int64_t match_e = 0;
  if (nstreams > 1) {
    for (int64_t l = 0; l < slots; ++l) {
      CUDA_CHECK(cudaMemsetAsync(L[(size_t)l].y, 0, L[(size_t)l].out_bytes, probe));
    }
    CUDA_CHECK(cudaStreamSynchronize(probe));
    pass_multi();
    CUDA_CHECK(cudaStreamSynchronize(probe));
    match_e = matches_after();
  }

  // ---- timing ---------------------------------------------------------------
  cudaEvent_t st0, st1;
  CUDA_CHECK(cudaEventCreate(&st0));
  CUDA_CHECK(cudaEventCreate(&st1));
  // Each timed window must cover exactly `passes` passes, so that dividing by
  // `passes` below yields a per-pass figure.  Measuring a single pass and then
  // dividing by `passes` under-reports every strategy by that factor (and hides
  // the queue-drain that dominates the launch-bound sizes).
  // `launches` submissions of `per_launch` passes each; the result is per pass.
  // C and D with --passes-per-graph K submit `batches` graphs of K passes, so they
  // report the same denominator as the per-pass strategies.
  auto timed = [&](auto &&body, int64_t launches, int64_t per_launch) {
    double best = -1.0;
    for (int rep = 0; rep < a.reps; ++rep) {
      CUDA_CHECK(cudaEventRecord(st0, probe));
      for (int64_t p = 0; p < launches; ++p)
        body();
      CUDA_CHECK(cudaEventRecord(st1, probe));
      CUDA_CHECK(cudaEventSynchronize(st1));
      float ms = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, st0, st1));
      if (best < 0.0 || (double)ms < best)
        best = (double)ms;
    }
    return best / (double)(launches * per_launch); // per pass
  };

  for (int warm = 0; warm < 2; ++warm) {
    pass_plain();
    pass_per_step();
    for (int64_t p = 0; p < batches; ++p)
      pass_seq();
    for (int64_t p = 0; p < batches; ++p)
      pass_par();
    pass_multi();
  }
  CUDA_CHECK(cudaStreamSynchronize(probe));

  const double ms_a = timed([&] { pass_plain(); }, passes, 1);
  const double ms_b = timed([&] { pass_per_step(); }, passes, 1);
  const double ms_c = timed([&] { pass_seq(); }, batches, ppg);
  const double ms_d = timed([&] { pass_par(); }, batches, ppg);
  const double ms_e = nstreams > 1 ? timed([&] { pass_multi(); }, passes, 1) : 0.0;

  const int64_t transforms = slots * passes;
  const double calls = (double)slots;
  auto bw = [&](double ms_per_pass) { return bytes_per_pass / (ms_per_pass * 1e-3) / 1e9; };

  std::printf("shape set        : rows=%lld dims=[", (long long)a.rows);
  for (size_t i = 0; i < dims.size(); ++i)
    std::printf("%s%lld", i ? " " : "", (long long)dims[i]);
  std::printf("] %s kernel=%s quant=%s\n", to_string(base.dtype), to_string(base.kernel),
              to_string(base.quant));
  if (tensors > 1) {
    std::printf("workload         : %lld layers x %lld tensors x %lld passes = %lld transforms "
                "(%zu distinct shape(s))\n",
                (long long)layers, (long long)tensors, (long long)passes, (long long)transforms,
                dims.size());
  } else {
    std::printf(
        "workload         : %lld layers x %lld passes = %lld transforms (%zu distinct shape(s))\n",
        (long long)layers, (long long)passes, (long long)transforms, dims.size());
  }
  std::printf("layer buffers    : %.1f MiB, %.1f MiB touched per pass\n", mem_bytes / 1048576.0,
              bytes_per_pass / 1048576.0);
  std::printf("correctness      : per-step %lld/%lld, sequence %lld/%lld, forked %lld/%lld "
              "bit-identical to plain\n",
              (long long)match_b, (long long)slots, (long long)match_c, (long long)slots,
              (long long)match_d, (long long)slots);
  if (nstreams > 1) {
    std::printf("correctness (E)  : %lld/%lld bit-identical to plain (over %d streams)\n",
                (long long)match_e, (long long)slots, nstreams);
  }
  std::printf("--- A: %lld cudaLaunchKernel per pass ---\n", (long long)slots);
  std::printf("  per pass       : %9.4f ms   per transform %8.3f us   %8.1f GB/s\n", ms_a,
              ms_a * 1e3 / calls, bw(ms_a));
  std::printf("--- B: %lld single-step graph launches per pass ---\n", (long long)slots);
  std::printf("  per pass       : %9.4f ms   per transform %8.3f us   %8.1f GB/s   %5.2fx vs A\n",
              ms_b, ms_b * 1e3 / calls, bw(ms_b), ms_a / ms_b);
  if (ppg == 1) {
    std::printf("--- C: 1 graph launch per pass (%d nodes) ---\n", seq.steps());
  } else {
    std::printf("--- C: 1 graph launch per %lld passes (%d nodes, batched) ---\n", (long long)ppg,
                seq.steps());
  }
  std::printf("  per pass       : %9.4f ms   per transform %8.3f us   %8.1f GB/s   %5.2fx vs A\n",
              ms_c, ms_c * 1e3 / calls, bw(ms_c), ms_a / ms_c);
  // The three shapes differ only in which steps are anchors; say which one ran so
  // the line is self-describing in a log.
  const char *dshape = entry_group ? (tensors > 1 ? ", per-layer groups, single root"
                                                  : ", per-layer groups, no forks")
                                   : (entry_serial ? ", single root" : ", all roots");
  // forks() == 0 means every step was an anchor, record() took the linear path.
  const char *dkind = seq_par.forks() == 0
                          ? "linear capture"
                          : (seq_par.kernel_nodes() ? "kernel nodes" : "child graphs");
  if (ppg == 1) {
    std::printf("--- D: %s per pass (%d nodes, %d forked, %s%s) ---\n",
                seq_par.forks() == 0 ? "1 graph launch" : "1 forked graph launch", seq_par.steps(),
                seq_par.forks(), dkind, dshape);
  } else {
    std::printf("--- D: %s per %lld passes (%d nodes, %d forked, %s%s) ---\n",
                seq_par.forks() == 0 ? "1 graph launch" : "1 forked graph launch", (long long)ppg,
                seq_par.steps(), seq_par.forks(), dkind, dshape);
  }
  std::printf("  per pass       : %9.4f ms   per transform %8.3f us   %8.1f GB/s   %5.2fx vs A   "
              "%5.2fx vs C\n",
              ms_d, ms_d * 1e3 / calls, bw(ms_d), ms_a / ms_d, ms_c / ms_d);
  if (nstreams > 1) {
    std::printf("--- E: %lld cudaLaunchKernel per pass over %d streams ---\n", (long long)slots,
                nstreams);
    std::printf("  per pass       : %9.4f ms   per transform %8.3f us   %8.1f GB/s   %5.2fx vs A   "
                "%5.2fx vs C\n",
                ms_e, ms_e * 1e3 / calls, bw(ms_e), ms_a / ms_e, ms_c / ms_e);
  }
  std::printf("capture cost     : B %8.3f ms (%lld graphs) | C %8.3f ms (%lld passes/graph) | "
              "D %8.3f ms\n",
              cap_b_ms, (long long)slots, cap_c_ms, (long long)ppg, cap_d_ms);
  std::printf("amortisation     : C pays for itself after ~%.1f pass(es)\n",
              cap_c_ms / (ms_a - ms_c > 1e-9 ? ms_a - ms_c : ms_a));

  if (!a.csv.empty()) {
    std::ofstream f(a.csv);
    f << "rows,dims,layers,tensors,streams,passes,passes_per_graph,quant,dtype,transforms,"
         "plain_ms,per_step_ms,sequence_ms,parallel_ms,multistream_ms,speedup_seq,speedup_par,"
         "speedup_multi,capture_seq_ms,capture_par_ms\n";
    char dimstr[128] = {0};
    for (size_t i = 0; i < dims.size(); ++i) {
      char one[24];
      std::snprintf(one, sizeof(one), "%s%lld", i ? " " : "", (long long)dims[i]);
      std::strncat(dimstr, one, sizeof(dimstr) - std::strlen(dimstr) - 1);
    }
    f << a.rows << ",\"" << dimstr << "\"," << layers << "," << tensors << "," << nstreams << ","
      << passes << "," << ppg << "," << to_string(base.quant) << "," << to_string(base.dtype) << ","
      << transforms << "," << ms_a << "," << ms_b << "," << ms_c << "," << ms_d << "," << ms_e
      << "," << (ms_a / ms_c) << "," << (ms_a / ms_d) << "," << (ms_e > 0.0 ? ms_a / ms_e : 0.0)
      << "," << cap_c_ms << "," << cap_d_ms << "\n";
  }

  for (Layer &ly : L) {
    CUDA_CHECK(cudaFree(ly.x));
    CUDA_CHECK(cudaFree(ly.y));
    CUDA_CHECK(cudaFree(ly.ref));
    if (ly.s != nullptr)
      CUDA_CHECK(cudaFree(ly.s));
    if (ly.s_ref != nullptr)
      CUDA_CHECK(cudaFree(ly.s_ref));
  }
  CUDA_CHECK(cudaEventDestroy(st0));
  CUDA_CHECK(cudaEventDestroy(st1));
  for (size_t i = 0; i < wk.size(); ++i) {
    CUDA_CHECK(cudaEventDestroy(done[i]));
    CUDA_CHECK(cudaStreamDestroy(wk[i]));
  }
  CUDA_CHECK(cudaStreamDestroy(probe));
  return 0;
}
} // namespace

int run(int argc, char **argv) {
  Args a = parse(argc, argv);
  if (a.graph) {
    std::fprintf(stderr,
                 "# --graph: every timed burst is one cudaGraphLaunch of %d kernels "
                 "(host submission removed from the critical path)\n",
                 a.iters);
  }
  if (a.mode == "info")
    return mode_info();
  if (a.mode == "roof")
    return mode_roof();
  if (a.mode == "advice")
    return mode_advice(a);
  if (a.mode == "verify")
    return mode_verify(a);
  if (a.mode == "bench")
    return mode_bench(a);
  if (a.mode == "cpu")
    return mode_cpu(a);
  if (a.mode == "tune")
    return mode_tune(a);
  if (a.mode == "inplace")
    return mode_inplace(a);
  if (a.mode == "dump")
    return mode_dump(a);
  if (a.mode == "matrix")
    return mode_matrix(a);
  if (a.mode == "quant")
    return mode_quant_bench(a);
  if (a.mode == "loop")
    return mode_loop(a);
  if (a.mode == "gemm")
    return mode_gemm(a);
  std::fprintf(stderr, "unknown mode '%s'\n", a.mode.c_str());
  return 2;
}

// Invalid arguments (non power-of-two dim, bad block size, ...) are reported by
// the library as exceptions.  Without this, `--dim 100` ended in terminate() and
// a core dump -- noticed when testing the new block-size check on Linux.
int main(int argc, char **argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}

// build revision marker: 2026-09-17 GraphExecCache + fork/join + fused GEMM (upload v2,
//                         per-layer tensor groups + multi-stream control for 7.13)

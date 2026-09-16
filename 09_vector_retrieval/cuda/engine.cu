#include "vsearch/engine.hpp"
#include "vsearch/binary_io.hpp"

#if defined(VSEARCH_COREX) && !defined(__CUDA_ACC__)
// Iluvatar CoreX devices emulate IEEE double-precision arithmetic with
// reduced mantissa width (measured ~1e-4 relative error per long sum), which
// would corrupt L2 / inner-product distances. NVIDIA builds keep double
// accumulation for maximum accuracy; CoreX builds accumulate in float, whose
// arithmetic on CoreX is IEEE-accurate (verified against the CPU reference).
#define VSEARCH_ACC_FLOAT 1
#endif

#ifdef VSEARCH_ACC_FLOAT
#define VSEARCH_ACC_TYPE float
#else
#define VSEARCH_ACC_TYPE double
#endif

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cublas_v2.h>

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace vs {

namespace {

#define VSCU_CHECK(expr)                                                     \
  do {                                                                       \
    cudaError_t err = (expr);                                                \
    if (err != cudaSuccess)                                                  \
      throwRuntime(std::string("CUDA error at ") + __FILE__ + ":" +          \
                   std::to_string(__LINE__) + ": " + cudaGetErrorString(err) + \
                   " [" #expr "]");                                           \
  } while (0)

#define VSCUBLAS_CHECK(expr)                                                \
  do {                                                                       \
    cublasStatus_t st = (expr);                                              \
    if (st != CUBLAS_STATUS_SUCCESS)                                          \
      throwRuntime(std::string("cuBLAS error at ") + __FILE__ + ":" +         \
                   std::to_string(__LINE__) + " code=" +                     \
                   std::to_string(static_cast<int>(st)));                     \
  } while (0)

// ---------------------------------------------------------------------------
// Small RAII device allocation helper.
// ---------------------------------------------------------------------------
template <typename T>
class DevMem {
 public:
  DevMem() = default;
  explicit DevMem(std::size_t n) { resize(n); }
  ~DevMem() {
    if (ptr_) cudaFree(ptr_);  // destructors must not throw
    ptr_ = nullptr;
    count_ = 0;
  }
  DevMem(const DevMem&) = delete;
  DevMem& operator=(const DevMem&) = delete;
  DevMem(DevMem&& o) noexcept { *this = std::move(o); }
  DevMem& operator=(DevMem&& o) noexcept {
    if (this != &o) {
      release();
      ptr_ = o.ptr_;
      count_ = o.count_;
      o.ptr_ = nullptr;
      o.count_ = 0;
    }
    return *this;
  }

  void resize(std::size_t n) {
    if (n == count_ && n != 0) return;
    release();
    if (n == 0) return;
    VSCU_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
    count_ = n;
  }

  void release() {
    if (ptr_) VSCU_CHECK(cudaFree(ptr_));
    ptr_ = nullptr;
    count_ = 0;
  }

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }
  std::size_t count() const { return count_; }
  std::size_t bytes() const { return count_ * sizeof(T); }

 private:
  T* ptr_ = nullptr;
  std::size_t count_ = 0;
};

// Reusable temporary allocation for CUB.
class TempStore {
 public:
  void* ensure(std::size_t bytes) {
    if (bytes <= bytes_) return ptr_;
    if (ptr_) VSCU_CHECK(cudaFree(ptr_));
    ptr_ = nullptr;
    bytes_ = 0;
    if (bytes) {
      VSCU_CHECK(cudaMalloc(&ptr_, bytes));
      bytes_ = bytes;
    }
    return ptr_;
  }
  ~TempStore() {
    if (ptr_) cudaFree(ptr_);
  }

 private:
  void* ptr_ = nullptr;
  std::size_t bytes_ = 0;
};

#define syncCheck()                                                        \
  do {                                                                     \
    cudaError_t e_ = cudaDeviceSynchronize();                              \
    if (e_ != cudaSuccess)                                                 \
      throwRuntime(std::string("sync failure at ") + __FILE__ + ":" +      \
                   std::to_string(__LINE__) + ": " + cudaGetErrorString(e_)); \
  } while (0)

// ---------------------------------------------------------------------------
// Score key helpers. Keys are 64-bit so the low 32 bits carry the id and make
// ties deterministic (smallest id first).
// ---------------------------------------------------------------------------
__device__ __forceinline__ i32 devFloatBits(float v) {
  return __float_as_int(v);
}

__device__ __forceinline__ float devBitsToFloat(i32 u) {
  return __int_as_float(u);
}

__device__ __forceinline__ i32 devSortable(float v) {
  const i32 u = devFloatBits(v);
  return (u & 0x80000000) ? ~u : (u | 0x80000000);
}

__device__ __forceinline__ float devSortableToFloat(i32 u) {
  const i32 raw = (u & 0x80000000) ? (u & 0x7fffffff) : ~u;
  return devBitsToFloat(raw);
}

// smallerBetter=true  -> low key == low score (L2 distance, cosine distance)
// smallerBetter=false -> low key == high score (inner product)
__device__ __forceinline__ unsigned long long packKey(float score,
                                                        bool smallerBetter,
                                                        i32 id) {
  const i32 s = devSortable(score);
  const i32 rank = smallerBetter ? s : ~s;
  return (static_cast<unsigned long long>(
              static_cast<unsigned int>(rank))
          << 32) |
         static_cast<unsigned int>(static_cast<unsigned int>(id));
}

__device__ __forceinline__ float unpackScore(unsigned long long key,
                                              bool smallerBetter) {
  const auto hi = static_cast<unsigned int>(key >> 32);
  const i32 rank = static_cast<i32>(hi);
  return devSortableToFloat(smallerBetter ? rank : ~rank);
}

__device__ __forceinline__ i32 unpackId(unsigned long long key) {
  return static_cast<i32>(key & 0xffffffffull);
}

// metric codes: 0 = L2, 1 = inner product, 2 = cosine (distance)
__device__ __forceinline__ bool metricSmaller(int metricCode) {
  return metricCode == 0 || metricCode == 2;
}

// ---------------------------------------------------------------------------
// Exact search kernels.
// ---------------------------------------------------------------------------
// One thread computes one full row distance and writes a packed key.
__global__ void exactKeysKernel(const float* __restrict__ x,
                                const float* __restrict__ queries,
                                const double* __restrict__ queryNorm,
                                const long long n, int dim, int metricCode,
                                unsigned long long* __restrict__ keysOut) {
  const long long row = static_cast<long long>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  if (row >= n) return;
  const float* q = queries + static_cast<size_t>(blockIdx.y) * dim;
  const float* xr = x + static_cast<size_t>(row) * dim;
  VSEARCH_ACC_TYPE dot = 0;
  VSEARCH_ACC_TYPE sq = 0;
  for (int d = 0; d < dim; ++d) {
    const float xi = xr[d];
    const float qi = q[d];
    const VSEARCH_ACC_TYPE diff =
        static_cast<VSEARCH_ACC_TYPE>(xi) - static_cast<VSEARCH_ACC_TYPE>(qi);
    dot += static_cast<VSEARCH_ACC_TYPE>(xi) * qi;
    sq += diff * diff;
  }
  float score;
  if (metricCode == 0) {  // L2
    score = static_cast<float>(sq);
  } else if (metricCode == 1) {  // inner product
    score = static_cast<float>(dot);
  } else {  // cosine: vectors normalized, distance = 1 - dot
    score = static_cast<float>(1) - static_cast<float>(dot);
  }
  keysOut[static_cast<size_t>(blockIdx.y) * n + row] =
      packKey(score, metricSmaller(metricCode), static_cast<i32>(row));
}

__global__ void decodeTopKeysKernel(
    const unsigned long long* __restrict__ sortedKeys, long long itemsPerQuery,
    int nq, int topK, int metricCode, int* __restrict__ ids,
    float* __restrict__ scores) {
  const int qi = blockIdx.y;
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= topK || k >= itemsPerQuery) return;
  const unsigned long long key =
      sortedKeys[static_cast<size_t>(qi) * itemsPerQuery + k];
  float score = unpackScore(key, metricSmaller(metricCode));
  if (metricCode == 0) score = sqrtf(fmaxf(score, 0.f));
  const size_t out = static_cast<size_t>(qi) * topK + k;
  ids[out] = unpackId(key);
  scores[out] = score;
}

// ---------------------------------------------------------------------------
// IVF kernels
// ---------------------------------------------------------------------------
__global__ void ivfCenterKeysKernel(const float* __restrict__ queries,
                                    const double* __restrict__ queryNorm,
                                    const float* __restrict__ centers,
                                    const float* __restrict__ centerNorm,
                                    int dim, int nlist, int metricCode,
                                    unsigned long long* __restrict__ keysOut) {
  const int qi = blockIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= nlist) return;
  const float* q = queries + static_cast<size_t>(qi) * dim;
  const float* cn = centers + static_cast<size_t>(c) * dim;
  VSEARCH_ACC_TYPE dot = 0;
  for (int d = 0; d < dim; ++d)
    dot += static_cast<VSEARCH_ACC_TYPE>(q[d]) * cn[d];
  float score;
  if (metricCode == 0)
    score = static_cast<float>(centerNorm[c]) +
            static_cast<float>(queryNorm[qi]) - 2.f * static_cast<float>(dot);
  else if (metricCode == 1)
    score = static_cast<float>(dot);
  else
    score = static_cast<float>(1) - static_cast<float>(dot);
  keysOut[static_cast<size_t>(qi) * nlist + c] =
      packKey(score, metricSmaller(metricCode), c);
}

__global__ void extractFirstKernel(
    const unsigned long long* __restrict__ sorted, long long perQuery,
    int nQuery, int take, int* __restrict__ out) {
  const int qi = blockIdx.y;
  for (int j = threadIdx.x; j < take; j += blockDim.x)
    out[static_cast<size_t>(qi) * take + j] =
        unpackId(sorted[static_cast<size_t>(qi) * perQuery + j]);
}

__global__ void probeCountsKernel(const int* __restrict__ probeIds,
                                  const long long* __restrict__ listOffsets,
                                  int nProbeSlots,
                                  long long* __restrict__ countsOut) {
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= nProbeSlots) return;
  const int c = probeIds[slot];
  countsOut[slot] = listOffsets[c + 1] - listOffsets[c];
}

__global__ void queryStartKernel(
    const long long* __restrict__ probeOffsets, int nProbe, int nQuery,
    long long* __restrict__ queryStart) {
  const int qi = blockIdx.x * blockDim.x + threadIdx.x;
  if (qi > nQuery) return;
  queryStart[qi] = probeOffsets[static_cast<size_t>(qi) * nProbe];
}

// One block per probe slot; a full inverted list is copied to the candidate
// arena for its query.
__global__ void gatherCandidatesKernel(
    const int* __restrict__ probeIds,
    const long long* __restrict__ probeOffsets,
    const long long* __restrict__ listOffsets,
    const int* __restrict__ listIds, int nProbeSlots,
    int* __restrict__ candidatesOut) {
  const int slot = blockIdx.x;
  if (slot >= nProbeSlots) return;
  const int c = probeIds[slot];
  const long long begin = listOffsets[c];
  const long long count = listOffsets[c + 1] - begin;
  const long long base = probeOffsets[slot];
  for (long long i = threadIdx.x; i < count; i += blockDim.x)
    candidatesOut[base + i] = listIds[begin + i];
}

__device__ __forceinline__ int locateQuery(
    const long long* __restrict__ queryStart, int nQuery, long long p) {
  int lo = 0, hi = nQuery;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (queryStart[mid] <= p)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo - 1;
}

__global__ void ivfFlatKeysKernel(
    const int* __restrict__ candidates,
    const float* __restrict__ vectors,
    const float* __restrict__ queries,
    const double* __restrict__ queryNorm,
    const long long* __restrict__ queryStart, int nQuery, int dim,
    int metricCode, long long totalCandidates,
    unsigned long long* __restrict__ keysOut) {
  const long long p = static_cast<long long>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
  if (p >= totalCandidates) return;
  const int qi = locateQuery(queryStart, nQuery, p);
  const int vid = candidates[p];
  const float* q = queries + static_cast<size_t>(qi) * dim;
  const float* x = vectors + static_cast<size_t>(vid) * dim;
  VSEARCH_ACC_TYPE dot = 0;
  VSEARCH_ACC_TYPE sq = 0;
  for (int d = 0; d < dim; ++d) {
    const float xi = x[d];
    const float qi = q[d];
    const VSEARCH_ACC_TYPE diff =
        static_cast<VSEARCH_ACC_TYPE>(xi) - static_cast<VSEARCH_ACC_TYPE>(qi);
    dot += static_cast<VSEARCH_ACC_TYPE>(xi) * qi;
    sq += diff * diff;
  }
  float score;
  if (metricCode == 0)
    score = static_cast<float>(sq);
  else if (metricCode == 1)
    score = static_cast<float>(dot);
  else
    score = static_cast<float>(1) - static_cast<float>(dot);
  keysOut[p] = packKey(score, metricSmaller(metricCode), vid);
}

__global__ void decodeVariableTopKernel(
    const unsigned long long* __restrict__ sorted,
    const long long* __restrict__ queryStart, int nQuery, int topK,
    int metricCode, int* __restrict__ ids, float* __restrict__ scores) {
  const int qi = blockIdx.y;
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= topK) return;
  const long long start = queryStart[qi];
  const long long count = queryStart[qi + 1] - start;
  if (k >= count) return;
  const unsigned long long key = sorted[start + k];
  float score = unpackScore(key, metricSmaller(metricCode));
  if (metricCode == 0) score = sqrtf(fmaxf(score, 0.f));
  const size_t out = static_cast<size_t>(qi) * topK + k;
  ids[out] = unpackId(key);
  scores[out] = score;
}

}  // namespace

struct GpuEngine::Impl {
  Impl(const Dataset& dataset, const SearchConfig& conf)
      : cfg(conf), metric(dataset.metric) {
    if (dataset.metric == Metric::None)
      throwRuntime("engine: dataset must declare a metric");
    if (dataset.dim <= 0 || dataset.n <= 0)
      throwRuntime("engine: empty dataset");
    dim = dataset.dim;
    n = dataset.n;
    x.resize(static_cast<std::size_t>(dataset.n) * dataset.dim);
    VSCU_CHECK(cudaMemcpy(x.get(), dataset.data.data(), x.bytes(),
                          cudaMemcpyHostToDevice));
  }

  SearchConfig cfg;
  Metric metric;
  int dim = 0;
  i64 n = 0;
  DevMem<float> x;

  // Index state (valid after buildIndex or loadIndex).
  bool hasIndex_ = false;
  IndexKind indexKind = IndexKind::IvfFlat;
  int nlist = 0;
  int pqM = 0;
  int pqKs = 0;
  int pqSubDim = 0;
  DevMem<float> centers;
  DevMem<float> centerNorm;
  DevMem<long long> listOffsets;  // nlist+1
  DevMem<int> listIds;            // n
  DevMem<unsigned char> pqCodes;  // n*pqM (IVF-PQ)
  DevMem<float> pqCodebooks;      // pqM*pqKs*pqSubDim

  std::size_t indexDeviceBytes() const {
    std::size_t b = centers.bytes() + centerNorm.bytes() +
                    listOffsets.bytes() + listIds.bytes();
    if (indexKind == IndexKind::IvfPq) b += pqCodes.bytes() + pqCodebooks.bytes();
    return b;
  }

  void requireIndex(SearchMode mode) const {
    if (!hasIndex_) throwRuntime("engine: index not built/loaded");
    if (mode == SearchMode::IvfPq && indexKind != IndexKind::IvfPq)
      throwRuntime("engine: requested IVF-PQ but loaded index is IVF-Flat");
    if (mode == SearchMode::IvfFlat && indexKind == IndexKind::IvfPq)
      throwRuntime("engine: requested IVF-Flat but loaded index is IVF-PQ");
  }
};

namespace {

// ---------------------------------------------------------------------------
// CUB / k-means support kernels.
// ---------------------------------------------------------------------------
__global__ void sqNormKernel(const float* __restrict__ rows, long long n,
                             int dim, float* __restrict__ out) {
  const long long i = static_cast<long long>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
  if (i >= n) return;
  const float* r = rows + static_cast<size_t>(i) * dim;
  float s = 0.f;
  for (int d = 0; d < dim; ++d) s += r[d] * r[d];
  out[i] = s;
}

__global__ void gatherEvenSampleKernel(const float* __restrict__ full,
                                       long long fullCount,
                                       long long sampleCount, int dim,
                                       float* __restrict__ out) {
  const long long i = static_cast<long long>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
  if (i >= sampleCount) return;
  const long long src = sampleCount >= fullCount
                            ? i
                            : i * fullCount / sampleCount;
  const float* r = full + static_cast<size_t>(src) * dim;
  float* o = out + static_cast<size_t>(i) * dim;
  for (int d = 0; d < dim; ++d) o[d] = r[d];
}

__global__ void assignNearestDotsKernel(
    const float* __restrict__ dots, long long chunkRows, long long chunkBase,
    int nlist, const float* __restrict__ pointNorm,
    const float* __restrict__ centerNorm, int* __restrict__ assignOut) {
  const long long local = static_cast<long long>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
  if (local >= chunkRows) return;
  const long long pid = chunkBase + local;
  float best = 1e30f;
  int bestC = 0;
  for (int c = 0; c < nlist; ++c) {
    const float dot = dots[local + static_cast<size_t>(c) * chunkRows];
    const float d = pointNorm[pid] + centerNorm[c] - 2.f * dot;
    if (d < best) {
      best = d;
      bestC = c;
    }
  }
  assignOut[pid] = bestC;
}

__global__ void accumulateMiniBatchKernel(
    const float* __restrict__ points, const int* __restrict__ assign,
    long long np, int dim, int nlist, float* __restrict__ sums,
    int* __restrict__ counts) {
  const long long p = static_cast<long long>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
  if (p >= np) return;
  const int c = assign[p];
  const float* r = points + static_cast<size_t>(p) * dim;
  float* s = sums + static_cast<size_t>(c) * dim;
  for (int d = 0; d < dim; ++d) atomicAdd(s + d, r[d]);
  atomicAdd(&counts[c], 1);
}

__global__ void divideCentersKernel(float* __restrict__ sums,
                                    const int* __restrict__ counts,
                                    int nlist, int dim) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= nlist) return;
  const float inv = counts[c] > 0 ? 1.f / static_cast<float>(counts[c]) : 0.f;
  float* s = sums + static_cast<size_t>(c) * dim;
  for (int d = 0; d < dim; ++d) s[d] *= inv;
}

__global__ void normalizeCenterKernel(float* __restrict__ centers, int nlist,
                                      int dim) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= nlist) return;
  float* s = centers + static_cast<size_t>(c) * dim;
  float nrm = 0.f;
  for (int d = 0; d < dim; ++d) nrm += s[d] * s[d];
  nrm = sqrtf(fmaxf(nrm, 1e-30f));
  for (int d = 0; d < dim; ++d) s[d] /= nrm;
}

__global__ void copyFixEmptyCentersKernel(
    const float* __restrict__ sample, long long sampleCount,
    const int* __restrict__ counts, float* __restrict__ centers,
    int nlist, int dim, int seed) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= nlist) return;
  if (counts[c] != 0) return;
  const long long pick =
      (static_cast<long long>(c) * 2654435761u + seed) % sampleCount;
  const float* r = sample + static_cast<size_t>(pick) * dim;
  float* s = centers + static_cast<size_t>(c) * dim;
  for (int d = 0; d < dim; ++d) s[d] = r[d];
}

__global__ void assignmentChangedKernel(const int* __restrict__ a,
                                        const int* __restrict__ b, long long n,
                                        int* __restrict__ out) {
  const long long i = static_cast<long long>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
  if (i < n && a[i] != b[i]) atomicAdd(out, 1);
}

void zeroCountsAndSums(float* sums, int* counts, int nlist, int dim) {
  VSCU_CHECK(cudaMemset(sums, 0, static_cast<size_t>(nlist) * dim * sizeof(float)));
  VSCU_CHECK(cudaMemset(counts, 0, static_cast<size_t>(nlist) * sizeof(int)));
}

void launchSqNorm(const float* rows, long long n, int dim, float* out) {
  const int block = 256;
  const int grid = static_cast<int>((n + block - 1) / block);
  sqNormKernel<<<grid, block>>>(rows, n, dim, out);
  syncCheck();
}

void launchAssignment(cublasHandle_t handle, const float* points,
                      long long np, const float* centers, int nlist, int dim,
                      const float* pointNorm, const float* centerNorm,
                      long long chunkRows, int* assignOut, float* dotsScratch) {
  const float one = 1.f, zero = 0.f;
  for (long long base = 0; base < np; base += chunkRows) {
    const long long m = std::min(chunkRows, np - base);
    VSCUBLAS_CHECK(cublasSgemm(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(m), nlist, dim,
        &one, points + static_cast<size_t>(base) * dim, dim, centers, dim,
        &zero, dotsScratch, static_cast<int>(m)));
    const int block = 256;
    const int grid = static_cast<int>((m + block - 1) / block);
    assignNearestDotsKernel<<<grid, block>>>(dotsScratch, m, base, nlist,
                                             pointNorm, centerNorm, assignOut);
    syncCheck();
  }
}

// Build an IVF-Flat center table with a deterministic mini-batch Lloyd loop.
void lloydMiniBatch(cublasHandle_t handle, const float* sample,
                    long long sampleCount, int dim, int nlist,
                    const SearchConfig& cfg, float* centersOut,
                    double* trainMs) {
  if (nlist > sampleCount)
    throwRuntime("nlist exceeds the k-means training sample size");
  DevMem<float> pointNorm(sampleCount);
  DevMem<float> centerNorm(nlist);
  DevMem<float> centerSum(static_cast<size_t>(nlist) * dim);
  DevMem<int> counts(nlist);
  DevMem<int> assign(sampleCount);
  DevMem<int> assignPrev(sampleCount);
  DevMem<int> changed(1);
  const long long chunk =
      std::min<long long>(cfg.kmeans_chunk_rows, 32768);
  DevMem<float> dots(static_cast<size_t>(chunk) * nlist);

  // Initialize centers from deterministically spaced sample rows.
  const long long stride =
      std::max<long long>(1, sampleCount / std::max(1, nlist));
  {
    std::vector<float> host(static_cast<size_t>(nlist) * dim);
    for (int c = 0; c < nlist; ++c) {
      const long long pick = std::min<long long>(
          sampleCount - 1, static_cast<long long>(c) * stride);
      VSCU_CHECK(cudaMemcpy(
          host.data() + static_cast<size_t>(c) * dim,
          sample + static_cast<size_t>(pick) * dim,
          static_cast<size_t>(dim) * sizeof(float), cudaMemcpyDeviceToHost));
    }
    VSCU_CHECK(cudaMemcpy(centersOut, host.data(), host.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
  }
  launchSqNorm(sample, sampleCount, dim, pointNorm.get());
  Timer timer;

  const int maxIter = std::max(1, cfg.kmeans_iters);
  bool converged = false;
  for (int it = 0; it < maxIter && !converged; ++it) {
    launchSqNorm(centersOut, nlist, dim, centerNorm.get());
    VSCU_CHECK(cudaMemcpy(assignPrev.get(), assign.get(),
                          assign.bytes(), cudaMemcpyDeviceToDevice));
    launchAssignment(handle, sample, sampleCount, centersOut, nlist, dim,
                     pointNorm.get(), centerNorm.get(), chunk, assign.get(),
                     dots.get());
    zeroCountsAndSums(centerSum.get(), counts.get(), nlist, dim);
    {
      const int block = 256;
      const int grid = static_cast<int>((sampleCount + block - 1) / block);
      accumulateMiniBatchKernel<<<grid, block>>>(
          sample, assign.get(), sampleCount, dim, nlist, centerSum.get(),
          counts.get());
      syncCheck();
    }
    divideCentersKernel<<<(nlist + 255) / 256, 256>>>(centerSum.get(),
                                                      counts.get(), nlist, dim);
    syncCheck();
    VSCU_CHECK(cudaMemcpy(centersOut, centerSum.get(),
                          static_cast<size_t>(nlist) * dim * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    if (cfg.cosine_normalize_centers) {
      normalizeCenterKernel<<<(nlist + 255) / 256, 256>>>(centersOut, nlist,
                                                          dim);
      syncCheck();
    }
    copyFixEmptyCentersKernel<<<(nlist + 255) / 256, 256>>>(
        sample, sampleCount, counts.get(), centersOut, nlist, dim,
        cfg.kmeans_seed);
    syncCheck();

    if (cfg.kmeans_assign_frac > 0.0) {
      VSCU_CHECK(cudaMemset(changed.get(), 0, sizeof(int)));
      const int block = 256;
      const int grid = static_cast<int>((sampleCount + block - 1) / block);
      assignmentChangedKernel<<<grid, block>>>(
          assign.get(), assignPrev.get(), sampleCount, changed.get());
      syncCheck();
      int hc = 0;
      VSCU_CHECK(cudaMemcpy(&hc, changed.get(), sizeof(hc),
                            cudaMemcpyDeviceToHost));
      if (static_cast<double>(hc) <=
          cfg.kmeans_assign_frac * static_cast<double>(sampleCount))
        converged = true;
    }
  }
  launchSqNorm(centersOut, nlist, dim, centerNorm.get());
  if (trainMs) *trainMs = timer.ms();
}

// ---------------------------------------------------------------------------
// Inverted-list construction.
// ---------------------------------------------------------------------------
__global__ void histogramClustersKernel(const int* __restrict__ assign,
                                        long long n, int* __restrict__ histOut) {
  const long long p = static_cast<long long>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
  if (p < n)
    atomicAdd(&histOut[assign[p]], 1);
}

// Initialize per-list 32-bit insertion cursors from the 64-bit list starts.
// CoreX (Iluvatar) devices do not implement 64-bit atomicAdd, so the cursor is
// kept in 32 bits; vector ids are bounded by INT32_MAX by the engine, which
// also bounds every per-list offset.
__global__ void initListCursor32Kernel(
    const long long* __restrict__ listOffsets, int nlist,
    int* __restrict__ cursor32) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < nlist) cursor32[c] = static_cast<int>(listOffsets[c]);
}

__global__ void fillInvertedListsKernel(
    const int* __restrict__ assign, long long n,
    const long long* __restrict__ listOffsets,
    int* __restrict__ cursor32, int* __restrict__ listIds) {
  const long long p = static_cast<long long>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
  if (p >= n) return;
  const int c = assign[p];
  const int pos = atomicAdd(&cursor32[c], 1);
  listIds[static_cast<size_t>(pos)] = static_cast<int>(p);
}

__global__ void sumToTotalKernel(const int* __restrict__ in, long long count,
                                 unsigned int* __restrict__ accOut) {
  unsigned int acc = 0;
  for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += static_cast<long long>(blockDim.x) * gridDim.x)
    acc += in[i];
  if (acc != 0) atomicAdd(accOut, acc);
}

long long buildInvertedLists(const int* assign, long long n, int nlist,
                             TempStore* temp, long long* listOffsetsOut,
                             int* listIdsOut) {
  DevMem<int> hist(nlist);
  VSCU_CHECK(cudaMemset(hist.get(), 0, hist.bytes()));
  {
    const int block = 256;
    const int grid = static_cast<int>((n + block - 1) / block);
    histogramClustersKernel<<<grid, block>>>(assign, n, hist.get());
    syncCheck();
  }
  // Exclusive scan: listOffsetsOut[0..nlist].
  size_t tmpBytes = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, tmpBytes, hist.get(),
                                listOffsetsOut, nlist, 0);
  void* tmp = temp->ensure(tmpBytes);
  cub::DeviceScan::ExclusiveSum(tmp, tmpBytes, hist.get(), listOffsetsOut,
                                nlist, 0);
  syncCheck();
  DevMem<unsigned int> totalDev(1);
  VSCU_CHECK(cudaMemset(totalDev.get(), 0, sizeof(unsigned int)));
  {
    const int block = 256;
    sumToTotalKernel<<<static_cast<unsigned>((nlist + block - 1) / block),
                       block>>>(hist.get(), nlist, totalDev.get());
    syncCheck();
  }
  unsigned int hTotal = 0;
  VSCU_CHECK(cudaMemcpy(&hTotal, totalDev.get(), sizeof(hTotal),
                        cudaMemcpyDeviceToHost));
  const long long total = static_cast<long long>(hTotal);
  const long long totalLL = total;
  VSCU_CHECK(cudaMemcpy(listOffsetsOut + nlist, &totalLL, sizeof(totalLL),
                        cudaMemcpyHostToDevice));
  DevMem<int> cursor32(nlist);
  initListCursor32Kernel<<<(nlist + 255) / 256, 256>>>(listOffsetsOut, nlist,
                                                       cursor32.get());
  syncCheck();
  {
    const int block = 256;
    const int grid = static_cast<int>((n + block - 1) / block);
    fillInvertedListsKernel<<<grid, block>>>(assign, n, listOffsetsOut,
                                             cursor32.get(), listIdsOut);
    syncCheck();
  }
  return total;
}

// ---------------------------------------------------------------------------
// IVF-PQ kernels and compact k-means used to learn sub-codebooks.
// ---------------------------------------------------------------------------
__global__ void pqAssignKernel(const float* __restrict__ points,
                               long long np, int dims, int ks,
                               const float* __restrict__ centers,
                               float* __restrict__ sums,
                               int* __restrict__ counts) {
  const long long p = static_cast<long long>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
  if (p >= np) return;
  const float* r = points + static_cast<size_t>(p) * dims;
  float best = 1e30f;
  int bestC = 0;
  for (int c = 0; c < ks; ++c) {
    const float* cc = centers + static_cast<size_t>(c) * dims;
    float d = 0.f;
    for (int j = 0; j < dims; ++j) {
      const float diff = r[j] - cc[j];
      d += diff * diff;
    }
    if (d < best) {
      best = d;
      bestC = c;
    }
  }
  float* s = sums + static_cast<size_t>(bestC) * dims;
  for (int j = 0; j < dims; ++j) atomicAdd(s + j, r[j]);
  atomicAdd(&counts[bestC], 1);
}

void pqLloyd(const float* points, long long np, int dims, int ks, int seed,
             int iters, float* centersOut, double* msOut) {
  if (ks > np) throwRuntime("PQ codebook larger than training sample");
  DevMem<float> sums(static_cast<size_t>(ks) * dims);
  DevMem<int> counts(ks);
  {
    std::vector<float> host(static_cast<size_t>(ks) * dims);
    const long long stride = std::max<long long>(1, np / ks);
    for (int c = 0; c < ks; ++c) {
      const long long pick = std::min<long long>(np - 1, stride * c);
      VSCU_CHECK(cudaMemcpy(host.data() + static_cast<size_t>(c) * dims,
                            points + static_cast<size_t>(pick) * dims,
                            static_cast<size_t>(dims) * sizeof(float),
                            cudaMemcpyDeviceToHost));
    }
    VSCU_CHECK(cudaMemcpy(centersOut, host.data(), host.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
  }
  Timer timer;
  const int block = 256;
  for (int it = 0; it < iters; ++it) {
    VSCU_CHECK(cudaMemset(sums.get(), 0, sums.bytes()));
    VSCU_CHECK(cudaMemset(counts.get(), 0, counts.bytes()));
    const int grid = static_cast<int>((np + block - 1) / block);
    pqAssignKernel<<<grid, block>>>(points, np, dims, ks, centersOut,
                                    sums.get(), counts.get());
    syncCheck();
    // Divide and copy back.
    divideCentersKernel<<<(ks + 255) / 256, 256>>>(sums.get(), counts.get(),
                                                   ks, dims);
    syncCheck();
    VSCU_CHECK(cudaMemcpy(centersOut, sums.get(),
                          static_cast<size_t>(ks) * dims * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    copyFixEmptyCentersKernel<<<(ks + 255) / 256, 256>>>(
        points, np, counts.get(), centersOut, ks, dims, seed);
    syncCheck();
  }
  if (msOut) *msOut = timer.ms();
}

__global__ void gatherSubspaceKernel(const float* __restrict__ full,
                                     long long np, int fullDim, int subDim,
                                     int subspace, float* __restrict__ out) {
  const long long p = static_cast<long long>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
  if (p >= np) return;
  const float* src =
      full + static_cast<size_t>(p) * fullDim + subspace * subDim;
  float* dst = out + static_cast<size_t>(p) * subDim;
  for (int j = 0; j < subDim; ++j) dst[j] = src[j];
}

__global__ void pqEncodeKernel(const float* __restrict__ full, long long n,
                               int dim, int m, int subDim, int ks,
                               const float* __restrict__ codebooks,
                               unsigned char* __restrict__ codesOut) {
  const long long p = static_cast<long long>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
  if (p >= n) return;
  const float* r = full + static_cast<size_t>(p) * dim;
  unsigned char* code = codesOut + static_cast<size_t>(p) * m;
  for (int s = 0; s < m; ++s) {
    const float* q = r + s * subDim;
    const float* cb = codebooks + (static_cast<size_t>(s) * ks) * subDim;
    float best = 1e30f;
    int bestC = 0;
    for (int c = 0; c < ks; ++c) {
      const float* cc = cb + static_cast<size_t>(c) * subDim;
      float d = 0.f;
      for (int j = 0; j < subDim; ++j) {
        const float diff = q[j] - cc[j];
        d += diff * diff;
      }
      if (d < best) {
        best = d;
        bestC = c;
      }
    }
    code[s] = static_cast<unsigned char>(bestC);
  }
}

__global__ void pqTableKernel(const float* __restrict__ queries, int nq,
                              int dim, int m, int subDim, int ks,
                              const float* __restrict__ codebooks,
                              float* __restrict__ tableOut, int metricCode) {
  const int qi = blockIdx.y;
  const int subspace = blockIdx.x;
  const int c = threadIdx.x;
  if (c >= ks) return;
  const float* q =
      queries + static_cast<size_t>(qi) * dim + subspace * subDim;
  const float* cb =
      codebooks + (static_cast<size_t>(subspace) * ks + c) * subDim;
  float acc = 0.f;
  if (metricCode == 0) {
    for (int j = 0; j < subDim; ++j) {
      const float diff = q[j] - cb[j];
      acc += diff * diff;
    }
  } else {
    for (int j = 0; j < subDim; ++j) acc += q[j] * cb[j];
  }
  tableOut[(static_cast<size_t>(qi) * m + subspace) * ks + c] = acc;
}

__global__ void pqPackKeysKernel(
    const int* __restrict__ candidates,
    const unsigned char* __restrict__ codes,
    const float* __restrict__ table, const long long* __restrict__ queryStart,
    int nQuery, int m, int ks, int metricCode, long long total,
    unsigned long long* __restrict__ keysOut) {
  const long long p = static_cast<long long>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
  if (p >= total) return;
  const int qi = locateQuery(queryStart, nQuery, p);
  const int vid = candidates[p];
  const unsigned char* code = codes + static_cast<size_t>(vid) * m;
  const float* tbl = table + static_cast<size_t>(qi) * m * ks;
  float acc = 0.f;
  for (int s = 0; s < m; ++s) acc += tbl[static_cast<size_t>(s) * ks + code[s]];
  if (metricCode == 2) acc = 1.f - acc;
  keysOut[p] = packKey(acc, metricSmaller(metricCode), vid);
}

// Collect the W best approximate candidates of every query (serial over the
// small per-batch query count) and compact them for exact rescoring.
__global__ void rerankPrepKernel(
    const unsigned long long* __restrict__ sortedApprox,
    const long long* __restrict__ origStart, int nQuery, int rerankWidth,
    long long* __restrict__ rerankStart, int* __restrict__ rerankCandidates) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  long long cum = 0;
  for (int qi = 0; qi < nQuery; ++qi) {
    rerankStart[qi] = cum;
    const long long begin = origStart[qi];
    const long long count = origStart[qi + 1] - begin;
    const long long takeLL = count < rerankWidth ? count : rerankWidth;
    const int take = static_cast<int>(takeLL);
    for (int j = 0; j < take; ++j)
      rerankCandidates[cum + j] =
          unpackId(sortedApprox[begin + j]);
    cum += take;
  }
  rerankStart[nQuery] = cum;
}

// ---------------------------------------------------------------------------
// CUB segmented sort wrappers.
// ---------------------------------------------------------------------------
void segmentedRadixSortKeys(TempStore* temp, const unsigned long long* in,
                            unsigned long long* out, const long long* offsets,
                            long long numItems, int numSegments) {
  size_t tmpBytes = 0;
  cub::DeviceSegmentedRadixSort::SortKeys(
      nullptr, tmpBytes, in, out, numItems, numSegments, offsets, offsets + 1,
      0, 64, 0);
  void* tmp = temp->ensure(tmpBytes);
  cub::DeviceSegmentedRadixSort::SortKeys(
      tmp, tmpBytes, in, out, numItems, numSegments, offsets, offsets + 1, 0,
      64, 0);
  syncCheck();
}

void exclusiveSum(TempStore* temp, const long long* in, long long* out,
                  long long count) {
  if (count <= 0) return;
  if (count > static_cast<long long>(std::numeric_limits<int>::max()))
    throwRuntime("exclusiveSum: segment count exceeds int range");
  const int n = static_cast<int>(count);
  size_t tmpBytes = 0;
  // CUB writes `count` prefix outputs into out[0..count-1]. Callers also need
  // out[count] (total), which CUB may not write, so pass an n+1 scratch
  // buffer and then append the total computed from the last prefix and the
  // final element.
  DevMem<long long> scanBuf(static_cast<size_t>(n) + 1);
  cub::DeviceScan::ExclusiveSum(nullptr, tmpBytes, in, scanBuf.get(), n, 0);
  void* tmp = temp->ensure(tmpBytes);
  cub::DeviceScan::ExclusiveSum(tmp, tmpBytes, in, scanBuf.get(), n, 0);
  syncCheck();
  VSCU_CHECK(cudaMemcpy(out, scanBuf.get(),
                        static_cast<size_t>(n) * sizeof(long long),
                        cudaMemcpyDeviceToDevice));
  long long prefixLast = 0, inLast = 0;
  VSCU_CHECK(cudaMemcpy(&prefixLast, scanBuf.get() + (n - 1),
                        sizeof(long long), cudaMemcpyDeviceToHost));
  VSCU_CHECK(cudaMemcpy(&inLast, in + (n - 1), sizeof(long long),
                        cudaMemcpyDeviceToHost));
  const long long total = prefixLast + inLast;
  VSCU_CHECK(cudaMemcpy(out + n, &total, sizeof(total),
                        cudaMemcpyHostToDevice));
}

// ---------------------------------------------------------------------------
// Search implementation helpers.
// ---------------------------------------------------------------------------
void hostQueryNorms(const GpuEngine::Impl& im, const float* queries, i64 nq,
                    std::vector<double>* norms) {
  norms->assign(static_cast<size_t>(nq), 0.0);
  for (i64 qi = 0; qi < nq; ++qi) {
    const float* q = queries + static_cast<size_t>(qi) * im.dim;
    double s = 0.0;
    for (int d = 0; d < im.dim; ++d)
      s += static_cast<double>(q[d]) * q[d];
    (*norms)[static_cast<size_t>(qi)] = s;
  }
}

void uploadQueryBatch(GpuEngine::Impl& im, const float* queries, i64 nq,
                      DevMem<float>* qDev, DevMem<double>* qNormDev) {
  qDev->resize(static_cast<size_t>(nq) * im.dim);
  std::vector<float> host(static_cast<size_t>(nq) * im.dim);
  std::memcpy(host.data(), queries,
              static_cast<size_t>(nq) * im.dim * sizeof(float));
  if (im.metric == Metric::Cosine)
    normalizeRowsToUnit(host.data(), nq, im.dim);
  VSCU_CHECK(cudaMemcpy(qDev->get(), host.data(), qDev->bytes(),
                        cudaMemcpyHostToDevice));

  std::vector<double> norms;
  hostQueryNorms(im, host.data(), nq, &norms);
  qNormDev->resize(static_cast<size_t>(nq));
  VSCU_CHECK(cudaMemcpy(qNormDev->get(), norms.data(),
                        static_cast<size_t>(nq) * sizeof(double),
                        cudaMemcpyHostToDevice));
}

int metricCode(Metric m) {
  if (m == Metric::L2) return 0;
  if (m == Metric::InnerProduct) return 1;
  return 2;
}

// One exact chunk: nq queries, all n vectors.
void exactChunk(GpuEngine::Impl& im, const float* queries, i64 nq, int topK,
                int metricCodeInt, TempStore* temp, i32* idsOut,
                float* scoresOut) {
  DevMem<float> qDev;
  DevMem<double> qNormDev;
  uploadQueryBatch(im, queries, nq, &qDev, &qNormDev);
  const long long items = static_cast<long long>(nq) * im.n;
  DevMem<unsigned long long> keysIn(items), keysOut(items);
  DevMem<long long> offsets(nq + 1);
  for (i64 qi = 0; qi <= nq; ++qi) {
    const long long v = im.n * qi;
    VSCU_CHECK(cudaMemcpy(offsets.get() + qi, &v,
                          sizeof(long long), cudaMemcpyHostToDevice));
  }

  const int block = 256;
  {
    const dim3 grid(static_cast<unsigned>((im.n + block - 1) / block),
                    static_cast<unsigned>(nq));
    exactKeysKernel<<<grid, block>>>(
        im.x.get(), qDev.get(), qNormDev.get(), im.n, im.dim, metricCodeInt,
        keysIn.get());
    syncCheck();
  }
  segmentedRadixSortKeys(temp, keysIn.get(), keysOut.get(), offsets.get(),
                         items, static_cast<int>(nq));
  DevMem<int> dIds(static_cast<size_t>(nq) * topK);
  DevMem<float> dScores(static_cast<size_t>(nq) * topK);
  {
    const dim3 grid((topK + 63) / 64, static_cast<unsigned>(nq));
    decodeTopKeysKernel<<<grid, 64>>>(keysOut.get(), im.n,
                                      static_cast<int>(nq), topK, metricCodeInt,
                                      dIds.get(), dScores.get());
    syncCheck();
  }
  VSCU_CHECK(cudaMemcpy(idsOut, dIds.get(), dIds.bytes(),
                        cudaMemcpyDeviceToHost));
  VSCU_CHECK(cudaMemcpy(scoresOut, dScores.get(), dScores.bytes(),
                        cudaMemcpyDeviceToHost));
}

// Returns total number of gathered candidates.
long long ivfProbeAndGather(GpuEngine::Impl& im,
                            const float* qDev, const double* qNormDev, i64 nq,
                            int metricCodeInt, int nprobe,
                            TempStore* temp, DevMem<int>* probeIds,
                            DevMem<int>* candidates,
                            DevMem<long long>* queryStart) {
  const long long items = static_cast<long long>(nq) * im.nlist;
  DevMem<unsigned long long> centerKeys(items), centerKeysSorted(items);
  DevMem<long long> offsets(nq + 1);
  for (i64 qi = 0; qi <= nq; ++qi) {
    const long long v = static_cast<long long>(im.nlist) * qi;
    VSCU_CHECK(cudaMemcpy(offsets.get() + qi, &v,
                          sizeof(long long), cudaMemcpyHostToDevice));
  }
  {
    const int block = 256;
    const dim3 grid((im.nlist + block - 1) / block,
                    static_cast<unsigned>(nq));
    ivfCenterKeysKernel<<<grid, block>>>(
        qDev, qNormDev, im.centers.get(), im.centerNorm.get(), im.dim,
        im.nlist, metricCodeInt, centerKeys.get());
    syncCheck();
  }
  segmentedRadixSortKeys(temp, centerKeys.get(), centerKeysSorted.get(),
                         offsets.get(), items, static_cast<int>(nq));

  const long long nSlots = static_cast<long long>(nq) * nprobe;
  probeIds->resize(static_cast<size_t>(nSlots));
  {
    const dim3 grid(1, static_cast<unsigned>(nq));
    extractFirstKernel<<<grid, 256>>>(centerKeysSorted.get(), im.nlist,
                                      static_cast<int>(nq), nprobe,
                                      probeIds->get());
    syncCheck();
  }

  DevMem<long long> probeCounts(nSlots), probeOffsets(nSlots + 1);
  {
    const int block = 256;
    probeCountsKernel<<<static_cast<unsigned>((nSlots + block - 1) / block),
                        block>>>(probeIds->get(), im.listOffsets.get(),
                                 static_cast<int>(nSlots), probeCounts.get());
    syncCheck();
  }
  exclusiveSum(temp, probeCounts.get(), probeOffsets.get(), nSlots);
  queryStart->resize(static_cast<size_t>(nq) + 1);
  {
    const int block = 256;
    queryStartKernel<<<static_cast<unsigned>((nq + block) / block), block>>>(
        probeOffsets.get(), nprobe, static_cast<int>(nq), queryStart->get());
    syncCheck();
  }
  long long total = 0;
  VSCU_CHECK(cudaMemcpy(&total, probeOffsets.get() + nSlots, sizeof(total),
                        cudaMemcpyDeviceToHost));
  if (total == 0) return 0;
  candidates->resize(static_cast<size_t>(total));
  gatherCandidatesKernel<<<static_cast<unsigned>(nSlots), 256>>>(
      probeIds->get(), probeOffsets.get(), im.listOffsets.get(),
      im.listIds.get(), static_cast<int>(nSlots), candidates->get());
  syncCheck();
  return total;
}

void ivfFlatChunk(GpuEngine::Impl& im, const float* queries, i64 nq, int topK,
                  int metricCodeInt, TempStore* temp, i32* idsOut,
                  float* scoresOut) {
  DevMem<float> qDev;
  DevMem<double> qNormDev;
  uploadQueryBatch(im, queries, nq, &qDev, &qNormDev);
  DevMem<int> probeIds, candidates;
  DevMem<long long> queryStart;
  const int nprobe = std::min<int>(im.cfg.nprobe, im.nlist);
  const long long total = ivfProbeAndGather(
      im, qDev.get(), qNormDev.get(), nq, metricCodeInt, nprobe, temp,
      &probeIds, &candidates, &queryStart);
  if (total == 0) return;

  DevMem<unsigned long long> keysIn(total), keysOut(total);
  {
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    ivfFlatKeysKernel<<<grid, block>>>(
        candidates.get(), im.x.get(), qDev.get(), qNormDev.get(),
        queryStart.get(), static_cast<int>(nq), im.dim, metricCodeInt, total,
        keysIn.get());
    syncCheck();
  }
  segmentedRadixSortKeys(temp, keysIn.get(), keysOut.get(), queryStart.get(),
                         total, static_cast<int>(nq));
  DevMem<int> dIds(static_cast<size_t>(nq) * topK);
  DevMem<float> dScores(static_cast<size_t>(nq) * topK);
  {
    const dim3 grid((topK + 63) / 64, static_cast<unsigned>(nq));
    decodeVariableTopKernel<<<grid, 64>>>(keysOut.get(), queryStart.get(),
                                          static_cast<int>(nq), topK,
                                          metricCodeInt, dIds.get(),
                                          dScores.get());
    syncCheck();
  }
  VSCU_CHECK(cudaMemcpy(idsOut, dIds.get(), dIds.bytes(),
                        cudaMemcpyDeviceToHost));
  VSCU_CHECK(cudaMemcpy(scoresOut, dScores.get(), dScores.bytes(),
                        cudaMemcpyDeviceToHost));
}

void ivfPqChunk(GpuEngine::Impl& im, const float* queries, i64 nq, int topK,
                int metricCodeInt, TempStore* temp, i32* idsOut,
                float* scoresOut) {
  DevMem<float> qDev;
  DevMem<double> qNormDev;
  uploadQueryBatch(im, queries, nq, &qDev, &qNormDev);
  DevMem<int> probeIds, candidates;
  DevMem<long long> queryStart;
  const int nprobe = std::min<int>(im.cfg.nprobe, im.nlist);
  const long long total = ivfProbeAndGather(
      im, qDev.get(), qNormDev.get(), nq, metricCodeInt, nprobe, temp,
      &probeIds, &candidates, &queryStart);
  if (total == 0) return;

  const int m = im.pqM;
  const int subDim = im.pqSubDim;
  const int ks = im.pqKs;
  DevMem<float> table(static_cast<size_t>(nq) * m * ks);
  {
    const dim3 grid(m, static_cast<unsigned>(nq));
    pqTableKernel<<<grid, ks>>>(qDev.get(), static_cast<int>(nq), im.dim, m,
                                subDim, ks, im.pqCodebooks.get(),
                                table.get(), metricCodeInt);
    syncCheck();
  }
  DevMem<unsigned long long> keysIn(total), keysOut(total);
  {
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    pqPackKeysKernel<<<grid, block>>>(
        candidates.get(), im.pqCodes.get(), table.get(), queryStart.get(),
        static_cast<int>(nq), m, ks, metricCodeInt, total, keysIn.get());
    syncCheck();
  }
  segmentedRadixSortKeys(temp, keysIn.get(), keysOut.get(), queryStart.get(),
                         total, static_cast<int>(nq));
  DevMem<int> dIds(static_cast<size_t>(nq) * topK);
  DevMem<float> dScores(static_cast<size_t>(nq) * topK);
  if (im.cfg.pq_rerank > 0) {
    // ADC (approximate) selection followed by exact rescoring on the top
    // rerank_width candidates.
    const int rerankWidth = im.cfg.pq_rerank;
    DevMem<long long> rerankStart(static_cast<size_t>(nq) + 1);
    DevMem<int> rerankCandidates(static_cast<size_t>(total));
    rerankPrepKernel<<<1, 1>>>(keysOut.get(), queryStart.get(),
                               static_cast<int>(nq), rerankWidth,
                               rerankStart.get(), rerankCandidates.get());
    syncCheck();
    long long rerankTotal = 0;
    VSCU_CHECK(cudaMemcpy(&rerankTotal, rerankStart.get() + nq,
                          sizeof(rerankTotal), cudaMemcpyDeviceToHost));
    if (rerankTotal > 0) {
      DevMem<unsigned long long> rerankKeys(rerankTotal),
          rerankKeysSorted(rerankTotal);
      {
        const int block = 256;
        const int grid = static_cast<int>((rerankTotal + block - 1) / block);
        ivfFlatKeysKernel<<<grid, block>>>(
            rerankCandidates.get(), im.x.get(), qDev.get(), qNormDev.get(),
            rerankStart.get(), static_cast<int>(nq), im.dim, metricCodeInt,
            rerankTotal, rerankKeys.get());
        syncCheck();
      }
      segmentedRadixSortKeys(temp, rerankKeys.get(),
                             rerankKeysSorted.get(), rerankStart.get(),
                             rerankTotal, static_cast<int>(nq));
      {
        const dim3 grid((topK + 63) / 64, static_cast<unsigned>(nq));
        decodeVariableTopKernel<<<grid, 64>>>(
            rerankKeysSorted.get(), rerankStart.get(), static_cast<int>(nq),
            topK, metricCodeInt, dIds.get(), dScores.get());
        syncCheck();
      }
      VSCU_CHECK(cudaMemcpy(idsOut, dIds.get(), dIds.bytes(),
                            cudaMemcpyDeviceToHost));
      VSCU_CHECK(cudaMemcpy(scoresOut, dScores.get(), dScores.bytes(),
                            cudaMemcpyDeviceToHost));
      return;
    }
  }
  {
    const dim3 grid((topK + 63) / 64, static_cast<unsigned>(nq));
    decodeVariableTopKernel<<<grid, 64>>>(keysOut.get(), queryStart.get(),
                                          static_cast<int>(nq), topK,
                                          metricCodeInt, dIds.get(),
                                          dScores.get());
    syncCheck();
  }
  VSCU_CHECK(cudaMemcpy(idsOut, dIds.get(), dIds.bytes(),
                        cudaMemcpyDeviceToHost));
  VSCU_CHECK(cudaMemcpy(scoresOut, dScores.get(), dScores.bytes(),
                        cudaMemcpyDeviceToHost));
}

}  // namespace

namespace {

double percentileOf(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double idx = p * static_cast<double>(v.size() - 1);
  const size_t lo = static_cast<size_t>(idx);
  const size_t hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (v[hi] - v[lo]) * (idx - static_cast<double>(lo));
}

void writeLE32(std::FILE* f, i32 v) {
  unsigned char b[4];
  for (int i = 0; i < 4; ++i)
    b[i] = static_cast<unsigned char>((static_cast<unsigned>(v) >> (8 * i)) & 0xff);
  if (std::fwrite(b, 1, 4, f) != 4) throwRuntime("index write failed");
}

void writeLE64(std::FILE* f, i64 v) {
  unsigned char b[8];
  for (int i = 0; i < 8; ++i)
    b[i] = static_cast<unsigned char>((static_cast<unsigned long long>(v) >> (8 * i)) & 0xff);
  if (std::fwrite(b, 1, 8, f) != 8) throwRuntime("index write failed");
}

i32 readLE32(std::FILE* f) {
  unsigned char b[4];
  if (std::fread(b, 1, 4, f) != 4) throwRuntime("index read failed");
  return static_cast<i32>(b[0]) | (static_cast<i32>(b[1]) << 8) |
         (static_cast<i32>(b[2]) << 16) | (static_cast<i32>(b[3]) << 24);
}

i64 readLE64(std::FILE* f) {
  unsigned char b[8];
  if (std::fread(b, 1, 8, f) != 8) throwRuntime("index read failed");
  unsigned long long v = 0;
  for (int i = 7; i >= 0; --i) v = (v << 8) | b[i];
  return static_cast<i64>(v);
}

constexpr i32 kIndexMagic = 0x58495653;  // "VSIX"
constexpr i32 kIndexVersion = 1;

void readAll(FILE* f, void* p, size_t n) {
  if (std::fread(p, 1, n, f) != n) throwRuntime("index read failed");
}

void writeAll(FILE* f, const void* p, size_t n) {
  if (std::fwrite(p, 1, n, f) != n) throwRuntime("index write failed");
}

template <typename T>
void copyToHost(const T* dev, std::vector<T>* host, size_t n) {
  host->resize(n);
  if (n) VSCU_CHECK(cudaMemcpy(host->data(), dev, n * sizeof(T),
                               cudaMemcpyDeviceToHost));
}

}  // namespace

// ---------------------------------------------------------------------------
// GpuEngine
// ---------------------------------------------------------------------------
GpuEngine::GpuEngine(const Dataset& ds, const SearchConfig& cfg)
    : impl_(std::make_unique<Impl>(ds, cfg)) {}

GpuEngine::~GpuEngine() = default;
GpuEngine::GpuEngine(GpuEngine&&) noexcept = default;
GpuEngine& GpuEngine::operator=(GpuEngine&&) noexcept = default;

std::size_t GpuEngine::deviceTotalBytes() {
  size_t total = 0, freeBytes = 0;
  VSCU_CHECK(cudaMemGetInfo(&freeBytes, &total));
  return total;
}

std::size_t GpuEngine::deviceFreeBytes() {
  size_t total = 0, freeBytes = 0;
  VSCU_CHECK(cudaMemGetInfo(&freeBytes, &total));
  return freeBytes;
}

bool GpuEngine::hasIndex() const { return impl_->hasIndex_; }

IndexBuildStats GpuEngine::buildIndex(SearchMode mode) {
  Impl& im = *impl_;
  if (mode == SearchMode::Exact)
    throwRuntime("buildIndex requires an approximate search mode");
  // Drop stale index state before allocating fresh buffers.
  im.hasIndex_ = false;
  im.pqCodes.release();
  im.pqCodebooks.release();
  if (im.n > std::numeric_limits<i32>::max())
    throwRuntime("this build path supports n <= INT32_MAX");
  const int nlist = im.cfg.nlist;
  if (nlist <= 0 || nlist > 1000000 || nlist > im.n)
    throwRuntime("bad nlist for index build");
  im.nlist = nlist;
  im.indexKind = mode == SearchMode::IvfFlat ? IndexKind::IvfFlat
                                             : IndexKind::IvfPq;
  im.pqM = mode == SearchMode::IvfPq ? im.cfg.pq_m : 0;
  im.pqKs = mode == SearchMode::IvfPq ? im.cfg.pq_ks : 0;
  im.pqSubDim = mode == SearchMode::IvfPq ? im.dim / im.cfg.pq_m : 0;
  if (mode == SearchMode::IvfPq) {
    if (im.cfg.pq_m <= 0 || im.dim % im.cfg.pq_m != 0)
      throwRuntime("pq_m must divide the vector dimension");
    if (im.cfg.pq_ks < 1 || im.cfg.pq_ks > 256)
      throwRuntime("pq_ks must be in [1, 256]");
  }
  const long long chunkRows =
      std::min<long long>(std::max<long long>(1, im.cfg.kmeans_chunk_rows),
                          32768);

  IndexBuildStats stats;
  Timer wall;
  cublasHandle_t handle = nullptr;
  VSCUBLAS_CHECK(cublasCreate(&handle));

  // 1. Deterministic training sample.
  const i64 sampleCount =
      std::min<i64>(im.n, std::max<i64>(1, im.cfg.kmeans_sample));
  DevMem<float> sample(static_cast<size_t>(sampleCount) * im.dim);
  {
    const int block = 256;
    const int grid =
        static_cast<int>((sampleCount + block - 1) / block);
    gatherEvenSampleKernel<<<grid, block>>>(
        im.x.get(), im.n, sampleCount, im.dim, sample.get());
    syncCheck();
  }

  // 2. Train centers (IVF partition + IVF-PQ shared coarse index).
  SearchConfig buildCfg = im.cfg;
  buildCfg.cosine_normalize_centers = (im.metric == Metric::Cosine);
  im.centers.resize(static_cast<size_t>(nlist) * im.dim);
  lloydMiniBatch(handle, sample.get(), sampleCount, im.dim, nlist, buildCfg,
                 im.centers.get(), &stats.trainMs);

  // 3. Assign every vector to its nearest center and build inverted lists.
  DevMem<float> xNorm(static_cast<size_t>(im.n));
  DevMem<float> cNorm(nlist);
  DevMem<float> dots(static_cast<size_t>(chunkRows) * nlist);
  DevMem<int> assign(static_cast<size_t>(im.n));
  launchSqNorm(im.x.get(), im.n, im.dim, xNorm.get());
  launchSqNorm(im.centers.get(), nlist, im.dim, cNorm.get());
  {
    Timer assignTimer;
    launchAssignment(handle, im.x.get(), im.n, im.centers.get(), nlist,
                     im.dim, xNorm.get(), cNorm.get(),
                     chunkRows, assign.get(), dots.get());
    stats.assignMs = assignTimer.ms();
  }
  im.centerNorm.resize(nlist);
  VSCU_CHECK(cudaMemcpy(im.centerNorm.get(), cNorm.get(), cNorm.bytes(),
                        cudaMemcpyDeviceToDevice));
  im.listOffsets.resize(static_cast<size_t>(nlist) + 1);
  im.listIds.resize(static_cast<size_t>(im.n));
  {
    TempStore temp;
    const i64 total = buildInvertedLists(assign.get(), im.n, nlist, &temp,
                                         im.listOffsets.get(),
                                         im.listIds.get());
    if (total != im.n)
      throwRuntime("inverted list total mismatch: total=" +
                   std::to_string(total) + " n=" + std::to_string(im.n));
  }

  // 4. Optional PQ encoding.
  if (mode == SearchMode::IvfPq) {
    Timer pqTimer;
    const int m = im.pqM;
    const int subDim = im.pqSubDim;
    const int ks = im.pqKs;
    im.pqCodebooks.resize(static_cast<size_t>(m) * ks * subDim);
    DevMem<float> subPoints(static_cast<size_t>(sampleCount) * subDim);
    const int block = 256;
    const int grid = static_cast<int>((sampleCount + block - 1) / block);
    for (int s = 0; s < m; ++s) {
      gatherSubspaceKernel<<<grid, block>>>(
          sample.get(), sampleCount, im.dim, subDim, s, subPoints.get());
      syncCheck();
      pqLloyd(subPoints.get(), sampleCount, subDim, ks, im.cfg.kmeans_seed,
              std::max(1, im.cfg.kmeans_iters),
              im.pqCodebooks.get() + static_cast<size_t>(s) * ks * subDim,
              nullptr);
    }
    im.pqCodes.resize(static_cast<size_t>(im.n) * m);
    pqEncodeKernel<<<static_cast<unsigned>((im.n + block - 1) / block),
                     block>>>(im.x.get(), im.n, im.dim, m, subDim, ks,
                              im.pqCodebooks.get(), im.pqCodes.get());
    syncCheck();
    stats.pqEncodeMs = pqTimer.ms();
  }

  VSCUBLAS_CHECK(cublasDestroy(handle));
  im.hasIndex_ = true;
  stats.buildMs = wall.ms();
  stats.deviceBytes = im.indexDeviceBytes();
  return stats;
}

void GpuEngine::saveIndex(const std::string& path) const {
  const Impl& im = *impl_;
  if (!im.hasIndex_) throwRuntime("saveIndex: no index");
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throwRuntime("cannot open index for writing: " + path);
  try {
    writeLE32(f, kIndexMagic);
    writeLE32(f, kIndexVersion);
    writeLE32(f, static_cast<i32>(im.indexKind));
    writeLE32(f, static_cast<i32>(im.metric));
    writeLE64(f, im.n);
    writeLE32(f, im.dim);
    writeLE32(f, im.nlist);
    writeLE32(f, im.pqM);
    writeLE32(f, im.pqKs);
    writeLE32(f, im.pqSubDim);

    std::vector<float> hCenters, hNorm;
    std::vector<long long> hOffsets;
    std::vector<int> hIds;
    copyToHost(im.centers.get(), &hCenters,
               static_cast<size_t>(im.nlist) * im.dim);
    copyToHost(im.centerNorm.get(), &hNorm, im.nlist);
    copyToHost(im.listOffsets.get(), &hOffsets, im.nlist + 1);
    copyToHost(im.listIds.get(), &hIds, im.n);
    writeAll(f, hCenters.data(), hCenters.size() * sizeof(float));
    writeAll(f, hNorm.data(), hNorm.size() * sizeof(float));
    writeAll(f, hOffsets.data(), hOffsets.size() * sizeof(long long));
    writeAll(f, hIds.data(), hIds.size() * sizeof(int));
    if (im.indexKind == IndexKind::IvfPq) {
      std::vector<unsigned char> hCodes;
      std::vector<float> hBooks;
      copyToHost(im.pqCodes.get(), &hCodes,
                 static_cast<size_t>(im.n) * im.pqM);
      copyToHost(im.pqCodebooks.get(), &hBooks,
                 static_cast<size_t>(im.pqM) * im.pqKs * im.pqSubDim);
      writeAll(f, hCodes.data(), hCodes.size());
      writeAll(f, hBooks.data(), hBooks.size() * sizeof(float));
    }
  } catch (...) {
    std::fclose(f);
    throw;
  }
  std::fclose(f);
}

void GpuEngine::loadIndex(const std::string& path) {
  Impl& im = *impl_;
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throwRuntime("cannot open index for reading: " + path);
  try {
    const i32 magic = readLE32(f);
    const i32 version = readLE32(f);
    if (magic != kIndexMagic || version != kIndexVersion)
      throwRuntime("bad index file: " + path);
    const auto kind = static_cast<IndexKind>(readLE32(f));
    const auto met = static_cast<Metric>(readLE32(f));
    const i64 n = readLE64(f);
    const i32 dim = readLE32(f);
    const int nlist = readLE32(f);
    const int pqM = readLE32(f);
    const int pqKs = readLE32(f);
    const int pqSub = readLE32(f);
    if (n != im.n || dim != im.dim || met != im.metric || nlist <= 0)
      throwRuntime("index/dataset mismatch: " + path);

    im.hasIndex_ = false;
    im.indexKind = kind;
    im.nlist = nlist;
    im.pqM = pqM;
    im.pqKs = pqKs;
    im.pqSubDim = pqSub;
    std::vector<float> hCenters(static_cast<size_t>(nlist) * dim);
    std::vector<float> hNorm(nlist);
    std::vector<long long> hOffsets(nlist + 1);
    std::vector<int> hIds(static_cast<size_t>(n));
    readAll(f, hCenters.data(), hCenters.size() * sizeof(float));
    readAll(f, hNorm.data(), hNorm.size() * sizeof(float));
    readAll(f, hOffsets.data(), hOffsets.size() * sizeof(long long));
    readAll(f, hIds.data(), hIds.size() * sizeof(int));
    im.centers.resize(hCenters.size());
    im.centerNorm.resize(nlist);
    im.listOffsets.resize(hOffsets.size());
    im.listIds.resize(hIds.size());
    VSCU_CHECK(cudaMemcpy(im.centers.get(), hCenters.data(),
                          hCenters.size() * sizeof(float), cudaMemcpyHostToDevice));
    VSCU_CHECK(cudaMemcpy(im.centerNorm.get(), hNorm.data(),
                          hNorm.size() * sizeof(float), cudaMemcpyHostToDevice));
    VSCU_CHECK(cudaMemcpy(im.listOffsets.get(), hOffsets.data(),
                          hOffsets.size() * sizeof(long long),
                          cudaMemcpyHostToDevice));
    VSCU_CHECK(cudaMemcpy(im.listIds.get(), hIds.data(),
                          hIds.size() * sizeof(int), cudaMemcpyHostToDevice));
    if (kind == IndexKind::IvfPq) {
      std::vector<unsigned char> hCodes(static_cast<size_t>(n) * pqM);
      std::vector<float> hBooks(
          static_cast<size_t>(pqM) * pqKs * pqSub);
      readAll(f, hCodes.data(), hCodes.size());
      readAll(f, hBooks.data(), hBooks.size() * sizeof(float));
      im.pqCodes.resize(hCodes.size());
      im.pqCodebooks.resize(hBooks.size());
      VSCU_CHECK(cudaMemcpy(im.pqCodes.get(), hCodes.data(), hCodes.size(),
                            cudaMemcpyHostToDevice));
      VSCU_CHECK(cudaMemcpy(im.pqCodebooks.get(), hBooks.data(),
                            hBooks.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
    }
  } catch (...) {
    std::fclose(f);
    throw;
  }
  std::fclose(f);
  im.hasIndex_ = true;
}

void GpuEngine::searchAll(SearchMode mode, const float* queries, i64 nq,
                          int topK, std::vector<i32>* ids, FloatVec* scores,
                          SearchStats* stats, int batchOverride) {
  if (mode == SearchMode::Exact) {
    // no index required
  } else {
    impl_->requireIndex(mode);
  }
  if (topK <= 0 || nq <= 0) return;
  ids->assign(static_cast<size_t>(nq) * topK, -1);
  scores->assign(static_cast<size_t>(nq) * topK, 0.f);

  int batch = batchOverride > 0 ? batchOverride : impl_->cfg.batch_size;
  if (mode == SearchMode::Exact) {
    const i64 bytesPerQuery = impl_->n * 32;  // in + out packed keys
    const i64 memBudget = std::max<i64>(1, impl_->cfg.exact_memory_mb) *
                          (1024 * 1024);
    batch = static_cast<int>(
        std::min<i64>(batch, std::max<i64>(1, memBudget / bytesPerQuery)));
  }
  batch = static_cast<int>(std::clamp<i64>(batch, 1, nq));

  const int metricCodeInt = metricCode(impl_->metric);
  TempStore temp;
  SearchStats local;
  cudaEvent_t startEv = nullptr, stopEv = nullptr;
  VSCU_CHECK(cudaEventCreate(&startEv));
  VSCU_CHECK(cudaEventCreate(&stopEv));
  Timer wall;
  for (i64 q0 = 0; q0 < nq; q0 += batch) {
    const i64 chunk = std::min<i64>(batch, nq - q0);
    const float* qs = queries + static_cast<size_t>(q0) * impl_->dim;
    i32* idOut = ids->data() + static_cast<size_t>(q0) * topK;
    float* sOut = scores->data() + static_cast<size_t>(q0) * topK;
    VSCU_CHECK(cudaEventRecord(startEv, 0));
    switch (mode) {
      case SearchMode::Exact:
        exactChunk(*impl_, qs, chunk, topK, metricCodeInt, &temp, idOut,
                   sOut);
        break;
      case SearchMode::IvfFlat:
        ivfFlatChunk(*impl_, qs, chunk, topK, metricCodeInt, &temp, idOut,
                     sOut);
        break;
      case SearchMode::IvfPq:
        ivfPqChunk(*impl_, qs, chunk, topK, metricCodeInt, &temp, idOut,
                   sOut);
        break;
    }
    VSCU_CHECK(cudaEventRecord(stopEv, 0));
    VSCU_CHECK(cudaEventSynchronize(stopEv));
    float ms = 0.f;
    VSCU_CHECK(cudaEventElapsedTime(&ms, startEv, stopEv));
    local.batchMs.push_back(ms);
  }
  local.gpuMs = 0.0;
  for (double b : local.batchMs) local.gpuMs += b;
  local.wallMs = wall.ms();
  local.qps = local.wallMs > 0 ? nq / (local.wallMs / 1e3) : 0.0;
  local.meanBatchMs = local.batchMs.empty()
                          ? 0.0
                          : local.gpuMs / static_cast<double>(local.batchMs.size());
  local.p50BatchMs = percentileOf(local.batchMs, 0.50);
  local.p99BatchMs = percentileOf(local.batchMs, 0.99);
  size_t total = 0, freeBytes = 0;
  VSCU_CHECK(cudaMemGetInfo(&freeBytes, &total));
  local.deviceUsedBytes = total - freeBytes;
  VSCU_CHECK(cudaEventDestroy(startEv));
  VSCU_CHECK(cudaEventDestroy(stopEv));
  if (stats) *stats = std::move(local);
}

}  // namespace vs

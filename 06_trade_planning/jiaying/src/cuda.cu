#include "flow.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include <numeric>
#include <stdexcept>

namespace trade {
namespace {
using U64 = unsigned long long;
void check(cudaError_t code, const char *operation) {
  if (code != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(code));
}
#define CUDA(call) check((call), #call)
template <class T> struct Buffer {
  T *p = nullptr;
  size_t count;
  explicit Buffer(size_t n) : count(n) {
    if (n)
      CUDA(cudaMalloc(&p, n * sizeof(T)));
  }
  ~Buffer() {
    if (p)
      cudaFree(p);
  }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  void upload(const std::vector<T> &v, cudaStream_t stream) {
    if (count)
      CUDA(cudaMemcpyAsync(p, v.data(), count * sizeof(T),
                           cudaMemcpyHostToDevice, stream));
  }
};

__global__ void saturate_source(const int *row, const int *col, const int *rev,
                                int *residual, U64 *excess, int source) {
  for (int64_t k = int64_t(row[source]) + blockIdx.x * blockDim.x + threadIdx.x;
       k < row[source + 1]; k += int64_t(gridDim.x) * blockDim.x) {
    const int e = int(k), v = col[e];
    if (v == source)
      continue;
    const int amount = residual[e];
    // The source row contains reverse arcs too, but those start with zero.
    // Separate original antiparallel edges have separate residual pairs.
    if (amount > 0) {
      residual[e] = 0;
      residual[rev[e]] += amount;
      atomicAdd(excess + v, U64(amount));
    }
  }
}

__global__ void init_labels(int n, int target, int *height, int *frontier) {
  const int u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u < n)
    height[u] = u == target ? 0 : n;
  if (u == 0)
    frontier[0] = target;
}

// Reverse residual BFS computes exact distance to the sink. Each warp visits
// one frontier vertex, avoiding serialization at a high-degree trade hub.
__global__ void reverse_bfs(const int *row, const int *col, const int *rev,
                            const int *residual, int *height,
                            const int *frontier, int count, int *next,
                            int *next_count, int n, int source, int distance) {
  const int lane = threadIdx.x & 31;
  const int index = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if (index >= count)
    return;
  const int u = frontier[index];
  for (int64_t k = int64_t(row[u]) + lane; k < row[u + 1]; k += 32) {
    const int e = int(k), v = col[e];
    if (v != source && residual[rev[e]] > 0 &&
        atomicCAS(height + v, n, distance) == n)
      next[atomicAdd(next_count, 1)] = v;
  }
}

// Heights and each vertex's starting excess remain a snapshot for the round.
// Incoming flow has its own buffer. Opposite pushes on a residual pair cannot
// both be admissible with fixed heights. Atomic residual reads also make
// concurrent reverse updates explicitly race-free under the CUDA memory model.
__global__ void push_thread(int n, const int *row, const int *col,
                            const int *rev, int *residual, U64 *excess,
                            U64 *incoming, const int *height,
                            const int *terminals) {
  const int u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= n || u == terminals[0] || u == terminals[1] || height[u] >= n)
    return;
  U64 left = excess[u];
  if (!left)
    return;
  for (int e = row[u]; e < row[u + 1] && left; ++e) {
    const int v = col[e];
    if (height[u] != height[v] + 1)
      continue;
    const int capacity = atomicAdd(residual + e, 0);
    const int amount = int(min(left, U64(capacity)));
    if (amount) {
      atomicSub(residual + e, amount);
      atomicAdd(residual + rev[e], amount);
      atomicAdd(incoming + v, U64(amount));
      left -= amount;
    }
  }
  excess[u] = left;
}

__global__ void push_warp(int n, const int *row, const int *col, const int *rev,
                          int *residual, U64 *excess, U64 *incoming,
                          const int *height, const int *terminals) {
  const int lane = threadIdx.x & 31;
  const int u = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if (u >= n || u == terminals[0] || u == terminals[1] || height[u] >= n)
    return;
  U64 left = excess[u];
  if (!left)
    return;
  for (int64_t base = row[u]; base < row[u + 1] && left; base += 32) {
    const int64_t k = base + lane;
    int v = 0, capacity = 0;
    if (k < row[u + 1]) {
      v = col[k];
      if (height[u] == height[v] + 1)
        capacity = atomicAdd(residual + k, 0);
    }
    U64 prefix = U64(capacity);
    for (int offset = 1; offset < 32; offset *= 2) {
      const U64 previous = __shfl_up_sync(0xffffffff, prefix, offset);
      if (lane >= offset)
        prefix += previous;
    }
    const U64 before = prefix - U64(capacity);
    const int amount =
        before < left ? int(min(U64(capacity), left - before)) : 0;
    const U64 total = __shfl_sync(0xffffffff, prefix, 31);
    if (amount) {
      atomicSub(residual + k, amount);
      atomicAdd(residual + rev[k], amount);
      atomicAdd(incoming + v, U64(amount));
    }
    left -= min(left, total);
  }
  if (lane == 0)
    excess[u] = left;
}

__global__ void update_relabel(int n, const int *row, const int *col,
                               const int *residual, U64 *excess, U64 *incoming,
                               const int *old_height, int *new_height,
                               const int *terminals, int *active_blocks,
                               bool count_active) {
  const int u = blockIdx.x * blockDim.x + threadIdx.x;
  bool active = false;
  if (u < n) {
    excess[u] += incoming[u];
    incoming[u] = 0;
    int h = old_height[u];
    if (u != terminals[0] && u != terminals[1] && excess[u] && h < n) {
      int best = n;
      for (int e = row[u]; e < row[u + 1]; ++e)
        if (residual[e] > 0)
          best = min(best, old_height[col[e]] + 1);
      h = best;
      active = h < n;
    }
    new_height[u] = h;
  }
  if (count_active) {
    const int count = __syncthreads_count(active);
    if (threadIdx.x == 0)
      active_blocks[blockIdx.x] = count;
  }
}
} // namespace

struct GpuSolver::Impl {
  const ResidualGraph &g;
  GpuOptions options;
  Buffer<int> row, col, rev, original, residual, height_a, height_b;
  Buffer<int> frontier_a, frontier_b, next_count, terminals, active;
  Buffer<U64> excess, incoming;
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  int *host_active = nullptr;
  std::string name;
  int blocks;

  Impl(const ResidualGraph &input, GpuOptions opt)
      : g(input), options(opt), row(g.row.size()), col(g.col.size()),
        rev(g.rev.size()), original(g.cap.size()), residual(g.cap.size()),
        height_a(g.n), height_b(g.n), frontier_a(g.n), frontier_b(g.n),
        next_count(1), terminals(2), active((g.n + 255LL) / 256), excess(g.n),
        incoming(g.n), blocks(int((g.n + 255LL) / 256)) {
    try {
      CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      CUDA(cudaMallocHost(&host_active, active.count * sizeof(int)));
      cudaDeviceProp prop{};
      CUDA(cudaGetDeviceProperties(&prop, 0));
      name = prop.name;
      row.upload(g.row, stream);
      col.upload(g.col, stream);
      rev.upload(g.rev, stream);
      original.upload(g.cap, stream);
      CUDA(cudaStreamSynchronize(stream));
      if (options.graphs) {
        CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        launch_batch();
        CUDA(cudaStreamEndCapture(stream, &graph));
        CUDA(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        CUDA(cudaGraphUpload(executable, stream));
        CUDA(cudaStreamSynchronize(stream));
      }
    } catch (...) {
      release();
      throw;
    }
  }
  void release() {
    if (executable)
      cudaGraphExecDestroy(executable);
    if (graph)
      cudaGraphDestroy(graph);
    if (host_active)
      cudaFreeHost(host_active);
    if (stream)
      cudaStreamDestroy(stream);
  }
  ~Impl() { release(); }

  void launch_batch() {
    int *h = height_a.p, *next_h = height_b.p;
    for (int i = 0; i < options.batch_rounds; ++i) {
      if (options.warp)
        push_warp<<<int((int64_t(g.n) * 32 + 255) / 256), 256, 0, stream>>>(
            g.n, row.p, col.p, rev.p, residual.p, excess.p, incoming.p, h,
            terminals.p);
      else
        push_thread<<<blocks, 256, 0, stream>>>(g.n, row.p, col.p, rev.p,
                                                residual.p, excess.p,
                                                incoming.p, h, terminals.p);
      update_relabel<<<blocks, 256, 0, stream>>>(
          g.n, row.p, col.p, residual.p, excess.p, incoming.p, h, next_h,
          terminals.p, active.p, i == options.batch_rounds - 1);
      std::swap(h, next_h);
    }
    CUDA(cudaGetLastError());
  }

  void global_relabel(int s, int t) {
    init_labels<<<blocks, 256, 0, stream>>>(g.n, t, height_a.p, frontier_a.p);
    int count = 1, distance = 1;
    int *front = frontier_a.p, *next = frontier_b.p;
    while (count) {
      CUDA(cudaMemsetAsync(next_count.p, 0, sizeof(int), stream));
      reverse_bfs<<<int((int64_t(count) * 32 + 255) / 256), 256, 0, stream>>>(
          row.p, col.p, rev.p, residual.p, height_a.p, front, count, next,
          next_count.p, g.n, s, distance);
      CUDA(cudaMemcpyAsync(host_active, next_count.p, sizeof(int),
                           cudaMemcpyDeviceToHost, stream));
      CUDA(cudaStreamSynchronize(stream));
      count = host_active[0];
      std::swap(front, next);
      ++distance;
    }
  }

  QueryResult solve(int s, int t) {
    const auto start = Clock::now();
    const int endpoints[2] = {s, t};
    CUDA(cudaMemcpyAsync(terminals.p, endpoints, sizeof(endpoints),
                         cudaMemcpyHostToDevice, stream));
    if (original.count)
      CUDA(cudaMemcpyAsync(residual.p, original.p, original.count * sizeof(int),
                           cudaMemcpyDeviceToDevice, stream));
    CUDA(cudaMemsetAsync(excess.p, 0, excess.count * sizeof(U64), stream));
    CUDA(cudaMemsetAsync(incoming.p, 0, incoming.count * sizeof(U64), stream));
    saturate_source<<<std::max(1, int((g.row[s + 1] - g.row[s] + 255LL) / 256)),
                      256, 0, stream>>>(row.p, col.p, rev.p, residual.p,
                                        excess.p, s);
    QueryResult result;
    global_relabel(s, t);
    ++result.global_relabels;
    for (;;) {
      if (options.graphs)
        CUDA(cudaGraphLaunch(executable, stream));
      else
        launch_batch();
      CUDA(cudaMemcpyAsync(host_active, active.p, active.count * sizeof(int),
                           cudaMemcpyDeviceToHost, stream));
      CUDA(cudaStreamSynchronize(stream));
      result.rounds += options.batch_rounds;
      if (std::all_of(host_active, host_active + active.count,
                      [](int x) { return x == 0; }))
        break;
      if (result.rounds >= uint64_t(options.max_rounds))
        throw std::runtime_error(
            "Maximum round limit reached; no approximate result is emitted");
      if (result.rounds % options.global_interval == 0) {
        global_relabel(s, t);
        ++result.global_relabels;
      }
    }
    U64 value = 0;
    CUDA(cudaMemcpyAsync(&value, excess.p + t, sizeof(value),
                         cudaMemcpyDeviceToHost, stream));
    CUDA(cudaStreamSynchronize(stream));
    result.flow = int64_t(value);
    result.ms = elapsed_ms(start);
    return result;
  }
};

GpuSolver::GpuSolver(const ResidualGraph &g, GpuOptions options) {
  if (options.batch_rounds < 2 || options.batch_rounds > 1024 ||
      options.batch_rounds % 2 ||
      options.global_interval < options.batch_rounds ||
      options.global_interval % options.batch_rounds ||
      options.max_rounds < options.batch_rounds)
    throw std::runtime_error(
        "Batch must be even in [2,1024]; global interval must be a positive "
        "multiple of batch; max rounds >= batch");
  impl_ = std::make_unique<Impl>(g, options);
}
GpuSolver::~GpuSolver() = default;
QueryResult GpuSolver::solve(int s, int t) { return impl_->solve(s, t); }
std::string GpuSolver::device_name() const { return impl_->name; }
size_t GpuSolver::device_bytes() const {
  const auto &p = *impl_;
  return sizeof(int) * (p.row.count + 4 * p.col.count + 4ULL * p.g.n + 3 +
                        p.active.count) +
         2 * sizeof(U64) * p.g.n;
}

void GpuSolver::verify(int s, int t, int64_t expected) const {
  const auto &p = *impl_;
  const auto &g = p.g;
  std::vector<int> r(g.cap.size());
  std::vector<U64> device_excess(g.n);
  if (!r.empty())
    CUDA(cudaMemcpy(r.data(), p.residual.p, r.size() * sizeof(int),
                    cudaMemcpyDeviceToHost));
  CUDA(cudaMemcpy(device_excess.data(), p.excess.p, g.n * sizeof(U64),
                  cudaMemcpyDeviceToHost));
  std::vector<int64_t> balance(g.n, 0);
  for (int u = 0; u < g.n; ++u)
    for (int e = g.row[u]; e < g.row[u + 1]; ++e) {
      if (r[e] < 0 ||
          int64_t(r[e]) + r[g.rev[e]] != int64_t(g.cap[e]) + g.cap[g.rev[e]])
        throw std::runtime_error("Certificate: invalid residual capacity pair");
      if (g.cap[e] > 0) {
        const int64_t f = int64_t(g.cap[e]) - r[e];
        if (f < 0 || f > g.cap[e])
          throw std::runtime_error("Certificate: capacity violation");
        balance[u] -= f;
        balance[g.col[e]] += f;
      }
    }
  std::vector<unsigned char> reaches_sink(g.n, 0);
  std::vector<int> q{t};
  reaches_sink[t] = 1;
  for (size_t i = 0; i < q.size(); ++i) {
    const int u = q[i];
    for (int e = g.row[u]; e < g.row[u + 1]; ++e) {
      const int v = g.col[e];
      if (r[g.rev[e]] > 0 && !reaches_sink[v]) {
        reaches_sink[v] = 1;
        q.push_back(v);
      }
    }
  }
  if (reaches_sink[s])
    throw std::runtime_error("Certificate: residual s-t path exists");
  int64_t cut = 0;
  for (int u = 0; u < g.n; ++u) {
    if (u != s && (balance[u] < 0 || U64(balance[u]) != device_excess[u]))
      throw std::runtime_error("Certificate: preflow balance mismatch");
    if (u != t && reaches_sink[u] && balance[u] != 0)
      throw std::runtime_error("Certificate: unfinished excess can reach sink");
    if (!reaches_sink[u])
      for (int e = g.row[u]; e < g.row[u + 1]; ++e)
        if (reaches_sink[g.col[e]])
          cut += g.cap[e];
  }
  if (cut != expected || balance[t] != expected)
    throw std::runtime_error(
        "Certificate: cut capacity differs from sink flow");
}
} // namespace trade

#include "nbody.hpp"
#include <chrono>
#include <cuda_runtime.h>
#include <stdexcept>

namespace nb {
namespace {
void check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}
#define CUDA(call) check((call), #call)
template <typename T> struct DeviceBuffer {
  T *ptr = nullptr;
  explicit DeviceBuffer(std::size_t n) {
    CUDA(cudaMalloc(&ptr, n * sizeof(T)));
  }
  ~DeviceBuffer() {
    if (ptr)
      cudaFree(ptr);
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
};
struct PinnedBuffer {
  Vec4 *ptr = nullptr;
  explicit PinnedBuffer(std::size_t n) {
    CUDA(cudaMallocHost(&ptr, n * sizeof(Vec4)));
  }
  ~PinnedBuffer() {
    if (ptr)
      cudaFreeHost(ptr);
  }
};
struct Event {
  cudaEvent_t value{};
  Event() { CUDA(cudaEventCreate(&value)); }
  ~Event() { cudaEventDestroy(value); }
};
__device__ __forceinline__ float3 interact(float4 pi, float4 pj, float3 a,
                                           float eps2, float G) {
  float dx = pj.x - pi.x, dy = pj.y - pi.y, dz = pj.z - pi.z;
  float inv = rsqrtf(dx * dx + dy * dy + dz * dz + eps2);
  float scale = G * pj.w * inv * inv * inv;
  a.x += dx * scale;
  a.y += dy * scale;
  a.z += dz * scale;
  return a;
}
// Every thread participates in both barriers, including threads beyond N.
// Positions are read-only here: a separate drift kernel prevents global races.
template <int Block, bool Tiled>
__global__ void acceleration(const float4 *__restrict__ p,
                             float4 *__restrict__ a, float4 *__restrict__ v,
                             int n, float eps2, float G, float kick) {
  __shared__ float4 tile[Block];
  const int i = blockIdx.x * Block + threadIdx.x;
  float4 pi = i < n ? p[i] : make_float4(0, 0, 0, 0);
  float3 sum = make_float3(0, 0, 0);
  if constexpr (Tiled) {
    for (int base = 0; base < n; base += Block) {
      const int j = base + threadIdx.x;
      tile[threadIdx.x] = j < n ? p[j] : make_float4(0, 0, 0, 0);
      __syncthreads();
      const int count = min(Block, n - base);
      if (i < n) {
#pragma unroll 8
        for (int k = 0; k < count; ++k)
          if (base + k != i)
            sum = interact(pi, tile[k], sum, eps2, G);
      }
      __syncthreads();
    }
  } else if (i < n) {
    for (int j = 0; j < n; ++j)
      if (j != i)
        sum = interact(pi, p[j], sum, eps2, G);
  }
  if (i < n) {
    a[i] = make_float4(sum.x, sum.y, sum.z, 0);
    if (kick != 0) {
      float4 vi = v[i];
      vi.x += kick * sum.x;
      vi.y += kick * sum.y;
      vi.z += kick * sum.z;
      v[i] = vi;
    }
  }
}
__global__ void drift(float4 *p, float4 *v, const float4 *a, int n, float dt,
                      bool leapfrog) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float4 pi = p[i], vi = v[i], ai = a[i];
  if (leapfrog) {
    vi.x += 0.5f * dt * ai.x;
    vi.y += 0.5f * dt * ai.y;
    vi.z += 0.5f * dt * ai.z;
  }
  pi.x += dt * vi.x;
  pi.y += dt * vi.y;
  pi.z += dt * vi.z;
  if (!leapfrog) {
    vi.x += dt * ai.x;
    vi.y += dt * ai.y;
    vi.z += dt * ai.z;
  }
  p[i] = pi;
  v[i] = vi;
}
template <int Block>
void launch(const float4 *p, float4 *a, float4 *v, int n, const Config &c,
            bool tiled, float kick) {
  if (tiled)
    acceleration<Block, true><<<(n + Block - 1) / Block, Block>>>(
        p, a, v, n, c.softening * c.softening, c.G, kick);
  else
    acceleration<Block, false><<<(n + Block - 1) / Block, Block>>>(
        p, a, v, n, c.softening * c.softening, c.G, kick);
  CUDA(cudaGetLastError());
}
} // namespace
Timing run_cuda(State &s, const Config &c, const Options &o,
                const Record &record) {
  static_assert(sizeof(Vec4) == sizeof(float4));
  using Clock = std::chrono::steady_clock;
  cudaDeviceProp prop{};
  CUDA(cudaGetDeviceProperties(&prop, 0));
  int n = static_cast<int>(s.position.size());
  std::size_t bytes = s.position.size() * sizeof(float4);
  DeviceBuffer<float4> p(n), v(n), a(n);
  PinnedBuffer host(n);
  Event start, stop;
  CUDA(cudaMemcpy(p.ptr, s.position.data(), bytes, cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(v.ptr, s.velocity.data(), bytes, cudaMemcpyHostToDevice));
  Timing timing;
  timing.device = prop.name;
  timing.device_bytes = bytes * 3;
  auto begin = Clock::now();
  if (record)
    record(0, s.position.data());
  auto force = [&](float kick) {
    if (o.block_size == 128)
      launch<128>(p.ptr, a.ptr, v.ptr, n, c, o.kernel == "tiled", kick);
    else
      launch<256>(p.ptr, a.ptr, v.ptr, n, c, o.kernel == "tiled", kick);
  };
  auto finish_segment = [&]() {
    CUDA(cudaEventRecord(stop.value));
    CUDA(cudaEventSynchronize(stop.value));
    float ms = 0;
    CUDA(cudaEventElapsedTime(&ms, start.value, stop.value));
    timing.compute_ms += ms;
  };
  CUDA(cudaEventRecord(start.value));
  const bool leapfrog = c.integrator == "leapfrog";
  if (c.steps > 0 && leapfrog)
    force(0);
  for (int step = 1; step <= c.steps; ++step) {
    if (!leapfrog)
      force(0);
    drift<<<(n + 255) / 256, 256>>>(p.ptr, v.ptr, a.ptr, n, c.dt, leapfrog);
    CUDA(cudaGetLastError());
    if (leapfrog)
      force(0.5f * c.dt); // Fuse the final half kick with force write-back.
    if (record && (step % c.record_interval == 0 || step == c.steps)) {
      finish_segment();
      CUDA(cudaMemcpy(host.ptr, p.ptr, bytes, cudaMemcpyDeviceToHost));
      record(step, host.ptr);
      CUDA(cudaEventRecord(start.value));
    }
  }
  finish_segment();
  CUDA(cudaMemcpy(s.position.data(), p.ptr, bytes, cudaMemcpyDeviceToHost));
  CUDA(cudaMemcpy(s.velocity.data(), v.ptr, bytes, cudaMemcpyDeviceToHost));
  timing.wall_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
  return timing;
}
} // namespace nb

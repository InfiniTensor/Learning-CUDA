#include "model.hpp"
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdexcept>

namespace pricing {
namespace {
void check(cudaError_t status, const char *call) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(call) + ": " +
                             cudaGetErrorString(status));
}
#define CUDA_CHECK(call) check((call), #call)
struct CurandBits {
  curandStatePhilox4_32_10_t state;
  __device__ CurandBits(uint64_t seed, uint64_t id) {
    curand_init(seed, id, 0, &state);
  }
  __device__ Words next() {
    const uint4 v = curand4(&state);
    return {v.x, v.y, v.z, v.w};
  }
};
struct Buffer {
  Moments *ptr = nullptr;
  explicit Buffer(size_t n) {
    CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(Moments)));
  }
  ~Buffer() { cudaFree(ptr); }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
};
struct Event {
  cudaEvent_t value = nullptr;
  Event() { CUDA_CHECK(cudaEventCreate(&value)); }
  ~Event() { cudaEventDestroy(value); }
  Event(const Event &) = delete;
  Event &operator=(const Event &) = delete;
};
__device__ Moments warp_reduce(Moments value) {
  for (int delta = 16; delta > 0; delta /= 2) {
    Moments other;
    other.n = __shfl_down_sync(0xffffffff,
                               static_cast<unsigned long long>(value.n), delta);
    for (int k = 0; k < 5; ++k) {
      other.mean[k] = __shfl_down_sync(0xffffffff, value.mean[k], delta);
      other.m2[k] = __shfl_down_sync(0xffffffff, value.m2[k], delta);
    }
    other.cross = __shfl_down_sync(0xffffffff, value.cross, delta);
    if ((threadIdx.x & 31) + delta < 32)
      value.merge(other);
  }
  return value;
}
__device__ Moments block_reduce(Moments value) {
  __shared__ Moments warps[32];
  value = warp_reduce(value);
  if ((threadIdx.x & 31) == 0)
    warps[threadIdx.x / 32] = value;
  __syncthreads();
  Moments aggregate;
  if (threadIdx.x < blockDim.x / 32)
    aggregate = warps[threadIdx.x];
  if (threadIdx.x < 32)
    aggregate = warp_reduce(aggregate);
  return aggregate;
}
template <class Bits, bool Anti, bool Control>
__global__ __launch_bounds__(512) void simulate(Parameters p, uint64_t seed,
                                                uint64_t n, uint64_t offset,
                                                Moments *blocks) {
  Moments local;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += uint64_t(gridDim.x) * blockDim.x) {
    double v[5];
    observation<Bits, Anti, Control>(p, seed, i + offset, v);
    local.add(v);
  }
  const Moments merged = block_reduce(local);
  if (threadIdx.x == 0)
    blocks[blockIdx.x] = merged;
}
__global__ void reduce_blocks(const Moments *blocks, int n, Moments *result) {
  Moments local;
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    local.merge(blocks[i]);
  const Moments merged = block_reduce(local);
  if (threadIdx.x == 0)
    *result = merged;
}
template <class Bits>
void launch(const Parameters &p, const Simulation &s, uint64_t n,
            uint64_t offset, int blocks, bool anti, Moments *output) {
  if (s.reduction == Reduction::ControlVariate)
    simulate<Bits, false, true>
        <<<blocks, s.block_size>>>(p, s.seed, n, offset, output);
  else if (anti)
    simulate<Bits, true, false>
        <<<blocks, s.block_size>>>(p, s.seed, n, offset, output);
  else
    simulate<Bits, false, false>
        <<<blocks, s.block_size>>>(p, s.seed, n, offset, output);
}
} // namespace
Run run_gpu(const Option &o, const Simulation &s) {
  using Clock = std::chrono::steady_clock;
  Run r;
  const auto setup_start = Clock::now();
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaFree(nullptr));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  r.setup_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - setup_start)
          .count();
  r.device = prop.name;
  r.block_size = s.block_size;
  if (s.block_size > prop.maxThreadsPerBlock)
    throw std::runtime_error("block_size exceeds GPU limit");
  const bool anti = s.reduction == Reduction::Antithetic;
  const uint64_t n = anti ? s.paths / 2 : s.paths;
  r.blocks = s.blocks
                 ? s.blocks
                 : std::min<uint64_t>((n + s.block_size - 1) / s.block_size,
                                      prop.multiProcessorCount * 16);
  const auto start = Clock::now();
  {
    Buffer partial(r.blocks), result(1);
    r.workspace_bytes = (r.blocks + 1) * sizeof(Moments);
    Event a, b, c;
    const auto p = parameters(o, s);
    auto batch = [&](uint64_t count, uint64_t offset, Moments &destination,
                     bool paired) {
      const int blocks = std::min<uint64_t>(
          r.blocks, (count + s.block_size - 1) / s.block_size);
      CUDA_CHECK(cudaEventRecord(a.value));
      if (s.curand)
        launch<CurandBits>(p, s, count, offset, blocks, paired, partial.ptr);
      else
        launch<CustomBits>(p, s, count, offset, blocks, paired, partial.ptr);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaEventRecord(b.value));
      reduce_blocks<<<1, 256>>>(partial.ptr, blocks, result.ptr);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaEventRecord(c.value));
      CUDA_CHECK(cudaEventSynchronize(c.value));
      float simulation_ms, reduction_ms;
      CUDA_CHECK(cudaEventElapsedTime(&simulation_ms, a.value, b.value));
      CUDA_CHECK(cudaEventElapsedTime(&reduction_ms, b.value, c.value));
      r.simulation_ms += simulation_ms;
      r.reduction_ms += reduction_ms;
      CUDA_CHECK(cudaMemcpy(&destination, result.ptr, sizeof(Moments),
                            cudaMemcpyDeviceToHost));
    };
    if (s.reduction == Reduction::ControlVariate)
      batch(s.pilot_paths, uint64_t(1) << 63, r.pilot, false);
    batch(n, 0, r.samples, anti);
  }
  r.total_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  return r;
}
} // namespace pricing

#include "nbody.hpp"
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>

namespace nbody {
namespace {
using Clock = std::chrono::steady_clock;
void check(cudaError_t result, const char *operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define CUDA_CHECK(call) check((call), #call)

template <class T> struct DeviceBuffer {
    T *data = nullptr;
    explicit DeviceBuffer(std::size_t count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
};
struct Event {
    cudaEvent_t value{};
    Event() { CUDA_CHECK(cudaEventCreate(&value)); }
    ~Event() { cudaEventDestroy(value); }
};
struct Stream {
    cudaStream_t value{};
    Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
    Stream(const Stream &) = delete;
    Stream &operator=(const Stream &) = delete;
};
struct Graph {
    cudaGraph_t value{};
    cudaGraphExec_t executable{};
    Graph() = default;
    ~Graph() {
        if (executable)
            cudaGraphExecDestroy(executable);
        if (value)
            cudaGraphDestroy(value);
    }
    Graph(const Graph &) = delete;
    Graph &operator=(const Graph &) = delete;
};

__device__ __forceinline__ void interact(Vec4 p, Vec4 q, float G, float eps2, Vec4 &a) {
    if (q.w == 0 || G == 0)
        return;
    const float dx = q.x - p.x, dy = q.y - p.y, dz = q.z - p.z;
    const float inv = rsqrtf(dx * dx + dy * dy + dz * dz + eps2);
    const float scale = G * q.w * inv * inv * inv;
    a.x += dx * scale;
    a.y += dy * scale;
    a.z += dz * scale;
}

// Each thread owns velocity[i]; force evaluation only reads positions, so the
// final half kick can safely share the force kernel without a grid barrier.
__device__ __forceinline__ void store_acceleration(Vec4 *acceleration, Vec4 *velocity,
                                                  int i, Vec4 a, float half_dt) {
    acceleration[i] = a;
    if (half_dt != 0) {
        velocity[i].x += half_dt * a.x;
        velocity[i].y += half_dt * a.y;
        velocity[i].z += half_dt * a.z;
    }
}

template <int Tile>
__global__ void acceleration_tiled(const Vec4 *__restrict__ position,
                                   Vec4 *__restrict__ acceleration, Vec4 *velocity,
                                   int n, float G, float eps2, float half_dt) {
    __shared__ Vec4 tile[Tile];
    const int i = blockIdx.x * Tile + threadIdx.x;
    const Vec4 p = i < n ? position[i] : Vec4{};
    Vec4 a{};
    // Inactive lanes still participate in BOTH barriers, including a partial last block.
    for (int base = 0; base < n; base += Tile) {
        const int j = base + threadIdx.x;
        tile[threadIdx.x] = j < n ? position[j] : Vec4{};
        __syncthreads();
        if (i < n) {
            // Only the tile containing this target needs self exclusion.
            // Keep source order unchanged while removing per-pair comparisons
            // from all other tiles (including their zero-mass padding).
            if (i >= base && i < base + Tile) {
#pragma unroll 32
                for (int k = 0; k < Tile; ++k) {
                    if (base + k != i)
                        interact(p, tile[k], G, eps2, a);
                }
            } else {
#pragma unroll 32
                for (int k = 0; k < Tile; ++k)
                    interact(p, tile[k], G, eps2, a);
            }
        }
        __syncthreads();
    }
    if (i < n)
        store_acceleration(acceleration, velocity, i, a, half_dt);
}

__global__ void acceleration_naive(const Vec4 *position, Vec4 *acceleration, Vec4 *velocity,
                                   int n, float G, float eps2, float half_dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const Vec4 p = position[i];
    Vec4 a{};
    for (int j = 0; j < n; ++j)
        if (i != j)
            interact(p, position[j], G, eps2, a);
    store_acceleration(acceleration, velocity, i, a, half_dt);
}

__global__ void drift_kick(Vec4 *position, Vec4 *velocity, const Vec4 *accel, int n, float dt,
                           bool euler) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    auto p = position[i];
    auto v = velocity[i];
    const auto a = accel[i];
    if (euler) {
        p.x += dt * v.x;
        p.y += dt * v.y;
        p.z += dt * v.z;
        v.x += dt * a.x;
        v.y += dt * a.y;
        v.z += dt * a.z;
    } else {
        v.x += 0.5f * dt * a.x;
        v.y += 0.5f * dt * a.y;
        v.z += 0.5f * dt * a.z;
        p.x += dt * v.x;
        p.y += dt * v.y;
        p.z += dt * v.z;
    }
    position[i] = p;
    velocity[i] = v;
}

} // namespace

std::string cuda_device_info() {
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    int runtime = 0, driver = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    std::ostringstream out;
    out << "{\"name\":" << json_string(p.name) << ",\"compute_capability\":\"" << p.major << '.'
        << p.minor << "\",\"total_memory_bytes\":" << p.totalGlobalMem
        << ",\"multiprocessors\":" << p.multiProcessorCount << ",\"runtime_version\":" << runtime
        << ",\"driver_version\":" << driver << '}';
    return out.str();
}

Timing run_cuda(State &s, const Config &c, const Options &o, const Recorder &record) {
    const auto wall_start = Clock::now();
    Timing timing;
    cudaDeviceProp property{};
    CUDA_CHECK(cudaGetDeviceProperties(&property, 0));
    timing.device = property.name;
    Stream stream;
    const int n = static_cast<int>(s.position.size());
    const auto bytes = s.position.size() * sizeof(Vec4);
    timing.device_bytes = 3 * bytes;
    DeviceBuffer<Vec4> position(n), velocity(n), accel(n);
    const bool zero_gravity = c.G == 0;
    const auto copy_start = Clock::now();
    CUDA_CHECK(cudaMemcpy(position.data, s.position.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(velocity.data, s.velocity.data(), bytes, cudaMemcpyHostToDevice));
    timing.transfer_seconds = std::chrono::duration<double>(Clock::now() - copy_start).count();
    if (zero_gravity)
        CUDA_CHECK(cudaMemsetAsync(accel.data, 0, bytes, stream.value));
    Event start, stop;
    const int block = o.block_size;
    const int grid = (n + block - 1) / block;
    auto acceleration = [&](float half_dt = 0) {
        if (zero_gravity)
            return;
        const float eps2 = c.softening * c.softening;
        if (o.kernel == "naive")
            acceleration_naive<<<grid, block, 0, stream.value>>>(
                position.data, accel.data, velocity.data, n, c.G, eps2, half_dt);
        else if (block == 128)
            acceleration_tiled<128><<<grid, 128, 0, stream.value>>>(
                position.data, accel.data, velocity.data, n, c.G, eps2, half_dt);
        else if (block == 512)
            acceleration_tiled<512><<<grid, 512, 0, stream.value>>>(
                position.data, accel.data, velocity.data, n, c.G, eps2, half_dt);
        else
            acceleration_tiled<256><<<grid, 256, 0, stream.value>>>(
                position.data, accel.data, velocity.data, n, c.G, eps2, half_dt);
        CUDA_CHECK(cudaGetLastError());
    };
    auto launch_step = [&]() {
        if (c.integrator == "euler")
            acceleration();
        drift_kick<<<grid, block, 0, stream.value>>>(
            position.data, velocity.data, accel.data, n, c.dt, c.integrator == "euler");
        CUDA_CHECK(cudaGetLastError());
        if (c.integrator == "leapfrog")
            acceleration(0.5f * c.dt);
    };
    Graph graph;
    const int span = record ? std::min(c.num_steps, c.record_interval) : c.num_steps;
    int batch = 0;
    std::int64_t best_submissions = std::int64_t(span) * 2 + 1;
    // One graph submission replaces two kernel submissions per step. Fit the
    // recording interval when beneficial: 200 steps use batches of 25, avoiding
    // 8 direct tail steps at every frame with a fixed batch of 32.
    for (int candidate = 1; candidate <= std::min(32, span); ++candidate) {
        const std::int64_t submissions = span / candidate + 2 * (span % candidate);
        if (submissions <= best_submissions) {
            batch = candidate;
            best_submissions = submissions;
        }
    }
    const bool use_graph = batch > 0 && (o.cuda_execution == "graph" ||
        (o.cuda_execution == "auto" && n <= 256 && c.num_steps >= 128 && batch >= 4));
    if (use_graph) {
        const auto setup_start = Clock::now();
        CUDA_CHECK(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
        try {
            for (int i = 0; i < batch; ++i)
                launch_step();
            CUDA_CHECK(cudaStreamEndCapture(stream.value, &graph.value));
        } catch (...) {
            // End a failed capture before destroying its stream/resources.
            cudaGraph_t abandoned{};
            cudaStreamEndCapture(stream.value, &abandoned);
            if (abandoned)
                cudaGraphDestroy(abandoned);
            throw;
        }
        CUDA_CHECK(cudaGraphInstantiateWithFlags(&graph.executable, graph.value, 0));
        CUDA_CHECK(cudaGraphUpload(graph.executable, stream.value));
        CUDA_CHECK(cudaStreamSynchronize(stream.value));
        timing.graph_setup_seconds = std::chrono::duration<double>(Clock::now() - setup_start).count();
        timing.graph_batch_steps = batch;
    }
    auto host_segment_start = Clock::now();
    auto begin_segment = [&]() {
        host_segment_start = Clock::now();
        CUDA_CHECK(cudaEventRecord(start.value, stream.value));
    };
    auto end_segment = [&]() {
        CUDA_CHECK(cudaEventRecord(stop.value, stream.value));
        CUDA_CHECK(cudaEventSynchronize(stop.value));
        timing.host_compute_seconds +=
            std::chrono::duration<double>(Clock::now() - host_segment_start).count();
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start.value, stop.value));
        timing.compute_seconds += milliseconds / 1000.0;
    };
    begin_segment();
    if (c.integrator == "leapfrog")
        acceleration();
    int step = 0;
    while (step < c.num_steps) {
        // Never batch across an output boundary, including an incomplete final interval.
        const int until_record = record ? c.record_interval - step % c.record_interval : c.num_steps - step;
        const int available = std::min(c.num_steps - step, until_record);
        if (use_graph && available >= batch) {
            CUDA_CHECK(cudaGraphLaunch(graph.executable, stream.value));
            step += batch;
            ++timing.graph_launches;
        } else {
            launch_step();
            ++step;
            ++timing.direct_steps;
        }
        if (record && (step % c.record_interval == 0 || step == c.num_steps)) {
            end_segment();
            const auto transfer_start = Clock::now();
            CUDA_CHECK(cudaMemcpy(s.position.data(), position.data, bytes, cudaMemcpyDeviceToHost));
            timing.transfer_seconds +=
                std::chrono::duration<double>(Clock::now() - transfer_start).count();
            record(step, s.position);
            begin_segment();
        }
    }
    end_segment();
    const auto transfer_start = Clock::now();
    if (!record)
        CUDA_CHECK(cudaMemcpy(s.position.data(), position.data, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(s.velocity.data(), velocity.data, bytes, cudaMemcpyDeviceToHost));
    timing.transfer_seconds += std::chrono::duration<double>(Clock::now() - transfer_start).count();
    timing.backend_wall_seconds = std::chrono::duration<double>(Clock::now() - wall_start).count();
    return timing;
}
} // namespace nbody

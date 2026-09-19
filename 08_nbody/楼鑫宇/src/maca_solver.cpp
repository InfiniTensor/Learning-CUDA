#include "nbody/maca_solver.hpp"

#include <mc_runtime.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nbody {
namespace {

using Clock = std::chrono::steady_clock;

enum class ForceKernel { Naive, Tiled };

// Local PODs make host/device layout explicit; no CUDA headers are required.
struct alignas(16) Position {
    float x, y, z, mass;
};

struct Velocity {
    float x, y, z;
};

void check_maca(mcError_t status, const char* expression, int line) {
    if (status != mcSuccess) {
        throw std::runtime_error(std::string("MACA 错误，行 ") +
                                 std::to_string(line) + "，" + expression +
                                 ": " + mcGetErrorString(status));
    }
}

#define NBODY_MACA_CHECK(expression) \
    check_maca((expression), #expression, __LINE__)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        NBODY_MACA_CHECK(mcMalloc(reinterpret_cast<void**>(&data_),
                                  count * sizeof(T)));
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) mcFree(data_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() { return data_; }

private:
    T* data_ = nullptr;
};

class EventPair {
public:
    EventPair() {
        NBODY_MACA_CHECK(mcEventCreate(&start_));
        try {
            NBODY_MACA_CHECK(mcEventCreate(&stop_));
        } catch (...) {
            mcEventDestroy(start_);
            throw;
        }
    }

    ~EventPair() {
        mcEventDestroy(start_);
        mcEventDestroy(stop_);
    }

    EventPair(const EventPair&) = delete;
    EventPair& operator=(const EventPair&) = delete;

    mcEvent_t start() const { return start_; }
    mcEvent_t stop() const { return stop_; }

private:
    mcEvent_t start_{};
    mcEvent_t stop_{};
};

__global__ void force_naive_kernel(const Position* positions,
                                   Velocity* accelerations,
                                   int particle_count,
                                   float gravitational_constant,
                                   float softening_squared) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) return;

    const Position target = positions[i];
    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    for (int j = 0; j < particle_count; ++j) {
        if (j == i) continue;
        const Position source = positions[j];
        const float dx = source.x - target.x;
        const float dy = source.y - target.y;
        const float dz = source.z - target.z;
        const float distance_squared =
            dx * dx + dy * dy + dz * dz + softening_squared;
        const float inverse_distance = rsqrtf(distance_squared);
        const float scale = gravitational_constant * source.mass *
                            inverse_distance * inverse_distance * inverse_distance;
        ax += dx * scale;
        ay += dy * scale;
        az += dz * scale;
    }
    accelerations[i] = {ax, ay, az};
}

__global__ void force_tiled_kernel(const Position* positions,
                                   Velocity* accelerations,
                                   int particle_count,
                                   float gravitational_constant,
                                   float softening_squared) {
    extern __shared__ Position tile[];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool valid_target = i < particle_count;
    const Position target = valid_target ? positions[i] : Position{0, 0, 0, 0};
    float ax = 0.0f, ay = 0.0f, az = 0.0f;

    for (int tile_begin = 0; tile_begin < particle_count;
         tile_begin += blockDim.x) {
        const int source_index = tile_begin + threadIdx.x;
        tile[threadIdx.x] = source_index < particle_count
                                ? positions[source_index]
                                : Position{0, 0, 0, 0};
        __syncthreads();

        if (valid_target) {
            const int remaining = particle_count - tile_begin;
            const int tile_count = remaining < blockDim.x ? remaining : blockDim.x;
            for (int source_in_tile = 0; source_in_tile < tile_count;
                 ++source_in_tile) {
                const int j = tile_begin + source_in_tile;
                if (j == i) continue;
                const Position source = tile[source_in_tile];
                const float dx = source.x - target.x;
                const float dy = source.y - target.y;
                const float dz = source.z - target.z;
                const float distance_squared =
                    dx * dx + dy * dy + dz * dz + softening_squared;
                const float inverse_distance = rsqrtf(distance_squared);
                const float scale = gravitational_constant * source.mass *
                                    inverse_distance * inverse_distance *
                                    inverse_distance;
                ax += dx * scale;
                ay += dy * scale;
                az += dz * scale;
            }
        }
        __syncthreads();
    }

    if (valid_target) accelerations[i] = {ax, ay, az};
}

__global__ void euler_update_kernel(Position* positions,
                                    Velocity* velocities,
                                    const Velocity* accelerations,
                                    int particle_count,
                                    float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) return;
    Position position = positions[i];
    Velocity velocity = velocities[i];
    const Velocity acceleration = accelerations[i];
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;
    position.z += velocity.z * dt;
    velocity.x += acceleration.x * dt;
    velocity.y += acceleration.y * dt;
    velocity.z += acceleration.z * dt;
    positions[i] = position;
    velocities[i] = velocity;
}

__global__ void kick_drift_kernel(Position* positions,
                                  Velocity* velocities,
                                  const Velocity* accelerations,
                                  int particle_count,
                                  float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) return;
    Position position = positions[i];
    Velocity velocity = velocities[i];
    const Velocity acceleration = accelerations[i];
    const float half_dt = 0.5f * dt;
    velocity.x += acceleration.x * half_dt;
    velocity.y += acceleration.y * half_dt;
    velocity.z += acceleration.z * half_dt;
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;
    position.z += velocity.z * dt;
    positions[i] = position;
    velocities[i] = velocity;
}

__global__ void final_kick_kernel(Velocity* velocities,
                                  const Velocity* accelerations,
                                  int particle_count,
                                  float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) return;
    Velocity velocity = velocities[i];
    const Velocity acceleration = accelerations[i];
    const float half_dt = 0.5f * dt;
    velocity.x += acceleration.x * half_dt;
    velocity.y += acceleration.y * half_dt;
    velocity.z += acceleration.z * half_dt;
    velocities[i] = velocity;
}

float launch_force(const Position* positions,
                   Velocity* accelerations,
                   int particle_count,
                   float gravitational_constant,
                   float softening_squared,
                   int grid_size,
                   int block_size,
                   const EventPair& events,
                   ForceKernel kernel) {
    NBODY_MACA_CHECK(mcEventRecord(events.start()));
    if (kernel == ForceKernel::Naive) {
        force_naive_kernel<<<grid_size, block_size>>>(
            positions, accelerations, particle_count, gravitational_constant,
            softening_squared);
    } else {
        const std::size_t shared_bytes =
            static_cast<std::size_t>(block_size) * sizeof(Position);
        force_tiled_kernel<<<grid_size, block_size, shared_bytes>>>(
            positions, accelerations, particle_count, gravitational_constant,
            softening_squared);
    }
    NBODY_MACA_CHECK(mcGetLastError());
    NBODY_MACA_CHECK(mcEventRecord(events.stop()));
    NBODY_MACA_CHECK(mcEventSynchronize(events.stop()));
    float elapsed_ms = 0.0f;
    NBODY_MACA_CHECK(mcEventElapsedTime(&elapsed_ms, events.start(), events.stop()));
    return elapsed_ms;
}

void record_snapshot(const std::vector<Position>& positions,
                     std::size_t record_index,
                     std::size_t record_count,
                     std::vector<float>& trajectory) {
    for (std::size_t particle = 0; particle < positions.size(); ++particle) {
        const Position position = positions[particle];
        if (!std::isfinite(position.x) || !std::isfinite(position.y) ||
            !std::isfinite(position.z)) {
            throw std::runtime_error("MACA 轨迹包含非有限坐标");
        }
        const std::size_t offset = (particle * record_count + record_index) * 3;
        trajectory[offset] = position.x;
        trajectory[offset + 1] = position.y;
        trajectory[offset + 2] = position.z;
    }
}

CpuRunResult simulate_maca(const std::vector<Particle>& initial_particles,
                           const Config& config,
                           int block_size,
                           ForceKernel kernel,
                           const RunOptions& options) {
    if (initial_particles.empty()) throw std::invalid_argument("至少需要一个粒子");
    if (initial_particles.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("MACA 粒子数量超出 int 范围");
    }
    if (block_size < 1 || block_size > 1024) {
        throw std::invalid_argument("MACA block size 必须在 1 到 1024 之间");
    }

    const std::size_t particle_count = initial_particles.size();
    const int n = static_cast<int>(particle_count);
    const int grid_size = (n + block_size - 1) / block_size;
    std::vector<Position> host_positions(particle_count);
    std::vector<Velocity> host_velocities(particle_count);

    CpuRunResult result;
    result.record_enabled = options.record_enabled;
    result.diagnostics_enabled = options.diagnostics_enabled;
    result.particles.resize(particle_count);
    for (std::size_t i = 0; i < particle_count; ++i) {
        const Particle& source = initial_particles[i];
        const Position position = {
            static_cast<float>(source.position.x),
            static_cast<float>(source.position.y),
            static_cast<float>(source.position.z),
            static_cast<float>(source.mass),
        };
        const Velocity velocity = {
            static_cast<float>(source.velocity.x),
            static_cast<float>(source.velocity.y),
            static_cast<float>(source.velocity.z),
        };
        if (!std::isfinite(position.x) || !std::isfinite(position.y) ||
            !std::isfinite(position.z) || !std::isfinite(position.mass) ||
            !std::isfinite(velocity.x) || !std::isfinite(velocity.y) ||
            !std::isfinite(velocity.z) || !(position.mass > 0.0f)) {
            throw std::runtime_error("初态无法用 MACA FP32 表示");
        }
        host_positions[i] = position;
        host_velocities[i] = velocity;
        result.particles[i] = {
            {position.x, position.y, position.z},
            {velocity.x, velocity.y, velocity.z}, position.mass,
        };
    }

    if (options.record_enabled) {
        result.recorded_steps =
            make_recorded_steps(config.num_steps, config.record_interval);
    }
    const std::size_t record_count = result.recorded_steps.size();
    if (options.record_enabled &&
        record_count > std::numeric_limits<std::size_t>::max() /
                           particle_count / 3) {
        throw std::overflow_error("MACA 轨迹大小溢出");
    }
    if (options.record_enabled) {
        result.trajectory.resize(particle_count * record_count * 3);
        record_snapshot(host_positions, 0, record_count, result.trajectory);
    }
    if (options.diagnostics_enabled) {
        result.initial_diagnostics = compute_diagnostics(
            result.particles, config.gravitational_constant, config.softening);
    }

    const float gravitational_constant =
        static_cast<float>(config.gravitational_constant);
    const float softening = static_cast<float>(config.softening);
    const float softening_squared = softening * softening;
    const float dt = static_cast<float>(config.dt);
    if (!std::isfinite(gravitational_constant) ||
        !std::isfinite(softening_squared) || !std::isfinite(dt) ||
        !(gravitational_constant > 0.0f) || !(softening > 0.0f) ||
        !(softening_squared > 0.0f) || !(dt > 0.0f)) {
        throw std::runtime_error("配置参数无法用 MACA FP32 表示");
    }

    DeviceBuffer<Position> device_positions(particle_count);
    DeviceBuffer<Velocity> device_velocities(particle_count);
    DeviceBuffer<Velocity> device_accelerations(particle_count);
    NBODY_MACA_CHECK(mcMemcpy(device_positions.get(), host_positions.data(),
                              particle_count * sizeof(Position),
                              mcMemcpyHostToDevice));
    NBODY_MACA_CHECK(mcMemcpy(device_velocities.get(), host_velocities.data(),
                              particle_count * sizeof(Velocity),
                              mcMemcpyHostToDevice));

    EventPair force_events;
    const auto simulation_begin = Clock::now();
    result.force_ms += launch_force(
        device_positions.get(), device_accelerations.get(), n,
        gravitational_constant, softening_squared, grid_size, block_size,
        force_events, kernel);
    result.force_evaluations = 1;
    std::size_t next_record = 1;

    for (int step = 1; step <= config.num_steps; ++step) {
        if (config.integrator == Integrator::Euler) {
            euler_update_kernel<<<grid_size, block_size>>>(
                device_positions.get(), device_velocities.get(),
                device_accelerations.get(), n, dt);
            NBODY_MACA_CHECK(mcGetLastError());
            if (step < config.num_steps) {
                result.force_ms += launch_force(
                    device_positions.get(), device_accelerations.get(), n,
                    gravitational_constant, softening_squared, grid_size,
                    block_size, force_events, kernel);
                ++result.force_evaluations;
            }
        } else {
            kick_drift_kernel<<<grid_size, block_size>>>(
                device_positions.get(), device_velocities.get(),
                device_accelerations.get(), n, dt);
            NBODY_MACA_CHECK(mcGetLastError());
            result.force_ms += launch_force(
                device_positions.get(), device_accelerations.get(), n,
                gravitational_constant, softening_squared, grid_size,
                block_size, force_events, kernel);
            ++result.force_evaluations;
            final_kick_kernel<<<grid_size, block_size>>>(
                device_velocities.get(), device_accelerations.get(), n, dt);
            NBODY_MACA_CHECK(mcGetLastError());
        }

        if (options.record_enabled && next_record < record_count &&
            result.recorded_steps[next_record] == step) {
            NBODY_MACA_CHECK(mcMemcpy(
                host_positions.data(), device_positions.get(),
                particle_count * sizeof(Position), mcMemcpyDeviceToHost));
            record_snapshot(host_positions, next_record, record_count,
                            result.trajectory);
            ++next_record;
        }
    }

    NBODY_MACA_CHECK(mcDeviceSynchronize());
    result.simulation_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - simulation_begin)
            .count();
    NBODY_MACA_CHECK(mcMemcpy(host_positions.data(), device_positions.get(),
                              particle_count * sizeof(Position),
                              mcMemcpyDeviceToHost));
    NBODY_MACA_CHECK(mcMemcpy(host_velocities.data(), device_velocities.get(),
                              particle_count * sizeof(Velocity),
                              mcMemcpyDeviceToHost));

    for (std::size_t i = 0; i < particle_count; ++i) {
        const Position position = host_positions[i];
        const Velocity velocity = host_velocities[i];
        if (!std::isfinite(position.x) || !std::isfinite(position.y) ||
            !std::isfinite(position.z) || !std::isfinite(velocity.x) ||
            !std::isfinite(velocity.y) || !std::isfinite(velocity.z)) {
            throw std::runtime_error("MACA 最终状态包含非有限数值");
        }
        result.particles[i] = {
            {position.x, position.y, position.z},
            {velocity.x, velocity.y, velocity.z}, position.mass,
        };
    }
    if (options.diagnostics_enabled) {
        result.final_diagnostics = compute_diagnostics(
            result.particles, config.gravitational_constant, config.softening);
    }
    return result;
}

}  // namespace

CpuRunResult simulate_maca_naive(const std::vector<Particle>& initial_particles,
                                 const Config& config,
                                 int block_size,
                                 const RunOptions& options) {
    return simulate_maca(initial_particles, config, block_size,
                         ForceKernel::Naive, options);
}

CpuRunResult simulate_maca_tiled(const std::vector<Particle>& initial_particles,
                                 const Config& config,
                                 int block_size,
                                 const RunOptions& options) {
    return simulate_maca(initial_particles, config, block_size,
                         ForceKernel::Tiled, options);
}

}  // namespace nbody

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t error__ = (call);                                          \
        if (error__ != cudaSuccess) {                                          \
            std::ostringstream message__;                                      \
            message__ << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << ": " << cudaGetErrorString(error__);                 \
            throw std::runtime_error(message__.str());                         \
        }                                                                       \
    } while (false)

constexpr int TILE_SIZE = 256;

struct Params {
    float dt = 1.0e-3f;
    int steps = 1000;
    int record_interval = 100;
    float G = 1.0f;
    float softening = 1.0e-4f;
    std::string integrator = "leapfrog";
};

struct Body {
    float x, y, z, vx, vy, vz, mass;
};

struct Diagnostics {
    double energy = 0.0;
    double px = 0.0, py = 0.0, pz = 0.0;
    double momentum_scale = 0.0;
    bool sampled_energy = false;
    std::uint64_t energy_pairs = 0;
};

static Diagnostics compute_diagnostics(const std::vector<Body>& bodies, float gravitational_constant,
                                       float softening) {
    Diagnostics result;
    for (const Body& body : bodies) {
        const double speed_squared = static_cast<double>(body.vx) * body.vx +
                                     static_cast<double>(body.vy) * body.vy +
                                     static_cast<double>(body.vz) * body.vz;
        result.energy += 0.5 * body.mass * speed_squared;
        result.px += static_cast<double>(body.mass) * body.vx;
        result.py += static_cast<double>(body.mass) * body.vy;
        result.pz += static_cast<double>(body.mass) * body.vz;
        result.momentum_scale += std::abs(static_cast<double>(body.mass)) * std::sqrt(speed_squared);
    }

    const std::uint64_t n = bodies.size();
    const std::uint64_t total_pairs = n * (n - 1) / 2;
    constexpr std::uint64_t MAX_ENERGY_PAIRS = 8'000'000;
    double potential_sum = 0.0;
    if (total_pairs <= MAX_ENERGY_PAIRS) {
        result.energy_pairs = total_pairs;
        for (std::uint64_t i = 0; i < n; ++i) {
            for (std::uint64_t j = i + 1; j < n; ++j) {
                const double dx = static_cast<double>(bodies[j].x) - bodies[i].x;
                const double dy = static_cast<double>(bodies[j].y) - bodies[i].y;
                const double dz = static_cast<double>(bodies[j].z) - bodies[i].z;
                const double distance = std::sqrt(dx * dx + dy * dy + dz * dz +
                                                  static_cast<double>(softening) * softening);
                potential_sum -= gravitational_constant * bodies[i].mass * bodies[j].mass / distance;
            }
        }
    } else {
        result.sampled_energy = true;
        result.energy_pairs = MAX_ENERGY_PAIRS;
        std::uint64_t state = 0x9e3779b97f4a7c15ULL;
        for (std::uint64_t sample = 0; sample < MAX_ENERGY_PAIRS; ++sample) {
            state ^= state >> 12; state ^= state << 25; state ^= state >> 27;
            const std::uint64_t i = (state * 0x2545F4914F6CDD1DULL) % n;
            state ^= state >> 12; state ^= state << 25; state ^= state >> 27;
            std::uint64_t j = (state * 0x2545F4914F6CDD1DULL) % (n - 1);
            if (j >= i) ++j;
            const double dx = static_cast<double>(bodies[j].x) - bodies[i].x;
            const double dy = static_cast<double>(bodies[j].y) - bodies[i].y;
            const double dz = static_cast<double>(bodies[j].z) - bodies[i].z;
            const double distance = std::sqrt(dx * dx + dy * dy + dz * dz +
                                              static_cast<double>(softening) * softening);
            potential_sum -= gravitational_constant * bodies[i].mass * bodies[j].mass / distance;
        }
        potential_sum *= static_cast<double>(total_pairs) / MAX_ENERGY_PAIRS;
    }
    result.energy += potential_sum;
    return result;
}

static double benchmark_cpu_force(const std::vector<Body>& input, const Params& params,
                                  int& benchmark_particles, int& benchmark_steps) {
    benchmark_particles = std::min<int>(static_cast<int>(input.size()), 1024);
    const std::uint64_t interactions_per_step = static_cast<std::uint64_t>(benchmark_particles) * benchmark_particles;
    benchmark_steps = std::max(1, std::min<int>(100, static_cast<int>(20'000'000 / std::max<std::uint64_t>(1, interactions_per_step))));
    std::vector<float4> acceleration(benchmark_particles);
    volatile float checksum = 0.0f;
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < benchmark_steps; ++step) {
        for (int i = 0; i < benchmark_particles; ++i) {
            float ax = 0, ay = 0, az = 0;
            for (int j = 0; j < benchmark_particles; ++j) {
                const float dx = input[j].x - input[i].x;
                const float dy = input[j].y - input[i].y;
                const float dz = input[j].z - input[i].z;
                const float r2 = dx * dx + dy * dy + dz * dz + params.softening * params.softening;
                const float inv_r = 1.0f / std::sqrt(r2);
                const float scale = params.G * input[j].mass * inv_r * inv_r * inv_r;
                ax += dx * scale; ay += dy * scale; az += dz * scale;
            }
            acceleration[i] = make_float4(ax, ay, az, 0);
        }
        checksum += acceleration[step % benchmark_particles].x;
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (!std::isfinite(checksum)) throw std::runtime_error("CPU benchmark produced non-finite values");
    return interactions_per_step * static_cast<double>(benchmark_steps) / seconds;
}

__global__ void acceleration_kernel(const float4* positions, float4* acceleration,
                                    int n, float gravitational_constant,
                                    float softening_squared) {
    __shared__ float4 tile[TILE_SIZE];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < n;
    const float4 pi = active ? positions[i] : make_float4(0, 0, 0, 0);
    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    for (int base = 0; base < n; base += TILE_SIZE) {
        const int j = base + threadIdx.x;
        tile[threadIdx.x] = (j < n) ? positions[j] : make_float4(0, 0, 0, 0);
        __syncthreads();
        const int count = min(TILE_SIZE, n - base);
        if (active) {
            for (int k = 0; k < count; ++k) {
                const float4 pj = tile[k];
                const float dx = pj.x - pi.x;
                const float dy = pj.y - pi.y;
                const float dz = pj.z - pi.z;
                const float distance_squared = dx * dx + dy * dy + dz * dz + softening_squared;
                const float inv_distance = rsqrtf(distance_squared);
                const float scale = gravitational_constant * pj.w * inv_distance * inv_distance * inv_distance;
                ax += dx * scale;
                ay += dy * scale;
                az += dz * scale;
            }
        }
        __syncthreads();
    }
    if (active) acceleration[i] = make_float4(ax, ay, az, 0.0f);
}

__global__ void euler_kernel(float4* positions, double4* precise_positions, double4* velocities,
                             const float4* acceleration, int n, float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double4 p = precise_positions[i], v = velocities[i];
    const float4 a = acceleration[i];
    v.x += static_cast<double>(a.x) * dt;
    v.y += static_cast<double>(a.y) * dt;
    v.z += static_cast<double>(a.z) * dt;
    p.x += v.x * dt; p.y += v.y * dt; p.z += v.z * dt;
    velocities[i] = v; precise_positions[i] = p;
    positions[i] = make_float4(static_cast<float>(p.x), static_cast<float>(p.y),
                              static_cast<float>(p.z), positions[i].w);
}

__global__ void kick_kernel(double4* velocities, const float4* acceleration, int n, float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    velocities[i].x += static_cast<double>(acceleration[i].x) * dt;
    velocities[i].y += static_cast<double>(acceleration[i].y) * dt;
    velocities[i].z += static_cast<double>(acceleration[i].z) * dt;
}

__global__ void drift_kernel(float4* positions, double4* precise_positions,
                             const double4* velocities, int n, float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Preserve sub-ULP drifts at large coordinates instead of rounding every step.
    double4 p = precise_positions[i];
    p.x += velocities[i].x * dt;
    p.y += velocities[i].y * dt;
    p.z += velocities[i].z * dt;
    precise_positions[i] = p;
    positions[i] = make_float4(static_cast<float>(p.x), static_cast<float>(p.y),
                              static_cast<float>(p.z), positions[i].w);
}

static void read_particles(const std::string& path, std::vector<Body>& bodies) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open particle file: " + path);
    Body body{};
    while (input >> body.x >> body.y >> body.z >> body.vx >> body.vy >> body.vz >> body.mass)
        bodies.push_back(body);
    if (bodies.empty() || !input.eof()) throw std::runtime_error("invalid particle file: " + path);
}

static void read_params(const std::string& path, Params& p) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open parameter file: " + path);
    std::string line;
    while (std::getline(input, line)) {
        const auto comment = line.find('#');
        if (comment != std::string::npos) line.resize(comment);
        std::istringstream row(line);
        std::string key, equals, value;
        if (!(row >> key)) continue;
        if (key == "dt") row >> equals >> p.dt;
        else if (key == "num_steps") row >> equals >> p.steps;
        else if (key == "record_interval") row >> equals >> p.record_interval;
        else if (key == "G") row >> equals >> p.G;
        else if (key == "softening") row >> equals >> p.softening;
        else if (key == "integrator") row >> equals >> p.integrator;
    }
    if (p.dt <= 0 || p.steps < 0 || p.record_interval <= 0 || p.softening < 0)
        throw std::runtime_error("invalid simulation parameters");
    if (p.integrator != "euler" && p.integrator != "leapfrog")
        throw std::runtime_error("integrator must be euler or leapfrog");
}

static void write_trajectory(const std::string& path, int n, int records,
                             const std::vector<float>& trajectory) {
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot create output file: " + path);
    const std::int32_t particle_count = n, record_count = records;
    output.write(reinterpret_cast<const char*>(&particle_count), sizeof(particle_count));
    output.write(reinterpret_cast<const char*>(&record_count), sizeof(record_count));
    output.write(reinterpret_cast<const char*>(trajectory.data()),
                 static_cast<std::streamsize>(trajectory.size() * sizeof(float)));
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 5) {
        std::cerr << "Usage: nbody <particles.txt> <params.txt> [trajectory.bin] [performance.log]\n";
        return 2;
    }
    try {
        std::vector<Body> bodies;
        Params params;
        read_particles(argv[1], bodies);
        read_params(argv[2], params);
        const std::string output_path = argc >= 4 ? argv[3] : "trajectory.bin";
        const std::string log_path = argc >= 5 ? argv[4] : "performance.log";
        const int n = static_cast<int>(bodies.size());
        const int records = params.steps / params.record_interval + 1;
        const Diagnostics initial_diagnostics = compute_diagnostics(bodies, params.G, params.softening);
        std::vector<float> trajectory(static_cast<size_t>(records) * n * 3);
        std::vector<float4> host_positions(n);
        std::vector<double4> host_precise_positions(n), host_velocities(n);
        for (int i = 0; i < n; ++i) {
            host_positions[i] = make_float4(bodies[i].x, bodies[i].y, bodies[i].z, bodies[i].mass);
            host_precise_positions[i] = make_double4(bodies[i].x, bodies[i].y, bodies[i].z, bodies[i].mass);
            host_velocities[i] = make_double4(bodies[i].vx, bodies[i].vy, bodies[i].vz, 0);
        }
        float4 *positions = nullptr, *acceleration = nullptr;
        double4 *precise_positions = nullptr, *velocities = nullptr;
        CUDA_CHECK(cudaMalloc(&positions, n * sizeof(float4)));
        CUDA_CHECK(cudaMalloc(&velocities, n * sizeof(double4)));
        CUDA_CHECK(cudaMalloc(&precise_positions, n * sizeof(double4)));
        CUDA_CHECK(cudaMalloc(&acceleration, n * sizeof(float4)));
        CUDA_CHECK(cudaMemcpy(positions, host_positions.data(), n * sizeof(float4), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(precise_positions, host_precise_positions.data(), n * sizeof(double4), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(velocities, host_velocities.data(), n * sizeof(double4), cudaMemcpyHostToDevice));
        const int blocks = (n + TILE_SIZE - 1) / TILE_SIZE;
        double trajectory_copy_seconds = 0.0;
        auto record = [&](int index) {
            const auto copy_start = std::chrono::steady_clock::now();
            CUDA_CHECK(cudaMemcpy(host_positions.data(), positions, n * sizeof(float4), cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; ++i) {
                trajectory[(static_cast<size_t>(i) * records + index) * 3 + 0] = host_positions[i].x;
                trajectory[(static_cast<size_t>(i) * records + index) * 3 + 1] = host_positions[i].y;
                trajectory[(static_cast<size_t>(i) * records + index) * 3 + 2] = host_positions[i].z;
            }
            trajectory_copy_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - copy_start).count();
        };
        record(0);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t gpu_start, gpu_stop;
        CUDA_CHECK(cudaEventCreate(&gpu_start));
        CUDA_CHECK(cudaEventCreate(&gpu_stop));
        CUDA_CHECK(cudaEventRecord(gpu_start));
        const auto start = std::chrono::steady_clock::now();
        int record_index = 1;
        if (params.integrator == "leapfrog") {
            acceleration_kernel<<<blocks, TILE_SIZE>>>(positions, acceleration, n, params.G, params.softening * params.softening);
            CUDA_CHECK(cudaGetLastError());
            kick_kernel<<<blocks, TILE_SIZE>>>(velocities, acceleration, n, 0.5f * params.dt);
            CUDA_CHECK(cudaGetLastError());
        }
        for (int step = 1; step <= params.steps; ++step) {
            if (params.integrator == "euler") {
                acceleration_kernel<<<blocks, TILE_SIZE>>>(positions, acceleration, n, params.G, params.softening * params.softening);
                euler_kernel<<<blocks, TILE_SIZE>>>(positions, precise_positions, velocities, acceleration, n, params.dt);
            } else {
                drift_kernel<<<blocks, TILE_SIZE>>>(positions, precise_positions, velocities, n, params.dt);
                acceleration_kernel<<<blocks, TILE_SIZE>>>(positions, acceleration, n, params.G, params.softening * params.softening);
                const float kick_dt = (step == params.steps) ? 0.5f * params.dt : params.dt;
                kick_kernel<<<blocks, TILE_SIZE>>>(velocities, acceleration, n, kick_dt);
            }
            CUDA_CHECK(cudaGetLastError());
            if (step % params.record_interval == 0) record(record_index++);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
         CUDA_CHECK(cudaEventRecord(gpu_stop));
         CUDA_CHECK(cudaEventSynchronize(gpu_stop));
         float gpu_milliseconds = 0.0f;
         CUDA_CHECK(cudaEventElapsedTime(&gpu_milliseconds, gpu_start, gpu_stop));
        CUDA_CHECK(cudaMemcpy(host_positions.data(), positions, n * sizeof(float4), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_velocities.data(), velocities, n * sizeof(double4), cudaMemcpyDeviceToHost));
         std::vector<Body> final_bodies(n);
         for (int i = 0; i < n; ++i) {
             final_bodies[i] = {host_positions[i].x, host_positions[i].y, host_positions[i].z,
                       static_cast<float>(host_velocities[i].x), static_cast<float>(host_velocities[i].y), static_cast<float>(host_velocities[i].z),
                       host_positions[i].w};
         }
         const Diagnostics final_diagnostics = compute_diagnostics(final_bodies, params.G, params.softening);
        const auto io_start = std::chrono::steady_clock::now();
         write_trajectory(output_path, n, records, trajectory);
         const double io_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - io_start).count();
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const double interactions = static_cast<double>(n) * n * params.steps;
         const double simulation_seconds = gpu_milliseconds / 1000.0;
         const double gpu_rate = interactions / simulation_seconds;
         int cpu_particles = 0, cpu_steps = 0;
         const double cpu_rate = benchmark_cpu_force(bodies, params, cpu_particles, cpu_steps);
         const double initial_momentum = std::sqrt(initial_diagnostics.px * initial_diagnostics.px +
                                 initial_diagnostics.py * initial_diagnostics.py +
                                 initial_diagnostics.pz * initial_diagnostics.pz);
         const double delta_px = final_diagnostics.px - initial_diagnostics.px;
         const double delta_py = final_diagnostics.py - initial_diagnostics.py;
         const double delta_pz = final_diagnostics.pz - initial_diagnostics.pz;
         const double momentum_error = std::sqrt(delta_px * delta_px + delta_py * delta_py + delta_pz * delta_pz) /
                           std::max(initial_diagnostics.momentum_scale, 1.0e-30);
         const double energy_error = std::abs(final_diagnostics.energy - initial_diagnostics.energy) /
                         std::max(std::abs(initial_diagnostics.energy), 1.0e-30);
         size_t free_memory = 0, total_memory = 0;
         CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
         cudaDeviceProp device{};
         CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
         std::ostringstream report;
         report << std::fixed << std::setprecision(6)
             << "device=" << device.name << "\n"
             << "particles=" << n << "\nsteps=" << params.steps << "\nrecords=" << records << "\n"
             << "integrator=" << params.integrator << "\ndt=" << params.dt << "\n"
             << "gpu_simulation_sec=" << simulation_seconds << "\n"
             << "gpu_kernel_compute_sec=" << simulation_seconds << "\n"
             << "host_trajectory_copy_and_record_sec=" << trajectory_copy_seconds << "\n"
             << "wall_total_sec=" << seconds << "\ntrajectory_io_sec=" << io_seconds << "\n"
             << "average_step_ms=" << (gpu_milliseconds / std::max(params.steps, 1)) << "\n"
             << "particle_steps_per_sec=" << (simulation_seconds > 0 ? n * params.steps / simulation_seconds : 0) << "\n"
             << "interactions_per_sec=" << gpu_rate << "\n"
             << "cpu_benchmark_particles=" << cpu_particles << "\ncpu_benchmark_steps=" << cpu_steps << "\n"
             << "cpu_interactions_per_sec=" << cpu_rate << "\n"
             << "estimated_gpu_vs_cpu_speedup=" << (cpu_rate > 0 ? gpu_rate / cpu_rate : 0) << "\n"
             << "integration_accumulator=float64\nforce_precision=float32\ntrajectory_precision=float32\n"
             << "device_array_bytes=" << static_cast<std::uint64_t>(n) * (2 * sizeof(float4) + 2 * sizeof(double4)) << "\n"
             << "gpu_memory_used_bytes=" << (total_memory - free_memory) << "\n"
             << "initial_momentum_norm=" << initial_momentum << "\n"
             << "relative_momentum_error=" << momentum_error << "\n"
             << "initial_energy=" << initial_diagnostics.energy << "\nfinal_energy=" << final_diagnostics.energy << "\n"
             << "relative_energy_error=" << energy_error << "\n"
             << "energy_diagnostic=" << (initial_diagnostics.sampled_energy ? "sampled" : "exact") << "\n"
             << "energy_pairs_evaluated=" << initial_diagnostics.energy_pairs << "\n"
             << "trajectory_layout=particle-major [P,R,xyz]\n";
         std::ofstream log(log_path);
         if (!log) throw std::runtime_error("cannot create performance log: " + log_path);
         log << report.str();
         std::cout << report.str() << "trajectory_file=" << output_path << "\nperformance_log=" << log_path << "\n";
         CUDA_CHECK(cudaEventDestroy(gpu_start));
         CUDA_CHECK(cudaEventDestroy(gpu_stop));
        cudaFree(positions); cudaFree(precise_positions); cudaFree(velocities); cudaFree(acceleration);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

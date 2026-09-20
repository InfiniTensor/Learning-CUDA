#include "nbody.cuh"

#include <cuda_runtime.h>

#ifdef NBODY_ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#define NBODY_NVTX_PUSH(name) nvtxRangePushA(name)
#define NBODY_NVTX_POP() nvtxRangePop()
#else
#define NBODY_NVTX_PUSH(name) ((void)0)
#define NBODY_NVTX_POP() ((void)0)
#endif

#include <algorithm>
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(call) cuda_check((call), #call, __FILE__, __LINE__)

namespace {

void cuda_check(cudaError_t error, const char* expression, const char* file, int line) {
    if (error == cudaSuccess) return;
    std::ostringstream message;
    message << "CUDA 调用失败: " << expression << "\n"
            << "位置: " << file << ':' << line << "\n"
            << "原因: " << cudaGetErrorString(error);
    throw std::runtime_error(message.str());
}

struct DeviceArrays {
    float* x = nullptr;
    float* y = nullptr;
    float* z = nullptr;
    float* vx = nullptr;
    float* vy = nullptr;
    float* vz = nullptr;
    float* mass = nullptr;
    float* ax = nullptr;
    float* ay = nullptr;
    float* az = nullptr;
    float* partial_ax = nullptr;
    float* partial_ay = nullptr;
    float* partial_az = nullptr;

    void allocate(int n) {
        const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
        CUDA_CHECK(cudaMalloc(&x, bytes));
        CUDA_CHECK(cudaMalloc(&y, bytes));
        CUDA_CHECK(cudaMalloc(&z, bytes));
        CUDA_CHECK(cudaMalloc(&vx, bytes));
        CUDA_CHECK(cudaMalloc(&vy, bytes));
        CUDA_CHECK(cudaMalloc(&vz, bytes));
        CUDA_CHECK(cudaMalloc(&mass, bytes));
        CUDA_CHECK(cudaMalloc(&ax, bytes));
        CUDA_CHECK(cudaMalloc(&ay, bytes));
        CUDA_CHECK(cudaMalloc(&az, bytes));
    }

    void allocate_partials(int n, int chunks) {
        const std::size_t bytes = static_cast<std::size_t>(n) * chunks * sizeof(float);
        CUDA_CHECK(cudaMalloc(&partial_ax, bytes));
        CUDA_CHECK(cudaMalloc(&partial_ay, bytes));
        CUDA_CHECK(cudaMalloc(&partial_az, bytes));
    }

    ~DeviceArrays() {
        // cudaFree(nullptr) 是安全的。析构路径只负责回收，不在这里抛异常。
        cudaFree(x); cudaFree(y); cudaFree(z);
        cudaFree(vx); cudaFree(vy); cudaFree(vz);
        cudaFree(mass);
        cudaFree(ax); cudaFree(ay); cudaFree(az);
        cudaFree(partial_ax); cudaFree(partial_ay); cudaFree(partial_az);
    }
};

struct DevicePackedArrays {
    float4* position_mass = nullptr;
    float4* velocity = nullptr;
    float4* acceleration = nullptr;

    void allocate(int n) {
        const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float4);
        CUDA_CHECK(cudaMalloc(&position_mass, bytes));
        CUDA_CHECK(cudaMalloc(&velocity, bytes));
        CUDA_CHECK(cudaMalloc(&acceleration, bytes));
    }

    ~DevicePackedArrays() {
        cudaFree(position_mass);
        cudaFree(velocity);
        cudaFree(acceleration);
    }
};

class SimulationStream {
public:
    SimulationStream() { CUDA_CHECK(cudaStreamCreate(&stream_)); }
    ~SimulationStream() { cudaStreamDestroy(stream_); }
    operator cudaStream_t() const { return stream_; }
private:
    cudaStream_t stream_{};
};

struct GraphStep {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    ~GraphStep() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
    }
};

class EventTimer {
public:
    EventTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~EventTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start(cudaStream_t stream = nullptr) { CUDA_CHECK(cudaEventRecord(start_, stream)); }
    float stop(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

__global__ void compute_acceleration_baseline(const float* x,
                                              const float* y,
                                              const float* z,
                                              const float* mass,
                                              float* ax,
                                              float* ay,
                                              float* az,
                                              int n,
                                              float G,
                                              float softening2) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // 这里一个线程负责一个粒子。最直接的做法就是把其他粒子全扫一遍。
    // 复杂度确实是 O(N^2)，但这个版本很适合先拿来检查公式和结果。
    const float xi = x[i];
    const float yi = y[i];
    const float zi = z[i];
    float axi = 0.0f;
    float ayi = 0.0f;
    float azi = 0.0f;
    for (int j = 0; j < n; ++j) {
        if (i == j) continue;
        const float dx = x[j] - xi;
        const float dy = y[j] - yi;
        const float dz = z[j] - zi;

        // softening 防止两个粒子靠得太近时 1/r^3 直接数值爆炸。
        const float dist2 = dx * dx + dy * dy + dz * dz + softening2;
        const float inv_dist = rsqrtf(dist2);
        const float inv_dist3 = inv_dist * inv_dist * inv_dist;
        const float factor = G * mass[j] * inv_dist3;
        axi += factor * dx;
        ayi += factor * dy;
        azi += factor * dz;
    }
    ax[i] = axi;
    ay[i] = ayi;
    az[i] = azi;
}

__global__ void compute_acceleration_partial(const float* x,
                                             const float* y,
                                             const float* z,
                                             const float* mass,
                                             float* partial_ax,
                                             float* partial_ay,
                                             float* partial_az,
                                             int n,
                                             int source_chunks,
                                             float G,
                                             float softening2) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int chunk = static_cast<int>(blockIdx.y);
    const int chunk_size = (n + source_chunks - 1) / source_chunks;
    const int source_begin = chunk * chunk_size;
    const int source_end = min(n, source_begin + chunk_size);
    const float xi = x[i], yi = y[i], zi = z[i];
    float axi = 0.0f, ayi = 0.0f, azi = 0.0f;

    // N=4096 时普通 target grid 只有 64 个 block，3080 上每个 SM 基本只分到一块。
    // 这个实验把 source 范围分给 grid.y 的多个 block，先各算一份 partial。
    // 代价也很直接：多了 partial 显存写入和后面的 reduction kernel。
    for (int j = source_begin; j < source_end; ++j) {
        if (j == i) continue;
        const float dx = x[j] - xi;
        const float dy = y[j] - yi;
        const float dz = z[j] - zi;
        const float dist2 = dx * dx + dy * dy + dz * dz + softening2;
        const float inv_dist = rsqrtf(dist2);
        const float scale = G * mass[j] * inv_dist * inv_dist * inv_dist;
        axi += scale * dx; ayi += scale * dy; azi += scale * dz;
    }
    const std::size_t offset = static_cast<std::size_t>(chunk) * n + i;
    partial_ax[offset] = axi;
    partial_ay[offset] = ayi;
    partial_az[offset] = azi;
}

__global__ void reduce_acceleration_partials(const float* partial_ax,
                                             const float* partial_ay,
                                             const float* partial_az,
                                             float* ax,
                                             float* ay,
                                             float* az,
                                             int n,
                                             int source_chunks) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float axi = 0.0f, ayi = 0.0f, azi = 0.0f;
    for (int chunk = 0; chunk < source_chunks; ++chunk) {
        const std::size_t offset = static_cast<std::size_t>(chunk) * n + i;
        axi += partial_ax[offset];
        ayi += partial_ay[offset];
        azi += partial_az[offset];
    }
    ax[i] = axi; ay[i] = ayi; az[i] = azi;
}

__global__ void compute_acceleration_tiled(const float* x,
                                           const float* y,
                                           const float* z,
                                           const float* mass,
                                           float* ax,
                                           float* ay,
                                           float* az,
                                           int n,
                                           float G,
                                           float softening2) {
    extern __shared__ float shared[];
    float* shared_x = shared;
    float* shared_y = shared + blockDim.x;
    float* shared_z = shared + 2 * blockDim.x;
    float* shared_mass = shared + 3 * blockDim.x;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < n;
    const float xi = active ? x[i] : 0.0f;
    const float yi = active ? y[i] : 0.0f;
    const float zi = active ? z[i] : 0.0f;
    float axi = 0.0f;
    float ayi = 0.0f;
    float azi = 0.0f;

    // 每个 block 一次搬一小批粒子到 shared memory。
    // 同一个 block 里的线程会重复使用这批数据，global memory 的读取量就少了很多。
    for (int tile_start = 0; tile_start < n; tile_start += blockDim.x) {
        const int source = tile_start + threadIdx.x;
        if (source < n) {
            shared_x[threadIdx.x] = x[source];
            shared_y[threadIdx.x] = y[source];
            shared_z[threadIdx.x] = z[source];
            shared_mass[threadIdx.x] = mass[source];
        }

        // 这里不能漏掉同步。否则有些线程还没搬完，别的线程已经开始读了。
        __syncthreads();
        const int tile_count = min(static_cast<int>(blockDim.x), n - tile_start);
        if (active) {
            for (int j = 0; j < tile_count; ++j) {
                if (tile_start + j == i) continue;
                const float dx = shared_x[j] - xi;
                const float dy = shared_y[j] - yi;
                const float dz = shared_z[j] - zi;
                const float dist2 = dx * dx + dy * dy + dz * dz + softening2;
                const float inv_dist = rsqrtf(dist2);
                const float inv_dist3 = inv_dist * inv_dist * inv_dist;
                const float factor = G * shared_mass[j] * inv_dist3;
                axi += factor * dx;
                ayi += factor * dy;
                azi += factor * dz;
            }
        }
        __syncthreads();
    }

    // 越界线程也必须走完上面的同步，所以不能像 baseline 那样在开头直接 return。
    if (active) {
        ax[i] = axi;
        ay[i] = ayi;
        az[i] = azi;
    }
}

__global__ void compute_acceleration_tiled_float4(
                                                  const float4* __restrict__ position_mass,
                                                  float4* __restrict__ acceleration,
                                                  int n,
                                                  float G,
                                                  float softening2) {
    extern __shared__ float4 shared_particles[];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < n;
    const float4 current = active
        ? position_mass[i]
        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float axi = 0.0f;
    float ayi = 0.0f;
    float azi = 0.0f;

    // 这个版本和 tiled 的算法完全一样，差别只在数据组织。
    // x、y、z、mass 打包成一个 16 字节对齐的 float4，加载 tile 时每个线程
    // 只发出一次向量读取。总字节数没有减少，所以它是单独的实验项，不假定一定更快。
    for (int tile_start = 0; tile_start < n; tile_start += blockDim.x) {
        const int source = tile_start + threadIdx.x;
        if (source < n) {
            shared_particles[threadIdx.x] = position_mass[source];
        }
        __syncthreads();

        // 最后一块不一定正好有 blockDim.x 个粒子。
        // 这里按实际数量遍历，N 不是 block size 整数倍时也不会读越界。
        const int valid_count = min(static_cast<int>(blockDim.x), n - tile_start);
        if (active) {
            #pragma unroll 4
            for (int j = 0; j < valid_count; ++j) {
                if (tile_start + j == i) continue;
                const float4 other = shared_particles[j];
                const float dx = other.x - current.x;
                const float dy = other.y - current.y;
                const float dz = other.z - current.z;
                const float dist2 = dx * dx + dy * dy + dz * dz + softening2;

                // rsqrtf 直接得到 1/sqrt(x)，后面的 1/r^3 用三次乘法得到。
                // 这里比在最内层调用 pow(dist2, 1.5) 更适合这种 kernel。
                const float inv_dist = rsqrtf(dist2);
                const float inv_dist3 = inv_dist * inv_dist * inv_dist;
                const float factor = G * other.w * inv_dist3;
                axi += factor * dx;
                ayi += factor * dy;
                azi += factor * dz;
            }
        }
        __syncthreads();
    }

    // ax、ay、az 在整个循环里只属于当前线程，先留在局部变量中。
    // 所有粒子处理完以后只写一次显存，避免在内层循环反复读写中间结果。
    if (active) {
        acceleration[i] = make_float4(axi, ayi, azi, 0.0f);
    }
}

__global__ void euler_update(float* x, float* y, float* z,
                             float* vx, float* vy, float* vz,
                             const float* ax, const float* ay, const float* az,
                             int n, float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // 显式 Euler：位置使用本步开始时的旧速度。
    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
    vx[i] += ax[i] * dt;
    vy[i] += ay[i] * dt;
    vz[i] += az[i] * dt;
}

__global__ void leapfrog_kick_drift(float* x, float* y, float* z,
                                    float* vx, float* vy, float* vz,
                                    const float* ax, const float* ay, const float* az,
                                    int n, float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float half_dt = 0.5f * dt;
    vx[i] += ax[i] * half_dt;
    vy[i] += ay[i] * half_dt;
    vz[i] += az[i] * half_dt;
    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
}

__global__ void leapfrog_second_kick(float* vx, float* vy, float* vz,
                                     const float* ax, const float* ay, const float* az,
                                     int n, float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float half_dt = 0.5f * dt;
    vx[i] += ax[i] * half_dt;
    vy[i] += ay[i] * half_dt;
    vz[i] += az[i] * half_dt;
}

__global__ void euler_update_float4(float4* position_mass,
                                    float4* velocity,
                                    const float4* acceleration,
                                    int n,
                                    float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 p = position_mass[i];
    float4 v = velocity[i];
    const float4 a = acceleration[i];
    p.x += v.x * dt;
    p.y += v.y * dt;
    p.z += v.z * dt;
    v.x += a.x * dt;
    v.y += a.y * dt;
    v.z += a.z * dt;
    position_mass[i] = p;
    velocity[i] = v;
}

__global__ void leapfrog_kick_drift_float4(float4* position_mass,
                                           float4* velocity,
                                           const float4* acceleration,
                                           int n,
                                           float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 p = position_mass[i];
    float4 v = velocity[i];
    const float4 a = acceleration[i];
    const float half_dt = 0.5f * dt;
    v.x += a.x * half_dt;
    v.y += a.y * half_dt;
    v.z += a.z * half_dt;
    p.x += v.x * dt;
    p.y += v.y * dt;
    p.z += v.z * dt;
    position_mass[i] = p;
    velocity[i] = v;
}

__global__ void leapfrog_second_kick_float4(float4* velocity,
                                            const float4* acceleration,
                                            int n,
                                            float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 v = velocity[i];
    const float4 a = acceleration[i];
    const float half_dt = 0.5f * dt;
    v.x += a.x * half_dt;
    v.y += a.y * half_dt;
    v.z += a.z * half_dt;
    velocity[i] = v;
}

void launch_acceleration(const DeviceArrays& d,
                         const DevicePackedArrays& packed,
                         int n,
                         const SimConfig& config,
                         int update_blocks,
                         cudaStream_t stream) {
    const int blocks = update_blocks;
    NBODY_NVTX_PUSH("acceleration");
    if (config.kernel == "baseline") {
        compute_acceleration_baseline<<<blocks, config.block_size, 0, stream>>>(
            d.x, d.y, d.z, d.mass, d.ax, d.ay, d.az,
            n, config.G, config.softening * config.softening);
    } else if (config.kernel == "tiled") {
        const std::size_t shared_bytes =
            4ULL * static_cast<std::size_t>(config.block_size) * sizeof(float);
        compute_acceleration_tiled<<<blocks, config.block_size, shared_bytes, stream>>>(
            d.x, d.y, d.z, d.mass, d.ax, d.ay, d.az,
            n, config.G, config.softening * config.softening);
    } else if (config.kernel == "split-source") {
        const dim3 split_grid(update_blocks, config.source_chunks);
        compute_acceleration_partial<<<split_grid, config.block_size, 0, stream>>>(
            d.x, d.y, d.z, d.mass,
            d.partial_ax, d.partial_ay, d.partial_az,
            n, config.source_chunks, config.G, config.softening * config.softening);
        reduce_acceleration_partials<<<update_blocks, config.block_size, 0, stream>>>(
            d.partial_ax, d.partial_ay, d.partial_az,
            d.ax, d.ay, d.az, n, config.source_chunks);
    } else if (config.kernel == "tiled-float4") {
        const std::size_t shared_bytes =
            static_cast<std::size_t>(config.block_size) * sizeof(float4);
        compute_acceleration_tiled_float4<<<update_blocks, config.block_size, shared_bytes, stream>>>(
            packed.position_mass, packed.acceleration,
            n, config.G, config.softening * config.softening);
    }
    CUDA_CHECK(cudaGetLastError());
    NBODY_NVTX_POP();
}

void copy_initial_to_device(const std::vector<Particle>& particles,
                            DeviceArrays& d,
                            std::vector<float>& x,
                            std::vector<float>& y,
                            std::vector<float>& z) {
    const int n = static_cast<int>(particles.size());
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    std::vector<float> vx(n), vy(n), vz(n), mass(n);
    x.resize(n); y.resize(n); z.resize(n);
    for (int i = 0; i < n; ++i) {
        x[i] = particles[i].x; y[i] = particles[i].y; z[i] = particles[i].z;
        vx[i] = particles[i].vx; vy[i] = particles[i].vy; vz[i] = particles[i].vz;
        mass[i] = particles[i].mass;
    }

    // 文件读取用 AoS 很直观；GPU 改成 SoA 后，同一 warp 读取 x[i] 时地址连续，
    // 比把七个字段交错放在一个结构体里更适合合并访存。
    CUDA_CHECK(cudaMemcpy(d.x, x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.y, y.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.z, z.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vx, vx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vy, vy.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vz, vz.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.mass, mass.data(), bytes, cudaMemcpyHostToDevice));
}

void copy_initial_to_device(const std::vector<Particle>& particles,
                            DevicePackedArrays& d,
                            std::vector<float>& x,
                            std::vector<float>& y,
                            std::vector<float>& z) {
    const int n = static_cast<int>(particles.size());
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float4);
    std::vector<float4> position_mass(n), velocity(n);
    x.resize(n); y.resize(n); z.resize(n);
    for (int i = 0; i < n; ++i) {
        const Particle& p = particles[i];
        x[i] = p.x; y[i] = p.y; z[i] = p.z;
        position_mass[i] = make_float4(p.x, p.y, p.z, p.mass);
        velocity[i] = make_float4(p.vx, p.vy, p.vz, 0.0f);
    }
    CUDA_CHECK(cudaMemcpy(d.position_mass, position_mass.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.velocity, velocity.data(), bytes, cudaMemcpyHostToDevice));
}

void copy_positions_from_device(const DeviceArrays& d,
                                std::vector<float>& x,
                                std::vector<float>& y,
                                std::vector<float>& z) {
    const std::size_t bytes = x.size() * sizeof(float);
    CUDA_CHECK(cudaMemcpy(x.data(), d.x, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(y.data(), d.y, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(z.data(), d.z, bytes, cudaMemcpyDeviceToHost));
}

void copy_positions_from_device(const DevicePackedArrays& d,
                                std::vector<float>& x,
                                std::vector<float>& y,
                                std::vector<float>& z) {
    const int n = static_cast<int>(x.size());
    std::vector<float4> position_mass(n);
    CUDA_CHECK(cudaMemcpy(position_mass.data(), d.position_mass,
                          static_cast<std::size_t>(n) * sizeof(float4),
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        x[i] = position_mass[i].x;
        y[i] = position_mass[i].y;
        z[i] = position_mass[i].z;
    }
}

std::vector<Particle> copy_final_from_device(const DeviceArrays& d,
                                             const std::vector<Particle>& initial) {
    const int n = static_cast<int>(initial.size());
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    std::vector<float> x(n), y(n), z(n), vx(n), vy(n), vz(n);
    CUDA_CHECK(cudaMemcpy(x.data(), d.x, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(y.data(), d.y, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(z.data(), d.z, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vx.data(), d.vx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vy.data(), d.vy, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vz.data(), d.vz, bytes, cudaMemcpyDeviceToHost));

    std::vector<Particle> result = initial;
    for (int i = 0; i < n; ++i) {
        result[i].x = x[i]; result[i].y = y[i]; result[i].z = z[i];
        result[i].vx = vx[i]; result[i].vy = vy[i]; result[i].vz = vz[i];
    }
    return result;
}

std::vector<Particle> copy_final_from_device(const DevicePackedArrays& d,
                                             const std::vector<Particle>& initial) {
    const int n = static_cast<int>(initial.size());
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float4);
    std::vector<float4> position_mass(n), velocity(n);
    CUDA_CHECK(cudaMemcpy(position_mass.data(), d.position_mass, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(velocity.data(), d.velocity, bytes, cudaMemcpyDeviceToHost));
    std::vector<Particle> result = initial;
    for (int i = 0; i < n; ++i) {
        result[i].x = position_mass[i].x;
        result[i].y = position_mass[i].y;
        result[i].z = position_mass[i].z;
        result[i].mass = position_mass[i].w;
        result[i].vx = velocity[i].x;
        result[i].vy = velocity[i].y;
        result[i].vz = velocity[i].z;
    }
    return result;
}

}  // namespace

GpuSimulationResult simulate_gpu(const std::vector<Particle>& initial_particles,
                                 const SimConfig& config,
                                 const std::string& trajectory_path) {
    const int n = static_cast<int>(initial_particles.size());
    cudaDeviceProp device_property{};
    CUDA_CHECK(cudaGetDeviceProperties(&device_property, 0));
    if (config.block_size > device_property.maxThreadsPerBlock) {
        throw std::runtime_error("block_size 超过当前 GPU 的 maxThreadsPerBlock");
    }

    const bool use_packed_layout = config.kernel == "tiled-float4";
    DeviceArrays d;
    DevicePackedArrays packed;
    if (use_packed_layout) {
        packed.allocate(n);
    } else {
        d.allocate(n);
        if (config.kernel == "split-source") {
            d.allocate_partials(n, config.source_chunks);
        }
    }
    GpuSimulationResult result;
    result.performance.gpu_allocated_bytes = use_packed_layout
        ? 3ULL * n * sizeof(float4)
        : 10ULL * n * sizeof(float);
    if (config.kernel == "split-source") {
        result.performance.gpu_allocated_bytes +=
            3ULL * n * config.source_chunks * sizeof(float);
    }
    CUDA_CHECK(cudaMemGetInfo(&result.performance.gpu_free_bytes_after_alloc,
                             &result.performance.gpu_total_bytes));

    std::vector<float> host_x, host_y, host_z;
    EventTimer kernel_timer;
    EventTimer transfer_timer;
    SimulationStream simulation_stream;
    const auto wall_start = std::chrono::steady_clock::now();
    transfer_timer.start();
    if (use_packed_layout) {
        copy_initial_to_device(initial_particles, packed, host_x, host_y, host_z);
    } else {
        copy_initial_to_device(initial_particles, d, host_x, host_y, host_z);
    }
    result.performance.h2d_ms = transfer_timer.stop();

    TrajectoryWriter trajectory(trajectory_path, n, expected_record_count(config));
    auto write_start = std::chrono::steady_clock::now();
    NBODY_NVTX_PUSH("trajectory_write");
    trajectory.write_record(host_x, host_y, host_z);
    NBODY_NVTX_POP();
    result.performance.trajectory_write_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - write_start).count();

    const int blocks = (n + config.block_size - 1) / config.block_size;
    auto launch_one_step = [&]() {
        if (use_packed_layout) {
            if (config.integrator == "euler") {
                launch_acceleration(d, packed, n, config, blocks, simulation_stream);
                NBODY_NVTX_PUSH("euler_update");
                euler_update_float4<<<blocks, config.block_size, 0, simulation_stream>>>(
                    packed.position_mass, packed.velocity, packed.acceleration,
                    n, config.dt);
                CUDA_CHECK(cudaGetLastError());
                NBODY_NVTX_POP();
            } else {
                NBODY_NVTX_PUSH("kick_drift");
                leapfrog_kick_drift_float4<<<blocks, config.block_size, 0, simulation_stream>>>(
                    packed.position_mass, packed.velocity, packed.acceleration,
                    n, config.dt);
                CUDA_CHECK(cudaGetLastError());
                NBODY_NVTX_POP();
                launch_acceleration(d, packed, n, config, blocks, simulation_stream);
                NBODY_NVTX_PUSH("final_kick");
                leapfrog_second_kick_float4<<<blocks, config.block_size, 0, simulation_stream>>>(
                    packed.velocity, packed.acceleration, n, config.dt);
                CUDA_CHECK(cudaGetLastError());
                NBODY_NVTX_POP();
            }
        } else {
            if (config.integrator == "euler") {
                launch_acceleration(d, packed, n, config, blocks, simulation_stream);
                NBODY_NVTX_PUSH("euler_update");
                euler_update<<<blocks, config.block_size, 0, simulation_stream>>>(
                    d.x, d.y, d.z, d.vx, d.vy, d.vz,
                    d.ax, d.ay, d.az, n, config.dt);
                CUDA_CHECK(cudaGetLastError());
                NBODY_NVTX_POP();
            } else {
                NBODY_NVTX_PUSH("kick_drift");
                leapfrog_kick_drift<<<blocks, config.block_size, 0, simulation_stream>>>(
                    d.x, d.y, d.z, d.vx, d.vy, d.vz,
                    d.ax, d.ay, d.az, n, config.dt);
                CUDA_CHECK(cudaGetLastError());
                NBODY_NVTX_POP();
                launch_acceleration(d, packed, n, config, blocks, simulation_stream);
                NBODY_NVTX_PUSH("final_kick");
                leapfrog_second_kick<<<blocks, config.block_size, 0, simulation_stream>>>(
                    d.vx, d.vy, d.vz, d.ax, d.ay, d.az, n, config.dt);
                CUDA_CHECK(cudaGetLastError());
                NBODY_NVTX_POP();
            }
        }
    };

    GraphStep graph_step;
    if (config.cuda_graph && config.num_steps > 0) {
        // nsys 确认 split-source 的 kernel 已经很短，多次 launch 的缝隙开始看得见。
        // 这里只 capture 固定的一步计算。轨迹 copy/写盘还在循环外层的记录点处，
        // 所以开 Graph 也不会把 record_interval 改掉。
        CUDA_CHECK(cudaStreamBeginCapture(simulation_stream, cudaStreamCaptureModeGlobal));
        launch_one_step();
        CUDA_CHECK(cudaStreamEndCapture(simulation_stream, &graph_step.graph));
        CUDA_CHECK(cudaGraphInstantiate(&graph_step.executable, graph_step.graph,
                                        nullptr, nullptr, 0));
    }

    bool kernel_batch_active = false;
    if (config.num_steps > 0) {
        // 原来每一步都 cudaEventSynchronize，会让 CPU 每步都等 GPU。
        // 现在从一个记录点连续提交到下一个记录点，再一次性结算这段 kernel 时间。
        // 轨迹 copy 和写文件发生在计时段之外，所以不会混进 gpu_simulation_ms。
        kernel_timer.start(simulation_stream);
        kernel_batch_active = true;
        if (config.integrator == "leapfrog") {
            launch_acceleration(d, packed, n, config, blocks, simulation_stream);
        }
    }

    for (int step = 1; step <= config.num_steps; ++step) {
        if (config.cuda_graph) {
            NBODY_NVTX_PUSH("cuda_graph_step");
            CUDA_CHECK(cudaGraphLaunch(graph_step.executable, simulation_stream));
            NBODY_NVTX_POP();
        } else {
            launch_one_step();
        }

        if (step % config.record_interval == 0) {
            result.performance.gpu_kernel_ms += kernel_timer.stop(simulation_stream);
            kernel_batch_active = false;

            // 轨迹只按间隔拷回。每步都拷的话，最后测到的主要会是 PCIe 和磁盘速度。
            transfer_timer.start();
            NBODY_NVTX_PUSH("trajectory_copy");
            if (use_packed_layout) {
                copy_positions_from_device(packed, host_x, host_y, host_z);
            } else {
                copy_positions_from_device(d, host_x, host_y, host_z);
            }
            NBODY_NVTX_POP();
            result.performance.d2h_ms += transfer_timer.stop();
            write_start = std::chrono::steady_clock::now();
            NBODY_NVTX_PUSH("trajectory_write");
            trajectory.write_record(host_x, host_y, host_z);
            NBODY_NVTX_POP();
            result.performance.trajectory_write_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - write_start).count();

            if (step < config.num_steps) {
                kernel_timer.start(simulation_stream);
                kernel_batch_active = true;
            }
        }
    }

    if (kernel_batch_active) {
        result.performance.gpu_kernel_ms += kernel_timer.stop(simulation_stream);
    }

    transfer_timer.start();
    NBODY_NVTX_PUSH("final_state_copy");
    result.particles = use_packed_layout
        ? copy_final_from_device(packed, initial_particles)
        : copy_final_from_device(d, initial_particles);
    NBODY_NVTX_POP();
    result.performance.d2h_ms += transfer_timer.stop();
    trajectory.close();
    CUDA_CHECK(cudaDeviceSynchronize());
    result.performance.simulation_wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - wall_start).count();
    return result;
}

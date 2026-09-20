// ============================================================
// nbody_kernels.cu —— CUDA N-body O(N^2) 引力模拟 (完整实现)
//   实现了:
//     [x] compute_accel_simple    (直接法, 1 thread/particle i, 遍历所有 j)
//     [x] compute_accel_tiling    (shared memory 分块, 减少 global mem 读, 性能更优)
//     [x] integrate_euler / integrate_leapfrog kernels
//     [x] nbody_integrator_init   (Leapfrog 起步: 先推 v 0.5dt)
//     [x] nbody_step_once         (主入口: accel + update)
//     [x] 全部内存管理 / GPU info 字符串 (stub 升级)
// ============================================================
#include "nbody_kernels.h"
#include "nbody_types.h"
#include <cuda_runtime.h>
#include <sstream>
#include <cstring>
#include <cstdio>

#ifndef NBODY_BLOCK_SIZE_DEFAULT
#define NBODY_BLOCK_SIZE_DEFAULT 256
#endif
#ifndef NBODY_TILE_SIZE_DEFAULT
#define NBODY_TILE_SIZE_DEFAULT 64
#endif

// -------- helper: SoA cudaMalloc N floats + check --------
static inline int alloc_float_dev(int N, float*& ptr) {
    cudaError_t e = cudaMalloc((void**)&ptr, (size_t)N * sizeof(float));
    return (e == cudaSuccess) ? 0 : (int)e;
}

int nbody_gpu_alloc_and_copy(const ParticleSet& h, ParticleSet& d) {
    if (h.N <= 0 || h.on_device)
        return -1;
    d.N = h.N;
    d.on_device = true;
    size_t sz = (size_t)h.N * sizeof(float);
    if (alloc_float_dev(h.N, d.x) || alloc_float_dev(h.N, d.y) || alloc_float_dev(h.N, d.z) ||
        alloc_float_dev(h.N, d.vx) || alloc_float_dev(h.N, d.vy) || alloc_float_dev(h.N, d.vz) ||
        alloc_float_dev(h.N, d.mass) || alloc_float_dev(h.N, d.ax) || alloc_float_dev(h.N, d.ay) ||
        alloc_float_dev(h.N, d.az)) {
        cudaFree(d.x);
        cudaFree(d.y);
        cudaFree(d.z);
        cudaFree(d.vx);
        cudaFree(d.vy);
        cudaFree(d.vz);
        cudaFree(d.mass);
        cudaFree(d.ax);
        cudaFree(d.ay);
        cudaFree(d.az);
        return -2;
    }
    auto cp = [&](float* dst, const float* src) {
        return cudaMemcpy(dst, src, sz, cudaMemcpyHostToDevice);
    };
    if (cp(d.x, h.x) != cudaSuccess || cp(d.y, h.y) != cudaSuccess || cp(d.z, h.z) != cudaSuccess ||
        cp(d.vx, h.vx) != cudaSuccess || cp(d.vy, h.vy) != cudaSuccess ||
        cp(d.vz, h.vz) != cudaSuccess || cp(d.mass, h.mass) != cudaSuccess)
        return -3;
    // ax/ay/az 初始清零 (避免未初始化读取)
    cudaMemset(d.ax, 0, sz);
    cudaMemset(d.ay, 0, sz);
    cudaMemset(d.az, 0, sz);
    return 0;
}

int nbody_gpu_free(ParticleSet& d) {
    if (!d.on_device)
        return -1;
    cudaFree(d.x);
    cudaFree(d.y);
    cudaFree(d.z);
    cudaFree(d.vx);
    cudaFree(d.vy);
    cudaFree(d.vz);
    cudaFree(d.mass);
    cudaFree(d.ax);
    cudaFree(d.ay);
    cudaFree(d.az);
    d.N = 0;
    d.on_device = false;
    d.x = d.y = d.z = d.vx = d.vy = d.vz = d.mass = d.ax = d.ay = d.az = nullptr;
    return 0;
}

int nbody_gpu_copy_positions_back(const ParticleSet& d, ParticleSet& h) {
    if (!d.on_device || h.on_device || d.N != h.N)
        return -1;
    size_t sz = (size_t)d.N * sizeof(float);
    if (cudaMemcpy(h.x, d.x, sz, cudaMemcpyDeviceToHost) != cudaSuccess)
        return -2;
    if (cudaMemcpy(h.y, d.y, sz, cudaMemcpyDeviceToHost) != cudaSuccess)
        return -3;
    if (cudaMemcpy(h.z, d.z, sz, cudaMemcpyDeviceToHost) != cudaSuccess)
        return -4;
    return 0;
}

// ============================================================
// [Kernel 1] Simple direct O(N^2) acceleration kernel
//   thread i -> compute a_i = Σ_{j!=i} G m_j (r_j - r_i) / (|r|^2 + eps^2)^(3/2)
// ============================================================
__global__ void k_accel_simple(int N, const float* __restrict__ x, const float* __restrict__ y,
                               const float* __restrict__ z, const float* __restrict__ mass,
                               float* __restrict__ ax, float* __restrict__ ay,
                               float* __restrict__ az, float G, float eps2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;
    float xi = x[i], yi = y[i], zi = z[i];
    float axi = 0.f, ayi = 0.f, azi = 0.f;
    for (int j = 0; j < N; ++j) {
        float dx = x[j] - xi;
        float dy = y[j] - yi;
        float dz = z[j] - zi;
        float mj = mass[j];
        float r2 = dx * dx + dy * dy + dz * dz + eps2;
        float inv_r3 = rsqrt(r2) / r2; // rsqrt = 1/sqrt (fast_math)
        float f = G * mj * inv_r3;
        axi += f * dx;
        ayi += f * dy;
        azi += f * dz;
    }
    ax[i] = axi;
    ay[i] = ayi;
    az[i] = azi;
}

// ============================================================
// [Kernel 2] Shared-memory tiling (经典 BHTree / direct nbody 通用优化)
//   tile_j 将 j 端的 (x,y,z,mass) 分块读进 shared memory,
//   每个 block 处理 blockDim 个 i's, 同时对同一 tile j 内的 N/tile 粒度减少 global 读
// ============================================================
template <int TILE>
__global__ void k_accel_tiling(int N, const float* __restrict__ x, const float* __restrict__ y,
                               const float* __restrict__ z, const float* __restrict__ mass,
                               float* __restrict__ ax, float* __restrict__ ay,
                               float* __restrict__ az, float G, float eps2) {
    __shared__ float sx[TILE], sy[TILE], sz[TILE], sm[TILE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    float xi, yi, zi;
    if (i < N) {
        xi = x[i];
        yi = y[i];
        zi = z[i];
    } else {
        xi = yi = zi = 0.f;
    }
    float axi = 0.f, ayi = 0.f, azi = 0.f;
    for (int tile = 0; tile < N; tile += TILE) {
        // 协作装填 shared (blockDim >= TILE 时没问题; 否则多线程轮转)
        for (int k = tx; k < TILE; k += blockDim.x) {
            int j = tile + k;
            if (j < N) {
                sx[k] = x[j];
                sy[k] = y[j];
                sz[k] = z[j];
                sm[k] = mass[j];
            } else {
                sx[k] = 0;
                sy[k] = 0;
                sz[k] = 0;
                sm[k] = 0;
            }
        }
        __syncthreads();
        if (i < N) {
            for (int k = 0; k < TILE; ++k) {
                float dx = sx[k] - xi;
                float dy = sy[k] - yi;
                float dz = sz[k] - zi;
                float mj = sm[k];
                float r2 = dx * dx + dy * dy + dz * dz + eps2;
                float inv_r3 = rsqrt(r2) / r2;
                float f = G * mj * inv_r3;
                axi += f * dx;
                ayi += f * dy;
                azi += f * dz;
            }
        }
        __syncthreads();
    }
    if (i < N) {
        ax[i] = axi;
        ay[i] = ayi;
        az[i] = azi;
    }
}

// ============================================================
// [Kernel 3] Euler integrate
//   v_new = v_old + a*dt;  x_new = x_old + v_new*dt   (显式 Euler 定义)
// ============================================================
__global__ void k_integrate_euler(int N, float dt, float* __restrict__ x, float* __restrict__ y,
                                  float* __restrict__ z, float* __restrict__ vx,
                                  float* __restrict__ vy, float* __restrict__ vz,
                                  const float* __restrict__ ax, const float* __restrict__ ay,
                                  const float* __restrict__ az) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;
    float d = dt;
    vx[i] += ax[i] * d;
    vy[i] += ay[i] * d;
    vz[i] += az[i] * d;
    x[i] += vx[i] * d;
    y[i] += vy[i] * d;
    z[i] += vz[i] * d;
}

// ============================================================
// [Kernel 4] Leapfrog step
//   按 v_{t-0.5dt} 已经存进 vx/vy/vz 的约定:
//     v_{t+0.5dt} = v_{t-0.5dt} + a(t) * dt
//     x(t+dt)     = x(t) + v_{t+0.5dt} * dt
// ============================================================
__global__ void k_integrate_leapfrog(int N, float dt, float* __restrict__ x, float* __restrict__ y,
                                     float* __restrict__ z, float* __restrict__ vx,
                                     float* __restrict__ vy, float* __restrict__ vz,
                                     const float* __restrict__ ax, const float* __restrict__ ay,
                                     const float* __restrict__ az) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;
    float d = dt;
    vx[i] += ax[i] * d;
    vy[i] += ay[i] * d;
    vz[i] += az[i] * d;
    x[i] += vx[i] * d;
    y[i] += vy[i] * d;
    z[i] += vz[i] * d;
}

// ============================================================
// [Kernel 5] Leapfrog init: v_half = v0 + 0.5*dt*a0
//   等价于先算 a0, 再推 v 半半步
// ============================================================
__global__ void k_leapfrog_init_kick(int N, float half_dt, float* __restrict__ vx,
                                     float* __restrict__ vy, float* __restrict__ vz,
                                     const float* __restrict__ ax, const float* __restrict__ ay,
                                     const float* __restrict__ az) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;
    vx[i] += ax[i] * half_dt;
    vy[i] += ay[i] * half_dt;
    vz[i] += az[i] * half_dt;
}

// ============================================================
// Host wrapper
// ============================================================
static inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

static int call_compute_accel(ParticleSet& d, const SimParams& sp) {
    int N = d.N;
    if (N <= 0)
        return 0;
    int bs = (sp.block_size > 0) ? sp.block_size : NBODY_BLOCK_SIZE_DEFAULT;
    int grid = ceil_div(N, bs);
    float G_f = (float)sp.G;
    float eps2 = (float)(sp.softening * sp.softening);
    if (sp.kernel_mode == KernelMode::SHARED_TILING) {
        constexpr int TILE = NBODY_TILE_SIZE_DEFAULT;
        k_accel_tiling<TILE><<<grid, bs>>>(N, d.x, d.y, d.z, d.mass, d.ax, d.ay, d.az, G_f, eps2);
    } else {
        k_accel_simple<<<grid, bs>>>(N, d.x, d.y, d.z, d.mass, d.ax, d.ay, d.az, G_f, eps2);
    }
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}

int nbody_integrator_init(ParticleSet& d, const SimParams& sp) {
    if (d.N <= 0)
        return 0;
    if (sp.integrator == Integrator::EULER)
        return 0; // Euler 无需起步半半步
    // Leapfrog: a(x0) → v_half = v0 + 0.5*dt*a0
    int r = call_compute_accel(d, sp);
    if (r != 0)
        return r;
    int bs = (sp.block_size > 0) ? sp.block_size : NBODY_BLOCK_SIZE_DEFAULT;
    int grid = ceil_div(d.N, bs);
    float half_dt = (float)(0.5 * sp.dt);
    k_leapfrog_init_kick<<<grid, bs>>>(d.N, half_dt, d.vx, d.vy, d.vz, d.ax, d.ay, d.az);
    return (int)cudaGetLastError();
}

int nbody_step_once(ParticleSet& d, const SimParams& sp) {
    if (d.N <= 0)
        return 0;
    int bs = (sp.block_size > 0) ? sp.block_size : NBODY_BLOCK_SIZE_DEFAULT;
    int grid = ceil_div(d.N, bs);
    int r;
    if (sp.integrator == Integrator::EULER) {
        // Euler: a(t) -> v(t+dt) -> x(t+dt) (用新 v 更新 x)
        r = call_compute_accel(d, sp);
        if (r)
            return r;
        k_integrate_euler<<<grid, bs>>>(d.N, (float)sp.dt, d.x, d.y, d.z, d.vx, d.vy, d.vz, d.ax,
                                        d.ay, d.az);
    } else {
        // Leapfrog: 这里的 v 已经是 v(t-0.5dt), 但主循环入口还没算 a(t)
        // 步骤: 1) accel(t)   2) v(t-0.5) + a(t)*dt = v(t+0.5)
        //       3) x(t) + v(t+0.5)*dt = x(t+dt)
        r = call_compute_accel(d, sp);
        if (r)
            return r;
        k_integrate_leapfrog<<<grid, bs>>>(d.N, (float)sp.dt, d.x, d.y, d.z, d.vx, d.vy, d.vz, d.ax,
                                           d.ay, d.az);
    }
    return (int)cudaGetLastError();
}

// ============================================================
// GPU 信息字符串 (C API, extern "C" 跨 g++/nvcc 稳定)
// ============================================================
extern "C" const char* nbody_gpu_info_string_c(void) {
    static char info_buf[1024];
    info_buf[0] = '\0';
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess || n <= 0) {
        std::snprintf(info_buf, sizeof(info_buf),
                      "GPU=N/A (cudaGetDeviceCount failed or no CUDA device: err=%d)", (int)e);
        return info_buf;
    }
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, 0);
    int rt_ver = 0, dr_ver = 0;
    cudaRuntimeGetVersion(&rt_ver);
    cudaDriverGetVersion(&dr_ver);
    int rtmaj = rt_ver / 1000, rtmin = (rt_ver % 1000) / 10;
    int drmaj = dr_ver / 1000, drmin = (dr_ver % 1000) / 10;
    size_t mem_mb = (size_t)p.totalGlobalMem / (1024UL * 1024UL);
    std::snprintf(info_buf, sizeof(info_buf),
                  "GPU=%s  SMs=%d  CC=%d.%d  VRAM=%zuMB  CUDA_RT=%d.%d  DRV=%d.%d", p.name,
                  p.multiProcessorCount, p.major, p.minor, mem_mb, rtmaj, rtmin, drmaj, drmin);
    return info_buf;
}

// Host wrapper so main.cpp does not depend on cuda_runtime.h
extern "C" int nbody_cuda_device_sync(void) {
    return (int)cudaDeviceSynchronize();
}

// 显存越界保护: 估算 SoA 10 数组 + 额外 20% 安全余量 是否能 fit
extern "C" int nbody_gpu_memory_ok_c(int64_t N, char* out_msg, int cap) {
    if (out_msg && cap > 0)
        out_msg[0] = '\0';
    if (N <= 0) {
        if (out_msg)
            std::snprintf(out_msg, cap, "N must be positive (got N=%lld)", (long long)N);
        return 0;
    }
    size_t free_bytes = 0, total_bytes = 0;
    cudaError_t e = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (e != cudaSuccess) {
        if (out_msg)
            std::snprintf(out_msg, cap, "cudaMemGetInfo failed (err=%d). 请检查 CUDA 驱动/环境",
                          (int)e);
        return 0;
    }
    // SoA 数组: x/y/z/vx/ky/vz/mass/ax/ay/az = 10 × N floats (4 bytes each) = 40N
    // 额外 + 1 float 安全防止 N=边界 时刚好精确分配, 再加 20% 的运行时中间量/tile buffer 冗余
    size_t need = (size_t)N * 10 * sizeof(float);
    need = (size_t)(need * 1.20f) + (1 << 20); // +20% + 1MB 安全带
    size_t free_mb = free_bytes / (1024UL * 1024UL);
    size_t need_mb = need / (1024UL * 1024UL) + (need % (1024UL * 1024UL) ? 1 : 0);
    size_t total_mb = total_bytes / (1024UL * 1024UL);
    bool ok = (free_bytes >= need);
    if (out_msg) {
        if (ok) {
            std::snprintf(out_msg, cap,
                          "OK: N=%lld need~%zuMB, GPU free=%zuMB / total=%zuMB (headroom=%zuMB)",
                          (long long)N, need_mb, free_mb, total_mb,
                          free_mb > need_mb ? free_mb - need_mb : 0);
        } else {
            std::snprintf(
                out_msg, cap,
                "ERR: GPU VRAM 不足! N=%lld need~%zuMB, 但当前 free=%zuMB / total=%zuMB. "
                "建议: 减小 N, 或关闭其它占用 GPU 的程序 (Chrome/IDE/浏览器WebGL/WebGPU/虚拟机).",
                (long long)N, need_mb, free_mb, total_mb);
        }
    }
    return ok ? 1 : 0;
}

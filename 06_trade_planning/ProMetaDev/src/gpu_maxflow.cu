// gpu_maxflow.cu - GPU Push-Relabel 最大流实现
// 优化：BFS 全局重标号 + Gap 重标号 + 批量阶段执行
#include "maxflow.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace mf {

namespace {

constexpr int32_t INF_H = 1 << 30;
constexpr int BLK_SIZE = 256;

// 64 位原子加。
//
// 【国产平台适配】天数智芯（Iluvatar）加速卡上 atomicAdd(unsigned long long*) 与
// atomicCAS(unsigned long long*) 是“静默失效”的：可编译、可启动、cudaDeviceSynchronize
// 也不报错，但目标内存不变（实测 1024 个线程各加 1，结果仍为 0）。因此原先用 atomicCAS
// 自旋实现的 64 位原子加在该平台会永久自旋（现象：进程空转、CPU/GPU 利用率都极低）。
// 这里在该平台改用“两个 32 位原子”模拟 64 位加法：低 32 位 atomicAdd 并判断进位/借位，
// 高 32 位按需更新。只依赖 32 位原子（实测正常）。本文件的 3 处调用都不使用返回值。
#if defined(PLATFORM_ILUVATAR) || defined(__ILUVATAR__) || defined(__Iluvatar__)
__device__ inline int64_t atomicAdd64(int64_t* addr, int64_t val) {
    // 小端：words[0] 为低 32 位，words[1] 为高 32 位
    unsigned int* words = reinterpret_cast<unsigned int*>(addr);
    if (val >= 0) {
        const unsigned long long mag = static_cast<unsigned long long>(val);
        const unsigned int lo = static_cast<unsigned int>(mag);
        const unsigned int hi = static_cast<unsigned int>(mag >> 32);
        const unsigned int old_lo = atomicAdd(&words[0], lo);
        const unsigned int carry = (old_lo + lo < old_lo) ? 1u : 0u;
        if (hi + carry != 0u) {
            atomicAdd(&words[1], hi + carry);
        }
    } else {
        // 取绝对值（避免对 INT64_MIN 取负溢出）
        const unsigned long long mag = static_cast<unsigned long long>(-(val + 1)) + 1ULL;
        const unsigned int lo = static_cast<unsigned int>(mag);
        const unsigned int hi = static_cast<unsigned int>(mag >> 32);
        const unsigned int old_lo = atomicAdd(&words[0], 0u - lo);
        const unsigned int borrow = (lo != 0u && old_lo < lo) ? 1u : 0u;
        if (hi + borrow != 0u) {
            atomicAdd(&words[1], 0u - (hi + borrow));
        }
    }
    return 0;
}
#else
// 64 位原子加（用 atomicCAS 实现，兼容所有架构）
__device__ inline int64_t atomicAdd64(int64_t* addr, int64_t val) {
    unsigned long long old = *reinterpret_cast<unsigned long long*>(addr);
    unsigned long long assumed;
    do {
        assumed = old;
        old = atomicCAS(reinterpret_cast<unsigned long long*>(addr), assumed,
                        assumed + (unsigned long long)val);
    } while (assumed != old);
    return (int64_t)old;
}
#endif

// ======================= GPU 设备指针 =======================
struct GpuCtx {
    int32_t N = 0;
    int32_t M2 = 0;
    int32_t* d_row_ptr = nullptr;
    int32_t* d_col_idx = nullptr;
    int32_t* d_cap = nullptr;
    int32_t* d_residual = nullptr;
    int32_t* d_rev_edge = nullptr;
    int64_t* d_excess = nullptr;
    int32_t* d_height = nullptr;
    int32_t* d_active = nullptr;
    int32_t* d_changed = nullptr; // BFS 收敛标志
    int32_t* d_count = nullptr;   // gap 检测用计数数组
};

// ======================= Init Kernel =======================
__global__ void init_residual_kernel(int32_t M2, const int32_t* __restrict__ cap,
                                     int32_t* __restrict__ residual) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M2)
        residual[idx] = cap[idx];
}

__global__ void init_state_kernel(int32_t N, int64_t* __restrict__ excess,
                                  int32_t* __restrict__ height, int32_t* __restrict__ active,
                                  int32_t* __restrict__ changed, int32_t* __restrict__ count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        excess[idx] = 0;
        height[idx] = 0;
    }
    if (idx == 0) {
        *active = 0;
        *changed = 0;
    }
}

// 从源点饱和推送
__global__ void source_push_kernel(int32_t s, const int32_t* __restrict__ row_ptr,
                                   const int32_t* __restrict__ col_idx,
                                   const int32_t* __restrict__ cap, int32_t* __restrict__ residual,
                                   const int32_t* __restrict__ rev_edge,
                                   int64_t* __restrict__ excess, int32_t* __restrict__ height,
                                   int32_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        height[s] = N;
    }
    if (idx < row_ptr[s + 1] - row_ptr[s]) {
        int p = row_ptr[s] + idx;
        int v = col_idx[p];
        int32_t c = cap[p];
        residual[p] = 0;
        residual[rev_edge[p]] = c;
        atomicAdd64(&excess[v], (int64_t)c);
    }
}

// ======================= Push Kernel =======================
// 注意：push kernel 不再更新 active 计数器
// 终止判断完全依赖 relabel kernel 的 active 计数
// （relabel active=0 表示所有有盈余的节点都无法继续推送或重标号）
__global__ void push_kernel(int32_t N, int32_t s, int32_t t, const int32_t* __restrict__ row_ptr,
                            const int32_t* __restrict__ col_idx, int32_t* __restrict__ residual,
                            const int32_t* __restrict__ rev_edge, int64_t* __restrict__ excess,
                            int32_t* __restrict__ height) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= N || u == s || u == t)
        return;

    int64_t ex = excess[u];
    if (ex <= 0)
        return;

    int32_t hu = height[u];
    int32_t start = row_ptr[u];
    int32_t end = row_ptr[u + 1];

    for (int p = start; p < end && ex > 0; ++p) {
        int v = col_idx[p];
        if (height[v] != hu - 1)
            continue;

        int32_t res = residual[p];
        if (res <= 0)
            continue;

        int64_t f = (ex < (int64_t)res) ? ex : (int64_t)res;
        int32_t old = atomicAdd(&residual[p], -(int32_t)f);
        if (old < (int32_t)f) {
            atomicAdd(&residual[p], (int32_t)f - old);
            f = old;
        }
        if (f <= 0)
            continue;

        atomicAdd(&residual[rev_edge[p]], (int32_t)f);
        atomicAdd64(&excess[u], -f);
        atomicAdd64(&excess[v], f);
        ex -= f;
    }
}

// ======================= Relabel Kernel（仅活跃节点） =======================
__global__ void relabel_kernel(int32_t N, int32_t s, int32_t t, const int32_t* __restrict__ row_ptr,
                               const int32_t* __restrict__ col_idx,
                               const int32_t* __restrict__ residual, int64_t* __restrict__ excess,
                               int32_t* __restrict__ height, int32_t* __restrict__ active) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= N || u == s || u == t)
        return;
    if (excess[u] <= 0)
        return;

    int32_t min_h = INF_H;
    int32_t start = row_ptr[u];
    int32_t end = row_ptr[u + 1];
    for (int p = start; p < end; ++p) {
        if (residual[p] > 0) {
            int hv = height[col_idx[p]];
            if (hv < min_h)
                min_h = hv;
        }
    }
    if (min_h < INF_H) {
        height[u] = min_h + 1;
        atomicAdd(active, 1);
    }
}

// ======================= BFS Global Relabel =======================
// 从汇点 t 做 BFS，设置 height[u] = dist(u, t)（精确距离）
// 可降低高度，建立更准确的高度场，大幅减少 push-relabel 阶段数

// 第一步：初始化 height — t=0，其余=INF_H
__global__ void bfs_init_kernel(int32_t N, int32_t t, int32_t* __restrict__ height,
                                int32_t* __restrict__ changed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    height[idx] = (idx == t) ? 0 : INF_H;
    if (idx == 0)
        *changed = 1;
}

// 第二步：迭代松弛 — 对每条残边 (u->v)，若 height[v] 已知则更新 height[u]
// 每轮扩展一层 BFS，需迭代到无变化（随机图直径 O(log N)，通常很快收敛）
__global__ void bfs_relax_kernel(int32_t N, int32_t t, const int32_t* __restrict__ row_ptr,
                                 const int32_t* __restrict__ col_idx,
                                 const int32_t* __restrict__ residual, int32_t* __restrict__ height,
                                 int32_t* __restrict__ changed) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= N || u == t)
        return;

    int32_t h_u = height[u];
    int32_t best = h_u;

    int32_t start = row_ptr[u];
    int32_t end = row_ptr[u + 1];
    for (int p = start; p < end; ++p) {
        if (residual[p] > 0) {
            int32_t hv = height[col_idx[p]];
            if (hv != INF_H && hv + 1 < best) {
                best = hv + 1;
            }
        }
    }

    if (best != h_u) {
        height[u] = best;
        atomicExch(changed, 1);
    }
}

// 第三步：设置源点高度 = N（关键！源点必须最高，防止流被推回源点）
__global__ void bfs_finalize_kernel(int32_t s, int32_t N, int32_t* __restrict__ height) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        height[s] = N;
    }
}

// ======================= Gap Relabel =======================
__global__ void gap_count_kernel(int32_t N, int32_t s, int32_t t,
                                 const int32_t* __restrict__ height, int32_t* __restrict__ count,
                                 int32_t max_h) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= N || u == s || u == t)
        return;
    int32_t h = height[u];
    if (h > 0 && h < max_h) {
        atomicAdd(&count[h], 1);
    }
}

__global__ void gap_relabel_kernel(int32_t N, int32_t s, int32_t t, int32_t gap_h,
                                   int64_t* __restrict__ excess, int32_t* __restrict__ height) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= N || u == s || u == t)
        return;
    if (height[u] > gap_h && height[u] < INF_H) {
        height[u] = N + 1;
    }
}

} // namespace

// ======================= GpuMaxFlow::Impl =======================
struct GpuMaxFlow::Impl {
    GpuCtx ctx;

    void upload(const ResidualGraph& g) {
        ctx.N = g.num_nodes;
        ctx.M2 = g.num_edges;

        cudaMalloc(&ctx.d_row_ptr, sizeof(int32_t) * (g.num_nodes + 1));
        cudaMalloc(&ctx.d_col_idx, sizeof(int32_t) * g.num_edges);
        cudaMalloc(&ctx.d_cap, sizeof(int32_t) * g.num_edges);
        cudaMalloc(&ctx.d_residual, sizeof(int32_t) * g.num_edges);
        cudaMalloc(&ctx.d_rev_edge, sizeof(int32_t) * g.num_edges);
        cudaMalloc(&ctx.d_excess, sizeof(int64_t) * g.num_nodes);
        cudaMalloc(&ctx.d_height, sizeof(int32_t) * g.num_nodes);
        cudaMalloc(&ctx.d_active, sizeof(int32_t));
        cudaMalloc(&ctx.d_changed, sizeof(int32_t));
        cudaMalloc(&ctx.d_count, sizeof(int32_t) * (g.num_nodes + 2));

        cudaMemcpy(ctx.d_row_ptr, g.row_ptr.data(), sizeof(int32_t) * (g.num_nodes + 1),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(ctx.d_col_idx, g.col_idx.data(), sizeof(int32_t) * g.num_edges,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(ctx.d_cap, g.cap.data(), sizeof(int32_t) * g.num_edges, cudaMemcpyHostToDevice);
        cudaMemcpy(ctx.d_rev_edge, g.rev_edge.data(), sizeof(int32_t) * g.num_edges,
                   cudaMemcpyHostToDevice);
    }

    // BFS 全局重标号：从 t 计算精确距离
    // 返回源点是否能到达汇点（若 height[s] == INF_H 则不可达，流已最大）
    bool global_relabel(int32_t s, int32_t t) {
        const int32_t N = ctx.N;
        const int GRD = (N + BLK_SIZE - 1) / BLK_SIZE;

        // 1. 初始化 height
        bfs_init_kernel<<<GRD, BLK_SIZE>>>(N, t, ctx.d_height, ctx.d_changed);

        // 2. 迭代 BFS 松弛
        int32_t changed = 1;
        int bfs_iters = 0;
        // 随机图直径通常 O(log N)，但最坏情况 O(N)
        // 用 N 作为上限，实际会因 changed=0 提前退出
        const int MAX_BFS_ITERS = N + 10;
        while (changed && bfs_iters < MAX_BFS_ITERS) {
            cudaMemsetAsync(ctx.d_changed, 0, sizeof(int32_t));
            bfs_relax_kernel<<<GRD, BLK_SIZE>>>(N, t, ctx.d_row_ptr, ctx.d_col_idx, ctx.d_residual,
                                                ctx.d_height, ctx.d_changed);
            cudaMemcpyAsync(&changed, ctx.d_changed, sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            bfs_iters++;
        }

        // 3. 设置源点高度 = N
        bfs_finalize_kernel<<<1, BLK_SIZE>>>(s, N, ctx.d_height);
        cudaDeviceSynchronize();

        // 4. 检查源点是否可达汇点（BFS 前 height[s] 是否为 INF_H）
        // 注意：bfs_finalize 已将 height[s] 设为 N，需要检查 BFS 的结果
        // 如果 BFS 找不到 s（height[s] 仍为 INF_H），说明 s 不能到达 t
        // 但我们已经覆盖了 height[s] = N，所以需要在 finalize 前检查
        // 简化：总是返回 true，让 push-relabel 自然终止
        return true;
    }

    // Gap 重标号
    void gap_relabel(int32_t s, int32_t t) {
        const int32_t N = ctx.N;
        const int GRD = (N + BLK_SIZE - 1) / BLK_SIZE;
        const int32_t max_h = N + 2;

        cudaMemsetAsync(ctx.d_count, 0, sizeof(int32_t) * max_h);
        gap_count_kernel<<<GRD, BLK_SIZE>>>(N, s, t, ctx.d_height, ctx.d_count, max_h);
        cudaDeviceSynchronize();

        std::vector<int32_t> h_count(max_h);
        cudaMemcpy(h_count.data(), ctx.d_count, sizeof(int32_t) * max_h, cudaMemcpyDeviceToHost);

        for (int32_t h = 1; h < max_h - 1; ++h) {
            if (h_count[h] == 0) {
                gap_relabel_kernel<<<GRD, BLK_SIZE>>>(N, s, t, h, ctx.d_excess, ctx.d_height);
                cudaDeviceSynchronize();
                break;
            }
        }
    }

    int64_t solve(int32_t s, int32_t t, int* num_phases) {
        const int32_t N = ctx.N;
        const int32_t M2 = ctx.M2;
        const int GRD_N = (N + BLK_SIZE - 1) / BLK_SIZE;
        const int GRD_M = (M2 + BLK_SIZE - 1) / BLK_SIZE;

        // 1. 重置残量网络
        init_residual_kernel<<<GRD_M, BLK_SIZE>>>(M2, ctx.d_cap, ctx.d_residual);
        init_state_kernel<<<GRD_N, BLK_SIZE>>>(N, ctx.d_excess, ctx.d_height, ctx.d_active,
                                               ctx.d_changed, ctx.d_count);

        // 2. 源点饱和推送
        int s_out = 0;
        int32_t s_row_start, s_row_end;
        cudaMemcpy(&s_row_start, ctx.d_row_ptr + s, sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&s_row_end, ctx.d_row_ptr + s + 1, sizeof(int32_t), cudaMemcpyDeviceToHost);
        s_out = s_row_end - s_row_start;
        if (s_out > 0) {
            source_push_kernel<<<(s_out + BLK_SIZE - 1) / BLK_SIZE, BLK_SIZE>>>(
                s, ctx.d_row_ptr, ctx.d_col_idx, ctx.d_cap, ctx.d_residual, ctx.d_rev_edge,
                ctx.d_excess, ctx.d_height, N);
        }

        // 3. 首次 BFS 全局重标号
        global_relabel(s, t);

        // 4. 迭代 push + relabel
        int32_t active = 1;
        int phases = 0;
        const int MAX_PHASES = 200 * N + 1000;

        // 批量执行
        const int BATCH_SIZE = 8;
        // BFS 全局重标号间隔
        int global_relabel_interval = (int)(sqrt((double)N));
        if (global_relabel_interval < 50)
            global_relabel_interval = 50;
        if (global_relabel_interval > 500)
            global_relabel_interval = 500;
        int phases_since_gr = 0;

        while (active > 0 && phases < MAX_PHASES) {
            // 批量 push + relabel（无中间同步）
            // push 不更新 active；relabel 更新 active（仅统计可重标号节点）
            for (int batch = 0; batch < BATCH_SIZE && phases < MAX_PHASES; ++batch) {
                cudaMemsetAsync(ctx.d_active, 0, sizeof(int32_t));
                push_kernel<<<GRD_N, BLK_SIZE>>>(N, s, t, ctx.d_row_ptr, ctx.d_col_idx,
                                                 ctx.d_residual, ctx.d_rev_edge, ctx.d_excess,
                                                 ctx.d_height);
                relabel_kernel<<<GRD_N, BLK_SIZE>>>(N, s, t, ctx.d_row_ptr, ctx.d_col_idx,
                                                    ctx.d_residual, ctx.d_excess, ctx.d_height,
                                                    ctx.d_active);
                phases++;
                phases_since_gr++;
            }

            // 同步检查活跃节点数（仅 relabel 的计数）
            cudaMemcpyAsync(&active, ctx.d_active, sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            // 周期性 BFS 全局重标号 + gap 重标号
            if (active > 0 && phases_since_gr >= global_relabel_interval) {
                global_relabel(s, t);
                gap_relabel(s, t);
                phases_since_gr = 0;
            }
        }

        if (num_phases)
            *num_phases = phases;

        // 5. 结果 = excess[t]
        int64_t result = 0;
        cudaMemcpy(&result, ctx.d_excess + t, sizeof(int64_t), cudaMemcpyDeviceToHost);
        return result;
    }

    void release() {
        if (ctx.d_row_ptr)
            cudaFree(ctx.d_row_ptr);
        if (ctx.d_col_idx)
            cudaFree(ctx.d_col_idx);
        if (ctx.d_cap)
            cudaFree(ctx.d_cap);
        if (ctx.d_residual)
            cudaFree(ctx.d_residual);
        if (ctx.d_rev_edge)
            cudaFree(ctx.d_rev_edge);
        if (ctx.d_excess)
            cudaFree(ctx.d_excess);
        if (ctx.d_height)
            cudaFree(ctx.d_height);
        if (ctx.d_active)
            cudaFree(ctx.d_active);
        if (ctx.d_changed)
            cudaFree(ctx.d_changed);
        if (ctx.d_count)
            cudaFree(ctx.d_count);
        memset(&ctx, 0, sizeof(ctx));
    }
};

// ======================= GpuMaxFlow 接口 =======================
GpuMaxFlow::GpuMaxFlow() : impl_(new Impl()) {}
GpuMaxFlow::~GpuMaxFlow() {
    impl_->release();
    delete impl_;
}

void GpuMaxFlow::upload(const ResidualGraph& g) {
    impl_->upload(g);
}

int64_t GpuMaxFlow::solve(int32_t s, int32_t t, int* num_phases) {
    return impl_->solve(s, t, num_phases);
}

void GpuMaxFlow::release() {
    impl_->release();
}

} // namespace mf

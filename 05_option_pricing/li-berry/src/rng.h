// rng.h - host/device 共享的随机数模块
//
// 设计契约（报告中详述）：
// 1. 每条路径拥有独立随机流，流位置仅由 (seed, path_index) 决定，与网格/块配置无关
//    → 同一配置重跑 bitwise 一致；N 较小的运行是 N 较大运行的精确子集（嵌套前缀）
// 2. custom RNG（xorshift128+）：CPU 与 GPU 的 32 位 uniform 流 bitwise 一致；
//    Box-Muller 正态因 glibc 与 CUDA 数学库 ~1ULP 差异，一致性到 ~1e-7
// 3. u32_to_uniform 映射到 (0,1) 开区间，保证 Box-Muller 中 log(u1) 不为 -inf
#pragma once

#include <cstdint>

#ifndef OP_HD
#ifdef __CUDACC__
#define OP_HD __host__ __device__ inline
#else
#define OP_HD inline
#endif
#endif

// ---------- 基础工具 ----------

// 24 位均匀整数 → (0,1) 开区间浮点。
// +0.5f 居中保证永不出现 0；对 0xFFFFFF 截断防止 float 舍入到 1.0
// （概率 6e-8 的映射畸变对蒙特卡洛可忽略，但 (0,1) 开区间是 Box-Muller 安全的前提）
OP_HD float u32_to_uniform(unsigned int x) {
    unsigned int v = x >> 8;
    if (v >= 0xFFFFFFu) v = 0xFFFFFEu;
    return (static_cast<float>(v) + 0.5f) * (1.0f / 16777216.0f);
}

// ---------- splitmix64：路径流播种混合器 ----------

OP_HD unsigned long long splitmix64(unsigned long long* x) {
    unsigned long long z = (*x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// ---------- xorshift128+（周期 2^128 - 1，通过 BigCrush）----------

struct XS128State {
    unsigned long long s0, s1;
};

OP_HD void xs128_seed(XS128State& st, unsigned long long seed, unsigned long long path) {
    // splitmix64 混合 (seed, path)，相邻路径得到不相关状态；s0 | 1 保证状态非全零
    unsigned long long s = seed ^ (path * 0x9E3779B97F4A7C15ull);
    st.s0 = splitmix64(&s) | 1ull;
    st.s1 = splitmix64(&s);
    // 预热 4 次，进一步消除播种相关性（每路径成本可忽略）
    for (int i = 0; i < 4; ++i) {
        unsigned long long x = st.s0;
        unsigned long long y = st.s1;
        st.s0 = y;
        x ^= x << 23;
        st.s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
    }
}

OP_HD unsigned long long xs128_next(XS128State& st) {
    unsigned long long x = st.s0;
    unsigned long long y = st.s1;
    st.s0 = y;
    x ^= x << 23;
    st.s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
    return st.s1 + y;
}

// 取高 32 位作为 uniform 整数（低位的线性性较弱，高位统计质量更好）
OP_HD unsigned int xs128_next_u32(XS128State& st) {
    return static_cast<unsigned int>(xs128_next(st) >> 32);
}

// ---------- Box-Muller：2 个 (0,1) uniform → 2 个标准正态 ----------

// CPU 用 sinf/cosf（glibc），GPU 用 sincosf（CUDA），两者结果差 ~1ULP
OP_HD void box_muller(float u1, float u2, float& n1, float& n2) {
    float r = sqrtf(-2.0f * logf(u1)); // u1 ∈ (0,1) 由 u32_to_uniform 保证
    float theta = 6.283185307179586f * u2;
#ifdef __CUDA_ARCH__
    float s, c;
    sincosf(theta, &s, &c);
#else
    float s = sinf(theta);
    float c = cosf(theta);
#endif
    n1 = r * c;
    n2 = r * s;
}

// ---------- 每步 4 个正态（2 次 Box-Muller）的生成器封装 ----------

// 自定义 RNG：2 次 xs128_next → 4 个 u32 → 4 个 uniform → 4 个正态
struct CustomGen4 {
    XS128State st;
    OP_HD void init(unsigned long long seed, unsigned long long path) {
        xs128_seed(st, seed, path);
    }
    OP_HD void next4(float* z) {
        float u0 = u32_to_uniform(xs128_next_u32(st));
        float u1 = u32_to_uniform(xs128_next_u32(st));
        float u2 = u32_to_uniform(xs128_next_u32(st));
        float u3 = u32_to_uniform(xs128_next_u32(st));
        box_muller(u0, u1, z[0], z[1]);
        box_muller(u2, u3, z[2], z[3]);
    }
};

#ifdef __CUDACC__
#include <curand_kernel.h>

// cuRAND Philox4x32-10：curand_uniform4 一次给出 4 个 (0,1] uniform
// 播种：curand_init(seed, subsequence = path_index, offset = 0)
// → 每条路径一条独立子序列（2^66 长度），流位置与网格配置无关
struct CurandGen4 {
    curandStatePhilox4_32_10_t st;
    __device__ inline void init(unsigned long long seed, unsigned long long path) {
        curand_init(seed, path, 0, &st);
    }
    __device__ inline void next4(float* z) {
        float4 u = curand_uniform4(&st); // (0,1]，u 不会为 0，log(u) 无 NaN 风险
        box_muller(u.x, u.y, z[0], z[1]);
        box_muller(u.z, u.w, z[2], z[3]);
    }
};
#endif // __CUDACC__

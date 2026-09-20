// ============================================================
// nbody_kernels.h —— CUDA kernel 包装函数声明 (host 端调用)
//   所有 <<<>>> kernel launch 都在 nbody_kernels.cu 内部完成;
//   其他 cpp 文件只需要 include 这个头文件并调用下面的函数即可。
// ============================================================
#ifndef NBODY_KERNELS_H
#define NBODY_KERNELS_H

#include "nbody_types.h"
#include <string>

// ============================================================
// Device 内存 / 拷贝 管理
// ============================================================

// 在 GPU 上分配 SoA 数组 + 从 host 粒子数据拷贝过去
//   返回值: 0 = 成功; 非0 = CUDA error
int nbody_gpu_alloc_and_copy(const ParticleSet& host_set, ParticleSet& dev_set);

// 释放 GPU 内存
int nbody_gpu_free(ParticleSet& dev_set);

// 把 dev_set.x/y/z 拉回到 host_set.x/y/z (轨迹记录用)
int nbody_gpu_copy_positions_back(const ParticleSet& dev_set, ParticleSet& host_set);

// ============================================================
// 积分 / 加速计算
// ============================================================

// 初始化 leapahead 半时间步速度 (如果 integrator == LEAPFROG, 在进入主循环前调用一次)
//   按 Glasserman / 数值辛积分教材标准: 先 v_half = v0 + 0.5*dt*a0
int nbody_integrator_init(ParticleSet& dev_set, const SimParams& sp);

// 推进一步: 计算加速度 → 更新 v, x (integrator 决定先后顺序)
//   每一步后 dev_set.{x,y,z,vx,vy,vz} 已被改写
int nbody_step_once(ParticleSet& dev_set, const SimParams& sp);

// ============================================================
// 设备信息字符串 / Device 同步封装
//  注意: 这些函数在 nbody_kernels.cu 中实现; 用 extern "C" 以便
//        g++ (main.cpp) 与 nvcc (nbody_kernels.cu) 链接时符号一致。
// ============================================================
#ifdef __cplusplus
extern "C" {
#endif

/* 返回 GPU info 的 C 字符串指针 (static buffer, 不释放, 线程不安全) */
const char* nbody_gpu_info_string_c(void);

/* 封装 cudaDeviceSynchronize, 返回 cudaError_t (int); main.cpp 禁止直接 include cuda_runtime.h */
int nbody_cuda_device_sync(void);

/* 显存预估: 1 -> 可容纳 N 粒子; 0 -> 不够, out_msg 写入原因 (out_msg_cap>=256) */
int nbody_gpu_memory_ok_c(int64_t N, char* out_msg, int out_msg_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

/* ---- C++ 便捷包装 (仅 .cpp / .cu 可用, 与旧代码兼容) ---- */
#if defined(__cplusplus)
#include <string>
static inline std::string nbody_gpu_info_string() {
    return std::string(nbody_gpu_info_string_c());
}
static inline bool nbody_gpu_check_memory_fit(int64_t N, std::string& out_msg) {
    char buf[512];
    buf[0] = 0;
    int ok = nbody_gpu_memory_ok_c(N, buf, (int)sizeof(buf));
    out_msg.assign(buf);
    return ok != 0;
}
#endif

#endif // NBODY_KERNELS_H

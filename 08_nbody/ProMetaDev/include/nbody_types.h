// ============================================================
// nbody_types.h —— 核心类型定义 (无函数实现, 仅结构体+枚举)
//   ParticleSet 使用 SoA (Structure-of-Arrays) 布局:
//     便于 CUDA kernel 连续读取，提高全局内存合并度
// ============================================================
#ifndef NBODY_TYPES_H
#define NBODY_TYPES_H

#include <cstdint>

// ---- 数值积分器 ----
enum class Integrator : int {
    EULER = 0,   // v(t+dt) = v(t)+a(t)*dt; x(t+dt)=x(t)+v(t)*dt    (能量漂移大, 简单)
    LEAPFROG = 1 // 辛积分: v(t+0.5dt)=v(t-0.5dt)+a(t)*dt; x(t+dt)=x(t)+v(t+0.5dt)*dt  (保能量)
};

// ---- 核函数加速实现模式 (预留钩子, 当前默认 SIMPLE) ----
enum class KernelMode : int {
    SIMPLE = 0,       // 直接 O(N^2): 1 thread per particle i, loop j=0..N-1
    SHARED_TILING = 1 // 分块 + shared memory 复用: 每块算 tile_i × tile_j, 减少 global mem 读
};

// ---- SoA 粒子集合 (host / device 内存布局完全一致) ----
//  host 端用 new / std::vector 分配; device 端用 cudaMalloc
struct ParticleSet {
    int32_t N;      // 粒子总数
    float* x;       // [N] 位置 x
    float* y;       // [N] 位置 y
    float* z;       // [N] 位置 z
    float* vx;      // [N] 速度 vx
    float* vy;      // [N] 速度 vy
    float* vz;      // [N] 速度 vz
    float* mass;    // [N] 质量
    float* ax;      // [N] 加速度 ax (kernel 输出)
    float* ay;      // [N] 加速度 ay
    float* az;      // [N] 加速度 az
    bool on_device; // true = 当前指针在 GPU 上; false = 在 CPU

    ParticleSet()
        : N(0), x(nullptr), y(nullptr), z(nullptr), vx(nullptr), vy(nullptr), vz(nullptr),
          mass(nullptr), ax(nullptr), ay(nullptr), az(nullptr), on_device(false) {}
};

// ---- 模拟参数 (从 params.txt 解析) ----
struct SimParams {
    double dt;               // 时间步长
    int64_t num_steps;       // 总步数
    int64_t record_interval; // 每多少步记录一次轨迹 (同时写轨迹文件+可选能量计算)
    double G;                // 引力常数 (通常 1.0)
    double softening;        // 软化因子 eps (防止 r→0 发散, 默认 1e-4)
    Integrator integrator;  // euler / leapfrog
    KernelMode kernel_mode; // SIMPLE / SHARED_TILING
    int block_size;         // CUDA block 大小 (默认 256)
    bool write_csv;         // 是否也写 CSV 轨迹 (默认只写二进制)
    bool skip_cpu;          // 跳过 CPU 参考 (不比较加速比)
    bool check_energy;      // 每 record_interval 计算总能量 (正确性用)
    int analyze_frames;     // 调试: 只运行前 K 帧 (0 = 不限制)

    // 构造函数: 给一组合理默认
    SimParams()
        : dt(1e-3), num_steps(1000), record_interval(100), G(1.0), softening(1e-4),
          integrator(Integrator::LEAPFROG), kernel_mode(KernelMode::SIMPLE), block_size(256),
          write_csv(false), skip_cpu(false), check_energy(false), analyze_frames(0) {}
};

// ---- 性能结果 (主机端填充) ----
struct PerfResult {
    double total_sim_ms;           // 总模拟时间 (含数据传输, 不含文件I/O)
    double avg_step_ms;            // 每步平均时间 (total/num_steps)
    double particle_steps_per_sec; // 吞吐: N * num_steps / total_time (s^-1)
    double gpu_mem_used_mb;        // 当前 GPU 上分配的显存 (MB)
    double cpu_time_ms;            // CPU 参考 O(N^2) 时间 (用于加速比)
    double speedup_vs_cpu;         // GPU 加速比 (cpu_time / total_sim)
};

#endif // NBODY_TYPES_H

// ============================================================
// file_io.h —— 粒子/参数解析 + 轨迹(二进制/CSV)/性能日志写
//   不依赖 CUDA runtime (纯 C++17 std 库即可编译)
// ============================================================
#ifndef NBODY_FILE_IO_H
#define NBODY_FILE_IO_H

#include "nbody_types.h"
#include <string>

// -------- 输入文件解析 --------
// 读取 particles.txt: 每行 "x y z vx vy vz mass"
//   host_set out: N、x/y/z/vx/vy/vz/mass 全部在 host 上分配并填充
//   返回 0 = ok
int parse_particles_file(const std::string& path, ParticleSet& host_set /* out */);

// 读取 params.txt: 键值对 (dt、num_steps、record_interval、G、softening、integrator)
//   sp out: 解析值写入 sp (未覆盖的字段保留 SimParams 默认值)
int parse_params_file(const std::string& path, SimParams& sp /* out */);

// 释放 host 上分配的 SoA 数组内存 (来自 parse_particles_file)
void free_host_particles(ParticleSet& host_set);

// -------- 轨迹输出 --------
// 写 "二进制" 轨迹文件 (题面推荐格式, 最紧凑)
//   头:  int32_t N = 粒子数;  int32_t R = 总帧数
//   每帧: N*3 floats (按粒子顺序 x_0 y_0 z_0 ... x_{N-1} y_{N-1} z_{N-1})
//   Usage:
//      TrajectoryBinWriter w;
//      w.open("traj.bin", N);
//      for each frame: w.append_frame(host_set.x, host_set.y, host_set.z);
//      w.close();  // 会把实际帧数回写到 header 的 R 位置
class TrajectoryBinWriter {
public:
    TrajectoryBinWriter();
    ~TrajectoryBinWriter();
    int open(const std::string& path, int32_t N);
    int append_frame(const float* x, const float* y, const float* z); // 长度均为 N
    void close();

private:
    struct Impl;
    Impl* m;
};

// 写 "CSV" 轨迹文件 (题面可选, 效率低, 但便于人工检查)
//   每帧写入 N 行: particle_id, step, x, y, z
//   step 从 0 开始 (调用一次 append_next_frame = step++)
class TrajectoryCsvWriter {
public:
    TrajectoryCsvWriter();
    ~TrajectoryCsvWriter();
    int open(const std::string& path);
    int append_frame(int32_t step_idx, int32_t N, const float* x, const float* y, const float* z);
    void close();

private:
    struct Impl;
    Impl* m;
};

// -------- 性能日志 --------
// 追加一行到 perf.log (格式自描述, 每条运行追加 1 行)
void write_perf_log(const std::string& path, const std::string& gpu_info_line, int N,
                    int64_t num_steps, int record_interval, Integrator integrator,
                    KernelMode kernel_mode, const PerfResult& perf);

#endif // NBODY_FILE_IO_H

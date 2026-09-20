// trajectory_io.h —— 粒子初值解析 + 轨迹文件写出
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "nbody_types.h"
#include "sim_params.h"

namespace nbody {

// 解析粒子初值文件：每行 `x y z vx vy vz mass`，允许 # 与 // 注释和空行。
HostBodies ParseBodies(const std::string &path);

// ---------------------------------------------------------------------------
// 轨迹缓冲：格式转置陷阱
//
//   需求（要求 4）规定文件是**粒子主序**：
//       int P, int R, 然后 for p in [0,P): for r in [0,R): (x, y, z)
//
//   但 GPU 天然产出**记录主序**：每隔 record_interval 步，把当前全体粒子
//   的位置作为一个快照拷回主机，即 for r: for p: (x,y,z)。
//
//   所以必须在写文件时做一次转置。这一步写反了动画会完全错乱但不报任何错
//   （文件大小、P、R 全都对得上），因此 scripts/validate.sh 段 1 专门用
//   字节级交叉判定守这一步（两轮写出：单帧 vs 多帧）。
//
//   内存开销：按记录主序累积，缓冲区大小 = P * R * 3 * 4 bytes。
//   N=65536, R=100 → 79 MB；N=262144, R=100 → 315 MB。内存无压力，
//   但 record_interval=1 时 N=65536 会产生 786 MB 文件，故有下限校验。
// ---------------------------------------------------------------------------
class TrajectoryRecorder {
public:
  TrajectoryRecorder(int num_particles, int num_records);

  // 追加一个快照。data 按 xyz 交错排列，长度 3*num_particles。
  void AppendSnapshot(const float *xyz_interleaved);

  int recorded() const { return recorded_; }
  int num_particles() const { return num_particles_; }
  int num_records() const { return num_records_; }

  // 写出为需求指定的粒子主序二进制格式。内部完成记录主序 → 粒子主序转置。
  void WriteBinary(const std::string &path) const;

  // 写出为 CSV：`particle_id,step,x,y,z`。
  // 需要 record_interval 才能把记录索引还原成真实步号。
  void WriteCsv(const std::string &path, int record_interval) const;

  // 预估文件大小（字节），用于启动时提前告警。
  static std::uint64_t EstimateBinarySize(int num_particles, int num_records);

private:
  int num_particles_;
  int num_records_;
  int recorded_ = 0;
  // 记录主序存储：[record][particle][xyz]
  std::vector<float> buffer_;
};

// 读回二进制轨迹（供测试与 C++ 侧校验使用），返回粒子主序的扁平数组。
std::vector<float> ReadBinaryTrajectory(const std::string &path, int *out_p,
                                        int *out_r);

} // namespace nbody

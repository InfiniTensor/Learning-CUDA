// trajectory_io.cpp —— 初值解析与轨迹写出（含记录主序 → 粒子主序转置）
#include "trajectory_io.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace nbody {
namespace {

void StripComment(std::string *s) {
  const std::size_t hash = s->find('#');
  if (hash != std::string::npos)
    s->erase(hash);
  const std::size_t slashes = s->find("//");
  if (slashes != std::string::npos)
    s->erase(slashes);
}

} // namespace

HostBodies ParseBodies(const std::string &path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("cannot open bodies file: " + path);

  HostBodies b;
  std::string line;
  int lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    StripComment(&line);
    std::istringstream ss(line);
    double x, y, z, vx, vy, vz, m;
    if (!(ss >> x >> y >> z >> vx >> vy >> vz >> m)) {
      // 空行或纯注释行直接跳过；有内容但字段不足则是真错误。
      std::string leftover;
      ss.clear();
      ss.str(line);
      if (ss >> leftover) {
        throw std::runtime_error("bodies line " + std::to_string(lineno) +
                                 ": expected 7 numbers `x y z vx vy vz mass`");
      }
      continue;
    }
    if (m <= 0.0) {
      throw std::runtime_error("bodies line " + std::to_string(lineno) +
                               ": mass must be positive");
    }
    b.pos_x.push_back(static_cast<real>(x));
    b.pos_y.push_back(static_cast<real>(y));
    b.pos_z.push_back(static_cast<real>(z));
    b.vel_x.push_back(static_cast<real>(vx));
    b.vel_y.push_back(static_cast<real>(vy));
    b.vel_z.push_back(static_cast<real>(vz));
    b.mass.push_back(static_cast<real>(m));
  }
  if (b.size() == 0)
    throw std::runtime_error("bodies file has no particles: " + path);
  return b;
}

TrajectoryRecorder::TrajectoryRecorder(int num_particles, int num_records)
    : num_particles_(num_particles), num_records_(num_records) {
  if (num_particles <= 0 || num_records <= 0) {
    throw std::runtime_error("TrajectoryRecorder: invalid dimensions");
  }
  buffer_.resize(static_cast<std::size_t>(num_particles) * num_records * 3);
}

void TrajectoryRecorder::AppendSnapshot(const float *xyz_interleaved) {
  if (recorded_ >= num_records_) {
    throw std::runtime_error(
        "TrajectoryRecorder: more snapshots than reserved");
  }
  const std::size_t offset =
      static_cast<std::size_t>(recorded_) * num_particles_ * 3;
  std::copy(xyz_interleaved,
            xyz_interleaved + static_cast<std::size_t>(num_particles_) * 3,
            buffer_.begin() + offset);
  ++recorded_;
}

std::uint64_t TrajectoryRecorder::EstimateBinarySize(int num_particles,
                                                     int num_records) {
  return 2ull * sizeof(std::int32_t) +
         static_cast<std::uint64_t>(num_particles) * num_records * 3ull *
             sizeof(float);
}

void TrajectoryRecorder::WriteBinary(const std::string &path) const {
  std::ofstream out(path, std::ios::binary);
  if (!out)
    throw std::runtime_error("cannot open output file: " + path);

  const std::int32_t p = num_particles_;
  const std::int32_t r = recorded_;
  out.write(reinterpret_cast<const char *>(&p), sizeof(p));
  out.write(reinterpret_cast<const char *>(&r), sizeof(r));

  // ------------------------------------------------------------------
  // 转置：buffer_ 是记录主序 [record][particle][xyz]，
  // 需求要求文件是粒子主序 [particle][record][xyz]。
  // 这里按粒子逐个收集其全部记录点，攒够一个粒子写一次，
  // 避免每个 float 都调一次 write（那样会慢一个数量级）。
  //
  // 写反了文件大小、P、R 全部对得上，程序不报任何错，只有动画会错乱 ——
  // 所以 scripts/validate.sh 段 1 专门守这一步：先跑 num_steps=1 +
  // 极大 record_interval 得到只有 1 帧的文件（此时两种展平等价，是初值
  // 的权威字节布局），再跑多帧版本，断言后者"每个粒子的第 0 帧"拼起来
  // 与前者载荷逐字节相等。
  // ------------------------------------------------------------------
  std::vector<float> per_particle(static_cast<std::size_t>(recorded_) * 3);
  for (int ip = 0; ip < num_particles_; ++ip) {
    for (int ir = 0; ir < recorded_; ++ir) {
      const std::size_t src =
          (static_cast<std::size_t>(ir) * num_particles_ + ip) * 3;
      const std::size_t dst = static_cast<std::size_t>(ir) * 3;
      per_particle[dst + 0] = buffer_[src + 0];
      per_particle[dst + 1] = buffer_[src + 1];
      per_particle[dst + 2] = buffer_[src + 2];
    }
    out.write(
        reinterpret_cast<const char *>(per_particle.data()),
        static_cast<std::streamsize>(per_particle.size() * sizeof(float)));
  }
  if (!out)
    throw std::runtime_error("failed while writing: " + path);
}

void TrajectoryRecorder::WriteCsv(const std::string &path,
                                  int record_interval) const {
  std::FILE *f = std::fopen(path.c_str(), "wb");
  if (!f)
    throw std::runtime_error("cannot open output file: " + path);
  std::fprintf(f, "particle_id,step,x,y,z\n");
  // 与二进制一致，按粒子主序输出，便于两种格式互相对照。
  for (int ip = 0; ip < num_particles_; ++ip) {
    for (int ir = 0; ir < recorded_; ++ir) {
      const std::size_t src =
          (static_cast<std::size_t>(ir) * num_particles_ + ip) * 3;
      std::fprintf(f, "%d,%d,%.7g,%.7g,%.7g\n", ip, ir * record_interval,
                   static_cast<double>(buffer_[src + 0]),
                   static_cast<double>(buffer_[src + 1]),
                   static_cast<double>(buffer_[src + 2]));
    }
  }
  std::fclose(f);
}

std::vector<float> ReadBinaryTrajectory(const std::string &path, int *out_p,
                                        int *out_r) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("cannot open trajectory: " + path);
  std::int32_t p = 0, r = 0;
  in.read(reinterpret_cast<char *>(&p), sizeof(p));
  in.read(reinterpret_cast<char *>(&r), sizeof(r));
  if (!in || p <= 0 || r <= 0) {
    throw std::runtime_error("trajectory header is malformed: " + path);
  }
  std::vector<float> data(static_cast<std::size_t>(p) * r * 3);
  in.read(reinterpret_cast<char *>(data.data()),
          static_cast<std::streamsize>(data.size() * sizeof(float)));
  if (!in)
    throw std::runtime_error("trajectory payload is truncated: " + path);
  *out_p = p;
  *out_r = r;
  return data;
}

} // namespace nbody

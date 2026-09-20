// test_io_roundtrip.cpp —— 轨迹格式转置的单元测试
//
// 为什么这个测试必须存在：
//   需求（要求 4）规定文件是粒子主序 [particle][record][xyz]，
//   而 GPU 天然产出记录主序 [record][particle][xyz]。写文件时要转置一次。
//   转置写反的话，文件大小、P、R 全都对得上，程序不报任何错，
//   只有动画会完全错乱——这是最难发现的一类 bug。
//   所以用"每个值都编码了它的 (particle, record, axis) 身份"的构造数据，
//   直接验证字节布局，并显式验证测试本身有区分能力。
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "trajectory_io.h"

namespace {

int g_failures = 0;

void Check(bool ok, const std::string &what) {
  std::printf("%-58s %s\n", what.c_str(), ok ? "PASS" : "FAIL");
  if (!ok)
    ++g_failures;
}

// 构造可自证身份的值：particle p、record r、轴 axis 的坐标编码为
//   p * 1000 + r * 10 + axis
// 这样任何布局错误都会立刻表现为读出的值对不上。
float Encode(int p, int r, int axis) {
  return static_cast<float>(p * 1000 + r * 10 + axis);
}

} // namespace

int main() {
  using namespace nbody;

  const int kP = 7; // 刻意用非 2 的幂，避免掩盖索引计算错误
  const int kR = 5;

  TrajectoryRecorder rec(kP, kR);

  // 按记录主序逐个快照喂进去——这正是主循环的做法。
  std::vector<float> snapshot(static_cast<std::size_t>(kP) * 3);
  for (int r = 0; r < kR; ++r) {
    for (int p = 0; p < kP; ++p) {
      snapshot[3 * p + 0] = Encode(p, r, 0);
      snapshot[3 * p + 1] = Encode(p, r, 1);
      snapshot[3 * p + 2] = Encode(p, r, 2);
    }
    rec.AppendSnapshot(snapshot.data());
  }
  Check(rec.recorded() == kR, "recorded snapshot count matches");

  const std::string path = "test_traj_roundtrip.bin";
  rec.WriteBinary(path);

  int got_p = 0, got_r = 0;
  const std::vector<float> data = ReadBinaryTrajectory(path, &got_p, &got_r);
  Check(got_p == kP, "header P matches");
  Check(got_r == kR, "header R matches");
  Check(data.size() == static_cast<std::size_t>(kP) * kR * 3,
        "payload length matches P*R*3");

  // 核心断言：文件必须是粒子主序。
  // 粒子 p 的全部记录点连续排列在 [p*R*3, (p+1)*R*3)。
  bool layout_ok = true;
  for (int p = 0; p < kP && layout_ok; ++p) {
    for (int r = 0; r < kR && layout_ok; ++r) {
      for (int axis = 0; axis < 3; ++axis) {
        const std::size_t idx =
            (static_cast<std::size_t>(p) * kR + r) * 3 + axis;
        if (data[idx] != Encode(p, r, axis)) {
          std::printf(
              "  layout mismatch at p=%d r=%d axis=%d: got %.0f want %.0f\n", p,
              r, axis, data[idx], Encode(p, r, axis));
          layout_ok = false;
          break;
        }
      }
    }
  }
  Check(layout_ok, "binary layout is particle-major (P,R,3)");

  // 反向验证：如果文件是记录主序，上面的断言应该失败。
  // 这里显式检查"记录主序读法"读出的是错的值，
  // 确认测试本身有区分能力（否则一个恒真的测试毫无价值）。
  bool record_major_reading_is_wrong = false;
  for (int r = 0; r < kR && !record_major_reading_is_wrong; ++r) {
    for (int p = 0; p < kP; ++p) {
      const std::size_t idx = (static_cast<std::size_t>(r) * kP + p) * 3;
      if (data[idx] != Encode(p, r, 0)) {
        record_major_reading_is_wrong = true;
        break;
      }
    }
  }
  Check(record_major_reading_is_wrong,
        "record-major reading yields wrong values (test discriminates)");

  // CSV 也要能写出来（内容格式在 visualize.py 侧验证）。
  rec.WriteCsv("test_traj_roundtrip.csv", 100);
  std::FILE *f = std::fopen("test_traj_roundtrip.csv", "rb");
  Check(f != nullptr, "csv file was created");
  if (f)
    std::fclose(f);

  std::remove(path.c_str());
  std::remove("test_traj_roundtrip.csv");

  std::printf("\n%s\n",
              g_failures == 0 ? "all checks passed" : "SOME CHECKS FAILED");
  return g_failures == 0 ? 0 : 1;
}

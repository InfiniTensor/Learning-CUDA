// sim_params.h —— 模拟参数文件解析
#pragma once

#include <string>

#include "nbody_types.h"

namespace nbody {

enum class Integrator { kEuler, kLeapfrog };

// 轨迹输出格式。二进制是需求指定格式（要求 4），CSV 仅作对照，
// 用来在报告里量化"CSV 效率较低"这个结论。
enum class TrajectoryFormat { kBinary, kCsv };

struct SimParams {
  // 显式 real(...) 构造：FP32 编译下 double 字面量会触发 MSVC C4305 截断警告。
  real dt = real(1e-3);
  int num_steps = 1000;
  int record_interval = 100;
  real G = real(1.0);
  real softening = real(1e-4);
  Integrator integrator = Integrator::kLeapfrog;
  TrajectoryFormat format = TrajectoryFormat::kBinary;

  // 记录点数：第 0 步（初始状态）也记录，之后每 record_interval 步记一次。
  int record_count() const { return num_steps / record_interval + 1; }
};

// 解析形如 `key = value  # comment` 的参数文件。
// 容忍空行、# 与 // 注释、等号两侧任意空白、以及 value 上的引号。
// 未出现的键保留默认值。遇到无法识别的键会抛异常而非静默忽略——
// 静默忽略拼错的键会导致"我明明设了 dt 却没生效"这类难查的问题。
SimParams ParseSimParams(const std::string &path);

const char *IntegratorName(Integrator integ);

} // namespace nbody

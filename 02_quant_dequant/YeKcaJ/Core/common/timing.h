#pragma once

// 主机墙钟毫秒计时；不包含 CUDA event 或文件读写。
#include <chrono>

namespace pipeline {

using Clock = std::chrono::steady_clock;

// 主机墙钟耗时（毫秒），与 CUDA event 测得的设备序列时间区分。
inline double elapsed(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

}  // namespace pipeline

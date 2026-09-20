// main.cu —— 模拟主程序
//
// 用法：
//   nbody_sim --bodies data/two_body.txt --params data/params_two_body.txt \
//             [--out traj.bin] [--block 128] [--cpu-check] [--diag-interval
//             100]
//
// 输出：
//   - 轨迹文件（二进制粒子主序，或 CSV）
//   - 性能日志（stdout + 可选 --log results.json）
//
// 本版力计算固定使用 V3（shared memory tiling + float4 + rsqrtf）。
// 完整的五档优化阶梯（V0..V4）与逐档扫参数据见开发版 n_version5 与
// n_bodyversion.md §3.1；本文件只保留一个 kernel 的调用点，因此没有
// --kernel / --coarsen 参数。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "cpu_reference.h"
#include "cuda_utils.cuh"
#include "diagnostics.h"
#include "diagnostics_gpu.cuh"
#include "kernels.cuh"
#include "nbody_types.h"
#include "sim_params.h"
#include "snapshot_stream.cuh"
#include "trajectory_io.h"

namespace {

struct Options {
  std::string bodies_path;
  std::string params_path;
  std::string out_path = "trajectory.bin";
  std::string log_path;
  int block_size = 128; // V3 在 4090D 上的最优值（扫参见 n_bodyversion.md §3.1）
  int diag_interval = 0;  // 0 = 只在首尾各算一次
  bool cpu_check = false; // 与 CPU 逐元素交叉验证加速度
  bool cpu_bench = false; // 跑 CPU 基线以计算加速比
  int device = 0;
  // 轨迹快照走 pinned 双缓冲 + 独立流，与后续计算重叠。
  // 默认开启（正确性已验证，无额外成本）；--sync-copy 回到同步 pageable
  // 路径，用于 A/B 对比。
  //
  // 说明：实测这条管线在当前负载下**收益为零**，因为拷贝量 O(N) 而计算量
  // O(N^2)，record_interval=1（最激进）时拷贝占比也仅 0.74%。
  // 保留它是因为"为什么没用"的量化分析本身是结论（见 n_bodyversion.md §1.4），
  // 且代码正确、无额外成本。
  bool async_copy = true;
  // 诊断量默认走 GPU。--cpu-diag 回到主机侧实现：它仍是参考答案，不是退路。
  bool cpu_diag = false;
  // 在初始状态上把 GPU 与 CPU 两条诊断路径逐项比一遍。
  // 两者数学相同、求和顺序不同，差异只应在浮点重排级别。
  bool check_diag = false;
};

void PrintUsage() {
  std::printf(
      "usage: nbody_sim --bodies <file> --params <file> [options]\n"
      "  --out <file>          trajectory output path (default "
      "trajectory.bin)\n"
      "  --log <file>          write performance log as JSON\n"
      "  --block <int>         CUDA block size (default 128)\n"
      "  --diag-interval <int> compute energy/momentum every N steps (O(N^2), "
      "keep large)\n"
      "  --cpu-check           cross-check GPU acceleration against CPU\n"
      "  --cpu-bench           run CPU baselines to report speedup\n"
      "  --sync-copy           record trajectories with a blocking pageable\n"
      "                        memcpy (for A/B comparison against the async "
      "path)\n"
      "  --cpu-diag            compute diagnostics on the host (O(N^2) on "
      "CPU,\n"
      "                        very slow at large N; the GPU path is the "
      "default)\n"
      "  --check-diag          cross-check GPU diagnostics against the CPU\n"
      "                        reference implementation\n"
      "  --device <int>        CUDA device index\n");
}

Options ParseArgs(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char *what) -> std::string {
      if (i + 1 >= argc)
        throw std::runtime_error(std::string(what) + " needs a value");
      return argv[++i];
    };
    if (a == "--bodies") {
      o.bodies_path = next("--bodies");
    } else if (a == "--params") {
      o.params_path = next("--params");
    } else if (a == "--out") {
      o.out_path = next("--out");
    } else if (a == "--log") {
      o.log_path = next("--log");
    } else if (a == "--block") {
      o.block_size = std::stoi(next("--block"));
    } else if (a == "--diag-interval") {
      o.diag_interval = std::stoi(next("--diag-interval"));
    } else if (a == "--cpu-check") {
      o.cpu_check = true;
    } else if (a == "--cpu-bench") {
      o.cpu_bench = true;
    } else if (a == "--sync-copy") {
      o.async_copy = false;
    } else if (a == "--cpu-diag") {
      o.cpu_diag = true;
    } else if (a == "--check-diag") {
      o.check_diag = true;
    } else if (a == "--device") {
      o.device = std::stoi(next("--device"));
    } else if (a == "-h" || a == "--help") {
      PrintUsage();
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + a);
    }
  }
  if (o.bodies_path.empty() || o.params_path.empty()) {
    PrintUsage();
    throw std::runtime_error("--bodies and --params are required");
  }
  return o;
}

// ---------------------------------------------------------------------------
// 数值容差。集中定义而非散落在各判据处：它们会被打印、被写进 JSON、
// 又被用来决定退出码，三处若各写一个字面量，改了一处忘了另一处就会出现
// "打印 PASS 但退出码非零"这种自相矛盾的结果。
//
// kGpuVsCpuTol = 2e-3（GPU 对 CPU）
//   FP32 有 24 位有效位（~6e-8 相对精度），但内层是 N 次累加，误差按
//   sqrt(N) 随机游走放大（N=65536 时约 256 倍，即 ~1.5e-5）；再叠加
//   GPU 与 CPU 在 FMA 融合与求和顺序上的差异，以及 rsqrtf 的 ~2^-22
//   近似误差（经 N 次累加同样放大 sqrt(N) 倍 → ~2e-4）。
//   2e-3 留了一个数量级余量，宽松但仍能抓住真实 bug
//   （写错的 kernel 差值是 O(1)，不是 O(1e-5)）。
// ---------------------------------------------------------------------------
constexpr double kGpuVsCpuTol = 2e-3;

// kDiagGpuVsCpuTol = 1e-10（GPU 诊断量 对 主机侧诊断量）
//   这一档**必须严得多**，理由与上面不同：两边内层都用 double 累加，
//   数学表达式也完全相同，唯一的差别是求和顺序（GPU 分块 + 树形归约，
//   CPU 顺序累加）。double 有 53 位尾数（~1.1e-16 相对精度），
//   N=65536 时 2e9 次累加按 sqrt 随机游走放大约 4.5e4 倍 → ~5e-12。
//   取 1e-10 留两个数量级余量。
//
//   为什么不能用 2e-3 那种宽容差：这个比对的目的是抓归约写错、
//   自交互项漏跳过、原子累加竞争。这些 bug 的典型量级恰好可能落在
//   1e-3 以下（例如漏跳一个自交互项，在 N=65536 时相对误差约 1/N ~ 1.5e-5），
//   宽容差会把它们全放过去。**门禁的容差必须比它要抓的 bug 更小。**
constexpr double kDiagGpuVsCpuTol = 1e-10;

// GPU 与 CPU 的加速度逐元素比对。返回最大相对误差。
//
// 这是本版最重要的正确性门禁：力计算 kernel 只有一份实现，
// 没有"阶梯各档互相比对"可用了，所以参照物必须是**独立的 CPU 实现**
// （cpu_reference.cpp 的 OpenMP 档，与 GPU 侧无共享代码）。
// 两者数学表达式相同、累加顺序也相同（都是 j=0..N-1），
// 唯一差异是 rsqrtf 的近似与 FMA 融合 —— 实测在 N=2 时位级重合，
// 在 N=4096 时误差在 1e-6 量级。
double CrossCheckAcc(const nbody::HostBodies &h, const nbody::DeviceBodies &d,
                     nbody::real G, nbody::real softening) {
  const int n = static_cast<int>(h.size());
  std::vector<nbody::real> gx(n), gy(n), gz(n);
  const size_t bytes = static_cast<size_t>(n) * sizeof(nbody::real);
  CUDA_CHECK(cudaMemcpy(gx.data(), d.acc_x, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gy.data(), d.acc_y, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gz.data(), d.acc_z, bytes, cudaMemcpyDeviceToHost));

  std::vector<nbody::real> cx(n), cy(n), cz(n);
  nbody::ComputeAccCpu(h, G, softening, cx.data(), cy.data(), cz.data(),
                       nbody::CpuVariant::kOpenMP);

  double worst = 0.0;
  for (int i = 0; i < n; ++i) {
    const double dx = double(gx[i]) - cx[i];
    const double dy = double(gy[i]) - cy[i];
    const double dz = double(gz[i]) - cz[i];
    const double err = std::sqrt(dx * dx + dy * dy + dz * dz);
    const double mag = std::sqrt(double(cx[i]) * cx[i] + double(cy[i]) * cy[i] +
                                 double(cz[i]) * cz[i]);
    // 相对误差；分母加一个下限避免加速度接近零时放大噪声
    const double rel = err / (mag + 1e-30);
    if (rel > worst)
      worst = rel;
  }
  return worst;
}

} // namespace

int main(int argc, char **argv) {
  using namespace nbody;
  try {
    const Options opt = ParseArgs(argc, argv);
    CUDA_CHECK(cudaSetDevice(opt.device));
    const DeviceInfo dev = QueryDevice(opt.device);

    HostBodies host = ParseBodies(opt.bodies_path);
    SimParams params = ParseSimParams(opt.params_path);
    const int n = static_cast<int>(host.size());

    // softening=0 会让自交互项产生 0/0 = NaN（kernel 里刻意不做分支跳过），
    // 这里挡住而不是让它悄悄污染全部结果。
    if (params.softening <= real(0)) {
      throw std::runtime_error("softening must be > 0 (the kernel relies on it "
                               "to skip self-interaction "
                               "without branching)");
    }
    if (opt.block_size <= 0 || opt.block_size > dev.max_threads_per_block) {
      throw std::runtime_error("--block out of range for this device");
    }
    // 力计算 kernel 的 tile 放在动态 shared memory 里，容量随 block size
    // 线性增长。超了的话 kernel launch 会失败（cudaErrorInvalidValue），
    // 而那个错误信息完全看不出是 block 开太大导致的 —— 在这里挡住并说清楚。
    {
      const size_t need = SharedBytesPerBlock(opt.block_size);
      if (need > static_cast<size_t>(dev.shared_mem_per_block)) {
        throw std::runtime_error(std::string("--block ") +
                                 std::to_string(opt.block_size) + " needs " +
                                 std::to_string(need) +
                                 " B of shared memory for the force kernel, "
                                 "but this device allows only " +
                                 std::to_string(dev.shared_mem_per_block) +
                                 " B per block; use a smaller --block");
      }
    }

    const int num_records = params.record_count();
    const std::uint64_t est_bytes =
        TrajectoryRecorder::EstimateBinarySize(n, num_records);

    std::printf("=== N-body simulation (%s, %s) ===\n", NBODY_PRECISION_NAME,
                kKernelName);
    PrintDeviceInfo(dev);
    std::printf("particles     : %d\n", n);
    std::printf("steps         : %d  dt=%g  integrator=%s\n", params.num_steps,
                double(params.dt), IntegratorName(params.integrator));
    std::printf("G / softening : %g / %g\n", double(params.G),
                double(params.softening));
    std::printf("record        : every %d steps -> %d records\n",
                params.record_interval, num_records);
    std::printf("est. traj size: %.1f MiB\n", est_bytes / 1024.0 / 1024.0);
    if (est_bytes > (1ull << 30)) {
      std::printf("  [warn] trajectory exceeds 1 GiB; consider a larger "
                  "record_interval\n");
    }

    DeviceBodies dbod{};
    AllocateDevice(&dbod, n);
    UploadBodies(&dbod, host);

    size_t mem_used = 0, mem_total = 0;
    QueryMemoryUsage(&mem_used, &mem_total);

    TrajectoryRecorder recorder(n, num_records);
    // 主机侧的快照缓冲不在这里分配：传输管线自己持有 pinned 双缓冲，
    // 数据经回调直接交给 recorder（见下方构造处）。

    // ------------------------------------------------------------------
    // 诊断量：默认走 GPU，--cpu-diag 回到主机侧实现。
    //
    // 主机侧实现**保留而非删除**，它是 GPU 版的参考答案：两者数学表达式相同
    // 但求和顺序不同（GPU 分块 + 树形归约），结果只应差在浮点重排级别。
    // --check-diag 会在初始状态上把两边逐项比一遍。
    //
    // scratch 仍然需要：CPU 路径要用，而 n==2 的轨道要素也在主机侧算。
    // ------------------------------------------------------------------
    DiagnosticsGpuWorkspace diag_ws{};
    if (!opt.cpu_diag || opt.check_diag)
      AllocateDiagnosticsGpu(&diag_ws);

    HostBodies scratch;
    // 取当前状态的诊断量。GPU 路径不需要把状态拷回主机。
    auto compute_diag = [&]() -> Diagnostics {
      if (opt.cpu_diag) {
        DownloadBodies(&scratch, dbod);
        return ComputeDiagnostics(scratch, params.G, params.softening);
      }
      return ComputeDiagnosticsGpu(dbod, params.G, params.softening, &diag_ws,
                                   opt.block_size);
    };

    auto record_diag = [&](int step, std::vector<double> *energies,
                           std::vector<double> *momenta,
                           std::vector<double> *ang_momenta) {
      const Diagnostics diag = compute_diag();
      energies->push_back(diag.total_energy);
      momenta->push_back(diag.momentum_magnitude);
      // 角动量与能量、动量一起采：诊断 kernel 本来就算好了这一项
      // （见 diagnostics.h 的 angular_momentum_magnitude），
      // 多存一列不增加任何 GPU 工作量，却补上了要求里点名的守恒量之一。
      ang_momenta->push_back(diag.angular_momentum_magnitude);
      // 位力比 2T/|W|：平衡态应为 1。
      //
      // 为什么值得多印这一列（kinetic 与 potential 本来就算好了，代价为零）：
      // 能量守恒只说明"积分器没把能量弄丢"，它对一个正在剧烈演化的系统同样成立
      // ——两球对撞过程中 |dE/E0| 可以是 1e-8，而系统离平衡差得远。
      // 位力比反映的是**动力学状态**：对撞开始时轨道动能让 2T/|W| 偏离 1，
      // 并合完成、系统重新弛豫后应当回到 1 附近。
      // 这就把"星团并合"从一段好看的动画变成了一个有判据的动力学验证。
      const double virial = std::abs(diag.potential) > 0.0
                                ? 2.0 * diag.kinetic / std::abs(diag.potential)
                                : 0.0;
      std::printf("  step %7d  E=%.10e  |p|=%.3e  |L|=%.6e  max|a|=%.3e  "
                  "2T/|W|=%.4f\n",
                  step, diag.total_energy, diag.momentum_magnitude,
                  diag.angular_momentum_magnitude, diag.max_acc_magnitude,
                  virial);
    };

    // ------------------------------------------------------------------
    // 诊断量的 GPU vs CPU 交叉验证（--check-diag）。
    //
    // 这是一个几乎免费但很强的门禁。GPU 版与主机版数学表达式相同，
    // 只有求和顺序不同（GPU 是分块 + 树形归约 + 原子累加），
    // 所以每一项都应该只差在 double 的浮点重排级别（~1e-14 相对）。
    //
    // 它能抓住的具体错误，都是不崩溃、不报错的那类：
    //   - 势能忘了折半（kernel 按 j != i 全求和，每对数了两次）→ 差 2 倍；
    //   - 自交互没跳过 → 势能多一个 -G*m_i^2/eps 的巨大伪项；
    //   - 块内归约的 __syncthreads() 漏了 → 结果随 block 数随机波动；
    //   - max|a| 的位模式 atomicMax 写错 → 只有那一项不对。
    // 每一种都会让某一项跳到 O(1) 相对误差，而其余项仍然正确 ——
    // 所以要逐项比而不是只比总能量。
    //
    // 容差 1e-10：double 的机器精度是 2.2e-16，N=65536 时约 2e9 次累加，
    // 随机游走放大 sqrt(2e9) ≈ 4.5e4 倍 → ~1e-11。取 1e-10 留一个数量级。
    // ------------------------------------------------------------------
    int diag_check_failures = 0;
    if (opt.check_diag) {
      std::printf("\n--- diagnostics gpu-vs-cpu (same state) ---\n");
      DownloadBodies(&scratch, dbod);
      const Diagnostics c =
          ComputeDiagnostics(scratch, params.G, params.softening);
      const Diagnostics g = ComputeDiagnosticsGpu(
          dbod, params.G, params.softening, &diag_ws, opt.block_size);
      struct Item {
        const char *name;
        double cpu, gpu;
      };
      const Item items[] = {
          {"kinetic", c.kinetic, g.kinetic},
          {"potential", c.potential, g.potential},
          {"energy", c.total_energy, g.total_energy},
          {"|p|", c.momentum_magnitude, g.momentum_magnitude},
          {"|L|", c.angular_momentum_magnitude, g.angular_momentum_magnitude},
          {"max|a|", c.max_acc_magnitude, g.max_acc_magnitude},
      };
      for (const Item &it : items) {
        const double denom = std::abs(it.cpu) > 1e-300 ? std::abs(it.cpu) : 1.0;
        const double rel = std::abs(it.gpu - it.cpu) / denom;
        const bool ok = rel < kDiagGpuVsCpuTol;
        if (!ok)
          ++diag_check_failures;
        std::printf("  %-10s cpu=%+.12e  gpu=%+.12e  rel=%.2e  %s\n", it.name,
                    it.cpu, it.gpu, rel, ok ? "PASS" : "FAIL");
      }
    }

    std::vector<double> energies, momenta, ang_momenta;
    std::printf("\n--- diagnostics ---\n");
    record_diag(0, &energies, &momenta, &ang_momenta);

    // ------------------------------------------------------------------
    // 主循环
    //
    // Leapfrog KDK 的力评估复用：初始算一次 a(t0)，之后每步末尾算出的
    // a(t+dt) 正是下一步开头需要的 a(t)，所以每步只有一次 O(N^2) 力评估。
    // 若每步开头重新算一次，会白白多一倍最贵的工作。
    // ------------------------------------------------------------------
    const real half_dt = params.dt * real(0.5);
    if (params.integrator == Integrator::kLeapfrog) {
      LaunchComputeAcc(dbod, params.G, params.softening, opt.block_size);
    }

    double cross_check_err = -1.0;
    if (opt.cpu_check) {
      // 在初始状态上做一次比对（此时 GPU 与 CPU 的输入完全相同）。
      if (params.integrator != Integrator::kLeapfrog) {
        LaunchComputeAcc(dbod, params.G, params.softening, opt.block_size);
      }
      CUDA_CHECK(cudaDeviceSynchronize());
      cross_check_err = CrossCheckAcc(host, dbod, params.G, params.softening);
    }

    // ------------------------------------------------------------------
    // 轨迹传输管线。
    //
    // sink 直接把主机缓冲交给 recorder。recorder 内部按记录主序累积，
    // 写文件时才转置成需求要求的粒子主序——这一点没有变。
    // 变的只是"数据怎么从设备到达主机"：async 模式下是
    // pinned 双缓冲 + 独立非阻塞流，与后续步的计算重叠。
    // ------------------------------------------------------------------
    SnapshotStreamer streamer(n, opt.async_copy, [&recorder](const float *xyz) {
      recorder.AppendSnapshot(xyz);
    });

    // 记录初始状态。要求 4 的 R 包含 t=0 这个记录点。
    streamer.Submit(dbod);

    CudaTimer timer;
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.Start();

    for (int step = 1; step <= params.num_steps; ++step) {
      if (params.integrator == Integrator::kLeapfrog) {
        LaunchKickDrift(dbod, params.dt, opt.block_size);
        LaunchComputeAcc(dbod, params.G, params.softening, opt.block_size);
        LaunchKick(dbod, half_dt, opt.block_size);
      } else {
        LaunchComputeAcc(dbod, params.G, params.softening, opt.block_size);
        LaunchEulerUpdate(dbod, params.dt, opt.block_size);
      }

      if (step % params.record_interval == 0) {
        streamer.Submit(dbod);
      }
      if (opt.diag_interval > 0 && step % opt.diag_interval == 0 &&
          step != params.num_steps) {
        record_diag(step, &energies, &momenta, &ang_momenta);
      }
    }

    // 收割仍在飞行中的快照。**必须在 timer.Stop() 之前**：
    // 轨迹数据在收割完成前是不完整的，把这段等待排除在计时之外
    // 等于让 async 模式白拿一段没算钱的工作，A/B 对比就不诚实了。
    // 同步模式下 pending_ 恒为空，Drain 是空操作，两侧口径一致。
    streamer.Drain();

    timer.Stop();
    const double total_ms = timer.ElapsedMs();
    CUDA_CHECK(cudaDeviceSynchronize());

    record_diag(params.num_steps, &energies, &momenta, &ang_momenta);

    // ------------------------------------------------------------------
    // 性能指标（输出定义 2 要求的全部项）
    // ------------------------------------------------------------------
    const double ms_per_step = total_ms / params.num_steps;
    const double particle_steps_per_sec =
        double(n) * params.num_steps / (total_ms / 1000.0);
    const double interactions = double(n) * double(n) * params.num_steps;
    const double gflops =
        interactions * kFlopsPerInteraction / (total_ms / 1000.0) / 1e9;

    std::printf("\n--- performance ---\n");
    std::printf("total time    : %.3f ms\n", total_ms);
    std::printf("per step      : %.4f ms\n", ms_per_step);
    std::printf("particle-steps: %.4g /s\n", particle_steps_per_sec);
    std::printf("throughput    : %.2f GFLOP/s (at %.0f flop/interaction)\n",
                gflops, kFlopsPerInteraction);
    std::printf("device memory : %.1f MiB used / %.1f MiB total\n",
                mem_used / 1024.0 / 1024.0, mem_total / 1024.0 / 1024.0);
    if (n < 8192) {
      // 小 N 上 kernel launch 开销（5-10 us/launch，每步 2-3 次 launch）
      // 会主导，测出的不是真实算力。这个提示避免把小 N 的低效误读成实现问题。
      std::printf(
          "  [note] at N=%d each step is launch-bound; scale to N>=32768 for "
          "meaningful throughput\n",
          n);
    }

    const double e0 = energies.front();
    const double e1 = energies.back();
    const double rel_energy_err =
        std::abs(e0) > 0.0 ? std::abs((e1 - e0) / e0) : std::abs(e1 - e0);
    std::printf("\n--- accuracy ---\n");
    std::printf("energy  E0=%.10e  E1=%.10e  |dE/E0|=%.3e\n", e0, e1,
                rel_energy_err);
    std::printf("momentum |p|: %.3e -> %.3e\n", momenta.front(),
                momenta.back());
    // 角动量是比能量更挑剔的判据：能量误差里混着辛积分器的有界振荡，
    // 而 |L| 的漂移只来自浮点舍入，以及"每个目标粒子独立求和"导致的
    // 力反对称性缺失（就是下面那条 note）。有心力下 L 严格守恒，
    // 所以这一列的漂移直接反映数值误差，没有物理成因可以背锅。
    const double l0 = ang_momenta.front();
    const double l1 = ang_momenta.back();
    const double rel_angular_err =
        std::abs(l0) > 0.0 ? std::abs((l1 - l0) / l0) : std::abs(l1 - l0);
    std::printf("angular |L|: %.6e -> %.6e  rel=%.3e\n", l0, l1,
                rel_angular_err);
    // 这一点对力计算路径成立：每个目标粒子独立地把 N 个源粒子的贡献加起来，
    // 没有利用 F_ij = -F_ji 把一对相互作用只算一次。
    std::printf(
        "  [note] the force kernel sums forces per-particle without using "
        "F_ij=-F_ji,\n"
        "         so total momentum is conserved only to rounding, not "
        "exactly.\n");
    if (cross_check_err >= 0.0) {
      std::printf(
          "gpu-vs-cpu acc: worst relative error %.3e (tol %.1e) -> %s\n",
          cross_check_err, kGpuVsCpuTol,
          cross_check_err < kGpuVsCpuTol ? "PASS" : "FAIL");
    }

    if (n == 2) {
      // 必须显式把末状态拷回主机，不能依赖 scratch。
      //
      // 这里踩过一个真实的回归：早期版本诊断量走主机侧实现，每次 record_diag
      // 都会 DownloadBodies 到 scratch，于是这一行"顺便"拿到了末状态。
      // 把诊断量默认搬到 GPU 之后，scratch 再也不会被填充，
      // ComputeTwoBodyElements 收到一个 size==0 的容器就直接返回全零默认值
      // ——二体验证打印出 a=0 e=0，而**程序退出码仍是 0**，validate.sh
      // 照样报 PASS。
      //
      // 教训：依赖"别的功能顺带留下的副作用"是脆的。需要末状态就明确要一次。
      // 代价是 N=2 时一次 7 个数组的小拷贝，可忽略。
      DownloadBodies(&scratch, dbod);
      const OrbitalElements el = ComputeTwoBodyElements(scratch, params.G);
      std::printf("two-body: a=%.6f e=%.6f separation=%.6f period=%.6f\n",
                  el.semi_major_axis, el.eccentricity, el.separation,
                  el.period);
    }

    // ------------------------------------------------------------------
    // CPU 基线与加速比
    //
    // 计时对象是**一次完整力评估**（O(N^2)），而非跑满 num_steps。
    // 理由：N=262144 时单线程朴素版一次力评估约 10 分钟，跑 1000 步不可能。
    // 力评估占每步耗时的 99% 以上（其余是 O(N) 的 elementwise 更新），
    // 所以「每步耗时 ≈ 一次力评估耗时」，加速比用这个比也是公平的。
    // 大 N 时进一步只测一部分目标粒子再线性外推，见 BenchmarkCpuForceEval。
    // ------------------------------------------------------------------
    double cpu_ms[3] = {-1, -1, -1};
    int cpu_measured[3] = {0, 0, 0};
    // GPU 侧的同一口径：一次力评估的平均耗时。
    // leapfrog 每步恰好一次力评估，Euler 也是一次，所以直接用 ms_per_step
    // 会略微高估 GPU 的力评估耗时（含了 elementwise kernel 与打包 kernel），
    // 这个偏差对 GPU 不利，属于保守估计，可以接受。
    const double gpu_force_ms = ms_per_step;

    if (opt.cpu_bench) {
      std::printf("\n--- cpu baselines (%d OpenMP threads) ---\n",
                  OpenMPMaxThreads());
      const CpuVariant variants[3] = {CpuVariant::kNaive, CpuVariant::kScalarO3,
                                      CpuVariant::kOpenMP};
      // 目标交互数：约 2e8 次。单线程朴素版约 2 秒，OpenMP 版约 0.1 秒，
      // 既够长到压过计时噪声，又不至于让基准跑几分钟。
      const double kTargetInteractions = 2e8;
      for (int k = 0; k < 3; ++k) {
        const double ms =
            BenchmarkCpuForceEval(host, params.G, params.softening, variants[k],
                                  kTargetInteractions, &cpu_measured[k]);
        cpu_ms[k] = ms;
        std::printf("%-14s: %10.3f ms per force eval (measured %d/%d targets)  "
                    "gpu speedup = %8.1fx\n",
                    CpuVariantName(variants[k]), ms, cpu_measured[k], n,
                    ms / gpu_force_ms);
      }
      std::printf("  [note] report the speedup against cpu_openmp; the naive "
                  "single-thread number is a strawman.\n");
      std::printf(
          "  [note] cpu timings are extrapolated from a subset of target "
          "particles (work per target is identical and independent).\n");
    }

    // ------------------------------------------------------------------
    // 写轨迹
    // ------------------------------------------------------------------
    if (params.format == TrajectoryFormat::kBinary) {
      recorder.WriteBinary(opt.out_path);
      std::printf("\nwrote %s (particle-major binary, P=%d R=%d)\n",
                  opt.out_path.c_str(), recorder.num_particles(),
                  recorder.recorded());
    } else {
      recorder.WriteCsv(opt.out_path, params.record_interval);
      std::printf(
          "\nwrote %s (csv; ~%.1fx larger than binary and much slower to "
          "parse)\n",
          opt.out_path.c_str(), 6.0);
    }

    if (!opt.log_path.empty()) {
      std::FILE *f = std::fopen(opt.log_path.c_str(), "wb");
      if (!f)
        throw std::runtime_error("cannot open log file: " + opt.log_path);
      std::fprintf(f,
                   "{\n"
                   "  \"kernel\": \"%s\",\n"
                   "  \"precision\": \"%s\",\n"
                   "  \"device\": \"%s\",\n"
                   "  \"sm\": \"%d.%d\",\n"
                   "  \"n\": %d,\n"
                   "  \"steps\": %d,\n"
                   "  \"block_size\": %d,\n"
                   "  \"smem_bytes_per_block\": %zu,\n"
                   "  \"integrator\": \"%s\",\n"
                   "  \"total_ms\": %.6f,\n"
                   "  \"ms_per_step\": %.6f,\n"
                   "  \"particle_steps_per_sec\": %.6e,\n"
                   "  \"gflops\": %.4f,\n"
                   "  \"device_mem_used_mib\": %.2f,\n"
                   "  \"energy_initial\": %.12e,\n"
                   "  \"energy_final\": %.12e,\n"
                   "  \"rel_energy_error\": %.6e,\n"
                   "  \"momentum_initial\": %.6e,\n"
                   "  \"momentum_final\": %.6e,\n"
                   "  \"angular_momentum_initial\": %.6e,\n"
                   "  \"angular_momentum_final\": %.6e,\n"
                   "  \"rel_angular_momentum_error\": %.6e,\n"
                   "  \"gpu_vs_cpu_worst_rel_err\": %.6e,\n"
                   "  \"cpu_force_eval_ms\": {\n"
                   "    \"naive\": %.4f,\n"
                   "    \"scalar_o3\": %.4f,\n"
                   "    \"openmp\": %.4f\n"
                   "  },\n"
                   "  \"cpu_measured_targets\": [%d, %d, %d],\n"
                   "  \"speedup_vs_cpu_openmp\": %.3f,\n"
                   "  \"openmp_threads\": %d\n"
                   "}\n",
                   kKernelName, NBODY_PRECISION_NAME, dev.name.c_str(),
                   dev.major, dev.minor, n, params.num_steps, opt.block_size,
                   SharedBytesPerBlock(opt.block_size),
                   IntegratorName(params.integrator), total_ms, ms_per_step,
                   particle_steps_per_sec, gflops, mem_used / 1024.0 / 1024.0,
                   e0, e1, rel_energy_err, momenta.front(), momenta.back(), l0,
                   l1, rel_angular_err, cross_check_err, cpu_ms[0], cpu_ms[1],
                   cpu_ms[2], cpu_measured[0], cpu_measured[1], cpu_measured[2],
                   cpu_ms[2] > 0.0 ? cpu_ms[2] / gpu_force_ms : -1.0,
                   OpenMPMaxThreads());
      std::fclose(f);
      std::printf("wrote %s\n", opt.log_path.c_str());
    }

    FreeDiagnosticsGpu(&diag_ws);
    FreeDevice(&dbod);

    // ------------------------------------------------------------------
    // 退出码：把"数值不对"变成脚本能看见的失败。
    //
    // GPU-vs-CPU 比对若只打印 FAIL 而仍然 return 0，validate.sh 里的
    // `if run; then report PASS` 就会在 kernel 写错的情况下报 PASS ——
    // 门禁形同虚设。所以超差必须非零退出。
    // ------------------------------------------------------------------
    int failures = diag_check_failures;
    if (cross_check_err >= 0.0 && cross_check_err >= kGpuVsCpuTol)
      ++failures;
    if (failures > 0) {
      std::fprintf(stderr, "error: %d numerical check(s) failed\n", failures);
      return 2;
    }
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}

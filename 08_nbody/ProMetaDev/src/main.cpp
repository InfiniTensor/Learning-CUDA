// ============================================================
// main.cpp —— N体引力模拟项目入口
//  CLI: nbody [particles.txt] [params.txt] [trajectory.bin] [perf.log]
//            [--dt N] [--steps N] [--record N] [--G N] [--softening N]
//            [--integrator euler|leapfrog] [--kernel simple|tiling]
//            [--csv] [--no-cpu] [--check-energy] [--analyze N]
//            [--info]
// ============================================================
#include "nbody_types.h"
#include "nbody_kernels.h"
#include "file_io.h"
#include "analysis.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iomanip>
#include <cmath>

using clk = std::chrono::high_resolution_clock;
static double elapsed_ms(clk::time_point t0, clk::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static void print_usage(const char* argv0) {
    std::cout << "Usage: " << argv0 << " [particles.txt] [params.txt] [trajectory.bin] [perf.log]\n"
              << "         [--dt N] [--steps N] [--record N] [--G N] [--softening N]\n"
              << "         [--integrator euler|leapfrog] [--kernel simple|tiling]\n"
              << "         [--csv] [--no-cpu] [--check-energy] [--analyze N]\n"
              << "         [--info]\n";
}

int main(int argc, char** argv) {
    // ---- 默认路径 ----
    std::string pcl_path = "data/particles_4096.txt";
    std::string prm_path = "data/params_default.txt";
    std::string traj_path = "outputs/trajectory.bin";
    std::string perf_path = "outputs/perf.log";

    SimParams sp;
    bool info_only = false;
    bool flag_csv = false;
    bool flag_no_cpu = false;
    bool flag_energy = false;
    int flag_analyze = 0; // 0 = 使用 params 中的 analyze_frames
    // CLI 明确指定的数值用 sentinel 保留，以便 parse_params_file 之后再次覆盖（CLI > params.txt）。
    bool sp_over_dt = false, sp_over_G = false, sp_over_eps = false;
    bool sp_over_steps = false, sp_over_rec = false, sp_over_block = false;
    double cli_dt = 0, cli_G = 0, cli_eps = 0;
    int64_t cli_steps = 0, cli_rec = 0;
    int cli_block = 0;
    Integrator cli_intg = Integrator::LEAPFROG;
    bool sp_over_intg = false;
    KernelMode cli_knl = KernelMode::SIMPLE;
    bool sp_over_knl = false;

    // 第一遍: 位置参数
    int positional = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--csv")
            continue;
        if (a == "--no-cpu")
            continue;
        if (a == "--check-energy")
            continue;
        if (a == "--info") {
            continue;
        }
        if ((a == "--dt" || a == "--steps" || a == "--record" || a == "--G" || a == "--softening" ||
             a == "--integrator" || a == "--kernel" || a == "--analyze") &&
            i + 1 < argc) {
            ++i;
            continue;
        }
        switch (positional) {
        case 0:
            pcl_path = a;
            break;
        case 1:
            prm_path = a;
            break;
        case 2:
            traj_path = a;
            break;
        case 3:
            perf_path = a;
            break;
        }
        ++positional;
    }

    // 第二遍: 覆盖
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](std::string& err) -> std::string {
            if (i + 1 >= argc) {
                err = "missing value for " + a;
                return "";
            }
            return std::string(argv[++i]);
        };
        std::string err;
        if (a == "--csv") {
            flag_csv = true;
        } else if (a == "--no-cpu") {
            flag_no_cpu = true;
        } else if (a == "--check-energy") {
            flag_energy = true;
        } else if (a == "--info") {
            info_only = true;
        } else if (a == "--dt") {
            std::string v = next(err);
            if (!err.empty()) {
                std::cerr << "[err] " << err << "\n";
                return 1;
            }
            cli_dt = std::stod(v);
            sp_over_dt = true;
        } else if (a == "--steps") {
            std::string v = next(err);
            if (!err.empty()) {
                std::cerr << "[err] " << err << "\n";
                return 1;
            }
            cli_steps = std::stoll(v);
            sp_over_steps = true;
        } else if (a == "--record") {
            std::string v = next(err);
            if (!err.empty()) {
                std::cerr << "[err] " << err << "\n";
                return 1;
            }
            cli_rec = std::stoll(v);
            sp_over_rec = true;
        } else if (a == "--G") {
            std::string v = next(err);
            if (!err.empty()) {
                std::cerr << "[err] " << err << "\n";
                return 1;
            }
            cli_G = std::stod(v);
            sp_over_G = true;
        } else if (a == "--softening") {
            std::string v = next(err);
            if (!err.empty()) {
                std::cerr << "[err] " << err << "\n";
                return 1;
            }
            cli_eps = std::stod(v);
            sp_over_eps = true;
        } else if (a == "--integrator") {
            std::string v = next(err);
            if (!err.empty()) {
                std::cerr << "[err] " << err << "\n";
                return 1;
            }
            cli_intg = (v == "euler") ? Integrator::EULER : Integrator::LEAPFROG;
            sp_over_intg = true;
        } else if (a == "--kernel") {
            std::string v = next(err);
            if (!err.empty()) {
                std::cerr << "[err] " << err << "\n";
                return 1;
            }
            cli_knl = (v == "tiling" || v == "shared_tiling") ? KernelMode::SHARED_TILING
                                                              : KernelMode::SIMPLE;
            sp_over_knl = true;
        } else if (a == "--block-size") {
            std::string v = next(err);
            if (!err.empty()) {
                std::cerr << "[err] " << err << "\n";
                return 1;
            }
            cli_block = std::stoi(v);
            sp_over_block = true;
        } else if (a == "--analyze") {
            std::string v = next(err);
            if (!err.empty()) {
                std::cerr << "[err] " << err << "\n";
                return 1;
            }
            flag_analyze = std::stoi(v);
        }
    }

    std::string gpu_info = nbody_gpu_info_string();
    if (info_only) {
        std::cout << "==== GPU Info ====\n" << gpu_info << "\n";
        return 0;
    }

    std::cout << "==== CUDA N-Body Gravity Simulation (2026-summer) ====\n" << gpu_info << "\n";

    ParticleSet host;
    if (parse_particles_file(pcl_path, host) != 0) {
        std::cerr << "[ERR] cannot parse particles: " << pcl_path << "\n";
        return 1;
    }
    // ----- 数值一致性: 质心 (CM) 与总动量归零 -----
    // 防止因 particles.txt 生成脚本版本差异或 10 位小数截断导致整体漂移
    if (host.N > 0 && host.mass && host.x && host.vx) {
        double Mtot = 0.0;
        double cx = 0.0, cy = 0.0, cz = 0.0;
        double px = 0.0, py = 0.0, pz = 0.0;
        for (int64_t i = 0; i < host.N; ++i) {
            double m = (double)host.mass[i];
            if (m <= 0.0)
                continue;
            Mtot += m;
            cx += m * (double)host.x[i];
            cy += m * (double)host.y[i];
            cz += m * (double)host.z[i];
            px += m * (double)host.vx[i];
            py += m * (double)host.vy[i];
            pz += m * (double)host.vz[i];
        }
        if (Mtot > 0.0) {
            cx /= Mtot;
            cy /= Mtot;
            cz /= Mtot;
            double vx_cm = px / Mtot, vy_cm = py / Mtot, vz_cm = pz / Mtot;
            double p_prev = std::sqrt(px * px + py * py + pz * pz);
            for (int64_t i = 0; i < host.N; ++i) {
                host.x[i] = (float)((double)host.x[i] - cx);
                host.y[i] = (float)((double)host.y[i] - cy);
                host.z[i] = (float)((double)host.z[i] - cz);
                host.vx[i] = (float)((double)host.vx[i] - vx_cm);
                host.vy[i] = (float)((double)host.vy[i] - vy_cm);
                host.vz[i] = (float)((double)host.vz[i] - vz_cm);
            }
            std::cout << std::fixed << std::setprecision(3) << "[Sim] CM+P zeroed: Mtot=" << Mtot
                      << "  |P|_before=" << p_prev << "  CM_before=(" << cx << "," << cy << ","
                      << cz << ")\n";
        }
    }
    // params 解析失败允许用默认 (如文件不存在)
    (void)parse_params_file(prm_path, sp);

    // CLI 的 --no-cpu / --check-energy / --csv 覆盖 params 文件值
    if (flag_no_cpu)
        sp.skip_cpu = true;
    if (flag_energy)
        sp.check_energy = true;
    if (flag_csv)
        sp.write_csv = true;
    if (flag_analyze > 0)
        sp.analyze_frames = flag_analyze;
    // CLI 显式给出的参数，优先级高于 params.txt (parse_params_file 之后强制覆盖)
    if (sp_over_dt)
        sp.dt = cli_dt;
    if (sp_over_steps)
        sp.num_steps = cli_steps;
    if (sp_over_rec)
        sp.record_interval = cli_rec;
    if (sp_over_G)
        sp.G = cli_G;
    if (sp_over_eps)
        sp.softening = cli_eps;
    if (sp_over_intg)
        sp.integrator = cli_intg;
    if (sp_over_knl)
        sp.kernel_mode = cli_knl;
    if (sp_over_block)
        sp.block_size = cli_block;

    std::cout << "[Sim] N=" << host.N << " steps=" << sp.num_steps << " dt=" << sp.dt
              << " record_int=" << sp.record_interval << " G=" << sp.G << " eps=" << sp.softening
              << " intg=" << (sp.integrator == Integrator::EULER ? "euler" : "leapfrog")
              << " knl=" << (sp.kernel_mode == KernelMode::SIMPLE ? "simple" : "tiling")
              << " skip_cpu=" << sp.skip_cpu << " check_E=" << sp.check_energy
              << " csv=" << sp.write_csv << "\n";

    if (host.N <= 0) {
        std::cerr << "[ERR] N<=0\n";
        return 2;
    }

    // ---- record frames R ----
    int64_t total_steps = sp.num_steps;
    int64_t rec_int = std::max<int64_t>(1, sp.record_interval);
    int64_t R_frames = total_steps / rec_int + 1;
    if (sp.analyze_frames > 0 && sp.analyze_frames < (int)R_frames) {
        R_frames = sp.analyze_frames;
        total_steps = (R_frames - 1) * rec_int;
        std::cout << "[Sim] analyze_frames=" << sp.analyze_frames << " → clamp steps to "
                  << total_steps << "\n";
    }
    std::cout << "[Sim] R_frames=" << R_frames << " effective_steps=" << total_steps << "\n";

    // ---- CPU 参考计时 (可选, 用于加速比) ----
    double cpu_time_ms = -1.0;
    if (!sp.skip_cpu && host.N > 0) {
        // 为大 N 安全: 仅跑若干"有效"步进行基准, 按比例外推, 避免几小时
        int64_t cpu_bench_steps = std::min<int64_t>(total_steps, host.N <= 2048 ? 5 : 2);
        if (cpu_bench_steps < 1)
            cpu_bench_steps = 1;
        std::cout << "[CPU] benchmark with " << cpu_bench_steps << " steps (N=" << host.N
                  << ") ... ";
        // 拷贝一份用于 CPU 基准
        ParticleSet hcpu;
        hcpu.N = host.N;
        hcpu.on_device = false;
        auto clone = [&](float*& dst, const float* src) {
            dst = new float[host.N];
            std::memcpy(dst, src, sizeof(float) * host.N);
        };
        clone(hcpu.x, host.x);
        clone(hcpu.y, host.y);
        clone(hcpu.z, host.z);
        clone(hcpu.vx, host.vx);
        clone(hcpu.vy, host.vy);
        clone(hcpu.vz, host.vz);
        clone(hcpu.mass, host.mass);
        hcpu.ax = new float[host.N]();
        hcpu.ay = new float[host.N]();
        hcpu.az = new float[host.N]();
        // leapfrog 起步 (保持和 GPU 相同约定)
        if (sp.integrator == Integrator::LEAPFROG) {
            // 用 host 参考: 先算 a0 再踢 v 半步
            std::vector<double> ax(host.N), ay(host.N), az(host.N);
            cpu_accelerations_naive(hcpu, sp.G, sp.softening, ax.data(), ay.data(), az.data());
            float half_dt = (float)(0.5 * sp.dt);
            for (int i = 0; i < host.N; ++i) {
                hcpu.vx[i] += (float)ax[i] * half_dt;
                hcpu.vy[i] += (float)ay[i] * half_dt;
                hcpu.vz[i] += (float)az[i] * half_dt;
            }
        }
        auto t0 = clk::now();
        for (int64_t s = 0; s < cpu_bench_steps; ++s) {
            cpu_step_once_naive(hcpu, sp);
        }
        auto t1 = clk::now();
        double ms_sample = elapsed_ms(t0, t1);
        cpu_time_ms = ms_sample * ((double)total_steps / (double)cpu_bench_steps);
        free_host_particles(hcpu);
        std::cout << std::fixed << std::setprecision(2) << ms_sample << " ms (sample) → ~"
                  << cpu_time_ms << " ms (extrapolated)\n";
    }

    // ---- GPU 分配 + 起步 (先做显存预估越界保护) ----
    {
        std::string vram_msg;
        if (!nbody_gpu_check_memory_fit(host.N, vram_msg)) {
            std::cerr << "[ERR] " << vram_msg << "\n";
            free_host_particles(host);
            return 11;
        }
        std::cout << "[VRAM] " << vram_msg << "\n";
    }
    ParticleSet dev;
    if (nbody_gpu_alloc_and_copy(host, dev) != 0) {
        std::cerr << "[ERR] GPU alloc/copy failed\n";
        free_host_particles(host);
        return 3;
    }
    if (nbody_integrator_init(dev, sp) != 0) {
        std::cerr << "[ERR] integrator init failed\n";
        nbody_gpu_free(dev);
        free_host_particles(host);
        return 4;
    }
    // 同步确保 init kernel 结束再计时 (总时间包含同步开销更真实)
    nbody_cuda_device_sync();

    // ---- 轨迹 writer ----
    TrajectoryBinWriter bw;
    if (bw.open(traj_path, host.N) != 0) {
        std::cerr << "[WARN] cannot open trajectory bin: " << traj_path << " (no write)\n";
    }
    std::string csv_path = traj_path + ".csv";
    TrajectoryCsvWriter cw;
    if (sp.write_csv) {
        if (cw.open(csv_path) != 0) {
            std::cerr << "[WARN] cannot open trajectory CSV: " << csv_path << " (no CSV write)\n";
            sp.write_csv = false;
        }
    }

    // ---- 写第 0 帧 ----
    int frame_idx = 0;
    // GPU → host 当前位置 (第 0 步就是初始, 已经在 host 有, 但保持统一流程)
    if (nbody_gpu_copy_positions_back(dev, host) == 0) {
        bw.append_frame(host.x, host.y, host.z);
        if (sp.write_csv)
            cw.append_frame((int32_t)(frame_idx * rec_int), host.N, host.x, host.y, host.z);
    }
    ++frame_idx;

    // 可选能量检查: E0
    double E0 = 0.0, E_last = 0.0;
    if (sp.check_energy) {
        E0 = calc_total_energy(host, sp.G, sp.softening);
        Vec3 P0 = calc_total_momentum(host);
        std::cout << "[E] step 0:  E=" << std::fixed << std::setprecision(6) << E0
                  << "   |P|=" << std::sqrt(P0.x * P0.x + P0.y * P0.y + P0.z * P0.z) << "\n";
        E_last = E0;
    }

    // ---- 主循环 ----
    auto t_sim0 = clk::now();
    int64_t step = 0;
    int64_t printed_tick = std::max<int64_t>(1, total_steps / 10);
    for (; step < total_steps; ++step) {
        int r = nbody_step_once(dev, sp);
        if (r != 0) {
            std::cerr << "[ERR] nbody_step_once failed, step=" << step << " (cuda err=" << r
                      << ")\n";
            break;
        }
        // record
        if ((step + 1) % rec_int == 0) {
            nbody_cuda_device_sync();
            if (nbody_gpu_copy_positions_back(dev, host) != 0) {
                std::cerr << "[WARN] D2H failed at step " << step + 1 << "\n";
            } else {
                bw.append_frame(host.x, host.y, host.z);
                if (sp.write_csv) {
                    cw.append_frame((int32_t)((step + 1)), host.N, host.x, host.y, host.z);
                }
            }
            if (sp.check_energy) {
                // 注意: Leapfrog host 中 v 是半速, calc_total_energy 用 host.v 会偏离.
                //   近似做法: 这里的能量仍用来观察"长期漂移"趋势.
                double E = calc_total_energy(host, sp.G, sp.softening);
                if (step + 1 == total_steps)
                    E_last = E;
                double rel = (std::abs(E0) > 1e-30) ? std::abs((E - E0) / E0) : std::abs(E - E0);
                if (sp.check_energy) {
                    std::cout << "[E] step " << (step + 1) << ":  E=" << std::fixed
                              << std::setprecision(6) << E << "  |ΔE/E0|=" << std::scientific << rel
                              << "\n";
                }
            }
            ++frame_idx;
        }
        if ((step + 1) % printed_tick == 0) {
            std::cout << "[step] " << (step + 1) << "/" << total_steps << "\n";
        }
    }
    nbody_cuda_device_sync();
    auto t_sim1 = clk::now();
    double total_sim_ms = elapsed_ms(t_sim0, t_sim1);

    // ---- 性能指标 ----
    PerfResult perf{};
    perf.total_sim_ms = total_sim_ms;
    perf.avg_step_ms = (step > 0) ? total_sim_ms / (double)step : 0.0;
    double total_sec = total_sim_ms / 1000.0;
    perf.particle_steps_per_sec = (total_sec > 0) ? (double)host.N * (double)step / total_sec : 0.0;
    // 显存使用量粗估: 10 arrays of N floats → 10*N*4 bytes
    perf.gpu_mem_used_mb = (double)host.N * 10.0 * 4.0 / (1024.0 * 1024.0);
    perf.cpu_time_ms = cpu_time_ms;
    perf.speedup_vs_cpu =
        (cpu_time_ms > 0 && total_sim_ms > 0) ? (cpu_time_ms / total_sim_ms) : 0.0;

    std::cout << "---- Simulation Stats ----\n"
              << std::fixed << std::setprecision(3) << "  total time    : " << perf.total_sim_ms
              << " ms\n"
              << "  avg step time : " << perf.avg_step_ms << " ms\n"
              << "  throughput    : " << perf.particle_steps_per_sec << " particle-steps/s\n"
              << "  gpu mem est   : " << perf.gpu_mem_used_mb << " MB\n";
    if (cpu_time_ms > 0) {
        std::cout << std::fixed << std::setprecision(3) << "  cpu est       : " << cpu_time_ms
                  << " ms\n"
                  << "  speedup       : " << perf.speedup_vs_cpu << " x\n";
    }
    if (sp.check_energy) {
        double drift =
            (std::abs(E0) > 1e-30) ? std::abs((E_last - E0) / E0) : std::abs(E_last - E0);
        std::cout << std::scientific << "  E_drift |ΔE/E0|: " << drift
                  << "  (0: good; <<1% for leapfrog expected)\n";
    }

    // ---- 写 perf.log ----
    write_perf_log(perf_path, gpu_info, host.N, total_steps, (int)rec_int, sp.integrator,
                   sp.kernel_mode, perf);

    // 关闭 writers (会回写 header R)
    bw.close();
    if (sp.write_csv)
        cw.close();

    // 清理
    nbody_gpu_free(dev);
    free_host_particles(host);
    (void)print_usage;
    return 0;
}

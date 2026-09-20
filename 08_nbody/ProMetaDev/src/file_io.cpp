// ============================================================
// file_io.cpp —— 纯 C++17 实现: 粒子/参数解析 + 轨迹(二进制/CSV)/性能日志写
// ============================================================
#include "file_io.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <algorithm>

// ============================================================
// 小工具: 去掉字符串首尾空白 + 注释 (# 开头去掉)
// ============================================================
static inline std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char)s[b]))
        ++b;
    while (e > b && std::isspace((unsigned char)s[e - 1]))
        --e;
    return s.substr(b, e - b);
}
static inline std::string strip_comment(const std::string& s) {
    auto p = s.find('#');
    if (p == std::string::npos)
        return s;
    return s.substr(0, p);
}
static inline std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

// ============================================================
// parse_particles_file
//   格式: 每行 "x y z vx vy vz mass" (空行/注释行跳过)
// ============================================================
int parse_particles_file(const std::string& path, ParticleSet& host_set) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[file_io] cannot open particles file: " << path << "\n";
        return -1;
    }
    // 先一次性读进 vector (行数通常已知, 否则用流式 push_back)
    std::vector<float> xs, ys, zs, vxs, vys, vzs, ms;
    xs.reserve(4096);
    std::string line;
    int lineno = 0;
    while (std::getline(f, line)) {
        ++lineno;
        std::string content = trim(strip_comment(line));
        if (content.empty())
            continue;
        std::istringstream iss(content);
        float x, y, z, vx, vy, vz, m;
        if (!(iss >> x >> y >> z >> vx >> vy >> vz >> m)) {
            std::cerr << "[file_io] particles parse err at line " << lineno << ": [" << content
                      << "]\n";
            return -2;
        }
        xs.push_back(x);
        ys.push_back(y);
        zs.push_back(z);
        vxs.push_back(vx);
        vys.push_back(vy);
        vzs.push_back(vz);
        ms.push_back(m);
    }
    int N = (int)xs.size();
    if (N <= 0) {
        std::cerr << "[file_io] empty particles file: " << path << "\n";
        return -3;
    }
    // 分配 SoA
    auto alloc = [&](float*& p, const std::vector<float>& v) {
        p = new float[N];
        std::memcpy(p, v.data(), sizeof(float) * N);
    };
    host_set.N = N;
    host_set.on_device = false;
    alloc(host_set.x, xs);
    alloc(host_set.y, ys);
    alloc(host_set.z, zs);
    alloc(host_set.vx, vxs);
    alloc(host_set.vy, vys);
    alloc(host_set.vz, vzs);
    alloc(host_set.mass, ms);
    host_set.ax = new float[N]();
    host_set.ay = new float[N]();
    host_set.az = new float[N]();
    return 0;
}

// ============================================================
// parse_params_file —— 键值对, "key = value", 支持 '#' 注释
// ============================================================
static bool parse_bool(const std::string& v, bool& out) {
    std::string s = to_lower(trim(v));
    if (s == "1" || s == "true" || s == "yes" || s == "on") {
        out = true;
        return true;
    }
    if (s == "0" || s == "false" || s == "no" || s == "off") {
        out = false;
        return true;
    }
    return false;
}
int parse_params_file(const std::string& path, SimParams& sp) {
    std::ifstream f(path);
    if (!f.is_open())
        return -1;
    std::string line;
    int lineno = 0, parsed = 0;
    while (std::getline(f, line)) {
        ++lineno;
        std::string s = trim(strip_comment(line));
        if (s.empty())
            continue;
        auto eq = s.find('=');
        if (eq == std::string::npos) {
            std::cerr << "[file_io] params parse skip (no '='): " << s << "\n";
            continue;
        }
        std::string k = to_lower(trim(s.substr(0, eq)));
        std::string v = trim(s.substr(eq + 1));
        if (k == "dt") {
            sp.dt = std::stod(v);
            parsed++;
        } else if (k == "num_steps") {
            sp.num_steps = std::stoll(v);
            parsed++;
        } else if (k == "record_interval") {
            sp.record_interval = std::stoll(v);
            parsed++;
        } else if (k == "g") {
            sp.G = std::stod(v);
            parsed++;
        } else if (k == "softening") {
            sp.softening = std::stod(v);
            parsed++;
        } else if (k == "integrator") {
            std::string iv = to_lower(v);
            if (iv == "euler")
                sp.integrator = Integrator::EULER;
            else
                sp.integrator = Integrator::LEAPFROG;
            parsed++;
        } else if (k == "kernel_mode") {
            std::string kv = to_lower(v);
            if (kv == "shared_tiling" || kv == "tiling")
                sp.kernel_mode = KernelMode::SHARED_TILING;
            else
                sp.kernel_mode = KernelMode::SIMPLE;
            parsed++;
        } else if (k == "block_size") {
            sp.block_size = std::stoi(v);
            parsed++;
        } else if (k == "write_csv") {
            parse_bool(v, sp.write_csv);
            parsed++;
        } else if (k == "skip_cpu") {
            parse_bool(v, sp.skip_cpu);
            parsed++;
        } else if (k == "check_energy") {
            parse_bool(v, sp.check_energy);
            parsed++;
        } else if (k == "analyze_frames") {
            sp.analyze_frames = std::stoi(v);
            parsed++;
        } else {
            std::cerr << "[file_io] params unknown key '" << k << "' at line " << lineno << "\n";
        }
    }
    return (parsed > 0) ? 0 : -2;
}

// ============================================================
// free_host_particles
// ============================================================
void free_host_particles(ParticleSet& h) {
    if (h.on_device)
        return; // 不负责 device
    delete[] h.x;
    delete[] h.y;
    delete[] h.z;
    delete[] h.vx;
    delete[] h.vy;
    delete[] h.vz;
    delete[] h.mass;
    delete[] h.ax;
    delete[] h.ay;
    delete[] h.az;
    h.x = h.y = h.z = nullptr;
    h.vx = h.vy = h.vz = nullptr;
    h.mass = nullptr;
    h.ax = h.ay = h.az = nullptr;
    h.N = 0;
}

// ============================================================
// TrajectoryBinWriter (pimpl)
// ============================================================
struct TrajectoryBinWriter::Impl {
    std::ofstream of;
    int32_t N;
    int64_t R_written;
    std::streamoff header_pos_R; // 保存 R 在文件中的 offset, 以便 close 时回写
};
TrajectoryBinWriter::TrajectoryBinWriter() : m(new Impl()) {}
TrajectoryBinWriter::~TrajectoryBinWriter() {
    close();
    delete m;
}

int TrajectoryBinWriter::open(const std::string& path, int32_t N) {
    m->of.open(path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!m->of.is_open())
        return -1;
    m->N = N;
    m->R_written = 0;
    // 头: N(4 bytes) + R(4 bytes, placeholder)
    int32_t R_placeholder = 0;
    m->of.write((const char*)&N, sizeof(int32_t));
    m->header_pos_R = m->of.tellp();
    m->of.write((const char*)&R_placeholder, sizeof(int32_t));
    if (!m->of.good())
        return -2;
    return 0;
}
int TrajectoryBinWriter::append_frame(const float* x, const float* y, const float* z) {
    if (!m->of.is_open())
        return -1;
    // 按题面: 按粒子顺序写入 xyz 连续 floats
    for (int i = 0; i < m->N; ++i) {
        float t[3] = {x[i], y[i], z[i]};
        m->of.write((const char*)t, sizeof(float) * 3);
    }
    if (!m->of.good())
        return -2;
    ++m->R_written;
    return 0;
}
void TrajectoryBinWriter::close() {
    if (!m->of.is_open())
        return;
    // 回写真实 R
    int32_t R = (int32_t)m->R_written;
    m->of.seekp(m->header_pos_R);
    m->of.write((const char*)&R, sizeof(int32_t));
    m->of.close();
}

// ============================================================
// TrajectoryCsvWriter (pimpl)
// ============================================================
struct TrajectoryCsvWriter::Impl {
    std::ofstream of;
};
TrajectoryCsvWriter::TrajectoryCsvWriter() : m(new Impl()) {}
TrajectoryCsvWriter::~TrajectoryCsvWriter() {
    close();
    delete m;
}
int TrajectoryCsvWriter::open(const std::string& path) {
    m->of.open(path, std::ios::out | std::ios::trunc);
    if (!m->of.is_open())
        return -1;
    m->of << "particle_id,step,x,y,z\n";
    return 0;
}
int TrajectoryCsvWriter::append_frame(int32_t step_idx, int32_t N, const float* x, const float* y,
                                      const float* z) {
    if (!m->of.is_open())
        return -1;
    m->of << std::fixed << std::setprecision(6);
    for (int i = 0; i < N; ++i) {
        m->of << i << "," << step_idx << "," << x[i] << "," << y[i] << "," << z[i] << "\n";
    }
    return m->of.good() ? 0 : -2;
}
void TrajectoryCsvWriter::close() {
    if (m->of.is_open())
        m->of.close();
}

// ============================================================
// write_perf_log
// ============================================================
void write_perf_log(const std::string& path, const std::string& gpu_info_line, int N,
                    int64_t num_steps, int record_interval, Integrator integrator,
                    KernelMode kernel_mode, const PerfResult& perf) {
    std::ofstream of(path, std::ios::app);
    if (!of.is_open())
        return;
    of << "# " << gpu_info_line << "\n";
    of << std::fixed << std::setprecision(3) << "[N=" << N << " steps=" << num_steps
       << " rec=" << record_interval
       << " intg=" << (integrator == Integrator::EULER ? "euler" : "leapfrog")
       << " knl=" << (kernel_mode == KernelMode::SIMPLE ? "simple" : "shared_tiling")
       << "] total_ms=" << perf.total_sim_ms << " avg_step_ms=" << perf.avg_step_ms
       << " psteps/s=" << perf.particle_steps_per_sec << " gpu_mem_MB=" << perf.gpu_mem_used_mb
       << " cpu_ms=" << perf.cpu_time_ms << " speedup=" << perf.speedup_vs_cpu << "\n";
}

// types.h - 项目公共类型定义（POD，host/device 均可包含，禁止 STL）
#pragma once

#include <cstdint>

// 期权类型：欧式看涨/看跌、亚式看涨（算术平均、固定行权价）、障碍看涨（向上敲出）
enum class OptionType : int {
    EuropeanCall = 0,
    EuropeanPut = 1,
    AsianCall = 2,
    BarrierCallUpOut = 3,
};

// 随机数生成器：cuRAND Philox4x32-10 或自定义 xorshift128+
enum class RngType : int {
    Curand = 0,
    Custom = 1,
};

// 方差缩减方法（互斥）：无 / 对偶变量 / 控制变量
enum class VarReduction : int {
    None = 0,
    Antithetic = 1,
    ControlVariate = 2,
};

// 计算引擎：CPU 单线程基线 / GPU / 两者对比
enum class Engine : int {
    Cpu = 0,
    Gpu = 1,
    Both = 2,
};

// 期权参数文件解析结果
struct OptionSpec {
    OptionType type = OptionType::EuropeanCall;
    double spot = 100.0;           // 标的初始价格 S0
    double strike = 100.0;         // 执行价格 K
    double risk_free_rate = 0.03;  // 无风险利率 r
    double volatility = 0.2;       // 波动率 sigma
    double maturity = 1.0;         // 到期时间 T（年）
    double barrier = 130.0;        // 障碍价格 H（仅障碍期权使用）
};

// 模拟参数文件解析结果
struct SimSpec {
    long long num_paths = 10000000;       // 模拟路径数
    int num_steps = 256;                  // 每条路径的时间步数（欧式精确模拟时无效）
    unsigned long long seed = 1234;       // 随机数种子
    RngType rng = RngType::Curand;        // 随机数生成器
    VarReduction vr = VarReduction::None; // 方差缩减方法
};

// 预计算后传给 device 的期权/市场参数（float，与核内路径循环精度一致）
// 欧式精确模拟 = 1 步递推（dt = T），与逐步模拟共用同一条代码路径
struct OptionParams {
    float s0 = 0.0f;          // 初始价格
    float k = 0.0f;           // 执行价格
    float r = 0.0f;           // 无风险利率
    float sigma = 0.0f;       // 波动率
    float T = 0.0f;           // 到期时间（年）
    float barrier = 0.0f;     // 障碍价格（可经 BGK 修正）
    float drift_step = 0.0f;  // (r - sigma^2/2) * dt
    float vol_step = 0.0f;    // sigma * sqrt(dt)
    float disc = 1.0f;        // 折现因子 exp(-r*T)
    float inv_steps = 0.0f;   // 1/num_steps（亚式平均用）
    int num_steps = 1;        // 实际执行的步数（欧式精确模拟 = 1）
};

// 控制变量参数（beta = 0 表示未启用）
struct CVParams {
    double beta = 0.0;   // 控制变量系数（pilot 核估计）
    float c_mean = 0.0f; // 控制变量期望 E[c]（解析值）
};

// 引擎运行选项（命令行参数）
struct EngineOptions {
    int block = 256;              // 线程块大小
    int grid_blocks = 0;          // 网格块数，0 = 自动（按占用率）
    long long cpu_paths = 500000; // CPU 基线路径数上限（10M×256 单线程太慢）
    int cpu_threads = 1;          // CPU 引擎线程数（>1 启用 OpenMP）
    int euro_steps = 1;           // 欧式模拟步数：1 = 精确终值公式，>1 = 逐步验证
    bool bgk = false;             // 障碍期权 BGK 边界修正
    bool quiet = false;           // 抑制非必要输出
    bool dump_rng = false;        // 输出 RNG 流（--dump-rng N）
    int dump_rng_count = 0;
    bool sweep = false;           // N = 1e4..1e7 收敛扫描
};

// 实际执行的每路径步数：欧式用精确终值（1 步），路径依赖期权用 num_steps 逐步模拟
inline int effective_steps(const OptionSpec& os, const SimSpec& ss, const EngineOptions& opts) {
    return (os.type == OptionType::EuropeanCall || os.type == OptionType::EuropeanPut)
               ? opts.euro_steps
               : ss.num_steps;
}

// 计时明细（GPU 用 cudaEvent，CPU 用 std::chrono）
struct Timing {
    double kernel_ms = 0.0;      // 主 mc_kernel 耗时
    double finalize_ms = 0.0;    // finalize_reduce 核耗时
    double reduction_ms = 0.0;   // 主核结束 → 主机拿到 (sum,sumsq) 并算完统计量
    double h2d_ms = 0.0;         // 参数上传
    double d2h_ms = 0.0;         // 结果回传
    double host_setup_ms = 0.0;  // 设备初始化/内存分配/参数准备
    double total_ms = 0.0;       // 端到端总耗时
    // 粒度配置（性能日志用）
    int grid_blocks = 0;
    int block_size = 0;
    int blocks_per_sm = 0;
    int sm_count = 0;
    int total_threads = 0;
    int registers_per_thread = 0;
    int shared_bytes_per_block = 0;
    // 设备信息
    int device_id = 0;
    char device_name[256] = {0};
    int compute_major = 0;
    int compute_minor = 0;
    int driver_version = 0;
    int runtime_version = 0;
};

// 定价结果（含误差/收敛指标）
struct PriceResult {
    double price = 0.0;            // 价格估计（已折现）
    double std_error = 0.0;        // 蒙特卡洛标准误
    double ci95_low = 0.0;         // 95% 置信区间下界
    double ci95_high = 0.0;        // 95% 置信区间上界
    long long samples = 0;         // 独立样本数（对偶时 = 对数）
    double mean_payoff = 0.0;      // 原始样本均值（折现后）
    double var_payoff = 0.0;       // 样本方差
    double reference = 0.0;        // 参考价格（解析解/高精度 MC）
    double abs_error = 0.0;        // |price - reference|
    double rel_error = 0.0;        // 相对误差
    double z_score = 0.0;          // (price - reference) / std_error
    double var_reduction_factor = 1.0; // 方差缩减因子（>= 1）
    double beta = 0.0;             // 控制变量系数（未用时为 0）
    double error_times_sqrt_n = 0.0; // |error| * sqrt(N)，收敛指标（期望近似常数）
    const char* reference_name = "";  // 参考价格来源名称
};

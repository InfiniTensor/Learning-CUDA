// ============================================================
// pricing_kernels.cu - CUDA 蒙特卡洛定价核
//   * Float 精度路径模拟 (P1: __expf / logf / float 累加)
//   * 控制变量法 CV (P2: 几何亚式作为控制)
//   * 对偶变量法 Antithetic (F1: Z 与 -Z 配对, payoff 平均)
//   * curand Philox4_32_10 可复现 RNG
//   * 共享内存两阶段归约
// 支持期权: 欧式 / 亚式(算术平均) / 障碍(up&down, in&out)
// ============================================================
#include "option_params.h"
#include "bs_formula.h"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// -------- __constant__ 内存定义 (头文件中 extern 声明) --------
__constant__ double c_spot;
__constant__ double c_strike;
__constant__ double c_risk_free;
__constant__ double c_volatility;
__constant__ double c_maturity;
__constant__ double c_barrier;
__constant__ int c_option_type;
__constant__ int c_barrier_dir;
__constant__ int c_barrier_kind;
__constant__ int c_num_steps;

// -------- 设备函数: 模拟一条路径 (float 精度) --------
// 输出: S_T, 算术平均, 几何平均, 是否触及障碍
// negate=true 时对所有正态随机数取负 (用于 Antithetic 对偶变量法)
__device__ inline void simulate_path_f(curandStatePhilox4_32_10_t& st, bool negate, float& S_T,
                                       float& arith_avg, float& geom_avg, int& barrier_hit) {
    const float S0 = (float)c_spot;
    const float dt = (float)(c_maturity / (double)c_num_steps);
    const float mu_d = (float)((c_risk_free - 0.5 * c_volatility * c_volatility) * (double)dt);
    const float sd = (float)(c_volatility * sqrt((double)dt));
    const float barr = (float)c_barrier;
    const int otype = c_option_type;
    const bool is_barrier =
        (otype == (int)OptionType::BARRIER_CALL || otype == (int)OptionType::BARRIER_PUT);

    float S = S0;
    float sum_S = 0.0f;
    float sum_log_S = 0.0f;
    int hit = 0;

    int step = 0;
    while (step < c_num_steps) {
        // Philox4_32_10 每次产生 4 个正态
        float4 n = curand_normal4(&st);
        if (negate) {
            n.x = -n.x;
            n.y = -n.y;
            n.z = -n.z;
            n.w = -n.w;
        }
#pragma unroll
        for (int k = 0; k < 4 && step < c_num_steps; ++k, ++step) {
            float z = (k == 0) ? n.x : (k == 1) ? n.y : (k == 2) ? n.z : n.w;
            // GBM 递推: S *= exp(mu*dt + sigma*sqrt(dt)*z)
            S = S * __expf(mu_d + sd * z);
            sum_S += S;
            sum_log_S += logf(S);
            if (is_barrier) {
                if (c_barrier_dir == 1 && S >= barr)
                    hit = 1;
                if (c_barrier_dir == -1 && S <= barr)
                    hit = 1;
            }
        }
    }
    S_T = S;
    arith_avg = sum_S / (float)c_num_steps;
    geom_avg = __expf(sum_log_S / (float)c_num_steps);
    barrier_hit = hit;
}

// -------- 设备函数: 计算 payoff (目标 X 与 控制 Y) --------
// X = 目标期权 payoff (折现前)
// Y = 几何亚式 payoff (作为 CV 控制, 仅亚式期权有效)
// 注意: 不使用 host 的 fmax, 用 device 内联 max0
__device__ inline double dmax0(double v) {
    return v > 0.0 ? v : 0.0;
}

__device__ inline void compute_payoffs(float S_T, float arith_avg, float& geom_avg, int barrier_hit,
                                       double& X, double& Y) {
    const double K = c_strike;
    const int otype = c_option_type;
    double Xv = 0.0, Yv = 0.0;

    switch (otype) {
    case (int)OptionType::EUROPEAN_CALL:
        Xv = dmax0((double)S_T - K);
        Yv = 0.0;
        break;
    case (int)OptionType::EUROPEAN_PUT:
        Xv = dmax0((double)K - S_T);
        Yv = 0.0;
        break;
    case (int)OptionType::ASIAN_CALL:
        Xv = dmax0((double)arith_avg - K);
        Yv = dmax0((double)geom_avg - K); // 控制变量
        break;
    case (int)OptionType::ASIAN_PUT:
        Xv = dmax0((double)K - arith_avg);
        Yv = dmax0((double)K - geom_avg);
        break;
    case (int)OptionType::BARRIER_CALL:
    case (int)OptionType::BARRIER_PUT: {
        double ep = (otype == (int)OptionType::BARRIER_CALL) ? dmax0((double)S_T - K)
                                                             : dmax0((double)K - S_T);
        // KNOCK_IN(0): 触及才有 payoff; KNOCK_OUT(1): 触及则 payoff=0
        if (c_barrier_kind == 0)
            Xv = (barrier_hit ? ep : 0.0);
        else
            Xv = (barrier_hit ? 0.0 : ep);
        Yv = 0.0;
        break;
    }
    default:
        Xv = 0.0;
        Yv = 0.0;
        break;
    }
    X = Xv;
    Y = Yv;
}

// -------- 主核函数 --------
// vr_mode: 0=NONE, 1=ANTITHETIC, 2=CONTROL_VARIATE (CV 在 host 端处理, kernel 端同 NONE)
// 每个 block 输出 5 个和: sum(X), sum(X^2), sum(Y), sum(Y^2), sum(XY)
__global__ void mc_pricing_kernel(unsigned long long seed, int num_paths, int vr_mode,
                                  double* block_sums) // [num_blocks][5]
{
    extern __shared__ double sm[];
    double* sx = sm;
    double* sx2 = sm + blockDim.x;
    double* sy = sm + 2 * blockDim.x;
    double* sy2 = sm + 3 * blockDim.x;
    double* sxy = sm + 4 * blockDim.x;

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int total = gridDim.x * blockDim.x;

    // 本地累加 (double 累加保证精度)
    double lx = 0, lx2 = 0, ly = 0, ly2 = 0, lxy = 0;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, (unsigned long long)gid, 0, &state);

    for (int p = gid; p < num_paths; p += total) {
        float S_T, arith_avg, geom_avg;
        int barrier_hit;

        // 路径 1: 正常 Z
        curandStatePhilox4_32_10_t saved = state; // 保存 state 供对偶路径复用
        simulate_path_f(state, false, S_T, arith_avg, geom_avg, barrier_hit);
        double X1, Y1;
        compute_payoffs(S_T, arith_avg, geom_avg, barrier_hit, X1, Y1);

        double X, Y;
        if (vr_mode == 1) {
            // Antithetic: 路径 2 用 -Z (恢复 state → 生成相同随机数但取负)
            state = saved;
            simulate_path_f(state, true, S_T, arith_avg, geom_avg, barrier_hit);
            double X2, Y2;
            compute_payoffs(S_T, arith_avg, geom_avg, barrier_hit, X2, Y2);
            X = 0.5 * (X1 + X2);
            Y = 0.5 * (Y1 + Y2);
        } else {
            X = X1;
            Y = Y1;
        }

        lx += X;
        lx2 += X * X;
        ly += Y;
        ly2 += Y * Y;
        lxy += X * Y;
    }

    // 写入 shared
    sx[tid] = lx;
    sx2[tid] = lx2;
    sy[tid] = ly;
    sy2[tid] = ly2;
    sxy[tid] = lxy;
    __syncthreads();

    // 树形归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sx[tid] += sx[tid + s];
            sx2[tid] += sx2[tid + s];
            sy[tid] += sy[tid + s];
            sy2[tid] += sy2[tid + s];
            sxy[tid] += sxy[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int bi = blockIdx.x;
        block_sums[bi * 5 + 0] = sx[0];
        block_sums[bi * 5 + 1] = sx2[0];
        block_sums[bi * 5 + 2] = sy[0];
        block_sums[bi * 5 + 3] = sy2[0];
        block_sums[bi * 5 + 4] = sxy[0];
    }
}

// ============================================================
// 主机端: 启动核 + 归约 + 计算结果
// ============================================================
#include <vector>
#include <chrono>
#include <sstream>

// -------- GPU 信息字符串 (P5: 可观测性) --------
std::string gpu_info_string() {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    int rt = 0, drv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    std::ostringstream ss;
    ss << "GPU=" << prop.name << "  SMs=" << prop.multiProcessorCount << "  CC=" << prop.major
       << "." << prop.minor << "  VRAM=" << (prop.totalGlobalMem >> 20) << "MB"
       << "  CUDA_RT=" << (rt / 1000) << "." << ((rt / 10) % 100) << "  DRV=" << (drv / 1000) << "."
       << ((drv / 10) % 100);
    return ss.str();
}

PricingResult run_monte_carlo(const OptionParams& o, const SimParams& s, double& occupancy_pct) {
    PricingResult r;
    r.option_type_str = OptionParams::to_string(o.type);

    // ---- 拷贝参数到 __constant__ ----
    cudaMemcpyToSymbol(c_spot, &o.spot, sizeof(double));
    cudaMemcpyToSymbol(c_strike, &o.strike, sizeof(double));
    cudaMemcpyToSymbol(c_risk_free, &o.risk_free, sizeof(double));
    cudaMemcpyToSymbol(c_volatility, &o.volatility, sizeof(double));
    cudaMemcpyToSymbol(c_maturity, &o.maturity, sizeof(double));
    cudaMemcpyToSymbol(c_barrier, &o.barrier, sizeof(double));
    int ot = (int)o.type, bd = (int)o.barrier_dir, bt = (int)o.barrier_t, ns = s.num_steps;
    cudaMemcpyToSymbol(c_option_type, &ot, sizeof(int));
    cudaMemcpyToSymbol(c_barrier_dir, &bd, sizeof(int));
    cudaMemcpyToSymbol(c_barrier_kind, &bt, sizeof(int));
    cudaMemcpyToSymbol(c_num_steps, &ns, sizeof(int));

    // ---- 配置 grid/block ----
    int block = s.block_size;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // 决定 grid 大小: 取足够多的 block 覆盖 GPU
    int num_sm = prop.multiProcessorCount;
    int max_warps_per_sm = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    int block_warps = block / prop.warpSize;
    int blocks_per_sm = (block_warps > 0) ? (max_warps_per_sm / block_warps) : 1;
    int grid = num_sm * blocks_per_sm * 2;
    int max_blocks = (s.num_paths + block - 1) / block;
    if (grid > max_blocks)
        grid = max_blocks;
    if (grid < 1)
        grid = 1;

    // ---- 占用率查询 ----
    int max_active_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, mc_pricing_kernel, block,
                                                  5 * block * sizeof(double));
    double occupancy =
        (prop.maxThreadsPerMultiProcessor > 0)
            ? (double)max_active_blocks * block / (double)(prop.maxThreadsPerMultiProcessor)
            : 0.0;
    occupancy_pct = occupancy * 100.0;

    // ---- 分配 block_sums ----
    std::vector<double> h_sums(grid * 5, 0.0);
    double* d_sums = nullptr;
    cudaMalloc(&d_sums, sizeof(double) * grid * 5);
    cudaMemset(d_sums, 0, sizeof(double) * grid * 5);

    size_t smem_bytes = 5 * block * sizeof(double);
    int vr_mode = (int)s.variance_reduction;

    // ---- 计时 ----
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0, 0);

    mc_pricing_kernel<<<grid, block, smem_bytes>>>((unsigned long long)s.seed, (int)s.num_paths,
                                                   vr_mode, d_sums);
    cudaEventRecord(t1, 0);
    cudaEventSynchronize(t1);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    r.gpu_time_ms = (double)ms;
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    // ---- 拷回 + 归约 ----
    cudaMemcpy(h_sums.data(), d_sums, sizeof(double) * grid * 5, cudaMemcpyDeviceToHost);
    cudaFree(d_sums);

    double sx = 0, sx2 = 0, sy = 0, sy2 = 0, sxy = 0;
    for (int i = 0; i < grid; ++i) {
        sx += h_sums[i * 5 + 0];
        sx2 += h_sums[i * 5 + 1];
        sy += h_sums[i * 5 + 2];
        sy2 += h_sums[i * 5 + 3];
        sxy += h_sums[i * 5 + 4];
    }
    int64_t N = s.num_paths;
    double mean_x = sx / N;
    double mean_x2 = sx2 / N;
    double mean_y = sy / N;
    double mean_y2 = sy2 / N;
    double mean_xy = sxy / N;

    double var_x = mean_x2 - mean_x * mean_x;
    double var_y = mean_y2 - mean_y * mean_y;
    double cov_xy = mean_xy - mean_x * mean_y;
    if (var_x < 0)
        var_x = 0;
    if (var_y < 0)
        var_y = 0;

    // 折现因子
    double disc = std::exp(-o.risk_free * o.maturity);

    // ---- 计算价格与 SE ----
    double price = 0.0, se = 0.0;
    bool is_asian = (o.type == OptionType::ASIAN_CALL || o.type == OptionType::ASIAN_PUT);
    bool is_european = (o.type == OptionType::EUROPEAN_CALL || o.type == OptionType::EUROPEAN_PUT);

    if (s.variance_reduction == VarianceReduction::CONTROL_VARIATE && is_asian && var_y > 0) {
        // 控制变量法
        double beta = cov_xy / var_y;
        double E_Y = bs_geometric_asian_price(o); // 已是折现后价格
        double E_Y_payoff = E_Y / disc;
        double cv_mean = mean_x - beta * (mean_y - E_Y_payoff);
        price = disc * cv_mean;
        double var_cv = var_x - 2.0 * beta * cov_xy + beta * beta * var_y;
        if (var_cv < 0)
            var_cv = 0;
        se = std::sqrt(var_cv / N) * disc;
        r.method_str = "MC+CV(geometric asian)";
    } else if (s.variance_reduction == VarianceReduction::CONTROL_VARIATE && is_european) {
        // 欧式: 直接用 BS 解析解 (SE = 0)
        price = bs_reference_price(o);
        se = 0.0;
        r.method_str = "BS analytic";
    } else if (s.variance_reduction == VarianceReduction::ANTITHETIC) {
        // 对偶变量法: kernel 已对 (X(Z), X(-Z)) 取平均
        // var_x 已是平均后样本的方差 (天然包含负相关带来的缩减)
        price = disc * mean_x;
        se = std::sqrt(var_x / N) * disc;
        r.method_str = "MC+Antithetic";
    } else {
        // 普通蒙特卡洛
        price = disc * mean_x;
        se = std::sqrt(var_x / N) * disc;
        r.method_str = "MC plain";
    }

    r.price = price;
    r.std_error = se;
    r.ci_half = 1.96 * se;
    r.ref_price = bs_reference_price(o);
    r.abs_error = std::fabs(price - r.ref_price);
    r.paths_per_sec = (double)N / (r.gpu_time_ms * 1e-3);

    return r;
}

// ============================================================
// 希腊字母计算 (P4)
//   欧式 → BS 解析 Greeks (精确, 无 MC 噪声)
//   亚式/障碍 → 有限差分 (bump-and-reprice) + 公共随机数
//     Delta = (P(S+h) - P(S-h)) / (2h)
//     Gamma = (P(S+h) - 2P(S) + P(S-h)) / h^2
//     Vega  = (P(sigma+h) - P(sigma-h)) / (2h)
//     Rho   = (P(r+h) - P(r-h)) / (2h)
//     Theta = -(P(T+h) - P(T-h)) / (2h)     [单位: /年]
// ============================================================
GreeksBS compute_greeks(const OptionParams& o, const SimParams& s) {
    bool is_call = (o.type == OptionType::EUROPEAN_CALL || o.type == OptionType::ASIAN_CALL ||
                    o.type == OptionType::BARRIER_CALL);
    if (o.type == OptionType::EUROPEAN_CALL || o.type == OptionType::EUROPEAN_PUT) {
        return bs_greeks(o.spot, o.strike, o.risk_free, o.volatility, o.maturity, is_call);
    }

    // ---- 亚式/障碍: 有限差分 (公共随机数) ----
    const double h_S = 0.01 * o.spot;
    const double h_sg = 0.01;
    const double h_r = 0.0001;
    const double h_T = 1.0 / 365.0;

    auto price_at = [&](OptionParams p) -> double {
        double occ;
        PricingResult r = run_monte_carlo(p, s, occ);
        return r.price;
    };

    double P0 = price_at(o);
    OptionParams o_up = o, o_dn = o;
    o_up.spot = o.spot + h_S;
    o_dn.spot = o.spot - h_S;
    double P_S_up = price_at(o_up), P_S_dn = price_at(o_dn);

    o_up = o;
    o_dn = o;
    o_up.volatility = o.volatility + h_sg;
    o_dn.volatility = o.volatility - h_sg;
    double P_sg_up = price_at(o_up), P_sg_dn = price_at(o_dn);

    o_up = o;
    o_dn = o;
    o_up.risk_free = o.risk_free + h_r;
    o_dn.risk_free = o.risk_free - h_r;
    double P_r_up = price_at(o_up), P_r_dn = price_at(o_dn);

    o_up = o;
    o_dn = o;
    o_up.maturity = o.maturity + h_T;
    o_dn.maturity = o.maturity - h_T;
    double P_T_up = price_at(o_up), P_T_dn = price_at(o_dn);

    GreeksBS g{};
    g.Delta = (P_S_up - P_S_dn) / (2.0 * h_S);
    g.Gamma = (P_S_up - 2.0 * P0 + P_S_dn) / (h_S * h_S);
    g.Vega = (P_sg_up - P_sg_dn) / (2.0 * h_sg);
    g.Rho = (P_r_up - P_r_dn) / (2.0 * h_r);
    g.Theta = -(P_T_up - P_T_dn) / (2.0 * h_T);
    return g;
}

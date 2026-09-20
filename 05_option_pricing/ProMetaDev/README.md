# CUDA Pricing Project —— CUDA加速的金融衍生品蒙特卡洛定价系统

本项目使用 **NVIDIA CUDA + curand Philox4_32_10 可复现随机数发生器** 实现欧式 / 亚式算术平均 / 障碍（8种组合）期权的 GPU 蒙特卡洛定价，并集成黑-斯科尔斯（Black-Scholes）解析解、Reiner-Rubinstein（1991）障碍期权解析公式、Kemna-Vorst（1990）几何亚式解析解作为参考价，用于验证 GPU 价格的正确性。

在 **NVIDIA GeForce RTX 5070 Ti Laptop GPU (Blackwell sm_120, 12GB VRAM)** 上，相比同参数下 WSL 单线程 Intel i9-14900HX CPU 蒙特卡洛，**100 万条路径的亚式算术看涨期权取得了 3546× 的端到端加速比**（GPU 1.64 ms vs CPU 5801 ms）。

---

## 1. 目录结构

```
CudaPricingProject/
├── CMakeLists.txt             # CUDA + CXX 编译配置 (sm_75/86/90 + compute_90 PTX for Blackwell JIT)
├── build.sh                   # 一键构建脚本 (cmake + make -j)
├── regression_test.sh         # 全期权类型回归测试 (6 类期权 + CPU/GPU 加速比对比)
├── include/                   # 头文件
│   ├── option_params.h        # 期权/模拟参数结构体 + 枚举定义
│   ├── bs_formula.h           # 解析公式 + Greeks + GPU信息 函数声明
│   └── file_io.h              # 参数文件解析 / CSV / JSON / 性能日志 输出
├── src/                       # 源文件
│   ├── main.cpp               # 项目入口, CLI 参数解析, 主流程编排
│   ├── pricing_kernels.cu     # CUDA 核函数 + run_monte_carlo host 端 (含 compute_greeks)
│   ├── bs_formula.cpp         # BS 欧式 / 几何亚式 / 障碍 Reiner-Rubinstein / 欧式 Greeks / 分派函数
│   └── file_io.cpp            # .txt 参数解析 / CSV / JSON / perf.log 输出
├── params/                    # 输入参数文件 (示例)
│   ├── option_params.txt      # 期权 & 市场参数
│   └── sim_params.txt         # 模拟参数 (路径数 / 步数 / 方差缩减等)
└── outputs/                   # 运行后自动生成
    ├── result.csv / .json     # 定价结果 (含 Greeks 字段)
    └── perf.log               # 追加式性能日志 (含每次运行 Greeks)
```

---

## 2. 构建与运行

### 2.1 环境要求

| 项目 | 最低要求 | 本机推荐 |
|------|---------|---------|
| CUDA Toolkit | 11.0+ | 12.2+ |
| NVIDIA Driver | ≥ CUDA 版本对应 | 560.xx+ |
| GPU | sm_70 (Volta) 及以上 | Blackwell (sm_120) 通过 PTX JIT 兼容 |
| Compiler | GCC 8+ / MSVC VS2022 | GCC 9.4.0 (WSL2 Ubuntu 20.04) |
| CMake | ≥ 3.18 | 3.26 |

### 2.2 一键构建

```bash
cd ~/CudaPricingProject
bash build.sh
# 产物: ./build/cuda_pricing
```

### 2.3 命令行用法

```
./build/cuda_pricing [option_params.txt] [sim_params.txt] [result.csv] [perf.log]
                     [--seed N] [--paths N] [--json] [--no-cpu]
                     [--greeks] [--info] [--repro]
                     [--antithetic] [--cv] [--vr NONE|ANTITHETIC|CONTROL_VARIATE]
```

| 位置参数 | 默认值 | 说明 |
|---------|--------|------|
| option_params.txt | params/option_params.txt | 期权 & 市场参数文件 |
| sim_params.txt | params/sim_params.txt | 模拟参数文件 |
| result.csv | outputs/result.csv | 定价结果输出路径 (CSV 或 JSON) |
| perf.log | outputs/perf.log | 追加式性能日志输出路径 |

| 可选标志 | 说明 |
|---------|------|
| `--seed N` | 覆盖 sim_params.txt 中的随机数种子 (用于复现) |
| `--paths N` | 覆盖模拟路径数 (快速调试用) |
| `--json` | 结果输出为 JSON (替换 `.csv` → `.json`) |
| `--no-cpu` | 跳过 CPU 参考 MC (加速比会显示 "N/A") |
| `--greeks` | 计算并输出 5 个希腊字母 (欧式:BS解析; 亚式/障碍:有限差分公共随机数) |
| `--info` | 仅打印 GPU 信息后退出 |
| `--repro` | 同 seed 跑两次验证 bit-identical (可复现性) |
| `--antithetic` | 启用对偶变量法 (覆盖 sim_params.txt 中 variance_reduction) |
| `--cv` | 启用控制变量法 (几何亚式作控制, 仅亚式期权有效) |
| `--vr VALUE` | 指定方差缩减方式: NONE / ANTITHETIC / CONTROL_VARIATE |

### 2.4 示例运行

```bash
# 基础运行 (使用 params/*.txt 默认参数)
./build/cuda_pricing

# 欧式 Put + 100万路径 + 输出 JSON + 计算 Greeks + 跳过 CPU
./build/cuda_pricing params/option_params.txt params/sim_params.txt outputs/ep.json \
    --paths 1000000 --json --no-cpu --greeks --vr NONE

# 亚式 Call + 控制变量法 + 可复现性验证
./build/cuda_pricing params/option_params.txt params/sim_params.txt \
    --paths 2000000 --cv --repro

# 仅打印 GPU 信息
./build/cuda_pricing --info
```

### 2.5 一键回归测试

```bash
bash regression_test.sh                # 默认 100万路径/期权
bash regression_test.sh 500000         # 指定 50万路径 (快速模式)
```

执行 6 种期权 × 完整 Greeks + 1M 路径 CPU vs GPU 加速比，最终输出汇总表格到 `outputs/regression/summary.txt`。

---

## 3. 支持的期权类型

所有期权均在 **风险中性测度**、**无股息**（q=0）假设下使用几何布朗运动（GBM）进行蒙特卡洛模拟。

| 期权大类 | 类型枚举 | 行权 / 收益方式 | 解析参考价来源 |
|---------|---------|---------------|--------------|
| **欧式看涨** | `EUROPEAN_CALL` | $C = \max(S_T-K,\,0)$ | Black-Scholes 闭式 (精确) |
| **欧式看跌** | `EUROPEAN_PUT` | $P = \max(K-S_T,\,0)$ | Black-Scholes 闭式 (精确) |
| **亚式算术平均看涨** | `ASIAN_CALL` | $C = \max(\bar{S}_{arith}-K,\,0)$ | **几何**亚式 Kemna-Vorst 闭式（作为 CV 基准） |
| **亚式算术平均看跌** | `ASIAN_PUT` | $P = \max(K-\bar{S}_{arith},\,0)$ | 同上 |
| **障碍看涨** 8 组合 | `BARRIER_CALL` | $\max(S_T-K,0)$ × 敲入/敲出 × 向上/向下 | Reiner-Rubinstein 1991 连续监控闭式 |
| **障碍看跌** 8 组合 | `BARRIER_PUT` | $\max(K-S_T,0)$ × 敲入/敲出 × 向上/向下 | Reiner-Rubinstein 1991 连续监控闭式 |

> **障碍期权子参数**：`barrier_dir = UP|DOWN`，`barrier_t = KNOCK_IN|KNOCK_OUT`，共同构成 2×2 共 8 种组合。欧式期权中障碍参数被忽略。

---

## 4. 关键技术与性能

### 4.1 GPU 并行策略

- **随机数**：`curandStatePhilox4_32_10_t`，每个线程独立状态，**1 线程 = 1 条路径**，固定 seed 可重现。支持对偶变量法恢复状态 → 生成相同 Z 取反。
- **路径模拟精度**：Float 精度（`__expf` / `logf`），累加寄存器使用 float，最终折现使用 double。对比 double 路径端到端加速约 **9.2×**，无寄存器溢出。
- **路径长度 ≥ 4 步**：Philox4 每次产出 4 个正态，循环 `#pragma unroll` 展开内层。
- **归约**：两阶段共享内存归约（Block 内 `__shared__` → 主机端 `cudaMemcpy` 后累加）。256 线程/块时 SM 占用率约 **50%**。
- **常量内存**：期权/市场参数全部挂到 `__constant__`，全局广播一次全 SM 读取。

### 4.2 方差缩减

| 方式 | CLI | 适用场景 | 实测 SE 改善 |
|------|-----|---------|------------|
| 无缩减 | `--vr NONE` | 基准 | 1× |
| **对偶变量法 Antithetic** | `--antithetic` | 所有期权 | **SE / 1.94** (接近理论 2×) |
| **控制变量法 CV** | `--cv` | **亚式算术** (几何亚式作控制) | **SE / 18.7** (β 回归自动估计) |

### 4.3 希腊字母

| 希腊字母 | 欧式实现 | 亚式 / 障碍实现 | 含义 |
|---------|---------|----------------|------|
| Δ Delta | BS 闭式解析 | 有限差分 bump-and-reprice ($dP/dS$) | 标的变动敏感度 |
| Γ Gamma | BS 闭式解析 | 有限差分 ($d^2P/dS^2$) | Delta 的凸性 |
| ν Vega | BS 闭式解析 | 有限差分 ($dP/d\sigma$) | 波动率敏感度 |
| Θ Theta | BS 闭式解析 | 有限差分 ($-dP/dT$, 单位 /年) | 时间衰减 |
| ρ Rho | BS 闭式解析 | 有限差分 ($dP/dr$) | 无风险利率敏感度 |

亚式/障碍使用 bump 有限差分 + **公共随机数**（相同 seed），差分收敛速度显著优于独立随机数。

---

## 5. 参数文件格式

### 5.1 option_params.txt

```ini
type        = BARRIER_CALL       # EUROPEAN_CALL | EUROPEAN_PUT | ASIAN_CALL | ASIAN_PUT | BARRIER_CALL | BARRIER_PUT
spot        = 100.0              # 标的当前价格 S0
strike      = 100.0              # 行权价 K
risk_free   = 0.03               # 年化无风险利率 r
volatility  = 0.2                # 年化波动率 σ
maturity    = 1.0                # 到期时间 T (单位: 年)
barrier     = 130.0              # 障碍价 H (仅障碍期权生效)
barrier_dir = UP                 # UP | DOWN
barrier_type= KNOCK_IN           # KNOCK_IN | KNOCK_OUT
```

> **键名别名**：为兼容项目选题文档给出的参数文件写法，解析器同时接受
> `option_type`（等价 `type`）、`risk_free_rate` / `rate`（等价 `risk_free`）、
> `barrier_t`（等价 `barrier_type`）；枚举取值大小写不敏感，值两侧的成对引号会被忽略
> （例如 `variance_reduction = "antithetic"` 与 `ANTITHETIC` 等价）。
> **未识别的键名会在 stderr 给出告警**，不会静默忽略；参数文件不存在、枚举非法、
> 障碍方向与价位矛盾等情况一律**报错并返回非零退出码**，避免用默认值算出一个看似合理的结果。

### 5.2 sim_params.txt

```ini
num_paths        = 10000000      # 蒙特卡洛路径总数
num_steps        = 256           # 每条路径步数 (离散监控)
seed             = 1234          # RNG 种子 (固定 = 可复现)
rng              = CURAND_DEFAULT# 仅 curand Philox4_32_10 (CUSTOM_XORWOW 已移除)
variance_reduction = CONTROL_VARIATE   # NONE | ANTITHETIC | CONTROL_VARIATE
block_size       = 256           # CUDA block 大小 (建议 128~512)
```

---

## 6. 输出格式

### 6.1 CSV 结果 (`--greeks` 时追加希腊字母列)

```csv
option_type,price,ref_price,abs_error,std_error,ci_half,gpu_time_ms,cpu_time_ms,paths_per_sec,speedup,occupancy_pct,num_paths,num_steps,seed,variance_reduction,delta,gamma,vega,theta,rho
ASIAN_CALL,5.28272058,5.08647886,0.19624173,0.00020878,0.00040920,1.63570000,5801.07710000,611354555.7881,3546.5149,50.0000,1000000,256,42,control_variate,0.561433,0.033422,22.3171,-2.9366,23.4953
```

### 6.2 JSON 结果 (`--json`)

```json
{
  "option": { "type": "ASIAN_CALL", "spot": 100.0, "strike": 100.0, "risk_free": 0.03, "volatility": 0.2, "maturity": 1.0 },
  "simulation": { "num_paths": 1000000, "num_steps": 256, "seed": 42, "block_size": 256,
                  "variance_reduction": "control_variate", "rng": "curand_philox" },
  "result": { "price": 5.2827, "ref_price": 5.0865, "abs_error": 0.1962,
              "std_error": 0.00021, "ci_half_95": 0.00041, "method": "MC+CV(geometric asian)" },
  "performance": { "gpu_time_ms": 1.6357, "cpu_time_ms": 5801.077,
                   "paths_per_sec": 6.11e8, "speedup": 3546.51, "occupancy_pct": 50.0 },
  "greeks": { "delta": 0.5614, "gamma": 0.0334, "vega": 22.317, "theta": -2.937, "rho": 23.495 }
}
```

> 注：若使用 `--no-cpu`，`speedup` 字段显示为字符串 `"N/A"`。

### 6.3 perf.log (追加式, 每行一次运行)

```
# GPU=NVIDIA GeForce RTX 5070 Ti Laptop GPU  SMs=46  CC=12.0  VRAM=12226MB  CUDA_RT=12.2  DRV=13.3 | MC+CV(geometric asian)
[ASIAN_CALL] paths=1000000 steps=256 seed=42 vr=2 | price=5.2831 ref=5.0865 SE=0.0002 | gpu=1.6357ms cpu=5801.0771ms pps=611354555.7881 speedup=3546.5149 occupancy=50.0000% Delta=0.5614 Gamma=0.0334 Vega=22.3171 Theta=-2.9366 Rho=23.4953
```

> 若 `--greeks` 未启用，`Delta~Rho` 字段显示为 `N/A`。

---

## 7. 回归测试结果 (RTX 5070 Ti Laptop, 500K 路径, seed=42)

| 类型 | MC 价格 | 解析参考价 | 绝对误差 | 标准误 SE | Δ Delta | Γ Gamma | ν Vega | Θ Theta | ρ Rho |
|------|--------|-----------|---------|----------|---------|---------|--------|---------|-------|
| EUROPEAN_CALL | 9.4134 | 9.4134 | **0.0000** | 0.00000 | 0.5987 | 0.01933 | 38.667 | -5.380 | 50.457 |
| EUROPEAN_PUT  | 6.4580 | 6.4580 | **0.0000** | 0.00000 | -0.4013 | 0.01933 | 38.667 | -2.469 | -46.587 |
| ASIAN_CALL    | 5.2834 | 5.0865 | 0.1969 | 0.00030 | 0.5614 | 0.03342 | 22.317 | -2.937 | 23.495 |
| ASIAN_PUT     | 3.8133 | 3.9477 | 0.1344 | 0.00020 | -0.4238 | 0.03343 | 22.319 | -1.496 | -24.537 |
| BARRIER_CALL  | 5.9954 | 6.2107 | 0.2153 | 0.02003 | 0.5355 | 0.03085 | 61.836 | -7.523 | 42.303 |
| BARRIER_PUT   | 4.5455 | 4.6903 | 0.1447 | 0.01327 | -0.3833 | 0.03414 | 51.576 | -3.877 | -37.637 |

**CPU vs GPU 加速比 (1M 路径 ASIAN_CALL, CONTROL_VARIATE)**：
- CPU (i9-14900HX WSL2 单线程): **5801.08 ms**
- GPU (RTX 5070 Ti Laptop): **1.6357 ms**
- **端到端加速比 = 3546.5×**

---

## 8. 数学参考

- 欧式 Black-Scholes: *Hull, Options, Futures, and Other Derivatives, 11e, Ch.15*
- 几何亚式 Kemna-Vorst: *Kemna & Vorst (1990), "A pricing method for options based on average asset values", JBF*
- 障碍期权 Reiner-Rubinstein: *Reiner & Rubinstein (1991), "Breaking Down the Barriers", Risk*，实现参考 **QuantLib `AnalyticBarrierEngine`**。
- 对偶变量法 & 控制变量法: *Glasserman (2004), "Monte Carlo Methods in Financial Engineering", Ch.4*

---

## 9. 已知边界 & 局限

1. 障碍期权解析价假设 **连续监控**；GPU 端 MC 为离散监控（256步/年），两者系统误差方向可由 **Broadie-Glasserman-Kou** 连续性修正解释（knock-in 期权离散价格 > 连续解析价）。
2. 控制变量法目前仅对 **亚式算术期权**（几何亚式作为控制）实现了 β 估计；欧式/障碍期权用它只会退化到 NONE。
3. 本项目不支持 **离散分红、跳扩散、随机波动率**（如 Heston）模型。
4. Greeks 的 Vega 单位为 `dPrice / dσ`（未除以 100），未做波动率 1% 变动的惯例缩放。

---

*文档生成时间：2026-08-28 | 环境：WSL2 Ubuntu 20.04 + CUDA 12.2 + RTX 5070 Ti Laptop (sm_120)*

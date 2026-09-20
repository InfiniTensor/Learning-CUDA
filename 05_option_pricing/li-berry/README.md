# 金融衍生品定价与风险估计（CUDA 蒙特卡洛）

2026 夏季训练营 CUDA 方向项目五实现。GPU 加速蒙特卡洛期权定价：欧式 / 亚式 / 障碍期权，
支持 cuRAND 与自定义 RNG、对偶变量与控制变量方差缩减，输出价格估计 + 95% 置信区间 +
误差/收敛指标，并与 Black-Scholes 等解析解对比。

## 构建与运行

```bash
# 环境：CUDA 12.4 + sm_89（RTX 4060），g++ 15，make
make                    # 构建 build/option_pricer
./build/option_pricer --option configs/option_european_call.txt --sim configs/sim_default.txt
```

输出：
- `results/pricing_result.txt` — 价格估计、标准误、95% CI、参考价格、绝对/相对误差、
  z 分数、`|error|·√N` 收敛指标、方差缩减因子
- `results/perf_log.txt` — 总耗时（ms）、paths/sec、粒度配置（块/网格/每 SM 块数/寄存器）、
  归约耗时、相对 CPU 单线程的加速比

### 命令行参数

| 参数 | 说明 |
|---|---|
| `--option <f>` / `--sim <f>` | 期权/模拟参数文件（`key = value`，`#` 注释，兼容 CRLF/BOM） |
| `--mode cpu\|gpu\|both` | 计算引擎（默认 gpu；both 附带 CPU 单线程基线对比） |
| `--block N` / `--grid N` | GPU 粒度（默认 256 / 按占用率自动） |
| `--cpu-paths N` / `--cpu-threads N` | CPU 基线路径上限（默认 500k）/ 线程数（>1 用 OpenMP） |
| `--euro-steps N` | 欧式模拟步数（默认 1 = 精确终值公式；>1 走逐步路径验证） |
| `--bgk` | 障碍期权 BGK 边界修正（离散监控 → 连续监控） |
| `--sweep` | N = 1e4..1e7 收敛扫描（stdout CSV + 拟合斜率） |
| `--dump-rng N` | 转储 RNG 流（CPU/GPU 一致性验证，N 为 16 的倍数） |

### 测试与基准

```bash
make test-units   # 纯 CPU 单元测试：解析解黄金值、解析器、RNG、统计量（43 项断言）
make e2e          # GPU 端到端回归 24 项（< 30s）
make convergence  # 收敛扫描 → data/convergence.csv
make bench        # block×grid×rng×方差缩减 基准 → data/bench.csv
make profile      # nsys + ncu 性能分析
```

黄金参考值（解析公式推导 + Python 交叉验证）见 `data/reference_values.txt`。
注意：S0=K=100, r=0.03, σ=0.2, T=1 的 BS 看涨参考值是 **9.413403**（10.45 是 r=5% 的值）。

## 设计要点

- **方法**：蒙特卡洛 + 精确 GBM 转移 `S_{i+1} = S_i·exp((r−σ²/2)Δt + σ√Δt·Z)`
  （非 Euler 离散，终值分布无时间离散偏差）。欧式 1 步精确模拟；亚式/障碍 256 步。
- **并行粒度**：一路径（对偶时一对）一线程单元，grid-stride；融合生成+模拟+收益，
  每线程 O(1) 寄存器状态，全 GPU 内存占用 < 1MB（10M×256 的 Z 矩阵 10GB 放不进 8GB 显存）。
- **随机数**：流位置仅由 `(seed, path_index)` 决定——curand Philox4x32-10 每路径一条独立
  子序列；自定义 xorshift128+（splitmix64 播种）。同一配置重跑 **bitwise 一致**，
  N 较小运行是 N 较大运行的精确子集（嵌套前缀 → 收敛曲线干净）。
- **精度**：路径循环 float（Ada 消费卡 FP64 = 1/64 FP32，double 会慢 30–60×），
  每线程/块/全局归约 double。FP32 路径误差 ~2.4e-4 ≪ MC 标准误 3.8e-3@N=10⁷。
- **归约**：两级确定性归约（warp shuffle → 块内 → finalize 单块核），无原子操作
  （FP 加法非结合，原子序不确定会破坏可复现性）。
- **方差缩减**：对偶变量（同线程锁步模拟 (Z,−Z) 对）；控制变量按期权类型选择——
  欧式用折现终值（E= S0）、亚式用几何平均收益（E=Kemna-Vorst 闭式解，**方差缩减 ~1400×**）、
  障碍用 vanilla 看涨（实测无效 ≈1.0，up-and-out 触障路径 payoff=0 而控制值大的协方差
  抵消所致，报告中有分析）。
- **障碍离散偏差**：256 步离散监控偏差 +0.146 且不随 N 收敛；BGK 边界平移
  `H·exp(−0.5826σ√Δt)` 将偏差降到 ~0.0035（42×），残差随 n 增大收敛（见报告）。

## 性能（RTX 4060 Laptop, 10M 路径 × 256 步）

| 场景 | GPU 吞吐 | CPU 单线程 | 加速比 |
|---|---|---|---|
| 亚式看涨（plain） | 2.48e8 paths/s（63 G steps/s） | 3.6e5 paths/s | **683×** |
| 障碍看涨（+BGK） | 3.31e8 paths/s（85 G steps/s） | 4.8e5 paths/s | **690×** |
| 欧式看涨（1 步精确） | 5.9e9 paths/s | 3.7e7 paths/s | 160× |

（CPU 基线为同参数 custom RNG 镜像实现；加速比按 paths/sec 比值，两者路径数不同时以
paths/sec 归一化比较，详见性能日志 `speedup_basis` 字段。）

## 目录结构

```
src/        核心代码（types/config/analytic/rng/payoff/sim_core/engine_common/
            stats/output/cpu_engine/gpu_engine/main）
configs/    示例参数文件（4 种期权 × 6 组模拟参数）
tests/      test_units.cpp（CPU 单元测试）+ run_e2e.sh（GPU 端到端 24 项）
scripts/    convergence.sh / bench.sh / check.sh
data/       黄金参考值 + 生成的 CSV（报告图表数据源）
report/     总结报告（见 report.md）
```

详细设计与分析见 `report/report.md`。

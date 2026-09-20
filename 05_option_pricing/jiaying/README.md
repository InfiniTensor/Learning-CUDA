# 选题五：金融衍生品定价与风险估计

作者目录：`jiaying`。C++17 / CUDA，已在 WSL2 Ubuntu 26.04、RTX 4070 Laptop GPU 上验证。

## 功能

- 欧式看涨、欧式看跌、离散算术平均亚式看涨、离散上敲出看涨期权。
- 精确 GBM 采样；普通蒙特卡洛、对偶采样、独立先导样本控制变量。
- 价格、样本标准误、渐近 95% 置信区间、欧式 Black–Scholes 参考值。
- Delta / Gamma；欧式和亚式同时支持 Vega。风险指标也输出标准误。
- cuRAND Philox 和自定义 Philox；CPU 单线程复现相同随机整数和正态变换。
- GPU 在线生成路径和归约，只使用 O(线程块数) 显存；支持千万路径。
- 严格配置检查、机器可读 JSON、自动测试、基准实验与绘图。

## 在 WSL 构建和运行

本机路径：

```bash
cd /mnt/d/Jiaying/Documents/ChatGPT/InfiniTensor/Learning-CUDA/05_option_pricing/jiaying
bash scripts/build_wsl.sh
./build/option_pricing --option examples/asian_call.cfg \
  --simulation examples/control_variate.cfg --backend both --output results/demo_asian
```

依赖：CMake ≥3.24、CUDA Toolkit（本机 12.4）、CUDA 兼容的 C++17 编译器（本机 g++-13），测试需要 Python 3，绘图需要 NumPy、Matplotlib。默认脚本为本机的 SM89 编译；其他 GPU 例如 Turing 使用 `CUDAARCHS=75 bash scripts/build_wsl.sh`。没有使用 Tensor Core 或 Ampere 专有指令。

通用构建：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_CUDA_HOST_COMPILER=g++-13 \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

纯 CPU 模式也可构建：

```bash
cmake -S . -B build-cpu -DOPTION_ENABLE_CUDA=OFF -DCMAKE_CXX_COMPILER=g++-13
cmake --build build-cpu -j4
ctest --test-dir build-cpu --output-on-failure
./build-cpu/option_pricing --option examples/european_call.cfg \
  --simulation examples/simulation.cfg --backend cpu --output results/cpu
```

## 输入和输出

输入是 `key=value` 文本，可带引号、`#` 注释、CRLF 和 UTF-8 BOM。未知键、重复键、非法数值都会报错。示例覆盖题面全部期权与方差缩减模式。未写出的键使用示例对应默认值。

| 配置项 | 含义 |
|---|---|
| `num_paths` | 实际主样本路径数，至少 4；对偶模式必须为偶数 |
| `num_steps` | 亚式/障碍监控次数；欧式自动用单步精确终值 |
| `rng` | `curand`（默认）或 `custom` / `philox` |
| `variance_reduction` | `none`、`antithetic` 或 `control_variate` |
| `pilot_paths` | 控制变量独立先导样本数，默认 8192，另计工作量 |
| `block_size` | 32–512，32 的倍数，默认 256 |
| `blocks` | 0 自动选择，或 1–65535 |
| `spot_bump` | Gamma / 障碍 Delta 的相对差分步长，默认 0.01 |

`--backend cpu|gpu|both` 选择计算后端，默认 `both`；`--output PREFIX` 创建：

- `PREFIX.json`：完整输入、方法、价格、置信区间、参考误差、风险指标。
- `PREFIX_performance.json`：总计算时间、初始化时间、核函数时间、最终归约时间、路径吞吐、粒度、显存与 CPU 加速比。

亚式平均取 `S(T/m), ..., S(T)`，不含 `S0`。障碍为上敲出：`S0` 或任何监控点达到/超过障碍即收益归零，无返还。没有把离散障碍冒充连续障碍。

`ci95` 只描述蒙特卡洛抽样误差，不覆盖模型误差、浮点误差或有限差分偏差。Gamma 使用共享随机数中心差分；障碍 Vega 未实现并输出 `null`。控制变量只修正价格，Greeks 保留原始估计与各自标准误。Vega 单位为波动率变化 1.0，而非 1 个百分点。

## 复现实验

```bash
python3 scripts/benchmark.py --exe build/option_pricing --output results
python3 scripts/analyze.py --results results
```

完整基准包括 3 次重复的百万路径 CPU/GPU 比较、8 个种子的收敛实验、千万路径压力测试和线程块/RNG 对照；CPU 路径需要数分钟。每次实验的配置与原始 JSON 保留在 `results/raw/`，总表为 `results/benchmark.csv`。测试程序与生产程序分离。

报告见 [REPORT.md](REPORT.md)，实测汇总见 [results/summary.md](results/summary.md)，图表在 `results/figures/`。硬件与软件信息见 `results/environment.json`。

CUDA 与纯 CPU 构建各通过 40 项功能验证。当前 WSL 的 Compute Sanitizer 调试接口不可用；Nsight Systems 仅成功导出 CUDA API 统计，详细限制和原始日志见 [results/profiling.md](results/profiling.md)。

## 文件结构

```text
include/          模型、Philox 随机数、稳定统计归约接口
src/              CPU、CUDA、配置与结果输出、命令行
examples/         合约和模拟配置
tests/            正确性、独立参考、边界条件验证
scripts/          WSL 构建、性能与收敛实验、绘图
results/          实测原始数据、图表、测试与分析日志
REPORT.md         模型、方法、优化和实测分析
```

提交目录符合 `2026-summer-project` 分支的 `05_option_pricing/jiaying/` 规范。编译产物、临时配置与大型 profiler 数据文件已排除；保留可复查的文本日志和测量结果。

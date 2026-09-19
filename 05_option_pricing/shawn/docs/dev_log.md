# Development Log

> 最终状态说明（2026-09-19）：以下条目保留开发当时的计划和问题状态。
> 正式单卡 RTX 5090 实验已完成并汇入 `REPORT.md`；多 GPU 仅有代码路径，
> 尚未进行多卡正确性或扩展性实验。早期“GPU 待测”的记录不代表当前单卡状态。

## Project Overview

CUDA 衍生品定价、风险估计与 CPU reference 工程。

## Implementation Progress

- 初版及功能扩展：基础/路径依赖期权、Greeks、American LSM、Heston、Local Vol、QMC、多 GPU 的 CPU/CUDA 路径已完成。
- 当前：迭代 #1，完成服务器验收产物与报告证据闭环。

## Development Log Entries

### 2026-09-20 — 提交前整理

压缩重复注释，保留路径采样、反向变量、回归及计时边界的说明；修正 LSM 主机分工
和多 GPU 独立性假设的注释。未改变计算语句、配置或实验结果。
独立提交目录以 Release、CUDA=OFF 构建，CTest 1/1 通过；本轮未重新运行 GPU。

### 2026-08-25 13:31 — 迭代 #1：诊断与设计冻结

**改动原因**：本地 CPU CTest 已通过，但 GPU 矩阵只生成 CSV，环境与报告表仍需人工整理。
**改动内容**：
- `docs/user_requirements.md`：固定 CPU/GPU 双结果与本地/远端边界。
- `docs/idea_report.md`：记录方法和实验问题。
- `docs/implementation.md`：定义报告型验收产物。
**预期效果**：一次服务器验证即可形成数值 gate、环境清单和报告草表，减少人工转录错误。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 否

## Known Issues

- 当前机器无 CUDA，GPU 数值与性能结论仍待服务器实测。
- REPORT 中正式数值表尚未填充。

### 2026-08-25 13:57 — 迭代 #1：生成服务器环境与报告草表

**改动原因**：原 GPU 矩阵只汇总数值 gate，无法自动追溯硬件/工具链，也需要人工把 CSV 转入报告。
**改动内容**：
- `scripts/run_gpu_validation.py`：解析绝对 binary 路径，采集 platform/NVIDIA/nvcc 环境，为每行加入配置名，并生成 `environment.txt` 与 `report_table.md`。
**预期效果**：一次服务器运行同时完成数值验收和报告证据整理，降低错配配置或抄写数字的风险。
**文档同步**：idea_report.md 否 | implementation.md 是 | configs/ 否

### 2026-08-25 14:00 — 迭代 #1：同步验收产物说明

**改动原因**：新生成的环境与 Markdown 产物需要进入用户运行说明和报告流程。
**改动内容**：
- `README.md`：记录三个汇总产物和逐案例日志保留规则。
- `REPORT.md`：指定从自动生成草表填充正式结果，并禁止从终端手工抄录 GPU 数值。
**预期效果**：服务器运行到最终报告之间形成明确、可追溯的交接。
**文档同步**：idea_report.md 否 | implementation.md 是 | configs/ 否

### 2026-08-25 14:04 — 迭代 #1 结果

| 指标 | 改动前 | 改动后 | 变化 |
|---|---:|---:|---|
| Release CPU CTest | 1/1 通过 | 1/1 通过 | 保持 |
| UBSan CPU CTest | 1/1 通过 | 1/1 通过 | 保持 |
| GPU 验收汇总 | `summary.csv` | CSV + 环境 + Markdown 草表 | 增强 |
| Python 静态检查 | 通过 | 通过 | 保持 |

**结论**：迭代有效，未改变算法结果或接口；本地完整 CPU/UBSan 回归通过。GPU 数值表只能在服务器执行 `run_gpu_validation.py` 后确认。

## Run Instructions

- 本地 CPU：`cmake --build build -j && ctest --test-dir build --output-on-failure`
- 服务器 GPU：`python3 scripts/run_gpu_validation.py --binary build/derivatives_pricer --output-dir outputs/gpu_validation`
- 多 GPU：在至少两张卡可见时追加 `--include-multi-gpu`。

### 2026-09-09 — 迭代 #2：American Gamma 诊断与实验设计更新

**改动原因**：RTX 5090 实测中，50,000 路径的 American pseudo 仅 Gamma 失败；提高到 200,000 路径后所有指标通过，且 American Halton 的 CPU/GPU 输出逐项一致。
**诊断证据**：
- 50k pseudo Gamma：CPU `0.0351127`、GPU `0.0233494`，绝对差约 `0.01176`，略高于 `0.01` 门限。
- 200k pseudo Gamma：CPU `0.0225246`、GPU `0.0233072`，绝对差约 `0.000783`，通过门限。
- Halton：price、Delta、Gamma、Vega 的 CPU/GPU 绝对差均为 `0`；这证明实现路径一致，但负 Gamma 不单独作为绝对精度结论。
**改动范围**：只调整 American 正式验证配置与矩阵容错/汇总行为，不修改定价模型、LSM 或容差。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 待完成

### 2026-09-09 — 迭代 #2：正式矩阵配置与失败汇总实现

**改动原因**：50k American pseudo 的二阶有限差分噪声会偶发触发 Gamma 门禁，且原驱动器在首个失败后中止，无法得到完整矩阵。
**改动内容**：
- `configs/simulation_american_validation.txt`：固定 200,000 路径、seed `20260824` 与 1% spot bump，专供 American pseudo 正式 gate。
- `scripts/run_gpu_validation.py`：American pseudo 使用专用配置；单案例失败后继续；删除陈旧 comparison；总表保留执行失败行；全部完成后统一非零退出。
- `README.md`、`REPORT.md`：同步正式运行口径，并区分 Halton 实现一致性证据与 CRR 绝对精度证据。
**预期效果**：不放宽统计容差的前提下稳定验证 American Gamma，并保证失败矩阵也能留下完整、可审计的报告产物。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 是

### 2026-09-09 — 迭代 #2 结果

| 指标 | 改动前 | 改动后 | 变化 |
|---|---:|---:|---|
| American pseudo 通过路径数 | 50,000（Gamma 失败） | 200,000（四项通过） | 稳定通过 |
| American pseudo Gamma 绝对差 | 约 0.01176 | 约 0.000783 | 降低约 93.3% |
| American Halton CPU/GPU 差 | 未执行 | 四项均为 0 | 实现一致 |
| Release CPU CTest | 1/1 通过 | 1/1 通过 | 保持 |
| 无 CUDA 失败路径覆盖 | 首个 case 后中止 | 8/8 case 均执行并汇总 | 完整 |

**结论**：迭代有效。服务器证据支持将原失败归因于 50k 路径下 American LSM Gamma 的统计波动；没有放宽门限或更改算法。本地无 CUDA 演练确认矩阵会保留完整失败清单和汇总产物。更新后的真实 CUDA 全矩阵仍需在 RTX 5090 上复跑后归档。

### 2026-09-09 — 迭代 #3：独立 reference gate 诊断与设计

**改动原因**：RTX 5090 矩阵已达到 8/8 cases、32/32 CPU/GPU 指标通过，但现有汇总只证明后端一致性，未展示 Black–Scholes/CRR 的独立价格与 Greeks reference。当前 American QMC Gamma 为 `-0.0031627`，虽然 CPU/GPU 完全相同，仍不能作为绝对精度证据。
**改动范围**：新增解析/CRR reference 输出与门禁，并为 American QMC 选择更稳定的正式配置；不修改 Monte Carlo、LSM 或 CUDA 算法。
**预期效果**：报告表能够直接区分 backend parity 和 financial-reference accuracy，防止共同误差被一致性 gate 掩盖。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 待完成

### 2026-09-09 — 迭代 #3：解析/CRR reference gate 实现

**改动内容**：
- `include/pricing/pricing.hpp`、`src/pricing_cpu.cpp`：新增 American 2,000 步 CRR 中心差分 Greeks reference，使用与估值端一致的 bump。
- `src/main.cpp`：结果文件新增 `reference_delta/gamma/vega` 与 reference method；European 使用 Black–Scholes 解析 Greeks，American 使用 CRR Greeks。
- `scripts/compare_cpu_gpu.py`：分别记录 backend parity 与每个后端的 reference error/tolerance，并合并为 overall gate。
- `scripts/run_gpu_validation.py`：报告草表展示 reference 误差与状态，American Halton 改用专用正式配置。
- `configs/simulation_american_qmc_validation.txt`：固定 131,072 路径、64 步、8 个 randomized-Halton replications 和 1% spot bump。
- `tests/test_pricing.cpp`、`README.md`、`REPORT.md`：补 reference API/config 回归与报告口径。
**预期效果**：CPU/GPU 即使共同偏离 Black–Scholes/CRR，也不能通过正式矩阵；无独立 reference 的模型明确标记为 `n/a`。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 是

### 2026-09-09 — 迭代 #3 本地结果

| 指标 | 旧 American QMC smoke | 新 American QMC validation | Reference |
|---|---:|---:|---:|
| Paths / steps | 16,384 / 32 | 131,072 / 64 | — |
| Price | 6.7995046 | 6.7413820 | CRR 6.7425208 |
| Price 绝对误差 | 0.0569838 | 0.0011388 | — |
| Gamma | -0.0031627 | 0.0133855 | CRR 0.0225197 |
| Gamma 绝对误差 | 0.0256824 | 0.0091342 | 门限 0.01 |

**验证**：Release CPU CTest 1/1 通过，UBSan CPU CTest 1/1 通过，Python compileall 通过；本机无 CUDA 的矩阵演练仍执行并记录全部 8 个失败 case。对新 American QMC 本地结果模拟同值 CPU/GPU 时，price/Delta/Gamma/Vega 四个 reference gates 全部通过。

**结论**：新 QMC 正式配置显著减小 American LSM 的价格与二阶差分偏差，并通过当前独立 CRR 门限。真实 CUDA 后端仍需在 RTX 5090 上重新构建并执行矩阵，确认 GPU reference gates。

### 2026-09-10 — 迭代 #4：解耦数值 reference 与 CPU 性能基线

**改动原因**：旧 CLI 复用 `--reference-paths` 同时控制路径依赖合约的独立数值 reference 和 CUDA 加速比中的 CPU 样本数，导致控制变量 reference 可能被误当成不同方差模式的性能基线。
**改动内容**：
- `src/main.cpp`：新增 `--cpu-benchmark-paths`；性能基线使用与被测配置相同的模型、步数、RNG 路径类型和方差缩减设置。
- 未提供新参数时继续使用 `--reference-paths`，保持旧命令行为兼容。
**预期效果**：报告中的单线程 CPU/GPU 吞吐对比和金融 reference 分别可追溯，避免混合两种实验口径。
**运行说明影响**：服务器性能实验需显式提供 `--cpu-benchmark-paths`；README 与最终实验入口将在本轮同步。

### 2026-09-10 — 迭代 #4：增强 sweep 统计与报告产物

**改动原因**：原 `aggregate.csv` 只有均值、经验标准差和吞吐，无法直接检查单次标准误是否被跨 seed 波动支持，也没有独立控制 CPU 性能基线。
**改动内容**：
- `scripts/run_sweep.py`：新增 `--cpu-benchmark-paths`，记录运行命令与 option 配置快照。
- 聚合结果新增经验 SD/平均 SE 比、相对标准误、平均 `SE²×runtime`、CPU 单线程加速比。
- 新增 `report_table.md`，同时保留 `summary.csv` 和 `aggregate.csv` 作为机器可读原始证据。
**预期效果**：收敛性、置信区间校准、方差缩减效率和吞吐对比均能从同一组可复现实验中提取。
**运行说明影响**：正式 sweep 应明确指定 `--reference-paths` 与 `--cpu-benchmark-paths`，两者不可再混用。

**本地复验修正**：CPU quick preset 首次完整运行后发现，未请求 reference/CPU benchmark 的聚合行会显示 `nan` 或 `0`。`run_sweep.py` 已改为将未测项留空，避免报告把缺失证据误读为实测零值。

**代码审查修正**：底层 performance log 已有 CPU benchmark 路径数、耗时和吞吐，但旧 sweep CSV 字段会忽略这些值。现已在 raw/aggregate/Markdown 三层保留 simulation kernel、reduction、kernel throughput、CPU paths、CPU ms、CPU throughput 与最终 speedup，直接覆盖题目性能日志和报告表要求。

### 2026-09-10 — 迭代 #4：强化 GPU 验收 provenance

**改动原因**：旧矩阵虽生成环境文件，但未记录 Git 工作区状态、compute capability、CMake CUDA cache 和逐案例控制台输出，服务器结果复制后仍可能缺少构建或失败上下文。
**改动内容**：
- `scripts/run_gpu_validation.py`：环境清单新增完整 commit、dirty 状态、compute capability、CMake/C++ 版本与关键 CUDA cache。
- 每个案例保存独立 `command.txt`、option/simulation 配置快照和 `console.log`。
- 根目录保存矩阵调用命令，数值汇总逻辑与严格失败状态保持不变。
**预期效果**：最新 GPU gate 的每个数字均可回溯到代码、构建配置、输入和原始运行日志。
**运行说明影响**：无新增必选参数，但必须完整下载输出目录而非只复制总表。

### 2026-09-10 — 迭代 #4：固定 profiler 代表性负载

**改动原因**：直接对 10M×256 正式性能配置执行 ncu full replay 成本过高，且不同手工配置会削弱 profile 的可比性。
**改动内容**：`configs/simulation_profile.txt` 固定 1,048,576 路径、256 步、control variate、Philox 请求和关闭 Greeks，作为 Asian CUDA kernel 的 ncu/nsys 代表性负载。
**预期效果**：性能基准仍使用 10M×256，profiler 则使用固定中等规模定位瓶颈，二者边界在报告中明确区分。
**运行说明影响**：ncu/nsys 命令统一引用该配置。

### 2026-09-10 — 迭代 #4：新增最终报告实验总入口

**改动原因**：正确性矩阵、sweep、性能重复和 profiler 原先需要人工逐条运行，容易漏掉阶段、复用旧目录或丢失失败日志。
**改动内容**：
- `scripts/run_task5_experiments.py`：新增 `quick`/`report` 固定 preset，正式 preset 覆盖 European/Asian MC、randomized Halton 和 10M×256 Asian 性能实验，可选 Barrier 与 ncu/nsys。
- 正式运行默认拒绝 dirty worktree 和非空输出目录；每个阶段流式显示并保存 console log，任一核心阶段失败即停止后续结论。
- 自动输出 `environment.txt`、`experiment_plan.json`、`manifest.json`、`artifact_index.md` 与合并后的 `report_tables.md`。
**预期效果**：一次命令可得到训练营报告所需的可复现证据树，quick 结果与 report 结果在 preset 层面显式区分。
**运行说明影响**：本地仅允许 `--backend cpu --skip-validation --preset quick --allow-dirty` 检查流水线；正式数字必须在 clean commit 的 CUDA 服务器上使用 `--preset report`。

**收敛图扩展**：新增 `scripts/generate_report_plots.py`，不依赖第三方包，从 `aggregate.csv` 生成各方法的 log-log 绝对误差与平均 SE 曲线；总入口新增 `report_plots` 阶段并把 SVG 索引合并进报告素材。缺失值不会被绘制为零。

### 2026-09-10 — 迭代 #4：同步 README 运行契约

**改动内容**：`README.md` 增加独立 CPU benchmark 参数、一键 quick/report 命令、正式 preset 的规模说明、证据目录结构和中等规模 profiler 配置。
**证据边界**：quick preset 明确禁止进入最终报告；report preset 必须从 clean commit 运行，且 ncu/nsys 缺失时不得伪称已完成 profiling。

### 2026-09-10 — 迭代 #4：同步报告证据门禁

**改动内容**：`REPORT.md` 标明当前仍待最新 RTX 5090 report preset，补充数值 reference/CPU 性能基线分离原则，并新增最终 evidence manifest 清单。
**预期效果**：报告撰写不能再把 smoke/quick、人工抄录或缺失 profiler 产物误写为正式结果。

### 2026-09-10 — 迭代 #4：本地代码审查与流水线复验

**验证结果**：
- Release CPU CTest：1/1 通过。
- Python `compileall`：通过。
- CPU quick preset：European MC、Asian MC、Barrier MC、European QMC、Asian performance、SVG plots 共 6/6 阶段通过。
- SVG 视觉检查：坐标、图例、log 轴和三种方差缩减曲线正常。
- 无 CUDA stub 演练：8 个案例均按预期失败，但 `summary.csv`、环境文件和逐案例 console log 完整保留，失败证据路径通过检查。
- 最终聚合字段复验：raw/aggregate/Markdown 均保留 kernel、reduction、CPU baseline 和 speedup 字段；Release CTest 再次 1/1 通过，`git diff --check` 通过。
**证据边界**：该 quick 运行位于临时目录，仅验证代码可运行；本机没有 NVIDIA GPU，因此没有生成或声称任何新 GPU 数值。

## Run Instructions — Iteration 4

### Local CPU regression

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
python3 -m compileall -q scripts
```

用于验证 C++ CPU 回归与 Python 实验驱动器语法，不产生 GPU 结论。

### RTX 5090 CUDA build

```bash
CUDA_ROOT_DIR=/usr/local/cuda-12.8
export PATH="$CUDA_ROOT_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT_DIR/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cmake -S . -B build-cuda128 \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER="$CUDA_ROOT_DIR/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
  -DCUDAToolkit_ROOT="$CUDA_ROOT_DIR" \
  -DCMAKE_CUDA_FLAGS=-lineinfo \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build-cuda128 --parallel "$(nproc)"
ctest --test-dir build-cuda128 --output-on-failure
```

### Quick CUDA pipeline check

```bash
TASK5_COMMIT="$(git rev-parse --short HEAD)"
python3 scripts/run_task5_experiments.py \
  --binary build-cuda128/derivatives_pricer \
  --preset quick --backend cuda --include-barrier \
  --output-dir "outputs/task5_quick_${TASK5_COMMIT}"
```

运行小规模 correctness/sweep/performance/plot 流水线。该目录只用于排错，不能写入最终报告。

### Final report experiments

```bash
TASK5_COMMIT="$(git rev-parse --short HEAD)"
python3 scripts/run_task5_experiments.py \
  --binary build-cuda128/derivatives_pricer \
  --preset report --backend cuda --include-barrier \
  --output-dir "outputs/task5_report_${TASK5_COMMIT}"
```

依次运行最新 CPU/GPU gate、四级路径规模的 MC/QMC 收敛和方差缩减、7 次 10M×256 Asian 性能实验，并生成 Markdown 表及 SVG 图。输出目录必须为空且 Git worktree 必须 clean。

### Optional Nsight profiles

```bash
python3 scripts/profile_cuda.py --tool nsys \
  --binary build-cuda128/derivatives_pricer \
  --option configs/asian_call.txt \
  --simulation configs/simulation_profile.txt \
  --output-dir "outputs/task5_report_${TASK5_COMMIT}/profile_nsys"

python3 scripts/profile_cuda.py --tool ncu \
  --binary build-cuda128/derivatives_pricer \
  --option configs/asian_call.txt \
  --simulation configs/simulation_profile.txt \
  --output-dir "outputs/task5_report_${TASK5_COMMIT}/profile_ncu"
```

分别生成 nsys 与 ncu 原始 profile，并在子目录保存完整命令、结果和性能日志；工具不可用时保留为报告待办，不能捏造分析。

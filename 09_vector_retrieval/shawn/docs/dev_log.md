# Development Log

> 最终状态说明（2026-09-19）：以下条目为开发历史，早期“GPU 待测”记录保留原时点。
> 单卡 RTX 5090 的正式验证、百万级检索实验和 Nsight Systems 分析现已汇入
> `REPORT.md`。FAISS 对照和真正并发的批量 GPU 调度仍未完成。

## Project Overview

GPU exact 与 IVF-Flat 向量检索、CPU ground truth 和性能评估工程。

## Implementation Progress

- 初版：CPU exact、CUDA exact、IVF build/load/search、数据生成和 sweep 已完成。
- 当前：迭代 #1，补 CUDA CTest 与服务器 CPU/GPU 验收矩阵。

## Development Log Entries

### 2026-09-20 — 提交前整理

精简重复注释和脚本文档字符串，保留同分 ID 排序、度量定义、索引格式及逐查询
执行方式的说明。未改变检索逻辑、配置或实验结果。
独立提交目录以 Release、CUDA=OFF 构建，CTest 2/2 通过；其中验证脚本测试不执行 GPU。

### 2026-08-25 13:31 — 迭代 #1：诊断与设计冻结

**改动原因**：本地 CPU CTest 已通过，但测试文件没有 CUDA gate，现有 outputs 仅为 10k × 128 smoke。
**改动内容**：
- `docs/user_requirements.md`：记录双后端、验证位置和最终规模要求。
- `docs/idea_report.md`：固定检索方法与实验问题。
- `docs/implementation.md`：定义 CUDA CTest 和服务器验证矩阵。
**预期效果**：CUDA 实现进入自动化正确性门禁，服务器运行可直接形成报告证据。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 否

## Known Issues

- 尚未在真实 GPU 上运行 CUDA CTest、1M 数据实验和 ncu/nsys。
- 当前 CUDA exact 仍逐 query 全量排序，不是最终性能实现。

### 2026-08-25 14:12 — 迭代 #1：CUDA 正确性进入 CTest

**改动原因**：CUDA exact、IVF assignment 与 candidate rerank 原先没有自动化测试入口。
**改动内容**：
- `tests/test_vector_search.cpp`：新增条件式 CUDA gate，覆盖 L2/IP/Cosine exact、非整 batch、GPU-built IVF index 和 CPU/CUDA rerank 对照。
**预期效果**：CUDA 服务器上的 CTest 会同时约束 Recall@K、rank ID agreement 与 score error。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 否

### 2026-08-25 14:20 — 迭代 #1：新增服务器 CPU/GPU 验收矩阵

**改动原因**：现有 sweep 面向性能探索，不会自动生成跨 metric/dtype/mode 的硬性正确性结论。
**改动内容**：
- `scripts/run_gpu_validation.py`：生成 3 metrics × 2 dtypes 数据，执行 K=1/10/50/100 exact 和 GPU-built IVF 的 CPU/GPU gate，保存哈希、命令、环境、CSV 与 Markdown 草表。
**预期效果**：服务器一条命令即可验证 exact、GPU assignment 与 IVF rerank，并为报告保留完整证据链。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 否

### 2026-08-25 14:23 — 迭代 #1：同步服务器验收说明

**改动原因**：新增脚本需要明确运行顺序、产物和 smoke/formal scale 边界。
**改动内容**：
- `README.md`：新增 CUDA CTest + CPU/GPU matrix 命令、覆盖范围和产物说明。
- `REPORT.md`：规定从自动产物填表并保留哈希/环境/命令，同时强调正式 1M 规模仍需另跑。
**预期效果**：不会把 4096-vector correctness gate 误写成任务要求的最终性能实验。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 否

### 2026-08-25 14:25 — 迭代 #1：格式化 CUDA gate 测试

**改动原因**：新增条件测试需保持项目 C++ 风格一致。
**改动内容**：
- `tests/test_vector_search.cpp`：执行 clang-format。
**预期效果**：降低格式噪声并便于服务器失败时定位。
**文档同步**：idea_report.md 否 | implementation.md 否 | configs/ 否

### 2026-08-25 14:26 — 迭代 #1：保护 IVF 训练样本下界

**改动原因**：自定义 `--clusters` 大于 2048 时，固定训练样本数可能小于 nlist。
**改动内容**：
- `scripts/run_gpu_validation.py`：令 training samples 至少等于 nlist，同时不超过数据库规模。
**预期效果**：合法的自定义矩阵参数不会因脚本内部默认值产生无效 IVF 配置。
**文档同步**：idea_report.md 否 | implementation.md 是 | configs/ 否

### 2026-08-25 14:29 — 迭代 #1：补 CUDA 排序 tie 与零向量门禁

**改动原因**：报告要求的重复分数确定性策略和 cosine 零向量此前只在 CPU 测试中覆盖。
**改动内容**：
- `tests/test_vector_search.cpp`：CUDA gate 新增 L2 重复分数排序对照，以及 cosine 零向量的有限 score 检查。
**预期效果**：GPU exact 的 tie policy 与零范数语义被明确纳入服务器 CTest。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 否

### 2026-08-25 14:30 — 迭代 #1：格式化新增边界 gate

**改动原因**：保持条件测试的 C++ 排版一致。
**改动内容**：
- `tests/test_vector_search.cpp`：再次执行 clang-format。
**预期效果**：新增 tie/zero-vector 代码保持可读。
**文档同步**：idea_report.md 否 | implementation.md 否 | configs/ 否

### 2026-08-25 14:34 — 迭代 #1 结果

| 指标 | 改动前 | 改动后 | 变化 |
|---|---:|---:|---|
| Release CPU CTest | 1/1 通过 | 1/1 通过 | 保持 |
| UBSan CPU CTest | 1/1 通过 | 1/1 通过 | 保持 |
| CUDA 条件测试语法 | 无 gate | host C++ 条件分支检查通过 | 新增 |
| 10k×128 CPU exact smoke | Recall@10=1 | Recall@10=1，QPS≈1570 | 正确性保持 |
| 10k×128 CPU IVF smoke | Recall@10=1 | Recall@10=1，QPS≈3902 | 正确性保持 |
| 服务器验收矩阵 | 无 | 30 个默认 gate | 新增，待 GPU 运行 |

**结论**：迭代有效。CPU exact/IVF 与索引回读回归通过，CUDA gate 已进入构建路径且验证脚本静态检查通过；真实 CUDA CTest、30-case matrix、1M 正式实验和 profiler 仍必须在服务器执行。上述 smoke QPS 仅用于确认程序运行，不用于最终性能结论。

## Run Instructions

- 本地 CPU：`cmake --build build -j && ctest --test-dir build --output-on-failure`
- 服务器 gate：`python3 scripts/run_gpu_validation.py --binary build/vector_search --output-dir outputs/gpu_validation`
- 正式实验：另行生成 N≥1,000,000、D=128、Q≥1,000 数据，再运行 sweep 与 ncu/nsys。

### 2026-09-09 — 迭代 #2：GPU 首跑前验证链审计

**改动原因**：现有 IVF gate 只比较 CPU/GPU 各自相对 ground truth 的聚合质量值；两个聚合误差可能相同，但具体 ID/score 仍可能不同。IVF 候选按 coarse-list 顺序拼接，也没有显式保证 CUDA 稳定排序在重复 score 时采用全局较小 ID。
**改动范围**：增强 tie policy 和验收证据，不改变距离公式、IVF 候选集合、K-means 或性能容差。
**预期效果**：CPU/GPU 输出逐项相同时才通过 parity gate；任何单项执行失败仍能得到完整矩阵汇总，便于 RTX 5090 首轮定位。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 否

### 2026-09-09 — 迭代 #2：直接结果门禁与失败汇总实现

**改动内容**：
- `src/ivf.cpp`：coarse probe 后按 vector ID 整理候选，固定 CUDA stable radix sort 的重复分数 tie-break。
- `scripts/run_gpu_validation.py`：逐 query/rank 解析 CPU/GPU result，新增 direct rank agreement、mean/max score error；清理陈旧产物；单项异常转为结构化 failure row 后继续。
- `tests/test_vector_search.cpp`：新增跨倒排表、乱序 ID、全重复 score 的 IVF tie 回归。
- `README.md`、`REPORT.md`：同步直接门禁和报告证据要求。
**预期效果**：30 项默认矩阵既验证相对 CPU exact 的质量，也验证两后端实际输出逐项一致。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 否

### 2026-09-09 — 迭代 #2 本地结果

| 指标 | 结果 |
|---|---:|
| Release CPU CTest | 1/1 通过 |
| UBSan CPU CTest | 1/1 通过 |
| Python py_compile/compileall | 通过 |
| 直接 result helper | identical file = rank 1.0、mean/max error 0 |
| CPU-as-CUDA 成功编排演练 | 30/30 case 通过，direct rank 1.0、max score error 0 |
| 无 CUDA 失败矩阵 | 30/30 case 均执行并写入 failure row |

**结论**：CPU/索引回归通过，验证驱动器在本机无 CUDA 的预期失败条件下仍生成 30 行、28 列完整汇总。ASan 在当前 macOS 沙箱未完成，不计为通过证据。真实 CUDA compile、CTest 和 30 项数值门禁仍需在 RTX 5090 上执行。

### 2026-09-10 — 迭代 #3：百万级正式实验闭环设计

**改动原因**：现有代码功能面已覆盖题目，但只有小规模 smoke/gate 和手工 sweep 命令，无法一键完成 `N=10^6, D=128, Q=1000`、重复统计、图表、profiler 与报告归档，也缺少长实验失败后的安全续跑。
**设计决策**：
- report preset 固定百万 FP32/L2 数据、Q=1000、nlist=4096，不根据中间结果调参。
- GPU exact/IVF 对全部 query 计时；CPU exact 对前 20 条 query 计算 ground truth/reference QPS，并把样本量写入每行。
- Exact 控制 K，IVF 分别控制 nprobe 与 batch；每配置独立 warm-up，保留所有 repetition。
- ncu 聚焦自研 score kernel，nsys 捕获包含 CUB/传输/同步/coarse CPU 的完整时间线。
**文档同步**：user_requirements.md 是 | idea_report.md 是 | implementation.md 是 | configs/ 由 driver 冻结生成

### 2026-09-10 — 迭代 #3：正式驱动器、统计与报告资产实现

**改动内容**：
- `scripts/run_task9_experiments.py`：新增 validation、data、index build、exact Top-K、IVF nprobe、IVF batch、plots 和可选 profiler stages；冻结 plan/manifest，校验 expected files，并支持同 plan `--resume`。
- `scripts/run_sweep.py`：同时支持 exact/IVF，加入 per-config warm-up、逐次 console/raw logs、命令、环境和输入/索引 SHA-256；aggregate 新增 QPS std、CPU reference 样本、candidate/memory/speedup，自动生成报告表。
- `scripts/generate_data.py`：NumPy 路径改为分块 FP32 生成，降低百万规模峰值内存；新增 metadata 与 hashes。
- `scripts/profile_vector_search.py`：分别封装 ncu/nsys，并固定结果、性能、质量、命令和 profiler 原始报告位置。
- `scripts/generate_report_plots.py`：基于 aggregate mean 生成 7 张 publication-style SVG/PNG，不挑选最佳 repetition。
- `README.md`、`REPORT.md`：新增一键运行、计时/内存口径、完整报告骨架和逐项 requirement traceability；明确 batch 尚未并发、FP16 仅为存储格式。
**预期效果**：RTX 5090 上一次正式命令即可生成满足题面规模和报告字段的完整证据目录，失败后无需重跑已验证 stage。
**文档同步**：idea_report.md 是 | implementation.md 是 | README/REPORT 是

### 2026-09-10 — 迭代 #3 本地结果

| Gate | Result |
|---|---|
| Release CPU build/CTest | 1/1 通过 |
| Python `py_compile` | 6 个脚本通过 |
| Quick CPU end-to-end orchestrator | data/build/exact/nprobe/batch 5/5 stages 通过 |
| `--resume` frozen-plan check | 5/5 expected files 验证后跳过 |
| NumPy FP16/Cosine generator branch | metadata/hash 与二进制生成通过 |
| Report plots | 7 SVG + 7 PNG 生成；代表图人工检查无标题/标注重叠 |
| No-CUDA validation failure path | 24/24 自定义 cases 均保留 failure row，最终按预期非零退出 |

**结论**：正式实验基础设施已完成本地 CPU 与编排验证。这里没有产生任何可用于最终报告的 GPU 数字；CUDA compile/CTest、默认 30-case matrix、百万规模三组 sweep 与 ncu/nsys 仍需在 RTX 5090 上执行后才能关闭任务九。

### 2026-09-10 — 迭代 #4：RTX 5090 correctness gate 诊断与设计修订

**改动原因**：首轮 30-case matrix 有 28 项通过；两个 Exact/K=100 案例 Recall@K=1、最大 score error 小于 `4.8e-7`，但严格逐 rank agreement 因 1–2 组近同分结果换位而略低于 1。原 gate 把数值等价排序与错误 ID 集合等同处理，严于跨 CPU/CUDA 浮点实现的实际一致性契约。
**改动内容**：
- `docs/idea_report.md`：把 CPU/GPU parity 定义为严格 ID 集合、按 ID 对齐的 score 和容差感知顺序三重门禁，严格 rank 保留为诊断值。
- `docs/implementation.md`：冻结 direct comparator 新字段、通过条件及确定性 exact-tie CTest 的职责边界。
**预期效果**：仍会拒绝漏邻居、错邻居、显著错序和超阈值 score，同时允许可由浮点舍入解释的近同分邻居换位。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 否

### 2026-09-10 — 迭代 #4：容差感知 CPU/GPU direct comparator 实现

**改动原因**：原 comparator 按 rank 比较 score，换位时没有按同一 vector ID 对齐；硬性要求 strict rank=1 也不能区分近同分换位与实际 Top-K 错误。
**改动内容**：
- `scripts/run_gpu_validation.py`：新增 strict rank、ID set、tolerance-aware rank 和 per-ID score 四类指标；Exact/IVF gate 改用后三类硬条件，并扩展 CSV、失败行和 Markdown 表。
- `tests/test_gpu_validation.py`：新增纯 Python 回归，覆盖完全一致、容差内换位、显著错序、错误 ID 集合及重复 ID。
- `CMakeLists.txt`：检测到 Python interpreter 时把 comparator 回归加入 CTest。
- `README.md`、`REPORT.md`：同步新门禁语义与报告字段，避免把 99.9% strict rank 误写成检索错误或隐藏不一致。
**预期效果**：RTX 5090 的两个已知 K=100 近同分案例应通过；ID 集合变化、超容差顺序变化和 score 超差仍保持失败。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 否

### 2026-09-10 — 迭代 #4 本地结果

| Gate | Result |
|---|---|
| Python comparator unit tests | 7/7 通过 |
| Release CPU CTest | 2/2 通过（C++ core + Python comparator） |
| Python `py_compile` | 通过 |
| Git whitespace check | 通过 |

**结论**：新门禁在合成近同分换位上按预期通过，并继续拒绝不同 ID 集合与超容差错序。本地没有 CUDA，30-case RTX 5090 定点复测仍是关闭本轮迭代的必要证据。

### 2026-09-10 — 迭代 #4 RTX 5090 结果

| Gate | Result |
|---|---|
| Default CPU/GPU correctness matrix | 30/30 通过 |
| Exact/IVF Recall@K | 全部为 1 |
| Top-K ID set agreement | 30/30 均为 1 |
| Tolerance-aware rank agreement | 30/30 均为 1 |
| 最大 per-ID score error | `1.53e-5`，低于 `2e-4` 容差 |
| Quick end-to-end stages | 7/7 通过 |

两个已知近同分案例的 strict rank 分别保持为 `0.999375`（L2/FP16/K=100）和 `0.99875`（Cosine/FP32/K=100），同时 ID set 与 tolerance-aware rank 均为 1，符合修订后的数值一致性契约。Quick 仅使用 N=4096、D=64、Q=32，不作为最终性能数据。

**结论**：迭代 #4 的真实 CUDA gate 已关闭；下一步进入 N=1,000,000、D=128、Q=1,000 正式 sweep 与 profiler。

### 2026-09-10 — 迭代 #5：服务器绘图字体可移植性设计

**改动原因**：Quick 的 7 张图均成功生成，但服务器缺少 Arial/Helvetica，console log 出现大量 `findfont` 回退警告。
**改动内容**：
- `docs/implementation.md`：规定 DejaVu Sans 优先的跨服务器字体策略，不改变图表数据编码。
**预期效果**：保留 publication-style sans-serif 视觉，同时消除正式运行中的重复字体警告。
**文档同步**：idea_report.md 否 | implementation.md 是 | configs/ 否

### 2026-09-10 — 迭代 #5：无桌面字体服务器的确定性回退实现

**改动原因**：Arial/Helvetica 位于字体列表首位时，Matplotlib 会为每个文本对象重复搜索不存在的字体。
**改动内容**：
- `scripts/generate_report_plots.py`：使用通用 sans-serif family，并将 Matplotlib 内置 DejaVu Sans 置于候选首位；保留 Arial/Helvetica 作为后备选项。
**预期效果**：服务器无需安装额外字体即可安静地导出相同的 7 张 SVG/PNG。
**文档同步**：idea_report.md 否 | implementation.md 是 | configs/ 否

### 2026-09-10 — 迭代 #5 本地结果

| Gate | Result |
|---|---|
| Matplotlib isolated font-cache smoke | PNG 成功生成，无 `findfont` 告警 |
| 输出人工检查 | 标题、坐标和线条正常，无裁切 |
| Python `py_compile` | 通过 |
| Git whitespace check | 通过 |

**结论**：字体回退修复有效；服务器更新后可直接开始正式规模实验，无需安装 Arial/Helvetica。

### 2026-09-10 — 正式规模与 Nsight Systems 结果存档（commit `b3e629a`）

**证据来源**：用户提供的 RTX 5090 服务器表格与 nsys CSV 输出；完整转录见 `docs/results/task9_report_b3e629a.md`，原始 `.nsys-rep`、SQLite、CSV 和 raw logs 仍位于服务器 `outputs/task9_report_b3e629a/`。

| Observation | Result |
|---|---|
| Formal scale | N=1,000,000, D=128, Q=1,000，GT Q=20 |
| Exact K=100 | 1486.49 QPS，P99=0.530 ms，sampled hot-query speedup=144.83× |
| IVF nprobe=1 | Recall@10=1，1327.59 QPS，P99=0.635 ms |
| IVF nprobe 1→256 | candidates 467.4→220088.4；QPS -92.62%；P99 17.94× |
| IVF batch 1/32/128 | QPS 范围差 2.02%，未观察到稳定 batch 收益 |
| IVF GPU kernel share | CUB radix-sort kernels 67.0%；score kernel 32.9% |
| IVF H2D | 728.576 MB/2001 calls；单次 512 MB 数据库上传耗时 162.737 ms |
| ncu | Exact/IVF 均因 `ERR_NVGPUCTRPERM` 未生成 `.ncu-rep` |

**证据边界**：Recall 曲线在当前 clustered synthetic 数据上从 nprobe=1 即饱和，不能据此声称对困难数据也保持相同行为；ncu 硬件计数器缺失时不得报告 DRAM/L2 throughput、achieved occupancy 或 stall reason。现有 nsys 证据支持优先分析全量 radix sort、冷启动数据库上传、逐 query candidate staging 和 host-side coarse selection。

**权限复核**：同一环境中 Task 5 也只有 `profile_ncu.console.log`，没有 `.ncu-rep`；`RmProfilingAdminOnly: 1` 且容器未获得 `cap_sys_admin`。因此这是服务器/容器级性能计数器权限限制，不是 Task 9 内核问题，也不能把 Task 5 的 console log 视为 ncu 成功导出。

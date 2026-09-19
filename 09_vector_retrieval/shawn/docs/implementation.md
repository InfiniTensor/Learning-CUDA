# Implementation Design

## Current Files

| 文件 | 责任 |
|---|---|
| `src/search_cpu.cpp` | exact/candidate CPU ground truth 与质量指标 |
| `src/search_cuda.cu` | CUDA score、CUB sort 与 IVF assignment |
| `src/ivf.cpp` | IVF 训练、索引组织、probe 与 rerank |
| `tests/test_vector_search.cpp` | CPU/I/O/索引及条件式 CUDA 门禁 |
| `scripts/run_gpu_validation.py` | 服务器端 exact/IVF CPU-GPU 验收矩阵 |
| `scripts/run_sweep.py` | exact/IVF warm-up、重复实验、聚合与可追溯日志 |
| `scripts/run_task9_experiments.py` | 正式规模一键编排、断点续跑与产物索引 |
| `scripts/profile_vector_search.py` | exact/IVF 的 ncu/nsys 原始采集 |
| `scripts/generate_report_plots.py` | 从 aggregate.csv 生成无第三方绘图库依赖的 SVG |

## Iteration 1 Design

1. 为 CUDA exact、candidate rerank 和 IVF assignment 增加条件式 CTest，对照 CPU 的 Recall、rank agreement 与 score tolerance。
2. 新增服务器验证脚本，覆盖三种 metric、FP32/FP16、多个 K，以及 IVF build/search；失败时非零退出。
3. 自动保存环境、summary CSV 与 Markdown 报告草表；README/REPORT 明确 smoke 与正式规模边界。

## Iteration 2 Design

1. IVF coarse probe 完成后对 candidate IDs 排序，令 CPU heap 与 CUDA 稳定 radix sort 在重复分数处共享全局 ID tie-break。
2. 验收脚本解析 CPU/GPU result 文件，记录逐 rank ID agreement、mean/max score error，并将其纳入 exact 与 IVF 最终 gate；quality log 仍用于各后端相对 CPU exact 的 Recall 证据。
3. 每次运行先删除该案例的陈旧输出；单项命令失败时保留 execution failure 行并继续剩余矩阵，最终统一非零退出，使第一次服务器调试也能留下完整失败分布。

## Result Format

每个案例保留 database/query/config/index/result/performance/quality；总目录保存 `summary.csv`、`environment.txt`、`commands.txt` 与 `report_table.md`。汇总同时记录 CPU/GPU 直接 rank agreement、mean/max score error 和各自相对 CPU exact 的质量指标。

## Iteration 3 Design

1. 正式 preset 固定生成 `1,000,000 x 128` FP32/L2 数据库和 1,000 条 query；quick preset 只验证流水线。
2. 同一持久化 IVF-Flat 索引用于所有 sweep，分别控制变量扫描 Top-K、`nprobe` 和 batch size；每个配置先 warm-up，再保留全部 repetition 原始日志。
3. exact CUDA 对所有 1,000 queries 计时；质量日志中的 CPU exact 仅运行 `quality_queries` 子集，并在表中同时记录样本量、reference QPS 与 speedup，避免口径混淆。
4. 正确性矩阵、正式数据生成、GPU 建索引、exact sweep、IVF `nprobe` sweep、IVF batch sweep、绘图和可选 profiler 均成为独立 stage。manifest 记录状态、耗时、命令和产物目录；`--resume` 只跳过同一 plan 中已通过且产物仍存在的 stage。
5. profiler 分 exact/IVF 两种 mode：ncu 聚焦 score kernel 并限制采集 launch 数，nsys 保留完整 CPU/GPU 时间线，以区分距离计算、radix sort、coarse selection、传输与同步开销。

## Formal Experiment Protocol

| 因素 | report preset | 说明 |
|---|---:|---|
| 数据规模 | N=1,000,000, D=128, Q=1,000 | 满足题面基础测试下界 |
| 输入 | FP32, L2, clustered synthetic | seed 与 SHA-256 固化 |
| Exact K | 1, 10, 50, 100 | 四种均覆盖 |
| IVF nlist | 4096 | 索引只构建一次并从文件回读 |
| IVF nprobe | 1, 4, 16, 64, 256 | K=10、batch=128 控制变量 |
| IVF batch | 1, 32, 128 | K=10、nprobe=16 控制变量 |
| 重复 | exact 3 次；IVF 5 次 | 每组另有 1 次 warm-up，不进入统计 |
| 质量样本 | 前 20 queries | 每次均以 CPU exact 计算，不与 Q=1,000 吞吐口径混写 |

## Iteration 3 Output Schema

- `environment.txt`：commit、dirty state、GPU/driver/compute capability、CUDA/CMake/C++/Python。
- `experiment_plan.json`、`manifest.json`：冻结 preset、stage 命令、返回码与 wall time。
- `data/metadata.json`、`data/hashes.sha256`：数据生成参数、大小和内容哈希。
- `*/commands.txt`、`summary.csv`、`aggregate.csv`、`report_table.md`：逐次命令、原始行、统计量和报告草表。
- `plots/*.svg`、`plots/plots_index.md`：Top-K、nprobe、batch 对 QPS/P99/Recall 的图。
- `profile_*/command.txt` 与 `.ncu-rep`/`.nsys-rep`：独立 profiler 证据。

## Iteration 4 Design

RTX 5090 首轮 30-case correctness matrix 中，28 项严格通过；`exact/L2/FP16/K=100` 与 `exact/Cosine/FP32/K=100` 的 Recall@K 均为 1，最大 score error 分别只有 `4.8e-7` 与 `3.57e-7`，但少量近同分邻居相邻换位，使严格 rank agreement 分别为 `0.999375` 与 `0.99875`。

修订后的 direct comparator 同时输出四类证据：严格逐 rank ID agreement、逐 query Top-K ID set agreement、score-tolerance-aware rank agreement，以及按同一 vector ID 对齐后的 mean/max score error。硬门禁要求 set agreement=1、tolerance-aware rank agreement=1、max per-ID score error 不超过 `--score-tolerance`；严格 rank agreement 保留在 CSV/报告中但不单独否决数值等价换位。重复 score 的较小 ID 优先策略继续由确定性 CUDA CTest 严格验证。

## Iteration 5 Design

RTX 5090 quick run 已生成全部 7 张 SVG/PNG，但服务器没有 Arial/Helvetica，Matplotlib 在每次文字布局时重复输出字体回退警告。绘图样式改用 Matplotlib 随包提供的 DejaVu Sans 作为首选 sans-serif，再保留 Arial/Helvetica 为可选回退；不改变数据、坐标、配色或导出格式。正式实验 console log 因此只保留真正的实验和错误信息。

## Implementation Order

设计文档 → 实验/Profiler 驱动器 → 报告图表 → README/REPORT → 本地 CPU/UBSan/Python 静态验证 → RTX 5090 正式实验。

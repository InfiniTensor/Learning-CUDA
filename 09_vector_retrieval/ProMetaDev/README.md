# GPU 向量检索引擎（CUDA）

2026 夏季训练营 CUDA 方向项目阶段 —— 选题九。

在**普通 CUDA GPU** 上实现一个简化但完整的向量检索引擎：GPU **精确检索 baseline**
（暴力扫描 + Top-K）与 **IVF-Flat 近似检索**（k-means 粗量化 + 倒排桶扫描），
并给出 recall@K / QPS / P50 / P99 / 显存 / 相对 CPU 加速比等完整指标。

## 1. 功能与要求对照

| 题目要求 | 实现情况 |
|---|---|
| GPU 精确检索 baseline（L2 或内积至少一种） | 支持 **L2 / 内积 / cosine** 三种度量 |
| 精确检索结果需与 CPU 参考一致 | 每个 query 的 top-K **id 集合完全一致**（1e6 规模实测 0/1000 不一致） |
| Top-K 按距离/相似度排序 | 已排序；并列时以 id 小者优先（全序，与 CPU 一致） |
| 支持批量查询 | `batch_size` 可配置，1→512 扫描（QPS 提升约 34 倍） |
| 至少实现一种近似检索 | **IVF-Flat**（k-means 粗量化 + 倒排表 + nprobe 扫描） |
| 需支持从文件加载已建索引 | 索引落盘为 `.bin`（聚类中心 + 倒排表），查询时直接加载，不重建 |
| 与精确检索对比 recall@K | 内置 recall@K 与平均距离误差统计 |
| 支持 K = 1,10,50,100 中至少三种 | **四种全支持**（模板特化） |
| 10^6 向量 / 128 维 / 1000 查询 | **已实测**（见 report.md） |
| 平台适配 | 默认英伟达；未使用任何新架构特性，可移植到国产平台 |

## 2. 目录结构

```
.
├── CMakeLists.txt
├── build.sh                  # 一键构建（自动探测 CUDA 环境）
├── .clang-format / .gitignore
├── include/
│   ├── types.h               # 配置、向量集、结果、索引、性能统计
│   ├── io.h                  # 向量库/查询/参数/索引/结果 的读写
│   ├── metrics.h             # recall@K、距离误差、延迟分位、结果比对
│   ├── search.h              # CPU 参考与 GPU 接口
│   ├── topk_merge.cuh        # 精确检索与 IVF 共用的两级 Top-K 归并
│   └── normalize.cuh         # cosine 用的行归一化
├── src/
│   ├── main.cpp              # CLI: gen / genq / exact / build / search / bench / selftest
│   ├── io.cpp                # I/O 实现（含 host 端 fp16 软件转换）
│   ├── metrics.cpp           # 指标实现
│   ├── cpu_ref.cpp           # CPU 参考（OpenMP 并行）：精确检索 + k-means
│   ├── exact.cu              # GPU 精确检索 kernel
│   └── ivf.cu                # IVF 建索引（k-means + 倒排表）与查询 kernel
├── configs/                  # 多组量化/检索配置
├── tests/run_tests.sh        # 端到端测试（13 项）
└── report.md                 # 总结报告
```

## 3. 构建

```bash
bash build.sh          # 默认 sm_90，可传参覆盖：bash build.sh 89
```

## 4. 使用

```bash
# 0) 自检
./build/vs selftest

# 1) 生成数据（kind = uniform | normal | clustered）
./build/vs gen clustered 1000000 128 outputs/data/base.txt --nlist 1024 --seed 7
./build/vs genq outputs/data/base.txt 1000 outputs/data/q.txt --kind clustered

# 2) 精确检索（默认同时跑 CPU 参考并比对）
./build/vs exact outputs/data/base.txt outputs/data/q.txt configs/exact_k10.cfg outputs/gt/exact
#    1e6 规模可加 --no-cpu 跳过 CPU 参考（耗时较长）

# 3) 建 IVF 索引（结果落盘，可复用）
./build/vs build outputs/data/base.txt configs/ivf_k10.cfg outputs/index/ivf.bin

# 4) IVF 查询 + 召回率
./build/vs search outputs/data/base.txt outputs/data/q.txt outputs/index/ivf.bin \
                configs/ivf_k10.cfg outputs/results/ivf --gt outputs/gt/exact.result

# 5) nprobe / batch_size 参数扫描
./build/vs bench outputs/data/base.txt outputs/data/q.txt outputs/index/ivf.bin \
               configs/ivf_k10.cfg outputs/results/bench --gt outputs/gt/exact.result

# 6) 端到端测试
bash tests/run_tests.sh
```

## 5. 文件格式

**向量库 / 查询集**（文本头 + 二进制数据段）：

```
[header]
num_vectors: 1000000        # 查询文件为 num_queries
dim: 128
dtype: fp32                 # fp32 | fp16
metric: l2                  # l2 | inner_product | cosine（查询文件可省略）

[data]
<num_vectors * dim 个元素，行主序，按 dtype 紧密排列>
```

**索引文件**（`.bin`，二进制）：magic `VSI1` + 头部（nlist/dim/metric/pq_m/n 及各数组长度）
+ 聚类中心 `nlist*dim` 个 float + `list_start`（nlist+1 个 int32）+ `list_ids`（n 个 int32）。

**检索结果**（`.result`，文本）：每行 `query_id rank vector_id distance`。

**日志**：`.log`（性能 + 质量）、`.bench.log`（参数扫描表）。

## 6. 算法要点

- **精确检索**：一个线程块处理一个 query；块内线程按步长扫描全库；
  每线程在本地维护有序 top-K（用第 K 名做阈值早退）；
  两级归并（warp 内 32 路 → 跨 warp）利用"各列表已有序"的性质，
  仅需 K 轮取最优，复杂度 O(K·log)，避免整体排序。
- **IVF 建索引**：从库中抽取子样本训练 k-means（避免在百万级数据上反复全量分配），
  每轮为 分配 → 计数 → 前缀和 → 散列倒排表 → 更新聚类中心；最后对全量数据分配一次。
- **IVF 查询**：Phase A 在共享内存中计算 query 到全部聚类中心的距离并选出 nprobe 个；
  Phase B 扫描这些倒排桶中的原始向量（IVF-Flat）并复用同一套 Top-K 归并。
- **一致性**：GPU 与 CPU 参考采用同一"全序"比较规则（分数更优者胜，并列时 id 小者胜），
  且 GPU 端保持与 CPU 相同的浮点累加顺序。

详见 [report.md](report.md)。

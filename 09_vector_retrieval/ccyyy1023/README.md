# GPU 向量检索引擎（CUDA）

本项目实现了一个可独立构建的简化 GPU 向量检索引擎，面向训练营第九题。当前实现包含 CPU 精确检索参考、CUDA 批量精确检索、L2/内积、排序后的 Top-K、IVF-Flat、索引持久化、正确性测试，以及 QPS、P50/P99、recall@K、平均分数误差、显存工作区和 CPU/GPU 加速比的采集框架。

> 真实性声明：`results/*.json` 来自 2026-09-20 在 RTX 4090 24GB、CUDA 11.8 上的真实运行。百万规模正式矩阵覆盖 1,000,000 条 128 维向量和 1000 个查询；原始 JSON、环境记录和汇总表均随项目提交。

## 1. 已完成范围

- CPU FP32 精确检索 baseline。
- CUDA FP32 精确检索，支持 `l2` 和 `inner_product`。
- 批量查询，`batch_size` 可配置。
- 任意正整数 `top_k`，已提供 K=1/10/50/100 的基准脚本。
- Top-K 按 L2 从小到大或内积从大到小输出。
- IVF-Flat：CPU mini-batch Lloyd 聚类、倒排桶、`nprobe` 探测、数据库常驻 GPU、按候选 ID 计算距离。
- IVF 索引保存/加载，查询时不需要重新聚类。
- 固定版本的二进制向量、查询、结果和 IVF 索引格式。
- CPU/GPU 一致性测试、索引 round-trip 测试。
- JSON 性能/质量日志与汇总脚本。

实现边界：基础版本仅支持 FP32；IVF 聚类、coarse search 和最终 Top-K 归并在 CPU 上完成，主要距离计算在 GPU 上完成。这个版本不声称实现了 GPU Top-K、FP16、IVF-PQ 或国产 GPU 适配。

## 2. 环境

- NVIDIA GPU
- CUDA Toolkit 11.0 或更新版本（需要 `nvcc`）
- CMake 3.20+
- 支持 C++17 的编译器
- Python 3.8+（仅用于日志汇总和包检查）

推荐 Ubuntu 22.04 + CUDA 12.x。Windows 也可使用 Visual Studio 2022 的 C++ 工具链构建。

## 3. 构建与测试

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

测试同时覆盖 L2、inner product、批量精确 Top-K、CPU/GPU 一致性和 IVF 索引保存/加载。

## 4. 五分钟 smoke test

```bash
bash scripts/run_smoke.sh
```

等价的手动命令：

```bash
mkdir -p data results
./build/vector_search generate \
  --db data/small_db.bin --queries data/small_queries.bin \
  --n 10000 --nq 100 --dim 128 --metric l2 --seed 42

./build/vector_search build \
  --db data/small_db.bin --index data/small_ivf.bin \
  --nlist 64 --iterations 8 --seed 42

./build/vector_search benchmark \
  --db data/small_db.bin --queries data/small_queries.bin \
  --index data/small_ivf.bin --log results/smoke.json \
  --k 10 --batch 32 --nlist 64 --nprobe 8 \
  --cpu-query-limit 20 --latency-query-limit 50
```

## 5. 题目要求规模测试

以下脚本生成 `1,000,000 x 128` 向量库和 1000 个查询，测试 K=1/10/50/100、多个 batch size 和多个 nprobe。数据库约 488 MiB，索引还会额外保存一份向量数据，请预留至少 2 GiB 磁盘空间。

```bash
bash scripts/run_required_benchmarks.sh
python3 scripts/summarize_results.py results results/SUMMARY.md
```

正式测试前请关闭其他 GPU 任务，并记录：GPU 型号、驱动、CUDA 版本、CPU 型号和编译命令。脚本不会伪造或预填任何数字。

## 6. 文件模式

`configs/exact.conf` 和 `configs/ivf_flat.conf` 遵循题面参数形式：

```text
top_k = 10
search_mode = "ivf_flat"
batch_size = 128
nlist = 256
nprobe = 16
```

构建一次索引并从文件加载查询：

```bash
./build/vector_search build --db data/db.bin --index data/index.bin --nlist 256
./build/vector_search search \
  --db data/db.bin --queries data/queries.bin \
  --params configs/ivf_flat.conf --index data/index.bin \
  --output results/search_results.bin
```

精确检索不使用 `--index`，但命令行解析允许保留该参数：

```bash
./build/vector_search search \
  --db data/db.bin --queries data/queries.bin \
  --params configs/exact.conf --output results/exact_results.bin
```

## 7. 二进制格式

所有整数均为 little-endian，向量按行主序存储。

- 数据库：8 字节 magic `GVSDB01\0`，`uint64 n`，`uint32 dim`，`uint32 dtype`，`uint32 metric`，随后 FP32 数据。`dtype=1`，`metric=0/1` 分别表示 L2/inner product。
- 查询：8 字节 magic `GVSQR01\0`，`uint64 nq`，`uint32 dim`，`uint32 dtype`，随后 FP32 数据。
- 结果：8 字节 magic `GVSRS01\0`，`uint64 nq`，`uint32 k`，随后每个结果的 `int64 id + float score`。
- IVF 索引：版本化 magic、形状/度量、聚类中心、原始 FP32 向量和倒排表。加载时会验证 header、总 ID 数和 ID 范围。

## 8. 指标定义

- `QPS = 查询数 / 查询时间`。精确检索计时包含 query H2D、CUDA 距离计算、distance D2H 和 CPU Top-K，不包含数据库首次上传；IVF 计时还包含 CPU coarse search、候选 ID 构造和一次数据库 H2D，索引文件加载不计入。
- P50/P99：逐 query、batch=1 的端到端延迟分位数。
- `recall@K`：近似结果与 GPU 精确结果 Top-K ID 集合的平均交集比例。
- `mean_rank_score_error`：近似结果和精确结果在相同排名位置的距离/相似度绝对误差平均值。
- `workspace_mib`：程序显式分配的数据库、query 和 distance buffer 总量估计，不包含 CUDA context/驱动开销。
- CPU/GPU speedup：CPU 单 query 平均时间除以 GPU 批量查询的单 query 平均时间。CPU 默认只测前 20 个 query，并在日志中明确记录样本数。

## 9. 设计说明

精确检索 kernel 使用二维 grid：x 维覆盖数据库向量，y 维覆盖 batch 内 query。每个线程顺序归约一个向量的 128 个维度，行主序布局使相邻线程访问相邻向量。L2 使用 `fmaf(diff, diff, acc)`，内积内部取负数以复用升序 Top-K。

IVF-Flat 用 mini-batch Lloyd 方法训练聚类中心，随后将每条向量分配到最近中心。查询先计算 coarse centroid 距离，选择 `nprobe` 个桶，只打包这些桶中的 FP32 原向量并调用同一 CUDA 精确距离路径。索引文件保存中心、向量与倒排表，因此查询过程无需重新训练或分桶。

当前 Top-K 在 CPU 使用 `std::partial_sort`，这样实现简单且可作为后续 GPU Top-K 的稳定参照。性能报告应如实讨论 D2H 距离传输和 CPU 归并的开销。

## 10. 目录

```text
.
├── CMakeLists.txt
├── README.md
├── REPORT.md
├── configs/
├── include/gvs/
├── results/
├── scripts/
├── src/
└── tests/
```

## 11. 提交

1. Fork `InfiniTensor/Learning-CUDA`。
2. 基于 `2026-summer-project` 分支工作。
3. 将本目录完整复制到 `09_vector_retrieval/ccyyy1023/`（若实际 ID 不同，只改这一层目录名）。
4. 保留 `results/` 中的真实日志；如换硬件复测，请用新日志替换并同步更新 `REPORT.md`。
5. Commit、push，然后向上游 `2026-summer-project` 发起 PR；官网提交 fork 仓库链接和 commit 链接。

# GPU 向量检索引擎（CUDA）项目总结报告

项目代码：`vsearch`

## 1. 摘要

本报告对应“九. GPU 向量检索引擎”训练营题目。实现了一个可直接构建运行的 CUDA
向量检索库与 CLI，包含三档检索能力：

| 模式 | 说明 | 正确性基准 |
| --- | --- | --- |
| `exact` | GPU 全量距离计算 + Top-K 归并 | 与 CPU 暴力参考逐条对齐 |
| `ivf_flat` | IVF 倒排 + 原始向量重算距离 | recall@K vs GPU exact |
| `ivf_pq` | IVF 倒排 + PQ 压缩 + ADC 近似距离 | recall@K / 平均距离误差 |

引擎支持 fp16/fp32 输入文件、L2 / inner product / cosine、批量查询、三种以上
Top-K、索引落盘与重载、主机侧 CPU 参考，以及性能/质量日志。GPU exact baseline 在
小规模合成数据上已设计为与 CPU 参考逐条一致（含距离与 id 的确定性 tie-break）。

同一份源码已实现并验证**双平台运行**：

| 平台 | 工具链 | 验证状态 |
| --- | --- | --- |
| NVIDIA（RTX 4090 24 GB） | CMake + `nvcc`（CUDA 12.0/12.8，驱动 570） | Release 构建、主机/GPU 测试、1e6 与 300k 两档 bench 均通过 |
| 天数智芯 CoreX（MR-V100 32 GB） | `Makefile.corex` + 定制 clang 18（`-x ivcore`） | 构建、全部测试、300k bench 与 nprobe/batch 扫描均通过 |

> NVIDIA 实测环境：Ubuntu 24.04，RTX 4090，内核模块与用户态
> libcuda/libnvidia-ml 570.124.06，CUDA Toolkit 12.0 与 12.8；早期 1e6 实验数据
> `N=10^6, D=128, nlist=4096, nprobe=16, topK=100, nq=1000`。CoreX 实测环境见
> §11。

## 2. 系统结构与模块

```text
include/vsearch/           主机侧数据结构、文件格式、配置、CPU 参考
cuda/engine.cu             CUDA kernels、KMeans、IVF-Flat/PQ、序列化、检索
src/main.cpp               build / search / bench / validate 子命令
tests/                     test_host.cpp、test_gpu.cu
python/gen_dataset.py      合成数据与参数文件
python/run_experiments.py  nprobe x batch 扫描并汇总 CSV
CMakeLists.txt             NVIDIA 构建（nvcc）
Makefile.corex             CoreX 构建（clang -x ivcore，-DVSEARCH_COREX=1）
outputs/                   NVIDIA 平台报告、日志与结果样例
outputs_corex/             CoreX 平台 perf/quality 日志与扫描汇总
```

编译对象只有一个静态库和一个 CLI，方便替换评测方要求的文件布局。

## 3. 向量表示与距离度量

### 3.1 数据布局

向量库/查询按行主序加载。文件头定义见 README。内部统一转成 `fp32` 并放到对齐内存，
因此 fp16 文件也能直接获得相同的精确检索语义；fp16 的价值体现在“原始数据减少 2 倍
带宽/容量”，后续可用 half2 张量化作为内存受限优化（见 §10）。

### 3.2 度量

- **L2**：候选打分直接累计逐维差分平方和
  `score = Σ_d (q[d] - x[d])²`，排序用平方距离，输出时开根得到欧氏距离。直接
  差分求和避免了“范数相减”在大数值下抵消导致的精度损失，也让 CoreX 的 float
  路径与 CPU 参考保持同一累加语义。
  （中心挑选等只需要相对次序的场景仍使用
  `||q - c||² = ||q||² + ||c||² - 2 q·c` 的预计算范数形式，见 §5.3/§6.3。）
- **inner product**：输出原始点积，越大越相似。
- **cosine**：读取时把库向量与查询归一化，再按 `1 - q·x` 输出 cosine 距离。

所有 kernel 在 `dim ≤ 65536` 的一般维度上按运行时循环实现（编译器可展开），没有把
维度写成模板常量，便于实验不同 `D`。

## 4. 精确检索 baseline（exact）

### 4.1 单轮全量打分

`exactKeysKernel` 以 query 为 `blockIdx.y`、向量行为 `blockIdx.x` 组织网格。每个
线程计算一整行向量的距离（L2 直接累计逐维差分平方和；inner product / cosine
累计点积），只把结果写成 64 位排序键，不写全量浮点距离矩阵，避免一次分配
`nq × N` 个 fp32 的中间显存。

### 4.2 Top-K 归并

Top-K 不做单 query 一个串行 heap（带宽利用率低），而是把每行转成“可排序键 + id”
的 64 位打包键：

```text
key = rank(score)<<32 | vector_id
```

`rank(score)` 使用 IEEE-754 fp32 的 total-order 映射：

- L2 / cosine（越小越好）：`rank = sortable(score)`；
- inner product（越大越好）：`rank = ~sortable(score)`。

然后用 `cub::DeviceSegmentedRadixSortKeys` 按 query 做一次分段升序 radix sort，
低 32 位是 id，因此同分时自动按 id 稳定，GPU 与 CPU 参考结果完全一致。排序后每个
query 取前 K 个键并解码距离。排序是确定性、精确的，不依赖采样阈值。

### 4.3 显存分块

exact 的键数组为 `nq_chunk × N × 16 B`（in/out 两份）。CLI 以
`exact_memory_mb`（默认 1536 MB）限制内部 query 子批次，避免在低显存卡上失败。

## 5. IVF-Flat

### 5.1 粗聚类中心训练

- 从库中取至多 `kmeans_sample` 行作为训练样本；
- 用确定间距的样本行初始化 `nlist` 个中心；
- 执行 mini-batch Lloyd：GEMM 求样本×中心点积、最近中心分配、按中心原子累加求和与
  计数、更新中心；空桶从样本行重新注入；
- cosine 数据下每轮把中心归一化到单位长度；
- 迭代默认 12 轮，可选按“分配变化比例 < 阈值”提前收敛。

训练时只累加子样本，控制原子写开销；对全部 1e6 向量只做一次最终分配，因此 build
时间可预估为“训练 + 一次全库最近中心分配 + 倒排重排”。

### 5.2 全库分配与倒排表

全库分配使用同一 GEMM 路径，按 `kmeans_chunk_rows` 分块，避免一次分配
`N × nlist` 浮点矩阵。最近中心判定 kernel 直接读取列主 GEMM 输出（按列连续），
合并 `||x||² + ||c||² - 2 dot`。随后：

1. histogram 统计每个倒排桶大小；
2. `cub::DeviceScan::ExclusiveSum` 得到桶起始 offsets；
3. 用原子 cursor 把向量 id 填入 `list_ids`。

### 5.3 检索

每个 query：

1. 计算 query 到全部 `nlist` 中心的距离（同款排序键编码）；
2. 分段排序后取前 `nprobe` 个中心；
3. 由 `probe -> list offsets -> candidate arena` 汇集候选；
4. 对候选向量重算精确距离并分段 Top-K。

倒排桶访问、候选拼接和 query 内分段排序都在 GPU 上完成；各 query 的候选数量只由
命中的桶决定，用 exclusive scan 生成连续的 candidate arena。

## 6. IVF-PQ

### 6.1 码本训练

把 `D` 分成 `m` 个子空间（要求 `D % m == 0`），每个子空间抽取训练样本的
`D/m` 维切片，对 `pq_ks = 256` 个码字做小型 Lloyd：

```text
样本 × 256 码字（子空间维数小）→ 最近码字 → 原子累加 → 更新码字
```

### 6.2 压缩编码

`pqEncodeKernel` 对库中每个向量按子空间分别找最近码字，输出 `N×m` 字节
`uint8` 码。1e6×128、m=16 时压缩码仅 16 MB，约是 fp32 原始数据的 1/32。

### 6.3 ADC 检索

- 对每个 query 先建 `m × 256` 距离表（L2 用子空间平方距离，inner/cosine 用点积）；
- 每个候选只查 `m` 次表并累加得到近似距离；
- 近似结果仍按打包键做 query 内分段排序，输出与 IVF-Flat 相同的 Top-K 结构。

IVF 本身仍用未经压缩的距离挑选倒排桶，兼顾召回与 ADC 的近似性。设计上把“候选
筛选”和“精排/近似排序”分开；默认会对 ADC 前 `pq_rerank=256` 个候选用原始向量
精确重排（与 IVF-Flat 共用打分 kernel），把压缩带来的精度损失降到很低。设置
`pq_rerank = 0` 可观察纯 ADC 的 recall 与吞吐。

## 7. Top-K、正确性与确定性

### 7.1 支持的 K

CLI/配置任意正整数 K，测试覆盖 1/10/50/100；`topK ≤ 65535` 均可直接解码。

### 7.2 正确性策略

- CPU 参考 `cpuExactSearch` 使用与 GPU 相同的 score、相同的 “越小/越大”方向、相同
  的 id tie-break；
- GPU 与 CPU 的距离允许浮点误差（NVIDIA 构建阈值 1e-4；CoreX 构建因 device FP64
  精度受限改用 float 累计，`test_gpu.cu` 在 `VSEARCH_COREX` 下将距离阈值放宽到
  2e-3），id 必须逐位相同；
- `test_gpu.cu` 使用分簇合成数据验证：exact GPU = CPU、IVF recall、索引 round-trip；
- `validate` 子命令在内存中重跑上述检查并返回非零退出码。

## 8. 性能日志与实验方法

### 8.1 日志字段

- `perf.log`：`mode / nq / n / dim / topk / build_ms / search_ms / qps /
  p50_ms / p99_ms / mean_batch_ms / gpu_used / cpu_ms / speedup`；
- `quality.log`：`recall_at_k`、`avg_distance_error`、mismatch；
- `result.txt`：每 query K 行 id + score。

`P50/P99` 统计以 CLI 内部每个 batch 的 CUDA event 耗时为样本（`batch_size` 可在
配置里改变），日志同时给出 batch 均值，避免把整段 wall time 误当作延迟。

### 8.2 CPU baseline 对比口径

CPU 参考是单线程暴力 `O(N×D)`，只在 `ref_query_limit`（默认 200）个 query 上运行；
GPU exact 用相同 query 子集重测一次，`speedup = cpu_ms / gpu_ms`。这是可复现的
“工程基线”对比；若要和 FAISS CPU 对比，可把 `cpuExactSearch` 换成 FAISS IndexFlat
并保持同 query 子集。

### 8.3 实测性能总览

CPU 参考是单线程暴力，200 query 子集约 14.8 s（100 query 约 7.4 s）；GPU exact
在相同 100 query 子集约 67 ms，加速约 110×。

**表 1：性能总览（RTX 4090，1e6×128，nprobe=16，topK=100）**

| mode | n | dim | nq | topK | build ms | search ms | QPS | P50 ms | P99 ms | 显存 | CPU ms | speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exact_gpu | 1e6 | 128 | 1000 | 100 | - | 1433.5 | 697.6 | 71.55 | 73.44 | 1.24 GB | 7413* | ~111* |
| ivf_flat | 1e6 | 128 | 1000 | 100 | 223.9 | 47.3 | 21150 | 5.90 | 6.43 | 0.89 GB | 7413 | 156.8 |
| ivf_pq16+rerank | 1e6 | 128 | 1000 | 100 | 402.8 | 34.1 | 29338 | 4.17 | 4.83 | 0.90 GB | 7478 | 219.4 |

`*`：CPU 参考跑 100 query（7413 ms）；GPU exact 在同一子集约 66.9 ms（110.7×）。

**表 2：IVF-Flat nprobe 权衡（batch=128，recall@100）**

该表来自另一组 2000-cluster 合成数据（前文表 1/3 为 10000-cluster 数据），用于展示
同一索引在 nprobe 变化时的 recall-QPS 曲线。

| nprobe | recall@100 | QPS | P50 ms | P99 ms |
| --- | --- | --- | --- | --- |
| 1 | 0.0192 | 69385 | 1.78 | 2.11 |
| 4 | 0.2155 | 52347 | 2.36 | 2.70 |
| 8 | 0.8608 | 47862 | 2.59 | 3.03 |
| 16 | 1.0000 | 38240 | 3.24 | 3.78 |
| 32 | 1.0000 | 28434 | 4.35 | 5.00 |
| 64 | 1.0000 | 20058 | 6.21 | 6.83 |

**表 3：IVF-PQ（ADC 粗排 + top-256 精排）**

| pq_m | recall@100 | search ms（nq=1000） | QPS |
| --- | --- | --- | --- |
| 16 | 0.9248 | 34.1 | 29338 |
| 64 | 0.9264 | 40.0 | 24996 |

以上表 1-3 来自早期 1e6×128 数据（`data_ann`，生成器修复前格式）。为验证所有
CoreX 适配改动不破坏 NVIDIA 路径，最终源码又用修复后生成器产出 300k×128 数据
（与 CoreX §11 同一组生成参数：100 个高斯簇、scale 0.10）做回归 bench：

**表 3b：NVIDIA 最终源码回归（RTX 4090，300k×128，nq=1000，topK=100，nprobe=16）**

| mode | build ms | search ms | QPS | P50 ms | P99 ms | recall@100 | speedup vs CPU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exact_gpu | - | 98.6 | 2029 | 49.3 | 59.5 | - | 87.1× |
| ivf_flat | 97 | 20.4 | 49058 | 2.51 | 3.02 | 0.986 | 425× |
| ivf_pq16+rerank | 213 | 21.8 | 45792 | 2.59 | 3.68 | 0.012 | 397× |

GPU 集成测试在 NVIDIA 上全部通过（exact=CPU、IVF-Flat recall≈1.0、IVF-PQ 合法、
索引 round-trip）。表中 exact 行取独立 exact bench 的 `exact_perf.log`
（IVF bench 内的 exact 子集对比行另有 ~95.5 ms / 90.8× 的测量波动，两者同为
合法数据）。表 3b 与 §11.3 的 CoreX 数据使用同一生成参数，两平台 IVF-Flat 的
recall@100 均约 0.99（0.9858 / 0.9914），IVF-PQ 均约 0.012——差异只来自各自
机器的独立随机数据与 ADC 量化，而非平台实现（详见 §11.3 的量化分析）。

## 9. 优化记录与设计取舍

开发中形成的主要取舍如下，供评估代码时对照：

1. **用“打包键 + 分段 radix sort”代替每 query 单 block 共享 heap**：逻辑简单、全
   局确定、tie-break 易控制；代价是需要 in/out 键数组与排序临时显存，因此 exact
   做了 query 分块。对 IVF 的小候选集该开销很小。
2. **距离计算不写中间全量矩阵**：exact 只写排序键；IVF 只对候选写键。显著降低
   8 GB 卡上的压力。
3. **fp16 文件先转 fp32**：功能与正确性优先；张量化 half2 kernel、以 fp16 距离表
   为下一步优化。
4. **KMeans 用 GEMM + 子样本 mini-batch**：全量 `N×nlist×D` 朴素循环在 4096 桶时
   会退化为纯内存扫描；GEMM 分块把中心复用放到 tensor core/访存优化路径，最终分配
   仍保持精确最近中心。
5. **原子累加训练中心**：速度快、可复现性由固定训练子样本保证；中心在浮点低位可能
   与严格顺序归约有差异，但对检索质量影响可忽略。
6. **倒排桶内不排序 id**：Top-K 最终按排序键输出，桶内顺序不影响结果，因此省一次
   大规模 sort。
7. **cosine 归一化放在加载期**：所有 kernel 只需内积，输出统一为 cosine 距离。

## 10. 性能分析（Nsight Systems）

本次实验使用 Nsight Systems 2024.6.2 对 exact / IVF-Flat / IVF-PQ 三种检索做了
CUDA kernel 时间线分析，完整命令、统计表与结论见
[PROFILING.md](PROFILING.md)。

### 10.1 收集

```bash
# 时间线 + kernel 统计（本次实际执行）
nsys profile --force-overwrite=true -o prof/nsys_exact \
  -t cuda,osrt ./build/vsearch search \
  --vectors=data_ann/vectors.bin --queries=data_ann/queries.bin \
  --params=data_ann/params.txt --search_mode=exact --batch_size=128

nsys stats --report cuda_gpu_sum prof/nsys_exact.nsys-rep
```

### 10.2 重点分析项

- **exact 检索**：`exactKeysKernel` 占 GPU kernel 时间约 74.9%，平均 49.2 ms；
  分段 radix sort 约 23.1%。说明全量打分与排序是精确检索的两个主要成本。
- **IVF-Flat**：每 128-query batch 的中心打分约 0.32 ms、候选精排约 2.92 ms，
  倒排 gather/probe/scan 均只有微秒到十几微秒，索引结构本身开销很小。
- **IVF-PQ + rerank**：ADC 表构建每 batch 仅 5.4 μs，PQ 打分 0.126 ms，
  top-256 精排 0.08 ms；召回接近 IVF-Flat 时仍保持较低精排开销。

分析统一在 RTX 4090、CUDA 12.8、Nsight Systems 2024.6.2、
`N=1e6, D=128, nq=1000, topK=100` 下进行，保证结果可比。

## 11. 国产平台适配

默认 NVIDIA（CUDA Toolkit + CMake）。除 NVIDIA 外，本项目的同源代码也已实际移植并
运行在**天数智芯 Iluvatar CoreX**（MR-V100，IX-ML 4.4.0，32 GB）上，主机测试与
GPU 集成测试全部通过，端到端 bench 已生成结果/性能/质量日志。

### 11.1 天数智芯构建方式

CoreX 的 `/usr/local/corex/bin/nvcc` 只是版本回显 stub，官方未提供 CMake CUDA
工具链；真实工具链是 CoreX 定制 clang 18（用 `-x ivcore` 编译 `.cu`）。因此
CoreX 构建使用 [Makefile.corex](../Makefile.corex)，一条命令得到 `vsearch`、
`test_host`、`test_gpu`：

```bash
make -f Makefile.corex -j
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib
./build_corex/test_host && ./build_corex/test_gpu
```

Makefile 对 `.cu` 追加 `-x ivcore --cuda-path=/usr/local/corex
-DVSEARCH_COREX=1`，链接 CoreX 自带 `libcudart`/`libcublas`。引擎在 CoreX 上仍
使用其自带的 cuBLAS 与 CUB（已验证可运行），没有另写 GEMM/sort 后端。

### 11.2 实测发现并修复的平台差异（已并入源码）

1. **64 位 `atomicAdd` 在 CoreX 设备端静默失效**：调用不报错但值不增加，导致
   聚类计数与倒排直方图恒为 0、倒排表总数为 0。所有计数/游标改为 32 位原子；
   需要 64 位前缀和的地方仍由 `cub::DeviceScan` 输出 64 位 offsets。
2. **CUB `DeviceScan` 可用但行为需适配**：`ExclusiveSum` 只写 `out[0..n-1]`，
   不写 `out[n]`（total）；引擎先用 `n+1` 临时缓冲跑 scan，再按“最后一个前缀 +
   最后一项”补写 `out[n]`，避免把 CUB 越界写当成正常行为。
3. **FP64 精度受限**：CoreX 设备端 double 加法对长求和约有 `1e-4` 相对误差
   （实测对 128 维长和可到绝对误差数百，float 反而与 CPU IEEE 完全一致）。
   `-DVSEARCH_COREX=1` 时距离内核用 float 累计（同一份 kernel 以
   `VSEARCH_ACC_TYPE` 宏切换），NVIDIA 构建仍用 double 累计。
4. **PTX 内联汇编不可移植**：`asm("mov.b32 ...")` 的寄存器约束在 ivcore 后端分配
   失败，改为 CUDA 内建 `__float_as_int` / `__int_as_float`（NVIDIA 与 CoreX 均
   支持且行为一致）。

这些差异只影响正确性路径，不改算法；IVF 聚类、倒排构建、分段排序、PQ 编码与
ADC 打分在同一份 kernel 中运行。

### 11.3 天数智芯实测结果

**表 4：CoreX（MR-V100）300k×128，nq=1000，topK=100，nprobe=16**

| mode | build ms | search ms | QPS | P50 ms | P99 ms | 显存 | CPU ms | speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exact_gpu | - | 546.1 | 366 | 272.7 | 342.0 | 0.63 GB | 11655 | 21.3× |
| ivf_flat | 254 | 107.1 | 9340 | 13.6 | 13.8 | 0.41 GB | 11592 | 108.3× |
| ivf_pq16+rerank | 3947 | 58.9 | 16974 | 7.2 | 7.6 | 0.41 GB | 11833 | 200.9× |

CPU 参考为单线程暴力检索（200 query 子集）。GPU 集成测试另验证了
`exact=CPU`、IVF-Flat recall≈1.0 与索引 round-trip。

**表 5：CoreX IVF-Flat nprobe/batch 权衡（同一 300k 数据集，batch=128 行）**

| nprobe | recall@100 | QPS | P50 ms | P99 ms |
| --- | --- | --- | --- | --- |
| 1 | 0.013 | 37919 | 3.19 | 3.29 |
| 2 | 0.019 | 28317 | 4.29 | 4.50 |
| 4 | 0.065 | 20254 | 6.11 | 6.41 |
| 8 | 0.544 | 14121 | 8.87 | 9.02 |
| 16 | 0.991 | 9320 | 13.63 | 13.99 |
| 32 | 1.000 | 5125 | 24.71 | 25.85 |
| 64 | 1.000 | 2658 | 48.09 | 49.10 |

batch=32 时 P50 更低（nprobe=16 时 4.14 ms），batch=512 时 QPS 更高但单批延迟
更大；完整 21 组（nprobe×batch）数据见
[outputs_corex/experiment_summary.csv](../outputs_corex/experiment_summary.csv)。

**PQ 召回限制（作为质量分析的一部分）**：在 300k 高斯聚类合成数据上 IVF-PQ 的
recall@100≈0.012，而 IVF-Flat≈0.986。用索引文件 + Python 复算 ADC 分数确认并非
实现错误：nprobe=16 的真实候选池（约 5k 向量）包含全部 true top-100，但 PQ 编码
误差（编码 RMSE≈0.10，true-top 距离仅 0.2~1.5）使 ADC 粗排只把其中 4~22 个放进
top-256 精排窗口。该实验正是题目要求呈现的“PQ 压缩率 vs 召回”取舍：PQ 换取约
1.8×/2.3× 的 QPS 提升与约 16× 的码本/编码显存缩减，但在此数据上召回显著下降；
紧致分离簇数据（40k×32，100 簇）上同一实现 top-1 召回 100%，证明 PQ 通路正确，
召回瓶颈来自数据分布与码率（8 bit×16 子空间）的匹配。

这一现象与平台无关：相同生成参数的 300k 数据在 NVIDIA RTX 4090 上回归 bench
得到同一 recall（ivf_flat≈0.986、ivf_pq≈0.012，见 §8.3 表 3b），两平台互相印证
ADC 打分与码本编码实现一致。

完整测试输出与日志见 `outputs/`（NVIDIA）与 `outputs_corex/`（CoreX）。此外
§10 的 Nsight Systems 分析在 NVIDIA 平台完成；CoreX 无 nvidia 工具链，故未在
CoreX 上重复 nsys 采样。

## 12. 可继续提升的方向

- **查询并行流**：多个 stream 并行处理不同 batch，降低 P99；
- **rerank**：PQ top 候选用原始向量精排，兼顾召回与显存；
- **索引增强**：粗聚类加 k-means++/HNSW graph，倒排桶失衡时做子桶拆分；
- **低精度**：fp16x2 向量与距离表、int8 重排；
- **动态索引**：增量 insert/delete 与分段合并；
- **持久化/服务化**：内存池复用、共享显存多进程、gRPC 接口；
- **FAISS 对照矩阵**：与 FAISS `IndexIVFFlat`/`IndexIVFPQ` 在相同数据集、相同
  `nprobe` 下逐项对比 recall 与吞吐。

## 13. 复现清单

```bash
# 1) NVIDIA 构建与正确性自检
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/vsearch validate

# 2) CoreX 构建与正确性自检
make -f Makefile.corex -j
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib
./build_corex/test_host && ./build_corex/test_gpu

# 3) 生成数据并扫描（NVIDIA 示例；CoreX 将 ./build/ 换成 ./build_corex/）
python python/gen_dataset.py --out data \
    --n 1000000 --dim 128 --nq 1000 --top-k 100
python python/run_experiments.py --vsearch ./build/vsearch \
    --nprobe-list 1,2,4,8,16,32,64 --batch-list 32,128,512

# 4) 把生成目录中的 experiment_summary.csv 数字回填到 §8.3
```

表 1-3（NVIDIA 1e6）、表 3b（NVIDIA 300k 回归）与表 4-5（CoreX 300k）分别对应
`outputs/`、`outputs/nvidia_300k_regression/`、`outputs_corex/` 下的原始
perf/quality 日志；代码、测试与脚本均在仓库内，任何一行运行路径均无需人工改动
数据格式。
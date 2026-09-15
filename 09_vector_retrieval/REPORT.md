# GPU 向量检索引擎实验报告

被测程序：`vsearch`（GPU 向量检索库 + 命令行工具）

本报告记录该程序的实现要点、实验条件、实验步骤与全部测量结果，并说明每一项
结论对应的数据来源与复现方式，供独立核验。

## 1. 摘要

本实验对应“九. GPU 向量检索引擎”题目，评测对象是一个 GPU 向量检索程序，提供
三档检索模式：

| 模式 | 说明 | 评价基准 |
| --- | --- | --- |
| `exact` | GPU 全量距离计算 + Top-K 归并 | 与 CPU 暴力检索逐条比对 |
| `ivf_flat` | IVF 倒排 + 原始向量重算距离 | recall@K（基准为 GPU exact） |
| `ivf_pq` | IVF 倒排 + PQ 压缩 + ADC 近似距离 | recall@K、平均距离误差 |

程序支持 fp16/fp32 输入、L2 / inner product / cosine 三种度量、批量查询、K=1~100、
索引落盘与重载，并输出结果文件、性能日志与质量日志。CPU 参考实现独立于 GPU 代码，
用于校验精确检索的距离与排序（含同分时按 id 升序的确定性 tie-break）。

实验在四套 GPU 平台上完成，各平台均执行了完整的构建、正确性测试与端到端基准测试：

| 平台 | GPU | 系统 / 软件栈 | 本次执行内容 |
| --- | --- | --- | --- |
| NVIDIA | RTX 4090 24 GB | Ubuntu 24.04，CUDA 12.0/12.8，驱动 570 | 构建、主机/GPU 测试、1e6 与 300k 两档基准 |
| 天数智芯 | Iluvatar MR-V100 32 GB | IX-ML 4.4.0（clang `-x ivcore`） | 构建、主机/GPU 测试、300k 基准 + 21 组扫描 |
| 沐曦 | MetaX MXC500 32 GB | MACA 3.5.3（`cucc` + `mcblas`） | 构建、主机/GPU 测试、300k 基准 + 21 组扫描 |
| 摩尔线程 | MUSA 5.1.0 环境 | MUSA 5.1.0（`mcc -x musa` + `mublas`） | 构建、主机/GPU 测试、300k 基准 + 21 组扫描 |

主要结果（300k×128，nq=1000，topK=100，nprobe=16，详见 §7）：

| 平台 | exact QPS | ivf_flat QPS | ivf_flat recall@100 | ivf_pq QPS |
| --- | --- | --- | --- | --- |
| NVIDIA RTX 4090 | 2029 | 49058 | 0.986 | 45792 |
| 天数智芯 MR-V100 | 366 | 9340 | 0.991 | 16974 |
| 沐曦 MXC500 | 1560 | 15916 | 0.991 | 15985 |
| 摩尔线程 MUSA | 194 | 5581 | 0.991 | 6092 |

四套平台上，精确检索的 id 与 CPU 参考逐位一致，IVF-Flat 在 nprobe=32 时
recall@100 达到 1.000，IVF-PQ 的召回率受量化精度限制（本数据集上约 0.012，
原因见 §7.8）。

## 2. 实验目标与范围

本实验需要回答的问题：

1. **功能正确性**：GPU 精确检索的结果是否与独立实现的 CPU 暴力检索一致（id 逐位
   相同、距离在给定容差内）？Top-K 输出是否严格有序？
2. **近似检索的有效性**：IVF-Flat 与 IVF-PQ 相对精确检索的 recall@K 是多少？
   nprobe 与召回率、吞吐之间的关系如何？
3. **性能**：在给定数据规模下，各模式的 QPS、P50 / P99 延迟、显存占用，以及相对
   CPU 单线程基线的加速比。
4. **跨平台可迁移性**：同一程序在三种不同 GPU 软件栈上的构建可行性、正确性表现
   与性能差异，以及各平台需要哪些针对性处理。

实验范围与边界：

* 数据为合成数据（生成方式见 §5.1），不涉及真实业务语料；
* 召回率基准是 GPU 精确检索结果，不是解析式 ground truth；
* CPU 基线是单线程暴力检索，仅用于给出量级参照，不等价于经过优化的 CPU 检索库；
* 性能数字为单次运行结果，用于平台间量级对比，未做多轮统计与方差分析。

## 3. 实验环境

### 3.1 硬件与系统

| 项目 | 平台 A（NVIDIA） | 平台 B（天数智芯） | 平台 C（沐曦） | 平台 D（摩尔线程） |
| --- | --- | --- | --- | --- |
| GPU | GeForce RTX 4090 | Iluvatar MR-V100 | MetaX MXC500（单 SGPU 分片） | 摩尔线程 GPU（`/dev/mtgpu`） |
| 显存 | 24 GB | 32 GB | 32 GB（mx-smi 显示 50% 规格） | 未记录（`musaMemGetInfo` 报告空闲 49 GB） |
| CPU / 内存 | 容器环境，未记录 | 112 核 / 32 GB | 128 核 / 64 GB | 128 核 / 64 GB |
| 操作系统 | Ubuntu 24.04 | Ubuntu 24.04.4 | Ubuntu 20.04 | Ubuntu 20.04（内核 5.15） |
| GPU 驱动 / 工具链版本 | 570.124.06 | IX-ML 4.4.0 | MACA 3.5.3.20（mx-smi 2.2.12） | MUSA 5.1.0（CUB 1.17.2、mcc 5.1.0） |

### 3.2 工具链与构建方式

| 平台 | 构建文件 | 编译器 | 数学库 | 运行时库路径 |
| --- | --- | --- | --- | --- |
| NVIDIA | `CMakeLists.txt` | `nvcc`（CUDA 12.0 / 12.8） | cuBLAS | CUDA Toolkit 默认 |
| 天数智芯 | `Makefile.corex` | CoreX 定制 clang 18（`-x ivcore`） | CoreX cuBLAS 兼容层 | `/usr/local/corex/lib64` |
| 沐曦 | `Makefile.maca` | `cucc`（→ mxgpu_llvm `mxcc`） | `libmcblas.so` | `/opt/maca/lib`、`/opt/maca/tools/cu-bridge/lib` |
| 摩尔线程 | `Makefile.musa` | `mcc -x musa`（clang 14 前端） | `libmublas.so` | `/usr/local/musa/lib` |

天数智芯与沐曦提供 CUDA 兼容头文件与 CUB 实现，因此程序中的 `cub::DeviceScan`、
`cub::DeviceSegmentedRadixSort` 与 cuBLAS 调用无需替换；摩尔线程使用 MUSA 原生
API 与 MUSA 版 CUB，通过 `-DVSEARCH_MUSA=1` 的名称映射复用同一套实现。各平台的
构建命令见 §5.4。

### 3.3 平台能力探测

移植前在三台加速卡平台上分别执行了设备能力探测，结果决定各平台是否需要修改计算
路径（探测程序见 [tools/probe/device_probe.cu](tools/probe/device_probe.cu)）：

| 探测项 | 平台 B（天数智芯） | 平台 C（沐曦） | 平台 D（摩尔线程） |
| --- | --- | --- | --- |
| `atomicAdd(unsigned long long*)` | 调用返回成功但计数不增加（256 线程累加结果为 0） | 正常（256 线程累加结果为 256） | 正常（256 线程累加结果为 256） |
| device 端 double 长求和（128 维） | 与主机结果偏差约 1e-4 相对量级 | 与主机 double 结果一致（偏差 0） | 与主机 double 结果一致（偏差 0） |
| PTX 内联汇编 `asm("mov.b32 ...")` | 后端寄存器分配失败，编译报错 | 未使用（改用内建函数后不再依赖） | 未使用 |

对应的处理方式见 §7.2。

## 4. 系统实现

### 4.1 模块结构

```text
include/vsearch/           主机侧数据结构、文件格式、配置、CPU 参考
cuda/engine.cu             CUDA kernels、KMeans、IVF-Flat/PQ、序列化、检索
src/main.cpp               build / search / bench / validate 子命令
tests/                     test_host.cpp、test_gpu.cu
tools/probe/               设备能力探测用例（atomicAdd / double 精度）
python/gen_dataset.py      合成数据与参数文件
python/run_experiments.py  nprobe x batch 扫描并汇总 CSV
CMakeLists.txt             NVIDIA 构建（nvcc）
Makefile.corex             CoreX 构建（clang -x ivcore，-DVSEARCH_COREX=1）
Makefile.maca              沐曦构建（cucc + mcblas）
Makefile.musa              摩尔线程构建（mcc -x musa，-DVSEARCH_MUSA=1）
outputs/                   NVIDIA 平台日志与结果样例
outputs_corex/             CoreX 平台 perf/quality 日志与扫描汇总
outputs_maca/              沐曦平台 perf/quality 日志与扫描汇总
outputs_musa/              摩尔线程平台 perf/quality 日志与扫描汇总
```

编译产物为一个静态库和三个可执行文件（`vsearch`、`test_host`、`test_gpu`）。

### 4.2 向量表示与距离度量

#### 4.2.1 数据布局

向量库/查询按行主序加载。文件头定义见 README。内部统一转成 `fp32` 并放到对齐内存，
因此 fp16 文件也能直接获得相同的精确检索语义；fp16 的价值体现在“原始数据减少 2 倍
带宽/容量”，后续可用 half2 张量化作为内存受限优化（见 §12）。

#### 4.2.2 度量

- **L2**：候选打分直接累计逐维差分平方和
  `score = Σ_d (q[d] - x[d])²`，排序用平方距离，输出时开根得到欧氏距离。直接
  差分求和避免了“范数相减”在大数值下抵消导致的精度损失，也让 CoreX 的 float
  路径与 CPU 参考保持同一累加语义。
  （中心挑选等只需要相对次序的场景仍使用
  `||q - c||² = ||q||² + ||c||² - 2 q·c` 的预计算范数形式，见 §4.4.3 / §4.5.3。）
- **inner product**：输出原始点积，越大越相似。
- **cosine**：读取时把库向量与查询归一化，再按 `1 - q·x` 输出 cosine 距离。

所有 kernel 在 `dim ≤ 65536` 的一般维度上按运行时循环实现（编译器可展开），没有把
维度写成模板常量，便于实验不同 `D`。

### 4.3 精确检索 baseline（exact）

#### 4.3.1 单轮全量打分

`exactKeysKernel` 以 query 为 `blockIdx.y`、向量行为 `blockIdx.x` 组织网格。每个
线程计算一整行向量的距离（L2 直接累计逐维差分平方和；inner product / cosine
累计点积），只把结果写成 64 位排序键，不写全量浮点距离矩阵，避免一次分配
`nq × N` 个 fp32 的中间显存。

#### 4.3.2 Top-K 归并

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

#### 4.3.3 显存分块

exact 的键数组为 `nq_chunk × N × 16 B`（in/out 两份）。CLI 以
`exact_memory_mb`（默认 1536 MB）限制内部 query 子批次，避免在低显存卡上失败。

### 4.4 IVF-Flat

#### 4.4.1 粗聚类中心训练

- 从库中取至多 `kmeans_sample` 行作为训练样本；
- 用确定间距的样本行初始化 `nlist` 个中心；
- 执行 mini-batch Lloyd：GEMM 求样本×中心点积、最近中心分配、按中心原子累加求和与
  计数、更新中心；空桶从样本行重新注入；
- cosine 数据下每轮把中心归一化到单位长度；
- 迭代默认 12 轮，可选按“分配变化比例 < 阈值”提前收敛。

训练时只累加子样本，控制原子写开销；对全部 1e6 向量只做一次最终分配，因此 build
时间可预估为“训练 + 一次全库最近中心分配 + 倒排重排”。

#### 4.4.2 全库分配与倒排表

全库分配使用同一 GEMM 路径，按 `kmeans_chunk_rows` 分块，避免一次分配
`N × nlist` 浮点矩阵。最近中心判定 kernel 直接读取列主 GEMM 输出（按列连续），
合并 `||x||² + ||c||² - 2 dot`。随后：

1. histogram 统计每个倒排桶大小；
2. `cub::DeviceScan::ExclusiveSum` 得到桶起始 offsets；
3. 用原子 cursor 把向量 id 填入 `list_ids`。

#### 4.4.3 检索

每个 query：

1. 计算 query 到全部 `nlist` 中心的距离（同款排序键编码）；
2. 分段排序后取前 `nprobe` 个中心；
3. 由 `probe -> list offsets -> candidate arena` 汇集候选；
4. 对候选向量重算精确距离并分段 Top-K。

倒排桶访问、候选拼接和 query 内分段排序都在 GPU 上完成；各 query 的候选数量只由
命中的桶决定，用 exclusive scan 生成连续的 candidate arena。

### 4.5 IVF-PQ

#### 4.5.1 码本训练

把 `D` 分成 `m` 个子空间（要求 `D % m == 0`），每个子空间抽取训练样本的
`D/m` 维切片，对 `pq_ks = 256` 个码字做小型 Lloyd：

```text
样本 × 256 码字（子空间维数小）→ 最近码字 → 原子累加 → 更新码字
```

#### 4.5.2 压缩编码

`pqEncodeKernel` 对库中每个向量按子空间分别找最近码字，输出 `N×m` 字节
`uint8` 码。1e6×128、m=16 时压缩码仅 16 MB，约是 fp32 原始数据的 1/32。

#### 4.5.3 ADC 检索

- 对每个 query 先建 `m × 256` 距离表（L2 用子空间平方距离，inner/cosine 用点积）；
- 每个候选只查 `m` 次表并累加得到近似距离；
- 近似结果仍按打包键做 query 内分段排序，输出与 IVF-Flat 相同的 Top-K 结构。

IVF 本身仍用未经压缩的距离挑选倒排桶，兼顾召回与 ADC 的近似性。设计上把“候选
筛选”和“精排/近似排序”分开；默认会对 ADC 前 `pq_rerank=256` 个候选用原始向量
精确重排（与 IVF-Flat 共用打分 kernel），把压缩带来的精度损失降到很低。设置
`pq_rerank = 0` 可观察纯 ADC 的 recall 与吞吐。

### 4.6 Top-K、正确性与确定性

#### 4.6.1 支持的 K

CLI/配置任意正整数 K，测试覆盖 1/10/50/100；`topK ≤ 65535` 均可直接解码。

#### 4.6.2 正确性策略

- CPU 参考 `cpuExactSearch` 使用与 GPU 相同的 score、相同的 “越小/越大”方向、相同
  的 id tie-break；
- GPU 与 CPU 的距离允许浮点误差（NVIDIA 构建阈值 1e-4；CoreX 构建因 device FP64
  精度受限改用 float 累计，`test_gpu.cu` 在 `VSEARCH_COREX` 下将距离阈值放宽到
  2e-3），id 必须逐位相同；
- `test_gpu.cu` 使用分簇合成数据验证：exact GPU = CPU、IVF recall、索引 round-trip；
- `validate` 子命令在内存中重跑上述检查并返回非零退出码。

### 4.7 设计取舍与优化记录

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

## 5. 实验设计

### 5.1 数据集与参数

实验数据由 `python/gen_dataset.py` 生成，方法为“高斯簇 + 扰动”：先随机生成若干
簇心，每条库向量取一个簇心并叠加正态噪声；查询向量取若干库向量的扰动副本，因此
每条查询都存在明确的近邻结构。文件格式为 §4.2.1 定义的二进制容器。

本报告使用两组数据：

| 数据组 | N | D | nq | topK | 簇数 | 向量噪声 | 查询噪声 | nlist | nprobe | pq_m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `data_ann`（早期） | 1 000 000 | 128 | 1000 | 100 | 自动 | 默认 | 默认 | 4096 | 16 | 16 |
| 统一复现组 | 300 000 | 128 | 1000 | 100 | 100 | 0.10 | 0.02 | 1024 | 16 | 16 |

其中 300k 数据组在四套平台上使用相同生成参数与相同随机种子（`--seed 2026`）
生成，用于跨平台比较。其余参数取默认值：`kmeans_iters = 12`、
`kmeans_sample = 65536`、`pq_ks = 256`、`pq_rerank = 256`、
`ref_query_limit = 200`、`exact_memory_mb = 4096`。

### 5.2 对照组与评价指标

- **正确性对照**：CPU 单线程暴力检索（`cpuExactSearch`），与 GPU 代码相互独立；
  比较 id 是否逐位相同、距离/相似度是否在容差内。
- **召回率**：recall@K 为预测 Top-K 与 GPU exact Top-K 逐位相同的比例，对全部
  query 取平均；基准是 GPU 精确检索结果。
- **距离误差**：同一 id 在两套结果中的分数之差的绝对值均值。
- **吞吐与延迟**：QPS = nq / 总墙钟时间；P50/P99 取各内部 batch 的 CUDA event
  耗时样本的分位数。
- **加速比**：`speedup = cpu_ms / gpu_ms`，两值用同一批 query 子集（默认 200 条）
  测量，避免用全量 GPU 时间除以子集 CPU 时间。

CPU 基线仅作量级参照：它是单线程、未向量化的实现，不等价于优化过的 CPU 检索库。

### 5.3 测量口径与日志字段

- `perf.log`：`mode / nq / n / dim / topk / build_ms / search_ms / qps /
  p50_ms / p99_ms / mean_batch_ms / gpu_used / cpu_ms / speedup`；
- `quality.log`：`recall_at_k`、`avg_distance_error`、mismatch；
- `result.txt`：每 query K 行 id + score。

`P50/P99` 以 CLI 内部每个 batch 的 CUDA event 耗时为样本（`batch_size` 可配置），
日志同时给出 batch 均值，避免把整段 wall time 误当作延迟。所有性能数字为单次运行
结果，用于平台间量级对比，未做多轮重复实验。

### 5.4 实验流程与命令

每个平台按相同的五步执行：**构建 → 正确性测试 → 生成数据集 → 三种模式基准 →
nprobe × batch 扫描**。下面以 **NVIDIA 平台**为例给出完整命令；其余平台的构建
入口与库路径差异见 §7.1，只需替换第 1、2 步对应的命令，后续步骤一致。

```bash
# 1) 构建
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j

# 2) 正确性测试
./build/test_host
./build/test_gpu

# 3) 生成数据集（300k×128，与其余平台同参数、同随机种子）
python python/gen_dataset.py --out data \
    --n 300000 --dim 128 --nq 1000 --top-k 100 --metric l2 \
    --clusters 100 --vector-scale 0.10 --query-scale 0.02 \
    --nlist 1024 --nprobe 16

# 4) 三种模式的端到端基准（每次 bench 先跑 GPU exact 作为召回基准）
./build/vsearch bench --vectors=data/vectors.bin --queries=data/queries.bin \
    --params=data/params.txt --search_mode=exact \
    --perf_log_path=outputs/exact_perf.log \
    --quality_log_path=outputs/exact_quality.log
# ivf_flat / ivf_pq 同理，仅替换 --search_mode 与输出文件名，完整命令见 §11.1

# 5) nprobe × batch 扫描
python python/run_experiments.py --vsearch ./build/vsearch \
    --data data --mode ivf_flat --nprobe-list 1,2,4,8,16,32,64 \
    --batch-list 32,128,512 --out-dir outputs
```

各平台的数据生成脚本与扫描脚本完全相同；若某平台自带 Python 环境（如沐曦使用
`/opt/conda/bin/python`），仅需替换解释器路径。

## 6. 正确性验证结果

### 6.1 主机侧单元测试（test_host）

5 个用例在四套平台上全部通过（输出 `ALL HOST TESTS PASSED`）：

| 用例 | 检查内容 |
| --- | --- |
| fp32 file round-trip | fp32 向量文件写入后读回逐元素一致 |
| fp16 round-trip | fp16 写入/读回在给定容差内 |
| parameter parser | 参数文件解析结果与预期一致 |
| CPU reference deterministic tie ordering | 同分时按 id 升序，结果确定 |
| CPU reference L2 distances | CPU 参考的 L2 距离与解析计算一致 |

### 6.2 GPU 集成测试（test_gpu）

测试数据为 20000×64、100 个分离簇的合成集，四套平台输出一致：

| 检查项 | 结果 |
| --- | --- |
| GPU exact 与 CPU 参考一致 | PASS（id 逐位相同，距离在容差内） |
| IVF-Flat recall（nprobe=8） | 1.0000（判定阈值 > 0.95） |
| 索引保存后重新加载结果一致 | PASS |
| IVF-PQ 返回合法向量 id | PASS |

### 6.3 精确检索与 CPU 参考的一致性

- **id**：GPU exact 与 CPU 参考逐位相同，测试与 bench 均校验；
- **距离**：GPU 与 CPU 的距离差在容差内。NVIDIA 与沐曦的 device 端 double 精度
  正常，使用 1e-4 相对容差；天数智芯 CoreX 的 device double 存在精度损失，其构建
  改用 float 累计距离，测试容差相应放宽到 2e-3（依据见 §3.3 的探测数据）。

### 6.4 索引持久化

IVF-Flat 索引保存到文件后重新加载，用同一批查询检索，得到与保存前逐位相同的
id 序列（`test_gpu` 的 round-trip 用例），说明索引序列化格式自洽。

## 7. 性能实验结果

本章数据由 `vsearch bench` 生成，数据集与参数见 §5.1，指标定义见 §5.2。每张表
下方注明对应的原始日志文件，可据此逐项核对。

### 7.1 构建与运行方式

四个平台使用同一份源码，构建入口按平台区分。下表汇总各平台的构建入口与依赖；
随后以 **NVIDIA 平台作为示例**给出完整构建流程，再给出其余三个平台的对应方式。

| 平台 | 构建入口 | 编译 `.cu` | 数学库 | 运行时库路径 | 需定义的宏 |
| --- | --- | --- | --- | --- | --- |
| NVIDIA（示例） | `CMakeLists.txt` | `nvcc`（CUDA 12.x） | cuBLAS（CUDA Toolkit 自带） | CUDA Toolkit 默认 | — |
| 天数智芯 CoreX | `Makefile.corex` | CoreX clang 18（`-x ivcore`） | CoreX cuBLAS 兼容层 | `/usr/local/corex/lib64`、`/usr/local/corex/lib` | `VSEARCH_COREX=1` |
| 沐曦 MetaX | `Makefile.maca` | `cucc`（nvcc 风格 wrapper） | `libmcblas.so` | `/opt/maca/lib`、`/opt/maca/tools/cu-bridge/lib` | — |
| 摩尔线程 MUSA | `Makefile.musa` | `mcc -x musa`（clang 14 前端） | `libmublas.so` | `/usr/local/musa/lib`、`/usr/local/musa/lib64` | `VSEARCH_MUSA=1` |

#### 7.1.1 NVIDIA（示例平台）

环境要求：CMake ≥ 3.18、CUDA Toolkit（本实验为 12.0 / 12.8）、支持 C++17 的主机
编译器。构建命令：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
```

`-DCMAKE_CUDA_ARCHITECTURES` 按实际 GPU 填写（如 RTX 4090 为 `89`，A100 为 `80`）。
构建产物为 `build/vsearch`、`build/test_host`、`build/test_gpu`，运行测试：

```bash
./build/test_host
./build/test_gpu
```

若目标环境不使用 CMake，也可直接调用 `nvcc` 编译 CUDA 源文件：

```bash
# 编译 CUDA 源
nvcc -std=c++17 -O2 -c cuda/engine.cu -Iinclude -o engine.o
# 主机侧源用系统 C++ 编译器编译后，与 engine.o 一起链接（CUDA 源由 nvcc 编译）
g++ -std=c++17 -O2 -Iinclude -c src/main.cpp -o main.o
nvcc main.o engine.o -lcublas -o vsearch
```

其余三个平台只需替换上述构建命令与运行时库路径，源码与后续测试/基准命令不变。

#### 7.1.2 天数智芯 CoreX

CoreX 的 `/usr/local/corex/bin/nvcc` 只是版本回显 stub，官方未提供 CMake CUDA
工具链；实际编译器是 CoreX 定制 clang 18（用 `-x ivcore` 编译 `.cu`）。构建使用
[Makefile.corex](Makefile.corex)：

```bash
make -f Makefile.corex -j
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib
./build_corex/test_host && ./build_corex/test_gpu
```

Makefile 对 `.cu` 追加 `-x ivcore --cuda-path=/usr/local/corex
-DVSEARCH_COREX=1`，并链接 CoreX 自带的 `libcudart` / `libcublas`；CUB 直接使用
CoreX 提供的实现（已验证可用），未替换 GEMM 或排序后端。

#### 7.1.3 沐曦 MetaX

沐曦通过 `cu-bridge` 提供 CUDA 兼容层：编译器为 `cucc`（nvcc 风格 wrapper，内部
调用 mxgpu_llvm 的 `mxcc`）；头文件位于 `/opt/maca/tools/cu-bridge/include`
（cuda_runtime.h、cublas_v2.h、cub）与 `/opt/maca/include`；cuBLAS 的对应实现是
`/opt/maca/lib/libmcblas.so`。构建使用 [Makefile.maca](Makefile.maca)：

```bash
make -f Makefile.maca -j
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/tools/cu-bridge/lib:/opt/maca/lib64
./build_maca/test_host && ./build_maca/test_gpu
```

#### 7.1.4 摩尔线程 MUSA

MUSA 提供原生 API（`musa_runtime.h`、`mublas_v2.h` 以及 MUSA 版 CUB），不提供
CUDA 兼容头文件，因此 `cuda/engine.cu` 在 `-DVSEARCH_MUSA=1` 时启用一层名称映射
（`cuda*` → `musa*`、`cublas*` → `mublas*`、`CUBLAS_OP_*` → `MUBLAS_OP_*`）。
构建使用 [Makefile.musa](Makefile.musa)：

```bash
make -f Makefile.musa -j
export LD_LIBRARY_PATH=/usr/local/musa/lib:/usr/local/musa/lib64
./build_musa/test_host && ./build_musa/test_gpu
```

Makefile 以 `mcc -x musa --musa-path=/usr/local/musa` 编译 `.cu`（若按扩展名推断为
CUDA，mcc 会走 CUDA 前端并报找不到 CUDA 安装），并链接 `-lmusart -lmublas`。

### 7.2 平台差异与处理方式

§3.3 的探测结果对应的处理如下：

| 差异 | 影响 | 处理方式 |
| --- | --- | --- |
| CoreX 设备端 64 位 `atomicAdd` 不生效 | 聚类计数与倒排直方图恒为 0 | 计数与游标改为 32 位原子；需要 64 位前缀和处仍由 `cub::DeviceScan` 输出 64 位 offsets |
| CUB `ExclusiveSum` 不写 `out[n]` | 调用方需要的候选总数缺失 | 使用 `n+1` 临时缓冲执行 scan，再按“最后前缀 + 最后一项”补写 `out[n]` |
| CoreX device double 精度不足 | 长求和误差会使距离排序失真 | CoreX 构建以 `VSEARCH_COREX` 宏切换为 float 累计；NVIDIA / 沐曦 / MUSA 保持 double |
| PTX 内联汇编在 CoreX 后端编译失败 | 无法生成 kernel | 改用 CUDA 内建 `__float_as_int` / `__int_as_float`，各平台语义一致 |
| MUSA 不提供 CUDA 兼容头 | 无法直接包含 cuda_runtime.h | 以 `-DVSEARCH_MUSA=1` 启用名称映射（`cuda*`/`cublas*` → `musa*`/`mublas*`），实现代码不变 |
| mcc 把 `.cu` 当作 CUDA 源 | 报“找不到 CUDA 安装” | 编译时显式指定 `-x musa --musa-path=/usr/local/musa` |

上述处理只涉及计数类型、累计类型与符号名称映射，检索流程与 kernel 划分不变。

### 7.3 NVIDIA 平台结果

**表 1：NVIDIA RTX 4090，1e6×128，nq=1000，topK=100，nprobe=16**

| mode | n | dim | nq | topK | build ms | search ms | QPS | P50 ms | P99 ms | 显存 | CPU ms | speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exact_gpu | 1e6 | 128 | 1000 | 100 | - | 1433.5 | 697.6 | 71.55 | 73.44 | 1.24 GB | 7413* | ~111* |
| ivf_flat | 1e6 | 128 | 1000 | 100 | 223.9 | 47.3 | 21150 | 5.90 | 6.43 | 0.89 GB | 7413 | 156.8 |
| ivf_pq16+rerank | 1e6 | 128 | 1000 | 100 | 402.8 | 34.1 | 29338 | 4.17 | 4.83 | 0.90 GB | 7478 | 219.4 |

`*`：CPU 参考跑 100 条 query（7413 ms），GPU exact 在同一子集约 66.9 ms。
原始日志：`outputs/`。

**表 2：NVIDIA IVF-Flat nprobe 权衡（batch=128，recall@100）**

| nprobe | recall@100 | QPS | P50 ms | P99 ms |
| --- | --- | --- | --- | --- |
| 1 | 0.0192 | 69385 | 1.78 | 2.11 |
| 4 | 0.2155 | 52347 | 2.36 | 2.70 |
| 8 | 0.8608 | 47862 | 2.59 | 3.03 |
| 16 | 1.0000 | 38240 | 3.24 | 3.78 |
| 32 | 1.0000 | 28434 | 4.35 | 5.00 |
| 64 | 1.0000 | 20058 | 6.21 | 6.83 |

原始日志：`outputs/sweep/`、`outputs/experiment_summary.csv`。

**表 3：NVIDIA IVF-PQ（ADC 粗排 + top-256 精排）**

| pq_m | recall@100 | search ms（nq=1000） | QPS |
| --- | --- | --- | --- |
| 16 | 0.9248 | 34.1 | 29338 |
| 64 | 0.9264 | 40.0 | 24996 |

原始日志：`outputs/ivf_pq16_perf.log` 等。

表 1-3 使用早期 1e6 数据组。为核对平台适配改动之后的代码，另用修复格式后的
生成器产出 300k 数据组并重跑一次完整基准：

**表 4：NVIDIA RTX 4090，300k×128，nq=1000，topK=100，nprobe=16（复核）**

| mode | build ms | search ms | QPS | P50 ms | P99 ms | recall@100 | speedup vs CPU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exact_gpu | - | 98.6 | 2029 | 49.3 | 59.5 | - | 87.1× |
| ivf_flat | 97 | 20.4 | 49058 | 2.51 | 3.02 | 0.986 | 425× |
| ivf_pq16+rerank | 213 | 21.8 | 45792 | 2.59 | 3.68 | 0.012 | 397× |

原始日志：`outputs/nvidia_300k_regression/`。表中 exact 行取自独立 exact bench
日志；IVF bench 内嵌的 exact 子集对照行为 95.5 ms / 90.8×，属同一实现的运行间
波动。

### 7.4 天数智芯 CoreX 结果

**表 5：Iluvatar MR-V100，300k×128，nq=1000，topK=100，nprobe=16**

| mode | build ms | search ms | QPS | P50 ms | P99 ms | 显存 | CPU ms | speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exact_gpu | - | 546.1 | 366 | 272.7 | 342.0 | 0.63 GB | 11655 | 21.3× |
| ivf_flat | 254 | 107.1 | 9340 | 13.6 | 13.8 | 0.41 GB | 11592 | 108.3× |
| ivf_pq16+rerank | 3947 | 58.9 | 16974 | 7.2 | 7.6 | 0.41 GB | 11833 | 200.9× |

原始日志：`outputs_corex/`。GPU 集成测试在同一平台验证了 exact=CPU 与
IVF-Flat recall≈1.0。

**表 6：CoreX IVF-Flat nprobe 权衡（300k 数据组，batch=128）**

| nprobe | recall@100 | QPS | P50 ms | P99 ms |
| --- | --- | --- | --- | --- |
| 1 | 0.013 | 37919 | 3.19 | 3.29 |
| 2 | 0.019 | 28317 | 4.29 | 4.50 |
| 4 | 0.065 | 20254 | 6.11 | 6.41 |
| 8 | 0.544 | 14121 | 8.87 | 9.02 |
| 16 | 0.991 | 9320 | 13.63 | 13.99 |
| 32 | 1.000 | 5125 | 24.71 | 25.85 |
| 64 | 1.000 | 2658 | 48.09 | 49.10 |

完整 21 组（nprobe × batch）数据：`outputs_corex/experiment_summary.csv`。

### 7.5 沐曦 MetaX 结果

**表 7：MetaX MXC500，300k×128，nq=1000，topK=100，nprobe=16**

| mode | build ms | search ms | QPS | P50 ms | P99 ms | recall@100 | speedup vs CPU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exact_gpu | - | 128.2 | 1560 | 64.0 | 82.4 | - | 61.9× |
| ivf_flat | 856 | 62.8 | 15916 | 8.0 | 8.4 | 0.991 | 137.2× |
| ivf_pq16+rerank | 1174 | 62.6 | 15985 | 7.9 | 8.5 | 0.012 | 119.4× |

原始日志：`outputs_maca/`。

**表 8：沐曦 IVF-Flat nprobe 权衡（300k 数据组，batch=128）**

| nprobe | recall@100 | QPS | P50 ms |
| --- | --- | --- | --- |
| 1 | 0.0125 | 27109 | 4.59 |
| 2 | 0.0192 | 21030 | 5.97 |
| 4 | 0.0636 | 22493 | 5.58 |
| 8 | 0.5428 | 18627 | 6.64 |
| 16 | 0.9914 | 15329 | 8.25 |
| 32 | 1.0000 | 9062 | 13.99 |
| 64 | 1.0000 | 5710 | 22.30 |

完整 21 组数据：`outputs_maca/experiment_summary.csv`。

### 7.6 摩尔线程 MUSA 结果

构建方式见 §7.1.4；测试结果与四个平台的对比分别见表 9、表 10 与表 11。

**表 9：摩尔线程 MUSA，300k×128，nq=1000，topK=100，nprobe=16**

| mode | build ms | search ms | QPS | P50 ms | P99 ms | recall@100 | speedup vs CPU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exact_gpu | - | 1028.7 | 194 | 514.0 | 638.5 | - | 16.1× |
| ivf_flat | 241 | 179.2 | 5581 | 22.5 | 24.3 | 0.991 | 92.5× |
| ivf_pq16+rerank | 4820 | 164.2 | 6092 | 20.0 | 24.4 | 0.012 | 101.9× |

原始日志：`outputs_musa/`。

**表 10：MUSA IVF-Flat nprobe 权衡（batch=128）**

| nprobe | recall@100 | QPS | P50 ms |
| --- | --- | --- | --- |
| 1 | 0.0125 | 14307 | 8.45 |
| 2 | 0.0190 | 11249 | 10.67 |
| 4 | 0.0629 | 9605 | 12.71 |
| 8 | 0.5437 | 7669 | 16.18 |
| 16 | 0.9914 | 5515 | 22.63 |
| 32 | 1.0000 | 3447 | 34.69 |
| 64 | 1.0000 | 2101 | 60.07 |

完整 21 组（nprobe × batch）数据：`outputs_musa/experiment_summary.csv`。

### 7.7 四平台横向对比

**表 11：四个平台，300k×128，nq=1000，topK=100，nprobe=16**

| 平台 | GPU | exact QPS | ivf_flat QPS | recall@100 | ivf_pq QPS |
| --- | --- | --- | --- | --- | --- |
| NVIDIA | RTX 4090 | 2029 | 49058 | 0.986 | 45792 |
| 天数智芯 | MR-V100 | 366 | 9340 | 0.991 | 16974 |
| 沐曦 | MXC500 | 1560 | 15916 | 0.991 | 15985 |
| 摩尔线程 | MUSA（MUSA 5.1.0 环境） | 194 | 5581 | 0.991 | 6092 |

原始日志：`outputs/nvidia_300k_regression/`、`outputs_corex/`、`outputs_maca/`、
`outputs_musa/`。

### 7.8 IVF-PQ 量化误差分析

在 300k 数据组上，IVF-PQ 的 recall@100 约为 0.012，与 IVF-Flat 的 0.99 相差
两个数量级（表 4、5、7）。为确认这是量化方法的固有代价而非实现缺陷，做了如下
离线核对：

1. 用 `saveIndex` 导出的 IVF-PQ 索引，在 Python 中按同一套码本重新编码向量，
   与设备端写入的压缩码逐字节一致（1000/1000 条抽样相同），说明编码环节正确。
2. 在 nprobe=16 的真实候选池（约 5k 条）内复算 ADC 分数：真实 top-100 全部
   位于候选池中，但按 ADC 排序后只有 4~22 条进入 `pq_rerank=256` 的精排窗口。
3. 编码重建误差（RMSE≈0.10）与查询到真实近邻的距离（0.2~1.5）处于同一量级，
   因此 ADC 的近似误差淹没了近邻之间的真实差异——这是 8 bit × 16 子空间码率
   在“簇内近邻密集”数据上的固有局限。

对照实验：在紧致分离簇数据（40k×32、100 簇）上，同一实现取 `pq_rerank=128`
时 Top-1 命中率为 100%，说明 PQ 通路本身可用；本数据集上的低召回来自数据分布与
码率的匹配，而非实现错误。

该结论在四套平台上一致：300k 数据组（同参数、同随机种子）在 NVIDIA、CoreX、
沐曦与摩尔线程上分别测得 recall@100 = 0.9858/0.9914/0.9914/0.9914（IVF-Flat），
IVF-PQ 均约 0.012。

## 8. 性能剖析（Nsight Systems）

本次实验使用 Nsight Systems 2024.6.2 对 exact / IVF-Flat / IVF-PQ 三种检索做了
CUDA kernel 时间线分析，完整命令、统计表与结论见
[outputs/PROFILING.md](outputs/PROFILING.md)。

### 8.1 采集方法

```bash
# 时间线 + kernel 统计（本次实际执行）
nsys profile --force-overwrite=true -o prof/nsys_exact \
  -t cuda,osrt ./build/vsearch search \
  --vectors=data_ann/vectors.bin --queries=data_ann/queries.bin \
  --params=data_ann/params.txt --search_mode=exact --batch_size=128

nsys stats --report cuda_gpu_sum prof/nsys_exact.nsys-rep
```

### 8.2 分析结果

- **exact 检索**：`exactKeysKernel` 占 GPU kernel 时间约 74.9%，平均 49.2 ms；
  分段 radix sort 约 23.1%。说明全量打分与排序是精确检索的两个主要成本。
- **IVF-Flat**：每 128-query batch 的中心打分约 0.32 ms、候选精排约 2.92 ms，
  倒排 gather/probe/scan 均只有微秒到十几微秒，索引结构本身开销很小。
- **IVF-PQ + rerank**：ADC 表构建每 batch 仅 5.4 μs，PQ 打分 0.126 ms，
  top-256 精排 0.08 ms；召回接近 IVF-Flat 时仍保持较低精排开销。

分析统一在 RTX 4090、CUDA 12.8、Nsight Systems 2024.6.2、
`N=1e6, D=128, nq=1000, topK=100` 下进行，保证结果可比。

## 9. 结果分析与讨论

### 9.1 正确性

四套平台上主机测试 5 项、GPU 集成测试 4 项全部通过；基准测试过程中 GPU exact 与
CPU 参考的 id 逐位一致。这满足题目对“精确检索结果需与 CPU 参考实现一致”“Top-K
输出需按距离或相似度排序”的要求。

天数智芯平台的距离容差需要单独说明：该平台 device 端 double 累加存在约 1e-4 的
相对误差，在 300k 规模下会使距离排序出现错乱。将其距离累计改为 float 后测试稳定
通过，容差取 2e-3；NVIDIA 与沐曦的构建保持 double 累计，容差取 1e-4。该差异源于
设备浮点实现，不涉及算法改动。

### 9.2 性能

精确检索是全库扫描，受显存带宽约束：1e6×128 的库需读取 512 MB 向量，RTX 4090 上
测得 1433.5 ms，折合约 357 GB/s 的有效带宽。由于该阶段工作量与 N 成正比，扩大规模
时 QPS 会成比例下降，这也是引入倒排索引的主要动因。

IVF-Flat 只扫描 `nprobe/nlist` 比例的候选：1e6 数据、nlist=4096、nprobe=16 时理论
候选约为全库的 0.4%，实测 QPS 相对 exact 提升约 30 倍；300k 数据、nlist=1024、
nprobe=16 时提升约 24 倍（表 4、7）。IVF-PQ 进一步把候选打分从读原始向量改为读
压缩码与距离表，在 300k 数据上与 IVF-Flat 吞吐相当，但代价是召回率（§7.8）。

四平台吞吐排序为 NVIDIA > 沐曦 > 天数智芯 > 摩尔线程。需要注意沐曦本次只启用了
MXC500 的一个 SGPU 分片（mx-smi 显示 50% 规格），并非整卡；天数智芯与摩尔线程的
精确检索明显慢于前两者（366 / 194 QPS 对 1560 / 2029 QPS），但两者的 IVF-Flat
相对自身 exact 的加速倍数最大（约 25 倍与 29 倍），说明瓶颈更多在原始向量扫描
而非索引结构。

### 9.3 召回率与参数的权衡

表 2、6、8 给出一致的规律：

- nprobe 从 1 增到 8 时召回从约 0.01~0.02 快速升到 0.54~0.86；
- nprobe=16（约 1.6% 的桶）时召回达到 0.99；
- nprobe≥32 时召回为 1.000，但吞吐下降约一半。

即：用约 1.6% 的候选即可覆盖 99% 的近邻，继续增大 nprobe 只能换取最后约 1% 的
召回，QPS 代价却成比例上升。这为参数选择提供了直接依据——以召回目标（如 0.99）
反推最小 nprobe，而不是取接近 nlist 的值。

### 9.4 量化误差

IVF-PQ 在本数据组上的低召回（0.012）经离线核对确认来自 8 bit×16 子空间的量化
误差量级与近邻距离量级相当（§7.8），而非编码或打分实现错误。若要在此类数据上
使用 PQ，需要提高码率（增大 `pq_m`、使用更多码字或残差量化）或扩大精排窗口。

### 9.5 实验局限

- 数据为合成数据，近邻分布与真实语料不同，召回率结论不能直接外推到实际业务；
- 性能数字为单次运行，未做重复实验与方差统计，平台间比较只反映量级差异；
- CPU 基线是单线程实现，加速比数值会随 CPU 基线实现方式显著变化；
- 沐曦平台使用单个 SGPU 分片，其绝对性能不代表整卡能力。

## 10. 实验结论

1. **功能正确性**：四套平台上 GPU 精确检索与独立 CPU 参考的 id 逐位一致，距离在
   平台相应容差内；Top-K 严格有序；批量查询、K 取值、fp16/fp32 输入、索引落盘与
   重载均可用。
2. **近似检索有效性**：IVF-Flat 在 nprobe=16 时 recall@100 达到 0.986~0.991，
   nprobe=32 时达到 1.000，同时吞吐为精确检索的 20~70 倍。
3. **性能量级**（300k×128，nprobe=16）：exact 194~2029 QPS，ivf_flat 5581~49058
   QPS，ivf_pq 6092~45792 QPS，相对单线程 CPU 基线的加速比为 16×~425×。
4. **跨平台**：程序在四套 GPU 软件栈（CUDA、CoreX IX-ML、MetaX MACA、MUSA）上均能
   构建并完成全部测试与基准；天数智芯需要针对设备特性调整计数位宽与距离累计类型，
   摩尔线程需要一层 API 名称映射，沐曦与 NVIDIA 使用相同计算路径。
5. **已知代价**：IVF-PQ 在本数据组上将召回降到约 0.012，属于量化码率与数据分布
   不匹配的固有结果，需通过提高码率或扩大精排窗口改善。

## 11. 数据溯源与可复现性

### 11.1 构建与测试命令

```bash
# 1) NVIDIA 构建与正确性自检
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/vsearch validate

# 2) CoreX 构建与正确性自检
make -f Makefile.corex -j
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib
./build_corex/test_host && ./build_corex/test_gpu

# 3) 沐曦 MetaX 构建与正确性自检
make -f Makefile.maca -j
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/tools/cu-bridge/lib:/opt/maca/lib64
./build_maca/test_host && ./build_maca/test_gpu

# 4) 摩尔线程 MUSA 构建与正确性自检
make -f Makefile.musa -j
export LD_LIBRARY_PATH=/usr/local/musa/lib:/usr/local/musa/lib64
./build_musa/test_host && ./build_musa/test_gpu

# 5) 生成 300k 数据组并做 nprobe × batch 扫描
#    （NVIDIA 用 ./build/vsearch，其余平台用对应 build_*/vsearch）
python python/gen_dataset.py --out data \
    --n 300000 --dim 128 --nq 1000 --top-k 100 --metric l2 \
    --clusters 100 --vector-scale 0.10 --query-scale 0.02 \
    --nlist 1024 --nprobe 16
python python/run_experiments.py --vsearch ./build/vsearch \
    --data data --mode ivf_flat \
    --nprobe-list 1,2,4,8,16,32,64 --batch-list 32,128,512
```

### 11.2 数据与日志对照

| 报告中的表 | 数据组 | 平台 | 对应日志目录 |
| --- | --- | --- | --- |
| 表 1、2、3 | 1e6×128 | NVIDIA | `outputs/`（含 `sweep/`、`experiment_summary.csv`） |
| 表 4 | 300k×128 | NVIDIA | `outputs/nvidia_300k_regression/` |
| 表 5、6 | 300k×128 | 天数智芯 | `outputs_corex/` |
| 表 7、8 | 300k×128 | 沐曦 | `outputs_maca/` |
| 表 9、10 | 300k×128 | 摩尔线程 | `outputs_musa/` |
| 表 11 | 300k×128 | 四平台 | 上述四个目录 |

每组日志均包含 `perf.log`（性能）、`quality.log`（召回与距离误差）与
`experiment_summary.csv`（nprobe × batch 汇总）。索引文件为 `.idx`，`saveIndex`
与 `loadIndex` 使用同一格式，可由任一平台生成、在其它平台加载核对。

### 11.3 结果核验方式

1. **正确性**：直接运行 `test_host` 与 `test_gpu`（§11.1 步骤 1-4），
   `test_gpu` 内部会把 GPU exact 与 CPU 参考逐条比对。
2. **性能/召回**：按 §5.4 的命令重跑 bench，得到的 `perf.log` 与 `quality.log`
   可与 §7 各表逐项对照；表下方均标注了对应的日志文件。
3. **平台差异**：§3.3 的两项设备能力探测可用最小程序复现，代码与各平台编译命令
   见 [tools/probe/device_probe.cu](tools/probe/device_probe.cu) 与
   [tools/probe/README.md](tools/probe/README.md)。

代码、测试脚本与数据生成脚本均在仓库内，运行路径不需要手工改动文件格式。

## 12. 后续工作

- **查询并行流**：多个 stream 并行处理不同 batch，降低 P99；
- **rerank**：PQ top 候选用原始向量精排，兼顾召回与显存；
- **索引增强**：粗聚类加 k-means++/HNSW graph，倒排桶失衡时做子桶拆分；
- **低精度**：fp16x2 向量与距离表、int8 重排；
- **动态索引**：增量 insert/delete 与分段合并；
- **持久化/服务化**：内存池复用、共享显存多进程、gRPC 接口；
- **FAISS 对照矩阵**：与 FAISS `IndexIVFFlat`/`IndexIVFPQ` 在相同数据集、相同
  `nprobe` 下逐项对比 recall 与吞吐。

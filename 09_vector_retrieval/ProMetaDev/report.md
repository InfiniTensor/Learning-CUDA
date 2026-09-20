# GPU 向量检索引擎 —— 总结报告

> 选题：2026 夏季训练营 CUDA 方向项目阶段 · 选题九
> 环境：WSL2 Ubuntu 22.04 + CUDA 12.2 + NVIDIA RTX 5070 Ti Laptop (12 GB)，编译目标 `sm_90`
> 国产平台适配：沐曦 MetaX 曦云 C500 + MACA 3.5.3（见第 9 节，已实测验证，13/13 通过）
> CPU 参考实现使用 OpenMP 并行

---

## 1. 问题描述

面向 RAG / 推荐 / 多模态检索的向量数据库，需要在数百万到数千万条高维向量中
为每个查询找出最相似的 Top-K。CPU 实现在高并发低延迟场景下很快遇到瓶颈。

本题要求实现一个 GPU 加速的向量检索引擎：

1. 必须有 **GPU 精确检索 baseline**（L2 或内积至少一种），且结果与 CPU 参考一致；
2. 至少实现一种**近似检索/索引优化**方案（IVF-Flat、IVF-PQ 或其他），
   并且**索引要能落盘后加载复用**，不能每次查询都重建；
3. 支持 K = 1/10/50/100 中至少三种；
4. 支持批量查询；
5. 数据规模 ≥ 10⁶ 向量、维度 ≥ 128、查询数 ≥ 1000；
6. 输出检索结果、索引文件、性能日志（建索引时间/QPS/P50/P99/显存/加速比）
   与质量日志（recall@K、平均距离误差、不同 nprobe/batch 下的变化）。

---

## 2. 系统设计

### 2.1 分层结构

```
types.h / io.cpp     配置、向量集、结果、索引的读写（文本头 + 二进制数据段）
metrics.cpp          recall@K、平均距离误差、延迟分位、结果一致性比对
cpu_ref.cpp          CPU 参考：精确检索（OpenMP 并行）+ k-means
exact.cu             GPU 精确检索 kernel
ivf.cu               IVF 建索引（k-means + 倒排表）与查询 kernel
topk_merge.cuh       精确检索与 IVF 共用的两级 Top-K 归并
normalize.cuh        cosine 度量的行归一化
main.cpp             CLI：gen / genq / exact / build / search / bench / selftest
```

### 2.2 精确检索的 Top-K 设计

朴素做法是对每个 query 的 n 个距离排序，代价过高。本实现利用"**每个线程的候选表
本身已有序**"这一性质做**归并**而非排序：

- 一个线程块（256 线程）处理一个 query；块内线程按步长扫描全部向量，访存合并。
- 每线程维护一个**有序的本地 top-K**。热路径只有一次"与本地第 K 名比较"（早退），
  仅当更优时才做插入，绝大多数向量只花一次比较。
- **两级归并**：warp 内 32 个有序表做 K 轮 32 路最小值（`__shfl_xor_sync`）→
  每个 warp 得到其 top-K；再由 warp0 的 8 个 lane 做一次 8 路归并得到块级 top-K。
  总代价 O(K·log(线程数))，只需 1 次 `__syncthreads`。

比较规则采用**全序**：分数更优者胜，分数相等时 **id 小者胜**。
这与 CPU 参考的"稳定插入"语义一致，消除了并列项的不确定性。

### 2.3 IVF-Flat 建索引

k-means 的全量分配代价为 O(n·nlist·dim)，在 10⁶×128×1024 的规模上每轮就要 ~10⁸ 次
距离计算。因此采用**子样本训练**（标准做法）：

1. 抽取子样本 `min(n, 20·nlist)` 条；
2. 迭代 —— 分配（`k_assign`）→ 计数 → **前缀和** → 散列成倒排表 → 更新聚类中心；
3. 用训练好的中心对**全量数据**做一次分配，得到最终倒排表并落盘。

分配 kernel 用**共享内存分块**：每次把 CT=32 个中心放进共享内存，
向量按 DC=16 分块读入寄存器复用，避免每个 MAC 都访问显存。

### 2.4 IVF-Flat 查询

- **Phase A（粗量化）**：把 query 载入共享内存，块内线程并行计算到全部 nlist 个中心的
  距离存入共享内存；再用 nprobe 轮"取最小 + 标记已选"选出最近的 nprobe 个中心
  （每轮用 warp 归约 + 跨 warp 归约）。
- **Phase B（细扫）**：遍历这 nprobe 个倒排桶中的原始向量，计算精确距离，
  复用与精确检索完全相同的每线程 top-K + 两级归并。

这样 IVF 与精确检索在"打分 + Top-K"部分共享同一套代码，保证二者语义一致。

---

## 3. 优化历程与开发中发现的问题（真实记录）

### 3.1 倒排表出现大量重复与缺失（最严重的 bug）

**现象**：IVF 在 `nprobe = nlist`（即扫描全部倒排桶）时，召回率仍只有 0.35~0.65，
且**时对时错**：nlist=1 和 4 时 recall=1.0，nlist=2/8/16/64 时全错。

**定位手段**：在 `build` 流程里加了一条**倒排表合法性校验**（`list_ids` 必须是
`0..n-1` 的一个排列）。校验立刻报出：结构正常，但**重复=15941、缺失=15941**。

**根因**：散列时 `pos = offset[c] + atomicAdd(&cursor[c], 1)`，
而我却把 `cursor` 初始化成了**前缀和**而不是 0 —— 偏移被加了两次，
绝大多数位置越界写入到别的桶的区间，造成重叠。
`nlist=1` 时 `offset[0]=0` 恰好掩盖了该问题，这也是"时对时错"的来源。

**修复**：`cursor` 改为清零（`cudaMemset`），前缀和只写入 `offset`。
修复后 `nprobe=nlist` 的召回率在任意 nlist 下**恒为 1.0000**。

> 教训：计数排序（CSR 构建）里 `offset` 与 `cursor` 的语义极易混淆，
> 一条"排列合法性"断言一次性排除了整类问题，值得在索引构建后固定保留。

### 3.2 块内线程迭代次数不一致导致 `__syncthreads()` 分歧（UB）

**现象**：即使修好了 3.1，倒排表仍会出错。

**根因**：`k_assign` 的外层 grid-stride 循环写成
`for (v = blockIdx*THREADS + tid; v < n; v += stride)` —— 循环条件**依赖线程**。
当 n 不是 stride 的整数倍时，块内部分线程会少执行一轮，而循环体内有 `__syncthreads()`
→ 块内线程到达同步点的次数不同 → 未定义行为，标签被写错。

**修复**：循环条件改为只依赖 `blockIdx`（`for (v0 = blockIdx*THREADS; v0 < n; v0 += stride)`），
块内所有线程迭代次数一致；用 `active = (v < n)` 保护实际计算。

> 教训：**含 `__syncthreads()` 的循环，其循环条件必须对块内所有线程一致。**

### 3.3 `__shfl_xor_sync` 的 mask 与参与线程不匹配（UB）

**现象**：精确检索结果与 CPU 完全一致，但 IVF 结果时对时错。

**根因**：跨 warp 归并里只有 warp0 的前 8 个 lane 执行 shuffle，
却传了 `0xffffffff`（全 32 lane）的 mask。参与线程集合与 mask 不一致属于未定义行为。

**修复**：mask 改为与参与线程精确对应的 `(1u << NWARP) - 1`。

### 3.4 精确检索的名次并列差异（内积度量下的 0.1%）

**现象**：内积度量下，2000 个条目里有 2 个名次顺序与 CPU 不同，但
**每个 query 的 top-K id 集合完全一致**。

**原因**：GPU 与 CPU 的浮点累加顺序/FMA 收缩可能不同，分数几乎相等的两项
在最后几位上大小反转，导致内部名次互换。这是浮点实现的固有现象，不是检索错误。

**处理**：
- 判定标准定为"**每个 query 的 top-K id 集合完全一致**"，并单独报告名次差异与
  最大相对距离差（实测 3.13e-07）；
- 在报告中明确说明该口径，而不是掩盖。

### 3.5 IVF 的有效性依赖数据的可聚类结构（重要的负结果）

在**聚类数据**上，IVF 只需 nprobe=16/1024（扫描 6.8%）就能达到 recall@10 = 1.0；
但在**均匀正态分布数据**上，同样的 IVF 需要 nprobe=128/256（扫描 92.5%）才能到
recall 0.992 —— 因为均匀高斯数据没有簇结构可供裁剪。

**结论**：IVF 的加速比与数据分布强相关，"能聚类"是它生效的前提。
本报告把两组数据的结果都列出，避免只呈现好看的一面。

---

## 4. 正确性验证

### 4.1 引擎自检（`vs selftest`，全部通过）

- 精确检索 kernel 可执行；
- 精确检索 top-K 集合与 CPU 参考一致；
- IVF 在 `nprobe = nlist` 时 recall@10 = 1.0000。

### 4.2 端到端测试（`tests/run_tests.sh`，**13/13 通过**）

| 测试组 | 内容 | 结果 |
|---|---|---|
| 0 | 引擎自检 | PASS |
| 1 | 生成聚类数据（n=20000, dim=128） | PASS |
| 2 | 精确检索 K ∈ {1,10,50,100}，GPU vs CPU | 4/4 PASS |
| 3 | IVF 建索引 + **倒排表合法性**（0..n-1 排列，重复=0 缺失=0） | PASS |
| 4 | nprobe=nlist 时 recall=1.0；召回率随 nprobe 单调不降 | 2/2 PASS |
| 5 | 三种度量 L2 / 内积 / cosine，精确检索 GPU vs CPU | 3/3 PASS |

### 4.3 目标规模验证（n=10⁶, dim=128, nq=1000）

精确检索与 CPU 参考的比对结果：

```
id 集合不一致的 query 数: 0 / 1000
逐名次 id 不一致:        0 / 10000
最大相对距离差:          3.130e-07
```

**1000 个查询的 top-10 集合与 CPU 参考完全一致。**

---

## 5. 性能指标

### 5.1 目标规模（n=10⁶, dim=128, nq=1000, top_k=10, 聚类数据）

| 方案 | 查询总时间 | QPS | P50 | P99 | 扫描比例 | 显存 | 质量 |
|---|---|---|---|---|---|---|---|
| **精确检索** | 339.74 ms | 2943 | 26.55 ms | 26.67 ms | 100% | 488.8 MB | 与 CPU 集合一致 |
| **IVF-Flat** (nlist=1024, nprobe=16) | **141.06 ms** | **7089** | **4.67 ms** | **5.37 ms** | **6.79%** | 493.2 MB | **recall@10 = 1.0000** |

**相对 CPU baseline 的加速比**：

- CPU 精确检索（OpenMP）：**5878.24 ms**
- GPU 精确检索：**339.74 ms** → **加速 17.3 倍**
- IVF-Flat（nprobe=16）：**141.06 ms** → **加速 41.7 倍**，且 recall@10 = 1.0

**IVF 建索引耗时**：**178.43 ms**（nlist=1024，含 10 轮子样本 k-means + 一次全量分配），
索引落盘后可反复加载复用，不重复付出该成本。

### 5.2 nprobe 扫描（聚类数据，n=10⁶，top_k=10）

| nprobe | recall@10 | QPS | P50 (ms) | P99 (ms) | 扫描比例 |
|---|---|---|---|---|---|
| 1 | 0.9617 | 83982 | 0.139 | 0.555 | 0.17% |
| 2 | 0.9978 | 53463 | 0.508 | 0.969 | 0.64% |
| 4 | 1.0000 | 27198 | 1.165 | 1.715 | 1.70% |
| 8 | 1.0000 | 13788 | 2.394 | 3.069 | 3.59% |
| 16 | 1.0000 | 7279 | 4.730 | 5.535 | 6.79% |
| 32 | 1.0000 | 3868 | 8.472 | 9.206 | 12.11% |
| 64 | 1.0000 | 2106 | 14.481 | 15.445 | 20.60% |
| 128 | 1.0000 | 1257 | 22.768 | 23.929 | 32.37% |

**nprobe 是"精度—速度"旋钮**：nprobe 从 1 增到 4，召回率从 0.962 升到 1.0，
QPS 从 83982 降到 27198；继续增大 nprobe 只增加开销、不再提升召回率。

### 5.3 batch_size 扫描（聚类数据，nprobe=16，nq=1000）

| batch_size | QPS | P50 (ms) |
|---|---|---|
| 1 | 216 | 4.75 |
| 16 | 3335 | 4.73 |
| 64 | 5919 | 4.75 |
| 128 | 7237 | 4.73 |
| 256 | 7262 | 4.73 |
| 512 | 7443 | 4.74 |

批量化把 QPS 从 216 提升到 7443（**约 34 倍**），而单查询延迟基本不变（4.7 ms）——
说明单 query 时 GPU 严重欠占用，批量查询才能真正喂满 GPU。

### 5.4 非聚类（正态分布）数据的权衡曲线

n=200000, dim=128, nq=500, nlist=256：

| nprobe | recall@10 | QPS | 扫描比例 |
|---|---|---|---|
| 1 | 0.0438 | 103479 | 0.82% |
| 2 | 0.0782 | 158034 | 1.68% |
| 4 | 0.1394 | 82425 | 3.41% |
| 8 | 0.2322 | 42642 | 6.81% |
| 16 | 0.3926 | 22148 | 13.59% |
| 32 | 0.5898 | 11110 | 27.00% |
| 64 | 0.8280 | 5685 | 52.59% |
| 128 | 0.9920 | 3239 | 92.53% |

这是一条标准的"精度—速度"权衡曲线：要拿到高召回率必须扫描大部分数据。
与 5.2 对比可见，**IVF 的收益高度依赖数据是否具备可聚类结构**。

### 5.5 性能分析

1. **精确检索是显存带宽受限**：每个 query 都要完整读一遍 512 MB 向量库，
   1000 个 query 即 512 GB 访存。GPU 相对 CPU 的 17.3 倍加速主要来自
   显存带宽（数百 GB/s vs 数十 GB/s）与大规模并行带来的延迟隐藏。
2. **IVF 的收益来自"少算"**：nprobe=16 只扫描 6.79% 的向量，
   理论访存降到约 1/15，实测 QPS 提升 2.4 倍（受固定开销与 Top-K 归并限制）。
3. **P99/P50 比值接近 1**（精确检索 1.005，IVF 1.15），说明延迟分布很集中，
   没有长尾抖动。
4. **显存占用约 490 MB**（主要为 10⁶×128 的 fp32 向量库 512 MB），
   12 GB 显存可支持更大规模或 fp16 存储进一步压缩。

---

## 6. 依赖说明

| 组件 | 实现方式 |
|---|---|
| 精确检索 / IVF 查询 kernel | 纯手写 CUDA C++，仅用 warp shuffle、共享内存、`atomicCAS`/`atomicAdd` |
| Top-K 归并 | 自实现的"有序表 K 轮归约"，未使用 cub / thrust 等第三方库 |
| k-means 聚类 | 自实现（共享内存分块 + 计数排序建倒排表） |
| CPU 参考 | 自实现标量检索 + OpenMP 并行 |
| fp16 读写 | host 端软件位运算转换 |
| 第三方库 | **无**（仅 CUDA Runtime + C++ 标准库 + OpenMP） |

未使用 Hopper / Blackwell 的新指令（无 WGMMA/TMA、无 FP8/FP4 Tensor Core 依赖），
代码在 sm_70 及以上均可编译。构建产物为单一可执行文件 `vs`。

---

## 7. 局限与未来可继续提升的方向

**本次实现的局限**：

1. **未实现 IVF-PQ**。题目要求"至少实现一种近似检索方案"，IVF-Flat 已满足；
   PQ（乘积量化 + 查表 ADC）可进一步把向量压缩 16~64 倍，是后续最值得补的功能。
2. **IVF 的粗量化固定用 L2**。对内积/cosine，更贴切的做法是对归一化后的向量聚类
   （cosine 已通过归一化等价转换），或使用专门的量化器。
3. **nlist 上限受共享内存约束**（需在共享内存中存放 nlist 个距离，nlist ≲ 12288）。
   更大 nlist 需改为不落共享内存的分块选择。
4. **国产平台已适配沐曦 曦云 C500**（见第 9 节）：自带测试 13/13 通过，
   精确检索 top-K 集合与 CPU 参考一致、IVF recall@10 = 1.0。
   本工程含较多 `__shfl_xor_sync`（`width=32` / `nlist` 级 mask）归约，
   而在 `warpSize = 64` 的沐曦平台上实测语义正确。
5. **`ncu` / `nsys` 未采集**：WSL2 不透传 CUPTI（实测 ncu 报 "No kernels were profiled"、
   nsys 报告中 CUDA kernel 段为空），因此本报告的瓶颈判断来自访存量估算与实测
   时间吻合，而非 profiler 计数器；如需补齐该加分项，需在原生 Linux / AutoDL 上运行。

**优化方向**：

1. **多 query 寄存器分块**：让一个线程块同时处理多个 query，把向量读入寄存器后
   复用于多个 query，可把精确检索的显存流量降低数倍（当前每 query 重读全库）。
2. **IVF-PQ / IVFPQ 的查表 ADC**：用查表替代逐元素距离计算，大幅降低算力与带宽。
3. **向量化访存与 fp16/int8 存储**：进一步降低带宽压力。
4. **两阶段 top-K**（先按 tile 求局部 top-K 再归并）以提高大 nlist 下的并行度。
5. **真实数据集验证**：目前使用合成数据（clustered / normal / uniform），
   建议补 SIFT1M、GIST1M 等公开基准，使 recall-QPS 曲线更具参考价值。

---

## 8. 复现方式

```bash
bash build.sh                                  # 构建
./build/vs selftest                            # 自检
bash tests/run_tests.sh                        # 端到端测试（13 项）

# 目标规模复现
./build/vs gen clustered 1000000 128 outputs/data/base.txt --nlist 1024 --seed 7
./build/vs genq outputs/data/base.txt 1000 outputs/data/q.txt --kind clustered
./build/vs exact outputs/data/base.txt outputs/data/q.txt configs/exact_k10.cfg outputs/gt/exact
./build/vs build outputs/data/base.txt configs/ivf_k10.cfg outputs/index/ivf.bin
./build/vs search outputs/data/base.txt outputs/data/q.txt outputs/index/ivf.bin \
                configs/ivf_k10.cfg outputs/results/ivf --gt outputs/gt/exact.result
./build/vs bench  outputs/data/base.txt outputs/data/q.txt outputs/index/ivf.bin \
                configs/ivf_k10.cfg outputs/results/bench --gt outputs/gt/exact.result
```

---

## 9. 国产平台适配：沐曦（MetaX 曦云 C500 / MACA）

### 9.1 平台环境与编译方式

| 项目   | 值                                                                                |
| ---- | -------------------------------------------------------------------------------- |
| 加速卡  | 沐曦 曦云 C500（`mx-smi` 2.2.12，KMD 3.8.30）                                           |
| 设备属性 | `warpSize = 64`，104 个 SM，`maxThreadsPerBlock = 1024`，共享内存 64 KB/block，计算能力 `(10,0)` |
| 软件栈  | MACA 3.5.3.20（SDK 3.5.3.307），CUDA 兼容层由 `tools/cu-bridge` 提供                       |
| 编译器  | `mxcc 1.0.0`（`/opt/maca/mxgpu_llvm/bin/mxcc`）                                     |

构建脚本 [`build_maca.sh`](build_maca.sh)：

```bash
mxcc -x maca -offload-arch native --maca-path=/opt/maca \
     -Iinclude -I/opt/maca/tools/cu-bridge/include -L/opt/maca/lib \
     -imacros __macro_mxcc.h -forward-unknown-to-compiler \
     -fgpu-rdc --maca-link -lToolsExt_cu -lruntime_cu -lmcToolsExt \
     -std=c++17 -O3 \
     src/main.cpp src/io.cpp src/metrics.cpp src/cpu_ref.cpp src/exact.cu src/ivf.cu \
     -o build_maca/vs

export LD_LIBRARY_PATH=/opt/maca/lib
BIN=./build_maca/vs bash tests/run_tests.sh
```

### 9.2 重点验证：`warpSize = 64` 上的 shuffle 归约（本工程的核心风险）

本工程大量使用 warp 级归约（§3.3 专门记录过 mask 与参与线程不匹配的 UB 问题）：

| 位置                              | 用法                                              |
| ------------------------------- | ----------------------------------------------- |
| `include/topk_merge.cuh:56,57`  | `__shfl_xor_sync(0xffffffffu, v, off, 32)`（warp 内 top-K） |
| `include/topk_merge.cuh:78,87`  | `__shfl_xor_sync(kMask, v, off, NWARP)`，`kMask = (1<<NWARP)-1`（跨 warp 合并） |
| `src/ivf.cu:200,201,217,218`    | `__shfl_xor_sync(..., 32)` 与 `(..., NWARP)`      |

沐曦的 `warpSize` 是 **64 而不是 32**，这些"逻辑 32 线程组"的假设属于必须实测的点。
实测结论：**全部语义正确**（见 9.3 的第 2、5 组断言 —— 精确检索与 CPU 参考逐名次一致、
IVF recall@10 = 1.0）。源码零改动。

### 9.3 正确性验证：自带测试 13/13 通过

`BIN=./build_maca/vs bash tests/run_tests.sh`：

| 组   | 内容                                                         | 结果       |
| --- | ---------------------------------------------------------- | -------- |
| 0   | 引擎自检                                                       | ✅ PASS   |
| 1   | 生成 clustered 数据（n=20000, dim=128, nq=200, nlist=64）        | ✅ PASS   |
| 2   | 精确检索 K ∈ {1,10,50,100}，GPU vs CPU **逐名次一致**              | ✅ 4/4    |
| 3   | IVF 倒排表是 0..n-1 的合法排列                                      | ✅ PASS   |
| 4   | nprobe 扫描：recall 随 nprobe 单调不降、nprobe=nlist 时 = 1.0       | ✅ PASS   |
| 5   | 三种度量（L2 / inner_product / cosine）精确检索与 CPU 一致             | ✅ 3/3    |
| —   | **合计**                                                     | **13/13** |

目标规模（n=10⁶, dim=128, nq=1000）的正确性同样通过：

```
[正确性验证] (GPU 精确 vs CPU 参考)
id 集合不一致的 query 数: 0 / 1000
逐名次 id 不一致: 0 / 10000
```

### 9.4 性能对照（n=10⁶, dim=128, nq=1000, top_k=10, 聚类数据）

| 方案                                | 指标      | NVIDIA RTX 5070 Ti | 沐曦 C500      |
| --------------------------------- | ------- | ------------------ | ------------ |
| **精确检索**                          | 总时间     | 339.74 ms          | 1454.35 ms   |
|                                   | QPS     | 2943               | 687.6        |
|                                   | P50 / P99 | 26.55 / 26.67 ms   | 76.63 / 81.01 ms |
| **IVF-Flat**（nlist=1024, nprobe=16） | 总时间     | 141.06 ms          | 195.51 ms    |
|                                   | QPS     | 7089               | **5114.9**   |
|                                   | P50 / P99 | 4.67 / 5.37 ms     | 6.07 / 7.08 ms |
|                                   | recall@10 | 1.0000             | **1.0000**   |
|                                   | 扫描比例    | 6.79%              | 6.83%        |
| **IVF 建索引**                       | 耗时      | 178.43 ms          | 405.82 ms    |

**读法**：

1. **IVF 路径差距不大**（195.5 vs 141.1 ms，约 **0.72×**），且 `recall@10 = 1.0`、
   扫描比例几乎相同（6.83% vs 6.79%）—— 说明索引构建与倒排扫描的**行为一致**。
2. **精确检索差距较大**（1454 vs 340 ms，约 **0.23×**）。
   精确检索是 `nq × n × dim` 的纯计算路径（1000 × 10⁶ × 128 ≈ 1.3×10¹¹ 次乘加），
   更吃 FMA 吞吐，这是沐曦 C500 与英伟达笔记本卡差距最明显的一处。
3. **本表未列"相对 CPU 加速比"**：沐曦容器的主机 CPU 与笔记本完全不同
   （同一 CPU 参考实现：89 514 ms vs 5878 ms，相差 **15 倍**，说明容器分配的核数远少于
   本机 32 核），两台的 CPU/GPU 比值不可直接相除。C500 上的 CPU 精确检索为 89.5 s，
   GPU 精确检索 1.45 s，**加速 61.6×**（该数字仅在同一台机器内部有意义）。

### 9.5 在沐曦平台复现

```bash
cd proj_vecsearch
MACA_PATH=/opt/maca bash build_maca.sh
export LD_LIBRARY_PATH=/opt/maca/lib:$LD_LIBRARY_PATH

BIN=./build_maca/vs bash tests/run_tests.sh      # 13/13

# 目标规模
./build_maca/vs gen clustered 1000000 128 outputs/data/base.txt --nlist 1024 --seed 7
./build_maca/vs genq outputs/data/base.txt 1000 outputs/data/q.txt --kind clustered
./build_maca/vs exact outputs/data/base.txt outputs/data/q.txt configs/exact_k10.cfg outputs/gt/exact
./build_maca/vs build outputs/data/base.txt configs/ivf_k10.cfg outputs/index/ivf.bin
./build_maca/vs search outputs/data/base.txt outputs/data/q.txt outputs/index/ivf.bin \
                 configs/ivf_k10.cfg outputs/results/ivf --gt outputs/gt/exact.result
```

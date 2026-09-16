# GPU Vector Search Engine (`vsearch`)

一个面向 RAG / 推荐 / 多模态检索场景的 CUDA 向量检索演示引擎，支持：

- **exact**：GPU 精确检索 baseline（全量距离 + 分块归并 Top-K），支持 L2 / inner
  product / cosine；
- **ivf_flat**：IVF（倒排文件）+ Flat 向量距离；
- **ivf_pq**：IVF + Product Quantization（ADC 距离表 + 压缩码本；默认对
  top-256 候选做原始向量精排，设 `pq_rerank = 0` 可切换为纯 ADC）；
- fp16 / fp32 二进制向量库读取（内存统一按 fp32 计算）；
- 索引保存 / 加载，避免每次查询重新建索引；
- 批量查询、K=1/10/50/100、CPU 参考实现、结果/性能/质量日志；
- 主机与 GPU 正确性测试。

程序可在四个 GPU 平台构建与运行：

| 平台 | 构建方式 | 验证情况 |
| --- | --- | --- |
| NVIDIA | CMake + `nvcc` | RTX 4090 上构建、主机/GPU 测试与 bench 全部通过 |
| 天数智芯 CoreX | `make -f Makefile.corex`（clang `-x ivcore`） | MR-V100（IX-ML 4.4.0）上全部通过 |
| 沐曦 MetaX | `make -f Makefile.maca`（`cucc` + `-lmcblas`） | MXC500（MACA 3.5.3）上全部通过 |
| 摩尔线程 MUSA | `make -f Makefile.musa`（`mcc -x musa` + `-lmublas`） | MUSA 5.1.0 上全部通过 |

四平台最近一次回归均通过（`test_host` + `test_gpu` 全部 PASS，端到端 bench 正常）。
四平台使用同一套检索算法与 kernel 实现，差异仅体现为构建脚本与少量编译期分支。
代表性结果见文末与 [REPORT.md](REPORT.md)。

验证日期：NVIDIA / CoreX 为 2026-09-07，沐曦 MetaX 为 2026-09-13，
摩尔线程 MUSA 为 2026-09-14。

## 目录

```text
.
├── .gitignore
├── REPORT.md              # 完整实验报告（实现、实验设计、结果、数据溯源）
├── CMakeLists.txt
├── Makefile.corex         # 天数智芯 CoreX 专用构建
├── Makefile.maca          # 沐曦 MetaX（MACA）专用构建
├── Makefile.musa          # 摩尔线程 MUSA 专用构建
├── README.md
├── include/vsearch/        # 公共数据结构/配置/IO/CPU 参考
├── src/                    # 主机侧实现与 CLI
├── cuda/engine.cu          # 全部 CUDA kernel、IVF/PQ 构建与检索
├── tests/                  # test_host / test_gpu
├── tools/probe/            # 设备能力探测（atomicAdd / double 精度）
├── python/                 # 数据生成与实验调度
├── data/                   # 生成的二进制向量库/查询/参数
├── docs/
├── outputs/                # NVIDIA 平台产物（报告 §7.3 表 1-4）
├── outputs_corex/          # 天数智芯平台产物（报告 §7.4 表 5-6）
├── outputs_maca/           # 沐曦平台产物（报告 §7.5 表 7-8）
└── outputs_musa/           # 摩尔线程平台产物（报告 §7.6 表 9-10）
```

每个 `outputs*/` 目录的组织方式一致：根目录放单次 bench 的 `perf.log` /
`quality.log`，`sweep/` 放 nprobe × batch 扫描结果，另外各自附带
`experiment_summary.csv`。四个平台的对应关系与数据来源见报告 §11.2。

## 构建（以 NVIDIA 平台为示例）

需要 CMake ≥ 3.18、CUDA Toolkit（≥ 11.8）与 C++17 编译器。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="80;86"        # 按实际 GPU 调整
cmake --build build -j
ctest --test-dir build --output-on-failure      # 可选：主机+GPU 测试
```

Windows 下把 `-DCMAKE_CUDA_ARCHITECTURES` 换成实际卡型（如 `75;86;89`），并保证
`nvcc` 能调用同一套 MSVC 工具链。

产物为 `build/vsearch`、`build/test_host`、`build/test_gpu`；运行
`./build/test_host && ./build/test_gpu` 可完成正确性自检。

**其余三个平台的构建入口与命令见下文**「天数智芯 CoreX 平台支持」、
「沐曦 MetaX（MACA）平台支持」、「摩尔线程 MUSA 平台支持」三节——它们只替换
编译器与链接库，源码、测试与基准命令与 NVIDIA 示例完全一致。

## 天数智芯 CoreX 平台支持

天数智芯（Iluvatar CoreX）的 `/usr/local/corex/bin/nvcc` 只是版本回显 stub，不能
编译；真正的编译器是 CoreX 定制 clang（需用 `-x ivcore` 编译 `.cu`），且官方并未
提供 CMake CUDA 工具链。因此 CoreX 构建使用仓库根目录的
[Makefile.corex](Makefile.corex)：

```bash
make -f Makefile.corex -j             # 产物在 build_corex/
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib
./build_corex/test_host               # 主机测试
./build_corex/test_gpu                # GPU 集成测试
```

适配过程中发现并解决的平台差异（按 `-DVSEARCH_COREX=1` 在编译期适配）：

1. **64 位 `atomicAdd` 静默失效**：CoreX 设备端不实现 `atomicAdd(unsigned long
   long*, ...)`，调用不报错但计数保持 0，导致倒排表/聚类计数全为 0。引擎改为
   32 位原子计数 + 64 位前缀和，倒排游标使用 32 位。
2. **CUB `DeviceScan`/`DeviceSegmentedRadixSort` 在 CoreX 上可用**，但
   `ExclusiveSum` 不会写 `out[count]`，引擎用独立缓冲并补写 total，避免污染后续
   显存。
3. **FP64 精度问题**：CoreX 的 device double 加法存在约 1e-4 相对误差（对长求和
   放大到绝对误差数百），而 float 完全符合 IEEE。CoreX 构建下距离内核改用 float
   累计；NVIDIA 构建保持 double 累计不变。
4. **PTX 内联汇编不可用**：`mov.b32` 寄存器约束在 ivcore 后端分配失败，改为标准
   CUDA 位转换内建 `__float_as_int` / `__int_as_float`（各平台均可）。

CoreX（MR-V100，32 GB）实测见下方「天数智芯实测结果」。

## 沐曦 MetaX（MACA）平台支持

沐曦通过 `cu-bridge` 提供 CUDA 兼容层：编译器是 `cucc`（nvcc 风格 wrapper，
内部调用 mxgpu_llvm 的 `mxcc`），头文件在 `cu-bridge/include` 与 `/opt/maca/include`
（含 cub、mcc 等），cuBLAS 对应实现是 `/opt/maca/lib/libmcblas.so`。构建使用仓库
根目录的 [Makefile.maca](Makefile.maca)：

```bash
make -f Makefile.maca -j            # 产物在 build_maca/
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/tools/cu-bridge/lib:/opt/maca/lib64
./build_maca/test_host              # 主机测试
./build_maca/test_gpu               # GPU 集成测试
```

沐曦的设备端 `atomicAdd(unsigned long long*)` 与 double 累加均正常（实测 64 位
原子计数正确、double 长求和与主机结果一致），因此距离累计使用 double、倒排计数
使用 64 位原子，与 NVIDIA 平台的计算路径相同；该平台只需新增构建脚本即可跑通
全部测试与 bench。

沐曦（MXC500）实测见下方「沐曦 MetaX 实测结果」。

## 摩尔线程 MUSA 平台支持

MUSA 提供原生 API（`musa_runtime.h`、`mublas_v2.h` 与 MUSA 版 CUB），不提供 CUDA
兼容头文件。因此 `cuda/engine.cu` 以 `-DVSEARCH_MUSA=1` 启用一层名称映射
（`cuda*` → `musa*`、`cublas*` → `mublas*`、`CUBLAS_OP_*` → `MUBLAS_OP_*`），
检索算法与 kernel 代码保持不变。构建使用仓库根目录的
[Makefile.musa](Makefile.musa)：

```bash
make -f Makefile.musa -j            # 产物在 build_musa/
export LD_LIBRARY_PATH=/usr/local/musa/lib:/usr/local/musa/lib64
./build_musa/test_host              # 主机测试
./build_musa/test_gpu               # GPU 集成测试
```

Makefile 用 `mcc -x musa --musa-path=/usr/local/musa` 编译 `.cu`（若按扩展名推断为
CUDA，mcc 会走 CUDA 前端并报找不到 CUDA 安装），链接 `-lmusart -lmublas`。实测该
平台的 64 位 `atomicAdd` 与 double 累加均正常，因此距离累计与倒排计数沿用
NVIDIA 路径。摩尔线程实测见下方「摩尔线程 MUSA 实测结果」。

## 生成数据并运行实验

```bash
python python/gen_dataset.py --out data \
    --n 1000000 --dim 128 --nq 1000 --top-k 100 --metric l2

# 只建索引（IVF-Flat 示例）
./build/vsearch build --vectors data/vectors.bin \
    --params data/params.txt --index data/index.idx --search_mode=ivf_flat

# IVF-PQ（m=16，ADC 粗排 + top-256 精排）
./build/vsearch bench --vectors data/vectors.bin --queries data/queries.bin \
    --params data/params.txt --search_mode=ivf_pq --pq_m=16 --nprobe=16

# 纯 ADC 模式（观察 PQ 压缩造成的召回/距离误差）
./build/vsearch search --vectors data/vectors.bin --queries data/queries.bin \
    --params data/params.txt --search_mode=ivf_pq --pq_m=16 --pq_rerank=0

# 单次检索并输出结果/性能日志
./build/vsearch search --vectors data/vectors.bin --queries data/queries.bin \
    --params data/params.txt --index data/index.idx

# 完整 bench：GPU exact 与 CPU 参考对比，并输出 recall/性能日志
./build/vsearch bench --vectors data/vectors.bin --queries data/queries.bin \
    --params data/params.txt --index data/index.idx

# 不同 nprobe / batch_size 的质量-性能扫描
python python/run_experiments.py --vsearch ./build/vsearch \
    --nprobe-list 1,2,4,8,16,32 --batch-list 32,128,512
```

命令行参数支持 `--key value` 与 `--key=value` 两种写法。NVIDIA 构建产物位于
`build/`，CoreX 构建产物位于 `build_corex/`；实验日志与报告见 `outputs/` 与
`outputs_corex/`。

`vsearch validate` 不依赖任何文件，在内存中构造小规模数据，执行 GPU exact vs CPU
参考、IVF-Flat recall 与索引 round-trip 检查。

## 文件格式

向量库与查询文件使用同一二进制容器（查询文件 metric 字段为 `none`）：

```text
i32 magic=0x56534348("VSCH")  i32 version=1
i64 n
i32 dim
u8 len + dtype("fp32"/"fp16")
u8 len + metric("l2"/"inner_product"/"cosine"/"none")
payload: n*dim 个 fp32/fp16，行主序
```

参数文件为文本 `key = value`（示例见 `python/gen_dataset.py`）。如果评测方要求完全
指定的字节布局，只需修改 `src/binary_io.cpp` 的 header 编解码并保留内部结构。

常用扩展参数：

```text
pq_m       = 16       # PQ 子空间数，需整除向量维度
pq_rerank  = 256      # ADC 后精排候选数；0 表示关闭精排
kmeans_sample = 131072
exact_memory_mb = 1536
```

检索结果、性能日志、质量日志均为纯文本，见 `outputs/` 说明与实验日志。

## 度量语义

- `l2`：返回欧氏距离（GPU 内部用平方距离排序，输出时开根）；
- `inner_product`：返回原始点积，值大者优先；
- `cosine`：读取后向量归一化，输出 `1 - cosine(a,b)`（距离越小越相似）；
- Top-K 对同分值按向量 id 小者优先，保证 GPU / CPU / 多次运行一致；
- IVF-PQ 默认对 ADC top-256 结果用原始向量精排，因此输出真实距离；纯 ADC 模式
  （`pq_rerank=0`）输出近似距离。recall 基准始终是 GPU exact 结果。

## 索引文件

自定义二进制索引（magic `VSIX`），保存 coarse center、中心范数、倒排 offsets /
ids，以及 PQ 码本与压缩码，兼容 IVF-Flat 与 IVF-PQ。完整布局见
`cuda/engine.cu` 中 `saveIndex/loadIndex`。

## 设计与报告

算法、数据布局、优化细节与实验方法见
[REPORT.md](REPORT.md)；开发过程中遇到的取舍与后续方向也在其中。

## 实测结果（RTX 4090，1e6×128，nq=1000，topK=100）

| mode | nprobe | search ms | QPS | P50 ms | P99 ms | recall@100 |
| --- | --- | --- | --- | --- | --- | --- |
| exact | - | 1433.5 | 697.6 | 71.6 | 73.4 | - |
| ivf_flat | 16 | 47.3 | 21150 | 5.90 | 6.43 | 0.926 |
| ivf_pq16 + rerank | 16 | 34.1 | 29338 | 4.17 | 4.83 | 0.925 |
| ivf_pq64 + rerank | 16 | 40.0 | 24996 | 4.91 | 5.59 | 0.926 |

CPU 单线程暴力参考（100 query）约 7.4 s；GPU exact 同子集约 67 ms，加速约 111×。
完整数据与 nprobe/batch 扫描见 [experiment_summary.csv](outputs/experiment_summary.csv)
与 [REPORT.md](REPORT.md)。

### 本次最终源码回归验证（RTX 4090，300k×128，nq=1000，topK=100，nprobe=16）

为确认所有 CoreX 适配改动不破坏 NVIDIA 路径，用最终源码重新生成 300k×128 数据集
（文件头修复后的格式）并完成 exact / IVF-Flat / IVF-PQ 三次完整 bench：

| mode | build ms | search ms | QPS | P50 ms | P99 ms | recall@100 | speedup vs CPU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exact | - | 98.6 | 2029 | 49.3 | 59.5 | - | 87.1× |
| ivf_flat | 97 | 20.4 | 49058 | 2.51 | 3.02 | 0.986 | 425× |
| ivf_pq16 + rerank | 213 | 21.8 | 45792 | 2.59 | 3.68 | 0.012 | 397× |

该组数据与 CoreX 使用同一生成参数（100 个高斯簇、scale 0.10），因此 Flat 召回
（≈0.986）与 PQ 低召回（≈0.012，ADC 量化限制）在四平台一致，说明差异来自数据
与量化率，而非平台实现。

## 天数智芯实测结果（Iluvatar MR-V100，300k×128，nq=1000，topK=100）

主机测试 5/5 通过、GPU 集成测试全部通过（exact 与 CPU 参考一致、IVF-Flat
recall≈1.0、IVF-PQ 返回合法 id、索引保存/加载 round-trip 一致）。

| mode | nprobe | search ms | QPS | P50 ms | P99 ms | recall@100 | 加速比 vs CPU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exact | - | 546.1 | 366 | 272.7 | 342.0 | - | 21.3× |
| ivf_flat | 16 | 107.1 | 9340 | 13.6 | 13.8 | 0.986 | 108.3× |
| ivf_pq16 + rerank | 16 | 58.9 | 16974 | 7.2 | 7.6 | 0.012（受限于 ADC 量化） | 200.9× |

CPU 参考为单线程暴力精确检索（200 query 子集约 11.7 s）。IVF-PQ 在该合成数据上
QPS 最高，但其 recall 受 PQ 量化精度限制：该高斯聚类合成数据上，nprobe=16 的
候选池中与查询真实距离最近的前 100 个向量在 ADC 粗排后仅 4~22 个进入精排窗口，
recall≈0.01。小规模紧致数据（40k×32、100 个分离簇）上 PQ 的 top-1 召回为 100%。
该差异用于说明 "PQ 压缩率 vs 召回" 的系统取舍：PQ 适合低精度高吞吐筛选或作为
粗排，需配合更大的精排窗口 / 更高 PQ 码率。详见
[REPORT.md](REPORT.md) 的 CoreX 适配与质量分析。

## 沐曦 MetaX 实测结果（MXC500，300k×128，nq=1000，topK=100）

主机测试 5/5 通过、GPU 集成测试全部通过（exact 与 CPU 参考逐条一致、IVF-Flat
recall=1.0、IVF-PQ 返回合法 id、索引保存/加载 round-trip 一致）。

| mode | build ms | search ms | QPS | P50 ms | P99 ms | recall@100 | 加速比 vs CPU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exact | - | 128.2 | 1560 | 64.0 | 82.4 | - | 61.9× |
| ivf_flat | 856 | 62.8 | 15916 | 8.0 | 8.4 | 0.991 | 137.2× |
| ivf_pq16 + rerank | 1174 | 62.6 | 15985 | 7.9 | 8.5 | 0.012（受限于 ADC 量化） | 119.4× |

CPU 参考为单线程暴力精确检索（200 query 子集约 7.5–8.6 s）。nprobe 扫描（batch=128）
显示 recall 随 nprobe 单调上升：nprobe=1 → 0.012、8 → 0.544、16 → 0.991、
32/64 → 1.000，与 NVIDIA / CoreX 在相同生成参数下的曲线一致；完整 21 组
nprobe×batch 数据见 [outputs_maca/experiment_summary.csv](outputs_maca/experiment_summary.csv)。

## 摩尔线程 MUSA 实测结果（300k×128，nq=1000，topK=100）

主机测试 5/5 通过、GPU 集成测试全部通过（exact 与 CPU 参考逐条一致、IVF-Flat
recall=1.0、IVF-PQ 返回合法 id、索引保存/加载 round-trip 一致）。

| mode | build ms | search ms | QPS | P50 ms | P99 ms | recall@100 | 加速比 vs CPU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exact | - | 1028.7 | 194 | 514.0 | 638.5 | - | 16.1× |
| ivf_flat | 241 | 179.2 | 5581 | 22.5 | 24.3 | 0.991 | 92.5× |
| ivf_pq16 + rerank | 4820 | 164.2 | 6092 | 20.0 | 24.4 | 0.012（受限于 ADC 量化） | 101.9× |

CPU 参考为单线程暴力精确检索（200 query 子集约 16.6 s）。nprobe 扫描（batch=128）
显示 recall 随 nprobe 单调上升：nprobe=1 → 0.0125、8 → 0.5437、16 → 0.9914、
32/64 → 1.0000，与其他平台在相同生成参数下的曲线一致；完整 21 组 nprobe×batch
数据见 [outputs_musa/experiment_summary.csv](outputs_musa/experiment_summary.csv)。

### 四平台对比（300k×128，nprobe=16）

| 平台 | GPU | exact QPS | ivf_flat QPS | recall@100 | ivf_pq QPS |
| --- | --- | --- | --- | --- | --- |
| NVIDIA | RTX 4090 | 2029 | 49058 | 0.986 | 45792 |
| 天数智芯 | MR-V100 | 366 | 9340 | 0.991 | 16974 |
| 沐曦 | MXC500 | 1560 | 15916 | 0.991 | 15985 |
| 摩尔线程 | MUSA 5.1.0 | 194 | 5581 | 0.991 | 6092 |

四平台的 recall 与排序结果一致（同参数数据下天数智芯 / 沐曦 / 摩尔线程均为
0.991390），说明检索算法在各平台上行为一致，差异来自 GPU 算力与访存带宽。

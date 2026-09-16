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

同一份源码可在 **NVIDIA CUDA** 与 **天数智芯 Iluvatar CoreX** 两个平台构建与运行：

- NVIDIA：CMake + `nvcc`（默认路径），已在 Ubuntu + RTX 4090 环境完成构建、
  主机/GPU 测试与 1e6×128 全量实验；
- CoreX：`make -f Makefile.corex`（定制 clang `-x ivcore` 编译 `.cu`），已在
  天数智芯 MR-V100（IX-ML 4.4.0）上完成构建、全部测试与端到端 bench。

两平台最近一次回归均通过：NVIDIA `test_host` + `test_gpu` 全部 PASS；CoreX 同样
全部 PASS（详见下文两节实测结果）。平台差异适配均体现在同一份源码中。代表性
结果见文末与 [outputs/REPORT.md](outputs/REPORT.md)。

验证日期：2026-09-07（NVIDIA RTX 4090 与天数智芯 MR-V100 均重新编译并跑完整测试
与 bench）。

## 目录

```text
.
├── CMakeLists.txt
├── Makefile.corex         # 天数智芯 CoreX 专用构建
├── README.md
├── include/vsearch/        # 公共数据结构/配置/IO/CPU 参考
├── src/                    # 主机侧实现与 CLI
├── cuda/engine.cu          # 全部 CUDA kernel、IVF/PQ 构建与检索
├── tests/                  # test_host / test_gpu
├── python/                 # 数据生成与实验调度
├── data/                   # 生成的二进制向量库/查询/参数
├── docs/
├── outputs/                # NVIDIA 平台报告、日志与交付物
└── outputs_corex/          # CoreX 平台日志（perf/quality/sweep 汇总）
```

## 构建

需要 CMake ≥ 3.18、CUDA Toolkit（≥ 11.8）与 C++17 编译器。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="80;86"        # 按实际 GPU 调整
cmake --build build -j
ctest --test-dir build --output-on-failure      # 可选：主机+GPU 测试
```

Windows 下把 `-DCMAKE_CUDA_ARCHITECTURES` 换成实际卡型（如 `75;86;89`），并保证
`nvcc` 能调用同一套 MSVC 工具链。

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

适配过程中发现并解决的平台差异（代码保持同一份，按
`-DVSEARCH_COREX=1` 在编译期适配）：

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
   CUDA 位转换内建 `__float_as_int` / `__int_as_float`（两平台均可）。

CoreX（MR-V100，32 GB）实测见下方「天数智芯实测结果」。

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
[outputs/REPORT.md](outputs/REPORT.md)；开发过程中遇到的取舍与后续方向也在其中。

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
（≈0.986）与 PQ 低召回（≈0.012，ADC 量化限制）在双平台一致，证明差异来自数据
与量化率而非平台实现。

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

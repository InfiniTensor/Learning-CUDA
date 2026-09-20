# GPU 向量检索引擎（CUDA）

一个不依赖 FAISS 的完整 L2 向量检索程序。包含 CPU 参考、GPU 精确检索、
IVF-Flat 建索引/持久化/加载、GPU 倒排桶搜索、分层 Top-K、批处理、recall 和性能日志。

## 功能边界

- metric：squared L2；省略 `sqrt` 不改变排序。
- 输入：FP32 或 FP16；FP16 读入后转成 FP32，距离使用 FP32 FMA 累加。
- Top-K：`1, 4, 8, 10, 16, 32, 50, 64, 100`。
- 模式：`exact`、`ivf_flat`。
- ID：文件输出为 uint64；当前 GPU 排序键支持最多 `UINT32_MAX` 个向量。
- 平分规则：距离相同时，较小原始 ID 优先；NaN/Inf 距离排到末尾。

## 构建

要求 CMake 3.24+、C++17、NVIDIA GPU 和支持目标架构的 CUDA Toolkit。
Blackwell/RTX 50 系建议 CUDA 12.8+。

```bash
cd 09_vector_retrieval/yangyeyuan
make CUDA_HOME=/usr/local/cuda CUDA_ARCH=120 -j
make test
```

也可直接使用 CMake：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## 使用

生成有聚类结构的开发数据：

```bash
./build/vector_search_cli generate \
  --vectors /tmp/vectors.bin --queries /tmp/queries.bin \
  --num-vectors 100000 --num-queries 100 --dim 128 --dtype fp32
```

GPU 精确搜索：

```bash
./build/vector_search_cli search \
  --vectors /tmp/vectors.bin --queries /tmp/queries.bin \
  --config configs/exact.toml --output /tmp/exact.bin \
  --performance /tmp/exact-performance.json --cpu-baseline
```

`--cpu-baseline` 会在同一数据上执行 CPU 参考并核对 ID；百万规模可能很慢。

构建、加载和查询 IVF-Flat：

```bash
./build/vector_search_cli build \
  --vectors /tmp/vectors.bin --index /tmp/database.ivf \
  --config configs/ivf_flat.toml --performance /tmp/build.json

./build/vector_search_cli search \
  --index /tmp/database.ivf --queries /tmp/queries.bin \
  --config configs/ivf_flat.toml --output /tmp/ivf.bin \
  --performance /tmp/ivf-performance.json \
  --ground-truth /tmp/exact.bin --quality /tmp/quality.json
```

百万规模与 `nprobe/batch_size` sweep：

```bash
./scripts/run_benchmarks.sh
```

可用环境变量缩小开发规模：

```bash
NUM_VECTORS=100000 NUM_QUERIES=100 NLIST=256 ./scripts/run_benchmarks.sh
```

## 文件格式

所有整数均为 little-endian。程序检查 magic、版本、形状、payload 长度和尾随字节。

向量库/查询文件头共 40 bytes：

```text
magic[8]       VECDB001 / VECQR001
version u32    1
dtype u32      1=fp32, 2=fp16
metric u32     1=l2；query 写 0，查询继承索引 metric
dim u32
count u64
payload u64
data           row-major [count, dim]
```

结果文件：

```text
magic[8]       VECRES01
version u32
metric u32
top_k u32
reserved u32
num_queries u64
ids u64[num_queries, top_k]
scores fp32[num_queries, top_k]
```

IVF 索引：

```text
magic[8]       IVFFLT01
version/dtype/metric/dim/nlist/reserved
count u64
payload u64
centroids fp32[nlist, dim]       row-major
offsets u64[nlist + 1]
original_ids u32[count]
vectors fp32[dim, count]         dimension-major；桶内连续
```

## 算法与数据布局

磁盘 row-major 数据在 Exact 索引加载时转置为 `[dim, N]`。一个 block 的线程在同一维
读取相邻候选，因此 global load 合并。query 缓存在 shared memory。

每个 block 计算 1024 个候选，用 CUB block radix sort 保留局部 K 个。后续 kernel
递归合并 `num_blocks × K`，不生成完整 `batch × N` 距离矩阵。排序使用
`distance_bits << 32 | id`，同时固定距离和 tie 顺序。

IVF 使用 GPU Exact Top-1 完成 Lloyd assignment；Host 只做 centroid 求和。
最终向量按桶重排并保存。查询先找最近 `nprobe` 个 centroid，再由一个 block 扫描一个
连续桶并保留局部 K，最后跨桶使用同一 Top-K kernel 合并。

关键正确性不变量：`nprobe == nlist` 时，IVF-Flat 必须逐 ID 等于 GPU Exact。

## 测试与 profiling

```bash
./tests/test_cli.sh
./scripts/profile_ncu.sh
./scripts/profile_nsys.sh
```

自测覆盖 CPU/GPU 对照、K=1/10/50/100、非整 block 的 N、batch 尾部、重复向量 tie、
FP16 文件、IVF 保存/加载和全桶等价性。

详细设计、实验口径和当前结果见 [总结报告](docs/report.zh-CN.md)。

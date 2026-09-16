# WarpFusion CLI 与 C++ API

本文给出面向使用者的最小稳定接口。二进制向量格式、完整参数和正式实验口径见
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)；性能结论见
[`FINAL_REPORT.md`](FINAL_REPORT.md)。

## CLI

### Exact 查询

```bash
./build/vector_search \
  --database data/database.bin \
  --queries data/queries.bin \
  --params configs/real_exact_k10.conf \
  --backend gpu \
  --warmup 1 \
  --repeat 5 \
  --output results/exact.txt
```

`--backend` 可取 `cpu` 或 `gpu`。GPU 路径在进程内常驻数据库和工作区，
`--warmup` 不计入正式测量，`--repeat` 的各轮共享同一个执行器。输出每行是
`query_id vector_id score`；同一 query 内按最好到最差排序，相同分数按较小
vector ID 排序。

### IVF-Flat 建库与查询

```bash
./build/ivf_build \
  --database data/database.bin \
  --output data/database.ivf \
  --nlist 1024 \
  --iterations 15 \
  --training-samples 100000 \
  --threads 16

./build/ivf_search \
  --index data/database.ivf \
  --queries data/queries.bin \
  --params configs/real_ivf_nlist1024_k10.conf \
  --nprobe 160 \
  --backend gpu \
  --warmup 1 \
  --repeat 5 \
  --output results/ivf.txt
```

建库和查询的 `nlist` 必须一致；`nprobe` 的合法范围是 `1..nlist`。CLI 的
`--nprobe` 会覆盖配置文件，便于扫描质量—吞吐曲线。`nprobe=nlist` 可用于
完整性检查，不代表推荐的近似检索配置。

所有 CLI 在参数缺失、输入格式不合法、维度/dtype/metric 不匹配或配置越界时
返回非零退出码并向标准错误输出原因。上层服务应检查退出码，不应只判断输出
文件是否存在。

## C++ API

公开头文件位于 `include/`，命名空间保持为空以兼容现有 CLI。调用者负责保证
数据库、查询和参数在执行器生命周期内语义一致；执行器不可复制但可移动。

### 常驻 Exact 执行器

```cpp
#include "file_io.h"
#include "search.h"

VectorDatabase database = readVectorDatabase("data/database.bin");
QuerySet initial = readQuerySet("data/queries.bin");
SearchParams params = readSearchParams("configs/real_exact_k10.conf");
validateInputs(database, initial, params);

GpuExactSearchEngine engine(database, initial, params);
SearchStats stats;
SearchResults neighbors = engine.search(initial, &stats);

double one_time_upload_ms = engine.databaseH2DMilliseconds();
std::size_t resident_bytes = engine.deviceBytes();
```

构造函数完成数据库上传与工作区分配；后续 `search()` 只上传 queries 并复用
常驻状态。`SearchStats` 分离 query H2D、距离/局部 Top-K、最终归并、结果 D2H
和 batch 延迟。只需一次查询时也可以调用 `gpuExactSearch()`；CPU 参考为
`cpuExactSearch()`。

### 常驻 IVF-Flat 执行器

```cpp
#include "file_io.h"
#include "ivf_flat.h"

IvfFlatIndex index = readIvfFlatIndex("data/database.ivf");
QuerySet queries = readQuerySet("data/queries.bin");
SearchParams params = readSearchParams("configs/real_ivf_nlist1024_k10.conf");

GpuIvfFlatSearchEngine engine(index, queries, params);
SearchStats stats;
SearchResults neighbors = engine.search(queries, &stats);

double one_time_upload_ms = engine.indexH2DMilliseconds();
std::size_t resident_bytes = engine.deviceBytes();
```

索引也可通过 `buildIvfFlatIndex()` 在进程内构建，并用
`writeIvfFlatIndex()`/`readIvfFlatIndex()` 持久化。`cpuIvfFlatSearch()` 是
正确性参考；`recallAtK(exact, approximate)` 计算候选 ID 集合 recall。

### 数据与结果对象

- `VectorDatabase`/`QuerySet` 支持 FP32 和 FP16 存储；计算使用 FP32 累加。
- `Metric` 支持 L2、inner product 和 cosine；L2 分数越小越好，另外两种越大
  越好。
- `SearchResults` 是按 query 分组的 `Neighbor{id, score}` 列表。
- `resultsMatch()` 用候选 ID 与分数容差比较两组 Exact 结果。
- `SearchStats` 中所有时间字段单位为毫秒；一次性数据库/索引上传不计入稳定
  查询 QPS。

## 默认选择与适用范围

Exact 默认使用 `distance_mode=warp`。正式大规模配置使用
`topk_mode=fused`；未经当前维度、K 和 GPU 验证的 query tile 会安全回退到
`exact_query_tile=1`。IVF 默认使用固定 `nprobe`；`score_mass` 是实验接口，
不是通用默认值。完整的 K/维度选择表与回退规则见
[`STABILITY_MATRIX.md`](STABILITY_MATRIX.md)。

当前 API 是单进程、单 GPU、内存内静态索引接口；不提供 ABI 稳定承诺、并发
请求调度、增量增删、压缩索引、超显存分片或多 GPU 协调。

# GPU 最大流求解器 — 贸易网络路由（选题六）

基于 CUDA 的 GPU 加速最大流求解器，采用 **Push-Relabel** 算法，支持 CSR 格式大规模图（≥10⁵ 节点、≥10⁶ 边）的多源汇查询。

## 目录结构

```
proj_maxflow/
├── include/
│   ├── types.h        # CSR 图、残量图、查询、性能统计类型
│   ├── io.h           # 图/查询读写接口
│   └── maxflow.h      # CPU 参考 + GPU 求解接口
├── src/
│   ├── main.cpp       # CLI：run / selftest 模式
│   ├── io.cpp         # CSR 二进制读写、残量图构建、结果输出
│   ├── cpu_ref.cpp    # CPU 参考实现（Edmonds-Karp）
│   └── gpu_maxflow.cu # GPU Push-Relabel 内核 + 调度
├── tools/
│   └── gen_graph.py   # 测试图生成器（random / grid / complete）
├── tests/
│   ├── run_tests.sh   # 端到端测试脚本
│   └── benchmark.sh   # 性能基准测试
├── CMakeLists.txt
├── build.sh           # NVIDIA / nvcc
├── build_iluvatar.sh  # 天数智芯 Iluvatar CoreX
├── build_maca.sh      # 沐曦 MetaX（曦云 C500 / MACA）
└── README.md
```

## 构建

依赖：CUDA Toolkit（≥11.0）、CMake（≥3.16）、C++17 编译器。

```bash
bash build.sh [SM_ARCH]
# SM_ARCH 默认为 90；同时生成 PTX 以便在更新架构的 GPU 上 JIT 运行
```

产物：`build/maxflow`

### 国产平台构建

同一份源码在两家国产加速卡上均已实测通过，各自提供了独立构建脚本：

```bash
# 天数智芯（Iluvatar CoreX）：clang++ -x ivcore
COREX_HOME=/usr/local/corex-4.4.0 bash build_iluvatar.sh

# 沐曦（MetaX / MACA）：mxcc -x maca
MACA_PATH=/opt/maca bash build_maca.sh
```

两者的详细环境、踩坑记录与性能对照见 [`report.md`](report.md) 第 9、10 节。

## 使用

### 1. 自测

```bash
./build/maxflow selftest
```

内置 4 节点测试图，验证 CPU 与 GPU 均输出最大流 5。

### 2. 运行查询

```bash
./build/maxflow run <graph.csr> <queries.txt> <result_out> [perf_log] [--cpu|--cpu-ref]
```

- `<graph.csr>`：CSR 二进制图文件
- `<queries.txt>`：查询文件，每行 `source target`
- `<result_out>`：结果输出文件，每行 `source target maxflow`
- `[perf_log]`：可选，性能日志输出路径
- `--cpu`：仅用 CPU 求解
- `--cpu-ref`：GPU 求解后用 CPU 逐查询比对，不一致则退出非零

### 3. 生成测试图

```bash
python3 tools/gen_graph.py <N> <M> <out.csr> <out.q> [--seed S] [--queries Q] [--type random|grid|complete]
```

### 4. 运行完整测试

```bash
bash tests/run_tests.sh [SM_ARCH]
```

包含：构建 → selftest → 四规模正确性比对（GPU vs CPU）→ 查询独立性验证。

### 5. 性能基准测试

```bash
bash tests/benchmark.sh [SM_ARCH]
```

输出各规模图的 T_preprocess / TTFQ / T_total / TPQ 指标。

## 输入格式

### CSR 二进制图文件

```
int32 N          节点数
int32 M          边数
int32[N+1]       row_ptr
int32[M]         col_idx
int32[M]         cap       每条边的容量（非负）
```

### 查询文件

```
source_0 target_0
source_1 target_1
...
```

## 算法设计

### GPU Push-Relabel + 全局重标号

1. **残量图常驻显存**：图拓扑（row_ptr / col_idx / rev_edge）和原始容量（cap）只上传一次，多查询复用。
2. **每查询独立初始化**：`init_residual_kernel` 从 cap 重置 residual；`init_state_kernel` 重置 excess/height。
3. **源点饱和推送**：`source_push_kernel` 将源点所有出边饱和，流量注入邻居 excess。
4. **首次 BFS 全局重标号**：从汇点 t 做 BFS，设置 `height[u] = dist(u, t)`（精确距离），建立最优高度场。
5. **迭代 push + relabel**（批量执行）：
   - `push_kernel`：沿 `height[v] = height[u]-1` 的边原子推送流量；64 位 excess 用 `atomicCAS` 实现 `atomicAdd64`。
   - `relabel_kernel`：无法推送的节点将高度提升为 `min(邻居高度)+1`；同时统计活跃节点数（仅可重标号节点）。
   - **批量执行**：连续 8 个 push+relabel 阶段无中间同步，减少 `cudaDeviceSynchronize` 开销。
   - 终止条件：relabel 活跃计数为 0（所有中间节点 excess 已推送至 t 或滞留）。
6. **周期性 BFS 全局重标号**：每 `sqrt(N)` 个阶段做一次 BFS 重标号 + gap 重标号，防止高度场退化。
7. **Gap 重标号**：检测高度断层（某高度无节点），将断层以上节点提升到 `N+1`，加速收敛。
8. **查询独立性**：每查询重置残量网络，打乱查询顺序结果一致（已验证）。

### 关键优化

| 优化 | 效果 |
|------|------|
| BFS 全局重标号 | 高度设为精确距离，阶段数从 O(N) 降至 O(log N) |
| 批量阶段执行 | 8 阶段/同步，减少 ~8x 同步开销 |
| Gap 重标号 | 检测高度断层，批量提升不可达节点 |
| Push kernel 不计 active | 避免滞留节点导致误判活跃，保证正确终止 |

### CPU 参考

Edmonds-Karp 算法（BFS 增广），作为正确性基准。

## 性能指标

| 指标 | 含义 |
|------|------|
| T_preprocess | 读图 + 建残量图 + 上传 GPU |
| TTFQ | 首次查询耗时（Time To First Query） |
| T_total | 所有查询总耗时 |
| TPQ | 平均每查询耗时（T_total / num_queries） |

## 验证结果

在 RTX 5070 Ti Laptop（sm_120，JIT 运行 sm_90 PTX）上：

### 正确性

`tests/run_tests.sh` 在六个规模上各用 **12 个查询**（题目要求「至少 10 个不同的源汇对查询」），
逐查询与 CPU 参考（Edmonds-Karp）比对，并验证查询顺序无关：

| 规模 (N, M) | 查询数 | GPU vs CPU | 查询独立性 |
|-------------|--------|------------|------------|
| 20, 100     | 12     | ✓ 全部一致  | ✓ |
| 100, 800    | 12     | ✓ 全部一致  | ✓ |
| 500, 4000   | 12     | ✓ 全部一致  | ✓ |
| 2000, 20000 | 12     | ✓ 全部一致  | ✓ |
| 10000, 100000 | 12   | ✓ 全部一致  | ✓ |
| 100000, 1000000 | 12 | ✓ 全部一致  | ✓ |

### 性能

`tests/benchmark.sh`：每档 12 个查询，构建后先预热一次（排除 PTX JIT 一次性开销），重复 3 次取中位数。

| 规模 (N, M) | 查询数 | 阶段数 | T_preprocess | TTFQ | T_total | TPQ |
|-------------|--------|--------|-------------|------|---------|-----|
| 20, 100     | 12     | 16     | 208.0 ms     | 5.94 ms | 40.25 ms | 3.35 ms |
| 100, 800    | 12     | 24     | 202.9 ms     | 5.91 ms | 35.14 ms | 2.93 ms |
| 500, 4000   | 12     | 64     | 205.7 ms     | 3.82 ms | 45.34 ms | 3.78 ms |
| 2000, 20000 | 12     | 56     | 216.4 ms     | 5.28 ms | 73.70 ms | 6.14 ms |
| 10000, 100000 | 12   | 112    | 211.6 ms     | 4.37 ms | 53.84 ms | 4.49 ms |
| **100000, 1000000** | 12 | 328 | 257.9 ms   | 17.65 ms | 144.98 ms | **12.08 ms** |

> 小规模档的 TPQ 受固定启动/同步开销支配，3 次之间波动可达 ±50%（如 2000, 20000 档
> 2.55 / 6.14 / 6.28 ms），故取中位数。

### 优化效果对比

| 规模 (N, M) | 优化前阶段数 | 优化后阶段数 | 优化前 TPQ | 优化后 TPQ | 加速比 |
|-------------|-------------|-------------|-----------|-----------|--------|
| 20, 100     | 78          | 16          | 13.19 ms  | 3.35 ms   | 3.9x   |
| 100, 800    | 1,263       | 24          | 179.36 ms | 2.93 ms   | 61x    |
| 500, 4000   | 3,886       | 64          | 666.40 ms | 3.78 ms   | 176x   |
| 2000, 20000 | 29,486      | 56          | 6,499.17 ms | 6.14 ms | 1,058x |
| 100000, 1000000 | 1,690,784 | 328      | 73,308.7 ms | 12.08 ms | 6,069x |

> 加速比按"优化后 TPQ 为 3 次中位数、优化前为单次测量"计算，宜作**量级**看待；
> 阶段数的下降是确定性的（如 29,486 → 56）。

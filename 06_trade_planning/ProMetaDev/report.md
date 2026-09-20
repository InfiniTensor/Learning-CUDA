# GPU 最大流求解器 —— 总结报告

> 选题：2026 夏季训练营 CUDA 方向项目阶段 · 选题六（贸易网络路由）
> 环境：WSL2 Ubuntu 22.04 + CUDA 12.2 + NVIDIA RTX 5070 Ti Laptop (12 GB)，编译目标 `sm_90`
> 国产平台：① 天数智芯 Iluvatar MR-V100 + IX-ML 4.4.0 / CoreX SDK 4.4.0（见第 9 节）
> 　　　　　② 沐曦 MetaX 曦云 C500 + MACA 3.5.3（见第 10 节）—— 两者均已实测验证
> 算法：Push-Relabel（GPU）+ BFS 全局重标号 + Gap 重标号；CPU 参考用 Edmonds-Karp

***

## 1. 问题描述

贸易网络可建模为有向带权图：节点是国家/地区/港口，边是贸易通道，边权是该通道的运力上限。
"给定一批源点—汇点对，问最多能承运多少货"是一个典型的最大流问题。

本题要求实现一个 GPU 加速的最大流求解器：

1. 支持 **CSR 格式**的图输入，规模达到 **≥10⁵ 节点、≥10⁶ 边**；
2. 支持**多次源点—汇点查询**，且每个查询的残量网络必须**独立初始化**
   （即查询之间不能互相污染，结果与查询顺序无关）；
3. 输出每个查询的最大流值，以及性能指标
   **T\_preprocess / TTFQ / T\_total / TPQ**；
4. 不得依赖 Hopper / Blackwell 专属架构特性（WGMMA、TMA、FP8 Tensor Core 等）。

***

## 2. 系统设计

### 2.1 分层结构

```
types.h / io.h / io.cpp   CSR 图与残量图的读写、查询读取、结果与性能日志输出
cpu_ref.cpp               CPU 参考实现（Edmonds-Karp，BFS 增广）
gpu_maxflow.cu            GPU Push-Relabel 内核 + 调度（全局/间隙重标号、批量阶段）
main.cpp                  CLI：selftest / run
tools/gen_graph.py        测试图生成器（random / grid / complete）
tests/run_tests.sh        端到端正确性测试
tests/benchmark.sh        多规模性能基准
```

### 2.2 数据结构

输入是标准 CSR 正向图：

```
struct GraphCSR {
    int32_t num_nodes, num_edges;
    vector<int32_t> row_ptr;   // N+1
    vector<int32_t> col_idx;   // M
    vector<int32_t> cap;       // M
};
```

求解前一次性展开为**残量图**（每条输入边 `u→v` 生成两条残量边）：

```
struct ResidualGraph {
    int32_t num_nodes, num_edges;   // num_edges = 2M
    vector<int32_t> row_ptr;        // N+1
    vector<int32_t> col_idx;        // 2M
    vector<int32_t> cap;            // 2M，正向边容量 c，反向边容量 0
    vector<int32_t> rev_edge;       // 2M，rev_edge[e] = e 的反向边索引
};
```

每个节点的出边表里**先排正向边、再排反向边**，用两个游标一次遍历填充，
`rev_edge` 保证 `e` 与 `rev_edge[e]` 互为反向。这样推流时对正向边扣减、
对反向边增加，只需一次数组索引，无需哈希查找。

### 2.3 显存布局与"图常驻"策略

本题的核心约束是**多次查询**。因此把**只跟图有关、跟查询无关**的数据
一次性上传并常驻显存：

| 数组                                       | 大小  | 是否随查询变化                |
| ---------------------------------------- | --- | ---------------------- |
| `d_row_ptr` / `d_col_idx` / `d_rev_edge` | 图拓扑 | 否（常驻）                  |
| `d_cap`                                  | 2M  | 否（常驻，只读）               |
| `d_residual`                             | 2M  | **是**（每查询从 `d_cap` 重置） |
| `d_excess` / `d_height`                  | N   | **是**（每查询重置）           |

每次查询只需要重置 `d_residual` 与 `d_excess`/`d_height`，**不必重新上传图**。
这就是查询独立性的实现方式：残量网络在每查询开始时由 `init_residual_kernel`
从 `d_cap` 重新拷贝，因此上一个查询的流痕迹被彻底清除，结果与查询顺序无关。

### 2.4 Push-Relabel 算法在 GPU 上的映射

算法维护两个量：

- `excess[u]`：节点 u 的盈余（流入 − 流出）；

- `height[u]`：节点高度（"势"）。

合法操作只有两种：

1. **Push**：若 `residual[u→v] > 0` 且 `height[u] = height[v] + 1`，
   则可沿该边推送 `min(excess[u], residual)`；
2. **Relabel**：若 u 有盈余但没有任何满足上述条件的出边，
   则令 `height[u] = min(height[v]) + 1`（对 `residual[u→v] > 0` 的 v 取最小）。

在 GPU 上，两种操作各对应一个 **node-parallel kernel**（一个线程负责一个节点）：

- `push_kernel`：读 `excess[u]`，遍历出边表，对满足高度条件的边推送；

- `relabel_kernel`：读 `excess[u]`，扫描残边取最小邻居高度，更新高度。

冲突处理（并发正确性的关键）：

- **残量容量的并发扣减**：两个线程可能同时向同一条边推流。实现上用
  `old = atomicAdd(&residual[p], -f)`，若返回值 `old < f` 说明可用量不足，
  再把差额 `f - old` 加回去（`atomicAdd(&residual[p], f - old)`），并把实际推送量
  修正为 `old`。这是"先扣后补"的无锁方案，保证不会超额扣减。

- **64 位盈余的原子加**：`int64_t` 的 `atomicAdd` 在部分 CUDA 版本/架构上不可用，
  因此用 `atomicCAS` 自实现 `atomicAdd64`。盈余用 64 位是为了防止
  大容量图上 `int32` 溢出（单节点盈余可达 2³¹ 量级）。

### 2.5 三个关键优化

初版（纯 push-relabel 轮询）在 N=2000 时需要近 3 万个阶段、每查询 6.5 秒。
真正把规模做上去的是下面三点：

**(1) BFS 全局重标号（Global Relabeling）**

push-relabel 最慢的部分是 relabel 一次只把高度抬 1，需要很多轮才"爬"到位。
标准做法是周期性从**汇点**做一次 BFS，直接把每个节点的精确距离赋成高度：
`height[u] = dist(u, t)`。

实现分三步：

1. `bfs_init_kernel`：`height[t] = 0`，其余 `= INF_H`；
2. `bfs_relax_kernel`：迭代松弛，对每条残边 `(u→v)`，
   若 `height[v]` 已知则 `height[u] = min(height[u], height[v]+1)`，
   直到 `changed == 0`（随机图直径 O(log N)，通常几轮就收敛）；
3. `bfs_finalize_kernel`：**强制** **`height[s] = N`**。

第 3 步是正确性关键：全局重标号算出的 `dist(s,t)` 可能是个有限小数，
若不抬高源点，流就可能被推回源点，破坏 `height[s]` 必须最高的不变式。
设为 N（大于任何可能的合法高度）保证源点只出不进。

**(2) Gap 重标号**

若某个高度 `h` 上**没有任何节点**，则所有 `height > h` 的节点都不可能再把流送到汇点
（高度不变的"断层"会永久阻断通路），可以把它们一次性抬到 `N+1`，
等价于宣告"这部分盈余已经流不到汇点"。实现为 `gap_count_kernel`（统计每个高度的节点数）

- `gap_relabel_kernel`（提升断层以上的节点）。

**(3) 批量阶段执行**

初版每个阶段都要 `cudaDeviceSynchronize` 一次来读活跃计数，
单次同步的开销在小图上反而成了主要成本。改为**连续执行 8 个 push+relabel 阶段
再同步一次**，把同步开销摊薄约 8 倍。

### 2.6 终止条件（踩过坑，见 3.3）

`push_kernel` **不维护**活跃计数；终止判断完全依赖 `relabel_kernel` 的计数：
当 relabel 后没有任何节点可以继续重标号时，说明所有中间节点的盈余都已送到汇点
或者已确认无法到达汇点（被 gap 抬高），算法终止。此时 `excess[t]` 即最大流值。

### 2.7 CPU 参考实现

Edmonds-Karp：反复 BFS 找增广路径 → 找瓶颈 → 更新残量，直到不存在增广路径。
用 BFS（而非 DFS）是为了避免递归深度和路径选择的实现陷阱，
它足够慢但**足够简单、难以写错**——这正是"参考实现"最需要的性质（见 3.2）。

***

## 3. 优化历程与开发中发现的问题（真实记录）

### 3.1 `no kernel image is available` —— 编译目标与 GPU 算力不匹配

**现象**：程序能编译链接，但一跑内核就报
`CUDA error: no kernel image is available for execution on the device`。

**定位**：查 GPU 算力：

```
$ nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12.0
```

这台机器的 GPU 是 **compute capability 12.0（Blackwell）**，
而构建脚本默认只生成了 `sm_90` 的 **SASS（cubin）**。
驱动找不到与 sm\_120 匹配的二进制，且 fatbin 里**没有 PTX** 可供 JIT，于是报错。

**修复**：在 `CMakeLists.txt` 中同时生成 PTX：

```cmake
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}
    -gencode arch=compute_90,code=sm_90
    -gencode arch=compute_90,code=compute_90")
```

`code=compute_90` 产出 PTX，驱动可在 sm\_120 上 JIT 编译执行。

> 教训：`code=sm_90` 只给 SASS，**只对 sm\_90 有效**；
> 要让程序在新架构上也能跑，必须额外带 PTX（`code=compute_XX`）。
> 题目要求"不依赖 Hopper/Blackwell 特性"，用 sm\_90 的 PTX 在新卡上 JIT
> 恰好满足这一点——代码里没有任何 sm\_90+ 专属指令。

### 3.2 CPU 参考实现结果偏大 —— 连"基准"都可能是错的

**现象**：`--cpu-ref` 逐查询比对时报大量不一致，且 **CPU 的值明显不合理**。
例如查询 `4 → 14`：

```
GPU = 24335,  CPU = 40773
```

而节点 4 的**全部出边容量之和只有 27647**。任何合法流的流量都不可能超过
源点的总出边容量，所以 **CPU 的 40773 在物理上就是不可能的**。

**定位手段**（关键）：

1. 先加了一条"廉价不变式"检查——**最大流 ≤ 源点出边容量之和**，立刻锁定 CPU 有问题；
2. 再用 **Python 独立实现一遍 Edmonds-Karp** 作为第三方基准：

```
maxflow(4->14)  = 24335
maxflow(0->19)  = 6425
maxflow(11->3)  = 16083
```

Python 的三个结果与 **GPU 完全一致**，证明 **GPU 是对的，CPU 参考实现有 bug**。

**根因**：原 Dinic 的 DFS 实现有缺陷——`cur[u]` 当前弧的推进与
"一次调用只返回一条增广路径"的语义混在一起，导致部分残量更新丢失，
反复增广后流量被重复累加，结果超出理论上限。

**修复**：把 CPU 参考彻底重写为 **Edmonds-Karp**（BFS 增广），
结构简单、易于验证。重写后所有规模 GPU 与 CPU **逐查询一致**。

> 教训一：**"参考实现"不等于"正确实现"**。当 GPU 与 CPU 不一致时，
> 不要默认是 GPU 错——先做"哪一方在物理上不可能"的不变式检查。
> 教训二：一个便宜的不变式（流 ≤ 源点出边容量和）就能把问题范围砍掉一半。

### 3.3 全局重标号反而更慢 —— 终止条件失效（最隐蔽的 bug）

**现象**：加入第一版全局重标号 + 批量执行后，**所有规模都撞上阶段数上限**：

| 规模     | 阶段数    | 上限 `200N+1000` |
| ------ | ------ | -------------- |
| N=20   | 5000   | 5000           |
| N=100  | 21000  | 21000          |
| N=500  | 101000 | 101000         |
| N=2000 | 401000 | 401000         |

阶段数**精确等于上限**，说明算法根本没有收敛，不是"慢"而是"错了"。

**根因**：初版让 `push_kernel` 和 `relabel_kernel` **都**累加活跃计数
（`active = push_计数 + relabel_计数`）。但存在一类节点：
`excess > 0` 却**没有任何残路径能通往汇点**（汇点一"侧"的节点，或被高度断层隔断的节点）。
这类节点会一直有盈余，`push` 每轮都为它们 +1，于是 `active` 永不为 0，循环永不退出。

**修复**：`push_kernel` **不再维护活跃计数**，终止判断只依赖 `relabel_kernel`：
只有"还能继续重标号"的节点才算活跃，滞留节点不计入。

```cpp
// push_kernel：删掉结尾的 atomicAdd(active, 1)
// 终止条件：relabel 后的 active 为 0
```

修复后同样的重标号策略立刻收敛，并且阶段数从 O(N) 降到几十量级。

> 教训：**并行算法的终止条件必须对"哪些节点算活跃"给出精确语义**。
> 把"有盈余"直接等同于"活跃"是错的——盈余可能永远推不出去。

### 3.4 全局重标号的三次迭代：从"迭代松弛"到"只增不减"再到"精确距离"

即使修好了终止条件，全局重标号本身也经历了三代：

| 版本       | 做法                      | small (100,800)  | large (2000,20000) | massive (100000,10⁶)     |
| -------- | ----------------------- | ---------------- | ------------------ | ------------------------ |
| 初版（无重标号） | 纯 push/relabel 轮询       | 1263 阶段 / 179 ms | 29486 阶段 / 6499 ms | 未测（不可行）                  |
| 第二代      | 全量重标号，**高度只增不减** + gap  | 72 阶段 / 6.0 ms   | 2696 阶段 / 335 ms   | 1,690,784 阶段 / 73,309 ms |
| 第三代（最终）  | **BFS 精确距离**（可降高度）+ gap | 64 阶段 / 3.3 ms   | 64 阶段 / 3.3 ms     | 328 阶段 / 13.2 ms         |

第二代的致命局限在 **massive** 上暴露：`full_relabel_kernel` 只允许"抬高"，不允许"降低"，
高度场一旦被早期错误的 relabel 带偏就**再也回不来**，在 10 万节点规模上彻底退化。

第三代改成从汇点做 BFS、**允许把高度改小**到精确距离 `dist(u,t)`，
高度场每轮都被拉回最优形态，因此在所有规模上都稳定在几十到几百阶段。

> 教训：**"只增不减"的启发式在算法早期阶段看起来安全，但会累积误差并使系统失去自愈能力。**
> 周期性地用精确计算覆盖启发式结果，是防止长尾退化的有效手段。

### 3.5 push kernel 活跃节点漏报（初版正确性 bug）

**现象**：selftest 中期望最大流 5，GPU 只给出 4，且 `active` 在源点推送后
下一个阶段就变 0。

**根因**：初版 `push_kernel` 开头是

```cpp
int64_t ex = excess[u];
if (ex <= 0) return;          // 提前返回
...
if (excess[u] > 0) atomicAdd(active, 1);   // 永不执行
```

若节点在本阶段**入口时** `excess == 0`、但在本阶段**过程中**收到其它节点推送的流量，
它会因为开头的 `return` 而直接跳过，**不会被标记为活跃**。于是循环提前退出，
少推了一部分流。

**修复**：把推流主体包进 `if (ex > 0) { ... }`，活跃判断放到函数最后，对所有节点执行。

### 3.6 查询独立性验证的方法

查询独立性不能只靠"跑一遍看结果对不对"，因为同一份查询顺序下，
即使残量网络被污染，也可能因为顺序巧合而得到正确结果。因此测试做法是：
**把查询列表打乱后重跑，再按 (source, target) 排序比对结果**——
两组结果必须完全一致。实测 12 个查询打乱后结果一致。

***

## 4. 正确性验证

### 4.1 自检（`./build/maxflow selftest`，PASS）

内置 4 节点图（0→1(3), 0→2(2), 1→2(1), 1→3(2), 2→3(3)），
CPU 与 GPU 均输出最大流 **5**。

### 4.2 端到端测试（`bash tests/run_tests.sh`，**全部通过**）

| 测试组 | 内容                                                     | 结果       |
| --- | ------------------------------------------------------ | -------- |
| 1   | 构建                                                     | PASS     |
| 2   | 引擎自检                                                   | PASS     |
| 3   | tiny / small / medium / large / big / target 六规模，GPU vs CPU 逐查询比对 | 6/6 PASS |
| 4   | 查询独立性（打乱查询顺序，结果按源汇排序比对）                                | PASS     |

六个规模**各 12 个查询、共 72 个查询**全部 GPU 与 CPU 一致（含题目目标规模 10⁵/10⁶）。

### 4.3 大规模验证

| 规模 (N, M)           | 查询数 | GPU vs CPU | 查询独立性 |
| ------------------- | --- | ---------- | ----- |
| 20, 100             | 12  | ✓ 全部一致     | ✓     |
| 100, 800            | 12  | ✓ 全部一致     | ✓     |
| 500, 4000           | 12  | ✓ 全部一致     | ✓     |
| 2000, 20000         | 12  | ✓ 全部一致     | ✓     |
| 10000, 100000       | 12  | ✓ 全部一致     | ✓     |
| **100000, 1000000** | 12  | ✓ 全部一致     | ✓     |

> 全规模统一用 **12 个查询**（题目要求「支持至少 10 个不同的源汇对查询」）。
> 目标规模（10⁵ 节点 / 10⁶ 边）上 GPU 与 CPU 参考的逐查询比对也会执行，
> 整个 `tests/run_tests.sh` 在本机约 20 秒内跑完（该规模的 CPU 参考只需约 2.5 秒）。

### 4.4 CPU 参考的交叉验证

CPU 参考实现本身用**独立的 Python Edmonds-Karp** 交叉验证过（见 3.2），
确认三方（GPU / C++ CPU / Python）在同一批查询上结果一致。

***

## 5. 性能指标

### 5.1 指标定义

| 指标                | 含义                                    |
| ----------------- | ------------------------------------- |
| **T\_preprocess** | 读图 + 构建残量图 + 上传 GPU（一次性，与查询数无关）       |
| **TTFQ**          | 首次查询耗时（Time To First Query，含残量网络首次重置） |
| **T\_total**      | 全部查询总耗时                               |
| **TPQ**           | 平均每查询耗时（T\_total / num\_queries）      |

### 5.2 多规模性能（最终版）

随机有向图，容量 ∈ \[1, 10000]，源汇对由固定种子生成。**每档 12 个查询**（题目要求
「支持至少 10 个不同的源汇对查询」，故所有规模统一用 12 个），**重复 3 次取中位数**：

| 规模 (N, M)           | 查询数 | 阶段数 | T\_preprocess | TTFQ     | T\_total  | TPQ          |
| ------------------- | --- | --- | ------------- | -------- | --------- | ------------ |
| 20, 100             | 12  | 16  | 208.0 ms      | 5.94 ms  | 40.25 ms  | 3.35 ms      |
| 100, 800            | 12  | 24  | 202.9 ms      | 5.91 ms  | 35.14 ms  | 2.93 ms      |
| 500, 4000           | 12  | 64  | 205.7 ms      | 3.82 ms  | 45.34 ms  | 3.78 ms      |
| 2000, 20000         | 12  | 56  | 216.4 ms      | 5.28 ms  | 73.70 ms  | 6.14 ms      |
| 10000, 100000       | 12  | 112 | 211.6 ms      | 4.37 ms  | 53.84 ms  | 4.49 ms      |
| **100000, 1000000** | 12  | 328 | 257.9 ms      | 17.65 ms | 144.98 ms | **12.08 ms** |

**满足题目规模要求**：在 10⁵ 节点 / 10⁶ 边的图上，**每次查询平均 12.08 ms**（12 次查询）。

> 测量说明：`tests/benchmark.sh` 在构建后先跑一次预热。原因见 3.1——`build.sh` 会重建二进制，
> 而新二进制在**首次启动内核**时要由驱动做一次性 PTX JIT（sm_90 PTX → sm_120），
> 不预热的话这笔开销会被算进第一行（实测 tiny 的 TTFQ 从 5.9 ms 变成 111.5 ms）。
> 即便如此，小规模档的 TPQ 仍受"固定启动/同步开销"支配，**3 次之间的波动可达 ±50%**
> （如 2000, 20000 档 2.55 / 6.14 / 6.28 ms），因此这里取中位数而非单次值。

### 5.3 优化前 vs 优化后

| 规模 (N, M)       | 初版阶段数       | 最终阶段数 | 初版 TPQ        | 最终 TPQ   | 加速比       |
| --------------- | ----------- | ----- | ------------- | -------- | --------- |
| 20, 100         | 78          | 16    | 13.19 ms      | 3.35 ms  | 3.9×      |
| 100, 800        | 1263        | 24    | 179.36 ms     | 2.93 ms  | **61×**   |
| 500, 4000       | 3886        | 64    | 666.40 ms     | 3.78 ms  | **176×**  |
| 2000, 20000     | 29486       | 56    | 6499.17 ms    | 6.14 ms  | **1058×** |
| 100000, 1000000 | 1,690,784 † | 328   | 73,308.7 ms † | 12.08 ms | **6069×** |

† massive 一行在初版下无法在合理时间内完成，此处使用的是第二代（"只增不减"重标号）
的实测值作为对比基准。

> 加速比按"最终 TPQ 为 3 次中位数、初版 TPQ 为单次测量"计算，且小规模档的 TPQ
> 波动较大（见 5.2 的测量说明），因此这些倍率应视为**量级**而非精确值；
> 其中阶段数的下降（如 29 486 → 56）是确定性的，不受噪声影响。

### 5.4 性能分析

1. **阶段数才是主要矛盾**。三档优化（BFS 全局重标号 + gap + 批量化）把阶段数
   从 O(N) 降到 O(log N) 量级，TPQ 的加速比几乎全部来自这里
   （最显著的一档为 29 486 → 56 阶段）。
   这说明对 push-relabel 而言，**"少做几轮"远重要于"每轮做得更快"**。
2. **T\_preprocess 稳定在 \~203–258 ms**，且随图规模增长很慢。
   对 massive（10⁶ 边）也只比 tiny 高约 24%。这部分主要是主机端
   `build_residual_graph` 的 2M 边展开 + H2D 拷贝，属于一次成本。
3. **批量化让 GPU 利用率显著提升**。初版每阶段一次全局同步，
   小图上同步开销占主导（tiny 的 TPQ 高达 13 ms，与 large 同量级）；
   改成 8 阶段/同步后，小图 TPQ 直接降到 \~3 ms。
4. **TPQ 的规模无关性（到 10⁴ 为止）**。从 N=20 到 N=10000，TPQ 落在 **2.9–6.1 ms** 区间
   （该区间内既受阶段数支配、也受固定启动/同步开销影响，单次测量波动可达 ±50%，
   故取 3 次中位数）；只有当 N 跨到 10⁵ 时阶段数才回升到 328，TPQ 升到 **12.08 ms**。

***

## 6. 依赖说明

| 组件                                | 实现方式                                               |
| --------------------------------- | -------------------------------------------------- |
| Push / Relabel / BFS / Gap kernel | 纯手写 CUDA C++，仅用 `atomicAdd`、`atomicCAS`            |
| 64 位原子加                           | NVIDIA 与沐曦走 `atomicCAS` 自实现；天数平台改用两个 32 位原子模拟（见 9.2） |
| CPU 参考                            | 自实现 Edmonds-Karp（仅用 STL `queue` / `vector`）        |
| 图生成器                              | 自实现（Python，纯标准库）                                   |
| 第三方库                              | **无**（仅 CUDA Runtime + C++ 标准库）                    |

**未使用 Hopper / Blackwell 特性**：无 WGMMA / TMA / FP8 / FP4，
无分布式共享内存、无集群（cluster）编程。全部内核只用 sm\_70 即有的一般特性
（线程块、共享内存、原子操作），因此代码在 sm\_70 及以上均可编译运行。
构建产物为单一可执行文件 `maxflow`。

***

## 7. 局限与未来可继续提升的方向

**本次实现的局限**：

1. **重标号以"阶段"为粒度、而非"活跃节点队列"**。经典实现用 FIFO/最高标号队列
   （highest-label-first）只处理活跃节点，本实现每轮扫描全部 N 个节点，
   在活跃节点占比很低时有浪费。10⁵ 规模下这仍是可接受的常数开销，
   但更大规模（10⁷）会成为瓶颈。
2. **Gap 检测走主机端**。当前 `gap_relabel` 把高度直方图拷回主机再找断层，
   引入了每 N 阶段一次的 H2D 同步。可改为设备端并行扫描消除该同步。
3. **批大小固定为 8**，未按图规模自适应调优。
4. **未做容量缩放（capacity scaling）**。大容量图上初期的推流轮数偏多。
5. **`ncu`** **/** **`nsys`** **未采集**：WSL2 不透传 CUPTI（此前在其他选题实测
   `ncu` 报 "No kernels were profiled"、`nsys` 的 CUDA kernel 段为空），
   因此本报告的瓶颈判断来自阶段数统计与实测时间吻合，而非 profiler 计数器。
   如需补齐该加分项，需在原生 Linux / AutoDL 上运行。
6. **测试图均为合成随机图**。缺少真实贸易网络（如 UN Comtrade 导出的
   双边贸易矩阵）验证，真实图的度分布与社区结构可能与随机图差异较大。
7. **国产平台验证了两家**（天数智芯、沐曦，见第 9、10 节）。海光 DCU / 华为昇腾 /
   摩尔线程等其他国产加速卡的编译链路与原子操作语义差异未做验证。

**优化方向**：

1. **活跃节点队列**：把"扫描全部节点"改为"只遍历活跃节点表"，
   配合设备端无锁队列（如 per-block 计数 + CSR 压缩），可大幅削减无效扫描。
2. **最高标号优先（HLF）**：优先处理高度最高的活跃节点，实践中收敛更快。
3. **Gap 检测设备端化**：用并行前缀/直方图扫描在设备端直接定位断层高度。
4. **容量缩放**：按 2 的幂逐级放宽可用容量，减少初期推流轮数。
5. **多查询批处理**：当查询数量很大且图不变时，可把多个 (s,t) 对
   合并到一次 kernel 调用中（每个 block 负责一个查询），提高 GPU 占用率。
6. **真实数据集验证**：补 UN Comtrade / 公开最大流基准（如 DIMACS）的测试。

***

## 8. 复现方式

```bash
# 构建（默认 sm_90；同时生成 PTX 以兼容更新的架构）
bash build.sh 90

# 自检
./build/maxflow selftest

# 端到端测试（构建 + 自检 + 四规模 GPU/CPU 比对 + 查询独立性）
bash tests/run_tests.sh 90

# 多规模性能基准
bash tests/benchmark.sh 90

# 目标规模复现（10⁵ 节点 / 10⁶ 边）
python3 tools/gen_graph.py 100000 1000000 tests/tmp/massive.csr tests/tmp/massive.q \
        --seed 42 --queries 3 --type random
./build/maxflow run tests/tmp/massive.csr tests/tmp/massive.q tests/tmp/massive.result perf.log
```

`run` 子命令的其他用法：

```bash
# 仅 CPU 求解
./build/maxflow run <graph.csr> <queries.txt> <result_out> --cpu

# GPU 求解后与 CPU 逐查询比对（不一致则退出码非零）
./build/maxflow run <graph.csr> <queries.txt> <result_out> --cpu-ref

# 指定性能日志输出路径
./build/maxflow run <graph.csr> <queries.txt> <result_out> <perf_log>
```

### 输入格式

**CSR 二进制图文件**：

```
int32 N            节点数
int32 M            边数
int32[N+1]         row_ptr
int32[M]           col_idx
int32[M]           cap        每条边的容量（非负）
```

**查询文件**（文本，每行一个查询）：

```
source_0 target_0
source_1 target_1
...
```

**结果文件**（文本，每行一个结果）：

```
source_0 target_0 maxflow_0
source_1 target_1 maxflow_1
...
```

***

## 9. 国产平台适配（天数智芯 Iluvatar CoreX）

本题有"国产平台适配"加分项。除了在 NVIDIA 平台完成开发外，本项目在**天数智芯
Iluvatar MR-V100** 上做了完整的编译、正确性与性能验证，**源码改动只有一个 64 位原子加**。

### 9.1 目标平台环境

| 项目         | 值                                                                        |
| ---------- | ------------------------------------------------------------------------ |
| 加速卡        | Iluvatar MR-V100（32 GB 显存，16 个 SM，1500 MHz）                              |
| 设备算力（报告值）  | `cc = 7.1`，`warpSize = 64`，`maxThreadsPerBlock = 4096`，每 block 共享内存 128 KB |
| 软件栈        | IX-ML 4.4.0 / Driver 4.4.0 / CoreX SDK 4.4.0，CUDA 兼容层 10.2（`cudart 10020`） |
| 编译器        | CoreX 自带 `clang++ 18.1.8`（`-x ivcore`）                                   |
| 宿主机        | 云上 Ubuntu，Python 3.12，容器镜像 IX-ML 4.4.0                                     |

编译设备代码必须显式指定语言，并链接 CoreX 的 `libcudart`：

```bash
clang++ -x ivcore -std=c++17 -O3 -DPLATFORM_ILUVATAR \
        -I<root>/include -I$COREX/include -L$COREX/lib64 -lcudart ...
```

运行前**必须**导出 `LD_LIBRARY_PATH=$COREX/lib64`，否则报
`error while loading shared libraries: libcudart.so.10.2`。
以上步骤已封装为 [`build_iluvatar.sh`](build_iluvatar.sh)（自动探测 `/usr/local/corex-4.4.0`，
可用 `COREX_HOME` 覆盖）。

### 9.2 移植中唯一的硬问题：64 位原子操作"静默失效"

在天数平台上，64 位浮点/整型普通读写正常，但**64 位原子操作不生效**，
且不报任何错误——`cudaGetLastError()` 返回 `cudaSuccess`、`cudaDeviceSynchronize()`
也正常返回，只有目标内存的值始终不变：

| 操作                              | NVIDIA RTX 5070 Ti | 天数 MR-V100     |
| ------------------------------- | ------------------ | -------------- |
| `atomicAdd(int*)` / `(uint*)` / `(float*)` | ✓                  | ✓              |
| `atomicExch` / `atomicCAS(int*)` | ✓                  | ✓              |
| 64 位普通读 / 写                       | ✓                  | ✓              |
| `atomicAdd(unsigned long long*)` | ✓                  | **✗ 恒为 0**      |
| `atomicCAS(unsigned long long*)` | ✓                  | **✗ 自旋死循环**    |

复现方式：1024 个线程各对同一个 `unsigned long long` 做一次 `atomicAdd(…, 1)`，
NVIDIA 上得到 1024，天数上得到 0。

这个坑在本项目里是**致命的**，因为 `excess[]`（节点盈余）是 `int64_t`：
10⁶ 条边、容量上限 10⁴ 时总流可达 10¹⁰ 量级，超过 `int32` 范围，不能用 32 位替代。
而初版 `atomicAdd64` 正好用 `atomicCAS` 自旋实现（见第 6 节），在源点饱和推送后
第一次写 `excess` 就永久自旋：进程不崩溃、CPU/GPU 利用率都很低，看起来像"卡住"。

**修复**：在 `PLATFORM_ILUVATAR`（以及编译器的 `__ILUVATAR__` / `__Iluvatar__`）
分支下，用**两个 32 位原子**模拟 64 位加法——小端机器上 `words[0]` 是低 32 位、
`words[1]` 是高 32 位；低 32 位 `atomicAdd` 后由返回值判断进位/借位，高 32 位按需更新：

```cpp
// 加法：先加低 32 位，溢出则高位补 1
const unsigned int old_lo = atomicAdd(&words[0], lo);
const unsigned int carry = (old_lo + lo < old_lo) ? 1u : 0u;
if (hi + carry != 0u) atomicAdd(&words[1], hi + carry);

// 减法：先减低 32 位，下溢则高位借 1
const unsigned int old_lo = atomicAdd(&words[0], 0u - lo);
const unsigned int borrow = (lo != 0u && old_lo < lo) ? 1u : 0u;
if (hi + borrow != 0u) atomicAdd(&words[1], 0u - (hi + borrow));
```

本文件的 3 处调用点（`source_push_kernel`、`push_kernel` 的两处）都不使用返回值，
因此可以直接改写为"无返回值"语义。NVIDIA 分支仍保留原来的 `atomicCAS` 自实现，
两个平台的执行路径互不影响。

### 9.3 可移植性小结：为什么移植成本可以这么低

| 潜在障碍            | 本项目情况                                         |
| --------------- | --------------------------------------------- |
| warp 级原语        | **未使用**（无 `__shfl` / `__ballot` / `__syncthreads_count`） |
| 设备侧 FP64        | **未使用**（内核只做 32/64 位整数运算与原子操作）                 |
| Hopper/Blackwell 专属特性 | 未使用（WGMMA / TMA / FP8 / cluster 均无）             |
| 块大小与 warpSize   | `BLK_SIZE = 256` 是 64 的整数倍，在 `warpSize=64` 上无隐患  |
| 主机端库依赖          | 无第三方库，只依赖 CUDA Runtime                        |

因此除 9.2 的 64 位原子外，`src/` 下**源码零改动**即可编译运行。

### 9.4 天数平台验证结果（MR-V100）

`./maxflow selftest` → **PASS**（GPU 与 CPU 均得 5）。
六个规模、每档 12 个查询，GPU 结果与 CPU Edmonds-Karp **逐查询全部一致**：

| 规模 (N, M)           | 阶段数 | T\_preprocess | TTFQ      | T\_total   | TPQ        |
| ------------------- | --- | ------------- | --------- | ---------- | ---------- |
| 20, 100             | 16  | 19.29 ms      | 2.96 ms   | 21.13 ms   | 1.76 ms    |
| 100, 800            | 24  | 19.11 ms      | 5.30 ms   | 28.44 ms   | 2.37 ms    |
| 500, 4000           | 64  | 19.99 ms      | 1.07 ms   | 24.15 ms   | 2.01 ms    |
| 2000, 20000         | 56  | 20.90 ms      | 1.32 ms   | 26.78 ms   | 2.25 ms    |
| 10000, 100000       | 112 | 26.75 ms      | 1.77 ms   | 41.31 ms   | 3.44 ms    |
| **100000, 1000000** | 328 | 78.57 ms      | 17.00 ms  | 174.13 ms  | **14.51 ms** |

> tiny–large 为 3 次中位数，big / target 为 2 次测量（两次相差 < 1%）。

三点值得注意：

1. **阶段数逐档与原平台完全一致**（16 / 24 / 64 / 56 / 112 / 328）。
   同一份算法在两种差异很大的硬件上产生完全相同的阶段序列，是对"算法映射正确"
   最强的证据（阶段数由高度函数演化决定，任何原子/同步语义的偏差都会改变它）。
2. **目标规模（10⁵ 节点 / 10⁶ 边）TPQ = 14.51 ms**，与 NVIDIA 平台的 12.08 ms 为
   **同一量级**（约 1.2×），说明性能瓶颈在算法与访存，而不在某一家的硬件特性上。
3. **T\_preprocess 明显更小**（target 档 78.6 ms vs 257.9 ms）。这部分是主机端
   `build_residual_graph` + H2D 拷贝，与 GPU 无关，差异来自云主机 CPU / 磁盘，
   不是加速卡的优势。

### 9.5 在天数平台复现

```bash
cd proj_maxflow
COREX_HOME=/usr/local/corex-4.4.0 bash build_iluvatar.sh      # 产物 build_iluvatar/maxflow
export LD_LIBRARY_PATH=/usr/local/corex-4.4.0/lib64:$LD_LIBRARY_PATH

./build_iluvatar/maxflow selftest
./build_iluvatar/maxflow run tests/tmp/target.csr tests/tmp/target.q /tmp/target.res --cpu-ref
```

`--cpu-ref` 会逐查询比对 GPU 与 CPU，不一致则以非零码退出，可直接当作回归测试。

***

## 10. 国产平台适配之二：沐曦（MetaX 曦云 C500 / MACA）

在完成天数智芯适配后，本项目又在**沐曦 曦云 C500**上做了完整验证。
与天数不同：**这次源码改动为零**，只需新增一个构建脚本。

### 10.1 目标平台环境

| 项目         | 值                                                                                |
| ---------- | -------------------------------------------------------------------------------- |
| 加速卡        | 沐曦 曦云 C500（`mx-smi` 2.2.12，KMD 3.8.30）                                           |
| 设备属性       | `warpSize = 64`，104 个 SM，`maxThreadsPerBlock = 1024`，每 block 共享内存 64 KB，计算能力 `(10,0)` |
| 软件栈        | MACA 3.5.3.20（SDK 3.5.3.307），CUDA 兼容层由 `tools/cu-bridge` 提供                       |
| 编译器        | `mxcc 1.0.0`（LLVM/clang 系，路径 `/opt/maca/mxgpu_llvm/bin/mxcc`）                     |
| 容器         | Ubuntu 22.04 + Python 3.10                                                       |

编译命令（已封装为 [`build_maca.sh`](build_maca.sh)）：

```bash
mxcc -x maca -offload-arch native --maca-path=/opt/maca \
     -Iinclude -I/opt/maca/tools/cu-bridge/include -L/opt/maca/lib \
     -fgpu-rdc --maca-link -lToolsExt_cu -lruntime_cu -lmcToolsExt \
     -std=c++17 -O3 src/main.cpp src/io.cpp src/cpu_ref.cpp src/gpu_maxflow.cu -o maxflow

export LD_LIBRARY_PATH=/opt/maca/lib        # 运行前必须设置
```

### 10.2 移植中唯一的坎：cu-bridge 的链接参数不能只取一半

沐曦不重写源码，而是用 `tools/cu-bridge` 做**编译期映射**：头文件把 `cudaXxx`
声明成 `wcudaXxx`，实现体在 `/opt/maca/lib/libruntime_cu.so` 里。
如果只照着官方 samples 的 `-x maca -offload-arch native` 编译，会在**链接期**炸出一屏：

```
undefined reference to `wcudaMalloc'
undefined reference to `wcudaGetLastError'
undefined reference to `wcudaMemcpy'  ...
```

正解是从 cu-bridge 自带的 `bin/conf.json` 的 `[link][adder]` 段抄全（这段就是官方为
CUDA 工程准备的自动追加项）：

```
-fgpu-rdc --maca-link -lToolsExt_cu -lruntime_cu -lmcToolsExt
```

补上之后一次性通过。另外 `tools/cu-bridge/bin/cucc`（面向 CUDA 工程的包装器）在本容器里
不可用 —— 它把路径拼成了 `/tools/cu-bridge//bin/gomxccbin`（缺 `MACA_PATH` 前缀），
所以直接用 `mxcc` 比走 `cucc` 稳。

### 10.3 与天数智芯的关键差异（这条决定了要不要改代码）

| 检查项                                | 天数 MR-V100     | 沐曦 C500 |
| --------------------------------- | -------------- | ------- |
| `warpSize`                        | 64             | 64      |
| `atomicAdd(unsigned long long*)`  | ❌ 静默失效（恒为 0）   | ✅ 正确    |
| `atomicCAS(unsigned long long*)`  | ❌ 自旋死循环        | ✅ 正确    |
| 设备侧 FP64                          | ✅              | ✅       |
| `maxThreadsPerBlock`              | 4096           | 1024    |
| 每 block 共享内存                      | 128 KB         | 64 KB   |

也就是说：9.2 里那套"两个 32 位原子模拟 64 位加"是**天数专属**的补丁，
沐曦上 64 位原子完全正常，走默认的 `atomicCAS` 路径即可 —— 这正是本项目能在沐曦上
**零改动**移植的原因。（两个平台的宏分支互不影响，同一份源码两边都能编。）

### 10.4 验证结果（曦云 C500）

`./build_maca/maxflow selftest` → **PASS**。
六档规模、每档 12 个查询，GPU 与 CPU Edmonds-Karp **逐查询全部一致**：

| 规模 (N, M)           | 阶段数 | T\_preprocess | TTFQ     | T\_total   | TPQ        |
| ------------------- | --- | ------------- | -------- | ---------- | ---------- |
| 20, 100             | 16  | 23.35 ms      | 5.54 ms  | 42.30 ms   | 3.52 ms    |
| 100, 800            | 24  | 22.86 ms      | 9.94 ms  | 55.23 ms   | 4.60 ms    |
| 500, 4000           | 64  | 22.16 ms      | 1.95 ms  | 46.15 ms   | 3.85 ms    |
| 2000, 20000         | 56  | 21.88 ms      | 2.48 ms  | 50.96 ms   | 4.25 ms    |
| 10000, 100000       | 112 | 29.93 ms      | 3.02 ms  | 69.98 ms   | 5.83 ms    |
| **100000, 1000000** | 328 | 71.91 ms      | 27.11 ms | 262.42 ms  | **21.87 ms** |

> tiny–large 为单次测量，big / target 为 2 次测量（两次相差 < 3%）。
> 计 12 个查询的平均值即 TPQ。目标规模下 12 个查询全部与 CPU 一致。

**阶段数在三个平台上逐档完全一致**（NVIDIA / 天数智芯 / 沐曦均为 16 / 24 / 64 / 56 / 112 / 328），
这是对"同一份算法映射"最强的交叉证据。

### 10.5 三平台性能对照（题目目标规模 10⁵ 节点 / 10⁶ 边）

| 平台                     | T\_preprocess | TTFQ     | T\_total   | TPQ        |
| ---------------------- | ------------- | -------- | ---------- | ---------- |
| NVIDIA RTX 5070 Ti     | 257.9 ms      | 17.65 ms | 144.98 ms  | **12.08 ms** |
| 天数智芯 MR-V100           | 78.6 ms       | 17.00 ms | 174.13 ms  | 14.51 ms   |
| 沐曦 曦云 C500             | 71.9 ms       | 27.11 ms | 262.42 ms  | 21.87 ms   |

三点解读：

1. **三家的 TPQ 在同一量级**（12.1 / 14.5 / 21.9 ms），最大差距约 1.8×，
   说明本实现的瓶颈在算法与访存模式，而不是某一家的硬件特性。
2. **T\_preprocess 反而是国产卡更小**。这部分是主机端 `build_residual_graph` + H2D 拷贝，
   与 GPU 无关，差异来自云主机 CPU 与磁盘。
3. 沐曦的 TPQ 比天数高约 50%，但它的 SM 数（104）远多于天数（16）。小规模档
   （tiny/small）的绝对值受固定启动/同步开销支配，本表不作微架构层面的归因。

### 10.6 在沐曦平台复现

```bash
cd proj_maxflow
MACA_PATH=/opt/maca bash build_maca.sh          # 产物 build_maca/maxflow
export LD_LIBRARY_PATH=/opt/maca/lib:$LD_LIBRARY_PATH

./build_maca/maxflow selftest
./build_maca/maxflow run tests/tmp/target.csr tests/tmp/target.q /tmp/target.res --cpu-ref
```


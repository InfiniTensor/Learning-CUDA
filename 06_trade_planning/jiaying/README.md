# 选题六：贸易网络最大流量规划

作者目录：`jiaying`。C++17 / CUDA 实现单源单汇整数最大流。图拓扑常驻 GPU，查询相互独立，输出精确的 int64 最大流值。

已经在 WSL2 Ubuntu 26.04、CUDA 12.4、RTX 4070 Laptop GPU 上编译并运行。程序、独立测试、数据生成器、CPU 基线及实测报告均在本目录。详细设计和性能分析见 [REPORT.md](REPORT.md)。

## 在 WSL 中运行

```bash
cd /mnt/d/Jiaying/Documents/ChatGPT/InfiniTensor/Learning-CUDA/06_trade_planning/jiaying
bash scripts/build_wsl.sh -DCMAKE_CUDA_ARCHITECTURES=89
bash scripts/demo_wsl.sh
```

第二条命令编译并执行正确性测试；第三条生成小规模贸易图，运行 10 个查询，并核对 CPU 结果及最小割证书。输出为 `output/demo_results.txt` 和 `output/demo_metrics.json`。

依赖：CUDA Toolkit、支持该 Toolkit 的 C++17 编译器、CMake >= 3.24、Python >= 3.9。脚本默认使用 `g++-13`，可通过 `CXX` 环境变量指定兼容编译器。脚本为 WSL 设置 `LD_LIBRARY_PATH=/usr/lib/wsl/lib`，避免加载 CUDA stub。原生 Linux 上直接使用 CMake 即可。默认编译 SM 75、86、89；本机实测使用 SM 89，其他型号未实机验证。

## 使用自己的图

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
build/trade_flow --graph graph.bin --queries queries.txt \
  --output output/results.txt --log output/metrics.json
```

输入严格遵循题目 CSR 二进制布局，按小端字节序排列：

| 字段 | 类型 | 数量 |
|---|---|---:|
| N、M | int32 | 各 1 |
| row_ptr | int32 | N+1 |
| col_indices | int32 | M |
| values | int32 | M |

容量支持 `0..2147483647`，覆盖验收要求 `1..10000`。输入边号和扩展后的残量边号使用 int32，因此要求 `2*M <= INT32_MAX`。节点至少 2 个。允许零容量边、平行边、自环、反向边和非连通图。自环不贡献 s-t 流。支持空边集。

查询文件每行 `source target`，使用 0-based 节点编号，可以含空行或 `#` 注释。源汇必须不同；相同源汇的最大流约定并不唯一，本程序拒绝这种输入。重复查询合法且每次重置状态，查询数没有固定为 10 的限制。

结果文件每行：

```text
source target maxflow
```

结果使用十进制 int64。输入长度不符、负容量、非法 CSR、越界节点、非法查询或未达到收敛上限都会报错退出，不输出近似值。输入、结果和日志路径必须不同。所有查询成功后才写结果文件。

## 运行选项

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--engine cuda` | cuda | GPU 并行最大预流 |
| `--engine cpu` | — | 单线程 Dinic，显式栈 DFS |
| `--engine cpu-pr` | — | 单线程 FIFO 推送—重标号，带全局重标号 |
| `--check` | 关闭 | CUDA 每个查询与 Dinic 对比，并检查残量、预流余额和最小割；额外验证耗时单列 |
| `--schedule thread` | thread | 每个节点一个 CUDA 线程 |
| `--schedule warp` | — | 每个节点一个 warp，用 64 位前缀和分配推送量 |
| `--batch-rounds` | 8 | 一次提交的同步轮数，范围 2–1024，必须为偶数 |
| `--global-interval` | 64 | 全局重标号轮间隔，必须为 batch 的正整数倍 |
| `--max-rounds` | 1000000 | 未收敛时显式失败的保护上限，至少等于 batch |
| `--no-graphs` | 关闭 | 逐个启动相同核函数，供 CUDA Graphs 对照 |

CPU 引擎不需要可用 GPU，但当前统一可执行文件仍需 CUDA 工具链构建。仅实现 NVIDIA CUDA；未实现 float 容量或国产平台后端。

## 复现实验

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
python3 scripts/benchmark.py
python3 scripts/plot.py
bash scripts/profile_wsl.sh
bash scripts/ubsan_wsl.sh
```

`benchmark.py` 生成三类 N=100020、M=1000000 的图，每类 10 个不同源汇，先执行完整正确性验证，再轮换顺序测试四种 GPU 配置和两种 CPU 算法，各重复 3 次。原始日志在 `reports/metrics`，汇总在 `reports/benchmark.csv` 和 `reports/summary.json`。绘图单独依赖 Matplotlib 和 NumPy；编译、测试及性能实验不需要它们。

也可只生成数据：

```bash
python3 scripts/generate.py --family layered --nodes 100020 --edges 1000000 \
  --queries 10 --seed 2026 --output data/my_graph
build/trade_flow --graph data/my_graph.bin --queries data/my_graph.queries \
  --check --output output/my_results.txt --log output/my_metrics.json
```

在支持 GPU 调试的环境运行 `bash scripts/sanitize_wsl.sh` 可执行 memcheck、racecheck、initcheck、synccheck。本机 memcheck 在 WDDM 调试接口初始化时失败，**没有取得 GPU sanitizer 通过结果**。详见报告中的原始记录与限制。

## 实测摘要

默认 thread + CUDA Graphs，三次独立运行中位数；GPU 工作缓冲区约 33.95 MiB。

| 图类型 | 10 次查询 GPU 总时间 | TPQ | 对较快 CPU 查询加速比 | 含启动与读写的加速比 |
|---|---:|---:|---:|---:|
| 分层随机容量 | 45.924 ms | 4.592 ms | 8.33× | 1.53× |
| 分层等容量 | 17.709 ms | 1.771 ms | 4.16× | 0.59× |
| 随机有向图 | 24.410 ms | 2.441 ms | 21.98× | 2.31× |

这些是合成图上的实测，不代表所有图。等容量简单图只查十次时，CUDA 初始化成本尚未摊薄，CPU 整体更快。详细计时口径、TTFQ 和优化对照见报告。

## 文件与提交

- `src/`、`include/`：正式程序，测试与实现分离。
- `tests/validate.py`：独立 Python Edmonds–Karp、边界、长链、输入验证。
- `scripts/`：WSL 构建、示例、数据生成、基准、绘图、分析与打包。
- `reports/`：可审计的实测日志、图表、环境记录。
- `REPORT.md`：项目总结。

遵循题目要求，将本目录保留在 `2026-summer-project` 分支的 `06_trade_planning/jiaying/` 下。选题六明确要求“包含测试”，因此保留独立 `tests/`，不在正式程序中混入测试入口。提交范围为本目录的源码、脚本、文档和实验记录；`build/`、`build-ubsan/`、`data/`、`output/` 不进入提交包。

```bash
python3 scripts/package.py
```

打包同时生成 `reports/SHA256.json` 文件清单，便于检查源码与报告是否匹配。

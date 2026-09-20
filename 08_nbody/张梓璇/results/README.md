# 结果目录

最终交付保留以下可复核数据：

| 目录 | 内容 |
|---|---|
| `final_graph_4096` | split-source 普通/Graph 各 5 次，含 raw CSV 与中位数 |
| `final_chunk_sweep` | N=4096/8192/16384，chunks=1..64 |
| `final_n_sweep` | N=1024..65536，四种 kernel 与 split Graph |
| `final_correctness` | 四种 kernel 与 split Graph 的误差日志 |
| `final_profiler` | 最终 split-source 普通/Graph nsys 报告与 SQLite |
| `profiling` | baseline、tiled、tiled-float4、split-source 的 ncu 报告 |

轨迹 `.bin`、构建目录和每次 benchmark 的临时输入没有放入最终 ZIP；它们都能按 `README.md` 的命令重新生成。性能表与失败结果保留在 `benchmark.csv` 和 `summary.csv`。

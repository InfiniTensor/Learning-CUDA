# Nsight Systems 剖析摘要

> 由 `tools/profile_summary.py` 从 `docs/profile/*.txt`（`nsys stats` 的原始输出）生成。

每台机器上 `nsys` 需要管理员权限才能采集 CPU 采样与上下文切换，因此这里只使用
GPU 侧数据：内核时长、启动间隔与时间线占空比。


同一批形状的硬件计数器（ncu）在 `docs/profile/ncu/`：访存/SM 占峰值
百分比、占用率、stall 分布、寄存器用量——报告 8.4 节。本工具不处理那批文件。


小规模用例（`small_iters20` / `small_iters2000`）的逐内核空隙分析见
`docs/profile/launch_bound_analysis.md`（由 `tools/gap_analysis.py` 从
nsys 导出的 sqlite 生成）——那是「这个窗口到底在等什么」的直接证据。

图内调度的节点级证据是三份手写的分析文件，本工具只把其中的内核汇总
列在下面：`loop_forkjoin_nodes.txt`（根节点个数，报告 5.8 / 8.6）、
`loop_group_width.txt`（每层一个分叉组的宽度 vs 串行链，报告 7.13）、
`loop_batch.txt`（一次提交覆盖多趟，报告 7.14）。
三者都用 `--cuda-graph-trace=node` 采集——默认粒度会把整张 CUDA Graph
当成一条不透明记录，不看节点就无从谈调度。


## `dim1024_dram`

| 内核 | 启动次数 | 中位时长 (µs) | 最小 (µs) | 最大 (µs) |
|---|---|---|---|---|
| `fhwt::fhwt_reg_kernel<__half, fhwt::Native2<__half>` | 60 | 698.7 | 659.1 | 936.9 |

- 捕获窗口 1126.9 ms（含各阶段之间的主机侧空闲），共 61 次内核启动
- 内核中位时长 698.6 µs，相邻内核中位间隔 2.62 µs = 内核时长的 0.38%：稳态下 GPU 几乎没有空隙，瓶颈在内核内部而非调度


## `dim256_dram`

| 内核 | 启动次数 | 中位时长 (µs) | 最小 (µs) | 最大 (µs) |
|---|---|---|---|---|
| `fhwt::fhwt_reg_kernel<__half, fhwt::Native2<__half>` | 60 | 689.2 | 669.3 | 1014.7 |

- 捕获窗口 1128.3 ms（含各阶段之间的主机侧空闲），共 61 次内核启动
- 内核中位时长 689.2 µs，相邻内核中位间隔 2.40 µs = 内核时长的 0.35%：稳态下 GPU 几乎没有空隙，瓶颈在内核内部而非调度


## `dim256_inplace`

| 内核 | 启动次数 | 中位时长 (µs) | 最小 (µs) | 最大 (µs) |
|---|---|---|---|---|
| `fhwt::fhwt_reg_kernel<__half, fhwt::Native2<__half>` | 60 | 78.7 | 74.4 | 333.1 |

- 捕获窗口 846.8 ms（含各阶段之间的主机侧空闲），共 61 次内核启动
- 内核中位时长 78.6 µs，相邻内核中位间隔 2.48 µs = 内核时长的 3.16%：稳态下 GPU 几乎没有空隙，瓶颈在内核内部而非调度


## `dim256_l2`

| 内核 | 启动次数 | 中位时长 (µs) | 最小 (µs) | 最大 (µs) |
|---|---|---|---|---|
| `fhwt::fhwt_reg_kernel<__half, fhwt::Native2<__half>` | 60 | 19.4 | 18.3 | 51.2 |

- 捕获窗口 780.5 ms（含各阶段之间的主机侧空闲），共 61 次内核启动
- 内核中位时长 19.4 µs，相邻内核中位间隔 2.88 µs = 内核时长的 14.88%：内核时长已接近启动/调度开销的量级，此时主机侧开销不可忽略


## `dim256_tc`

| 内核 | 启动次数 | 中位时长 (µs) | 最小 (µs) | 最大 (µs) |
|---|---|---|---|---|
| `fhwt::fhwt_tc_kernel<__half, fhwt::Native2<__half>` | 60 | 690.1 | 643.6 | 905.0 |

- 捕获窗口 1133.4 ms（含各阶段之间的主机侧空闲），共 61 次内核启动
- 内核中位时长 690.1 µs，相邻内核中位间隔 2.26 µs = 内核时长的 0.33%：稳态下 GPU 几乎没有空隙，瓶颈在内核内部而非调度


## `loop_batch`

| 内核 | 启动次数 | 中位时长 (µs) | 最小 (µs) | 最大 (µs) |
|---|---|---|---|---|
| `fhwt::fhwt_reg_kernel<__half, fhwt::Native2<__half>` | 15744 | 1.0 | 0.8 | 389.9 |

- 未解析到时间线数据。


## `loop_forkjoin`

| 内核 | 启动次数 | 中位时长 (µs) | 最小 (µs) | 最大 (µs) |
|---|---|---|---|---|
| `fhwt::fhwt_reg_kernel<__half, fhwt::Native2<__half>` | 69 | 1.1 | 1.0 | 2.0 |
| `fhwt::fhwt_reg_kernel<__half, fhwt::Native2<__half>` | 69 | 1.0 | 1.0 | 1.6 |
| `fhwt::fhwt_reg_kernel<__half, fhwt::Native2<__half>` | 46 | 1.2 | 1.1 | 2.2 |

- 未解析到时间线数据。


## `loop_forkjoin_nodes`

| 内核 | 启动次数 | 中位时长 (µs) | 最小 (µs) | 最大 (µs) |
|---|---|---|---|---|
| `fhwt::fhwt_reg_kernel<__half, fhwt::Native2<__half>` | 2016 | 1.0 | 0.9 | 2.1 |
| `fhwt::fhwt_reg_kernel<__half, fhwt::Native2<__half>` | 2016 | 1.0 | 0.8 | 3.5 |

- 未解析到时间线数据。


## `loop_group_width`

| 内核 | 启动次数 | 中位时长 (µs) | 最小 (µs) | 最大 (µs) |
|---|---|---|---|---|
| `fhwt::fhwt_reg_kernel<__half, fhwt::Native2<__half>` | 15120 | 1.4 | 0.8 | 517.1 |

- 未解析到时间线数据。


## `quant_fp8_256`

| 内核 | 启动次数 | 中位时长 (µs) | 最小 (µs) | 最大 (µs) |
|---|---|---|---|---|
| `fhwt::fhwt_reg_kernel<__half, fhwt::Native2<__half>` | 60 | 689.7 | 685.9 | 899.6 |
| `fhwt::fhwt_quant_kernel<__half, fhwt::Native2<__half>` | 60 | 505.2 | 460.8 | 722.4 |
| `fhwt::quantize_rows_kernel<__half>` | 60 | 506.4 | 503.1 | 699.0 |

- 捕获窗口 1189.0 ms（含各阶段之间的主机侧空闲），共 122 次内核启动
- 内核中位时长 686.0 µs，相邻内核中位间隔 2.37 µs = 内核时长的 0.35%：稳态下 GPU 几乎没有空隙，瓶颈在内核内部而非调度

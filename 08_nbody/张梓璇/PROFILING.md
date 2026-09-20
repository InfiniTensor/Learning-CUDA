# 最终 Profiler 与性能报告

本报告只使用真实运行数据。测试机为 RTX 3080（68 SM）、Windows 11、CUDA 12.8；Nsight Systems 2024.6.2，Nsight Compute 2025.1.0。GPU simulation 由 CUDA Event 计时，wall time 包含模拟流程中的记录点同步、D2H 和写盘。Profiler replay 数据只用于微架构分析。

## 1. 调用流程与计时边界

Leapfrog 进入循环前计算一次 `a(t0)`，每步按以下顺序执行：

```text
kick_drift -> acceleration(t + dt) -> final_kick
```

新得到的 `a(t+dt)` 直接作为下一步输入。200 步普通路径为 201 次 acceleration、200 次 kick-drift、200 次 final-kick，共 601 个 kernel；split-source 每次 force evaluation 含 partial 与 reduction，共 802 个 kernel。

模拟循环没有逐步 `cudaDeviceSynchronize`。CUDA Event 在相邻轨迹记录点之间批量计时；trajectory D2H 只在 `step % record_interval == 0` 时发生，文件写入位于 Event 范围外。模拟结束另做一次状态 D2H，供 CPU reference 和守恒量检查使用。

## 2. Nsight Systems 与 CUDA Graph

### 原始 baseline，N=4096、200 步、block=64

| 项目 | 实测 |
|---|---:|
| acceleration | 201 次，57.937 ms 合计，288.244 us 平均 |
| acceleration 占 kernel 时间 | 98.7% |
| 601 个 kernel 的 GPU timeline | 60.054 ms |
| kernel gap | 1.346 ms，2.2% |
| CPU `cudaLaunchKernel` | 3.533 ms |

force kernel 主导原始 baseline，launch gap 占比很小。Graph 对这条路径的上限有限。

### split-source，N=4096、chunks=32、200 步

最终 nsys 重新采集使用相同输入，并为 Graph 开启 node trace：

| 项目 | 普通提交 | CUDA Graph |
|---|---:|---:|
| kernel 数 | 802 | 802 |
| GPU timeline span | 16.921 ms | 15.538 ms |
| kernel gap | 1.767 ms（10.44%） | 1.040 ms（6.69%） |
| CPU 提交 | 802 次 `cudaLaunchKernel`，4.424 ms | 200 次 `cudaGraphLaunch`，1.537 ms |

5 次普通运行中位数：

| 路径 | GPU simulation | wall time | particle-steps/s |
|---|---:|---:|---:|
| split-source | 15.074 ms | 17.177 ms | 54,344,136 |
| split-source + Graph | 13.830 ms | 16.172 ms | 59,232,934 |

Graph 的 GPU Event 中位数改善 8.25%，wall 中位数改善 5.85%。它减少固定短 kernel sequence 的 host launch cost；force calculation 仍为 `O(N²)`。trajectory copy/write 留在 Graph 外，继续遵守 `record_interval`。Graph 默认关闭，因为长 kernel 主导时收益会消失，部分规模还会略慢。

## 3. Nsight Compute 证据

### N=4096：并行度不足

| 指标 | baseline bs=64 | tiled bs=64 | tiled-float4 bs=128 |
|---|---:|---:|---:|
| acceleration duration | 360.64 us | 406.27 us | 388.38 us |
| L1 hit | 97.71% | 1.15% | 2.87% |
| L2 hit | 98.30% | 98.44% | 96.88% |
| DRAM throughput | 0.03% | 0.03% | 0.03% |
| achieved occupancy | 4.17% | 4.17% | 8.33% |
| registers/thread | 40 | 28 | 25 |
| barrier stall | 0% | 0.51% | 0.35% |
| long scoreboard stall | 39.57% | 4.41% | 1.62% |
| wait/execution dependency | 32.05% | 45.88% | 42.23% |

block=64 时 baseline 只有 64 blocks，少于 68 个 SM。理论 occupancy 为 66.67%，实测仅 4.17%，说明限制来自小网格。baseline 的 warp 同址 source load 配合 97% 以上 cache hit，DRAM 压力几乎为零；shared tiling 增加的指令、barrier 和依赖链抵消了逻辑 load 减少。

split-source 把 source dimension 拆成 32 份，partial grid 从 64 扩成 2048 blocks：

| 指标 | baseline | split partial | split reduction |
|---|---:|---:|---:|
| grid blocks | 64 | 2048 | 64 |
| duration | 360.64 us | 66.14 us | 6.94 us |
| achieved occupancy | 4.17% | 55.55% | 4.12% |
| active warps/scheduler | 1.00 | 6.66 | 0.96 |
| eligible warps/scheduler | 0.20 | 2.48 | 0.07 |
| registers/thread | 40 | 32 | 40 |

chunks=32 需要 `3 * N * chunks * sizeof(float)` 的 partial buffer；N=4096 时为 1.5 MiB。更高并行度隐藏了读取与距离计算延迟，代价是 partial 写回、reduction 和额外 launch。

### N=65536：L1/TEX pipeline 压力

| 指标 | baseline bs=64 | tiled-float4 bs=64 |
|---|---:|---:|
| acceleration duration | 17.89 ms | 13.74 ms |
| L1/TEX throughput | 77.20% | 23.40% |
| DRAM throughput | 2.05% | 0.66% |
| L1 hit | 99.00% | 39.89% |
| active warps/scheduler | 5.57 | 6.03 |
| eligible warps/scheduler | 1.96 | 2.22 |
| achieved occupancy | 43.38% | 47.34% |

大 N 已提供足够 target blocks。baseline 的 L1/TEX pipeline 压力明显升高；tiled-float4 把 source reuse 放进 shared memory，并降低 L1/TEX 流量，单次 force 因此从 17.89 ms 降到 13.74 ms。DRAM 仍未接近带宽上限。

## 4. Source chunk sweep

block=64，每组 3 次取中位数；N=4096/8192/16384 分别运行 200/100/50 步。单位为 ms：

| chunks | N=4096 | N=8192 | N=16384 |
|---:|---:|---:|---:|
| 1 | 86.885 | 100.235 | 91.407 |
| 2 | 43.570 | 46.344 | 51.249 |
| 4 | 25.829 | 28.120 | 44.179 |
| 8 | 15.670 | 24.548 | **41.185** |
| 16 | 14.782 | **22.283** | 43.162 |
| 32 | **14.236** | 23.239 | 42.196 |
| 64 | 14.592 | 24.466 | 43.450 |

实测最优 chunks 随 N 增大而下降：4096 → 32，8192 → 16，16384 → 8。chunks 太少时 grid 仍小；继续增加后 partial buffer、reduction 与调度成本超过新增并行度。chunks=64 在三个规模都未获胜，失败结果保留在 CSV 中。

## 5. 最终 N sweep

block=64，每组 5 次中位数。步数随规模自动降低：200、200、200、100、50、25、12。split-source 采用 chunks 64、64、32、16、8、4、2；表中单位为 GPU simulation ms。

| N | baseline | tiled | tiled-float4 | split | split + Graph |
|---:|---:|---:|---:|---:|---:|
| 1024 | 17.562 | 20.520 | 20.793 | 6.404 | **3.548** |
| 2048 | 31.668 | 37.900 | 34.258 | 8.209 | **4.402** |
| 4096 | 60.289 | 66.773 | 66.122 | 13.687 | **11.922** |
| 8192 | 57.881 | 70.655 | 66.123 | 21.803 | **20.316** |
| 16384 | 59.629 | 70.916 | 69.433 | **41.747** | 45.683 |
| 32768 | 85.060 | 86.294 | **80.565** | 84.900 | 83.742 |
| 65536 | 157.351 | 162.138 | **138.976** | 175.721 | 176.681 |

对应 particle-steps/s：

| N | baseline | tiled | tiled-float4 | split | split + Graph |
|---:|---:|---:|---:|---:|---:|
| 1024 | 11.66 M | 9.98 M | 9.85 M | 31.98 M | **57.72 M** |
| 2048 | 12.93 M | 10.81 M | 11.96 M | 49.89 M | **93.04 M** |
| 4096 | 13.59 M | 12.27 M | 12.39 M | 59.85 M | **68.71 M** |
| 8192 | 14.15 M | 11.59 M | 12.39 M | 37.57 M | **40.32 M** |
| 16384 | 13.74 M | 11.55 M | 11.80 M | **19.62 M** | 17.93 M |
| 32768 | 9.63 M | 9.49 M | **10.17 M** | 9.65 M | 9.78 M |
| 65536 | 5.00 M | 4.85 M | **5.66 M** | 4.48 M | 4.45 M |

本机 crossover 位于 16K 到 32K 区间。小规模由 split-source 扩大网格获得明显优势；32K 时四条普通路径已经接近，tiled-float4 略快；65K 时 shared reuse 成为主要收益，split-source 的 partial/reduction 开销转为负担。Graph 在 16K 与 65K 分别慢于普通 split，说明它只处理 launch overhead。

## 6. 正确性

最终 CTest 覆盖 baseline、tiled、tiled-float4、split-source、split-source + Graph、Euler/Leapfrog 与两体轨道。Windows Release 与 WSL2 Release 均通过。

4 粒子、20 步、CPU reference 开启的最终实测：

| 路径 | max position error | max velocity error | momentum error | relative energy error |
|---|---:|---:|---:|---:|
| baseline | 0 | 0 | 0 | 8.39e-8 |
| tiled | 0 | 0 | 0 | 8.39e-8 |
| tiled-float4 | 0 | 0 | 0 | 8.39e-8 |
| split-source | 0 | 1.0e-10 | 0 | 8.39e-8 |
| split-source + Graph | 0 | 1.0e-10 | 0 | 8.39e-8 |

两体 7000 步实测最大位置误差 `2.12748e-6`、最大速度误差 `2.91149e-6`、动量误差 `6.18642e-9`、相对能量误差 `1.17527e-5`。split-source 在 chunk 内先累加再 reduction，浮点加法分组发生变化；现有误差保持在测试容限内。Graph 重放相同 kernel nodes，不改变 kick-drift-kick 数学顺序。

## 7. 结果解释与复现

没有收益的实验也保留在最终 CSV：N≤16384 的 tiled/tiled-float4 慢于 baseline；chunks=64 慢于中间 chunk；Graph 在 16384 与 65536 变慢；N=65536 的 split-source 慢于 baseline。对应因果链分别是 cache 已充分复用 → shared 指令与依赖增加；过度拆分 → partial/reduction 成本增加；长 force kernel 主导 → launch 优化占比太小；大 target grid 已饱和 → source splitting 失去作用。

可复现数据：

- `results/final_graph_4096/summary.csv`：Graph 5 次对比；
- `results/final_chunk_sweep/summary.csv`：1 到 64 chunks；
- `results/final_n_sweep/summary.csv`：7 个 N、5 条运行路径；
- `results/final_correctness/summary.csv`：最终误差；
- `results/final_profiler/`：最终 nsys report 与 SQLite 导出。

下一阶段可研究按 N/设备属性自动选 chunks、partial buffer/reduction 布局、pinned host buffer + `cudaMemcpyAsync` 轨迹流水，以及在不同 GPU 上重新标定 crossover。它们应继续作为独立实验，保留当前四条路径和 CSV 基线。

# CUDA逐项优化实验

`v0_baseline`冻结完整功能版本；`v1`到`v8`依次对应路线图八个方向。每个方向先保持字节级正确性，再在同一RTX 3050 Laptop GPU、相同输入和计时协议下比较。

晋升规则：对候选执行多轮kernel median比较；只要综合结果优于上一最佳版本、核心用例没有实测回退且正确性不变，就晋升为下一方向的基础。若没有收益，则下一方向回到上一最佳版本。没有适用瓶颈的方向保留分析记录，不强行合入代码。

## 最终晋级链

`v0` → `v1` → `v3` → `v6` → `v7` → `v8`

| 方向 | 版本目录 | 结论 | 主要结果 |
|---:|---|---|---|
| 1 | `v1_thread_block` | 晋级 | 128 threads/block，反量化综合约+0.57% |
| 2 | `v2_coalescing` | 不产生代码候选 | 地址已经连续；窄加载转交方向3 |
| 3 | `v3_vectorized_memory` | 晋级 | 反量化综合约1.53x |
| 4 | `v4_warp_shuffle` | 未晋级 | 综合约+0.02%，MXFP8有轻微回退 |
| 5 | `v5_shared_reuse` | 未晋级 | NVFP4 block明显回退 |
| 6 | `v6_bank_conflict` | 晋级 | NVFP4 block约+0.023% |
| 7 | `v7_branch_reduction` | 晋级 | 量化综合约3.55x |
| 8 | `v8_ilp_unroll` | 晋级 | 在v7上量化综合再提升约3.06% |

最终候选可执行文件为`build/cuda/Release/luxorion_v8.exe`；每个目录都保留独立源码和测试结论。

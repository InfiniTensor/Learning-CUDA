# NVFP4 Nsight Compute 分析

来源：2026-09-14 的 Nsight Compute 采集。原先记录在根目录的一份阶段性分析文件里，
提交前清理时把结论移到这里，数据与报告为同一批。

采集条件：4M FP32 输入，RTX 3060 Laptop，Compute Capability 8.6。
原始报告：`nvfp4-test.ncu-rep`、`nvfp4-maximum.ncu-rep`、`nvfp4-maximum-1024.ncu-rep`。

| kernel | DRAM throughput | SM throughput | achieved occupancy | duration |
|---|---:|---:|---:|---:|
| NVFP4 fused quantize | 56.08% | 84.46% | 88.33% | 106.59 us |
| NVFP4 `maximum` | 88.01% | 32.50% | 88.55% | 61.31 us |

NVFP4 的 fused quantize 是计算受限（SM 84.46%，DRAM 56.08%），而 `maximum` 是访存受限
（DRAM 88.01%，SM 32.50%）。因此有两个明确方向：

1. `nvfp4_quantize_fused_kernel`：减少编码、除法、分支和谓词开销；测试每线程处理更多
   packed byte，但要观察寄存器和 active threads/warp。
2. `maximum`：测试每线程多加载几个 FP32、调整 partial block 数和线程块大小；
   目标是减少约 61 us 的全局最大值归约时间。

注意：`ncu --set full` 会为采集指标重复执行 kernel，报告中的单次耗时是 profiler 下的值，
不能与普通 benchmark 直接比较。正式耗时使用 CUDA event benchmark 和 Nsight Systems。

## 已落地情况

方向 2（`maximum` 的 partial 数量）已在[第 6 次优化](../../OPTIMIZATION_LOG.md)中试验：
partial 上限从 4096 降为 1024，1M 有 1.16x 改善，4M/16M 基本不变。

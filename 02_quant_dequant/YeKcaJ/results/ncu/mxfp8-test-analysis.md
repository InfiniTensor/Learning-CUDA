# MXFP8 Nsight Compute 分析

报告：`mxfp8-test.ncu-rep`，4M FP32 输入，`mxfp8_quantize_vectorized_kernel`，Compute Capability 8.6。

| 指标 | 数值 | 判断 |
|---|---:|---|
| Kernel duration | 71.14 us | profiler 下的单次时间，不能与普通 benchmark 直接比较 |
| DRAM throughput | 90.28% | 已接近显存吞吐上限 |
| Memory throughput | 302.97 GB/s | 说明访存是主要限制之一 |
| SM throughput | 81.95% | ALU/整数逻辑流水线也较忙 |
| Achieved occupancy | 88.92% | occupancy 不是主要瓶颈 |
| Registers/thread | 25 | 没有明显寄存器限制 |
| Local memory spilling | 0 | 没有寄存器溢出 |
| Active threads/warp | 26.45 | 尾部判断和线程谓词造成部分空闲 |

Nsight Compute 估计全局写入访问模式仍有约 4.5% 的潜在收益：部分 L2 sector 没有被完全利用。下一步若继续优化 MXFP8，应优先检查 packed data 的 32-bit 写入布局和尾部谓词；不应先增加每线程工作量，因为当前 DRAM 已达到约 90%，并且 SM 已达到约 82%。

注意：`ncu --set full` 会为采集指标重复执行 kernel，终端中的 343 ms 量化耗时是 profiler 开销，不是正式性能数据。正式耗时仍使用 CUDA event benchmark 和 Nsight Systems。

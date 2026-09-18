# Hopper 缺口补测

GPU：NVIDIA H200。30 次 NCU 成功采集；量化为三个独立进程，每进程九轮交错。

## NCU：同卡、同输入、单 kernel

FP16，16384 tokens，normalize=true；随机输入避免特殊值内存压缩偏置。
四次 launch 取第四次；cache-control=all/none；clock-control=none，未锁时钟。
下表为 all 策略，时间 μs、shared 字节；stall 是每 active issue 的比值，不是百分比。

| d | 路径 | μs | dynamic shared | registers | short scoreboard | MIO throttle |
|---|---|---:|---:|---:|---:|---:|
| 128 | ptx_fast | 4.992 | 0.0 | 32.0 | 0.97643 | 2.111512 |
| 256 | ptx_fast | 6.880 | 0.0 | 32.0 | 1.05554 | 2.289621 |
| 64 | ptx_fast | 3.200 | 0.0 | 32.0 | 1.211077 | 0.518172 |
| 128 | ptx_split | 4.832 | 0.0 | 32.0 | 0.715732 | 1.455231 |
| 256 | ptx_split | 7.168 | 0.0 | 32.0 | 0.615137 | 1.210565 |
| 64 | ptx_split | 3.264 | 0.0 | 32.0 | 0.940698 | 0.398771 |
| 128 | warp | 4.928 | 0.0 | 21.0 | 1.87413 | 1.181387 |
| 256 | warp | 6.432 | 0.0 | 24.0 | 1.25863 | 0.929677 |
| 64 | warp | 3.840 | 0.0 | 16.0 | 2.857348 | 0.962307 |
| 128 | wmma_fast | 6.304 | 9728.0 | 32.0 | 6.163585 | 9.674514 |
| 256 | wmma_fast | 10.144 | 9728.0 | 32.0 | 8.024887 | 14.375102 |
| 64 | wmma_fast | 4.320 | 9728.0 | 32.0 | 5.825285 | 3.692472 |
| 128 | wmma_split | 6.880 | 12800.0 | 32.0 | 5.675343 | 9.630259 |
| 256 | wmma_split | 11.264 | 12800.0 | 32.0 | 7.119047 | 13.855894 |
| 64 | wmma_split | 4.576 | 12800.0 | 32.0 | 5.306178 | 4.082422 |

![Hopper NCU Roofline](roofline.png)

Roofline 工作量为有用 FHT 加减次数 N·d·log2(d)，不计 dense MMA 冗余操作。
横轴使用实际 DRAM 流量（缺少 bytes 时使用同次 NCU rate×duration，CSV 标明来源）。
FP32 参考 roof 不是 Tensor Core 峰值；all/none 不等于真实应用的冷/热缓存。
未刷新缓存时可命中 L2，DRAM 点可越过 DRAM 斜线；不代表超越硬件带宽。

## 同卡大规模性能

131072 tokens，normalize=true，五轮 Graph 摊销设备时间中位数，μs。
不与单次 NCU 时间混算；TC 仍须单独看严格正确性，快不等于可用于提交。

| dtype | d | warp | WMMA fast | PTX fast | WMMA split | PTX split |
|---|---:|---:|---:|---:|---:|---:|
| float16 | 64 | 13.671 | 17.485 | 9.841 | 19.484 | 10.746 |
| float16 | 128 | 22.316 | 32.906 | 20.187 | 36.691 | 20.254 |
| float16 | 256 | 35.959 | 62.748 | 36.690 | 70.454 | 37.670 |
| bfloat16 | 64 | 13.754 | 18.376 | 9.847 | 21.168 | 11.077 |
| bfloat16 | 128 | 22.428 | 34.934 | 20.196 | 40.335 | 20.586 |
| bfloat16 | 256 | 37.287 | 66.834 | 36.596 | 77.901 | 38.028 |

![Hopper 同卡性能](ptx_comparison.png)

## INT4 宽写回独立复测

对照固定 f4_q0；f4_q1 隔离写回宽度，f8_q1 同时改变融合 block 大小。
表内为三个进程各自九轮中位数的 speedup，不挑最快轮；全部输出检查通过。

| dtype | tokens | d | 候选 | 进程 1 / 2 / 3 加速 | 三次均 >1 |
|---|---:|---:|---|---|---|
| float16 | 16384 | 512 | f4_q1/fused_warp | 1.016× / 1.006× / 1.015× | True |
| float16 | 16384 | 512 | f8_q1/fused_warp | 1.003× / 0.994× / 0.996× | False |
| float16 | 16384 | 1024 | f4_q1/fused_warp | 1.055× / 1.057× / 1.055× | True |
| float16 | 16384 | 1024 | f8_q1/fused_warp | 1.058× / 1.056× / 1.058× | True |
| float16 | 131072 | 512 | f4_q1/fused_warp | 0.974× / 0.974× / 0.975× | False |
| float16 | 131072 | 512 | f8_q1/fused_warp | 0.976× / 0.976× / 0.975× | False |
| float16 | 131072 | 1024 | f4_q1/fused_warp | 1.060× / 1.062× / 1.063× | True |
| float16 | 131072 | 1024 | f8_q1/fused_warp | 1.060× / 1.062× / 1.060× | True |
| bfloat16 | 16384 | 512 | f4_q1/fused_warp | 0.979× / 0.977× / 0.982× | False |
| bfloat16 | 16384 | 512 | f8_q1/fused_warp | 0.958× / 0.953× / 0.961× | False |
| bfloat16 | 16384 | 1024 | f4_q1/fused_warp | 1.066× / 1.060× / 1.063× | True |
| bfloat16 | 16384 | 1024 | f8_q1/fused_warp | 1.062× / 1.057× / 1.059× | True |
| bfloat16 | 131072 | 512 | f4_q1/fused_warp | 0.973× / 0.972× / 0.972× | False |
| bfloat16 | 131072 | 512 | f8_q1/fused_warp | 0.975× / 0.975× / 0.974× | False |
| bfloat16 | 131072 | 1024 | f4_q1/fused_warp | 1.060× / 1.060× / 1.061× | True |
| bfloat16 | 131072 | 1024 | f8_q1/fused_warp | 1.062× / 1.061× / 1.062× | True |

![独立进程复测](quant_repeat.png)

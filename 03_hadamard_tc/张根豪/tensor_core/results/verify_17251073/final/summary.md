# 严格验收与 A/B 调优结果

时间为同一作业、相同输入的三次独立扫描中位数；每次 20 warmup / 100 CUDA Events。
计时 GPU：NVIDIA H200
计时来源：`tensor_core/results/verify_17251073`；严格验证来源：`tensor_core/results/verify_17251073`。
验证与计数器可来自独立作业，其编译器和 GPU 见各自 environment.txt；不混算性能。

| dtype | d | 路径 | 原配置 μs | 最快候选 | 中位数 μs | 加速 |
|---|---:|---|---:|---|---:|---:|
| fp16 | 64 | warp | 24.850 | w0_tc4_v2 | 16.110 | 1.543x |
| fp16 | 64 | tc_fast | 26.053 | w0_tc4_v2 | 20.378 | 1.278x |
| fp16 | 64 | tc_split | 31.227 | w0_tc4_v2 | 22.369 | 1.396x |
| fp16 | 64 | fused_warp_int4 | 25.212 | w8_tc8_v0 | 22.411 | 1.125x |
| fp16 | 64 | fused_tc_fast_int4 | 26.297 | w0_tc4_v2 | 20.713 | 1.270x |
| fp16 | 64 | fused_tc_split_int4 | 31.394 | w0_tc4_v2 | 22.669 | 1.385x |
| fp16 | 128 | warp | 25.684 | w8_tc8_v0 | 25.188 | 1.020x |
| fp16 | 128 | tc_fast | 46.907 | w4_tc4_v1 | 36.147 | 1.298x |
| fp16 | 128 | tc_split | 57.128 | w4_tc4_v1 | 39.898 | 1.432x |
| fp16 | 128 | fused_warp_int4 | 30.919 | w4_tc4_v0 | 30.919 | 1.000x |
| fp16 | 128 | fused_tc_fast_int4 | 47.000 | w4_tc4_v1 | 36.261 | 1.296x |
| fp16 | 128 | fused_tc_split_int4 | 57.160 | w4_tc4_v1 | 40.377 | 1.416x |
| fp16 | 256 | warp | 39.715 | w4_tc4_v1 | 39.638 | 1.002x |
| fp16 | 256 | tc_fast | 87.910 | w4_tc4_v1 | 65.985 | 1.332x |
| fp16 | 256 | tc_split | 108.390 | w4_tc4_v1 | 73.656 | 1.472x |
| fp16 | 256 | fused_warp_int4 | 51.089 | w4_tc4_v0 | 51.089 | 1.000x |
| fp16 | 256 | fused_tc_fast_int4 | 88.711 | w0_tc4_v2 | 69.890 | 1.269x |
| fp16 | 256 | fused_tc_split_int4 | 108.905 | w0_tc4_v2 | 76.298 | 1.427x |
| bf16 | 64 | warp | 24.872 | w0_tc4_v1 | 16.119 | 1.543x |
| bf16 | 64 | tc_fast | 27.180 | w0_tc4_v1 | 21.339 | 1.274x |
| bf16 | 64 | tc_split | 32.621 | w0_tc4_v1 | 24.170 | 1.350x |
| bf16 | 64 | fused_warp_int4 | 25.152 | w8_tc8_v0 | 22.768 | 1.105x |
| bf16 | 64 | fused_tc_fast_int4 | 27.449 | w0_tc4_v1 | 21.791 | 1.260x |
| bf16 | 64 | fused_tc_split_int4 | 33.096 | w4_tc4_v1 | 24.555 | 1.348x |
| bf16 | 128 | warp | 25.646 | w8_tc8_v0 | 25.485 | 1.006x |
| bf16 | 128 | tc_fast | 49.113 | w4_tc4_v1 | 38.308 | 1.282x |
| bf16 | 128 | tc_split | 59.928 | w0_tc4_v2 | 43.486 | 1.378x |
| bf16 | 128 | fused_warp_int4 | 31.485 | w4_tc4_v1 | 31.450 | 1.001x |
| bf16 | 128 | fused_tc_fast_int4 | 49.279 | w4_tc4_v1 | 38.404 | 1.283x |
| bf16 | 128 | fused_tc_split_int4 | 60.544 | w4_tc4_v1 | 43.868 | 1.380x |
| bf16 | 256 | warp | 41.008 | w4_tc4_v0 | 41.008 | 1.000x |
| bf16 | 256 | tc_fast | 92.332 | w0_tc4_v2 | 70.639 | 1.307x |
| bf16 | 256 | tc_split | 114.086 | w0_tc4_v1 | 81.113 | 1.407x |
| bf16 | 256 | fused_warp_int4 | 52.188 | w8_tc8_v0 | 52.129 | 1.001x |
| bf16 | 256 | fused_tc_fast_int4 | 93.272 | w0_tc4_v1 | 72.045 | 1.295x |
| bf16 | 256 | fused_tc_split_int4 | 115.741 | w0_tc4_v1 | 82.830 | 1.397x |

最快候选是已测参数中的事后比较，不等于可推广的自动分发策略。

## 严格绝对误差与融合校验

| 变体 | 参考 | 路径 | 通过/总数 | 最大绝对误差 |
|---|---|---|---:|---:|
| w0_tc4_v1 | cpu | baseline | 132/132 | 0 |
| w0_tc4_v1 | cpu | fused_opt_int4 | 132/132 | n/a |
| w0_tc4_v1 | cpu | fused_tc_fast_int4 | 108/108 | n/a |
| w0_tc4_v1 | cpu | fused_tc_split_int4 | 108/108 | n/a |
| w0_tc4_v1 | cpu | fused_warp_int4 | 132/132 | n/a |
| w0_tc4_v1 | cpu | optimized | 132/132 | 0 |
| w0_tc4_v1 | cpu | tc_fast | 96/132 | 1 |
| w0_tc4_v1 | cpu | tc_split | 121/132 | 0.5 |
| w0_tc4_v1 | cpu | unfused_int4 | 132/132 | n/a |
| w0_tc4_v1 | cpu | warp | 132/132 | 0 |
| w0_tc4_v2 | cpu | baseline | 132/132 | 0 |
| w0_tc4_v2 | cpu | fused_opt_int4 | 132/132 | n/a |
| w0_tc4_v2 | cpu | fused_tc_fast_int4 | 108/108 | n/a |
| w0_tc4_v2 | cpu | fused_tc_split_int4 | 108/108 | n/a |
| w0_tc4_v2 | cpu | fused_warp_int4 | 132/132 | n/a |
| w0_tc4_v2 | cpu | optimized | 132/132 | 0 |
| w0_tc4_v2 | cpu | tc_fast | 96/132 | 1 |
| w0_tc4_v2 | cpu | tc_split | 121/132 | 0.5 |
| w0_tc4_v2 | cpu | unfused_int4 | 132/132 | n/a |
| w0_tc4_v2 | cpu | warp | 132/132 | 0 |
| w4_tc4_v0 | cpu | baseline | 132/132 | 0 |
| w4_tc4_v0 | cpu | fused_opt_int4 | 132/132 | n/a |
| w4_tc4_v0 | cpu | fused_tc_fast_int4 | 108/108 | n/a |
| w4_tc4_v0 | cpu | fused_tc_split_int4 | 108/108 | n/a |
| w4_tc4_v0 | cpu | fused_warp_int4 | 132/132 | n/a |
| w4_tc4_v0 | cpu | optimized | 132/132 | 0 |
| w4_tc4_v0 | cpu | tc_fast | 96/132 | 1 |
| w4_tc4_v0 | cpu | tc_split | 121/132 | 0.5 |
| w4_tc4_v0 | cpu | unfused_int4 | 132/132 | n/a |
| w4_tc4_v0 | cpu | warp | 132/132 | 0 |
| w4_tc4_v1 | cpu | baseline | 132/132 | 0 |
| w4_tc4_v1 | cpu | fused_opt_int4 | 132/132 | n/a |
| w4_tc4_v1 | cpu | fused_tc_fast_int4 | 108/108 | n/a |
| w4_tc4_v1 | cpu | fused_tc_split_int4 | 108/108 | n/a |
| w4_tc4_v1 | cpu | fused_warp_int4 | 132/132 | n/a |
| w4_tc4_v1 | cpu | optimized | 132/132 | 0 |
| w4_tc4_v1 | cpu | tc_fast | 96/132 | 1 |
| w4_tc4_v1 | cpu | tc_split | 121/132 | 0.5 |
| w4_tc4_v1 | cpu | unfused_int4 | 132/132 | n/a |
| w4_tc4_v1 | cpu | warp | 132/132 | 0 |
| w8_tc8_v0 | cpu | baseline | 132/132 | 0 |
| w8_tc8_v0 | cpu | fused_opt_int4 | 132/132 | n/a |
| w8_tc8_v0 | cpu | fused_tc_fast_int4 | 108/108 | n/a |
| w8_tc8_v0 | cpu | fused_tc_split_int4 | 108/108 | n/a |
| w8_tc8_v0 | cpu | fused_warp_int4 | 132/132 | n/a |
| w8_tc8_v0 | cpu | optimized | 132/132 | 0 |
| w8_tc8_v0 | cpu | tc_fast | 96/132 | 1 |
| w8_tc8_v0 | cpu | tc_split | 121/132 | 0.5 |
| w8_tc8_v0 | cpu | unfused_int4 | 132/132 | n/a |
| w8_tc8_v0 | cpu | warp | 132/132 | 0 |

同一用例/后端在不同调优参数下输出 SHA256 不一致的组数：0。
独立检查器与 C++ 判据不一致的行数：0。

## NCU 缓存策略与 Roofline

NCU GPU：NVIDIA H100 80GB HBM3；共 40 次单 kernel 采集。
独立计数器目录：`tensor_core/results/ncu_17251063`。其 GPU/环境见该目录的 environment.txt；不与计时作业跨 GPU 混算。
`all` 表示 replay 前刷新，`none` 表示不由 profiler 刷新，不直接等同于应用 cold/warm。
横轴使用实际 DRAM 字节（或明确标注的 NCU rate×duration），不是逻辑张量字节。
纵轴统一使用有用 FHT 加减次数；TC 的冗余 MMA FLOPs 不算成算法收益。
FP32 roof 仅为非 TC 参考线，不是 Tensor Core 峰值或利用率。
![缓存策略 Roofline](roofline_cache.png)

单次 profiler replay，不能与上面的 131072 tokens 无 profiler 中位数混算。
以下固定 FP16、normalize=true、16384 tokens、cache-control=all；stall 是
`per_issue_active.ratio`（每次发射对应的 stalled warp 周期），不是百分比。

| 配置 | 路径 | d | NCU μs | short scoreboard | MIO throttle | tensor active % |
|---|---|---:|---:|---:|---:|---:|
| w0_tc4_v1 | tc_fast | 128 | 6.784 | 5.994 | 9.561 | 3.053 |
| w0_tc4_v1 | tc_split | 128 | 7.360 | 5.568 | 9.724 | 4.267 |
| w0_tc4_v1 | warp | 128 | 5.408 | 1.605 | 0.910 | 0.000 |
| w0_tc4_v1 | warp | 64 | 4.512 | 2.678 | 1.576 | 0.000 |
| w4_tc4_v0 | tc_fast | 128 | 8.096 | 12.922 | 10.305 | 2.574 |
| w4_tc4_v0 | tc_split | 128 | 9.600 | 10.957 | 12.478 | 3.271 |
| w4_tc4_v0 | warp | 128 | 5.376 | 1.599 | 0.953 | 0.000 |
| w4_tc4_v0 | warp | 64 | 5.248 | 2.485 | 0.580 | 0.000 |

![重复扫描性能对比](tuning.png)

# 性能测试结果

GPU 驻留计时使用 CUDA event；host_api 包括量化的分配、传输和释放。
内部对照保留 shared scale 归约与枚举元素编码；两种格式的默认 block+nearest 均使用 scale/编码融合 kernel，NVFP4 另含两级全局归约。内部对照的 scale 也可能共享优化，优化前后请比较留存的 quant_optimized 记录。
中位数采用偶数样本中间两值均值，P95 使用 nearest-rank。CPU 仅测试 1M，预热一次后计时三次。

| 格式 | 操作 | 范围 | 元素数 | median ms | P95 ms | 逻辑 GB/s |
|---|---|---|---:|---:|---:|---:|
| mxfp8 | quant_enumeration | resident_gpu | 1048576 | 0.570368 | 0.910336 | 9.250 |
| mxfp8 | quant_optimized | resident_gpu | 1048576 | 0.021504 | 0.033792 | 245.333 |
| mxfp8 | dequant_fp32 | resident_gpu | 1048576 | 0.031744 | 0.036640 | 166.194 |
| mxfp8 | dequant_fp16 | resident_gpu | 1048576 | 0.024576 | 0.029472 | 129.333 |
| mxfp8 | dequant_bf16 | resident_gpu | 1048576 | 0.027392 | 0.027648 | 116.037 |
| mxfp8 | quant_optimized | host_api | 1048576 | 2.220704 | 2.521108 | 2.376 |
| mxfp8 | quant_reference | cpu | 1048576 | 1282.667695 | 1333.807234 | 0.004 |
| mxfp8 | dequant_fp32 | cpu | 1048576 | 13.930944 | 17.388098 | 0.379 |
| mxfp8 | quant_enumeration | resident_gpu | 4194304 | 2.235904 | 2.246656 | 9.438 |
| mxfp8 | quant_optimized | resident_gpu | 4194304 | 0.071648 | 0.082944 | 294.531 |
| mxfp8 | dequant_fp32 | resident_gpu | 4194304 | 0.111616 | 0.117760 | 189.064 |
| mxfp8 | dequant_fp16 | resident_gpu | 4194304 | 0.104448 | 0.114688 | 121.725 |
| mxfp8 | dequant_bf16 | resident_gpu | 4194304 | 0.109568 | 0.110592 | 116.037 |
| mxfp8 | quant_optimized | host_api | 4194304 | 5.204752 | 5.704295 | 4.054 |
| mxfp8 | quant_enumeration | resident_gpu | 16777216 | 7.856096 | 9.134080 | 10.745 |
| mxfp8 | quant_optimized | resident_gpu | 16777216 | 0.270672 | 0.307200 | 311.855 |
| mxfp8 | dequant_fp32 | resident_gpu | 16777216 | 0.401760 | 0.404480 | 210.101 |
| mxfp8 | dequant_fp16 | resident_gpu | 16777216 | 0.362496 | 0.365568 | 140.294 |
| mxfp8 | dequant_bf16 | resident_gpu | 16777216 | 0.377856 | 0.381952 | 134.591 |
| mxfp8 | quant_optimized | host_api | 16777216 | 18.417972 | 19.439252 | 4.583 |
| nvfp4 | quant_enumeration | resident_gpu | 1048576 | 0.196608 | 0.198656 | 24.333 |
| nvfp4 | quant_optimized | resident_gpu | 1048576 | 0.050176 | 0.052096 | 95.347 |
| nvfp4 | dequant_fp32 | resident_gpu | 1048576 | 0.028672 | 0.029696 | 166.857 |
| nvfp4 | dequant_fp16 | resident_gpu | 1048576 | 0.020320 | 0.023552 | 132.233 |
| nvfp4 | dequant_bf16 | resident_gpu | 1048576 | 0.026032 | 0.026624 | 103.218 |
| nvfp4 | quant_optimized | host_api | 1048576 | 1.802313 | 1.896407 | 2.654 |
| nvfp4 | quant_reference | cpu | 1048576 | 87.956852 | 88.718376 | 0.054 |
| nvfp4 | dequant_fp32 | cpu | 1048576 | 11.534422 | 11.841971 | 0.415 |
| nvfp4 | quant_enumeration | resident_gpu | 4194304 | 0.721728 | 0.726016 | 26.515 |
| nvfp4 | quant_optimized | resident_gpu | 4194304 | 0.141824 | 0.144384 | 134.931 |
| nvfp4 | dequant_fp32 | resident_gpu | 4194304 | 0.089568 | 0.092160 | 213.653 |
| nvfp4 | dequant_fp16 | resident_gpu | 4194304 | 0.082944 | 0.087040 | 129.580 |
| nvfp4 | dequant_bf16 | resident_gpu | 4194304 | 0.090112 | 0.091136 | 119.273 |
| nvfp4 | quant_optimized | host_api | 4194304 | 5.087974 | 5.757672 | 3.761 |
| nvfp4 | quant_enumeration | resident_gpu | 16777216 | 2.819072 | 3.649440 | 27.153 |
| nvfp4 | quant_optimized | resident_gpu | 16777216 | 0.519088 | 0.800768 | 147.463 |
| nvfp4 | dequant_fp32 | resident_gpu | 16777216 | 0.338960 | 0.344064 | 225.826 |
| nvfp4 | dequant_fp16 | resident_gpu | 16777216 | 0.302480 | 0.306176 | 142.130 |
| nvfp4 | dequant_bf16 | resident_gpu | 16777216 | 0.332640 | 0.337920 | 129.244 |
| nvfp4 | quant_optimized | host_api | 16777216 | 17.437179 | 19.044315 | 4.390 |

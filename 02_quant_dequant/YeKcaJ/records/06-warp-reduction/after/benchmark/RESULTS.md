# 性能测试结果

GPU 驻留计时使用 CUDA event；host_api 包括量化的分配、传输和释放。
内部对照保留 shared scale 归约与枚举元素编码；两种格式的默认 block+nearest 均使用 scale/编码融合 kernel，NVFP4 另含两级全局归约。内部对照的 scale 也可能共享优化，优化前后请比较留存的 quant_optimized 记录。
中位数采用偶数样本中间两值均值，P95 使用 nearest-rank。CPU 仅测试 1M，预热一次后计时三次。

| 格式 | 操作 | 范围 | 元素数 | median ms | P95 ms | 逻辑 GB/s |
|---|---|---|---:|---:|---:|---:|
| mxfp8 | quant_enumeration | resident_gpu | 1048576 | 0.566784 | 1.245184 | 9.308 |
| mxfp8 | quant_optimized | resident_gpu | 1048576 | 0.034816 | 0.055072 | 151.529 |
| mxfp8 | dequant_fp32 | resident_gpu | 1048576 | 0.031744 | 0.036864 | 166.194 |
| mxfp8 | dequant_fp16 | resident_gpu | 1048576 | 0.025584 | 0.029696 | 124.238 |
| mxfp8 | dequant_bf16 | resident_gpu | 1048576 | 0.027600 | 0.032512 | 115.163 |
| mxfp8 | quant_optimized | host_api | 1048576 | 3.868450 | 5.940636 | 1.364 |
| mxfp8 | quant_reference | cpu | 1048576 | 1294.859316 | 1301.408979 | 0.004 |
| mxfp8 | dequant_fp32 | cpu | 1048576 | 14.235217 | 14.439547 | 0.371 |
| mxfp8 | quant_enumeration | resident_gpu | 4194304 | 2.253824 | 3.468288 | 9.363 |
| mxfp8 | quant_optimized | resident_gpu | 4194304 | 0.123904 | 0.150528 | 170.314 |
| mxfp8 | dequant_fp32 | resident_gpu | 4194304 | 0.110592 | 0.115680 | 190.815 |
| mxfp8 | dequant_fp16 | resident_gpu | 4194304 | 0.103424 | 0.105472 | 122.931 |
| mxfp8 | dequant_bf16 | resident_gpu | 4194304 | 0.108544 | 0.109568 | 117.132 |
| mxfp8 | quant_optimized | host_api | 4194304 | 5.159724 | 6.890078 | 4.090 |
| mxfp8 | quant_enumeration | resident_gpu | 16777216 | 7.881728 | 8.727552 | 10.710 |
| mxfp8 | quant_optimized | resident_gpu | 16777216 | 0.433120 | 0.444128 | 194.889 |
| mxfp8 | dequant_fp32 | resident_gpu | 16777216 | 0.403456 | 0.413408 | 209.218 |
| mxfp8 | dequant_fp16 | resident_gpu | 16777216 | 0.362496 | 0.371424 | 140.294 |
| mxfp8 | dequant_bf16 | resident_gpu | 16777216 | 0.379232 | 0.386048 | 134.102 |
| mxfp8 | quant_optimized | host_api | 16777216 | 18.963694 | 21.623600 | 4.451 |
| nvfp4 | quant_enumeration | resident_gpu | 1048576 | 0.195936 | 0.200512 | 24.417 |
| nvfp4 | quant_optimized | resident_gpu | 1048576 | 0.048128 | 0.049056 | 99.404 |
| nvfp4 | dequant_fp32 | resident_gpu | 1048576 | 0.025600 | 0.029696 | 186.880 |
| nvfp4 | dequant_fp16 | resident_gpu | 1048576 | 0.020384 | 0.022528 | 131.818 |
| nvfp4 | dequant_bf16 | resident_gpu | 1048576 | 0.022528 | 0.026624 | 119.273 |
| nvfp4 | quant_optimized | host_api | 1048576 | 1.883629 | 2.387774 | 2.540 |
| nvfp4 | quant_reference | cpu | 1048576 | 88.877629 | 91.723530 | 0.054 |
| nvfp4 | dequant_fp32 | cpu | 1048576 | 11.492957 | 11.938990 | 0.416 |
| nvfp4 | quant_enumeration | resident_gpu | 4194304 | 0.722944 | 0.951296 | 26.470 |
| nvfp4 | quant_optimized | resident_gpu | 4194304 | 0.140288 | 0.148480 | 136.409 |
| nvfp4 | dequant_fp32 | resident_gpu | 4194304 | 0.089088 | 0.093088 | 214.805 |
| nvfp4 | dequant_fp16 | resident_gpu | 4194304 | 0.079728 | 0.089088 | 134.807 |
| nvfp4 | dequant_bf16 | resident_gpu | 4194304 | 0.087040 | 0.091136 | 123.482 |
| nvfp4 | quant_optimized | host_api | 4194304 | 4.976443 | 5.874972 | 3.845 |
| nvfp4 | quant_enumeration | resident_gpu | 16777216 | 2.834432 | 3.135488 | 27.006 |
| nvfp4 | quant_optimized | resident_gpu | 16777216 | 0.518144 | 0.534528 | 147.731 |
| nvfp4 | dequant_fp32 | resident_gpu | 16777216 | 0.340016 | 0.346016 | 225.125 |
| nvfp4 | dequant_fp16 | resident_gpu | 16777216 | 0.302080 | 0.307200 | 142.319 |
| nvfp4 | dequant_bf16 | resident_gpu | 16777216 | 0.336848 | 0.355328 | 127.629 |
| nvfp4 | quant_optimized | host_api | 16777216 | 16.801077 | 18.888079 | 4.556 |

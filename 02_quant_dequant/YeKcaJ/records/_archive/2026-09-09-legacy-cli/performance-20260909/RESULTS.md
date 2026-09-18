# 性能测试结果

GPU 驻留计时使用 CUDA event；host_api 包括量化的分配、传输和释放。
基线与优化版共用两级归约及相同显存，仅元素编码不同；不是原始 CLI 的完整实现对比。
中位数采用偶数样本中间两值均值，P95 使用 nearest-rank。CPU 仅测试 1M，预热一次后计时三次。

| 格式 | 操作 | 范围 | 元素数 | median ms | P95 ms | 逻辑 GB/s |
|---|---|---|---:|---:|---:|---:|
| mxfp8 | quant_enumeration | resident_gpu | 1048576 | 0.568320 | 0.573440 | 9.283 |
| mxfp8 | quant_binary_search | resident_gpu | 1048576 | 0.186368 | 0.188416 | 28.308 |
| mxfp8 | dequant_fp32 | resident_gpu | 1048576 | 0.033792 | 0.038912 | 156.121 |
| mxfp8 | dequant_fp16 | resident_gpu | 1048576 | 0.029664 | 0.029696 | 107.150 |
| mxfp8 | dequant_bf16 | resident_gpu | 1048576 | 0.030720 | 0.031744 | 103.467 |
| mxfp8 | quant_binary_search | host_api | 1048576 | 2.545216 | 2.913950 | 2.073 |
| mxfp8 | quant_reference | cpu | 1048576 | 1326.554471 | 1369.790218 | 0.004 |
| mxfp8 | dequant_fp32 | cpu | 1048576 | 13.776810 | 15.850128 | 0.383 |
| mxfp8 | quant_enumeration | resident_gpu | 4194304 | 2.245632 | 2.622464 | 9.397 |
| mxfp8 | quant_binary_search | resident_gpu | 4194304 | 0.725504 | 1.187840 | 29.087 |
| mxfp8 | dequant_fp32 | resident_gpu | 4194304 | 0.118784 | 0.124928 | 177.655 |
| mxfp8 | dequant_fp16 | resident_gpu | 4194304 | 0.113664 | 0.113664 | 111.856 |
| mxfp8 | dequant_bf16 | resident_gpu | 4194304 | 0.118784 | 0.119808 | 107.034 |
| mxfp8 | quant_binary_search | host_api | 4194304 | 5.572835 | 6.550620 | 3.787 |
| mxfp8 | quant_enumeration | resident_gpu | 16777216 | 8.804864 | 9.055232 | 9.587 |
| mxfp8 | quant_binary_search | resident_gpu | 16777216 | 2.516416 | 3.424256 | 33.544 |
| mxfp8 | dequant_fp32 | resident_gpu | 16777216 | 0.423424 | 0.721920 | 199.352 |
| mxfp8 | dequant_fp16 | resident_gpu | 16777216 | 0.394240 | 0.402432 | 128.997 |
| mxfp8 | dequant_bf16 | resident_gpu | 16777216 | 0.414720 | 0.611328 | 122.627 |
| mxfp8 | quant_binary_search | host_api | 16777216 | 19.885462 | 23.811995 | 4.245 |
| nvfp4 | quant_enumeration | resident_gpu | 1048576 | 0.317440 | 0.367328 | 15.071 |
| nvfp4 | quant_binary_search | resident_gpu | 1048576 | 0.484352 | 0.897024 | 9.877 |
| nvfp4 | dequant_fp32 | resident_gpu | 1048576 | 0.055296 | 0.056320 | 86.519 |
| nvfp4 | dequant_fp16 | resident_gpu | 1048576 | 0.047040 | 0.048032 | 57.121 |
| nvfp4 | dequant_bf16 | resident_gpu | 1048576 | 0.047104 | 0.048128 | 57.044 |
| nvfp4 | quant_binary_search | host_api | 1048576 | 2.199397 | 2.928146 | 2.175 |
| nvfp4 | quant_reference | cpu | 1048576 | 101.457136 | 103.035131 | 0.047 |
| nvfp4 | dequant_fp32 | cpu | 1048576 | 11.397470 | 11.774341 | 0.420 |
| nvfp4 | quant_enumeration | resident_gpu | 4194304 | 1.148480 | 1.155968 | 16.662 |
| nvfp4 | quant_binary_search | resident_gpu | 4194304 | 1.798144 | 2.647040 | 10.642 |
| nvfp4 | dequant_fp32 | resident_gpu | 4194304 | 0.203776 | 0.210944 | 93.910 |
| nvfp4 | dequant_fp16 | resident_gpu | 4194304 | 0.175088 | 0.185344 | 61.386 |
| nvfp4 | dequant_bf16 | resident_gpu | 4194304 | 0.177152 | 0.178176 | 60.671 |
| nvfp4 | quant_binary_search | host_api | 4194304 | 6.548274 | 7.656907 | 2.922 |
| nvfp4 | quant_enumeration | resident_gpu | 16777216 | 4.530688 | 5.469184 | 16.895 |
| nvfp4 | quant_binary_search | resident_gpu | 16777216 | 7.735296 | 8.236032 | 9.896 |
| nvfp4 | dequant_fp32 | resident_gpu | 16777216 | 0.813056 | 1.487840 | 94.146 |
| nvfp4 | dequant_fp16 | resident_gpu | 16777216 | 0.684912 | 0.962560 | 62.770 |
| nvfp4 | dequant_bf16 | resident_gpu | 16777216 | 0.700928 | 1.089536 | 61.335 |
| nvfp4 | quant_binary_search | host_api | 16777216 | 23.688713 | 26.312011 | 3.231 |

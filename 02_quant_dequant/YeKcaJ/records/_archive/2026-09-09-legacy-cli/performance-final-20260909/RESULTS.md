# 性能测试结果

GPU 驻留计时使用 CUDA event；host_api 包括量化的分配、传输和释放。
基线与优化版共用两级归约及相同显存，仅元素编码不同；不是原始 CLI 的完整实现对比。
中位数采用偶数样本中间两值均值，P95 使用 nearest-rank。CPU 仅测试 1M，预热一次后计时三次。

| 格式 | 操作 | 范围 | 元素数 | median ms | P95 ms | 逻辑 GB/s |
|---|---|---|---:|---:|---:|---:|
| mxfp8 | quant_enumeration | resident_gpu | 1048576 | 0.566272 | 0.942080 | 9.316 |
| mxfp8 | quant_optimized | resident_gpu | 1048576 | 0.184320 | 0.185344 | 28.622 |
| mxfp8 | dequant_fp32 | resident_gpu | 1048576 | 0.031664 | 0.031744 | 166.613 |
| mxfp8 | dequant_fp16 | resident_gpu | 1048576 | 0.023552 | 0.023552 | 134.957 |
| mxfp8 | dequant_bf16 | resident_gpu | 1048576 | 0.026048 | 0.026624 | 122.025 |
| mxfp8 | quant_optimized | host_api | 1048576 | 2.536448 | 2.746771 | 2.080 |
| mxfp8 | quant_reference | cpu | 1048576 | 1396.516374 | 1408.468844 | 0.004 |
| mxfp8 | dequant_fp32 | cpu | 1048576 | 14.500998 | 18.164678 | 0.364 |
| mxfp8 | quant_enumeration | resident_gpu | 4194304 | 2.668000 | 3.343200 | 7.910 |
| mxfp8 | quant_optimized | resident_gpu | 4194304 | 0.714240 | 1.135552 | 29.546 |
| mxfp8 | dequant_fp32 | resident_gpu | 4194304 | 0.110592 | 0.111616 | 190.815 |
| mxfp8 | dequant_fp16 | resident_gpu | 4194304 | 0.103888 | 0.108608 | 122.382 |
| mxfp8 | dequant_bf16 | resident_gpu | 4194304 | 0.113040 | 0.137216 | 112.473 |
| mxfp8 | quant_optimized | host_api | 4194304 | 6.898732 | 8.063666 | 3.059 |
| mxfp8 | quant_enumeration | resident_gpu | 16777216 | 9.152496 | 9.948160 | 9.223 |
| mxfp8 | quant_optimized | resident_gpu | 16777216 | 2.507248 | 3.289088 | 33.667 |
| mxfp8 | dequant_fp32 | resident_gpu | 16777216 | 0.404480 | 0.409600 | 208.689 |
| mxfp8 | dequant_fp16 | resident_gpu | 16777216 | 0.363520 | 0.373760 | 139.899 |
| mxfp8 | dequant_bf16 | resident_gpu | 16777216 | 0.380352 | 0.399360 | 133.708 |
| mxfp8 | quant_optimized | host_api | 16777216 | 21.241967 | 24.994602 | 3.974 |
| nvfp4 | quant_enumeration | resident_gpu | 1048576 | 0.317952 | 0.329728 | 15.047 |
| nvfp4 | quant_optimized | resident_gpu | 1048576 | 0.316416 | 0.348224 | 15.120 |
| nvfp4 | dequant_fp32 | resident_gpu | 1048576 | 0.025600 | 0.030720 | 186.880 |
| nvfp4 | dequant_fp16 | resident_gpu | 1048576 | 0.020480 | 0.023552 | 131.200 |
| nvfp4 | dequant_bf16 | resident_gpu | 1048576 | 0.022528 | 0.027648 | 119.273 |
| nvfp4 | quant_optimized | host_api | 1048576 | 2.160224 | 2.815966 | 2.215 |
| nvfp4 | quant_reference | cpu | 1048576 | 99.184757 | 101.032502 | 0.048 |
| nvfp4 | dequant_fp32 | cpu | 1048576 | 11.896237 | 12.297040 | 0.402 |
| nvfp4 | quant_enumeration | resident_gpu | 4194304 | 1.155072 | 1.604608 | 16.567 |
| nvfp4 | quant_optimized | resident_gpu | 4194304 | 1.145856 | 1.385472 | 16.701 |
| nvfp4 | dequant_fp32 | resident_gpu | 4194304 | 0.088064 | 0.093920 | 217.302 |
| nvfp4 | dequant_fp16 | resident_gpu | 4194304 | 0.078848 | 0.084992 | 136.312 |
| nvfp4 | dequant_bf16 | resident_gpu | 4194304 | 0.087040 | 0.092896 | 123.482 |
| nvfp4 | quant_optimized | host_api | 4194304 | 6.287036 | 7.263313 | 3.044 |
| nvfp4 | quant_enumeration | resident_gpu | 16777216 | 4.822432 | 5.686272 | 15.873 |
| nvfp4 | quant_optimized | resident_gpu | 16777216 | 4.671488 | 5.403648 | 16.386 |
| nvfp4 | dequant_fp32 | resident_gpu | 16777216 | 0.341984 | 0.346112 | 223.829 |
| nvfp4 | dequant_fp16 | resident_gpu | 16777216 | 0.303072 | 0.307200 | 141.853 |
| nvfp4 | dequant_bf16 | resident_gpu | 16777216 | 0.334848 | 0.338944 | 128.391 |
| nvfp4 | quant_optimized | host_api | 16777216 | 23.824296 | 28.803087 | 3.213 |

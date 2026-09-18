# 性能测试结果

GPU 驻留计时使用 CUDA event；host_api 包括量化的分配、传输和释放。
基线使用逐小块 shared scale 归约与枚举编码；优化版使用多组 warp 归约和快速编码。两者共用全局两级归约与显存；不是原始 CLI 的完整实现对比。
中位数采用偶数样本中间两值均值，P95 使用 nearest-rank。CPU 仅测试 1M，预热一次后计时三次。

| 格式 | 操作 | 范围 | 元素数 | median ms | P95 ms | 逻辑 GB/s |
|---|---|---|---:|---:|---:|---:|
| mxfp8 | quant_enumeration | resident_gpu | 1048576 | 0.565248 | 0.911360 | 9.333 |
| mxfp8 | quant_optimized | resident_gpu | 1048576 | 0.055728 | 0.058368 | 94.668 |
| mxfp8 | dequant_fp32 | resident_gpu | 1048576 | 0.031744 | 0.032704 | 166.194 |
| mxfp8 | dequant_fp16 | resident_gpu | 1048576 | 0.026624 | 0.026624 | 119.385 |
| mxfp8 | dequant_bf16 | resident_gpu | 1048576 | 0.028096 | 0.028672 | 113.130 |
| mxfp8 | quant_optimized | host_api | 1048576 | 2.622489 | 3.338045 | 2.012 |
| mxfp8 | quant_reference | cpu | 1048576 | 1462.654363 | 1466.461919 | 0.004 |
| mxfp8 | dequant_fp32 | cpu | 1048576 | 15.887773 | 16.746001 | 0.332 |
| mxfp8 | quant_enumeration | resident_gpu | 4194304 | 2.236416 | 3.262464 | 9.436 |
| mxfp8 | quant_optimized | resident_gpu | 4194304 | 0.199680 | 0.215040 | 105.682 |
| mxfp8 | dequant_fp32 | resident_gpu | 4194304 | 0.111616 | 0.119808 | 189.064 |
| mxfp8 | dequant_fp16 | resident_gpu | 4194304 | 0.103424 | 0.104448 | 122.931 |
| mxfp8 | dequant_bf16 | resident_gpu | 4194304 | 0.108544 | 0.109568 | 117.132 |
| mxfp8 | quant_optimized | host_api | 4194304 | 5.870815 | 6.268961 | 3.594 |
| mxfp8 | quant_enumeration | resident_gpu | 16777216 | 8.895984 | 9.241600 | 9.489 |
| mxfp8 | quant_optimized | resident_gpu | 16777216 | 0.708608 | 1.051648 | 119.121 |
| mxfp8 | dequant_fp32 | resident_gpu | 16777216 | 0.402944 | 0.414720 | 209.484 |
| mxfp8 | dequant_fp16 | resident_gpu | 16777216 | 0.361472 | 0.386848 | 140.691 |
| mxfp8 | dequant_bf16 | resident_gpu | 16777216 | 0.378560 | 0.380672 | 134.340 |
| mxfp8 | quant_optimized | host_api | 16777216 | 20.476395 | 22.335401 | 4.122 |
| nvfp4 | quant_enumeration | resident_gpu | 1048576 | 0.315392 | 0.333632 | 15.169 |
| nvfp4 | quant_optimized | resident_gpu | 1048576 | 0.143872 | 0.147456 | 33.253 |
| nvfp4 | dequant_fp32 | resident_gpu | 1048576 | 0.025600 | 0.026624 | 186.880 |
| nvfp4 | dequant_fp16 | resident_gpu | 1048576 | 0.019456 | 0.020480 | 138.105 |
| nvfp4 | dequant_bf16 | resident_gpu | 1048576 | 0.022528 | 0.027456 | 119.273 |
| nvfp4 | quant_optimized | host_api | 1048576 | 2.137247 | 2.522571 | 2.238 |
| nvfp4 | quant_reference | cpu | 1048576 | 104.046515 | 119.244268 | 0.046 |
| nvfp4 | dequant_fp32 | cpu | 1048576 | 13.070339 | 13.486333 | 0.366 |
| nvfp4 | quant_enumeration | resident_gpu | 4194304 | 1.143808 | 1.397760 | 16.731 |
| nvfp4 | quant_optimized | resident_gpu | 4194304 | 0.453632 | 0.830464 | 42.185 |
| nvfp4 | dequant_fp32 | resident_gpu | 4194304 | 0.088064 | 0.092160 | 217.302 |
| nvfp4 | dequant_fp16 | resident_gpu | 4194304 | 0.078896 | 0.080768 | 136.229 |
| nvfp4 | dequant_bf16 | resident_gpu | 4194304 | 0.087040 | 0.088000 | 123.482 |
| nvfp4 | quant_optimized | host_api | 4194304 | 5.814199 | 6.463159 | 3.291 |
| nvfp4 | quant_enumeration | resident_gpu | 16777216 | 4.507536 | 5.241856 | 16.982 |
| nvfp4 | quant_optimized | resident_gpu | 16777216 | 1.747456 | 2.686976 | 43.804 |
| nvfp4 | dequant_fp32 | resident_gpu | 16777216 | 0.339920 | 0.346112 | 225.188 |
| nvfp4 | dequant_fp16 | resident_gpu | 16777216 | 0.303104 | 0.313344 | 141.838 |
| nvfp4 | dequant_bf16 | resident_gpu | 16777216 | 0.332800 | 0.333824 | 129.182 |
| nvfp4 | quant_optimized | host_api | 16777216 | 19.882888 | 22.213678 | 3.850 |

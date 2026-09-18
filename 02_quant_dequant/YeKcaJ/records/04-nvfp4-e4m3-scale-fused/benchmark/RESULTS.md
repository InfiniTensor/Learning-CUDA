# 性能测试结果

GPU 驻留计时使用 CUDA event；host_api 包括量化的分配、传输和释放。
内部对照使用 shared scale 归约与枚举编码；MXFP8 默认使用 scale/直接编码融合 kernel，NVFP4 默认保留分组归约和快速编码。不是原始 CLI 的完整实现对比。
中位数采用偶数样本中间两值均值，P95 使用 nearest-rank。CPU 仅测试 1M，预热一次后计时三次。

| 格式 | 操作 | 范围 | 元素数 | median ms | P95 ms | 逻辑 GB/s |
|---|---|---|---:|---:|---:|---:|
| mxfp8 | quant_enumeration | resident_gpu | 1048576 | 0.566272 | 0.572416 | 9.316 |
| mxfp8 | quant_optimized | resident_gpu | 1048576 | 0.034816 | 0.034816 | 151.529 |
| mxfp8 | dequant_fp32 | resident_gpu | 1048576 | 0.031120 | 0.031744 | 169.526 |
| mxfp8 | dequant_fp16 | resident_gpu | 1048576 | 0.024576 | 0.024576 | 129.333 |
| mxfp8 | dequant_bf16 | resident_gpu | 1048576 | 0.026624 | 0.027648 | 119.385 |
| mxfp8 | quant_optimized | host_api | 1048576 | 2.155924 | 2.518721 | 2.447 |
| mxfp8 | quant_reference | cpu | 1048576 | 1256.908394 | 1258.304892 | 0.004 |
| mxfp8 | dequant_fp32 | cpu | 1048576 | 13.043707 | 14.032415 | 0.404 |
| mxfp8 | quant_enumeration | resident_gpu | 4194304 | 2.237296 | 2.245536 | 9.432 |
| mxfp8 | quant_optimized | resident_gpu | 4194304 | 0.123632 | 0.123904 | 170.689 |
| mxfp8 | dequant_fp32 | resident_gpu | 4194304 | 0.111616 | 0.114688 | 189.064 |
| mxfp8 | dequant_fp16 | resident_gpu | 4194304 | 0.104448 | 0.143360 | 121.725 |
| mxfp8 | dequant_bf16 | resident_gpu | 4194304 | 0.128000 | 0.129024 | 99.328 |
| mxfp8 | quant_optimized | host_api | 4194304 | 4.934837 | 5.365214 | 4.276 |
| mxfp8 | quant_enumeration | resident_gpu | 16777216 | 8.912384 | 9.305952 | 9.471 |
| mxfp8 | quant_optimized | resident_gpu | 16777216 | 0.477184 | 0.500736 | 176.893 |
| mxfp8 | dequant_fp32 | resident_gpu | 16777216 | 0.402432 | 0.406528 | 209.751 |
| mxfp8 | dequant_fp16 | resident_gpu | 16777216 | 0.362496 | 0.366592 | 140.294 |
| mxfp8 | dequant_bf16 | resident_gpu | 16777216 | 0.379792 | 0.384832 | 133.905 |
| mxfp8 | quant_optimized | host_api | 16777216 | 16.499964 | 19.381248 | 5.116 |
| nvfp4 | quant_enumeration | resident_gpu | 1048576 | 0.210944 | 0.215040 | 22.680 |
| nvfp4 | quant_optimized | resident_gpu | 1048576 | 0.063488 | 0.064512 | 75.355 |
| nvfp4 | dequant_fp32 | resident_gpu | 1048576 | 0.025600 | 0.026624 | 186.880 |
| nvfp4 | dequant_fp16 | resident_gpu | 1048576 | 0.019856 | 0.020480 | 135.323 |
| nvfp4 | dequant_bf16 | resident_gpu | 1048576 | 0.022528 | 0.023552 | 119.273 |
| nvfp4 | quant_optimized | host_api | 1048576 | 1.700697 | 2.080735 | 2.813 |
| nvfp4 | quant_reference | cpu | 1048576 | 88.366582 | 89.400289 | 0.054 |
| nvfp4 | dequant_fp32 | cpu | 1048576 | 11.727503 | 11.817327 | 0.408 |
| nvfp4 | quant_enumeration | resident_gpu | 4194304 | 0.726528 | 0.741376 | 26.340 |
| nvfp4 | quant_optimized | resident_gpu | 4194304 | 0.142336 | 0.143360 | 134.446 |
| nvfp4 | dequant_fp32 | resident_gpu | 4194304 | 0.088064 | 0.089088 | 217.302 |
| nvfp4 | dequant_fp16 | resident_gpu | 4194304 | 0.079200 | 0.082944 | 135.706 |
| nvfp4 | dequant_bf16 | resident_gpu | 4194304 | 0.086896 | 0.087808 | 123.687 |
| nvfp4 | quant_optimized | host_api | 4194304 | 4.390481 | 4.858732 | 4.359 |
| nvfp4 | quant_enumeration | resident_gpu | 16777216 | 2.827792 | 3.054592 | 27.069 |
| nvfp4 | quant_optimized | resident_gpu | 16777216 | 0.519136 | 0.523264 | 147.449 |
| nvfp4 | dequant_fp32 | resident_gpu | 16777216 | 0.338944 | 0.346112 | 225.837 |
| nvfp4 | dequant_fp16 | resident_gpu | 16777216 | 0.303104 | 0.319488 | 141.838 |
| nvfp4 | dequant_bf16 | resident_gpu | 16777216 | 0.333568 | 0.334688 | 128.884 |
| nvfp4 | quant_optimized | host_api | 16777216 | 15.863228 | 18.460876 | 4.825 |

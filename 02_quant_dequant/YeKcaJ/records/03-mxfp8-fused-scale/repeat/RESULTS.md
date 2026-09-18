# 性能测试结果

GPU 驻留计时使用 CUDA event；host_api 包括量化的分配、传输和释放。
内部对照使用 shared scale 归约与枚举编码；MXFP8 默认使用 scale/直接编码融合 kernel，NVFP4 默认保留分组归约和快速编码。不是原始 CLI 的完整实现对比。
中位数采用偶数样本中间两值均值，P95 使用 nearest-rank。CPU 仅测试 1M，预热一次后计时三次。

| 格式 | 操作 | 范围 | 元素数 | median ms | P95 ms | 逻辑 GB/s |
|---|---|---|---:|---:|---:|---:|
| mxfp8 | quant_enumeration | resident_gpu | 1048576 | 0.566272 | 0.942080 | 9.316 |
| mxfp8 | quant_optimized | resident_gpu | 1048576 | 0.034816 | 0.039936 | 151.529 |
| mxfp8 | dequant_fp32 | resident_gpu | 1048576 | 0.031744 | 0.032768 | 166.194 |
| mxfp8 | dequant_fp16 | resident_gpu | 1048576 | 0.024544 | 0.030720 | 129.502 |
| mxfp8 | dequant_bf16 | resident_gpu | 1048576 | 0.026624 | 0.031744 | 119.385 |
| mxfp8 | quant_optimized | host_api | 1048576 | 2.777097 | 3.458764 | 1.900 |
| mxfp8 | quant_reference | cpu | 1048576 | 1429.318340 | 1487.301541 | 0.004 |
| mxfp8 | dequant_fp32 | cpu | 1048576 | 16.892544 | 18.300872 | 0.312 |
| mxfp8 | quant_enumeration | resident_gpu | 4194304 | 2.237952 | 3.779584 | 9.429 |
| mxfp8 | quant_optimized | resident_gpu | 4194304 | 0.122880 | 0.127936 | 171.733 |
| mxfp8 | dequant_fp32 | resident_gpu | 4194304 | 0.111520 | 0.112640 | 189.227 |
| mxfp8 | dequant_fp16 | resident_gpu | 4194304 | 0.104448 | 0.110592 | 121.725 |
| mxfp8 | dequant_bf16 | resident_gpu | 4194304 | 0.109568 | 0.113664 | 116.037 |
| mxfp8 | quant_optimized | host_api | 4194304 | 6.179045 | 7.800763 | 3.415 |
| mxfp8 | quant_enumeration | resident_gpu | 16777216 | 9.211888 | 10.056704 | 9.163 |
| mxfp8 | quant_optimized | resident_gpu | 16777216 | 0.432128 | 0.828416 | 195.336 |
| mxfp8 | dequant_fp32 | resident_gpu | 16777216 | 0.404480 | 0.583680 | 208.689 |
| mxfp8 | dequant_fp16 | resident_gpu | 16777216 | 0.362496 | 0.369664 | 140.294 |
| mxfp8 | dequant_bf16 | resident_gpu | 16777216 | 0.378864 | 0.382976 | 134.233 |
| mxfp8 | quant_optimized | host_api | 16777216 | 22.352135 | 23.644252 | 3.776 |
| nvfp4 | quant_enumeration | resident_gpu | 1048576 | 0.320368 | 0.321536 | 14.933 |
| nvfp4 | quant_optimized | resident_gpu | 1048576 | 0.146432 | 0.148352 | 32.671 |
| nvfp4 | dequant_fp32 | resident_gpu | 1048576 | 0.025600 | 0.026624 | 186.880 |
| nvfp4 | dequant_fp16 | resident_gpu | 1048576 | 0.020368 | 0.020480 | 131.922 |
| nvfp4 | dequant_bf16 | resident_gpu | 1048576 | 0.023040 | 0.045056 | 116.622 |
| nvfp4 | quant_optimized | host_api | 1048576 | 2.142094 | 2.559161 | 2.233 |
| nvfp4 | quant_reference | cpu | 1048576 | 105.698345 | 108.144029 | 0.045 |
| nvfp4 | dequant_fp32 | cpu | 1048576 | 13.564043 | 14.376366 | 0.353 |
| nvfp4 | quant_enumeration | resident_gpu | 4194304 | 1.145824 | 1.460000 | 16.701 |
| nvfp4 | quant_optimized | resident_gpu | 4194304 | 0.449536 | 0.463872 | 42.569 |
| nvfp4 | dequant_fp32 | resident_gpu | 4194304 | 0.088064 | 0.093184 | 217.302 |
| nvfp4 | dequant_fp16 | resident_gpu | 4194304 | 0.078848 | 0.081920 | 136.312 |
| nvfp4 | dequant_bf16 | resident_gpu | 4194304 | 0.087040 | 0.090976 | 123.482 |
| nvfp4 | quant_optimized | host_api | 4194304 | 6.129867 | 6.989836 | 3.122 |
| nvfp4 | quant_enumeration | resident_gpu | 16777216 | 4.505440 | 5.185536 | 16.990 |
| nvfp4 | quant_optimized | resident_gpu | 16777216 | 2.107392 | 2.736128 | 36.323 |
| nvfp4 | dequant_fp32 | resident_gpu | 16777216 | 0.339984 | 0.353280 | 225.146 |
| nvfp4 | dequant_fp16 | resident_gpu | 16777216 | 0.302080 | 0.306176 | 142.319 |
| nvfp4 | dequant_bf16 | resident_gpu | 16777216 | 0.333296 | 0.345088 | 128.989 |
| nvfp4 | quant_optimized | host_api | 16777216 | 21.932022 | 30.106969 | 3.490 |

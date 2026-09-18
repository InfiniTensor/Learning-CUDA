# 性能测试结果

GPU 驻留计时使用 CUDA event；host_api 包括量化的分配、传输和释放。
基线使用逐小块 shared scale 归约与枚举编码；优化版使用多组 warp 归约和快速编码。两者共用全局两级归约与显存；不是原始 CLI 的完整实现对比。
中位数采用偶数样本中间两值均值，P95 使用 nearest-rank。CPU 仅测试 1M，预热一次后计时三次。

| 格式 | 操作 | 范围 | 元素数 | median ms | P95 ms | 逻辑 GB/s |
|---|---|---|---:|---:|---:|---:|
| mxfp8 | quant_enumeration | resident_gpu | 1048576 | 0.566272 | 0.982016 | 9.316 |
| mxfp8 | quant_optimized | resident_gpu | 1048576 | 0.124880 | 0.152576 | 42.246 |
| mxfp8 | dequant_fp32 | resident_gpu | 1048576 | 0.031680 | 0.031744 | 166.529 |
| mxfp8 | dequant_fp16 | resident_gpu | 1048576 | 0.025600 | 0.026624 | 124.160 |
| mxfp8 | dequant_bf16 | resident_gpu | 1048576 | 0.027648 | 0.028480 | 114.963 |
| mxfp8 | quant_optimized | host_api | 1048576 | 2.346497 | 3.103601 | 2.248 |
| mxfp8 | quant_reference | cpu | 1048576 | 1345.309175 | 1360.570139 | 0.004 |
| mxfp8 | dequant_fp32 | cpu | 1048576 | 15.701619 | 16.094164 | 0.336 |
| mxfp8 | quant_enumeration | resident_gpu | 4194304 | 2.237392 | 3.174400 | 9.432 |
| mxfp8 | quant_optimized | resident_gpu | 4194304 | 0.468352 | 0.847872 | 45.057 |
| mxfp8 | dequant_fp32 | resident_gpu | 4194304 | 0.111616 | 0.115712 | 189.064 |
| mxfp8 | dequant_fp16 | resident_gpu | 4194304 | 0.103424 | 0.104448 | 122.931 |
| mxfp8 | dequant_bf16 | resident_gpu | 4194304 | 0.109392 | 0.115712 | 116.224 |
| mxfp8 | quant_optimized | host_api | 4194304 | 5.638488 | 6.329715 | 3.743 |
| mxfp8 | quant_enumeration | resident_gpu | 16777216 | 8.769024 | 9.433088 | 9.626 |
| mxfp8 | quant_optimized | resident_gpu | 16777216 | 1.642496 | 2.674688 | 51.392 |
| mxfp8 | dequant_fp32 | resident_gpu | 16777216 | 0.406048 | 0.414720 | 207.883 |
| mxfp8 | dequant_fp16 | resident_gpu | 16777216 | 0.365056 | 0.378560 | 139.310 |
| mxfp8 | dequant_bf16 | resident_gpu | 16777216 | 0.381392 | 0.401088 | 133.343 |
| mxfp8 | quant_optimized | host_api | 16777216 | 19.634121 | 22.877268 | 4.299 |
| nvfp4 | quant_enumeration | resident_gpu | 1048576 | 0.322560 | 0.331680 | 14.832 |
| nvfp4 | quant_optimized | resident_gpu | 1048576 | 0.143360 | 0.156672 | 33.371 |
| nvfp4 | dequant_fp32 | resident_gpu | 1048576 | 0.025600 | 0.026624 | 186.880 |
| nvfp4 | dequant_fp16 | resident_gpu | 1048576 | 0.020480 | 0.020480 | 131.200 |
| nvfp4 | dequant_bf16 | resident_gpu | 1048576 | 0.022528 | 0.026624 | 119.273 |
| nvfp4 | quant_optimized | host_api | 1048576 | 1.826102 | 2.332746 | 2.620 |
| nvfp4 | quant_reference | cpu | 1048576 | 93.764216 | 95.079143 | 0.051 |
| nvfp4 | dequant_fp32 | cpu | 1048576 | 12.472241 | 13.985013 | 0.384 |
| nvfp4 | quant_enumeration | resident_gpu | 4194304 | 1.153536 | 1.569792 | 16.589 |
| nvfp4 | quant_optimized | resident_gpu | 4194304 | 0.452608 | 0.872448 | 42.281 |
| nvfp4 | dequant_fp32 | resident_gpu | 4194304 | 0.088064 | 0.089088 | 217.302 |
| nvfp4 | dequant_fp16 | resident_gpu | 4194304 | 0.079872 | 0.093184 | 134.564 |
| nvfp4 | dequant_bf16 | resident_gpu | 4194304 | 0.087040 | 0.088064 | 123.482 |
| nvfp4 | quant_optimized | host_api | 4194304 | 5.058294 | 5.811210 | 3.783 |
| nvfp4 | quant_enumeration | resident_gpu | 16777216 | 4.548096 | 5.478400 | 16.830 |
| nvfp4 | quant_optimized | resident_gpu | 16777216 | 1.755648 | 2.736128 | 43.600 |
| nvfp4 | dequant_fp32 | resident_gpu | 16777216 | 0.341856 | 0.348160 | 223.913 |
| nvfp4 | dequant_fp16 | resident_gpu | 16777216 | 0.304032 | 0.308224 | 141.405 |
| nvfp4 | dequant_bf16 | resident_gpu | 16777216 | 0.334848 | 0.339744 | 128.391 |
| nvfp4 | quant_optimized | host_api | 16777216 | 20.015056 | 21.861885 | 3.824 |

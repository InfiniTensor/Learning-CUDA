# 性能测试结果

GPU 驻留计时使用 CUDA event；host_api 包括量化的分配、传输和释放。
基线使用逐小块 shared scale 归约与枚举编码；优化版使用多组 warp 归约和快速编码。两者共用全局两级归约与显存；不是原始 CLI 的完整实现对比。
中位数采用偶数样本中间两值均值，P95 使用 nearest-rank。CPU 仅测试 1M，预热一次后计时三次。

| 格式 | 操作 | 范围 | 元素数 | median ms | P95 ms | 逻辑 GB/s |
|---|---|---|---:|---:|---:|---:|
| mxfp8 | quant_enumeration | resident_gpu | 1048576 | 0.566784 | 0.577536 | 9.308 |
| mxfp8 | quant_optimized | resident_gpu | 1048576 | 0.124928 | 0.129888 | 42.230 |
| mxfp8 | dequant_fp32 | resident_gpu | 1048576 | 0.031744 | 0.035840 | 166.194 |
| mxfp8 | dequant_fp16 | resident_gpu | 1048576 | 0.025472 | 0.025600 | 124.784 |
| mxfp8 | dequant_bf16 | resident_gpu | 1048576 | 0.027648 | 0.027648 | 114.963 |
| mxfp8 | quant_optimized | host_api | 1048576 | 2.845047 | 3.641677 | 1.854 |
| mxfp8 | quant_reference | cpu | 1048576 | 1506.161993 | 1561.378891 | 0.004 |
| mxfp8 | dequant_fp32 | cpu | 1048576 | 18.882875 | 18.925033 | 0.279 |
| mxfp8 | quant_enumeration | resident_gpu | 4194304 | 2.444032 | 3.439616 | 8.634 |
| mxfp8 | quant_optimized | resident_gpu | 4194304 | 0.468992 | 0.478208 | 44.996 |
| mxfp8 | dequant_fp32 | resident_gpu | 4194304 | 0.112064 | 0.116736 | 188.308 |
| mxfp8 | dequant_fp16 | resident_gpu | 4194304 | 0.103424 | 0.113664 | 122.931 |
| mxfp8 | dequant_bf16 | resident_gpu | 4194304 | 0.109936 | 0.114688 | 115.649 |
| mxfp8 | quant_optimized | host_api | 4194304 | 7.155556 | 9.405614 | 2.949 |
| mxfp8 | quant_enumeration | resident_gpu | 16777216 | 8.919024 | 9.476064 | 9.464 |
| mxfp8 | quant_optimized | resident_gpu | 16777216 | 1.628672 | 3.079168 | 51.828 |
| mxfp8 | dequant_fp32 | resident_gpu | 16777216 | 0.402432 | 0.409568 | 209.751 |
| mxfp8 | dequant_fp16 | resident_gpu | 16777216 | 0.362496 | 0.366592 | 140.294 |
| mxfp8 | dequant_bf16 | resident_gpu | 16777216 | 0.379392 | 0.382976 | 134.046 |
| mxfp8 | quant_optimized | host_api | 16777216 | 20.667298 | 22.801836 | 4.084 |
| nvfp4 | quant_enumeration | resident_gpu | 1048576 | 0.319328 | 0.320512 | 14.982 |
| nvfp4 | quant_optimized | resident_gpu | 1048576 | 0.145968 | 0.148384 | 32.775 |
| nvfp4 | dequant_fp32 | resident_gpu | 1048576 | 0.028704 | 0.029696 | 166.671 |
| nvfp4 | dequant_fp16 | resident_gpu | 1048576 | 0.019456 | 0.024576 | 138.105 |
| nvfp4 | dequant_bf16 | resident_gpu | 1048576 | 0.022528 | 0.023552 | 119.273 |
| nvfp4 | quant_optimized | host_api | 1048576 | 2.013645 | 2.718667 | 2.376 |
| nvfp4 | quant_reference | cpu | 1048576 | 105.043118 | 105.878674 | 0.046 |
| nvfp4 | dequant_fp32 | cpu | 1048576 | 14.282957 | 15.347272 | 0.335 |
| nvfp4 | quant_enumeration | resident_gpu | 4194304 | 1.143808 | 1.524672 | 16.731 |
| nvfp4 | quant_optimized | resident_gpu | 4194304 | 0.452096 | 0.789504 | 42.328 |
| nvfp4 | dequant_fp32 | resident_gpu | 4194304 | 0.088064 | 0.096992 | 217.302 |
| nvfp4 | dequant_fp16 | resident_gpu | 4194304 | 0.078848 | 0.082944 | 136.312 |
| nvfp4 | dequant_bf16 | resident_gpu | 4194304 | 0.087040 | 0.091136 | 123.482 |
| nvfp4 | quant_optimized | host_api | 4194304 | 5.331207 | 6.657581 | 3.590 |
| nvfp4 | quant_enumeration | resident_gpu | 16777216 | 4.770784 | 5.660416 | 16.045 |
| nvfp4 | quant_optimized | resident_gpu | 16777216 | 1.741824 | 2.703104 | 43.946 |
| nvfp4 | dequant_fp32 | resident_gpu | 16777216 | 0.339408 | 0.343040 | 225.528 |
| nvfp4 | dequant_fp16 | resident_gpu | 16777216 | 0.302832 | 0.316416 | 141.965 |
| nvfp4 | dequant_bf16 | resident_gpu | 16777216 | 0.332800 | 0.333824 | 129.182 |
| nvfp4 | quant_optimized | host_api | 16777216 | 20.392497 | 23.420690 | 3.754 |

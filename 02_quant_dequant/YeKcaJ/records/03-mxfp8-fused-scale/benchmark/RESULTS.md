# 性能测试结果

GPU 驻留计时使用 CUDA event；host_api 包括量化的分配、传输和释放。
基线使用逐小块 shared scale 归约与枚举编码；优化版使用多组 warp 归约和快速编码。两者共用全局两级归约与显存；不是原始 CLI 的完整实现对比。
中位数采用偶数样本中间两值均值，P95 使用 nearest-rank。CPU 仅测试 1M，预热一次后计时三次。

| 格式 | 操作 | 范围 | 元素数 | median ms | P95 ms | 逻辑 GB/s |
|---|---|---|---:|---:|---:|---:|
| mxfp8 | quant_enumeration | resident_gpu | 1048576 | 0.566272 | 1.007616 | 9.316 |
| mxfp8 | quant_optimized | resident_gpu | 1048576 | 0.034816 | 0.040960 | 151.529 |
| mxfp8 | dequant_fp32 | resident_gpu | 1048576 | 0.031680 | 0.037728 | 166.529 |
| mxfp8 | dequant_fp16 | resident_gpu | 1048576 | 0.025600 | 0.026624 | 124.160 |
| mxfp8 | dequant_bf16 | resident_gpu | 1048576 | 0.027648 | 0.027680 | 114.963 |
| mxfp8 | quant_optimized | host_api | 1048576 | 2.721380 | 3.144927 | 1.939 |
| mxfp8 | quant_reference | cpu | 1048576 | 1453.315685 | 1500.523489 | 0.004 |
| mxfp8 | dequant_fp32 | cpu | 1048576 | 16.080121 | 16.714027 | 0.328 |
| mxfp8 | quant_enumeration | resident_gpu | 4194304 | 2.237440 | 3.018752 | 9.432 |
| mxfp8 | quant_optimized | resident_gpu | 4194304 | 0.123312 | 0.128896 | 171.132 |
| mxfp8 | dequant_fp32 | resident_gpu | 4194304 | 0.111376 | 0.112480 | 189.472 |
| mxfp8 | dequant_fp16 | resident_gpu | 4194304 | 0.103424 | 0.104448 | 122.931 |
| mxfp8 | dequant_bf16 | resident_gpu | 4194304 | 0.109568 | 0.111616 | 116.037 |
| mxfp8 | quant_optimized | host_api | 4194304 | 5.958700 | 7.361336 | 3.541 |
| mxfp8 | quant_enumeration | resident_gpu | 16777216 | 8.663040 | 9.287680 | 9.744 |
| mxfp8 | quant_optimized | resident_gpu | 16777216 | 0.435648 | 0.737280 | 193.758 |
| mxfp8 | dequant_fp32 | resident_gpu | 16777216 | 0.401936 | 0.405568 | 210.009 |
| mxfp8 | dequant_fp16 | resident_gpu | 16777216 | 0.361984 | 0.367616 | 140.492 |
| mxfp8 | dequant_bf16 | resident_gpu | 16777216 | 0.378880 | 0.384928 | 134.227 |
| mxfp8 | quant_optimized | host_api | 16777216 | 20.027994 | 21.548169 | 4.215 |
| nvfp4 | quant_enumeration | resident_gpu | 1048576 | 0.319488 | 0.320512 | 14.974 |
| nvfp4 | quant_optimized | resident_gpu | 1048576 | 0.143360 | 0.152576 | 33.371 |
| nvfp4 | dequant_fp32 | resident_gpu | 1048576 | 0.026624 | 0.030720 | 179.692 |
| nvfp4 | dequant_fp16 | resident_gpu | 1048576 | 0.019904 | 0.020480 | 134.997 |
| nvfp4 | dequant_bf16 | resident_gpu | 1048576 | 0.022528 | 0.025600 | 119.273 |
| nvfp4 | quant_optimized | host_api | 1048576 | 2.132826 | 2.455789 | 2.243 |
| nvfp4 | quant_reference | cpu | 1048576 | 102.982454 | 103.390445 | 0.046 |
| nvfp4 | dequant_fp32 | cpu | 1048576 | 13.405879 | 14.026079 | 0.357 |
| nvfp4 | quant_enumeration | resident_gpu | 4194304 | 1.144832 | 1.418240 | 16.716 |
| nvfp4 | quant_optimized | resident_gpu | 4194304 | 0.451072 | 0.754688 | 42.425 |
| nvfp4 | dequant_fp32 | resident_gpu | 4194304 | 0.088064 | 0.089824 | 217.302 |
| nvfp4 | dequant_fp16 | resident_gpu | 4194304 | 0.078848 | 0.079872 | 136.312 |
| nvfp4 | dequant_bf16 | resident_gpu | 4194304 | 0.087040 | 0.088064 | 123.482 |
| nvfp4 | quant_optimized | host_api | 4194304 | 5.874244 | 7.390733 | 3.258 |
| nvfp4 | quant_enumeration | resident_gpu | 16777216 | 4.504976 | 4.751360 | 16.991 |
| nvfp4 | quant_optimized | resident_gpu | 16777216 | 2.068464 | 2.738176 | 37.006 |
| nvfp4 | dequant_fp32 | resident_gpu | 16777216 | 0.339968 | 0.344064 | 225.157 |
| nvfp4 | dequant_fp16 | resident_gpu | 16777216 | 0.302480 | 0.307200 | 142.130 |
| nvfp4 | dequant_bf16 | resident_gpu | 16777216 | 0.332800 | 0.333824 | 129.182 |
| nvfp4 | quant_optimized | host_api | 16777216 | 23.022030 | 26.027851 | 3.325 |

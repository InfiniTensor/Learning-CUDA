# 性能测试结果

GPU 驻留计时使用 CUDA event；host_api 包括量化的分配、传输和释放。
基线使用逐小块 shared scale 归约与枚举编码；优化版使用多组 warp 归约和快速编码。两者共用全局两级归约与显存；不是原始 CLI 的完整实现对比。
中位数采用偶数样本中间两值均值，P95 使用 nearest-rank。CPU 仅测试 1M，预热一次后计时三次。

| 格式 | 操作 | 范围 | 元素数 | median ms | P95 ms | 逻辑 GB/s |
|---|---|---|---:|---:|---:|---:|
| mxfp8 | quant_enumeration | resident_gpu | 1048576 | 0.566272 | 0.588800 | 9.316 |
| mxfp8 | quant_optimized | resident_gpu | 1048576 | 0.123904 | 0.130048 | 42.579 |
| mxfp8 | dequant_fp32 | resident_gpu | 1048576 | 0.031744 | 0.032736 | 166.194 |
| mxfp8 | dequant_fp16 | resident_gpu | 1048576 | 0.024576 | 0.025600 | 129.333 |
| mxfp8 | dequant_bf16 | resident_gpu | 1048576 | 0.027648 | 0.029600 | 114.963 |
| mxfp8 | quant_optimized | host_api | 1048576 | 2.755266 | 3.062598 | 1.915 |
| mxfp8 | quant_reference | cpu | 1048576 | 1454.409902 | 1470.301119 | 0.004 |
| mxfp8 | dequant_fp32 | cpu | 1048576 | 15.727446 | 18.053103 | 0.335 |
| mxfp8 | quant_enumeration | resident_gpu | 4194304 | 2.236416 | 2.247680 | 9.436 |
| mxfp8 | quant_optimized | resident_gpu | 4194304 | 0.467968 | 0.474112 | 45.094 |
| mxfp8 | dequant_fp32 | resident_gpu | 4194304 | 0.111616 | 0.112640 | 189.064 |
| mxfp8 | dequant_fp16 | resident_gpu | 4194304 | 0.103424 | 0.104448 | 122.931 |
| mxfp8 | dequant_bf16 | resident_gpu | 4194304 | 0.108944 | 0.109568 | 116.702 |
| mxfp8 | quant_optimized | host_api | 4194304 | 6.803138 | 8.464148 | 3.102 |
| mxfp8 | quant_enumeration | resident_gpu | 16777216 | 8.290304 | 9.019104 | 10.182 |
| mxfp8 | quant_optimized | resident_gpu | 16777216 | 1.634304 | 1.679360 | 51.649 |
| mxfp8 | dequant_fp32 | resident_gpu | 16777216 | 0.404928 | 0.406528 | 208.458 |
| mxfp8 | dequant_fp16 | resident_gpu | 16777216 | 0.364544 | 0.368640 | 139.506 |
| mxfp8 | dequant_bf16 | resident_gpu | 16777216 | 0.381952 | 0.384000 | 133.147 |
| mxfp8 | quant_optimized | host_api | 16777216 | 24.443100 | 27.221220 | 3.453 |
| nvfp4 | quant_enumeration | resident_gpu | 1048576 | 0.316416 | 0.321280 | 15.120 |
| nvfp4 | quant_optimized | resident_gpu | 1048576 | 0.142336 | 0.178048 | 33.612 |
| nvfp4 | dequant_fp32 | resident_gpu | 1048576 | 0.025600 | 0.030720 | 186.880 |
| nvfp4 | dequant_fp16 | resident_gpu | 1048576 | 0.020272 | 0.025280 | 132.546 |
| nvfp4 | dequant_bf16 | resident_gpu | 1048576 | 0.022528 | 0.027648 | 119.273 |
| nvfp4 | quant_optimized | host_api | 1048576 | 2.359228 | 2.761083 | 2.028 |
| nvfp4 | quant_reference | cpu | 1048576 | 104.559687 | 107.272974 | 0.046 |
| nvfp4 | dequant_fp32 | cpu | 1048576 | 13.683468 | 15.661115 | 0.350 |
| nvfp4 | quant_enumeration | resident_gpu | 4194304 | 1.144832 | 1.150976 | 16.716 |
| nvfp4 | quant_optimized | resident_gpu | 4194304 | 0.450944 | 0.470016 | 42.437 |
| nvfp4 | dequant_fp32 | resident_gpu | 4194304 | 0.088064 | 0.088864 | 217.302 |
| nvfp4 | dequant_fp16 | resident_gpu | 4194304 | 0.078848 | 0.080896 | 136.312 |
| nvfp4 | dequant_bf16 | resident_gpu | 4194304 | 0.086032 | 0.087040 | 124.929 |
| nvfp4 | quant_optimized | host_api | 4194304 | 6.129750 | 6.910311 | 3.122 |
| nvfp4 | quant_enumeration | resident_gpu | 16777216 | 4.514304 | 4.890624 | 16.956 |
| nvfp4 | quant_optimized | resident_gpu | 16777216 | 1.742800 | 2.141184 | 43.921 |
| nvfp4 | dequant_fp32 | resident_gpu | 16777216 | 0.339824 | 0.344928 | 225.252 |
| nvfp4 | dequant_fp16 | resident_gpu | 16777216 | 0.302080 | 0.305152 | 142.319 |
| nvfp4 | dequant_bf16 | resident_gpu | 16777216 | 0.332800 | 0.337920 | 129.182 |
| nvfp4 | quant_optimized | host_api | 16777216 | 21.812065 | 22.643258 | 3.509 |

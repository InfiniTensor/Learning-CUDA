# 性能测试结果

GPU 驻留计时使用 CUDA event；host_api 包括量化的分配、传输和释放。
基线使用逐小块 shared scale 归约与枚举编码；优化版使用多组 warp 归约和快速编码。两者共用全局两级归约与显存；不是原始 CLI 的完整实现对比。
中位数采用偶数样本中间两值均值，P95 使用 nearest-rank。CPU 仅测试 1M，预热一次后计时三次。

| 格式 | 操作 | 范围 | 元素数 | median ms | P95 ms | 逻辑 GB/s |
|---|---|---|---:|---:|---:|---:|
| mxfp8 | quant_enumeration | resident_gpu | 1048576 | 0.567296 | 0.569344 | 9.300 |
| mxfp8 | quant_optimized | resident_gpu | 1048576 | 0.123904 | 0.131072 | 42.579 |
| mxfp8 | dequant_fp32 | resident_gpu | 1048576 | 0.031744 | 0.031744 | 166.194 |
| mxfp8 | dequant_fp16 | resident_gpu | 1048576 | 0.025600 | 0.025600 | 124.160 |
| mxfp8 | dequant_bf16 | resident_gpu | 1048576 | 0.027648 | 0.027648 | 114.963 |
| mxfp8 | quant_optimized | host_api | 1048576 | 3.026753 | 3.928085 | 1.743 |
| mxfp8 | quant_reference | cpu | 1048576 | 1467.721210 | 1550.168343 | 0.004 |
| mxfp8 | dequant_fp32 | cpu | 1048576 | 15.567961 | 18.878772 | 0.339 |
| mxfp8 | quant_enumeration | resident_gpu | 4194304 | 2.236416 | 2.280448 | 9.436 |
| mxfp8 | quant_optimized | resident_gpu | 4194304 | 0.468992 | 0.472992 | 44.996 |
| mxfp8 | dequant_fp32 | resident_gpu | 4194304 | 0.129536 | 0.167904 | 162.909 |
| mxfp8 | dequant_fp16 | resident_gpu | 4194304 | 0.108544 | 0.121728 | 117.132 |
| mxfp8 | dequant_bf16 | resident_gpu | 4194304 | 0.108544 | 0.121856 | 117.132 |
| mxfp8 | quant_optimized | host_api | 4194304 | 7.071962 | 7.527101 | 2.984 |
| mxfp8 | quant_enumeration | resident_gpu | 16777216 | 7.932368 | 8.225792 | 10.641 |
| mxfp8 | quant_optimized | resident_gpu | 16777216 | 1.650528 | 1.969152 | 51.141 |
| mxfp8 | dequant_fp32 | resident_gpu | 16777216 | 0.403456 | 0.407552 | 209.218 |
| mxfp8 | dequant_fp16 | resident_gpu | 16777216 | 0.362496 | 0.366592 | 140.294 |
| mxfp8 | dequant_bf16 | resident_gpu | 16777216 | 0.379248 | 0.380640 | 134.097 |
| mxfp8 | quant_optimized | host_api | 16777216 | 24.302298 | 25.962615 | 3.473 |
| nvfp4 | quant_enumeration | resident_gpu | 1048576 | 0.316416 | 0.320512 | 15.120 |
| nvfp4 | quant_optimized | resident_gpu | 1048576 | 0.142336 | 0.146432 | 33.612 |
| nvfp4 | dequant_fp32 | resident_gpu | 1048576 | 0.025600 | 0.033792 | 186.880 |
| nvfp4 | dequant_fp16 | resident_gpu | 1048576 | 0.020000 | 0.061152 | 134.349 |
| nvfp4 | dequant_bf16 | resident_gpu | 1048576 | 0.022528 | 0.023488 | 119.273 |
| nvfp4 | quant_optimized | host_api | 1048576 | 2.201631 | 2.585676 | 2.173 |
| nvfp4 | quant_reference | cpu | 1048576 | 99.081328 | 102.703459 | 0.048 |
| nvfp4 | dequant_fp32 | cpu | 1048576 | 14.457638 | 15.911876 | 0.331 |
| nvfp4 | quant_enumeration | resident_gpu | 4194304 | 1.145216 | 1.150976 | 16.710 |
| nvfp4 | quant_optimized | resident_gpu | 4194304 | 0.451584 | 0.460832 | 42.376 |
| nvfp4 | dequant_fp32 | resident_gpu | 4194304 | 0.088064 | 0.093056 | 217.302 |
| nvfp4 | dequant_fp16 | resident_gpu | 4194304 | 0.079728 | 0.087968 | 134.807 |
| nvfp4 | dequant_bf16 | resident_gpu | 4194304 | 0.087040 | 0.087840 | 123.482 |
| nvfp4 | quant_optimized | host_api | 4194304 | 6.288401 | 6.858841 | 3.043 |
| nvfp4 | quant_enumeration | resident_gpu | 16777216 | 4.520960 | 4.765696 | 16.931 |
| nvfp4 | quant_optimized | resident_gpu | 16777216 | 1.746432 | 1.753088 | 43.830 |
| nvfp4 | dequant_fp32 | resident_gpu | 16777216 | 0.339920 | 0.343040 | 225.188 |
| nvfp4 | dequant_fp16 | resident_gpu | 16777216 | 0.303008 | 0.307200 | 141.883 |
| nvfp4 | dequant_bf16 | resident_gpu | 16777216 | 0.335232 | 0.344064 | 128.244 |
| nvfp4 | quant_optimized | host_api | 16777216 | 22.733686 | 24.075311 | 3.367 |

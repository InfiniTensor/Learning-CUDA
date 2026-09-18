# 性能测试结果

GPU 驻留计时使用 CUDA event；host_api 包括量化的分配、传输和释放。
host_api 中 quant_reused_workspace 复用显存和 event，不计首次创建/最终销毁；仍包含上传、量化、下载及主机结果分配/释放，与 quant_optimized 交替测量。此项衡量同进程重复调用，不代表单次 CLI 自动获得相同收益。
内部对照保留 shared scale 归约与枚举元素编码；两种格式的默认 block+nearest 均使用 scale/编码融合 kernel，NVFP4 另含两级全局归约。内部对照的 scale 也可能共享优化，优化前后请比较留存的 quant_optimized 记录。
中位数采用偶数样本中间两值均值，P95 使用 nearest-rank。CPU 仅测试 1M，预热一次后计时三次。

| 格式 | 操作 | 范围 | 元素数 | median ms | P95 ms | 逻辑 GB/s |
|---|---|---|---:|---:|---:|---:|
| mxfp8 | quant_enumeration | resident_gpu | 1048576 | 0.568320 | 0.577568 | 9.283 |
| mxfp8 | quant_optimized | resident_gpu | 1048576 | 0.021504 | 0.027648 | 245.333 |
| mxfp8 | dequant_fp32 | resident_gpu | 1048576 | 0.031728 | 0.035840 | 166.277 |
| mxfp8 | dequant_fp16 | resident_gpu | 1048576 | 0.024928 | 0.029568 | 127.507 |
| mxfp8 | dequant_bf16 | resident_gpu | 1048576 | 0.027648 | 0.031520 | 114.963 |
| mxfp8 | quant_optimized | host_api | 1048576 | 1.870341 | 2.381972 | 2.821 |
| mxfp8 | quant_reused_workspace | host_api | 1048576 | 0.805730 | 1.044301 | 6.548 |
| mxfp8 | quant_reference | cpu | 1048576 | 1274.090945 | 1275.156572 | 0.004 |
| mxfp8 | dequant_fp32 | cpu | 1048576 | 14.275321 | 14.713824 | 0.370 |
| mxfp8 | quant_enumeration | resident_gpu | 4194304 | 2.237952 | 2.699168 | 9.429 |
| mxfp8 | quant_optimized | resident_gpu | 4194304 | 0.070656 | 0.077536 | 298.667 |
| mxfp8 | dequant_fp32 | resident_gpu | 4194304 | 0.111008 | 0.112640 | 190.100 |
| mxfp8 | dequant_fp16 | resident_gpu | 4194304 | 0.104288 | 0.104448 | 121.912 |
| mxfp8 | dequant_bf16 | resident_gpu | 4194304 | 0.109568 | 0.113504 | 116.037 |
| mxfp8 | quant_optimized | host_api | 4194304 | 4.878609 | 5.138800 | 4.326 |
| mxfp8 | quant_reused_workspace | host_api | 4194304 | 2.726317 | 3.284191 | 7.740 |
| mxfp8 | quant_enumeration | resident_gpu | 16777216 | 8.946176 | 9.804800 | 9.435 |
| mxfp8 | quant_optimized | resident_gpu | 16777216 | 0.267728 | 0.300032 | 315.284 |
| mxfp8 | dequant_fp32 | resident_gpu | 16777216 | 0.403456 | 0.408576 | 209.218 |
| mxfp8 | dequant_fp16 | resident_gpu | 16777216 | 0.363520 | 0.366592 | 139.899 |
| mxfp8 | dequant_bf16 | resident_gpu | 16777216 | 0.384000 | 0.443392 | 132.437 |
| mxfp8 | quant_optimized | host_api | 16777216 | 16.600079 | 18.977148 | 5.085 |
| mxfp8 | quant_reused_workspace | host_api | 16777216 | 11.079116 | 15.232327 | 7.619 |
| nvfp4 | quant_enumeration | resident_gpu | 1048576 | 0.179200 | 0.184320 | 26.697 |
| nvfp4 | quant_optimized | resident_gpu | 1048576 | 0.041344 | 0.043008 | 115.715 |
| nvfp4 | dequant_fp32 | resident_gpu | 1048576 | 0.024576 | 0.025600 | 194.667 |
| nvfp4 | dequant_fp16 | resident_gpu | 1048576 | 0.020352 | 0.020480 | 132.025 |
| nvfp4 | dequant_bf16 | resident_gpu | 1048576 | 0.022528 | 0.034560 | 119.273 |
| nvfp4 | quant_optimized | host_api | 1048576 | 1.614376 | 1.887988 | 2.963 |
| nvfp4 | quant_reused_workspace | host_api | 1048576 | 0.727201 | 1.021837 | 6.579 |
| nvfp4 | quant_reference | cpu | 1048576 | 90.823209 | 90.967171 | 0.053 |
| nvfp4 | dequant_fp32 | cpu | 1048576 | 12.555596 | 12.779901 | 0.381 |
| nvfp4 | quant_enumeration | resident_gpu | 4194304 | 0.721936 | 0.953184 | 26.507 |
| nvfp4 | quant_optimized | resident_gpu | 4194304 | 0.140224 | 0.156672 | 136.471 |
| nvfp4 | dequant_fp32 | resident_gpu | 4194304 | 0.088544 | 0.089088 | 216.124 |
| nvfp4 | dequant_fp16 | resident_gpu | 4194304 | 0.079792 | 0.090112 | 134.699 |
| nvfp4 | dequant_bf16 | resident_gpu | 4194304 | 0.087040 | 0.088064 | 123.482 |
| nvfp4 | quant_optimized | host_api | 4194304 | 4.464255 | 5.004990 | 4.287 |
| nvfp4 | quant_reused_workspace | host_api | 4194304 | 2.671873 | 3.602070 | 7.162 |
| nvfp4 | quant_enumeration | resident_gpu | 16777216 | 2.842608 | 3.160992 | 26.928 |
| nvfp4 | quant_optimized | resident_gpu | 16777216 | 0.518656 | 0.553952 | 147.585 |
| nvfp4 | dequant_fp32 | resident_gpu | 16777216 | 0.339952 | 0.343040 | 225.167 |
| nvfp4 | dequant_fp16 | resident_gpu | 16777216 | 0.303104 | 0.305152 | 141.838 |
| nvfp4 | dequant_bf16 | resident_gpu | 16777216 | 0.333824 | 0.334848 | 128.785 |
| nvfp4 | quant_optimized | host_api | 16777216 | 14.626202 | 17.495761 | 5.233 |
| nvfp4 | quant_reused_workspace | host_api | 16777216 | 12.121707 | 13.584647 | 6.315 |

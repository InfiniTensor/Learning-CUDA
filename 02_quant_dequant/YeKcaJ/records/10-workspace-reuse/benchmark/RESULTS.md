# 性能测试结果

GPU 驻留计时使用 CUDA event；host_api 包括量化的分配、传输和释放。
host_api 中 quant_reused_workspace 复用显存和 event，不计首次创建/最终销毁；仍包含上传、量化、下载及主机结果分配/释放，与 quant_optimized 交替测量。此项衡量同进程重复调用，不代表单次 CLI 自动获得相同收益。
内部对照保留 shared scale 归约与枚举元素编码；两种格式的默认 block+nearest 均使用 scale/编码融合 kernel，NVFP4 另含两级全局归约。内部对照的 scale 也可能共享优化，优化前后请比较留存的 quant_optimized 记录。
中位数采用偶数样本中间两值均值，P95 使用 nearest-rank。CPU 仅测试 1M，预热一次后计时三次。

| 格式 | 操作 | 范围 | 元素数 | median ms | P95 ms | 逻辑 GB/s |
|---|---|---|---:|---:|---:|---:|
| mxfp8 | quant_enumeration | resident_gpu | 1048576 | 0.568832 | 0.574464 | 9.275 |
| mxfp8 | quant_optimized | resident_gpu | 1048576 | 0.024576 | 0.025600 | 214.667 |
| mxfp8 | dequant_fp32 | resident_gpu | 1048576 | 0.034816 | 0.034816 | 151.529 |
| mxfp8 | dequant_fp16 | resident_gpu | 1048576 | 0.028672 | 0.030528 | 110.857 |
| mxfp8 | dequant_bf16 | resident_gpu | 1048576 | 0.030720 | 0.031552 | 103.467 |
| mxfp8 | quant_optimized | host_api | 1048576 | 2.171398 | 3.004206 | 2.430 |
| mxfp8 | quant_reused_workspace | host_api | 1048576 | 1.037373 | 1.737259 | 5.086 |
| mxfp8 | quant_reference | cpu | 1048576 | 1304.556152 | 1341.786778 | 0.004 |
| mxfp8 | dequant_fp32 | cpu | 1048576 | 14.337760 | 14.898560 | 0.368 |
| mxfp8 | quant_enumeration | resident_gpu | 4194304 | 2.236416 | 3.311616 | 9.436 |
| mxfp8 | quant_optimized | resident_gpu | 4194304 | 0.072192 | 0.078720 | 292.312 |
| mxfp8 | dequant_fp32 | resident_gpu | 4194304 | 0.113024 | 0.119808 | 186.709 |
| mxfp8 | dequant_fp16 | resident_gpu | 4194304 | 0.103936 | 0.114688 | 122.325 |
| mxfp8 | dequant_bf16 | resident_gpu | 4194304 | 0.110592 | 0.113664 | 114.963 |
| mxfp8 | quant_optimized | host_api | 4194304 | 4.777892 | 5.551659 | 4.417 |
| mxfp8 | quant_reused_workspace | host_api | 4194304 | 2.792794 | 3.193150 | 7.556 |
| mxfp8 | quant_enumeration | resident_gpu | 16777216 | 8.716800 | 9.269248 | 9.684 |
| mxfp8 | quant_optimized | resident_gpu | 16777216 | 0.272384 | 0.301056 | 309.895 |
| mxfp8 | dequant_fp32 | resident_gpu | 16777216 | 0.407040 | 0.412512 | 207.376 |
| mxfp8 | dequant_fp16 | resident_gpu | 16777216 | 0.366624 | 0.370688 | 138.714 |
| mxfp8 | dequant_bf16 | resident_gpu | 16777216 | 0.382976 | 0.403456 | 132.791 |
| mxfp8 | quant_optimized | host_api | 16777216 | 16.660836 | 17.843077 | 5.066 |
| mxfp8 | quant_reused_workspace | host_api | 16777216 | 11.450560 | 14.257691 | 7.372 |
| nvfp4 | quant_enumeration | resident_gpu | 1048576 | 0.199088 | 0.214016 | 24.030 |
| nvfp4 | quant_optimized | resident_gpu | 1048576 | 0.051200 | 0.066464 | 93.440 |
| nvfp4 | dequant_fp32 | resident_gpu | 1048576 | 0.028672 | 0.029472 | 166.857 |
| nvfp4 | dequant_fp16 | resident_gpu | 1048576 | 0.022512 | 0.023264 | 119.358 |
| nvfp4 | dequant_bf16 | resident_gpu | 1048576 | 0.025568 | 0.026368 | 105.092 |
| nvfp4 | quant_optimized | host_api | 1048576 | 1.746878 | 2.255017 | 2.739 |
| nvfp4 | quant_reused_workspace | host_api | 1048576 | 0.963798 | 1.332427 | 4.964 |
| nvfp4 | quant_reference | cpu | 1048576 | 92.750940 | 93.309709 | 0.052 |
| nvfp4 | dequant_fp32 | cpu | 1048576 | 11.915275 | 14.047686 | 0.402 |
| nvfp4 | quant_enumeration | resident_gpu | 4194304 | 0.726880 | 0.959488 | 26.327 |
| nvfp4 | quant_optimized | resident_gpu | 4194304 | 0.142832 | 0.147456 | 133.979 |
| nvfp4 | dequant_fp32 | resident_gpu | 4194304 | 0.091136 | 0.093152 | 209.978 |
| nvfp4 | dequant_fp16 | resident_gpu | 4194304 | 0.081920 | 0.083904 | 131.200 |
| nvfp4 | dequant_bf16 | resident_gpu | 4194304 | 0.090112 | 0.091136 | 119.273 |
| nvfp4 | quant_optimized | host_api | 4194304 | 4.598268 | 6.197192 | 4.162 |
| nvfp4 | quant_reused_workspace | host_api | 4194304 | 2.726448 | 3.332351 | 7.019 |
| nvfp4 | quant_enumeration | resident_gpu | 16777216 | 2.852864 | 3.653632 | 26.831 |
| nvfp4 | quant_optimized | resident_gpu | 16777216 | 0.522096 | 0.562240 | 146.613 |
| nvfp4 | dequant_fp32 | resident_gpu | 16777216 | 0.342880 | 0.346784 | 223.244 |
| nvfp4 | dequant_fp16 | resident_gpu | 16777216 | 0.305952 | 0.307968 | 140.518 |
| nvfp4 | dequant_bf16 | resident_gpu | 16777216 | 0.336944 | 0.341760 | 127.593 |
| nvfp4 | quant_optimized | host_api | 16777216 | 14.593807 | 16.115012 | 5.245 |
| nvfp4 | quant_reused_workspace | host_api | 16777216 | 9.878531 | 12.850389 | 7.749 |

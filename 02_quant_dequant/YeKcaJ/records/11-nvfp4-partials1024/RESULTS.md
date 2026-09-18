# 性能测试结果

GPU 驻留计时使用 CUDA event；host_api 包括量化的分配、传输和释放。
host_api 中 quant_reused_workspace 复用显存和 event，不计首次创建/最终销毁；仍包含上传、量化、下载及主机结果分配/释放，与 quant_optimized 交替测量。此项衡量同进程重复调用，不代表单次 CLI 自动获得相同收益。
内部对照保留 shared scale 归约与枚举元素编码；两种格式的默认 block+nearest 均使用 scale/编码融合 kernel，NVFP4 另含两级全局归约。内部对照的 scale 也可能共享优化，优化前后请比较留存的 quant_optimized 记录。
中位数采用偶数样本中间两值均值，P95 使用 nearest-rank。CPU 仅测试 1M，预热一次后计时三次。

| 格式 | 操作 | 范围 | 元素数 | median ms | P95 ms | 逻辑 GB/s |
|---|---|---|---:|---:|---:|---:|
| mxfp8 | quant_enumeration | resident_gpu | 1048576 | 0.566272 | 0.945152 | 9.316 |
| mxfp8 | quant_optimized | resident_gpu | 1048576 | 0.021504 | 0.025600 | 245.333 |
| mxfp8 | dequant_fp32 | resident_gpu | 1048576 | 0.031744 | 0.035776 | 166.194 |
| mxfp8 | dequant_fp16 | resident_gpu | 1048576 | 0.026592 | 0.026624 | 119.528 |
| mxfp8 | dequant_bf16 | resident_gpu | 1048576 | 0.028432 | 0.028672 | 111.793 |
| mxfp8 | quant_optimized | host_api | 1048576 | 1.730113 | 2.138125 | 3.049 |
| mxfp8 | quant_reused_workspace | host_api | 1048576 | 0.732415 | 0.951016 | 7.203 |
| mxfp8 | quant_reference | cpu | 1048576 | 1308.790197 | 1330.536107 | 0.004 |
| mxfp8 | dequant_fp32 | cpu | 1048576 | 13.703281 | 14.810404 | 0.385 |
| mxfp8 | quant_enumeration | resident_gpu | 4194304 | 2.239872 | 2.999296 | 9.421 |
| mxfp8 | quant_optimized | resident_gpu | 4194304 | 0.072704 | 0.075776 | 290.254 |
| mxfp8 | dequant_fp32 | resident_gpu | 4194304 | 0.111456 | 0.115712 | 189.336 |
| mxfp8 | dequant_fp16 | resident_gpu | 4194304 | 0.106496 | 0.107520 | 119.385 |
| mxfp8 | dequant_bf16 | resident_gpu | 4194304 | 0.112608 | 0.114400 | 112.905 |
| mxfp8 | quant_optimized | host_api | 4194304 | 4.773344 | 6.179053 | 4.421 |
| mxfp8 | quant_reused_workspace | host_api | 4194304 | 2.935726 | 3.783344 | 7.188 |
| mxfp8 | quant_enumeration | resident_gpu | 16777216 | 8.510864 | 9.372416 | 9.918 |
| mxfp8 | quant_optimized | resident_gpu | 16777216 | 0.271872 | 0.276480 | 310.478 |
| mxfp8 | dequant_fp32 | resident_gpu | 16777216 | 0.410528 | 0.415488 | 205.614 |
| mxfp8 | dequant_fp16 | resident_gpu | 16777216 | 0.366608 | 0.373760 | 138.720 |
| mxfp8 | dequant_bf16 | resident_gpu | 16777216 | 0.380848 | 0.384832 | 133.533 |
| mxfp8 | quant_optimized | host_api | 16777216 | 18.680256 | 22.460208 | 4.519 |
| mxfp8 | quant_reused_workspace | host_api | 16777216 | 11.932201 | 14.553150 | 7.074 |
| nvfp4 | quant_enumeration | resident_gpu | 1048576 | 0.176128 | 0.180224 | 27.163 |
| nvfp4 | quant_optimized | resident_gpu | 1048576 | 0.041952 | 0.047104 | 114.038 |
| nvfp4 | dequant_fp32 | resident_gpu | 1048576 | 0.025600 | 0.025600 | 186.880 |
| nvfp4 | dequant_fp16 | resident_gpu | 1048576 | 0.020352 | 0.020480 | 132.025 |
| nvfp4 | dequant_bf16 | resident_gpu | 1048576 | 0.022528 | 0.026624 | 119.273 |
| nvfp4 | quant_optimized | host_api | 1048576 | 1.728344 | 2.253737 | 2.768 |
| nvfp4 | quant_reused_workspace | host_api | 1048576 | 0.771278 | 0.975629 | 6.203 |
| nvfp4 | quant_reference | cpu | 1048576 | 93.669478 | 100.641025 | 0.051 |
| nvfp4 | dequant_fp32 | cpu | 1048576 | 12.395709 | 12.844850 | 0.386 |
| nvfp4 | quant_enumeration | resident_gpu | 4194304 | 0.727040 | 0.942816 | 26.321 |
| nvfp4 | quant_optimized | resident_gpu | 4194304 | 0.144368 | 0.153600 | 132.554 |
| nvfp4 | dequant_fp32 | resident_gpu | 4194304 | 0.089600 | 0.180448 | 213.577 |
| nvfp4 | dequant_fp16 | resident_gpu | 4194304 | 0.079872 | 0.090112 | 134.564 |
| nvfp4 | dequant_bf16 | resident_gpu | 4194304 | 0.092000 | 0.097280 | 116.825 |
| nvfp4 | quant_optimized | host_api | 4194304 | 5.670380 | 6.484118 | 3.375 |
| nvfp4 | quant_reused_workspace | host_api | 4194304 | 3.349003 | 4.545353 | 5.714 |
| nvfp4 | quant_enumeration | resident_gpu | 16777216 | 2.840064 | 3.186528 | 26.952 |
| nvfp4 | quant_optimized | resident_gpu | 16777216 | 0.519952 | 0.787456 | 147.218 |
| nvfp4 | dequant_fp32 | resident_gpu | 16777216 | 0.339952 | 0.341824 | 225.167 |
| nvfp4 | dequant_fp16 | resident_gpu | 16777216 | 0.304640 | 0.320448 | 141.123 |
| nvfp4 | dequant_bf16 | resident_gpu | 16777216 | 0.333824 | 0.339872 | 128.785 |
| nvfp4 | quant_optimized | host_api | 16777216 | 14.968088 | 17.883021 | 5.114 |
| nvfp4 | quant_reused_workspace | host_api | 16777216 | 9.712428 | 11.560076 | 7.881 |

# 性能测试结果

GPU 驻留计时使用 CUDA event；host_api 包括量化的分配、传输和释放。
host_api 中 quant_reused_workspace 复用显存和 event，不计首次创建/最终销毁；仍包含上传、量化、下载及主机结果分配/释放，与 quant_optimized 交替测量。此项衡量同进程重复调用，不代表单次 CLI 自动获得相同收益。
内部对照保留 shared scale 归约与枚举元素编码；两种格式的默认 block+nearest 均使用 scale/编码融合 kernel，NVFP4 另含两级全局归约。内部对照的 scale 也可能共享优化，优化前后请比较留存的 quant_optimized 记录。
中位数采用偶数样本中间两值均值，P95 使用 nearest-rank。CPU 仅测试 1M，预热一次后计时三次。

| 格式 | 操作 | 范围 | 元素数 | median ms | P95 ms | 逻辑 GB/s |
|---|---|---|---:|---:|---:|---:|
| mxfp8 | quant_enumeration | resident_gpu | 1048576 | 0.568704 | 0.932864 | 9.277 |
| mxfp8 | quant_optimized | resident_gpu | 1048576 | 0.024448 | 0.025408 | 215.791 |
| mxfp8 | dequant_fp32 | resident_gpu | 1048576 | 0.049152 | 0.054272 | 107.333 |
| mxfp8 | dequant_fp16 | resident_gpu | 1048576 | 0.041984 | 0.045056 | 75.707 |
| mxfp8 | dequant_bf16 | resident_gpu | 1048576 | 0.045056 | 0.060352 | 70.545 |
| mxfp8 | quant_optimized | host_api | 1048576 | 1.784800 | 2.012705 | 2.956 |
| mxfp8 | quant_reused_workspace | host_api | 1048576 | 0.721718 | 1.107199 | 7.310 |
| mxfp8 | quant_reference | cpu | 1048576 | 1299.364942 | 1334.393881 | 0.004 |
| mxfp8 | dequant_fp32 | cpu | 1048576 | 14.253641 | 15.124957 | 0.370 |
| mxfp8 | quant_enumeration | resident_gpu | 4194304 | 2.237856 | 3.457024 | 9.430 |
| mxfp8 | quant_optimized | resident_gpu | 4194304 | 0.073216 | 0.078752 | 288.224 |
| mxfp8 | dequant_fp32 | resident_gpu | 4194304 | 0.113056 | 0.119808 | 186.656 |
| mxfp8 | dequant_fp16 | resident_gpu | 4194304 | 0.105984 | 0.107520 | 119.961 |
| mxfp8 | dequant_bf16 | resident_gpu | 4194304 | 0.109568 | 0.112640 | 116.037 |
| mxfp8 | quant_optimized | host_api | 4194304 | 4.901184 | 5.528165 | 4.306 |
| mxfp8 | quant_reused_workspace | host_api | 4194304 | 2.858666 | 3.479562 | 7.382 |
| mxfp8 | quant_enumeration | resident_gpu | 16777216 | 8.656896 | 9.396224 | 9.751 |
| mxfp8 | quant_optimized | resident_gpu | 16777216 | 0.271360 | 0.277504 | 311.064 |
| mxfp8 | dequant_fp32 | resident_gpu | 16777216 | 0.406528 | 0.452608 | 207.637 |
| mxfp8 | dequant_fp16 | resident_gpu | 16777216 | 0.367504 | 0.369376 | 138.382 |
| mxfp8 | dequant_bf16 | resident_gpu | 16777216 | 0.383808 | 0.386048 | 132.504 |
| mxfp8 | quant_optimized | host_api | 16777216 | 15.552419 | 16.578942 | 5.427 |
| mxfp8 | quant_reused_workspace | host_api | 16777216 | 10.545030 | 11.091512 | 8.005 |
| nvfp4 | quant_enumeration | resident_gpu | 1048576 | 0.198656 | 0.203776 | 24.082 |
| nvfp4 | quant_optimized | resident_gpu | 1048576 | 0.051200 | 0.052224 | 93.440 |
| nvfp4 | dequant_fp32 | resident_gpu | 1048576 | 0.028672 | 0.029696 | 166.857 |
| nvfp4 | dequant_fp16 | resident_gpu | 1048576 | 0.022528 | 0.023456 | 119.273 |
| nvfp4 | dequant_bf16 | resident_gpu | 1048576 | 0.025600 | 0.025600 | 104.960 |
| nvfp4 | quant_optimized | host_api | 1048576 | 1.569210 | 1.976124 | 3.049 |
| nvfp4 | quant_reused_workspace | host_api | 1048576 | 0.696192 | 0.887609 | 6.872 |
| nvfp4 | quant_reference | cpu | 1048576 | 86.841097 | 88.208038 | 0.055 |
| nvfp4 | dequant_fp32 | cpu | 1048576 | 11.842162 | 12.377686 | 0.404 |
| nvfp4 | quant_enumeration | resident_gpu | 4194304 | 0.726032 | 0.731136 | 26.358 |
| nvfp4 | quant_optimized | resident_gpu | 4194304 | 0.143360 | 0.145408 | 133.486 |
| nvfp4 | dequant_fp32 | resident_gpu | 4194304 | 0.092160 | 0.103424 | 207.644 |
| nvfp4 | dequant_fp16 | resident_gpu | 4194304 | 0.081920 | 0.082944 | 131.200 |
| nvfp4 | dequant_bf16 | resident_gpu | 4194304 | 0.090112 | 0.091136 | 119.273 |
| nvfp4 | quant_optimized | host_api | 4194304 | 4.527283 | 5.648417 | 4.227 |
| nvfp4 | quant_reused_workspace | host_api | 4194304 | 2.657489 | 3.340039 | 7.201 |
| nvfp4 | quant_enumeration | resident_gpu | 16777216 | 2.844672 | 3.845120 | 26.909 |
| nvfp4 | quant_optimized | resident_gpu | 16777216 | 0.522752 | 0.532480 | 146.429 |
| nvfp4 | dequant_fp32 | resident_gpu | 16777216 | 0.343552 | 0.347136 | 222.808 |
| nvfp4 | dequant_fp16 | resident_gpu | 16777216 | 0.307120 | 0.310272 | 139.983 |
| nvfp4 | dequant_bf16 | resident_gpu | 16777216 | 0.338944 | 0.342912 | 126.840 |
| nvfp4 | quant_optimized | host_api | 16777216 | 15.212290 | 17.087802 | 5.032 |
| nvfp4 | quant_reused_workspace | host_api | 16777216 | 10.305515 | 13.555922 | 7.428 |

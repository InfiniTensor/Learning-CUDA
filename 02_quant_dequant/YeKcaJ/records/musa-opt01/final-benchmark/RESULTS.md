# 性能测试结果

后端：MUSA。不同平台的内部对照路径和编译器可能不同，性能不能直接归因于硬件。
GPU 驻留计时使用 MUSA event；host_api 包括量化的分配、传输和释放。
host_api 中 quant_reused_workspace 复用显存和 event，不计首次创建/最终销毁；仍包含上传、量化、下载及主机结果分配/释放，与 quant_optimized 交替测量。此项衡量同进程重复调用，不代表单次 CLI 自动获得相同收益。
内部对照保留 shared scale 归约与枚举元素编码；两种格式的默认 block+nearest 均使用 scale/编码融合 kernel，NVFP4 另含两级全局归约。内部对照的 scale 也可能共享优化，优化前后请比较留存的 quant_optimized 记录。
中位数采用偶数样本中间两值均值，P95 使用 nearest-rank。CPU 仅测试 1M，预热一次后计时三次。

| 格式 | 操作 | 范围 | 元素数 | median ms | P95 ms | 逻辑 GB/s |
|---|---|---|---:|---:|---:|---:|
| mxfp8 | quant_enumeration | resident_gpu | 1048576 | 3.488343 | 3.489029 | 1.512 |
| mxfp8 | quant_optimized | resident_gpu | 1048576 | 0.505040 | 0.505349 | 10.446 |
| mxfp8 | dequant_fp32 | resident_gpu | 1048576 | 0.081600 | 0.082743 | 64.653 |
| mxfp8 | dequant_fp16 | resident_gpu | 1048576 | 0.081177 | 0.081691 | 39.155 |
| mxfp8 | dequant_bf16 | resident_gpu | 1048576 | 0.080194 | 0.081486 | 39.635 |
| mxfp8 | quant_optimized | host_api | 1048576 | 1.793011 | 1.802663 | 2.942 |
| mxfp8 | quant_reused_workspace | host_api | 1048576 | 1.744496 | 1.758384 | 3.024 |
| mxfp8 | quant_reference | cpu | 1048576 | 1924.480578 | 1960.819773 | 0.003 |
| mxfp8 | dequant_fp32 | cpu | 1048576 | 16.638046 | 17.048141 | 0.317 |
| mxfp8 | quant_enumeration | resident_gpu | 4194304 | 13.725646 | 13.726628 | 1.537 |
| mxfp8 | quant_optimized | resident_gpu | 4194304 | 1.624251 | 1.624937 | 12.992 |
| mxfp8 | dequant_fp32 | resident_gpu | 4194304 | 0.143691 | 0.144137 | 146.860 |
| mxfp8 | dequant_fp16 | resident_gpu | 4194304 | 0.142583 | 0.144389 | 89.169 |
| mxfp8 | dequant_bf16 | resident_gpu | 4194304 | 0.142469 | 0.142811 | 89.241 |
| mxfp8 | quant_optimized | host_api | 4194304 | 4.750016 | 4.766436 | 4.443 |
| mxfp8 | quant_reused_workspace | host_api | 4194304 | 4.642449 | 4.666271 | 4.546 |
| mxfp8 | quant_enumeration | resident_gpu | 16777216 | 54.604103 | 54.609188 | 1.546 |
| mxfp8 | quant_optimized | resident_gpu | 16777216 | 6.327497 | 6.332503 | 13.340 |
| mxfp8 | dequant_fp32 | resident_gpu | 16777216 | 0.472983 | 0.473897 | 178.464 |
| mxfp8 | dequant_fp16 | resident_gpu | 16777216 | 0.464023 | 0.465943 | 109.598 |
| mxfp8 | dequant_bf16 | resident_gpu | 16777216 | 0.463851 | 0.465417 | 109.638 |
| mxfp8 | quant_optimized | host_api | 16777216 | 16.438681 | 16.561473 | 5.135 |
| mxfp8 | quant_reused_workspace | host_api | 16777216 | 16.230203 | 16.324407 | 5.201 |
| nvfp4 | quant_enumeration | resident_gpu | 1048576 | 2.184126 | 2.189006 | 2.190 |
| nvfp4 | quant_optimized | resident_gpu | 1048576 | 0.597931 | 0.603360 | 8.001 |
| nvfp4 | dequant_fp32 | resident_gpu | 1048576 | 0.093851 | 0.094606 | 50.976 |
| nvfp4 | dequant_fp16 | resident_gpu | 1048576 | 0.093211 | 0.093600 | 28.827 |
| nvfp4 | dequant_bf16 | resident_gpu | 1048576 | 0.093006 | 0.095794 | 28.890 |
| nvfp4 | quant_optimized | host_api | 1048576 | 1.794339 | 1.804731 | 2.666 |
| nvfp4 | quant_reused_workspace | host_api | 1048576 | 1.754341 | 1.762619 | 2.727 |
| nvfp4 | quant_reference | cpu | 1048576 | 134.970259 | 140.247909 | 0.035 |
| nvfp4 | dequant_fp32 | cpu | 1048576 | 15.148323 | 16.043286 | 0.316 |
| nvfp4 | quant_enumeration | resident_gpu | 4194304 | 8.307154 | 8.313395 | 2.304 |
| nvfp4 | quant_optimized | resident_gpu | 4194304 | 1.747714 | 1.751543 | 10.949 |
| nvfp4 | dequant_fp32 | resident_gpu | 4194304 | 0.222503 | 0.227269 | 86.006 |
| nvfp4 | dequant_fp16 | resident_gpu | 4194304 | 0.221337 | 0.225760 | 48.559 |
| nvfp4 | dequant_bf16 | resident_gpu | 4194304 | 0.221509 | 0.221920 | 48.521 |
| nvfp4 | quant_optimized | host_api | 4194304 | 5.014895 | 5.040526 | 3.816 |
| nvfp4 | quant_reused_workspace | host_api | 4194304 | 4.932152 | 4.963601 | 3.880 |
| nvfp4 | quant_enumeration | resident_gpu | 16777216 | 32.955658 | 32.957554 | 2.323 |
| nvfp4 | quant_optimized | resident_gpu | 16777216 | 6.718057 | 6.720297 | 11.394 |
| nvfp4 | dequant_fp32 | resident_gpu | 16777216 | 0.738651 | 0.739063 | 103.629 |
| nvfp4 | dequant_fp16 | resident_gpu | 16777216 | 0.739669 | 0.740937 | 58.123 |
| nvfp4 | dequant_bf16 | resident_gpu | 16777216 | 0.740811 | 0.745029 | 58.033 |
| nvfp4 | quant_optimized | host_api | 16777216 | 18.276919 | 18.351268 | 4.188 |
| nvfp4 | quant_reused_workspace | host_api | 16777216 | 18.039917 | 18.156196 | 4.243 |

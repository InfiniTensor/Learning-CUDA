# 性能测试结果

后端：MUSA。不同平台的内部对照路径和编译器可能不同，性能不能直接归因于硬件。
GPU 驻留计时使用 MUSA event；host_api 包括量化的分配、传输和释放。
host_api 中 quant_reused_workspace 复用显存和 event，不计首次创建/最终销毁；仍包含上传、量化、下载及主机结果分配/释放，与 quant_optimized 交替测量。此项衡量同进程重复调用，不代表单次 CLI 自动获得相同收益。
内部对照保留 shared scale 归约与枚举元素编码；两种格式的默认 block+nearest 均使用 scale/编码融合 kernel，NVFP4 另含两级全局归约。内部对照的 scale 也可能共享优化，优化前后请比较留存的 quant_optimized 记录。
中位数采用偶数样本中间两值均值，P95 使用 nearest-rank。CPU 仅测试 1M，预热一次后计时三次。

| 格式 | 操作 | 范围 | 元素数 | median ms | P95 ms | 逻辑 GB/s |
|---|---|---|---:|---:|---:|---:|
| mxfp8 | quant_enumeration | resident_gpu | 1048576 | 3.488023 | 3.488480 | 1.513 |
| mxfp8 | quant_optimized | resident_gpu | 1048576 | 0.670057 | 0.670263 | 7.873 |
| mxfp8 | dequant_fp32 | resident_gpu | 1048576 | 0.081017 | 0.081760 | 65.118 |
| mxfp8 | dequant_fp16 | resident_gpu | 1048576 | 0.079954 | 0.080297 | 39.754 |
| mxfp8 | dequant_bf16 | resident_gpu | 1048576 | 0.080000 | 0.080206 | 39.731 |
| mxfp8 | quant_optimized | host_api | 1048576 | 1.939668 | 1.957353 | 2.720 |
| mxfp8 | quant_reused_workspace | host_api | 1048576 | 1.894206 | 1.927646 | 2.785 |
| mxfp8 | quant_reference | cpu | 1048576 | 1916.467197 | 1917.259782 | 0.003 |
| mxfp8 | dequant_fp32 | cpu | 1048576 | 15.914737 | 16.973565 | 0.331 |
| mxfp8 | quant_enumeration | resident_gpu | 4194304 | 13.729828 | 13.730948 | 1.537 |
| mxfp8 | quant_optimized | resident_gpu | 4194304 | 2.539737 | 2.540480 | 8.309 |
| mxfp8 | dequant_fp32 | resident_gpu | 4194304 | 0.144446 | 0.149326 | 146.094 |
| mxfp8 | dequant_fp16 | resident_gpu | 4194304 | 0.142823 | 0.148046 | 89.019 |
| mxfp8 | dequant_bf16 | resident_gpu | 4194304 | 0.142720 | 0.147749 | 89.083 |
| mxfp8 | quant_optimized | host_api | 4194304 | 6.237722 | 6.290070 | 3.383 |
| mxfp8 | quant_reused_workspace | host_api | 4194304 | 6.124994 | 6.172137 | 3.445 |
| mxfp8 | quant_enumeration | resident_gpu | 16777216 | 54.603714 | 54.608253 | 1.546 |
| mxfp8 | quant_optimized | resident_gpu | 16777216 | 9.997166 | 10.001554 | 8.443 |
| mxfp8 | dequant_fp32 | resident_gpu | 16777216 | 0.473143 | 0.478423 | 178.404 |
| mxfp8 | dequant_fp16 | resident_gpu | 16777216 | 0.463954 | 0.468343 | 109.614 |
| mxfp8 | dequant_bf16 | resident_gpu | 16777216 | 0.464229 | 0.469120 | 109.549 |
| mxfp8 | quant_optimized | host_api | 16777216 | 21.210941 | 21.288470 | 3.980 |
| mxfp8 | quant_reused_workspace | host_api | 16777216 | 20.994718 | 21.066699 | 4.021 |
| nvfp4 | quant_enumeration | resident_gpu | 1048576 | 2.186880 | 2.189097 | 2.188 |
| nvfp4 | quant_optimized | resident_gpu | 1048576 | 0.619989 | 0.621189 | 7.716 |
| nvfp4 | dequant_fp32 | resident_gpu | 1048576 | 0.100914 | 0.102789 | 47.408 |
| nvfp4 | dequant_fp16 | resident_gpu | 1048576 | 0.099360 | 0.100869 | 27.043 |
| nvfp4 | dequant_bf16 | resident_gpu | 1048576 | 0.095451 | 0.100846 | 28.150 |
| nvfp4 | quant_optimized | host_api | 1048576 | 1.830013 | 1.840914 | 2.614 |
| nvfp4 | quant_reused_workspace | host_api | 1048576 | 1.790788 | 1.802466 | 2.672 |
| nvfp4 | quant_reference | cpu | 1048576 | 133.935264 | 134.509642 | 0.036 |
| nvfp4 | dequant_fp32 | cpu | 1048576 | 15.214665 | 15.896878 | 0.314 |
| nvfp4 | quant_enumeration | resident_gpu | 4194304 | 8.317154 | 8.321006 | 2.301 |
| nvfp4 | quant_optimized | resident_gpu | 4194304 | 2.089954 | 2.092251 | 9.156 |
| nvfp4 | dequant_fp32 | resident_gpu | 4194304 | 0.227554 | 0.228069 | 84.096 |
| nvfp4 | dequant_fp16 | resident_gpu | 4194304 | 0.226160 | 0.227040 | 47.523 |
| nvfp4 | dequant_bf16 | resident_gpu | 4194304 | 0.225966 | 0.227223 | 47.564 |
| nvfp4 | quant_optimized | host_api | 4194304 | 5.531990 | 5.551285 | 3.459 |
| nvfp4 | quant_reused_workspace | host_api | 4194304 | 5.455171 | 5.480432 | 3.508 |
| nvfp4 | quant_enumeration | resident_gpu | 16777216 | 32.955509 | 32.956802 | 2.323 |
| nvfp4 | quant_optimized | resident_gpu | 16777216 | 8.061703 | 8.064228 | 9.495 |
| nvfp4 | dequant_fp32 | resident_gpu | 16777216 | 0.739543 | 0.743931 | 103.505 |
| nvfp4 | dequant_fp16 | resident_gpu | 16777216 | 0.741817 | 0.746377 | 57.954 |
| nvfp4 | dequant_bf16 | resident_gpu | 16777216 | 0.740377 | 0.745646 | 58.067 |
| nvfp4 | quant_optimized | host_api | 16777216 | 18.774826 | 18.815519 | 4.077 |
| nvfp4 | quant_reused_workspace | host_api | 16777216 | 18.541258 | 18.600690 | 4.128 |

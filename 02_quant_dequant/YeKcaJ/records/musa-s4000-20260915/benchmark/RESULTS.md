# 性能测试结果

后端：MUSA。不同平台的内部对照路径和编译器可能不同，性能不能直接归因于硬件。
GPU 驻留计时使用 MUSA event；host_api 包括量化的分配、传输和释放。
host_api 中 quant_reused_workspace 复用显存和 event，不计首次创建/最终销毁；仍包含上传、量化、下载及主机结果分配/释放，与 quant_optimized 交替测量。此项衡量同进程重复调用，不代表单次 CLI 自动获得相同收益。
内部对照保留 shared scale 归约与枚举元素编码；两种格式的默认 block+nearest 均使用 scale/编码融合 kernel，NVFP4 另含两级全局归约。内部对照的 scale 也可能共享优化，优化前后请比较留存的 quant_optimized 记录。
中位数采用偶数样本中间两值均值，P95 使用 nearest-rank。CPU 仅测试 1M，预热一次后计时三次。

| 格式 | 操作 | 范围 | 元素数 | median ms | P95 ms | 逻辑 GB/s |
|---|---|---|---:|---:|---:|---:|
| mxfp8 | quant_enumeration | resident_gpu | 1048576 | 3.488823 | 3.493623 | 1.512 |
| mxfp8 | quant_optimized | resident_gpu | 1048576 | 0.681371 | 0.681989 | 7.743 |
| mxfp8 | dequant_fp32 | resident_gpu | 1048576 | 0.091429 | 0.092457 | 57.702 |
| mxfp8 | dequant_fp16 | resident_gpu | 1048576 | 0.089817 | 0.091634 | 35.389 |
| mxfp8 | dequant_bf16 | resident_gpu | 1048576 | 0.090297 | 0.091063 | 35.200 |
| mxfp8 | quant_optimized | host_api | 1048576 | 1.940393 | 1.954846 | 2.719 |
| mxfp8 | quant_reused_workspace | host_api | 1048576 | 1.895458 | 1.900689 | 2.783 |
| mxfp8 | quant_reference | cpu | 1048576 | 1917.615780 | 1919.490118 | 0.003 |
| mxfp8 | dequant_fp32 | cpu | 1048576 | 16.177046 | 17.263479 | 0.326 |
| mxfp8 | quant_enumeration | resident_gpu | 4194304 | 13.726971 | 13.728571 | 1.537 |
| mxfp8 | quant_optimized | resident_gpu | 4194304 | 2.560949 | 2.564411 | 8.240 |
| mxfp8 | dequant_fp32 | resident_gpu | 4194304 | 0.148331 | 0.149211 | 142.266 |
| mxfp8 | dequant_fp16 | resident_gpu | 4194304 | 0.146903 | 0.148091 | 86.547 |
| mxfp8 | dequant_bf16 | resident_gpu | 4194304 | 0.146777 | 0.147589 | 86.621 |
| mxfp8 | quant_optimized | host_api | 4194304 | 6.076304 | 6.095872 | 3.473 |
| mxfp8 | quant_reused_workspace | host_api | 4194304 | 5.978815 | 5.994205 | 3.530 |
| mxfp8 | quant_enumeration | resident_gpu | 16777216 | 54.603313 | 54.607750 | 1.546 |
| mxfp8 | quant_optimized | resident_gpu | 16777216 | 10.092674 | 10.096823 | 8.364 |
| mxfp8 | dequant_fp32 | resident_gpu | 16777216 | 0.473634 | 0.478400 | 178.218 |
| mxfp8 | dequant_fp16 | resident_gpu | 16777216 | 0.464857 | 0.469303 | 109.401 |
| mxfp8 | dequant_bf16 | resident_gpu | 16777216 | 0.464480 | 0.468709 | 109.490 |
| mxfp8 | quant_optimized | host_api | 16777216 | 21.976261 | 22.059542 | 3.841 |
| mxfp8 | quant_reused_workspace | host_api | 16777216 | 21.788971 | 21.811378 | 3.874 |
| nvfp4 | quant_enumeration | resident_gpu | 1048576 | 2.186160 | 2.191177 | 2.188 |
| nvfp4 | quant_optimized | resident_gpu | 1048576 | 0.617417 | 0.623863 | 7.749 |
| nvfp4 | dequant_fp32 | resident_gpu | 1048576 | 0.100651 | 0.102286 | 47.532 |
| nvfp4 | dequant_fp16 | resident_gpu | 1048576 | 0.096000 | 0.100686 | 27.989 |
| nvfp4 | dequant_bf16 | resident_gpu | 1048576 | 0.095577 | 0.100549 | 28.113 |
| nvfp4 | quant_optimized | host_api | 1048576 | 1.831563 | 1.839341 | 2.612 |
| nvfp4 | quant_reused_workspace | host_api | 1048576 | 1.788198 | 1.795783 | 2.675 |
| nvfp4 | quant_reference | cpu | 1048576 | 134.460637 | 134.530027 | 0.036 |
| nvfp4 | dequant_fp32 | cpu | 1048576 | 15.226129 | 16.007598 | 0.314 |
| nvfp4 | quant_enumeration | resident_gpu | 4194304 | 8.312091 | 8.319543 | 2.302 |
| nvfp4 | quant_optimized | resident_gpu | 4194304 | 2.092537 | 2.096754 | 9.145 |
| nvfp4 | dequant_fp32 | resident_gpu | 4194304 | 0.226709 | 0.227520 | 84.410 |
| nvfp4 | dequant_fp16 | resident_gpu | 4194304 | 0.225771 | 0.226126 | 47.605 |
| nvfp4 | dequant_bf16 | resident_gpu | 4194304 | 0.226000 | 0.226400 | 47.557 |
| nvfp4 | quant_optimized | host_api | 4194304 | 5.334411 | 5.364535 | 3.587 |
| nvfp4 | quant_reused_workspace | host_api | 4194304 | 5.260079 | 5.298287 | 3.638 |
| nvfp4 | quant_enumeration | resident_gpu | 16777216 | 32.958811 | 32.964157 | 2.322 |
| nvfp4 | quant_optimized | resident_gpu | 16777216 | 8.093646 | 8.099040 | 9.458 |
| nvfp4 | dequant_fp32 | resident_gpu | 16777216 | 0.738949 | 0.743886 | 103.588 |
| nvfp4 | dequant_fp16 | resident_gpu | 16777216 | 0.742126 | 0.744526 | 57.930 |
| nvfp4 | dequant_bf16 | resident_gpu | 16777216 | 0.743600 | 0.746149 | 57.816 |
| nvfp4 | quant_optimized | host_api | 16777216 | 19.410166 | 19.484475 | 3.944 |
| nvfp4 | quant_reused_workspace | host_api | 16777216 | 19.161915 | 19.196060 | 3.995 |

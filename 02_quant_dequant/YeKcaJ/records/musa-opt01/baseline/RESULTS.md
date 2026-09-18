# 性能测试结果

后端：MUSA。不同平台的内部对照路径和编译器可能不同，性能不能直接归因于硬件。
GPU 驻留计时使用 MUSA event；host_api 包括量化的分配、传输和释放。
host_api 中 quant_reused_workspace 复用显存和 event，不计首次创建/最终销毁；仍包含上传、量化、下载及主机结果分配/释放，与 quant_optimized 交替测量。此项衡量同进程重复调用，不代表单次 CLI 自动获得相同收益。
内部对照保留 shared scale 归约与枚举元素编码；两种格式的默认 block+nearest 均使用 scale/编码融合 kernel，NVFP4 另含两级全局归约。内部对照的 scale 也可能共享优化，优化前后请比较留存的 quant_optimized 记录。
中位数采用偶数样本中间两值均值，P95 使用 nearest-rank。CPU 仅测试 1M，预热一次后计时三次。

| 格式 | 操作 | 范围 | 元素数 | median ms | P95 ms | 逻辑 GB/s |
|---|---|---|---:|---:|---:|---:|
| mxfp8 | quant_enumeration | resident_gpu | 1048576 | 3.492366 | 3.493280 | 1.511 |
| mxfp8 | quant_optimized | resident_gpu | 1048576 | 0.676629 | 0.681691 | 7.797 |
| mxfp8 | dequant_fp32 | resident_gpu | 1048576 | 0.088891 | 0.092754 | 59.349 |
| mxfp8 | dequant_fp16 | resident_gpu | 1048576 | 0.090160 | 0.091269 | 35.254 |
| mxfp8 | dequant_bf16 | resident_gpu | 1048576 | 0.087554 | 0.090766 | 36.303 |
| mxfp8 | quant_optimized | host_api | 1048576 | 1.832640 | 1.838559 | 2.879 |
| mxfp8 | quant_reused_workspace | host_api | 1048576 | 1.784093 | 1.797263 | 2.957 |
| mxfp8 | quant_reference | cpu | 1048576 | 1916.670269 | 1917.109469 | 0.003 |
| mxfp8 | dequant_fp32 | cpu | 1048576 | 16.061959 | 16.912292 | 0.328 |
| mxfp8 | quant_enumeration | resident_gpu | 4194304 | 13.729303 | 13.730468 | 1.537 |
| mxfp8 | quant_optimized | resident_gpu | 4194304 | 2.564229 | 2.564892 | 8.230 |
| mxfp8 | dequant_fp32 | resident_gpu | 4194304 | 0.148674 | 0.148937 | 141.938 |
| mxfp8 | dequant_fp16 | resident_gpu | 4194304 | 0.146983 | 0.147589 | 86.500 |
| mxfp8 | dequant_bf16 | resident_gpu | 4194304 | 0.147131 | 0.147954 | 86.412 |
| mxfp8 | quant_optimized | host_api | 4194304 | 5.305940 | 5.338720 | 3.977 |
| mxfp8 | quant_reused_workspace | host_api | 4194304 | 5.199494 | 5.210933 | 4.059 |
| mxfp8 | quant_enumeration | resident_gpu | 16777216 | 54.604342 | 54.608070 | 1.546 |
| mxfp8 | quant_optimized | resident_gpu | 16777216 | 10.094629 | 10.097554 | 8.362 |
| mxfp8 | dequant_fp32 | resident_gpu | 16777216 | 0.474834 | 0.478126 | 177.768 |
| mxfp8 | dequant_fp16 | resident_gpu | 16777216 | 0.465657 | 0.469440 | 109.213 |
| mxfp8 | dequant_bf16 | resident_gpu | 16777216 | 0.464194 | 0.469577 | 109.557 |
| mxfp8 | quant_optimized | host_api | 16777216 | 17.145151 | 17.186585 | 4.923 |
| mxfp8 | quant_reused_workspace | host_api | 16777216 | 16.895739 | 16.937988 | 4.996 |
| nvfp4 | quant_enumeration | resident_gpu | 1048576 | 2.186571 | 2.193166 | 2.188 |
| nvfp4 | quant_optimized | resident_gpu | 1048576 | 0.622149 | 0.627771 | 7.690 |
| nvfp4 | dequant_fp32 | resident_gpu | 1048576 | 0.100846 | 0.104206 | 47.440 |
| nvfp4 | dequant_fp16 | resident_gpu | 1048576 | 0.099074 | 0.100000 | 27.121 |
| nvfp4 | dequant_bf16 | resident_gpu | 1048576 | 0.096331 | 0.103726 | 27.893 |
| nvfp4 | quant_optimized | host_api | 1048576 | 1.793826 | 1.806666 | 2.667 |
| nvfp4 | quant_reused_workspace | host_api | 1048576 | 1.752028 | 1.762240 | 2.731 |
| nvfp4 | quant_reference | cpu | 1048576 | 134.781886 | 135.836933 | 0.035 |
| nvfp4 | dequant_fp32 | cpu | 1048576 | 15.266031 | 15.964620 | 0.313 |
| nvfp4 | quant_enumeration | resident_gpu | 4194304 | 8.311657 | 8.315337 | 2.302 |
| nvfp4 | quant_optimized | resident_gpu | 4194304 | 2.090983 | 2.096137 | 9.152 |
| nvfp4 | dequant_fp32 | resident_gpu | 4194304 | 0.223143 | 0.227840 | 85.759 |
| nvfp4 | dequant_fp16 | resident_gpu | 4194304 | 0.224194 | 0.226240 | 47.940 |
| nvfp4 | dequant_bf16 | resident_gpu | 4194304 | 0.221840 | 0.226217 | 48.449 |
| nvfp4 | quant_optimized | host_api | 4194304 | 4.594476 | 4.616930 | 4.165 |
| nvfp4 | quant_reused_workspace | host_api | 4194304 | 4.512172 | 4.547134 | 4.241 |
| nvfp4 | quant_enumeration | resident_gpu | 16777216 | 32.959896 | 32.962719 | 2.322 |
| nvfp4 | quant_optimized | resident_gpu | 16777216 | 8.095337 | 8.100206 | 9.456 |
| nvfp4 | dequant_fp32 | resident_gpu | 16777216 | 0.738971 | 0.744091 | 103.585 |
| nvfp4 | dequant_fp16 | resident_gpu | 16777216 | 0.742606 | 0.747771 | 57.893 |
| nvfp4 | dequant_bf16 | resident_gpu | 16777216 | 0.741406 | 0.747977 | 57.987 |
| nvfp4 | quant_optimized | host_api | 16777216 | 14.785834 | 14.832927 | 5.177 |
| nvfp4 | quant_reused_workspace | host_api | 16777216 | 14.574765 | 14.644125 | 5.252 |

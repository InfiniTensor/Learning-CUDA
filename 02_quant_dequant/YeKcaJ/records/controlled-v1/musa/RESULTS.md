# 性能测试结果

协议 fixed-fp32-v1：冻结 FP32 正态输入文件，seed=20260909，1M/4M/16M；block/nearest，预热3次、测量20次。输入与二进制 SHA256 见 environment.json。
后端：MUSA。不同平台的内部对照路径和编译器可能不同，性能不能直接归因于硬件。
GPU 驻留计时使用 MUSA event；host_api 包括量化的分配、传输和释放。
host_api 中 quant_reused_workspace 复用显存和 event，不计首次创建/最终销毁；仍包含上传、量化、下载及主机结果分配/释放，与 quant_optimized 交替测量。此项衡量同进程重复调用，不代表单次 CLI 自动获得相同收益。
内部对照保留 shared scale 归约与枚举元素编码；两种格式的默认 block+nearest 均使用 scale/编码融合 kernel，NVFP4 另含两级全局归约。内部对照的 scale 也可能共享优化，优化前后请比较留存的 quant_optimized 记录。
中位数采用偶数样本中间两值均值，P95 使用 nearest-rank。CPU 仅测试 1M，预热一次后计时三次。

| 格式 | 操作 | 范围 | 元素数 | median ms | P95 ms | 逻辑 GB/s |
|---|---|---|---:|---:|---:|---:|
| mxfp8 | quant_enumeration | resident_gpu | 1048576 | 2.571760 | 2.574583 | 2.051 |
| mxfp8 | quant_optimized | resident_gpu | 1048576 | 0.172377 | 0.172960 | 30.605 |
| mxfp8 | dequant_fp32 | resident_gpu | 1048576 | 0.091634 | 0.092389 | 57.573 |
| mxfp8 | dequant_fp16 | resident_gpu | 1048576 | 0.090274 | 0.090720 | 35.209 |
| mxfp8 | dequant_bf16 | resident_gpu | 1048576 | 0.090034 | 0.091451 | 35.303 |
| mxfp8 | quant_optimized | host_api | 1048576 | 1.227603 | 1.240791 | 4.298 |
| mxfp8 | quant_reused_workspace | host_api | 1048576 | 1.181888 | 1.191304 | 4.464 |
| mxfp8 | quant_reference | cpu | 1048576 | 1924.620252 | 1925.716630 | 0.003 |
| mxfp8 | dequant_fp32 | cpu | 1048576 | 17.103306 | 17.527295 | 0.308 |
| mxfp8 | quant_enumeration | resident_gpu | 4194304 | 10.046675 | 10.051337 | 2.100 |
| mxfp8 | quant_optimized | resident_gpu | 4194304 | 0.400114 | 0.405394 | 52.741 |
| mxfp8 | dequant_fp32 | resident_gpu | 4194304 | 0.143771 | 0.148640 | 146.779 |
| mxfp8 | dequant_fp16 | resident_gpu | 4194304 | 0.144594 | 0.147657 | 87.929 |
| mxfp8 | dequant_bf16 | resident_gpu | 4194304 | 0.143109 | 0.147314 | 88.842 |
| mxfp8 | quant_optimized | host_api | 4194304 | 3.463144 | 3.527439 | 6.093 |
| mxfp8 | quant_reused_workspace | host_api | 4194304 | 3.364277 | 3.426229 | 6.273 |
| mxfp8 | quant_enumeration | resident_gpu | 16777216 | 39.919075 | 39.922672 | 2.115 |
| mxfp8 | quant_optimized | resident_gpu | 16777216 | 1.363303 | 1.368183 | 61.916 |
| mxfp8 | dequant_fp32 | resident_gpu | 16777216 | 0.472651 | 0.480160 | 178.589 |
| mxfp8 | dequant_fp16 | resident_gpu | 16777216 | 0.465006 | 0.469943 | 109.366 |
| mxfp8 | dequant_bf16 | resident_gpu | 16777216 | 0.465394 | 0.469737 | 109.275 |
| mxfp8 | quant_optimized | host_api | 16777216 | 9.821629 | 9.878330 | 8.594 |
| mxfp8 | quant_reused_workspace | host_api | 16777216 | 9.592405 | 9.641349 | 8.800 |
| nvfp4 | quant_enumeration | resident_gpu | 1048576 | 0.620709 | 0.625097 | 7.708 |
| nvfp4 | quant_optimized | resident_gpu | 1048576 | 0.244434 | 0.248640 | 19.572 |
| nvfp4 | dequant_fp32 | resident_gpu | 1048576 | 0.101337 | 0.106286 | 47.210 |
| nvfp4 | dequant_fp16 | resident_gpu | 1048576 | 0.097749 | 0.102446 | 27.489 |
| nvfp4 | dequant_bf16 | resident_gpu | 1048576 | 0.099349 | 0.100526 | 27.046 |
| nvfp4 | quant_optimized | host_api | 1048576 | 1.237760 | 1.246283 | 3.865 |
| nvfp4 | quant_reused_workspace | host_api | 1048576 | 1.198208 | 1.216101 | 3.993 |
| nvfp4 | quant_reference | cpu | 1048576 | 133.711301 | 134.875070 | 0.036 |
| nvfp4 | dequant_fp32 | cpu | 1048576 | 15.150220 | 15.975648 | 0.316 |
| nvfp4 | quant_enumeration | resident_gpu | 4194304 | 2.022091 | 2.025006 | 9.464 |
| nvfp4 | quant_optimized | resident_gpu | 4194304 | 0.506709 | 0.508434 | 37.766 |
| nvfp4 | dequant_fp32 | resident_gpu | 4194304 | 0.227326 | 0.228411 | 84.181 |
| nvfp4 | dequant_fp16 | resident_gpu | 4194304 | 0.226263 | 0.227337 | 47.502 |
| nvfp4 | dequant_bf16 | resident_gpu | 4194304 | 0.226400 | 0.226971 | 47.473 |
| nvfp4 | quant_optimized | host_api | 4194304 | 3.878069 | 3.896266 | 4.935 |
| nvfp4 | quant_reused_workspace | host_api | 4194304 | 3.803528 | 3.851654 | 5.031 |
| nvfp4 | quant_enumeration | resident_gpu | 16777216 | 7.900514 | 7.906263 | 9.689 |
| nvfp4 | quant_optimized | resident_gpu | 16777216 | 1.618926 | 1.650674 | 47.282 |
| nvfp4 | dequant_fp32 | resident_gpu | 16777216 | 0.738754 | 0.743954 | 103.615 |
| nvfp4 | dequant_fp16 | resident_gpu | 16777216 | 0.739520 | 0.741623 | 58.134 |
| nvfp4 | dequant_bf16 | resident_gpu | 16777216 | 0.740389 | 0.743520 | 58.066 |
| nvfp4 | quant_optimized | host_api | 16777216 | 24.385869 | 24.542682 | 3.139 |
| nvfp4 | quant_reused_workspace | host_api | 16777216 | 24.145824 | 24.341085 | 3.170 |

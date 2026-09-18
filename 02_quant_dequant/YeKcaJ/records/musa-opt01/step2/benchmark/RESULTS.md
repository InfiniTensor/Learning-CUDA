# 性能测试结果

后端：MUSA。不同平台的内部对照路径和编译器可能不同，性能不能直接归因于硬件。
GPU 驻留计时使用 MUSA event；host_api 包括量化的分配、传输和释放。
host_api 中 quant_reused_workspace 复用显存和 event，不计首次创建/最终销毁；仍包含上传、量化、下载及主机结果分配/释放，与 quant_optimized 交替测量。此项衡量同进程重复调用，不代表单次 CLI 自动获得相同收益。
内部对照保留 shared scale 归约与枚举元素编码；两种格式的默认 block+nearest 均使用 scale/编码融合 kernel，NVFP4 另含两级全局归约。内部对照的 scale 也可能共享优化，优化前后请比较留存的 quant_optimized 记录。
中位数采用偶数样本中间两值均值，P95 使用 nearest-rank。CPU 仅测试 1M，预热一次后计时三次。

| 格式 | 操作 | 范围 | 元素数 | median ms | P95 ms | 逻辑 GB/s |
|---|---|---|---:|---:|---:|---:|
| mxfp8 | quant_enumeration | resident_gpu | 1048576 | 3.488206 | 3.488914 | 1.512 |
| mxfp8 | quant_optimized | resident_gpu | 1048576 | 0.505200 | 0.506034 | 10.443 |
| mxfp8 | dequant_fp32 | resident_gpu | 1048576 | 0.081291 | 0.081669 | 64.898 |
| mxfp8 | dequant_fp16 | resident_gpu | 1048576 | 0.079909 | 0.080297 | 39.777 |
| mxfp8 | dequant_bf16 | resident_gpu | 1048576 | 0.080000 | 0.085897 | 39.731 |
| mxfp8 | quant_optimized | host_api | 1048576 | 1.730597 | 1.743071 | 3.048 |
| mxfp8 | quant_reused_workspace | host_api | 1048576 | 1.683349 | 1.690960 | 3.134 |
| mxfp8 | quant_reference | cpu | 1048576 | 1925.886001 | 1927.175100 | 0.003 |
| mxfp8 | dequant_fp32 | cpu | 1048576 | 16.776270 | 17.573542 | 0.314 |
| mxfp8 | quant_enumeration | resident_gpu | 4194304 | 13.724011 | 13.724983 | 1.538 |
| mxfp8 | quant_optimized | resident_gpu | 4194304 | 1.623531 | 1.624457 | 12.998 |
| mxfp8 | dequant_fp32 | resident_gpu | 4194304 | 0.144286 | 0.144709 | 146.256 |
| mxfp8 | dequant_fp16 | resident_gpu | 4194304 | 0.142274 | 0.142743 | 89.362 |
| mxfp8 | dequant_bf16 | resident_gpu | 4194304 | 0.142480 | 0.142949 | 89.233 |
| mxfp8 | quant_optimized | host_api | 4194304 | 5.073078 | 5.111500 | 4.160 |
| mxfp8 | quant_reused_workspace | host_api | 4194304 | 4.970650 | 5.062013 | 4.245 |
| mxfp8 | quant_enumeration | resident_gpu | 16777216 | 54.602663 | 54.607677 | 1.546 |
| mxfp8 | quant_optimized | resident_gpu | 16777216 | 6.326149 | 6.327657 | 13.343 |
| mxfp8 | dequant_fp32 | resident_gpu | 16777216 | 0.472263 | 0.473669 | 178.736 |
| mxfp8 | dequant_fp16 | resident_gpu | 16777216 | 0.463771 | 0.465303 | 109.657 |
| mxfp8 | dequant_bf16 | resident_gpu | 16777216 | 0.463520 | 0.464937 | 109.717 |
| mxfp8 | quant_optimized | host_api | 16777216 | 16.927298 | 21.264428 | 4.987 |
| mxfp8 | quant_reused_workspace | host_api | 16777216 | 16.616960 | 19.557725 | 5.080 |
| nvfp4 | quant_enumeration | resident_gpu | 1048576 | 2.183977 | 2.189280 | 2.191 |
| nvfp4 | quant_optimized | resident_gpu | 1048576 | 0.597554 | 0.602651 | 8.006 |
| nvfp4 | dequant_fp32 | resident_gpu | 1048576 | 0.093920 | 0.094331 | 50.938 |
| nvfp4 | dequant_fp16 | resident_gpu | 1048576 | 0.093063 | 0.093417 | 28.873 |
| nvfp4 | dequant_bf16 | resident_gpu | 1048576 | 0.092880 | 0.093851 | 28.930 |
| nvfp4 | quant_optimized | host_api | 1048576 | 1.796390 | 1.866765 | 2.663 |
| nvfp4 | quant_reused_workspace | host_api | 1048576 | 1.741898 | 1.813057 | 2.747 |
| nvfp4 | quant_reference | cpu | 1048576 | 135.175025 | 135.386270 | 0.035 |
| nvfp4 | dequant_fp32 | cpu | 1048576 | 15.453993 | 16.230297 | 0.310 |
| nvfp4 | quant_enumeration | resident_gpu | 4194304 | 8.307131 | 8.312321 | 2.304 |
| nvfp4 | quant_optimized | resident_gpu | 4194304 | 1.747669 | 1.754446 | 10.950 |
| nvfp4 | dequant_fp32 | resident_gpu | 4194304 | 0.222606 | 0.224914 | 85.966 |
| nvfp4 | dequant_fp16 | resident_gpu | 4194304 | 0.221680 | 0.221966 | 48.484 |
| nvfp4 | dequant_bf16 | resident_gpu | 4194304 | 0.221383 | 0.221783 | 48.549 |
| nvfp4 | quant_optimized | host_api | 4194304 | 5.123500 | 6.051128 | 3.735 |
| nvfp4 | quant_reused_workspace | host_api | 4194304 | 5.038970 | 5.174904 | 3.798 |
| nvfp4 | quant_enumeration | resident_gpu | 16777216 | 32.954845 | 32.957073 | 2.323 |
| nvfp4 | quant_optimized | resident_gpu | 16777216 | 6.718789 | 6.721989 | 11.393 |
| nvfp4 | dequant_fp32 | resident_gpu | 16777216 | 0.738126 | 0.740343 | 103.703 |
| nvfp4 | dequant_fp16 | resident_gpu | 16777216 | 0.738789 | 0.741257 | 58.192 |
| nvfp4 | dequant_bf16 | resident_gpu | 16777216 | 0.739394 | 0.742034 | 58.144 |
| nvfp4 | quant_optimized | host_api | 16777216 | 18.560997 | 21.104286 | 4.124 |
| nvfp4 | quant_reused_workspace | host_api | 16777216 | 18.172561 | 20.818259 | 4.212 |

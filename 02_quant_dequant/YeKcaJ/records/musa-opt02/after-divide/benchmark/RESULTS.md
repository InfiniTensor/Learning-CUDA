# 性能测试结果

后端：MUSA。不同平台的内部对照路径和编译器可能不同，性能不能直接归因于硬件。
GPU 驻留计时使用 MUSA event；host_api 包括量化的分配、传输和释放。
host_api 中 quant_reused_workspace 复用显存和 event，不计首次创建/最终销毁；仍包含上传、量化、下载及主机结果分配/释放，与 quant_optimized 交替测量。此项衡量同进程重复调用，不代表单次 CLI 自动获得相同收益。
内部对照保留 shared scale 归约与枚举元素编码；两种格式的默认 block+nearest 均使用 scale/编码融合 kernel，NVFP4 另含两级全局归约。内部对照的 scale 也可能共享优化，优化前后请比较留存的 quant_optimized 记录。
中位数采用偶数样本中间两值均值，P95 使用 nearest-rank。CPU 仅测试 1M，预热一次后计时三次。

| 格式 | 操作 | 范围 | 元素数 | median ms | P95 ms | 逻辑 GB/s |
|---|---|---|---:|---:|---:|---:|
| mxfp8 | quant_enumeration | resident_gpu | 1048576 | 2.569223 | 2.569577 | 2.053 |
| mxfp8 | quant_optimized | resident_gpu | 1048576 | 0.167451 | 0.171291 | 31.506 |
| mxfp8 | dequant_fp32 | resident_gpu | 1048576 | 0.081497 | 0.082126 | 64.734 |
| mxfp8 | dequant_fp16 | resident_gpu | 1048576 | 0.080320 | 0.085211 | 39.573 |
| mxfp8 | dequant_bf16 | resident_gpu | 1048576 | 0.080149 | 0.081897 | 39.658 |
| mxfp8 | quant_optimized | host_api | 1048576 | 1.436416 | 1.484869 | 3.673 |
| mxfp8 | quant_reused_workspace | host_api | 1048576 | 1.389433 | 1.469053 | 3.797 |
| mxfp8 | quant_reference | cpu | 1048576 | 1918.042886 | 1927.711071 | 0.003 |
| mxfp8 | dequant_fp32 | cpu | 1048576 | 16.287541 | 16.789086 | 0.324 |
| mxfp8 | quant_enumeration | resident_gpu | 4194304 | 10.046423 | 10.047292 | 2.101 |
| mxfp8 | quant_optimized | resident_gpu | 4194304 | 0.399829 | 0.400777 | 52.779 |
| mxfp8 | dequant_fp32 | resident_gpu | 4194304 | 0.143760 | 0.144114 | 146.790 |
| mxfp8 | dequant_fp16 | resident_gpu | 4194304 | 0.142411 | 0.142743 | 89.276 |
| mxfp8 | dequant_bf16 | resident_gpu | 4194304 | 0.142309 | 0.142606 | 89.341 |
| mxfp8 | quant_optimized | host_api | 4194304 | 3.719712 | 4.232818 | 5.673 |
| mxfp8 | quant_reused_workspace | host_api | 4194304 | 3.623886 | 4.232643 | 5.823 |
| mxfp8 | quant_enumeration | resident_gpu | 16777216 | 39.918308 | 39.922329 | 2.115 |
| mxfp8 | quant_optimized | resident_gpu | 16777216 | 1.362297 | 1.367954 | 61.962 |
| mxfp8 | dequant_fp32 | resident_gpu | 16777216 | 0.472377 | 0.473417 | 178.693 |
| mxfp8 | dequant_fp16 | resident_gpu | 16777216 | 0.464549 | 0.465531 | 109.474 |
| mxfp8 | dequant_bf16 | resident_gpu | 16777216 | 0.463840 | 0.465669 | 109.641 |
| mxfp8 | quant_optimized | host_api | 16777216 | 12.411309 | 12.437197 | 6.801 |
| mxfp8 | quant_reused_workspace | host_api | 16777216 | 12.191919 | 12.221895 | 6.923 |
| nvfp4 | quant_enumeration | resident_gpu | 1048576 | 0.618549 | 0.623200 | 7.734 |
| nvfp4 | quant_optimized | resident_gpu | 1048576 | 0.242834 | 0.248709 | 19.701 |
| nvfp4 | dequant_fp32 | resident_gpu | 1048576 | 0.093909 | 0.094377 | 50.945 |
| nvfp4 | dequant_fp16 | resident_gpu | 1048576 | 0.093131 | 0.093737 | 28.851 |
| nvfp4 | dequant_bf16 | resident_gpu | 1048576 | 0.093040 | 0.093646 | 28.880 |
| nvfp4 | quant_optimized | host_api | 1048576 | 1.439934 | 1.450578 | 3.322 |
| nvfp4 | quant_reused_workspace | host_api | 1048576 | 1.392398 | 1.413786 | 3.436 |
| nvfp4 | quant_reference | cpu | 1048576 | 134.481261 | 136.362826 | 0.036 |
| nvfp4 | dequant_fp32 | cpu | 1048576 | 15.681100 | 15.973598 | 0.305 |
| nvfp4 | quant_enumeration | resident_gpu | 4194304 | 2.011337 | 2.013029 | 9.514 |
| nvfp4 | quant_optimized | resident_gpu | 4194304 | 0.494149 | 0.496480 | 38.726 |
| nvfp4 | dequant_fp32 | resident_gpu | 4194304 | 0.222983 | 0.223200 | 85.821 |
| nvfp4 | dequant_fp16 | resident_gpu | 4194304 | 0.221703 | 0.221989 | 48.479 |
| nvfp4 | dequant_bf16 | resident_gpu | 4194304 | 0.221577 | 0.222217 | 48.506 |
| nvfp4 | quant_optimized | host_api | 4194304 | 3.899312 | 3.915442 | 4.908 |
| nvfp4 | quant_reused_workspace | host_api | 4194304 | 3.815641 | 3.836812 | 5.015 |
| nvfp4 | quant_enumeration | resident_gpu | 16777216 | 7.874948 | 7.903566 | 9.720 |
| nvfp4 | quant_optimized | resident_gpu | 16777216 | 1.649154 | 1.654057 | 46.415 |
| nvfp4 | dequant_fp32 | resident_gpu | 16777216 | 0.737943 | 0.738583 | 103.729 |
| nvfp4 | dequant_fp16 | resident_gpu | 16777216 | 0.739954 | 0.744251 | 58.100 |
| nvfp4 | dequant_bf16 | resident_gpu | 16777216 | 0.739143 | 0.742743 | 58.164 |
| nvfp4 | quant_optimized | host_api | 16777216 | 13.765572 | 13.804670 | 5.561 |
| nvfp4 | quant_reused_workspace | host_api | 16777216 | 13.547902 | 13.598257 | 5.650 |

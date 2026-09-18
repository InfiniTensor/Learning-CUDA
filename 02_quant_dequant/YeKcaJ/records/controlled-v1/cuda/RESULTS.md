# 性能测试结果

协议 fixed-fp32-v1：冻结 FP32 正态输入文件，seed=20260909，1M/4M/16M；block/nearest，预热3次、测量20次。输入与二进制 SHA256 见 environment.json。
后端：CUDA。不同平台的内部对照路径和编译器可能不同，性能不能直接归因于硬件。
GPU 驻留计时使用 CUDA event；host_api 包括量化的分配、传输和释放。
host_api 中 quant_reused_workspace 复用显存和 event，不计首次创建/最终销毁；仍包含上传、量化、下载及主机结果分配/释放，与 quant_optimized 交替测量。此项衡量同进程重复调用，不代表单次 CLI 自动获得相同收益。
内部对照保留 shared scale 归约与枚举元素编码；两种格式的默认 block+nearest 均使用 scale/编码融合 kernel，NVFP4 另含两级全局归约。内部对照的 scale 也可能共享优化，优化前后请比较留存的 quant_optimized 记录。
中位数采用偶数样本中间两值均值，P95 使用 nearest-rank。CPU 仅测试 1M，预热一次后计时三次。

| 格式 | 操作 | 范围 | 元素数 | median ms | P95 ms | 逻辑 GB/s |
|---|---|---|---:|---:|---:|---:|
| mxfp8 | quant_enumeration | resident_gpu | 1048576 | 0.565248 | 0.567296 | 9.333 |
| mxfp8 | quant_optimized | resident_gpu | 1048576 | 0.021504 | 0.031744 | 245.333 |
| mxfp8 | dequant_fp32 | resident_gpu | 1048576 | 0.031744 | 0.032768 | 166.194 |
| mxfp8 | dequant_fp16 | resident_gpu | 1048576 | 0.025600 | 0.026624 | 124.160 |
| mxfp8 | dequant_bf16 | resident_gpu | 1048576 | 0.027648 | 0.033536 | 114.963 |
| mxfp8 | quant_optimized | host_api | 1048576 | 1.934799 | 2.447315 | 2.727 |
| mxfp8 | quant_reused_workspace | host_api | 1048576 | 0.997368 | 1.307940 | 5.290 |
| mxfp8 | quant_reference | cpu | 1048576 | 1336.533592 | 1450.237950 | 0.004 |
| mxfp8 | dequant_fp32 | cpu | 1048576 | 13.825012 | 15.759902 | 0.382 |
| mxfp8 | quant_enumeration | resident_gpu | 4194304 | 2.235392 | 2.247680 | 9.440 |
| mxfp8 | quant_optimized | resident_gpu | 4194304 | 0.070656 | 0.075776 | 298.667 |
| mxfp8 | dequant_fp32 | resident_gpu | 4194304 | 0.112640 | 0.113664 | 187.345 |
| mxfp8 | dequant_fp16 | resident_gpu | 4194304 | 0.103424 | 0.108544 | 122.931 |
| mxfp8 | dequant_bf16 | resident_gpu | 4194304 | 0.108544 | 0.109568 | 117.132 |
| mxfp8 | quant_optimized | host_api | 4194304 | 5.697663 | 5.979177 | 3.704 |
| mxfp8 | quant_reused_workspace | host_api | 4194304 | 3.446899 | 3.856739 | 6.122 |
| mxfp8 | quant_enumeration | resident_gpu | 16777216 | 7.902208 | 8.329216 | 10.682 |
| mxfp8 | quant_optimized | resident_gpu | 16777216 | 0.269728 | 0.275456 | 312.946 |
| mxfp8 | dequant_fp32 | resident_gpu | 16777216 | 0.403392 | 0.407552 | 209.251 |
| mxfp8 | dequant_fp16 | resident_gpu | 16777216 | 0.363344 | 0.367616 | 139.966 |
| mxfp8 | dequant_bf16 | resident_gpu | 16777216 | 0.379904 | 0.387072 | 133.865 |
| mxfp8 | quant_optimized | host_api | 16777216 | 21.197699 | 23.905482 | 3.982 |
| mxfp8 | quant_reused_workspace | host_api | 16777216 | 13.784016 | 18.024612 | 6.124 |
| nvfp4 | quant_enumeration | resident_gpu | 1048576 | 0.177136 | 0.184320 | 27.008 |
| nvfp4 | quant_optimized | resident_gpu | 1048576 | 0.041984 | 0.090112 | 113.951 |
| nvfp4 | dequant_fp32 | resident_gpu | 1048576 | 0.025600 | 0.026560 | 186.880 |
| nvfp4 | dequant_fp16 | resident_gpu | 1048576 | 0.020288 | 0.024288 | 132.442 |
| nvfp4 | dequant_bf16 | resident_gpu | 1048576 | 0.022528 | 0.022528 | 119.273 |
| nvfp4 | quant_optimized | host_api | 1048576 | 1.990703 | 2.413260 | 2.403 |
| nvfp4 | quant_reused_workspace | host_api | 1048576 | 0.922667 | 1.109451 | 5.185 |
| nvfp4 | quant_reference | cpu | 1048576 | 102.909364 | 106.148228 | 0.046 |
| nvfp4 | dequant_fp32 | cpu | 1048576 | 12.785104 | 12.815503 | 0.374 |
| nvfp4 | quant_enumeration | resident_gpu | 4194304 | 0.719328 | 0.730112 | 26.603 |
| nvfp4 | quant_optimized | resident_gpu | 4194304 | 0.139776 | 0.154368 | 136.908 |
| nvfp4 | dequant_fp32 | resident_gpu | 4194304 | 0.088112 | 0.092160 | 217.184 |
| nvfp4 | dequant_fp16 | resident_gpu | 4194304 | 0.078848 | 0.079872 | 136.312 |
| nvfp4 | dequant_bf16 | resident_gpu | 4194304 | 0.087040 | 0.087040 | 123.482 |
| nvfp4 | quant_optimized | host_api | 4194304 | 5.824294 | 6.669517 | 3.286 |
| nvfp4 | quant_reused_workspace | host_api | 4194304 | 3.537434 | 4.242945 | 5.410 |
| nvfp4 | quant_enumeration | resident_gpu | 16777216 | 2.833824 | 2.847744 | 27.012 |
| nvfp4 | quant_optimized | resident_gpu | 16777216 | 0.521216 | 0.522240 | 146.861 |
| nvfp4 | dequant_fp32 | resident_gpu | 16777216 | 0.339968 | 0.344064 | 225.157 |
| nvfp4 | dequant_fp16 | resident_gpu | 16777216 | 0.303104 | 0.307200 | 141.838 |
| nvfp4 | dequant_bf16 | resident_gpu | 16777216 | 0.333824 | 0.338944 | 128.785 |
| nvfp4 | quant_optimized | host_api | 16777216 | 15.750887 | 20.215779 | 4.860 |
| nvfp4 | quant_reused_workspace | host_api | 16777216 | 9.917916 | 12.866623 | 7.718 |

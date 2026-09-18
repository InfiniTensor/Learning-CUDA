# 性能测试结果

GPU 驻留计时使用 CUDA event；host_api 包括量化的分配、传输和释放。
基线使用逐小块 shared scale 归约与枚举编码；优化版使用多组 warp 归约和快速编码。两者共用全局两级归约与显存；不是原始 CLI 的完整实现对比。
中位数采用偶数样本中间两值均值，P95 使用 nearest-rank。CPU 仅测试 1M，预热一次后计时三次。

| 格式 | 操作 | 范围 | 元素数 | median ms | P95 ms | 逻辑 GB/s |
|---|---|---|---:|---:|---:|---:|
| mxfp8 | quant_enumeration | resident_gpu | 1048576 | 0.566272 | 0.568128 | 9.316 |
| mxfp8 | quant_optimized | resident_gpu | 1048576 | 0.124928 | 0.129024 | 42.230 |
| mxfp8 | dequant_fp32 | resident_gpu | 1048576 | 0.031744 | 0.031744 | 166.194 |
| mxfp8 | dequant_fp16 | resident_gpu | 1048576 | 0.024976 | 0.025600 | 127.262 |
| mxfp8 | dequant_bf16 | resident_gpu | 1048576 | 0.026624 | 0.027648 | 119.385 |
| mxfp8 | quant_optimized | host_api | 1048576 | 2.378176 | 2.766810 | 2.218 |
| mxfp8 | quant_reference | cpu | 1048576 | 1307.871305 | 1375.808707 | 0.004 |
| mxfp8 | dequant_fp32 | cpu | 1048576 | 14.971469 | 15.949533 | 0.352 |
| mxfp8 | quant_enumeration | resident_gpu | 4194304 | 2.236416 | 2.237440 | 9.436 |
| mxfp8 | quant_optimized | resident_gpu | 4194304 | 0.468320 | 0.468992 | 45.060 |
| mxfp8 | dequant_fp32 | resident_gpu | 4194304 | 0.111616 | 0.113472 | 189.064 |
| mxfp8 | dequant_fp16 | resident_gpu | 4194304 | 0.103424 | 0.106464 | 122.931 |
| mxfp8 | dequant_bf16 | resident_gpu | 4194304 | 0.109568 | 0.127776 | 116.037 |
| mxfp8 | quant_optimized | host_api | 4194304 | 5.197806 | 5.714604 | 4.060 |
| mxfp8 | quant_enumeration | resident_gpu | 16777216 | 7.887248 | 8.230912 | 10.702 |
| mxfp8 | quant_optimized | resident_gpu | 16777216 | 1.636352 | 1.650688 | 51.584 |
| mxfp8 | dequant_fp32 | resident_gpu | 16777216 | 0.403456 | 0.406528 | 209.218 |
| mxfp8 | dequant_fp16 | resident_gpu | 16777216 | 0.362496 | 0.366592 | 140.294 |
| mxfp8 | dequant_bf16 | resident_gpu | 16777216 | 0.378880 | 0.380928 | 134.227 |
| mxfp8 | quant_optimized | host_api | 16777216 | 17.685712 | 18.705655 | 4.773 |
| nvfp4 | quant_enumeration | resident_gpu | 1048576 | 0.315392 | 0.378880 | 15.169 |
| nvfp4 | quant_optimized | resident_gpu | 1048576 | 0.142336 | 0.200704 | 33.612 |
| nvfp4 | dequant_fp32 | resident_gpu | 1048576 | 0.025600 | 0.059552 | 186.880 |
| nvfp4 | dequant_fp16 | resident_gpu | 1048576 | 0.019920 | 0.064352 | 134.889 |
| nvfp4 | dequant_bf16 | resident_gpu | 1048576 | 0.022528 | 0.051040 | 119.273 |
| nvfp4 | quant_optimized | host_api | 1048576 | 1.969030 | 2.353787 | 2.430 |
| nvfp4 | quant_reference | cpu | 1048576 | 94.743895 | 102.993983 | 0.050 |
| nvfp4 | dequant_fp32 | cpu | 1048576 | 11.920125 | 13.289637 | 0.401 |
| nvfp4 | quant_enumeration | resident_gpu | 4194304 | 1.140736 | 1.148928 | 16.776 |
| nvfp4 | quant_optimized | resident_gpu | 4194304 | 0.449520 | 0.456448 | 42.571 |
| nvfp4 | dequant_fp32 | resident_gpu | 4194304 | 0.088064 | 0.090016 | 217.302 |
| nvfp4 | dequant_fp16 | resident_gpu | 4194304 | 0.078848 | 0.079872 | 136.312 |
| nvfp4 | dequant_bf16 | resident_gpu | 4194304 | 0.086976 | 0.088000 | 123.573 |
| nvfp4 | quant_optimized | host_api | 4194304 | 4.758016 | 5.968172 | 4.022 |
| nvfp4 | quant_enumeration | resident_gpu | 16777216 | 4.503552 | 4.510720 | 16.997 |
| nvfp4 | quant_optimized | resident_gpu | 16777216 | 1.737728 | 1.744896 | 44.050 |
| nvfp4 | dequant_fp32 | resident_gpu | 16777216 | 0.339968 | 0.340992 | 225.157 |
| nvfp4 | dequant_fp16 | resident_gpu | 16777216 | 0.301984 | 0.303040 | 142.364 |
| nvfp4 | dequant_bf16 | resident_gpu | 16777216 | 0.332800 | 0.338944 | 129.182 |
| nvfp4 | quant_optimized | host_api | 16777216 | 17.083996 | 18.895231 | 4.481 |

# 性能测试结果

GPU 驻留计时使用 CUDA event；host_api 包括量化的分配、传输和释放。
内部对照保留 shared scale 归约与枚举元素编码；两种格式的默认 block+nearest 均使用 scale/编码融合 kernel，NVFP4 另含两级全局归约。内部对照的 scale 也可能共享优化，优化前后请比较留存的 quant_optimized 记录。
中位数采用偶数样本中间两值均值，P95 使用 nearest-rank。CPU 仅测试 1M，预热一次后计时三次。

| 格式 | 操作 | 范围 | 元素数 | median ms | P95 ms | 逻辑 GB/s |
|---|---|---|---:|---:|---:|---:|
| mxfp8 | quant_enumeration | resident_gpu | 1048576 | 0.569344 | 0.571392 | 9.266 |
| mxfp8 | quant_optimized | resident_gpu | 1048576 | 0.025600 | 0.094080 | 206.080 |
| mxfp8 | dequant_fp32 | resident_gpu | 1048576 | 0.034816 | 0.036864 | 151.529 |
| mxfp8 | dequant_fp16 | resident_gpu | 1048576 | 0.025968 | 0.029696 | 122.400 |
| mxfp8 | dequant_bf16 | resident_gpu | 1048576 | 0.027648 | 0.032736 | 114.963 |
| mxfp8 | quant_optimized | host_api | 1048576 | 2.338218 | 2.616691 | 2.256 |
| mxfp8 | quant_reference | cpu | 1048576 | 1303.793877 | 1337.660196 | 0.004 |
| mxfp8 | dequant_fp32 | cpu | 1048576 | 13.649684 | 15.183241 | 0.387 |
| mxfp8 | quant_enumeration | resident_gpu | 4194304 | 2.234768 | 2.254720 | 9.443 |
| mxfp8 | quant_optimized | resident_gpu | 4194304 | 0.070656 | 0.073728 | 298.667 |
| mxfp8 | dequant_fp32 | resident_gpu | 4194304 | 0.111616 | 0.115712 | 189.064 |
| mxfp8 | dequant_fp16 | resident_gpu | 4194304 | 0.103424 | 0.108544 | 122.931 |
| mxfp8 | dequant_bf16 | resident_gpu | 4194304 | 0.109408 | 0.109568 | 116.207 |
| mxfp8 | quant_optimized | host_api | 4194304 | 5.398639 | 5.850200 | 3.909 |
| mxfp8 | quant_enumeration | resident_gpu | 16777216 | 7.912816 | 9.019392 | 10.668 |
| mxfp8 | quant_optimized | resident_gpu | 16777216 | 0.269824 | 0.279552 | 312.835 |
| mxfp8 | dequant_fp32 | resident_gpu | 16777216 | 0.408480 | 0.425888 | 206.645 |
| mxfp8 | dequant_fp16 | resident_gpu | 16777216 | 0.363888 | 0.366592 | 139.757 |
| mxfp8 | dequant_bf16 | resident_gpu | 16777216 | 0.379808 | 0.382976 | 133.899 |
| mxfp8 | quant_optimized | host_api | 16777216 | 18.411036 | 21.696017 | 4.585 |
| nvfp4 | quant_enumeration | resident_gpu | 1048576 | 0.198656 | 0.205664 | 24.082 |
| nvfp4 | quant_optimized | resident_gpu | 1048576 | 0.048928 | 0.052224 | 97.779 |
| nvfp4 | dequant_fp32 | resident_gpu | 1048576 | 0.028672 | 0.031744 | 166.857 |
| nvfp4 | dequant_fp16 | resident_gpu | 1048576 | 0.020368 | 0.023552 | 131.922 |
| nvfp4 | dequant_bf16 | resident_gpu | 1048576 | 0.022528 | 0.026624 | 119.273 |
| nvfp4 | quant_optimized | host_api | 1048576 | 1.843497 | 2.104586 | 2.595 |
| nvfp4 | quant_reference | cpu | 1048576 | 94.798739 | 104.689433 | 0.050 |
| nvfp4 | dequant_fp32 | cpu | 1048576 | 12.740212 | 13.055763 | 0.376 |
| nvfp4 | quant_enumeration | resident_gpu | 4194304 | 0.723872 | 0.735232 | 26.436 |
| nvfp4 | quant_optimized | resident_gpu | 4194304 | 0.143360 | 0.147456 | 133.486 |
| nvfp4 | dequant_fp32 | resident_gpu | 4194304 | 0.091136 | 0.093184 | 209.978 |
| nvfp4 | dequant_fp16 | resident_gpu | 4194304 | 0.082944 | 0.086016 | 129.580 |
| nvfp4 | dequant_bf16 | resident_gpu | 4194304 | 0.088576 | 0.091136 | 121.341 |
| nvfp4 | quant_optimized | host_api | 4194304 | 5.088929 | 5.613888 | 3.760 |
| nvfp4 | quant_enumeration | resident_gpu | 16777216 | 2.840064 | 3.702784 | 26.952 |
| nvfp4 | quant_optimized | resident_gpu | 16777216 | 0.522144 | 0.855040 | 146.600 |
| nvfp4 | dequant_fp32 | resident_gpu | 16777216 | 0.341504 | 0.344064 | 224.144 |
| nvfp4 | dequant_fp16 | resident_gpu | 16777216 | 0.304976 | 0.307072 | 140.967 |
| nvfp4 | dequant_bf16 | resident_gpu | 16777216 | 0.336896 | 0.339872 | 127.611 |
| nvfp4 | quant_optimized | host_api | 16777216 | 17.913119 | 20.339999 | 4.273 |

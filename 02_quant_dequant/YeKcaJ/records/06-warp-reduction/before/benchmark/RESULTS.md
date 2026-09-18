# 性能测试结果

GPU 驻留计时使用 CUDA event；host_api 包括量化的分配、传输和释放。
内部对照保留 shared scale 归约与枚举元素编码；两种格式的默认 block+nearest 均使用 scale/编码融合 kernel，NVFP4 另含两级全局归约。内部对照的 scale 也可能共享优化，优化前后请比较留存的 quant_optimized 记录。
中位数采用偶数样本中间两值均值，P95 使用 nearest-rank。CPU 仅测试 1M，预热一次后计时三次。

| 格式 | 操作 | 范围 | 元素数 | median ms | P95 ms | 逻辑 GB/s |
|---|---|---|---:|---:|---:|---:|
| mxfp8 | quant_enumeration | resident_gpu | 1048576 | 0.566272 | 0.567296 | 9.316 |
| mxfp8 | quant_optimized | resident_gpu | 1048576 | 0.034816 | 0.034816 | 151.529 |
| mxfp8 | dequant_fp32 | resident_gpu | 1048576 | 0.031744 | 0.038912 | 166.194 |
| mxfp8 | dequant_fp16 | resident_gpu | 1048576 | 0.025072 | 0.029696 | 126.775 |
| mxfp8 | dequant_bf16 | resident_gpu | 1048576 | 0.026624 | 0.027648 | 119.385 |
| mxfp8 | quant_optimized | host_api | 1048576 | 2.517653 | 2.852864 | 2.095 |
| mxfp8 | quant_reference | cpu | 1048576 | 1316.604992 | 1360.133527 | 0.004 |
| mxfp8 | dequant_fp32 | cpu | 1048576 | 14.073665 | 14.507570 | 0.375 |
| mxfp8 | quant_enumeration | resident_gpu | 4194304 | 2.241536 | 3.239872 | 9.414 |
| mxfp8 | quant_optimized | resident_gpu | 4194304 | 0.126976 | 0.127808 | 166.194 |
| mxfp8 | dequant_fp32 | resident_gpu | 4194304 | 0.111616 | 0.122880 | 189.064 |
| mxfp8 | dequant_fp16 | resident_gpu | 4194304 | 0.103424 | 0.111616 | 122.931 |
| mxfp8 | dequant_bf16 | resident_gpu | 4194304 | 0.108544 | 0.109568 | 117.132 |
| mxfp8 | quant_optimized | host_api | 4194304 | 5.542144 | 6.379204 | 3.808 |
| mxfp8 | quant_enumeration | resident_gpu | 16777216 | 8.874864 | 10.167296 | 9.511 |
| mxfp8 | quant_optimized | resident_gpu | 16777216 | 0.434688 | 0.787456 | 194.186 |
| mxfp8 | dequant_fp32 | resident_gpu | 16777216 | 0.404480 | 0.420864 | 208.689 |
| mxfp8 | dequant_fp16 | resident_gpu | 16777216 | 0.365936 | 0.384000 | 138.975 |
| mxfp8 | dequant_bf16 | resident_gpu | 16777216 | 0.381808 | 0.390144 | 133.198 |
| mxfp8 | quant_optimized | host_api | 16777216 | 19.293125 | 20.977906 | 4.375 |
| nvfp4 | quant_enumeration | resident_gpu | 1048576 | 0.212480 | 0.216064 | 22.516 |
| nvfp4 | quant_optimized | resident_gpu | 1048576 | 0.063488 | 0.064512 | 75.355 |
| nvfp4 | dequant_fp32 | resident_gpu | 1048576 | 0.025600 | 0.029696 | 186.880 |
| nvfp4 | dequant_fp16 | resident_gpu | 1048576 | 0.020480 | 0.020480 | 131.200 |
| nvfp4 | dequant_bf16 | resident_gpu | 1048576 | 0.023552 | 0.027584 | 114.087 |
| nvfp4 | quant_optimized | host_api | 1048576 | 1.834073 | 2.075167 | 2.608 |
| nvfp4 | quant_reference | cpu | 1048576 | 89.942499 | 93.041085 | 0.053 |
| nvfp4 | dequant_fp32 | cpu | 1048576 | 11.761140 | 11.959852 | 0.407 |
| nvfp4 | quant_enumeration | resident_gpu | 4194304 | 0.719872 | 0.727040 | 26.583 |
| nvfp4 | quant_optimized | resident_gpu | 4194304 | 0.145856 | 0.148480 | 131.201 |
| nvfp4 | dequant_fp32 | resident_gpu | 4194304 | 0.090544 | 0.092992 | 211.350 |
| nvfp4 | dequant_fp16 | resident_gpu | 4194304 | 0.079872 | 0.106496 | 134.564 |
| nvfp4 | dequant_bf16 | resident_gpu | 4194304 | 0.086992 | 0.091136 | 123.551 |
| nvfp4 | quant_optimized | host_api | 4194304 | 5.018383 | 5.626707 | 3.813 |
| nvfp4 | quant_enumeration | resident_gpu | 16777216 | 2.820608 | 3.425280 | 27.138 |
| nvfp4 | quant_optimized | resident_gpu | 16777216 | 0.520064 | 0.539648 | 147.186 |
| nvfp4 | dequant_fp32 | resident_gpu | 16777216 | 0.342016 | 0.344960 | 223.808 |
| nvfp4 | dequant_fp16 | resident_gpu | 16777216 | 0.304512 | 0.307040 | 141.182 |
| nvfp4 | dequant_bf16 | resident_gpu | 16777216 | 0.332800 | 0.340960 | 129.182 |
| nvfp4 | quant_optimized | host_api | 16777216 | 17.840104 | 19.249860 | 4.291 |

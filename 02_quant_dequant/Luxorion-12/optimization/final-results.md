# v0与最终v8直测

设备：NVIDIA GeForce RTX 3050 Laptop GPU。v0与v8交错运行3轮，每轮预热5次、计时20次，表中取3轮kernel median的中位数。

| 场景 | v0量化 ms | v8量化 ms | 量化加速 | v0反量化 ms | v8反量化 ms | 反量化加速 |
|---|---:|---:|---:|---:|---:|---:|
| MXFP8 block / FP16输入 | 12.328944 | 5.456864 | 2.26x | 0.053248 | 0.036272 | 1.47x |
| MXFP8 tensor / FP32输入 | 113.272831 | 10.684112 | 10.60x | 0.039936 | 0.025600 | 1.56x |
| NVFP4 block / FP32输入 | 8.883200 | 1.436048 | 6.19x | 0.046592 | 0.036864 | 1.26x |
| NVFP4 tensor / FP16输入 | 9.499648 | 7.846768 | 1.21x | 0.039936 | 0.024576 | 1.63x |

四场景几何平均：量化约3.66x，反量化约1.47x。

## 最终验证

- v8四种完整流程生成的`weights.luxq`和`restored.tensor.bin`与此前正确版本逐字节一致。
- 覆盖MXFP8/NVFP4、block/tensor scale、nearest/stochastic、FP32/原生FP16输入，以及FP32/FP16/BF16输出。
- 奇数元素数和非4倍数尾部已覆盖。
- CTest：13/13通过。

最终程序：`build/cuda/Release/luxorion_v8.exe`。

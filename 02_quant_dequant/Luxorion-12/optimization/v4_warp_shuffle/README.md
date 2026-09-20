# v4 方向4：Warp shuffle归约

基础版本：已晋级的`v3_vectorized_memory`。本版本未晋级。

## 修改

- 32线程MXFP8 block最大值归约由共享内存和多次`__syncthreads()`改为warp shuffle。
- 16线程NVFP4 block归约改为width=16的warp shuffle，并用shuffle传递相邻量化code完成打包。
- tensor-scale的256线程归约改为warp内shuffle加8个warp结果的两级归约。

## 正确性

四个完整流程的`weights.luxq`和`restored.tensor.bin`均与v3逐字节一致。

## 性能与决策

每个进程预热5次、计时20次，v3/v4交错运行3轮，取median的中位数：

| 场景 | v3量化 ms | v4量化 ms | 变化 |
|---|---:|---:|---:|
| MXFP8 block | 12.309952 | 12.312928 | -0.024% |
| MXFP8 tensor | 113.158142 | 113.158653 | -0.0005% |
| NVFP4 block | 8.884224 | 8.877568 | +0.075% |
| NVFP4 tensor | 9.482736 | 9.478960 | +0.040% |

综合差异约+0.02%，处于测量噪声量级，且两个MXFP8核心场景出现轻微回退。按“优于上一最佳版本且核心场景不回退”的规则不晋级；方向5继续基于v3。

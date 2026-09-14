# 低精度量化/反量化（CUDA 方向）
选题二，实现 NVFP4（e2m1）和 MXFP8（e4m3）两种格式的量化与反量化
cpu端和gpu端都实现了，以此作为对比

## 1. 低精度格式
NVFP4（e2m1）：最大值6
MXFP8（e4m3）：最大值448

## 2. 缩放策略
量化时除以缩放因子s，再将其映射到对应精度的格点上，格点全部算出来存储在数组上

### 2.1 量化/反量化公式
NVFP4：MAX = 6.0，格点表 [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] 及负值
MXFP8：MAX= 448，E4M3 位模式生成 128 个非负格点
'''
s = |max| / MAX
code = round(x / s)
dequant = TABLE[code] * s
'''

### 2.2 缩放因子
tensor：整个张量共享一个缩放因子， s = |全局max| / MAX
block：每个block共享一个缩放因子， s = |块max| / MAX

## 3. 打包
NVFP4：每2个4bit的打包成一个字节，低4位在前，高4位在后，一个字节两个元素
MXFP8：一个元素8bit，所以一个字节就是一个元素


## 4. 运行
```
python setup.py build_ext --inplace

# 生成三种输入矩阵（random / normal / outlier）
python gen_data.py --matrix random
python gen_data.py --matrix normal
python gen_data.py --matrix outlier

# 跑量化/反量化，--config 选格式，--matrix 选数据分布
python main.py --config configs/mxfp8_block.txt --matrix random
python main.py --config configs/nvfp4_block.txt --matrix normal
```

## 5. 产物
```
outputs/<matrix>/<format>/cpu/report.log
outputs/<matrix>/<format>/cpu/quantized.bin  #量化文件
outputs/<matrix>/<format>/cpu/dequant.bin

outputs/<matrix>/<format>/cuda/report.log
outputs/<matrix>/<format>/cuda/quantized.bin
outputs/<matrix>/<format>/cuda/dequantized.bin
```
report.log 包含：max_abs_error、mae、mse、compression_ratio、quant_time_s、dequant_time_s、quant_bandwidth_gbps

## 6. 目录结构
```
王玉环/
├── main.py            # 程序入口（CPU/GPU 量化、误差、对比、保存）
├── gen_data.py        # 生成输入矩阵
├── io_utils.py        # 文件读写、误差计算、日志
├── quant_cpu.py       # CPU 参考实现
├── quant_common.py    # 格式格点表、转换、打包/解包
├── quant_cuda.py      # CUDA 扩展封装
├── setup.py           # 构建脚本
├── cuda/
│   ├── nvfp4.cu       # NVFP4 量化/反量化内核
│   └── mxfp8.cu       # MXFP8 量化/反量化内核
├── configs/           # 参数文件
├── inputs/            # 输入矩阵
└── outputs/           # 运行产物
```

## 7. 运行结果（RTX 3090, 4096×4096 FP32 输入）
### MXFP8（block_size=32, scale_mode=tensor）

| 分布 | max_abs | MAE | MSE | 压缩比 | quant 时间 (GPU) | dequant 时间 (GPU) | 带宽 (GB/s) |
|---|---|---|---|---|---|---|---|
| random  | 0.0357 | 0.0111 | 2.17e-4 | 3.56 | 0.0286 s | 0.00167 s | 3.00 |
| normal  | 0.1866 | 0.0180 | 7.01e-4 | 3.56 | 0.0274 s | 0.00168 s | 3.14 |
| outlier | 0.2744 | 0.0180 | 7.01e-4 | 3.56 | 0.0264 s | 0.00167 s | 3.26 |

CPU 端量化约 42 s（GPU 的 ~1500 倍），反量化约 2.9 s。

### NVFP4（block_size=16, scale_mode=block）

| 分布 | max_abs | MAE | MSE | 压缩比 | quant 时间 (GPU) | dequant 时间 (GPU) | 带宽 (GB/s) |
|---|---|---|---|---|---|---|---|
| random  | 0.167 | 0.0429 | 3.38e-3 | 5.33 | 0.0241 s | 0.00512 s | 3.30 |
| normal  | 0.684 | 0.0686 | 8.86e-3 | 5.33 | 0.0229 s | 0.00488 s | 3.48 |
| outlier | 3.169 | 0.0686 | 8.86e-3 | 5.33 | 0.0223 s | 0.00482 s | 3.57 |

CPU 端量化约 8.5 s（GPU 的 ~350 倍），反量化约 2.9 s。

## 8. 实现说明

### 软件模拟部分（不依赖特定硬件）

- **E2M1 / E4M3 编码**：纯位运算 + 查找表，普通 CUDA kernel 实现，不使用 FP8/FP4 原生指令
- **量化 kernel**：nearest rounding，block max 由线程串行计算，tensor max 由 host 端归约后传入
- **打包/解包**：手动位操作，4bit 元素每两个打包成一个字节
- **反量化 kernel**：查表 + 乘 scale，普通 CUDA kernel
- 以上代码我是在RTX3090上运行的

### 依赖的第三方库

- **PyTorch**：使用 `torch.utils.cpp_extension` 构建 CUDA extension，Tensor 作为数据容器；host 端全局 max 调用 `Tensor::abs().max().item()`
- **CUDA Runtime**：仅使用标准 CUDA Runtime API（kernel launch、`cudaMemcpyToSymbol`），未使用 cuBLAS/cuDNN/cuBLASLt 等

## 9.Nsight Compute 性能分析

使用 `ncu --set full --kernel-name regex:quant_` 对四个 kernel 进行 profile：

| kernel | Duration | Memory Throughput | Compute Throughput | Achieved BW | Occupancy |
|---|---|---|---|---|---|
| quant_mxfp8  | 2.69 ms | 48.7% | 38.8% | 389 GB/s | 94.9% |
| dequant_mxfp8 | 2.77 ms | 52.7% | 6.2% | 334 GB/s | 92.7% |
| quant_nvfp4  | 0.17 ms | 91.9% | 38.2% | 508 GB/s | 82.9% |
| dequant_nvfp4 | 0.44 ms | 80.6% | 5.5% | 451 GB/s | 92.4% |
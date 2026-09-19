# MXFP8 / NVFP4 软件模拟与反量化

本项目在不依赖 Hopper、Blackwell 低精度硬件指令的前提下，用普通 CUDA kernel 实现：

- MXFP8 E4M3 与 MXFP8 E5M2；
- NVFP4 E2M1、E4M3 block scale 和 FP32 global scale；
- 编码、分块缩放、4-bit 打包、解包和反量化；
- CPU 参考实现与 GPU 逐字节一致性测试；
- 量化误差、压缩率、CPU/GPU 性能评估。

实现依据：

- [OCP Microscaling Formats (MX) Specification 1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [NVIDIA Transformer Engine NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)
- [NVIDIA cuDNN Block Scaling](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/BlockScaling.html)

## 格式与内存布局

### MXFP8

每 32 个连续 FP32 元素组成一个 block，一个 warp 负责一个 block：

1. warp reduction 计算有限输入的最大绝对值；
2. 按 OCP 规则选择 E8M0 共享 scale；
3. 输入除以 scale 后编码为 E4M3 或 E5M2；
4. 反量化时解码 FP8 并乘回共享 scale。

存储由 N 字节 FP8 数据和 ceil(N/32) 字节 E8M0 scale 组成。相对于 FP32，渐近压缩率为 4 / 1.03125 = 3.88x。

E4M3/E5M2 转换使用 round-to-nearest, ties-to-even，并采用饱和到最大有限值的溢出策略。NaN 编码为对应格式的 NaN；无穷在 SAT 模式下饱和。

### NVFP4

每 16 个连续元素共享一个 E4M3 block scale，整个张量共享一个 FP32 global scale：

    dequantized = E2M1_value * E4M3_block_scale * FP32_global_scale

量化流程：

1. GPU 全局归约得到 global_amax；
2. global_scale = global_amax / (448 * 6)；
3. 每 16 个值归约得到 block_amax；
4. (block_amax / 6) / global_scale 向上舍入到 E4M3，避免 block 最大值溢出；
5. 输入除以两级 scale 后，以 ties-to-even 编码为 E2M1；
6. 两个 E2M1 nibble 打包到一个字节。

存储由 ceil(N/2) 字节 E2M1 数据、ceil(N/16) 字节 E4M3 scale 和一个 4 字节 global scale 组成，渐近压缩率约为 7.11x。

E2M1 没有 NaN/Inf 编码。本实现将 NaN 转换为零，将正负无穷饱和到正负 6；此策略在测试中固定并由 CPU/GPU 共同验证。

## GPU 优化

优化依据 Nsight Compute 的硬件计数器，而不是只观察单次计时：

- MXFP8 E4M3 量化直接解析 FP32 指数和尾数，以整数 guard/round/sticky 规则完成 ties-to-even，避免每个元素调用通用 `frexp/ldexp` 路径；
- MXFP8 反量化每个线程读取 4 个连续 FP8 byte，并用 `float4` 写回 4 个 FP32；
- NVFP4 的 global amax 使用与 30 个 SM 驻留波次匹配的 180 个 reduction block，减少 partial wave；
- NVFP4 block scale 的 E4M3 向上舍入和正值解码使用位级实现，E2M1 使用无分支区间选择；
- NVFP4 量化由半个 warp 处理一个 16 元素 block，反量化每线程读取 4 个 packed byte 并写回 8 个 FP32。

所有向量化路径都有非整块尾部处理，并由 Compute Sanitizer 和多种非对齐长度测试覆盖。

## 构建

本机环境：

- NVIDIA GeForce RTX 3060 Laptop GPU，6 GiB，SM 8.6；
- Ubuntu 24.04 on WSL；
- NVIDIA Driver 591.74；
- CUDA Toolkit 12.8；
- CMake 3.28.3，GCC 13.3。

nvcc 未加入默认 PATH 时可直接指定完整路径：

    cd /home/xu/projects/Learning-CUDA/02_quant_dequant/shuaize6

    cmake -S . -B build \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
      -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DCMAKE_BUILD_TYPE=Release

    cmake --build build --parallel

本实现是独立 CUDA C++ 程序，不需要 PyTorch，也不受 PyTorch wheel 所带 CUDA 13.0 runtime 与系统 CUDA Toolkit 12.8 版本不同的影响。

## 测试

    ctest --test-dir build --output-on-failure

或者：

    ./build/quant_dequant --test

测试覆盖：

- E2M1 全部 16 个 bit pattern；
- E4M3/E5M2 全部有限 bit pattern 的 encode/decode round trip；
- MXFP8 所有相邻正数编码的中点、上下相邻 FP32 值及对应负数的 GPU 舍入边界；
- NVFP4 E2M1 全部舍入中点、上下相邻 FP32 值及对应负数的 GPU 舍入边界；
- ties-to-even 边界；
- E8M0 scale 选择；
- 0、负零、NaN、正负无穷、极值、subnormal；
- 长度小于 block、等于 block、跨 block 和非整块尾部；
- CPU 与 GPU 的 encoded value、scale、packed byte 完全一致；
- CPU 与 GPU 反量化结果一致；
- 零长度 API。

## 误差与性能评估

完整测试与默认 2^24 元素 benchmark：

    ./build/quant_dequant --all

只测试：

    ./build/quant_dequant --test

指定 benchmark 元素数：

    ./build/quant_dequant --benchmark --size 16777216

程序报告：

- max absolute error；
- MAE、RMSE、max relative error；
- cosine similarity、SQNR；
- 实际 encoded bytes 与压缩率；
- CUDA event 测得的 quantize/dequantize kernel 时间和有效带宽；
- 单线程 CPU 参考实现时间与 GPU 加速比。

性能计时不包含主机与显存之间的数据传输，以单独衡量低精度编码和反量化 kernel。

## RTX 3060 Laptop 实测结果

测试日期为 2026-09-19。Release 构建，输入为 2^24 个 FP32 元素（64 MiB）。每次 GPU 时间由 CUDA event 统计 50 次执行的平均值，下表取 3 次独立运行的中位数，不包含 PCIe 传输。CPU 为单线程参考实现。

| 格式 | 压缩率 | GPU 量化 | GPU 反量化 | CPU 量化加速 | CPU 反量化加速 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MXFP8 E4M3 | 3.879x | 0.898 ms / 93.955 GB/s | 0.336 ms / 251.409 GB/s | 440.140x | 940.236x |
| MXFP8 E5M2 | 3.879x | 0.949 ms / 88.902 GB/s | 0.347 ms / 243.091 GB/s | 400.352x | 687.603x |
| NVFP4 | 7.111x | 0.949 ms / 80.661 GB/s | 0.304 ms / 251.453 GB/s | 115.767x | 883.677x |

| 格式 | MAE | RMSE | Cosine similarity | SQNR |
| --- | ---: | ---: | ---: | ---: |
| MXFP8 E4M3 | 1.423369e-1 | 5.225359e-1 | 0.998980 | 25.982368 dB |
| MXFP8 E5M2 | 2.195838e-1 | 6.414765e-1 | 0.998298 | 24.201077 dB |
| NVFP4 | 5.328790e-1 | 1.077206 | 0.995479 | 19.698717 dB |

CTest 结果为 1/1 passed。测试可执行程序内部覆盖 14 种长度、所有标量格式检查以及 MXFP8/NVFP4 全舍入边界，其中最大常规一致性用例为 65,539 个元素。

启用 Windows GPU debugger interface 后，CUDA Compute Sanitizer 复查结果：

- `memcheck`：0 errors；
- `initcheck`：0 errors；
- `racecheck`：0 hazards（0 errors，0 warnings）；
- `synccheck`：0 errors。

Nsight Compute 2025.1 能在当前 WSL 环境读取 RTX 3060 Laptop 的硬件性能计数器。以下是 2^20 元素、本轮优化前与最终版本的同类 kernel 报告对比：

| NCU kernel | 优化前 | 最终版 | 执行指令：优化前 → 最终版 |
| --- | ---: | ---: | ---: |
| MXFP8 E4M3 量化 | 87.71 us | 77.25 us | 6,972,160 → 6,026,989 |
| MXFP8 E4M3 反量化 | 45.63 us | 24.86 us | 2,651,166 → 1,544,553 |
| NVFP4 global amax | 25.12 us | 22.24 us | 505,856 → 462,688 |
| NVFP4 量化主 kernel | 118.88 us | 71.94 us | 10,424,340 → 6,524,948 |
| NVFP4 反量化 | 29.79 us | 25.79 us | 2,129,562 → 598,016 |

完整 `.ncu-rep` 报告生成在本地、被 Git 忽略的 `build/` 目录中。

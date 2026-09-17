# 选题二总结报告：MXFP8 / NVFP4 软件模拟与反量化

本地实现与实测日期：2026-09-13。

## 1. 目标与完成情况

实现普通 CUDA GPU 上的低精度权重处理流程：FP32/FP16 文件读取 → GPU 缩放、编码与
真实位宽打包 → 权重保存 → GPU 解包反量化 → FP32/FP16/BF16 文件及误差、性能日志。
权重文件支持单独加载，不需要原始输入。实现、测试、配置和复现实验脚本均已提供。

所有数值转换为软件模拟。没有使用 Tensor Core、原生 FP8/FP4 指令或第三方量化库。
GPU 执行全局/块最大值归约、编码、打包、解码及输出转换；CPU 负责文件检查、配置、
误差统计和参考验证。该分工将与张量规模相关的主要转换计算放在 GPU 上。

标准配置是 MXFP8 的 32 元素块及 NVFP4 的 16 元素块。任务要求的逐张量缩放、
其他二次幂块长作为实验扩展；它们不属于原生格式的固定块长硬件 ABI。

## 2. 格式、缩放与舍入

### 2.1 MXFP8

每个元素保存 E4M3 的 1 位符号、4 位指数、3 位尾数。指数偏置为 7；
指数域为零时保留次正规数；最大有限值为 448，`0x7f/0xff` 为 NaN。
每个标准块共享一个 8 bit E8M0 scale，编码 `b` 表示 `2^(b-127)`，
有效编码为 0–254。格式定义参考 [OCP MX Specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)。

对块最大绝对值 `a`，本项目采用以下指数选择策略：

```text
e = clamp(floor(log2(a)) - 8, -127, 127)
s = 2^e
q_i = round_E4M3(x_i / s)
x_hat_i = decode_E4M3(q_i) * s
```

零块使用 `s=1`。实现通过 `frexpf` 提取整数指数，避免浮点 `log2` 在幂次边界的误判。
这一选择与 [Microsoft microxcaling 的 max shared exponent 方法](https://github.com/microsoft/microxcaling/blob/main/mx/mx_ops.py)
一致，随后对元素执行有限饱和。它会让部分接近块区间上沿的值超过 448 而截断；
采用该策略不意味着保证无截断，也不意味着块缩放误差总是小于逐张量缩放。

### 2.2 NVFP4

E2M1 的正数编码表如下，最高位为符号，保留负零：

| 低三位 | 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 数值 | 0 | 0.5 | 1 | 1.5 | 2 | 3 | 4 | 6 |

采用一个 FP32 全局缩放和每 16 元素一个 E4M3 局部缩放：

```text
A = max_i(abs(x_i))
g = A / (448 * 6)
l_b = round_E4M3((amax_b / g) / 6)
q_i = round_E2M1((x_i / g) / decode_E4M3(l_b))
x_hat_i = (decode_E2M1(q_i) * decode_E4M3(l_b)) * g
```

两级结构和缩放公式参考 [NVIDIA Transformer Engine NVFP4 文档](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)。
元素量化使用**已编码再解码后的**局部 scale，避免保存的尺度和计算时的尺度不一致。
算式分阶段相除/相乘，避免先把两个尺度相乘造成中间下溢。全零张量使用 `g=1`；
非零 `g` 最低钳制到 `2^-149`；量化后的 local 为零时输出带原符号的零。
这些极小值处理是本模拟器明确规定的数值策略。

逐张量模式令 `amax_b=A`、局部 scale 数组只有一个元素；一般有限范围下它等于 448。
这种模式可观察全局离群值对所有元素的影响。

### 2.3 舍入与异常值策略

默认 `nearest` 为最近偶数舍入：中点选择编码最低有效位为偶数的一侧。
`stochastic` 对相邻可表示值 `lo, hi`，以 `(abs(x)-lo)/(hi-lo)` 的概率选 `hi`；
超出有限范围仍饱和。按元素行主序索引和 64 bit seed 进行 SplitMix64 混合，
取高 24 位产生 `[0,1)` 随机数，CPU/GPU 可逐字节重现。有限 24 位随机分辨率
意味着极小概率会有量化误差；对未饱和区间，随机舍入近似无偏。
局部缩放因子始终使用最近偶数舍入。

NaN/Inf 输入在 CPU 读取阶段拒绝，不参与最大值归约。有限异常值正常处理。
FP16/BF16 输出采用软件 IEEE 最近偶数转换，包含次正规数和溢出 Inf；
若输出有非有限值，日志记录数量并将整体误差写为 `null`，避免 JSON 中出现非法 Infinity。

## 3. 打包布局与文件格式

矩阵沿列方向分块，各行独立，尾块仅归约有效元素。MXFP8 每元素 1 字节；
NVFP4 每字节存两元素，偶数列为低四位、奇数列为高四位，行跨度为 `ceil(cols/2)` 字节。
奇数列时每行最后的高四位必须为零，读取器会检查。

scale 数组按 `[row, block_column]` 行主序排列，元素 1 字节。
矩阵 header 为 32 字节；权重 header 为 96 字节，记录形状、类型、块长、舍入、seed、
全局 scale 和两段载荷长度，后接 packed data 和 scale bytes。字段偏移详见
[README.md](README.md)。读取器在分配内存之前检查维度乘法溢出和实际文件长度，
并拒绝非法 scale、非零保留字段、NaN 权重码及异常 padding。

本项目定义自己的紧凑磁盘协议；它表达相应低精度数值，但没有实现 NVIDIA GEMM
内部使用的 swizzle 或矩阵乘法专用 padding，因此不宣称可以直接交给原生 GEMM。

## 4. CUDA 设计与优化

### 4.1 分块量化

每个 CUDA CTA 使用 256 线程，每个 scale 块分配 `min(block_size,32)` 个线程组成的子组。
默认 MXFP8 使用一个 warp，默认 NVFP4 使用半个 warp。子组通过 shuffle 归约 amax，
融合尺度生成、元素编码和打包，省去单独写读块最大值的全局内存缓冲。
默认块长下每个输入元素在归约前载入寄存器，编码阶段继续使用同一份数据。
大于 32 的扩展块长以 lane stride 扫描额外元素。

NVFP4 编码后，相邻 lane 交换 4 bit 结果，偶数 lane 独占一个字节的 packed store。
反量化一个线程读一个 packed byte，解出两个元素，写两个输出值。
即使行数或列数不能整除块长，也不产生跨行半字节写冲突。

### 4.2 全局归约与逐张量模式

NVFP4 和逐张量 MXFP8 先用至多 4096 个 CTA 进行 grid-stride amax 归约，
再用单 CTA 汇总 partial 并写 global scale，随后发射量化内核。
三个步骤均在 GPU 上完成，量化计时包含全部三步。
逐张量量化采用独立并行逐元素/逐字节内核，不让一个子组串行扫描整张量。

### 4.3 可移植性与正确性取舍

避免使用新架构低精度转换及 BF16 指令；基础类型通过位操作和普通 FP32 运算实现。
不启用 `--use_fast_math`，关闭 FMA contraction，保留次正规数和精确除法路径。
64 位张量索引和按行布局保证大矩阵与奇数列行为明确，但运行时 64 位除法及通用类型
分支仍有开销。当前选择先保持完整边界支持；可进一步为连续、对齐矩阵增加专门内核。

实际运行 GPU 为 Ada。编译产物包含 Pascal `sm_60`、Turing `sm_75` 和 Ada `sm_89`，
见 [cuda_architectures.txt](reports/cuda_architectures.txt)。未在真实 Pascal/Turing 设备实测。

## 5. 验证方法与结果

构建和验证均在 WSL 运行。

1. C++ 基础格式单测：全部 16 个 FP4 码、256 个 FP8 码、全部 FP16/BF16 编码往返，
   全部低精度中点及相邻 FP32 值、饱和、次正规数、符号零和随机舍入均值。
2. 独立 NumPy oracle：用显式码表与 `searchsorted` 选择相邻值，独立计算缩放和打包。
   没有调用 C++ 转换函数，也没有复制其按位截断算法。
3. CPU/GPU 两后端各 111 个集成用例：格式、模式、输入/输出 dtype、舍入全组合；
   块长 2–1024；奇数列、单元素、尾块；全零、有限离群值、FP32 极小/极大值；
   重新加载；破损头部/载荷、NaN/Inf 输入、错误配置和路径冲突。
4. 每次大矩阵 benchmark 额外核对 CPU/GPU 权重、scale、global 和最终输出字节完全相同。
5. 无 CUDA 的 CPU Debug 构建在 `UBSAN_OPTIONS=halt_on_error=1` 下执行同一 CPU 验收。

**以上测试全部通过。** 测试明细见 [tests.txt](reports/tests.txt)、
[tests_details.txt](reports/tests_details.txt) 和 [cpu_ubsan.txt](reports/cpu_ubsan.txt)。
GPU Compute Sanitizer 未能启动 WDDM debugger interface，未取得可用的 sanitizer 检查结果；
原始错误保留在 [sanitizer_mxfp8_memcheck.txt](reports/sanitizer_mxfp8_memcheck.txt)。

## 6. 实验环境与计时口径

| 项目 | 实测环境 |
|---|---|
| 操作系统 | WSL2 Ubuntu 26.04，Linux 6.18.33.2-microsoft-standard-WSL2 |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU，8188 MiB，SM89 |
| Windows NVIDIA 驱动 | 616.56 |
| CUDA Toolkit | 12.4.131 |
| CPU | Intel Core i7-12800HX；基线单线程，未使用 OpenMP |
| 编译器 | GCC 13.4.0，CMake 4.2.3，Release |
| Python / NumPy | 3.14.4 / 2.3.5 |
| 输入 | 1024×4096，FP32 或 FP16，seed=42 |
| 输出 | FP32，nearest ties-to-even |
| 分布 | uniform(-1,1)；标准正态；标准正态中随机 0.1% 元素乘 100 |

GPU kernel 耗时：CUDA Events、预热 5 次、连续运行 50 次取均值；设备内存预先分配，
不包含文件 I/O、内存分配和 H2D/D2H。NVFP4 的 amax 归约及全局 scale 生成计入量化。
CPU 使用同一数值契约和预分配缓冲，预热一次，重复三次取均值；是可读的单线程基线，
没有作为极限 SIMD/OpenMP 性能代表。

另测 `gpu_transfer_pipeline_ms`：在已初始化、预分配状态下，从完整输入 H2D 开始，
经过 GPU 量化、反量化，将全部 packed data、scales、global 和输出拷回 CPU，
同步墙钟计时 5 次平均。使用 pageable host memory，未隐藏传输成本。
此口径代表内存中的处理流程；不含磁盘 I/O、设备初始化、CPU 对照验证和误差扫描。
`file_io_ms` 记录输入/配置读取与两份二进制写入（含输入有限值验证）；
`process_total_ms` 包括基线、多次计时和验证，在写日志前取值，不能当作单次算法时间。

有效带宽为逻辑读写字节数除以 kernel 时间。量化包括输入、packed/scales 写入，
有全局归约时加一次输入读取、partial 读写；大块额外读取按有效元素精确计数。
反量化为 packed bytes + scale bytes + 输出 bytes，NVFP4 再计全局 4 字节。
这是逻辑有效带宽，未计每线程重复 scale load、缓存事务或内存事务对齐成本。
重复访问同一缓冲可能命中 L2，因此 GB/s 可以高于设备 DRAM 带宽；不把它等同于实测 DRAM 吞吐。

## 7. 最终误差、压缩与性能

以下为 FP32 输入、FP32 输出、默认块长；时间单位为毫秒。完整 24 组数据、MAE/MSE、
最大绝对误差、带宽和文件压缩率见 [完整结果表](reports/results/TABLE.md)。

| 分布 | 格式 | MAE | MSE | 量化 ms | 反量化 ms | 含传输 ms | 含传输 CPU/GPU 加速比 |
|---|---|---:|---:|---:|---:|---:|---:|
| uniform | MXFP8 | 0.0162841 | 0.000797306 | 0.160874 | 0.080878 | 4.14228 | 16.56× |
| uniform | NVFP4 | 0.0442829 | 0.00344397 | 0.214221 | 0.045220 | 4.04214 | 24.58× |
| normal | MXFP8 | 0.0187231 | 0.000864783 | 0.155484 | 0.083190 | 4.12883 | 15.16× |
| normal | NVFP4 | 0.0714289 | 0.00904100 | 0.213176 | 0.054866 | 3.99528 | 26.95× |
| outliers | MXFP8 | 0.0211448 | 0.0170052 | 0.163758 | 0.066517 | 6.83754 | 9.34× |
| outliers | NVFP4 | 0.0820370 | 0.0271959 | 0.221041 | 0.045199 | 3.95163 | 27.54× |

标准正态输入时，MXFP8 驻留 GPU 的量化+反量化为 0.238674 ms，CPU 为 62.6076 ms；
NVFP4 分别为 0.268042 ms 和 107.680 ms。含传输仍分别达到 15.16× 和 26.95×。
显示型笔记本 GPU、动态频率、WSL 和 pageable 拷贝会带来抖动，例如异常值 MXFP8 的
传输流程测得 6.84 ms；本报告保留原始结果，不筛除较慢数据。

### 7.1 压缩率

压缩率是原始字节数除以压缩后字节数。分块且列数对齐时：

```text
MXFP8 = N + N/32 bytes                      # 8.25 bits/element
NVFP4 = N/2 + N/16 + 4 bytes                # 4.5 bits/element + global FP32
```

对本实验 4,194,304 元素，MXFP8 有效载荷为 4,325,376 字节，相对 FP32 为 3.878788×、
相对 FP16 为 1.939394×；NVFP4 为 2,359,300 字节，分别为 7.111099× 和 3.555550×。
`compression_payload` 计入 scale 和 NV 全局 4 字节；`compression_file` 则比较实际磁盘
文件长度，计入张量的 32 字节头与权重的 96 字节头。很小或奇数列张量的头部/padding
会降低压缩率，不能套用渐近的 4×/8× 理想值。

### 7.2 缩放与数据分布的影响

NVFP4 分块能把有限异常值的影响限制在局部：正态输入 MAE 从逐张量的 0.118038 降为
0.0714289，异常值输入从 0.806947 降为 0.0820370，约改善 9.84 倍。
两个尺度虽然带来每元素 0.5 bit 的元数据成本，但对异常值数据提供了明显收益。

MXFP8 的 floor exponent + 有限饱和策略存在截断权衡。本次异常值输入中，分块的 MAE
0.0211448 大于逐张量的 0.0197277，不能声称更细块必然更精确。FP16 输入也可能在
恰好舍入到 2 的幂次处改变共享指数：uniform 输入逐张量最大值在 FP16 下可能成为 1，
改变指数选择，因而与 FP32 输入的误差不同。这是明确的缩放边界行为，不是 CPU/GPU 不一致。

### 7.3 性能取舍

NVFP4 减少 packed 数据读取，反量化通常比 MXFP8 更快；量化还需要全局 amax 与
局部 E4M3 scale 编码，因此本实验分块量化比 MXFP8 更慢。逐张量模式减少子组归约和
尺度元数据，部分内核更快，但 NVFP4 的异常值误差显著增加。
数据常驻 GPU 时，kernel 总时间低于 0.3 ms；一旦包含传输，约 4 ms 的流程由传输主导。
因此实际权重加载应尽量复用显存中的压缩权重，在需要时直接反量化到设备缓冲。

## 8. 分析工具与局限

### 8.1 Nsight Systems 实际分析

使用 Nsight Systems 2023.4.4，`--trace=cuda --sample=none --cpuctxsw=none`，
对正态 FP32、1024×4096、分块 NVFP4 采集，预热 5 次、kernel 重复 20 次。
Ubuntu 包的 importer 路径未被 CLI 自动找到，脚本显式调用 `QdstrmImporter` 完成导入，
并强制更新 SQLite 导出，避免使用上一次的缓存。完整记录见
[nsys_capture.txt](reports/nsys_capture.txt)、[nsys_import.txt](reports/nsys_import.txt) 和
[nsys_summary.csv](reports/nsys_summary.csv)。原始 `.nsys-rep` 保存在本地 `output/`，
可用 Nsight Systems 打开；复现命令为 `bash scripts/profile_wsl.sh`。

| CUDA API | 次数 | 主机 API 总时间 ms | 观察 |
|---|---:|---:|---|
| cudaMalloc | 6 | 122.031 | 最大单次 121.132 ms，冷启动/分配在该次运行中占据明显主机时间 |
| cudaMemcpy | 27 | 21.541 | 同步拷贝、等待及 host staging 的整体 API 时间 |
| cudaLaunchKernel | 120 | 6.864 | 中位数 5.311 µs；包含预热及传输流程中的额外发射 |
| cudaEventSynchronize | 2 | 6.042 | 两个独立 kernel 批次的终点同步 |

120 次发射与程序结构一致：25 轮三阶段量化 + 25 轮反量化 + 5 轮四阶段传输流程。
只有 6 次设备分配，确认计时循环内没有重复 cudaMalloc。分配调用的较大冷启动耗时
也说明必须明确区分首次启动与预分配后的稳态性能，不能将表 7 的速度称为整个冷启动进程加速比。

**本次 nsys 报告仅取得 CUDA API 记录；GPU kernel 和 GPU memory 报表明确显示无数据。**
因此上述 API 时间不替代 GPU 内核耗时，也不能直接解释为 DMA 时间或设备吞吐。
GPU 内核数值仍以第 6 节 CUDA Events 的非 profiler 实测为准。

### 8.2 其他工具限制

Nsight Compute 2024.1.1 尝试采集内核时间、DRAM 字节和 warp 占用率，
但当前 Windows 驱动未授予 GPU performance counter 权限，返回 `ERR_NVGPUCTRPERM`。
原始输出见 [ncu_nvfp4.csv](reports/ncu_nvfp4.csv)。没有据此推导 occupancy、DRAM 吞吐
或声称通过 ncu 验证了瓶颈。CUDA Events 的常规运行时间不依赖这些计数器，结果有效。

Compute Sanitizer 2024.1.1 返回 WDDM debugger interface 未启用，因此没有取得 GPU
memcheck/racecheck/synccheck/initcheck 的有效通过记录。复查脚本已保留，可在已配置
GPU 调试接口的设备上重跑。CPU 未定义行为检查和 GPU 独立数值核对属于不同的验证证据。

## 9. 后续优化方向

- 为标准对齐矩阵增加固定块长、固定 dtype 的向量化访存路径，减少 64 位除法和通用分支。
- 使用 pinned memory、异步拷贝与多流分块流水线，优化权重首次加载延迟。
- 在小于 L2 与显著大于 L2 的数据上分别测试，并在支持计数器的机器验证内存瓶颈。
- 对 MX 共享尺度提供无截断的 ceil 策略作为对照，记录精度和动态范围取舍。
- 在真实 Pascal/Turing 设备及国产平台验证；有新架构硬件时再加入原生路径对照。

本次提交基础实现不包含随机 Hadamard 旋转、训练配方、硬件 GEMM、国产平台或新硬件路径；
这些均不作为已实现功能计入结果。

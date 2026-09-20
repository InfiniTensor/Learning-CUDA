# MXFP8 / NVFP4 低精度软件模拟与反量化总结报告

## 1. 范围与结论

本实现只使用常规 CUDA kernel，在 GPU 上完成 scale 计算、编码、NVFP4 nibble
打包、解包和 FP16/BF16/FP32 反量化。基础路径没有使用 CUDA 原生 FP8/FP4 类型、
Tensor Core、Transformer Engine 或 cuDNN，因而不以 Hopper/Blackwell 为前提。
代码已为 SM75（T4）成功编译；本文实测机器为 NVIDIA GeForce RTX 5060 Ti，
CUDA 12.9，编译目标 SM120，但仍运行相同的软件模拟 kernel；基于原生硬件指令
与 cuBLASLt 的对照路径见第 9 节。

格式语义主要依据 OCP 的
[Microscaling Formats (MX) v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
和 NVIDIA Transformer Engine 的
[MXFP8](https://nvidia.github.io/TransformerEngine/features/low_precision_training/mxfp8/mxfp8.html)、
[NVFP4](https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html)
说明。

## 2. 数据表示

### 2.1 FP8 E4M3

元素为 1-bit sign、4-bit exponent、3-bit mantissa，指数 bias 为 7。非规格化数
按 `mantissa × 2^-9` 解码；规格化数按
`(-1)^sign × (1 + mantissa/8) × 2^(exponent-7)` 解码。`0x7e/0xfe`
为最大有限值 `±448`，`0x7f/0xff` 为 NaN。本实现量化有限溢出时饱和到
`±448`。

### 2.2 E8M0 scale

E8M0 只有 8-bit exponent，`byte=e` 解码为 `2^(e-127)`；`255` 保留为 NaN。
因此 MXFP8 的 scale 是 2 的整数次幂。全零分组使用 byte 0，元素仍编码为零，
不会产生除零。

### 2.3 FP4 E2M1 与 packed layout

E2M1 的正值 codebook 为：

| 低 3 bit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 值 | 0 | 0.5 | 1 | 1.5 | 2 | 3 | 4 | 6 |

最高 bit 为 sign，因此完整范围为 `{-6,-4,-3,-2,-1.5,-1,-0.5,-0,
0,0.5,1,1.5,2,3,4,6}`。每个 byte 严格保存两个元素：

```text
packed[i] = (q[2i] & 0x0f) | ((q[2i+1] & 0x0f) << 4)
```

即 row-major 线性下标的偶数元素位于低 nibble，奇数元素位于高 nibble。总字节
数为 `ceil(N/2)`；当 N 为奇数时，最后一个未使用的高 nibble 固定为 0。

## 3. 缩放、量化与反量化公式

令 `Q_E4M3`、`Q_E2M1` 分别表示向对应有限 codebook 取整并饱和，`amax(G)`
为分组 G 的绝对值最大值。

### 3.1 MXFP8

canonical block mode 中 `|G|=32`，本项目也支持配置其他 block size。解码
scale 采用向上取 2 的幂，避免最大元素因 scale 向下舍入而越界：

```text
s_G = 2 ^ ceil(log2(amax(G) / 448))
q_i = Q_E4M3(x_i / s_G)
x_hat_i = decode_E4M3(q_i) * s_G
```

`scale_mode=tensor` 时 `G` 是整个张量，磁盘中只有一个 E8M0 scale；block mode
则每 `block_size` 个连续 row-major 元素保存一个 scale。

### 3.2 NVFP4

NVFP4 使用两级 scale。首先由整张量的最大值计算 FP32 全局 scale：

```text
s_global = amax(tensor) / (448 * 6)
```

然后每个局部分组（canonical 大小 16）计算并保存一个 E4M3 scale：

```text
s_block = Q_E4M3((amax(G) / 6) / s_global)
q_i = Q_E2M1(x_i / (decode_E4M3(s_block) * s_global))
x_hat_i = decode_E2M1(q_i) * decode_E4M3(s_block) * s_global
```

全零张量令 `s_global=1`、局部 scale 和所有元素均为零。局部 E4M3 scale 用最近
偶数舍入；scale 量化后极少量边界溢出由 E2M1 饱和处理。tensor mode 仍保留
NVFP4 的两级结构，但全张量只使用一个 E4M3 局部 scale；这是为了完成参数文件
要求的 per-tensor scaling，canonical NVFP4 应使用 block mode。

### 3.3 舍入

`nearest` 在相邻 codepoint 距离相同时选择最低有效 mantissa bit 为 0 的编码。
`stochastic` 找到输入两侧相邻 codepoint，以线性距离作为选择上界的概率；每个
元素使用 `seed + linear_index` 经 SplitMix64 风格散列得到可复现随机数。Scale
始终采用上述确定性规则，随机舍入只作用于元素。

## 4. CUDA 实现

量化按以下 kernel 流水执行：

1. tensor-wide amax 用 grid-stride loop、warp shuffle 和 `atomicMax` 归约；
2. block amax 每个逻辑分组对应一个 CUDA block，用共享内存归约；
3. scale 立即编码为真实 E8M0/E4M3 byte，而不是保留 FP32 替代品；
4. MXFP8 每线程编码一个 byte；NVFP4 每线程读取两个元素并完成 packed store，
   从而没有两个线程更新同一 byte 的竞争；
5. 反量化采用 grid-stride loop，NVFP4 用 shift/mask packed load，再写目标类型。

E4M3 编码器在单调正 codebook 上二分查找相邻值，避免遍历全部 127 个正编码。
反量化只含位运算、少量整数判断和乘法，是主要优化路径。所有 launch API 接收
CUDA stream 且异步；CLI 用 CUDA event 计时，不包含分配、H2D/D2H、文件 I/O
和 CPU 误差统计。

有效带宽不是 profiler 的 DRAM 实测值，而是按算法最少字节量定义：

```text
quant_GB/s   = (input + packed + scales + global) / quant_time
dequant_GB/s = (packed + scales + global + output) / dequant_time
```

## 5. 文件格式

`quantized.lpq` 由固定 88-byte packed header、packed data、scale bytes 顺序组成。
header 保存 magic `LPQBIN1`、版本、格式、scale mode、rounding、输入 dtype、二维
shape、元素数、block size、scale 编码、两个 payload 长度和 FP32 global scale。
scale 编码 1 表示 E8M0，2 表示 E4M3。当前 interchange 文件按 little-endian
主机布局生成。

`dequantized.bin` 不带 header，完全由 row-major 输出元素组成。`metrics.json`
记录足够的信息来解释该文件。header 不参与 payload 压缩率，以免小张量被固定
元数据大小主导；NVFP4 的 4-byte global scale 参与计算。

以 FP32 输入和 canonical block size 计，理论/实测 payload 压缩率为：

```text
MXFP8: 4 / (1 + 1/32)       = 3.8788x
NVFP4: 4 / (1/2 + 1/16)     ≈ 7.1111x（另有每张量 4 bytes）
```

若输入为 FP16，基线字节数减半，相应压缩率也约减半。

## 6. 正确性与性能结果

测试矩阵为 `1024×1024 FP32`，输出 FP32，nearest rounding，canonical block
scaling；每项 warmup 20 次、计时 200 次。异常值分布主体为 `N(0,0.1)`，每
4093 个元素插入 `20/-16`。原始 JSON 位于 `results/`。

| 分布 | 格式 | max abs | MAE | MSE | 压缩率 | 量化 ms | 反量化 ms | 量化 GB/s | 反量化 GB/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | MXFP8 | 0.03125 | 0.010437 | 0.000187 | 3.8788 | 0.2717 | 0.0261 | 19.42 | 202.10 |
| uniform | NVFP4 | 0.16667 | 0.044310 | 0.003448 | 7.1111 | 0.8223 | 0.0776 | 5.82 | 61.67 |
| normal | MXFP8 | 0.24428 | 0.017959 | 0.000704 | 3.8788 | 0.2932 | 0.0201 | 17.99 | 262.09 |
| normal | NVFP4 | 0.58459 | 0.071372 | 0.009029 | 7.1111 | 0.8059 | 0.0439 | 5.94 | 108.91 |
| outlier | MXFP8 | 0.01562 | 0.001791 | 0.00000696 | 3.8788 | 0.2955 | 0.0181 | 17.85 | 291.17 |
| outlier | NVFP4 | 0.36814 | 0.007442 | 0.000135 | 7.1111 | 0.7792 | 0.0668 | 6.14 | 71.57 |

结果符合预期：MXFP8 以约一半于 NVFP4 的压缩收益换来明显更小误差。Block
scaling 把异常值影响限制在 16/32 元素内，所以含稀疏异常值数据的大多数普通值
仍保持很低 MAE。NVFP4 量化明显慢于反量化，主要来自 tensor amax、block amax、
E4M3 scale 编码和 E2M1 打包的多阶段开销，而反量化接近纯内存流操作。

CTest 的 12 组主要分布/格式/mode 组合全部通过；此外还通过了已知编码逐 byte
对比、全零张量、stochastic rounding 可复现性测试。CLI 测试验证了 FP16 输入、
BF16/FP16 输出和 35 元素 NVFP4 的 18-byte packed payload。

## 7. Nsight 与限制

仓库提供 `profile_ncu.sh`（限定一个 dequantize kernel launch）和
`profile_nsys.sh`（CUDA timeline）脚本。本机 WSL2 环境下 `ncu` 的性能
计数器被 Windows 驱动侧拒绝（`ERR_NVGPUCTRPERM`，root 下相同），无法
采样；`nsys` 可用，已完成的 timeline 分析见第 8 节。安装原生 Linux 版
Nsight Compute 后应重点检查：

- dequantize 的 global load/store throughput 和 load efficiency；
- NVFP4 packed load 是否形成连续 transaction；
- block scale kernel 的 occupancy、同步开销和尾 block 利用率；
- 多 kernel 之间的 launch gap，以及小张量是否应做融合。

当前实现是一维连续 block scaling，不包含 Transformer Engine 权重使用的可选
16×16 二维 scale layout。其他限制包括：
E4M3 编码二分查找仍偏计算密集；tensor amax 使用单个 global atomic 目标；磁盘
header 尚未做跨大端主机转换。原生 Blackwell Tensor Core 对照已在第 9 节补齐。

后续可增加 warp-specialized block reduction、查表/整数位级 E4M3 encoder、
scale+encode 融合、向量化 packed load/store、CUDA Graph 小张量路径，并在有
Blackwell 的环境中与 cuDNN block scaling、CUDA Math API 或 Transformer Engine
逐编码和逐性能对照。国产平台可在保持文件格式与 host API 不变的前提下替换
kernel runtime 层。

## 8. nsys profiling 分析

本节数据来自本机真实采集，profiler 为 Nsight Systems 2025.6.3
（`~/.local/cuda-13.2/bin/nsys`）。为让 kernel 时间信号更干净，输入改用
4096×4096 FP32 normal 分布（`tests/generate_inputs.py --rows 4096 --cols 4096`
生成，保存在 `results/nsys_fixture_4096/`），每种格式单独 profile，warmup 3
次、计时 20 次：

```bash
nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true \
  -o results/nsys/mxfp8_4096 \
  ./build/low_precision_cli --input results/nsys_fixture_4096/normal_fp32.bin \
  --config results/nsys_fixture_4096/mxfp8_fp32.toml \
  --output-dir results/nsys/mxfp8_run --warmup 3 --iters 20
# NVFP4 同理，报告为 results/nsys/nvfp4_4096.nsys-rep
```

`nsys stats` 完整输出见 `results/nsys/nsys_kern_sum_mxfp8_sm120.txt` 和
`results/nsys/nsys_kern_sum_nvfp4_sm120.txt`；下同的实例数 23 = 3 warmup +
20 计时，平均值包含 warmup，首次迭代偏慢时以中位数为准。

### MXFP8（cuda_gpu_kern_sum，每 kernel 23 个实例）

| kernel | 占比 | 平均 | 中位 |
|---|---:|---:|---:|
| `mx_block_scale_kernel` | 75.2% | 2657.8 µs | 2581.5 µs |
| `mx_quantize_kernel` | 14.8% | 521.9 µs | 521.6 µs |
| `dequantize_kernel` | 10.0% | 354.7 µs | 308.4 µs |
| `finish_global_scale_kernel` | 0.0% | 0.44 µs | 0.45 µs |

量化路径三个 kernel 合计约 3.180 ms，对同次运行 CLI 的 CUDA event 计时
`quant_kernel_ms=3.196 ms`（`results/nsys/mxfp8_run/metrics.json`）；反量化
中位 308.4 µs 对 event 的 313.5 µs。两侧差异在 2% 以内：kernel 已到毫秒
量级，launch 开销可以忽略，event 与 timeline 相互印证。

### NVFP4（每 kernel 23 个实例）

| kernel | 占比 | 平均 | 中位 |
|---|---:|---:|---:|
| `nv_block_scale_kernel` | 65.6% | 8701.4 µs | 7964.7 µs |
| `dequantize_kernel` | 17.9% | 2378.5 µs | 1962.0 µs |
| `nv_quantize_kernel` | 15.2% | 2015.2 µs | 1790.3 µs |
| `tensor_amax_kernel` | 1.3% | 177.6 µs | 173.7 µs |
| `finish_global_scale_kernel` | 0.0% | 0.62 µs | 0.64 µs |

量化四段合计约 10.895 ms，对 event 的 `quant_kernel_ms=10.943 ms`；反量化
平均 2378.5 µs 对 event 的 2371.9 µs，同样吻合（差异 <0.5%）。

### 结论

- block scale kernel 是两条路径量化阶段的瓶颈：MXFP8 中约占 84%
  （2657.8/3180），NVFP4 中约占 80%（8701.4/10895），验证了第 6 节关于
  NVFP4 多阶段开销主要来自 scale 流水的判断；后续优化（归约方式、
  scale+encode 融合）应优先针对该 kernel。
- tensor amax 只占 NVFP4 量化的约 1.6%（177.6 µs），单一 global atomic
  目标在当前尺寸下不是瓶颈。
- 4096×4096 下 NVFP4 反量化（2.37 ms）显著慢于 MXFP8 反量化
  （0.31 ms），差距来自 packed nibble 的逐元素解包和两级 scale 乘法，
  与第 6 节 1024×1024 结果的排序一致。
- 曾尝试 `scripts/profile_ncu.sh` 采集 DRAM 吞吐与 load efficiency 等
  计数器，但 WSL2 下性能计数器被 Windows 驱动侧拒绝
  （`ERR_NVGPUCTRPERM`，root 相同），ncu 无法采样；本节以 nsys
  timeline 代替，第 7 节列出的计数器指标留待原生 Linux 环境补测。

## 9. 硬件路径对照（加分项）

基础验收路径（第 1–8 节）全部为**纯软件模拟**：只使用常规 CUDA core 和位操作，
不依赖任何特定 GPU 架构或第三方库。本节的三条对照路径均为可选实现，通过
`make WITH_HW_PATH=1`（CMake 选项 `WITH_HW_PATH`）打开；默认关闭时构建产物
与基础版完全一致，CLI 忽略 `hardware_path` 配置键，第 6 节的所有结果不受影响。
本节所有数字来自真实运行，可在 `results/hw_path/` 用
`./scripts/run_hw_benchmarks.sh` 复现（1024² 与 4096² FP32 uniform 输入，
warmup 20、计时 200 次，与第 6 节口径一致）。

### 9.1 实现方式与依赖标注

| 路径 | 实现 | 依赖 |
|---|---|---|
| cvt | `src/hw_path.cu` 的 `mx_encode_cvt_kernel` / `mx_decode_cvt_kernel`，用 `cuda_fp8.h` 的 `__nv_cvt_float_to_fp8`（`cvt.rn.satfinite.e4m3`）/ `__nv_cvt_fp8_to_halfraw` 做 MXFP8 元素编解码 | **依赖 GPU 架构**：cvt 指令要求 sm_89+（实测 sm_120）；不依赖第三方库 |
| gemm（MXFP8） | cuBLASLt 原生 FP8 GEMM（`CUDA_R_8F_E4M3` + `CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0` 块缩放），对照软件「反量化到 FP16 + `cublasGemmEx`」流水线 | **依赖第三方库** cuBLAS 12.9（pip nvidia-cublas 包）+ **依赖 Blackwell 低精度 Tensor Core** |
| gemm（NVFP4） | cuBLASLt 原生 FP4 GEMM（`CUDA_R_4F_E2M1` + `VEC16_UE4M3` 块缩放），FP32 全局 scale 折进 alpha | 同上 |

GEMM 对照模拟推理线性层 Y = X·Wᵀ（M=N=K=side，X/W 共用同一输入矩阵，K 连续
存储，恰好匹配 FP8/FP4 kernel 要求的 TN 布局）。量化本身仍由软件路径完成，两条
路径共用同一份量化结果，比较的只是「反量化 + FP16 GEMM」与「原生低精度 GEMM」
的计算段；scale 重排为一次性 setup，不计入计时。

### 9.2 cvt 指令路径：元素级编解码对照

软件路径的 E4M3 编码（二分查找 + RNE + SATFINITE 饱和）语义与硬件
`cvt.rn.satfinite.e4m3` 指令完全一致，两种尺寸下逐字节比对结果均为
`encode_byte_mismatches=0`、`decode_bit_mismatches=0`（解码输出按 FP32 位级
比对）。唯一已知语义差异是硬件指令保留 -0.0 的符号位（编码为 0x80），软件
路径把 -0.0 视为非负数（0x00）；两者解码后数值相等，硬件 kernel 内已把 -0.0
归一化为 +0.0 以保持逐字节一致。

| 尺寸 | kernel | 软件 ms | 硬件 ms | 软件 GB/s | 硬件 GB/s | 加速比 |
|---|---|---:|---:|---:|---:|---:|
| 1024² | 编码 | 0.0388 | 0.0147 | 135.9 | 358.6 | 2.64x |
| 1024² | 解码 | 0.0150 | 0.0140 | 352.9 | 377.1 | 1.07x |
| 4096² | 编码 | 0.6187 | 0.3486 | 136.4 | 242.1 | 1.77x |
| 4096² | 解码 | 0.3453 | 0.2951 | 244.4 | 286.0 | 1.17x |

编码加速明显（二分查找换成单条指令）；解码两侧的算术都很少、接近内存带宽
瓶颈，加速比相应较小。4096² 下绝对带宽低于 1024²，因为采集时机器上有常驻
llama-server 占用约 13GB 显存并带来偶发干扰（绝对数字受影响，同条件两侧的
相对加速比仍有效）。注意元素编码只是完整量化流水的一段：4096² 下软件完整
量化 4.57 ms（`gemm_mxfp8_4096/metrics.json`），其中 block scale kernel 占约
84%（第 8 节），元素编码本身占比很小，因此 cvt 路径对端到端量化时间的改善
有限——它的意义在于验证了软件编码与硬件指令的位级等价性。

### 9.3 GEMM 路径：原生低精度 Tensor Core 对照

| 格式 | 尺寸 | 原生 GEMM ms | 原生 TFLOPS | FP16 GEMM ms | FP16 TFLOPS | 软件反量化 ms | 流水线加速比 |
|---|---|---:|---:|---:|---:|---:|---:|
| MXFP8 | 1024³ | 0.0232 | 92.4 | 0.0764 | 28.1 | 0.0363 | 4.85x |
| MXFP8 | 4096³ | 1.0865 | 126.5 | 3.4327 | 40.0 | 0.8873 | 3.98x |
| NVFP4 | 1024³ | 0.0164 | 130.7 | 0.0577 | 37.2 | 0.0928 | 9.16x |
| NVFP4 | 4096³ | 0.5096 | 269.7 | 3.4573 | 39.8 | 1.1930 | 9.12x |

「流水线加速比」=（软件反量化 X 和 W + FP16 GEMM）/ 原生低精度 GEMM。
数值正确性：`hw_gemm.json` 中 `self_check_*_rel_error` 为 256³ 上对 CPU
double 参考的相对误差，MXFP8 两条路径同为 2.1e-4（MXFP8 反量化值在 FP16 中
精确可表示，软件 FP16 GEMM 与原生 FP8 GEMM 只差累加顺序）；NVFP4 原生路径
（2.0e-4）反而略优于软件路径（2.6e-4），因为 FP4 GEMM 的 FP32 全局 scale 在
FP32 累加后才乘入，避免了中间 FP16 舍入。基准尺寸上两条路径输出的最大相对
差 ≤ 7.0e-4。

结果直观展示了推理场景下原生低精度 GEMM 的价值：FP8 Tensor Core 峰值约为
FP16 的 3.2 倍（126.5 vs 40.0 TFLOPS），FP4 再翻一倍（269.7 TFLOPS），同时省
掉反量化 kernel 和 FP16 中间张量的显存流量。NVFP4 流水线加速比（约 9x）高于
GEMM 本身加速比（4096³ 下 6.8x），因为软件路径的 NVFP4 反量化（nibble 解包 +
两级 scale）比 MXFP8 贵得多。

### 9.4 遇到的问题与调研结论

- **cuBLAS 块缩放的 scale 布局不是自然的 row-major**：文档要求
  128(outer)×4(inner) tile 布局（cuBLAS 文档 "1D Block Scaling Factors
  Layout"）。最初直接传入自然布局 scale，256³ 自检相对误差 0.14；按文档公式
  重排后降到 2.1e-4。重排在 host 侧一次性完成（`swizzle_scales`），不计入
  GEMM 计时。
- **布局约束**：sm_120 上 cuBLAS 12.9 的 FP8/FP4 块缩放 GEMM 只接受 K 连续的
  TN 布局（A 转置、B 不转置）；NN/NT/TT 组合的 `cublasLtMatmulAlgoGetHeuristic`
  返回 `NOT_SUPPORTED`。本对照的 Y = X·Wᵀ 形态天然满足。
- **NVFP4 可行性结论**：cuBLAS 12.9 头文件（`library_types.h`）已暴露
  `CUDA_R_4F_E2M1`，且 sm_120 上 FP4 E2M1 + VEC16_UE4M3 块缩放 GEMM 实测可用
  （上面的数字即实测），不需要回退或借助 CUDA 13.2 的安装。
- 探测程序与校准程序为一次性工具，未进仓库；等价性与数值自检逻辑已固化在
  `src/hw_path.cu` 的 `run_gemm_comparison` 中，每次 `hardware_path=gemm`
  运行都会重新校验并写入 JSON。

# Hadamard 变换加速项目报告

## 问题背景

低比特量化对激活中的异常值非常敏感。量化前加入正交旋转，例如 Hadamard
变换，可以把异常值能量扩散到多个通道，从而降低单个通道的动态范围，改善
FP8/INT4 量化精度。这个方向是 QuaRot、SpinQuant、FlashAttention-3 等工作的
关键思想之一。

本项目实现一个 CUDA Hadamard 变换程序，输入张量形状为：

```text
[batch_size, seq_len, num_heads, head_dim]
```

Hadamard 矩阵大小为 `head_dim`，要求为 2 的幂。当前实现支持并测试了
`64`、`128`、`256`。

## 核心实现

主 kernel 使用归一化快速 Hadamard 变换：

```text
y = Hx / sqrt(D)
```

其中 `D = head_dim`。每个 CUDA block 负责一个展平后的 `(batch, seq, head)`
行向量，将该向量读入 shared memory，然后按标准蝶形结构迭代：

```text
(a, b) -> (a + b, a - b)
```

这样每行只需要 `D log2(D)` 次加减法，不需要显式存储稠密 Hadamard 矩阵。

支持的数据类型：

- FP16：使用 CUDA `half` 存储和读写。
- BF16：使用 `uint16_t` 保存 BF16 位模式，并通过软件完成 FP32 与 BF16 的转换。

BF16 路径不依赖原生 BF16 Tensor Core，因此可作为普通 NVIDIA CUDA GPU 上的可移植
基础实现。

## 融合量化

项目实现了两条融合路径。

INT4 路径在一个 kernel 中完成 Hadamard 变换和逐行对称量化。输出为真实 packed
INT4 格式，每个字节保存两个 signed 4-bit 元素。量化公式为：

```text
scale = max(abs(y)) / 7
q = clamp(round(y / scale), -8, 7)
```

E4M3 路径在 Hadamard 变换后直接把结果编码为软件 FP8 E4M3 字节。

为了满足题目中“融合量化结果与先变换后量化一致”的要求，fused kernel 在量化前会
模拟对应输出 dtype 的舍入：FP16 路径按 `half` 舍入，BF16 路径按 BF16 位模式舍入。
测试中会逐字节比较 fused 输出与分离路径输出。

## Tensor Core 对照实现

对于 FP16，本项目额外实现了 WMMA/Tensor Core 对照路径：

```text
Y = X * H_scaled
```

其中：

```text
H_scaled = H / sqrt(D)
```

该路径展示了 Tensor Core 的使用方式，但它需要显式构造 Hadamard 矩阵，并且每行
计算复杂度为 `O(D^2)`。对于 `head_dim <= 256` 的逐 head 向量变换，shared-memory
FHT 的 `O(D log D)` 更有算法优势，因此主实现仍采用 FHT。

## 正确性验证

测试程序使用 CPU FHT 作为参考实现，并与 GPU 输出比较：

- FP16：最大绝对误差 `< 1e-2`
- BF16：最大绝对误差 `< 5e-2`
- fused INT4：packed bytes 和 scales 与“先 Hadamard、再 INT4 量化”一致
- fused E4M3：编码字节与“先 Hadamard、再 E4M3 编码”一致

运行方式：

```powershell
.\build\Release\hadamard_tests.exe
```

或在 Linux 上直接运行：

```bash
./hadamard_tests
```

## 性能测试

benchmark 程序统计平均 kernel 时间和有效带宽：

```powershell
.\build\Release\hadamard_bench.exe --batch 8 --seq 1024 --heads 16 --dim 128 --dtype fp16 --repeats 100 --log perf.csv
```

输出包括：

- shared FHT 时间
- fused INT4 时间
- fused E4M3 时间
- FP16 WMMA 时间

已在 RTX 4090 D（`sm_89`，CUDA 12.8）上验证，测试参数为
`batch=2, seq=512, heads=16, head_dim=128, repeats=50`。profiling 构建使用
`nvcc -O3 -lineinfo -arch=sm_89`，正确性测试仍然全部通过：

```text
Hadamard 正确性测试和融合量化一致性测试全部通过。
```

profiling 构建下的一次 benchmark 结果如下：

| Kernel | DType | 时间（ms） | 有效带宽（GB/s） |
| --- | --- | ---: | ---: |
| shared_fht | FP16 | 0.0219034 | 382.983 |
| fused_int4 | FP16 | 0.0322765 | 164.467 |
| fused_e4m3 | FP16 | 0.0188205 | 334.288 |
| wmma_fp16 | FP16 | 0.0402227 | 209.369 |
| shared_fht | BF16 | 0.0208454 | 402.419 |
| fused_int4 | BF16 | 0.0322355 | 164.676 |
| fused_e4m3 | BF16 | 0.0188819 | 333.200 |

结果会同时追加到 CSV 日志，方便后续比较不同形状、不同 dtype 和不同 kernel。

## CPU 基线和加速比

项目新增 `hadamard_cpu_bench`，使用与 CPU reference 相同的 `O(D log D)` FHT
算法。运行示例：

```bash
./hadamard_cpu_bench --rows 16384 --dim 128 --repeats 10
```

GPU benchmark 和 CPU benchmark 均只统计核心计算时间，不包含首次 Host-to-Device
数据拷贝。这样可以单独分析 kernel 加速；若需要端到端结果，应另行报告数据传输
时间。在 RTX 4090 D 上，CPU 实测结果为：

```text
cpu_fht_ms=6.431319 effective_gb_s=2.608674
```

与 profiling 构建下的 FP16 shared FHT `0.0219034 ms` 对比，核心计算加速比约为
`293.6x`。该比较用于说明 GPU 并行收益；CPU 基线使用 FP32 工作数组，而 GPU
输入/输出为 FP16，GPU kernel 内部使用 FP32 shared memory 计算，因此报告中不把
这个数字宣称为严格同 dtype 端到端加速比。

## Nsight 分析与 Tensor Core 证据

项目包含 WMMA/Tensor Core 对照路径。源代码使用
`wmma::mma_sync(c_frag, a_frag, b_frag, c_frag)`，在 RTX 4090 D 上通过
`cuobjdump --dump-sass` 得到：

```text
Function: hadamard_wmma_fp16_kernel
HMMA.16816.F32 R8, R12.reuse, R16, R8
HMMA.16816.F32 R4, R12, R18, R4
```

因此可以确认 WMMA 路径实际生成并执行了 Tensor Core 的 `HMMA` 指令。需要注意：
主路径 `hadamard_shared_kernel` 使用 `O(D log D)` 的 shared-memory 蝶形算法，
不使用 Tensor Core；WMMA 仅作为进阶对照路径。

Nsight Systems 已成功采集 CUDA kernel 时间线。参数为
`batch=2, seq=512, heads=16, head_dim=128, repeats=5`，kernel 汇总为：

| Kernel | 调用次数 | 平均时间（ns） | 总时间占比 |
| --- | ---: | ---: | ---: |
| `hadamard_wmma_fp16_kernel` | 5 | 40262.6 | 36.9% |
| `fused_hadamard_int4_kernel2` | 5 | 31648.2 | 29.0% |
| `hadamard_shared_kernel` | 5 | 18694.6 | 17.2% |
| `fused_hadamard_e4m3_kernel` | 5 | 18374.6 | 16.9% |

该结果说明：Tensor Core 虽然被 WMMA 路径使用，但显式矩阵乘法的 `O(D^2)` 计算
量和矩阵装载开销抵消了其吞吐优势；对于本题的逐 head 向量变换，FHT 主路径更合适。
fused E4M3 路径省去了中间 Hadamard 张量写回，因此在当前测试形状下最快。

Nsight Compute 也已实际尝试，命令和结果见 `PROFILING.md`。当前远端容器返回：

```text
ERR_NVGPUCTRPERM
The user does not have permission to access NVIDIA GPU Performance Counters
```

因此本次环境无法读取 NCU 的 occupancy、Tensor Core utilization 等硬件计数器。
这是容器/驱动的 performance counter 权限限制，并非 kernel 编译或运行失败。项目
保留了 `scripts/profile_linux.sh`，在开放 NCU 权限的环境中可以直接补采。

## 沐曦曦云 C500 适配与实测

为了支持国产 GPU，项目新增独立的 MACA 源文件，而不是让 C500 直接复用包含
NVIDIA WMMA 对照代码的 `.cu` 主文件。适配文件如下：

- `src/hadamard_maca.maca`：C500 的共享内存 FHT、融合 INT4 和融合 E4M3。
- `src/hadamard_bench_maca.maca`：C500 性能基准入口。
- `tests/test_hadamard_maca.maca`：C500 正确性测试入口。
- `scripts/build_maca.sh`：MXCC 构建和运行时链接脚本。

C500 的 CUDA 兼容运行时会把 CUDA API 映射为 `wcuda*` 符号。只链接
`libmcruntime.so` 不足以完成最终链接，脚本和 CMake 已显式加入：

- `libruntime_cu.so`
- `libsymbol_cu.so`
- `libmcruntime.so`
- `libmxc-runtime64.so`

这样既保留了 CUDA 兼容 API 的公共接口，又避免将 NVIDIA WMMA 头文件和
Tensor Core 对照 kernel 带入 C500 构建。

在沐曦曦云 C500 主机上，以下测试已经完成：

```text
GPU: MetaX C500 sm_100
Hadamard 正确性测试和融合量化一致性测试全部通过。
```

测试覆盖 FP16/BF16、`head_dim = 64/128/256`、归一化 FHT、融合 INT4 的 packed
字节和 scale 一致性，以及融合 E4M3 的逐字节一致性。

C500 实测形状为 `batch=2, seq=512, heads=16, head_dim=128, repeats=50`：

| 路径 | 数据类型 | 时间（ms） | 有效带宽（GB/s） |
| --- | --- | ---: | ---: |
| shared FHT | FP16 | 0.0598118 | 140.250 |
| fused INT4 | FP16 | 0.0754278 | 70.3774 |
| fused E4M3 | FP16 | 0.0431053 | 145.956 |
| shared FHT | BF16 | 0.0609280 | 137.681 |
| fused INT4 | BF16 | 0.0749773 | 70.8003 |
| fused E4M3 | BF16 | 0.0435098 | 144.599 |

上述时间是设备端 kernel 平均时间，不包含首次 Host-to-Device 拷贝。原始 CSV
日志保存在 `profiling/maca_perf_fp16.csv` 和 `profiling/maca_perf_bf16.csv`。
C500 版本明确不宣称使用 NVIDIA Tensor Core；NVIDIA RTX 4090 D 上的 WMMA
对照路径仍由 `cuobjdump` 的 `HMMA.16816.F32` 证据单独验证。

## 摩尔线程 MTT S4000 适配与实测

项目新增独立的 MUSA `.mu` 后端，不直接复用包含 CUDA 运行时和 NVIDIA WMMA
的 `.cu` 文件：

- `include/hadamard_musa.muh`：MUSA 公共接口。
- `src/hadamard_musa.mu`：MUSA FP16/BF16 FHT 和两条融合量化路径。
- `src/hadamard_bench_musa.mu`：MUSA 性能基准。
- `tests/test_hadamard_musa.mu`：MUSA 正确性测试。
- `scripts/build_musa.sh`：MUSA 4.3 构建脚本。

MUSA 后端使用 `musaMalloc`、`musaMemcpy`、`musaEvent` 等原生运行时接口，
编译命令使用 `mcc -mtgpu -O3 -lmusart`。FP16 和 BF16 在 kernel 内使用位级
转换，避免依赖 NVIDIA 的 `half` 类型和 WMMA 接口。

在 MTT S4000 主机上完成的验证结果：

```text
GPU: MTT S4000 compute_capability=2.2
所有 MUSA Hadamard 正确性和融合量化测试通过。
```

测试覆盖 FP16/BF16、`head_dim = 64/128/256`、归一化 FHT、融合 INT4 packed
字节和 scale 一致性，以及融合 E4M3 逐字节一致性。

MTT S4000 实测形状为 `batch=2, seq=512, heads=16, head_dim=128, repeats=50`：

| 路径 | 数据类型 | 时间（ms） | 有效带宽（GB/s） |
| --- | --- | ---: | ---: |
| shared FHT | FP16 | 0.0948366 | 88.4533 |
| fused INT4 | FP16 | 0.172274 | 30.8137 |
| fused E4M3 | FP16 | 0.130696 | 48.1380 |
| shared FHT | BF16 | 0.0718725 | 116.715 |
| fused INT4 | BF16 | 0.137085 | 38.7236 |
| fused E4M3 | BF16 | 0.0962757 | 65.3484 |

上述数据是设备端 kernel 平均时间，不包含首次 Host-to-Device 拷贝。MUSA
版本不运行 NVIDIA WMMA；Tensor Core 证据仅适用于 NVIDIA 独立对照路径。

## 项目提交材料

- `CMakeLists.txt`：CUDA/C++ 构建和 CTest 配置。
- `include/hadamard_cuda.cuh`：公开接口。
- `src/hadamard_cuda.cu`：CUDA 主实现、fused 量化和 WMMA 对照路径。
- `src/hadamard_bench.cu`：GPU 性能测试。
- `src/hadamard_maca.maca`：沐曦 C500 主实现。
- `src/hadamard_bench_maca.maca`：沐曦 C500 性能测试入口。
- `include/hadamard_musa.muh`：摩尔线程 MUSA 公共接口。
- `src/hadamard_musa.mu`：摩尔线程 MTT S4000 主实现。
- `src/hadamard_bench_musa.mu`：摩尔线程 MTT S4000 性能测试入口。
- `src/hadamard_cpu_bench.cpp`：CPU 性能基线。
- `tests/test_hadamard.cu`：FP16/BF16 正确性、融合量化一致性和 WMMA 测试。
- `tests/test_hadamard_maca.maca`：沐曦 C500 正确性测试入口。
- `tests/test_hadamard_musa.mu`：摩尔线程 MTT S4000 正确性测试入口。
- `scripts/build_maca.sh`：沐曦 C500 一键构建脚本。
- `scripts/build_musa.sh`：摩尔线程 MTT S4000 一键构建脚本。
- `scripts/profile_maca.sh`：沐曦 C500 正确性和性能日志脚本。
- `scripts/profile_musa.sh`：摩尔线程 MTT S4000 正确性和性能日志脚本。
- `scripts/profile_linux.sh`：一键 profiling 脚本。
- `PROFILING.md`：Tensor Core、Nsight Systems、Nsight Compute 分析记录。
- `README.md`：构建和运行说明。
- `REPORT.md`：项目总结报告。

## 后续优化方向

- 为 `head_dim <= 64` 增加 warp-level 特化版本，减少 shared memory 同步开销。
- 使用向量化 load/store 优化 FP16/BF16 访存。
- 将 Hadamard 直接融合进 attention 或 MLP 的量化入口，减少中间张量读写。
- 在 Hopper/Blackwell 上增加原生 FP8 路径，同时保留当前软件模拟作为可移植基线。
- 使用 Nsight Compute / Nsight Systems 分析 occupancy、访存效率和同步开销。

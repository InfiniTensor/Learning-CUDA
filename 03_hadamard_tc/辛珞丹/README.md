# FWHT-CUDA：Hadamard 变换的 CUDA 加速

2026 夏季训练营 CUDA 方向，选题三「Hadamard 变换加速」。

这是一个给 LLM 量化前旋转（QuaRot / SpinQuant 那一类用法）准备的快速 Walsh–Hadamard 变换库。
输入是 `[batch, seq_len, num_heads, head_dim]` 的连续张量，按 `[rows, dim]` 处理，`dim` 为 2 的幂，
支持 FP16 / BF16（也支持 FP32）。里面有三套内核（寄存器 / 共享内存 / Tensor Core），一个把旋转和
FP8 / INT4 量化合在一起的融合内核，以及做实验用的命令行工具。

同一份源码在 NVIDIA（`nvcc`）和摩尔线程 MTT S4000（MUSA 3.1，`mcc`）上都能编译运行，
平台差异只放在 `include/fhwt/backend.cuh` 一个文件里。

完整的设计、优化过程和实验数据在 [`总结报告.md`](总结报告.md)，这份 README 只是入口。
报告与选题书对总结报告的要求是这样对应的：**实现思路与优化方法**在第 3、4 节（第 5 节是
优化历程，10 个小节，含一个被自己推翻的假设与两次被证伪的测量）；**最终性能指标与分析**
在第 7、8、9 节；**未来可继续提升的地方**在第 10 节；**nsys / ncu 的采集与分析**在第 8 节，
原始输出在 `docs/profile/` 与 `docs/profile/ncu/`。

## 构建与运行

```bash
# Linux
make -j            # -> build/fhwt.exe
make check         # 正确性自检（见下）
make bench         # 跑全部实验，结果写到 data/
make report        # data/*.csv -> docs/结果表格.md
```

```powershell
# Windows（我的主力开发机，通过 vcvars64.bat 调 nvcc）
powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -Target all
build\fhwt.exe --mode verify --dim 256 --rows 4096
powershell -ExecutionPolicy Bypass -File scripts\run_experiments.ps1
```

```bash
# 摩尔线程 MUSA 3.1
bash scripts/build_musa.sh all
bash scripts/run_experiments_musa.sh            # -> data/musa/
```

依赖：CUDA 12.x 和支持 C++17 的主机编译器。Tensor Core 路径需要 sm_80 以上，硬件 FP8 编码指令
需要 sm_89 以上（更低的架构自动走软件编码）。没有开 `--use_fast_math`，因为它会改变 IEEE 语义，
误差数字就没法和参考实现比了。

## 对照选题要求

| 选题要求 | 做到的程度 | 在哪看 |
|---|---|---|
| 快速 Hadamard 变换核，支持多种尺寸 | dim 32–16384，三套内核 | `include/fhwt/kernel_*.cuh`，报告第 3 节 |
| 输入 FP16 / BF16，head_dim 为 2 的幂 | 都支持，另支持 FP32 | `fhwt.h` |
| 输出核函数执行时间（ms） | `--mode bench / matrix / quant` 都输出 ms，CSV 在 `data/` | `docs/结果表格.md` |
| FP16 误差 < 1e-2，BF16 < 5e-2（对比 fast_hadamard_transform） | FP16 最大 3.9e-3，BF16 最大 3.1e-2，80 项比对全部通过 | `data/ref_check.log`，报告 7.1 |
| 融合量化，且结果与"先变换后量化"一致 | FP8 与 INT4，逐位一致 | `fhwt --mode quant` 每次都会比对，报告 7.7 |
| 使用 Tensor Core，并与不用 TC 的实现对比 | 做了，结论是带宽受限时两者持平 | 报告 7.3、8.4 |
| 国产平台（加分） | 摩尔线程 MTT S4000 实测 | 报告 7.15，`docs/国产平台适配.md` |
| 报告，ncu / nsys（加分） | 两种都用了 | 报告第 5、7、8 节，`docs/profile/` |

关于测试代码：通用要求第 1 条写的是"无测试代码"，选题三的提交内容又写"包含测试"。这份提交按
通用要求没有放 `tests/`，但程序自己带了自检：

- `fhwt --mode verify`：内核结果和 double 精度的 CPU 参考逐元素比，按选题的阈值判 PASS / FAIL；
- `fhwt --mode quant`：计时之后把融合结果和"先旋转再量化"的结果逐字节比较，打印 PASS / FAIL；
- `make check` 就是把这两种检查跑一遍（7 条 verify + 5 条 quant），最近一次的输出在 `data/check.log`。

开发时用的 5 个测试程序（最后一轮 1088 项，0 失败）的输出留在 `data/tests.log`、`data/tests_sm89.log`。

## 目录

```
include/fhwt/   内核与公共头文件
  fhwt.h          对外接口：hadamard / hadamard_quant / hadamard_gemm / Graph / GraphSequence
  kernel_reg.cuh  主内核：寄存器 + warp shuffle（dim 32–1024）
  kernel_smem.cuh 大 dim：寄存器 + 共享内存两级（dim 64–16384）
  kernel_tc.cuh   Tensor Core：块对角 radix-16，mma.m16n8k16
  kernel_quant.cuh / quant_pack.cuh  融合"旋转 + 量化"与打包存储
  kernel_gemm.cuh 把旋转折进 GEMM 的 A-tile 载入
  backend.cuh     CUDA / MUSA 的名字映射
src/            ops.cu（发射配置、图、设备信息），main.cu（命令行）
scripts/        构建、跑实验、采集 nsys / ncu
tools/          基线对比、和参考实现逐元素比对、出表、量化误差仿真等 Python 脚本
data/           NVIDIA 上的实测数据；data/musa/ 是摩尔线程上的
总结报告.md     总结报告正文（放在根目录，打开即读）
docs/           调研笔记、结果表格、国产平台适配、数据索引、profile/（nsys 与 ncu 原始输出）
```

`docs/数据索引.md` 列了 `data/` 下每个文件是哪条命令跑出来的、被报告哪一节引用。

## 做了什么

**寄存器内核（主路径）。** 每行由 TPR 个线程（2 的幂，≤ 32）负责，每个线程拿连续的 dim/TPR 个元素。
蝶形的低位在线程内的寄存器里做，高位用 `__shfl_xor_sync` 在线程间做，全程不用共享内存，也就没有
bank conflict 和 `__syncthreads()`。FP16 / BF16 两个元素打包在一个 32 位寄存器里，一次蝶形是
一条 PRMT 加一条 HFMA2。dim 小的时候一个 block 放多行，避免 dim = 64 时一个 block 只有 8 个线程。

**大 dim。** 超过 1024 就放不进一个 warp 的寄存器了，先在寄存器里做低 8 位，剩下的在共享内存里做，
索引映射保证同一个 warp 的访问连续、没有 bank conflict。

**Tensor Core。** 稠密矩阵乘做 Hadamard 太浪费（dim = 256 时计算量是蝶形的 64 倍），能用上 TC 的
只有块对角结构：`H_d = H_16 ⊗ … ⊗ H_16 ⊗ H_{2^r}`，每个 16 点变换正好是一条 `mma.m16n8k16`，
剩下的 2/4/8 点在寄存器里做，所以任意 2 的幂都能处理。

**融合量化。** 旋转完直接在寄存器里量化成 FP8(E4M3) 或 INT4 写回，中间结果不回显存。
每元素的显存访问从 7 B（FP8）/ 6.5 B（INT4）降到 3 B / 2.5 B。量化之前先把旋转结果舍入回
FP16/BF16，这样和"先写回显存再读出来量化"的结果逐位相同，而不只是近似。

**旋转折进 GEMM。** QuaRot 里旋转后的激活马上要进矩阵乘，所以在 GEMM 读 A 的 tile 时顺手做旋转，
旋转后的矩阵根本不写回显存，结果和先旋转再乘逐位相同。

**减少发射开销。** 小 shape 上测出来的其实主要是 CPU 提交内核的开销，所以做了
`Graph` / `GraphSequence`，把一整趟 forward 里的旋转录进一张 CUDA Graph 一次提交。

## 主要结果

主测机 RTX 5060 Ti（sm_120，36 SM，L2 32 MiB，CUDA 12.9，Windows）。显存实测拷贝上限约 380 GB/s。

| 项目 | 结果 |
|---|---|
| 寄存器内核，数据大于 L2 时 | 378–392 GB/s，基本等于实测拷贝上限 |
| 寄存器内核，数据能放进 L2 时 | 最高 1658 GB/s |
| 对比 fast_hadamard_transform | 大数据量持平（1.00x），小 dim 放得进 L2 时最高 6.45x |
| Tensor Core 对比寄存器内核 | 大数据量 0.99–1.02x；放得进 L2 时 0.34–0.86x（更慢） |
| 融合量化（dim 256） | FP8 2.39x，INT4 2.64x，和理论访存比 2.33x / 2.6x 基本对得上 |
| 原地变换 | 只有工作集正好等于 L2 容量时有用，2.12–2.18x |
| CUDA Graph | 小 shape 5–29x，大 shape 1.00x |
| 旋转折进 GEMM | M 大 N 小时 1.27–1.33x，N 大时没有收益 |
| 旋转对量化精度 | 有异常值时 INT4 的 SQNR 提高 9–11 dB；FP8 反而变差（报告 7.7.1） |
| ncu | DRAM 吞吐 87–93%，SM 吞吐 20–24%，是带宽受限的算子 |
| 对比 8 线程 CPU | 快 40.8 倍（按有效带宽算 20.4 倍，CPU 那边是 FP32） |

Tensor Core 没有带来加速这一点我一开始没想到，查下去发现这个算子每字节只有约 1 次运算，
寄存器内核已经把显存带宽用满了，TC 多出来的算力没处用；而 TC 路径每个阶段都要在共享内存里
来回搬一次数据，数据放得进 L2 时这部分开销就露出来了。这和 HadaCore 论文的结论一致。

测量时要注意：这台机器的 GPU 同时在驱动桌面，同一组参数在显存受限区会读出 300 和 385 GB/s
两种结果，表里取的是多次测量的最好值。换机器复现时，更可靠的是"占实测拷贝上限的比例"这类比值。

## 摩尔线程 MTT S4000

同一份源码用 `mcc` 编译，5 个测试共 781 项，0 失败（少的那些是平台确实没有的功能，比如 `mma.sync`，
测试里会跳过并打印原因）。有三条在 NVIDIA 上成立的结论到了这块卡上不成立：

1. 内核只跑到实测上限的 47–89%，还有余量，最有效的参数变成了每个 block 放几行；
2. 原地变换只有 1.10–1.17x，因为这块卡的 L2 带宽相对显存只高 2 倍左右；
3. CUDA Graph 单节点重放每次有约 0.24 ms 的固定开销，逐次重放是负优化，要把多个节点录进一张图才划算。

另外，换编译器后暴露出 `kernel_gemm.cuh` 里一处违反严格别名规则的写法（nvcc 下碰巧正确，
mcc 下读到旧数据），细节在报告 5.9。

## 命令行

```
fhwt --mode verify --dim 256 --rows 4096 [--dtype bf16] [--kernel tc]   正确性自检
fhwt --mode bench  --dim 256 --rows 262144 --iters 50                   计时
fhwt --mode quant  --quant fp8 --dim 256 --rows 262144 [--block-size 32]  融合量化：计时 + 一致性检查
fhwt --mode matrix --kernel reg --csv data/kernels_reg.csv              全规模扫描
fhwt --mode info | roof                                                  设备信息 / 实测带宽上限
fhwt --mode inplace | cpu | advice | loop | gemm | tune | dump           其他实验，见报告第 11 节
```

常用参数：`--kernel {auto,reg,smem,tc}`、`--dtype {f16,bf16,f32}`、`--acc {native,fp32}`、
`--tpr`、`--rpb`、`--scale {norm,raw,<数值>}`、`--graph`、`--csv`。

和 fast_hadamard_transform 逐元素比对（需要先编译好参考实现并放进 `PYTHONPATH`）：

```bash
make refcheck
```

## 已知的不足

- MUSA 上没有 Tensor Core 路径（工具链没有对应的 `mma` 指令），TC 对比只在 NVIDIA 上做了；
- 旋转对量化精度的影响只在合成数据上用 CPU 仿真过，没有在真实模型上验证；
- 报告 7.15.8 里有四个 MUSA 读数是手动跑的，没有留下原始日志；
- 更多见报告 10.2。

## 参考

Tensor Core 分解参考了 HadaCore（arXiv 2412.08832），正确性和性能基准是 Tri Dao 的
`fast_hadamard_transform`。其他文献见 `docs/research/调研笔记.md`。

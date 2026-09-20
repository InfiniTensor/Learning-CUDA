# CUDA Hadamard 变换加速与融合量化总结报告

## 1. 目标与结论

本项目实现了 FP16/BF16、`head_dim=64/128/256` 的归一化快速 Hadamard
变换，并融合逐行 FP8 E4M3/INT8/INT4 动态量化。主核使用每行一个 warp 的寄存器蝶形，
另提供共享内存基线和 cuBLAS Tensor Core GEMM 对照。

在 RTX 5060（sm_120，WSL2 环境）上，输入形状 `[4,128,32,D]`（16384 行）：

- warp 主核相对共享内存基线为 **1.81–2.69×**；维度越大，同步消除的收益越明显。
- D=128/256 时，`O(D log D)` warp 主核比 Tensor Core 的 `O(D²)` GEMM
  快 **1.28–2.00×**（以 TC 时间/warp 时间计）。
- D=64 时 Tensor Core GEMM 与 warp 核基本持平（TC/warp 约 0.94–1.04×），
  小矩阵 batched GEMM 仍有值得保留的调度区间。
- 融合 FP8/INT4 与“先变换后量化”的输出逐字节一致，同时避免中间张量写回和第二次
  kernel launch。

早期数据来自 RTX 5060 Ti 16 GB（单轮 50/500 计时），结论方向一致；
本文数字以第 7 节描述的复测方法为准。

## 2. 背景

QuaRot 使用计算等价的旋转消除隐藏状态异常值，使权重、激活和 KV cache 的
4-bit 量化成为可能。SpinQuant 进一步学习旋转矩阵，说明旋转选择也会影响最终
精度。FlashAttention-3 把随机符号与 Hadamard 组合为 incoherent processing，
利用 `O(D log D)` 变换降低 FP8 注意力量化误差。这些工作的共同工程问题是：
旋转本身必须足够便宜，最好融合在原有的内存受限算子中。

参考资料：

1. [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456)
2. [SpinQuant: LLM Quantization with Learned Rotations](https://arxiv.org/abs/2405.16406)
3. [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)
4. [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

## 3. 算法与数据布局

采用 Sylvester Hadamard：

```text
H_1 = [1]
H_2n = [ H_n  H_n ]
       [ H_n -H_n ]
y = H_D x / sqrt(D)
```

归一化使 `H_D/sqrt(D)` 正交。若指定非零 seed，输入加载时融合确定性的
Rademacher 对角矩阵，计算 `H_D D_sign x / sqrt(D)`，不额外访问 sign 数组。

输入的前三维展平为 `rows=batch×sequence×heads`。最后一维连续，每个 warp
负责一行。lane `l` 持有 `x[l+32k]`：

- stride 1/2/4/8/16 的蝶形通过 `__shfl_xor_sync` 交换 lane；
- stride 32 以上的蝶形在同一线程的寄存器数组内完成；
- 每个元素只读一次、写一次，不使用 shared memory，也没有 block 级同步。

一个 block 放 8 个 warp，在本次 64–256 维范围内兼顾占用率与 launch 数量。
sm_120 的 `cuobjdump -res-usage` 实测显示，纯 warp FHT 随 D=64/128/256 分别
使用 16/19/29 个寄存器/线程，SHARED=0、LOCAL=0（无 barrier、无 spill）；
融合量化的最重实例（D=256，INT4）为 34 个寄存器，同样没有 spill。共享内存
基线的 kernel 自有 shared 为 256/512/1024 bytes（sm_120 运行时另保留 1 KB，
`cuobjdump` 显示 1280/1536/2048 bytes）和 1 个硬件 barrier（源代码在每一级
多次到达该 barrier）。

## 4. 融合量化

FWHT 结束后，融合核继续执行：

1. 按输出类型舍入到 FP16/BF16，以严格复现显式 `FHT -> quantize` 的语义；
2. 每个线程计算其寄存器局部 absmax；
3. `__shfl_down_sync` 完成 warp 最大值归约；
4. lane 0 写行 scale；所有 lane 并行 round、clamp、写量化结果；FP8 E4M3
   使用 finite 最大值 448；
5. INT4 中偶数 lane 通过 shuffle 取得相邻奇数 lane 的值，并写一个 packed byte。

对零行，scale 定义为 1，输出为 0，避免除零。INT4 不使用 `-8`，保持对称范围
`[-7,7]`；INT8 对称范围为 `[-127,127]`。

融合核不写中间 FP16/BF16 结果，因此理论最少流量从：

```text
独立：read(T) + write(T) + read(T) + write(Q) + write(scale)
融合：read(T) + write(Q) + write(scale)
```

减少为一次输入读取与一次低比特输出。

## 5. Tensor Core 路径为何不是主路径

Tensor Core 擅长矩阵乘加，因此项目用 `cublasGemmEx` 实现
`[rows,D] × [D,D]` 的 FP16/BF16 Hadamard GEMM，并启用 Tensor Op math mode。
这条路径是可移植且容易由 ncu 验证 Tensor Pipe 的对照，但其操作数为 `O(D²)`；
蝶形 FWHT 只有 `O(D log D)`，而且 Hadamard 系数只是 ±1，不需要真正乘法。

实测表明没有单一赢家：D=64 时 cuBLAS 的调度与 Tensor Core 吞吐使 GEMM 与
warp 核基本持平；D 增大后冗余计算迅速超过收益（D=256 时 GEMM 约为 warp 核的
2 倍耗时）。因此推荐运行时策略是小 D/超大 rows 可评估 GEMM，常见 D=128/256
走 warp FWHT；若能把 FWHT 融进上游 QKV/rope 或下游量化，则优先融合路径。

## 6. 正确性

测试输入先转换为真实 FP16/BF16，再以 FP32 CPU 蝶形作为参考，避免把输入
舍入误差错误计入内核误差。测试覆盖 18 个 dtype/dim/quant 组合，且使用 37 行
检查尾 block。

RTX 5060（sm_120，CUDA 12.9.1，WSL2）复测自测结果：

```text
FP16: D=64/128/256 × FP8/INT8/INT4 PASS
BF16: D=64/128/256 × FP8/INT8/INT4 PASS
All 18 correctness cases passed.
```

完整 benchmark 中观测到（输入由固定种子生成，误差完全确定，两次平台复测
数值相同）：

- FP16 最大绝对误差：`1.95e-3` 至 `3.89e-3`（阈值 `1e-2`）；
- BF16 最大绝对误差：`2.22e-2` 至 `3.12e-2`（阈值 `5e-2`）；
- fused FP8/INT8/INT4 的量化 bytes 与 scales 均和独立组合完全一致。

## 7. 性能方法与结果

平台：NVIDIA GeForce RTX 5060 8 GB，compute capability 12.0；CUDA 12.9.1
编译器，`sm_120`；运行环境为 WSL2（Ubuntu 24.04，Windows 宿主机）。
输入为 `[4,128,32,D]`，即 16384 行。

WSL2 复测环境存在两个已观测的干扰因素，方法上做了针对性处理：

- SM 时钟被 Windows 侧电源策略限制在约 2025 MHz（标称 boost 3090 MHz），
  WSL2 内无权限也无法锁频；
- 该 GPU 同时承担 Windows 桌面显示，DWM、浏览器等宿主机负载使整套扫描之间
  内存敏感 kernel 偶发最高约 2× 的漂移（`nvidia-smi` 曾显示约 14% 利用率
  而 Linux 侧无任何可见计算进程，即 Windows 侧占用）。

因此每个 kernel 以 warmup 500 次把频率拉升到稳态，再以 CUDA event 计时
1000 次取平均为一轮，独立重复 5 轮取中位数（程序内 `--rounds`，默认 5）；
`scripts/run_benchmarks.sh` 完整扫描执行 2 遍，CSV 取逐行较小值作为无干扰
样本的估计。计时只包围 launch；不包含内存分配、拷贝、CPU 参考以及 Tensor
Core 常量矩阵构造。

### FP16（毫秒）

| D | warp FHT | shared baseline | baseline/warp | fused INT4 | TC GEMM | TC/warp |
|---:|---:|---:|---:|---:|---:|---:|
| 64  | 0.013877 | 0.027457 | 1.98× | 0.014392 | 0.013096 | 0.94× |
| 128 | 0.018615 | 0.044110 | 2.37× | 0.021618 | 0.023887 | 1.28× |
| 256 | 0.034071 | 0.089578 | 2.63× | 0.036540 | 0.065231 | 1.91× |

### BF16（毫秒）

| D | warp FHT | shared baseline | baseline/warp | fused INT4 | TC GEMM | TC/warp |
|---:|---:|---:|---:|---:|---:|---:|
| 64  | 0.014390 | 0.026111 | 1.81× | 0.014934 | 0.014945 | 1.04× |
| 128 | 0.019414 | 0.044943 | 2.31× | 0.027123 | 0.025995 | 1.34× |
| 256 | 0.031729 | 0.085378 | 2.69× | 0.034645 | 0.063412 | 2.00× |

FP8 E4M3 融合路径 D=64/128/256 时 FP16 输入分别为
0.013736/0.022336/0.032336 ms，BF16 输入分别为
0.014958/0.025804/0.031829 ms；所有 case 均与独立组合逐字节一致。

原始 CSV 位于 `results/performance_sm120.csv`。早期 RTX 5060 Ti 16 GB
单轮（50/500）数据方向一致但绝对值略低（时钟更高），已被本表取代。

## 8. ncu / nsys 实际使用与分析

复测机器（WSL2）装有 Nsight Systems 2025.6.3（CUDA 13.2 自带）与
Nsight Compute 2025.2.1（CUDA 12.9.1 自带），`scripts/profile_nsys.sh`
与 `scripts/profile_ncu.sh` 均已实际执行。nsys 可正常采集 kernel 级
时间线，见 8.1 节；ncu 性能计数器与 compute-sanitizer 受 WSL2 平台
限制不可用，见 8.2、8.3 节。

### 8.1 nsys：kernel 级时间线

命令与 `scripts/profile_nsys.sh` 相同，输出位于 `results/nsys/`：

```bash
nsys profile --trace=cuda,nvtx,cublas --force-overwrite=true \
  -o results/nsys/hadamard_d128_fp16_int4 \
  ./build/hadamard_bench --batch 4 --seq 128 --heads 32 --dim 128 \
  --dtype fp16 --quant int4 --warmup 20 --iters 100 \
  --csv results/nsys/profile_run_d128_fp16_int4.csv
```

配置为第 7 节的代表性形状 `[4,128,32,128]` FP16 + 融合 INT4；同一次
运行同时导出 CUDA event CSV，便于与 timeline 交叉验证。`nsys stats`
完整输出见 `results/nsys/nsys_kern_sum_sm120.txt`（kernel 与 memcpy
汇总）和 `results/nsys/nsys_gpu_trace_sm120.txt`（逐次 launch 明细）。
CUDA 12.9.1 自带的 Nsight Systems 2025.1.3 在 WSL2 下采不到 kernel
活动（报告中无 `CUPTI_ACTIVITY_KIND_KERNEL` 数据），kernel 级采集需
使用 2025.6.3。

**各 kernel 的 GPU 时间**（cuda_gpu_kern_sum，每 kernel 121–122 个实例）：

| kernel | 占比 | 平均 | 中位 | 说明 |
|---|---:|---:|---:|---|
| `shared_memory_fht_kernel<half,128>` | 44.3% | 31.92 µs | 31.90 µs | 共享内存基线 |
| cutlass `s16816gemm`（cuBLAS TC GEMM） | 21.7% | 15.63 µs | 15.52 µs | Tensor Core 对照 |
| `warp_fht_kernel<half,128,true,4>` | 19.4% | 13.98 µs | 13.95 µs | 融合 INT4 |
| `warp_fht_kernel<half,128,false,0>` | 14.5% | 10.34 µs | 10.21 µs | 纯 warp FHT |
| `row_quant_kernel<half,128,4>` | 0.1% | 8.00 µs（仅 1 次） | — | 独立量化正确性对照 |

三条路径的绝对耗时排序与第 7 节一致：纯 warp FHT 最快，Tensor Core
GEMM 为其 1.51×，共享内存基线为其 3.09×。cuBLAS 实际派发
`cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x64_32x6_nn_align8`，
确认启用了 Tensor Op（cuBLAS 选择 sm_80 的 CUTLASS tile 在 sm_120 上
运行）。trace 还显示纯 warp FHT 实测 19 寄存器/线程、融合 INT4 为 25、
共享基线为 16 加 1 KB 动态 shared memory，与第 3 节的编译信息互相印证。

**与 CUDA event 计时的交叉验证。** 同一次运行的 event 计时
（`profile_run_d128_fp16_int4.csv`）为 warp 15.17 µs、shared 34.84 µs、
融合 INT4 16.45 µs、TC GEMM 21.08 µs，系统性高于 nsys 的 kernel 纯
执行时长。原因是 event 包住整段异步 launch 循环，测得的是 wall-clock
发射间隔而不是 kernel 占用：trace 中最后 100 次纯 warp FHT 的平均
start-to-start 间隔为 15.10 µs，与 event 的 15.17 µs 几乎相同，而
kernel 实际只占 10.34 µs——CPU 发射速度成为间隔下限。两种口径的相对
排序与比例一致（TC/warp：event 1.39×，nsys 1.51×；shared/warp：
event 2.30×，nsys 3.09×），第 7 节结论不受影响；需要精确 GPU 耗时时
以 nsys kernel 时长为准，event 更适合工程上的相对比较。

**launch API 开销。** CUDA API 级轨迹（`cuda_api_sum`，同配置一次采集
的 380 次 `cudaLaunchKernel`，见 `results/nsys_hadamard.sqlite`）
中位数 3.9 µs、平均 14.3 µs（尾部为首次模块加载）；cuBLAS 路径
`cuLaunchKernel` 中位数同为 3.9 µs。一次性开销（`cudaMalloc` 共
125 ms、`cuLibraryLoadData` 共 104 ms）均发生在计时窗口之外，验证了
“event 只包围 launch”的计时设计。launch API 开销可达 kernel 本体的
约 25%，这是第 10 节 CUDA Graph / persistent kernel 方向的实测依据。

### 8.2 ncu：WSL2 不暴露 GPU 性能计数器

`profile_ncu.sh` 在采集第一个 launch 时返回：

```text
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device 0.
```

本机 GPU 由 Windows 宿主驱动经 dxgkrnl 半虚拟化，Linux 侧不存在
nvidia 内核模块，CUPTI 性能计数器（PM sampling）在 WSL2 上不被支持，
以 root 运行结果相同——这是平台限制而非权限配置问题。完整 ncu 分析
需在 Windows 宿主机（Nsight Compute Windows 版完整支持 RTX 50 系
消费级 GPU，配合 MSVC + CUDA Toolkit 原生构建）或原生 Linux 上进行；
原生 Linux 下若遇到同样的 `ERR_NVGPUCTRPERM` 才是权限问题，以 root
运行 ncu，或由管理员写入
`options nvidia NVreg_RestrictProfilingToAdminUsers=0` 到
`/etc/modprobe.d/` 并重载 nvidia 模块即可。

### 8.3 compute-sanitizer：WSL2 不支持

`compute-sanitizer --tool memcheck` 报告 `Failed to initialize WDDM
debugger interface` 与 `Device not supported`：sanitizer 检测到 GPU 由
Windows 显示驱动模型（WDDM）管理并拒绝工作。内存错误校验需在 Windows
侧或原生 Linux 进行。

### 8.4 无计数器条件下的替代分析

- 静态资源：`cuobjdump -res-usage` 实测（第 3 节）确认 warp 主核 16–34
  寄存器、0 shared、0 local、无 spill，与 ptxas 设计预期一致；
- 有效带宽：FP16 D=256 的 warp 主核达到 492–571 GB/s effective（一次读+
  一次写的最少流量模型），已接近本机观测条件下内存路径的实际上限，说明
  进一步优化空间主要在融合与 launch 开销而非访存模式；
- ncu 计数器指标（`dram__bytes_*` 与 L2 命中率、achieved occupancy、warp
  stall 构成、shared baseline 的 barrier stall）仍有待在原生
  Linux/Windows 评测机上采集，列入第 10 节后续工作。

## 9. 优化历程与发现

初版基线让每个蝶形 stage 都经过 shared memory，并在 stage 间同步。它容易验证，
但同步成本随 `log2(D)` 增长。将索引改成 lane 加寄存器分片后，前 5 级变成 shuffle，
高阶变成线程内交换，完全消除了 shared memory 与 `__syncthreads()`。

融合量化的第一次实现直接量化 FP32 寄存器值，这与显式路径先写 FP16/BF16 再读
的舍入位置不同，少量边界值会得到不同整数。最终在融合核中显式执行一次目标类型
round-trip；由此获得逐字节一致，而不是仅仅数值接近。这一细节对定义“融合等价”
非常重要。

另一个发现是“Tensor Core 必然更快”不成立。Hadamard 的代数结构能把乘法矩阵
降成蝶形；是否用 Tensor Core 应由尺寸、batch 和是否能融合决定，而不是硬件标签。

复测阶段还积累了计时方法层面的经验：几十微秒量级的 kernel 对时钟状态极其
敏感，单轮 50/500 计时在两次扫描间可出现约 2× 漂移，且漂移集中作用于内存
敏感 kernel（warp/baseline/fused 几乎同比例恶化，计算密集的 cuBLAS GEMM 仅
变化约 17%）。排查确认漂移源是 Windows 桌面显示负载（该 GPU 同时驱动
Windows 桌面）与 WDDM 调度，而非 kernel 本身不稳定。最终采用“长 warmup +
多轮中位数 + 多遍扫描逐行取小”的组合，在无法锁频的 WSL2 里得到了可复现的
对比结论。

## 10. 后续工作

1. 把 FWHT epilogue 直接嵌入 QKV GEMM、RoPE 或 attention kernel，进一步消除
   独立 launch 和输入读写。
2. 对 D=64 增加 half2/bfloat162 向量化及多个 row/warp 的布局实验，缩小与 cuBLAS
   的差距。
3. 使用 CUDA Graph 或 persistent kernel 优化 rows 很少时的 launch latency——
   nsys 实测 launch API 中位开销约 3.9 µs，对 13–35 µs 的 kernel 不可忽略。
4. 在 Hopper/Blackwell 上用 CUTLASS/CuTe 实验 Tensor Core 分块 Hadamard，并建立
   D、rows 到实现选择的 autotune 表。
5. 扩展 per-group scale 和下游 FP8/INT4 GEMM 直接消费接口；当前提交已经生成
   scaled FP8 E4M3 与 packed INT4，但没有把量化输出继续融合进矩阵乘。
6. 国产平台适配可把 warp shuffle、BF16 转换和 launch wrapper 抽象成后端 traits，
   再移植到 HIP/昇腾等环境；本提交只验证 NVIDIA CUDA。
7. 在 Windows 宿主机（Nsight Compute Windows 版 + MSVC 原生构建）或原生
   Linux 上补做完整 ncu 指标分析（脚本已就绪，见第 8 节），并补跑
   compute-sanitizer memcheck。

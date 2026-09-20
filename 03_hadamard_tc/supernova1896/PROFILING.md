# 性能分析说明与结果

## Tensor Core 是否被使用

项目包含独立的 FP16 WMMA 对照路径，kernel 中调用了：

```cpp
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
```

在 RTX 4090 D（`sm_89`）上使用 `cuobjdump --dump-sass` 检查编译结果，得到：

```text
Function: hadamard_wmma_fp16_kernel
HMMA.16816.F32 R8, R12.reuse, R16, R8
HMMA.16816.F32 R4, R12, R18, R4
```

`HMMA` 是 NVIDIA Tensor Core 矩阵乘加指令，因此可以确认 WMMA 对照路径实际
使用了 Tensor Core。需要区分的是：主路径 `hadamard_shared_kernel` 使用快速
Hadamard 蝶形算法，不使用 Tensor Core；WMMA 路径用于满足 Tensor Core 探索和
性能对比要求。

## Nsight Systems

已在远端 RTX 4090 D / CUDA 12.8 上执行：

```bash
nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true \
  --output=nsys_hadamard ./hadamard_bench \
  --batch 2 --seq 512 --heads 16 --dim 128 --dtype fp16 --repeats 5
nsys stats --report cuda_gpu_kern_sum --format csv nsys_hadamard.nsys-rep
```

kernel 汇总结果如下：

| Kernel | 调用次数 | 平均时间（ns） | 总时间占比 |
| --- | ---: | ---: | ---: |
| `hadamard_wmma_fp16_kernel` | 5 | 40,262.6 | 36.9% |
| `fused_hadamard_int4_kernel2` | 5 | 31,648.2 | 29.0% |
| `hadamard_shared_kernel` | 5 | 18,694.6 | 17.2% |
| `fused_hadamard_e4m3_kernel` | 5 | 18,374.6 | 16.9% |

Nsight Systems 可以正常采集 CUDA kernel 时间线和汇总，结果支持以下结论：

1. shared-memory FHT 的单次 kernel 时间低于 WMMA 矩阵乘法对照路径。
2. fused E4M3 路径在本测试形状下最快，因为省去了 Hadamard 中间张量写回。
3. WMMA 路径虽然使用 Tensor Core，但显式矩阵乘法的 `O(D^2)` 计算量抵消了
   Tensor Core 的吞吐优势，因此它更适合作为进阶对照，不是本题主路径。

## Nsight Compute

已执行针对 WMMA 和 shared FHT 的 NCU 命令：

```bash
ncu --target-processes all \
  --kernel-name regex:hadamard_wmma_fp16_kernel \
  --launch-count 1 \
  --section SpeedOfLight --section InstructionStats --section Occupancy \
  --csv ./hadamard_bench --batch 2 --seq 512 --heads 16 --dim 128 \
  --dtype fp16 --repeats 1
```

远端容器返回：

```text
ERR_NVGPUCTRPERM
The user does not have permission to access NVIDIA GPU Performance Counters
```

因此本次环境无法得到 NCU 的 occupancy、Tensor Core utilization 和 memory
throughput 计数器。这个限制来自远端 GPU 容器/驱动的 performance counter 权限，
不是 kernel 编译或运行失败。提交到具备管理员权限的环境后，直接运行
`scripts/profile_linux.sh` 即可补采 NCU 报告。

## 一键性能分析

Linux 环境执行：

```bash
chmod +x scripts/profile_linux.sh
CUDA_ARCH=89 ./scripts/profile_linux.sh
```

脚本会：

- 使用 `-lineinfo` 编译 benchmark 和正确性测试；
- 运行全套正确性测试；
- 尝试采集 NCU 指标；
- 采集 Nsight Systems CUDA kernel 时间线；
- 使用 `cuobjdump` 导出 HMMA/MMA 指令证据。

## C500 适配说明

C500 使用独立的 `.maca` 主实现和性能基准，不运行 NVIDIA WMMA 对照内核。
C500 的正确性测试和性能日志见项目目录下的：

- `profiling/maca_perf_fp16.csv`
- `profiling/maca_perf_bf16.csv`

C500 构建必须链接 CUDA 兼容运行时对应的 `libruntime_cu.so` 和
`libsymbol_cu.so`。这一点已经固化在 `scripts/build_maca.sh` 和 CMake 的
`HADAMARD_ENABLE_MACA` 分支中。C500 上的 kernel 仍然使用 shared-memory 蝶形
FHT，因而不依赖 NVIDIA 的 `mma.h` 或 WMMA 接口。

使用 `scripts/profile_maca.sh` 可以重新构建 C500 版本、运行完整正确性测试，
并生成 FP16/BF16 的 CSV 性能日志。

## MTT S4000 适配说明

MTT S4000 使用独立的 `.mu` 主实现和 MUSA 运行时，不运行 NVIDIA WMMA 对照
kernel。构建工具链为 MUSA 4.3.6 的 `mcc`，设备架构环境变量为
`MUSA_ARCH_LIST=22`。正确性和性能日志由 `scripts/profile_musa.sh` 生成，核心
数据保存在：

- `profiling/musa_correctness.log`
- `profiling/musa_perf_fp16.csv`
- `profiling/musa_perf_bf16.csv`

MUSA 路径使用 shared-memory 蝶形 FHT，FP16/BF16 采用软件位级转换，因而不依赖
NVIDIA `cuda_runtime.h`、`half` 或 WMMA。

## CPU 对照

项目新增了 `hadamard_cpu_bench`，使用同样的 `O(D log D)` CPU FHT 算法，避免只
报告 GPU 自身时间而缺少基线：

```bash
./hadamard_cpu_bench --rows 16384 --dim 128 --repeats 10
```

GPU benchmark 统计的是 kernel-only 时间，不包含首次 Host-to-Device 拷贝；CPU
对照也只统计 FHT 计算时间。RTX 4090 D 上 CPU 实测约为 `6.431319 ms`，对应
`rows=16384, dim=128`。报告性能时应明确这一统计口径：CPU 使用 FP32 工作数组，
GPU 使用 FP16 输入/输出和 FP32 shared memory，因此该数据用于说明并行收益，而非
严格同 dtype 的端到端对比。

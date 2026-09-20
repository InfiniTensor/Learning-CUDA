# CUDA Hadamard 与动态 INT8 量化融合优化

本项目在 NVIDIA GeForce RTX 5060 Laptop GPU（SM 12.0）上实现并逐步优化 Fast Walsh-Hadamard Transform（FWHT），随后将 FWHT、逐行绝对值最大值归约和动态对称 INT8 量化融合为单个 CUDA kernel。项目包含独立 Hadamard 的 `v0/v1/v2`、融合算子的 `fused_v0/v1/v2`、严格正确性门禁、CUDA Event benchmark、PyTorch/Dao FHT 对比和 Nsight Compute 分析。

统一计算为：

```text
y = x @ H(D) / sqrt(D)
amax[row]  = max(abs(y[row, :]))
scale[row] = amax[row] / 127          # 全零行使用 scale = 1
q          = round(clamp(y / scale, -127, 127)).to(int8)
```

`H(D)` 是 Sylvester/Walsh Hadamard 矩阵，输入是 contiguous `[batch, D]`，`D` 为 32–8192 范围内的 2 的幂。Dao FHT 固定使用 `scale=1/sqrt(D)`，保证排列和归一化一致。正式延迟由 warmup 后的 CUDA Event 或 `torch.cuda.Event` 给出；NCU 会重放 kernel，因此 NCU duration 只用于定位瓶颈，不能替代正常运行延迟。

## 1. Hadamard 算子优化

独立工程位于 [`hadamard_cudacore`](hadamard_cudacore)。三个版本计算相同的归一化 FWHT，只改变数据驻留位置、launch 数量和 butterfly 实现。

### 1.1 hadamard_v0：global-memory 多 stage 基线

`hadamard_v0` 是直接迭代实现。D=4096 时共有 12 个 butterfly stage，每个 stage 单独启动一个 kernel：从 global memory 读取一对元素，计算和与差，再写入另一个 global buffer；下一 stage 交换输入输出缓冲区，最后一级完成归一化。

该版本便于验证算法，但每一级都产生完整张量读写和一次 launch，成本随 `log2(D)` 累积。FP16 `[1024,4096]` 的 CUDA Event 延迟为 **772.726 μs**。

NCU 在 warmup 100 后采集一次代表性 `hadamard_stage`；它不是 v0 完整 12-stage 延迟：

| 指标 | 实测值 |
|---|---:|
| 单 stage NCU duration | 74.56 μs |
| DRAM throughput | 41.20% |
| Compute (SM) throughput | 63.10% |
| Achieved occupancy | 80.85% |
| Registers/thread | 28 |
| Shared memory/block | 1.02 KiB |
| Stall long scoreboard | 6.26 inst/issued warp |
| Stall barrier | 0.00 inst/issued warp |

单 stage 没有 block barrier，但存在 global-memory 依赖；完整算子还需要 12 次 launch 和 12 轮全量 global-memory 往返。因此 v0 的端到端瓶颈是多 launch 与中间结果物化，而不是单 stage occupancy。

### 1.2 hadamard_v1：整行驻留 shared memory

`hadamard_v1` 改为每行一个 block。线程先把输入转换为 FP32 并加载到 shared memory，在一个 kernel 内完成所有 butterfly stage，最后归一化并写回输入 dtype。它消除了 stage 间的 global-memory 中间结果和 11 次额外 launch。

FP16 `[1024,4096]` 的 CUDA Event 延迟下降到 **230.106 μs**，相对 v0 加速 **3.358x**。完整 NCU 结果为：

| 指标 | 实测值 |
|---|---:|
| NCU duration | 946.46 μs |
| DRAM throughput | 23.50% |
| Compute (SM) throughput | 55.43% |
| Achieved occupancy | 79.47% |
| Registers/thread | 34 |
| Shared memory/block | 17.41 KiB |
| Stall long scoreboard | 2.63 inst/issued warp |
| Stall barrier | 0.71 inst/issued warp |

DRAM 利用率下降说明中间数据已留在片上；但所有 stage 都依赖 shared-memory load/store 和 `__syncthreads()`。NCU 同时提示 compute 与 bandwidth 均低于峰值 60%，说明当前更接近同步与指令依赖造成的 latency-bound，而非显存带宽饱和。

### 1.3 hadamard_v2：warp shuffle 与向量化访存

`hadamard_v2` 保留每行一个 block，但使用向量化 global-memory 访问；前五个 butterfly stage 在寄存器中通过 warp shuffle 完成，只有跨 warp stage 才进入 shared memory，最终结果向量化写回。

FP16 `[1024,4096]` 的 CUDA Event 延迟进一步下降到 **163.796 μs**，相对 v1 加速 **1.405x**、相对 v0 加速 **4.718x**。完整 NCU 结果为：

| 指标 | 实测值 |
|---|---:|
| NCU duration | 210.91 μs |
| DRAM throughput | 22.62% |
| Compute (SM) throughput | 85.07% |
| Achieved occupancy | 79.37% |
| Registers/thread | 38 |
| Shared memory/block | 17.41 KiB |
| Stall long scoreboard | 0.46 inst/issued warp |
| Stall barrier | 0.87 inst/issued warp |

相比 v1，long-scoreboard 从 2.63 降到 0.46，说明寄存器 shuffle 有效缩短 shared-memory 数据依赖；SM throughput 提升至 85.07%。剩余成本主要来自跨 warp stage 的 shared-memory 访问和 block barrier，继续增加显存带宽不会直接解决该瓶颈。

### 1.4 性能结果与当前瓶颈

代表性矩阵使用 warmup=100、repeat=1000、3 次 trial 中位数，并与数学等价的 Dao FHT 比较。`speed_vs_dao = Dao latency / 当前版本 latency`，大于 1 表示自研更快。

| dtype | 版本 | 相对 Dao 几何平均速度 | 快于 Dao 的 case |
|---|---|---:|---:|
| FP16 | v0 / v1 / v2 | 0.179x / 0.910x / **1.000x** | 0 / 8 / 8（共 12） |
| BF16 | v0 / v1 / v2 | 0.172x / 0.890x / **0.974x** | 0 / 8 / 8（共 12） |
| FP32 | v0 / v1 / v2 | 0.173x / 0.978x / **1.057x** | 0 / 8 / 9（共 12） |

v2 在小 batch 和中等 D 上可达到或超过 Dao，但 batch 很大时仍落后。例如 FP16 `[4096,8192]` 中 v2 为 1221.425 μs，Dao 为 500.739 μs，速度为 Dao 的 0.410x。下一版应减少跨 warp barrier，并针对小 D 采用一个 block 处理多行、针对大 D 研究更高并行度的 register/shared-memory tiling。

## 2. Hadamard + 动态 INT8 量化融合优化

融合工程位于 [`Quantification_hadamard_cudacore`](Quantification_hadamard_cudacore)。基线 `separate` 依次执行 Hadamard、逐行 `amax(abs())` 和量化三个 CUDA kernel，并将完整 FP32 Hadamard 中间结果写入显存。三个融合版本均在一个 launch 内输出 INT8、FP32 amax 和 FP32 scale。

以下 NCU workload 固定为 FP16 `[1024,4096]`，warmup 100 后只采集一次正式 launch。

### 2.1 fused_v0：共享内存 FWHT 与传统归约

`fused_v0` 每行一个 block，标量加载输入；完整 FP32 FWHT 放在 shared memory 中；随后使用 256 个 float 的传统 shared-memory block reduction 求 amax，最后标量写出 INT8。

| NCU 指标 | 实测值 |
|---|---:|
| Duration | 295.104 μs |
| DRAM / SM throughput | 13.66% / 76.23% |
| Active warps | 79.40% |
| Registers/thread | 38 |
| Shared memory/block | 18.00 KiB |
| Barrier / long-scoreboard stall | 6.13% / 19.07% |

DRAM 仅为峰值的 13.66%，而 long-scoreboard 达 19.07%，说明 v0 不是外部带宽受限，主要损失来自 shared-memory 依赖、传统归约和同步等待。

### 2.2 fused_v1：向量化加载与 warp reduction

`fused_v1` 将输入改为向量化加载，并用 warp shuffle 完成局部 amax reduction，仅使用少量 shared memory 合并 warp 结果。FWHT 本身仍全部在 shared memory 中完成。

| NCU 指标 | 实测值 |
|---|---:|
| Duration | 264.672 μs |
| DRAM / SM throughput | 21.89% / 79.44% |
| Active warps | 79.56% |
| Registers/thread | 38 |
| Shared memory/block | 17.03 KiB |
| Barrier / long-scoreboard stall | 8.90% / 3.19% |

相对 v0，NCU duration 降低 **10.3%**，long-scoreboard 从 19.07% 降至 3.19%，证明向量化加载和 warp reduction 有效。与此同时 barrier stall 升至 8.90%，瓶颈已转移到全 shared-memory FWHT 的逐 stage 同步。

### 2.3 fused_v2：shuffle FWHT 与打包 INT8 写回

`fused_v2` 在 v1 基础上把前五个 FWHT stage 移入寄存器并使用 warp shuffle；跨 warp stage 才使用 shared memory；amax 继续采用 warp reduction，INT8 输出使用 `char4` 打包写回。

| NCU 指标 | 实测值 |
|---|---:|
| Duration | 216.192 μs |
| DRAM / SM throughput | 13.51% / 83.27% |
| Active warps | 79.48% |
| Registers/thread | 38 |
| Shared memory/block | 17.03 KiB |
| Barrier / long-scoreboard stall | 6.62% / 4.55% |

相对 v1，NCU duration 再降低 **18.3%**，SM throughput 提高至 83.27%。DRAM 仍只有 13.51%，所以剩余限制是 FP32 butterfly 计算和跨 warp shared-memory barrier，而非全局内存带宽。

### 2.4 优化结果

36 个代表性 shape/dtype case 全部通过正确性门禁。相对等价 CUDA `separate` 的几何平均加速为：

| 版本 | 相对 separate 几何平均加速 | 获胜 case |
|---|---:|---:|
| fused_v0 | 2.181x | 36/36 |
| fused_v1 | 2.355x | 36/36 |
| fused_v2 | **2.672x** | 36/36 |

同进程 PyTorch 对比包含 Dao FHT 和完整量化链：

| 版本 | 相对 PyTorch eager | 相对 PyTorch compiled |
|---|---:|---:|
| fused_v0 | 8.046x | 4.295x |
| fused_v1 | 8.446x | 4.508x |
| fused_v2 | **9.054x** | **4.833x** |

NCU 中 PyTorch eager 完整 pipeline 启动 12 个 kernel，累计 733.46 μs；大 elementwise/reduction stage 接近 95% DRAM throughput，主要成本是中间 tensor 重复物化。`torch.compile` 将量化部分融合为一个 Triton kernel，使完整 pipeline 减为 2 个 kernel、累计 155.23 μs；其中量化 kernel 为 74.21 μs，DRAM/SM throughput 为 59.30%/64.47%。Dao FHT 本身在两条路径中约 81 μs。自研融合 kernel 通过单 launch 完成 FWHT、amax 和 INT8 写回，避免完整 Hadamard 中间结果落盘与重读。

后续 v3 应保留单 launch 数据流，同时减少跨 warp barrier；小 D 可让一个 block 处理多行，大 D 可研究 register/shared-memory tiling 和更高 block 并行度。

## 3. 正确性与公平性

- FP32、FP16、BF16 分别测试随机输入、单位基向量、确定性序列和全零输入。
- 独立 Hadamard 验证自研版本、Dao FHT 和 FP32 iterative reference。
- 融合版本验证 INT8、逐行 amax 和 scale，并与 CPU reference、CUDA separate、PyTorch eager/compiled 交叉比较。
- INT8 最大允许 1 LSB；浮点 amax/scale 按 dtype 使用明确容差。
- 代表性 PyTorch 对比共 216 条正确性记录，失败数为 0；只有全部交叉门禁通过才输出有效性能 ratio。
- Dao 的 FP16/BF16 Hadamard 输出先舍入到输入 dtype，自研融合 butterfly 使用 FP32 中间值；最终 INT8、amax 和 scale 通过门禁后才认为性能比较有效。

## 4. 构建与复现

### 4.1 独立 Hadamard

```bash
cd /home/czw1/infinitensor/Hadamard/hadamard_cudacore
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./scripts/bootstrap_env.sh
./scripts/run_benchmark.sh
```

完整矩阵使用 `./scripts/run_benchmark.sh --full`。单 case：

```bash
./build/benchmark_cuda --kernel v2 --batch 1024 --dim 4096 \
  --dtype fp16 --warmup 100 --repeat 1000
```

### 4.2 融合算子与 PyTorch 对比

```bash
cd /home/czw1/infinitensor/Hadamard/Quantification_hadamard_cudacore
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./scripts/run_pytorch_comparison.sh
```

完整矩阵使用 `./scripts/run_pytorch_comparison.sh --full`。

### 4.3 NCU

独立 Hadamard 单版本：

```bash
ncu --profile-from-start off --set full --launch-count 1 \
  ./build/benchmark_cuda --kernel v2 --batch 1024 --dim 4096 \
  --dtype fp16 --warmup 100 --profile
```

融合版本、PyTorch eager/compiled 和可由 NCU-UI 打开的完整报告：

```bash
./scripts/profile_ncu.sh
./scripts/profile_ncu_ui.sh
./scripts/profile_pytorch_comparison_ncu.sh
```

大型 `.ncu-rep`、构建目录、虚拟环境和第三方源码不进入 Git，均可由上述命令重新生成。
